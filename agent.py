import os
from typing import TypedDict, Annotated, List, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver 
from langgraph.graph import add_messages 
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, streaming=True) 
classification_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2") 
db = FAISS.load_local("vectorstore/faiss_index2", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={'k': 3}) #

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: List[Document]
    rewritten_query: str 
    query_type: Literal["simple_rag", "comparative_rag", "conversational"] 
    sub_queries: List[str]

def format_history_for_prompt(messages: list[BaseMessage]) -> str:
    buffer = []
    for msg in messages:
        if isinstance(msg, HumanMessage): buffer.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage): buffer.append(f"AI: {msg.content}")
    return "\n".join(buffer)

def format_docs_for_prompt(docs: List[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in docs])


def inject_system_prompt(state: AgentState) -> dict:
    print("---NODE: INJECT_SYSTEM_PROMPT (START)---")
    has_system_message = any(isinstance(msg, SystemMessage) for msg in state["messages"])
    if not has_system_message:
        system_prompt = (
            "You are a helpful and professional assistant for IIITDMJ. "
            "You must answer user questions based *only* on the retrieved context. "
            "If the context does not contain the answer, you must state that "
            "you do not have that information. Do not make up answers."
        )
        return {"messages": [SystemMessage(content=system_prompt)]}
    return {}

def rewrite_query_node(state: AgentState) -> dict:
    print("---NODE: REWRITE_QUERY---")
    last_human_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_message = msg
            break
    last_query = last_human_message.content if last_human_message else ""
    chat_history = format_history_for_prompt(state["messages"][:-1])
    
    if not chat_history:
        print(f"--- Standalone Query: {last_query} ---")
        return {"rewritten_query": last_query}

    prompt = ChatPromptTemplate.from_template(
        """Given the following chat history and the user's latest question,
        rewrite the user's question to be a standalone question...
        Chat History: {chat_history}
        Latest Question: {query}
        Standalone Question:"""
    )
    rewrite_chain = prompt | classification_llm | StrOutputParser()
    rewritten_query = rewrite_chain.invoke({"chat_history": chat_history, "query": last_query})
    print(f"--- Rewritten Query: {rewritten_query} ---")
    return {"rewritten_query": rewritten_query}

def classify_query_node(state: AgentState) -> dict:
    print("---NODE: CLASSIFY_QUERY---")
    query = state["rewritten_query"]
    prompt = ChatPromptTemplate.from_template(
        """Classify the user's query into one of three categories:
        1.  **simple_rag**: ...
        2.  **comparative_rag**: ...
        3.  **conversational**: ...
        Query: {query}
        """
    )
    classification_chain = prompt | classification_llm | StrOutputParser()
    result = classification_chain.invoke({"query": query})
    
    decision = "simple_rag" 
    if "comparative_rag" in result.lower(): decision = "comparative_rag"
    elif "conversational" in result.lower(): decision = "conversational"
    print(f"--- Decision: {decision} ---")
    return {"query_type": decision}

def handle_chat_node(state: AgentState) -> dict:
    """
    Path A: Generates an answer based *only* on the chat history.
    """
    print("---NODE: HANDLE_CHAT---")
    # query = state["rewritten_query"] 
    chat_history = format_history_for_prompt(state["messages"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful college assistant. Answer the user's question based on the chat history. Be conversational."),
        ("user", "Here is the chat history (including my last question):\n{chat_history}\n\nNow, please provide a conversational answer.")
    ])
    generation_chain = prompt | llm | StrOutputParser()
    answer = generation_chain.invoke({"chat_history": chat_history})

    print(f"--- HANDLE_CHAT generated answer: {answer} ---")

    return {"messages": [AIMessage(content=answer)]}

def retrieve_docs_node(state: AgentState) -> dict:
    print("---NODE: RETRIEVE_DOCS (SIMPLE)---")
    query = state["rewritten_query"]
    documents = retriever.invoke(query)
    print("\n--- RETRIEVED CONTEXT ---")
    if documents:
        for i, doc in enumerate(documents):
            print(f"DOC {i+1}: Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
    else: print("!!! No context retrieved. !!!")
    print("---------------------------\n")
    return {"context": documents}

def generate_answer_node(state: AgentState) -> dict:
    print("---NODE: GENERATE_ANSWER (SIMPLE)---")
    query = state["rewritten_query"]
    context_docs = state["context"]
    context_str = format_docs_for_prompt(context_docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant. Answer the user's question based *only* on the retrieved context. "
            "If the context is empty or irrelevant, you *must* state that you do not have the information "
            "and recommend visiting the official Indian Institute of Information Technology, Design and Manufacturing, Jabalpur (IIITDM Jabalpur) website (https://www.iiitdmj.ac.in/) for more details."
         )),
        ("user", "Context:\n{context}\n\nQuestion:\n{query}")
    ])
    
    generation_chain = prompt | llm | StrOutputParser()
    answer = generation_chain.invoke({"context": context_str, "query": query})
    
    sources = []
    if context_docs:
        for i, doc in enumerate(context_docs):
            source_file = doc.metadata.get('source', 'N/A')
            source_name = source_file.split('/')[-1]
            page_num = doc.metadata.get('page', 'N/A')
            sources.append(f"  {i+1}. {source_name} (Page: {page_num})")
    
    if sources and "website" not in answer: 
        pretty_answer = answer + "\n--- \n**Sources:**\n" + "\n".join(sources)
    else:
        pretty_answer = answer

    return {"messages": [AIMessage(content=pretty_answer)]}

def decompose_query_node(state: AgentState) -> dict:
    print("---NODE: DECOMPOSE_QUERY---")
    query = state["rewritten_query"]
    prompt = ChatPromptTemplate.from_template(
        """You are a query decomposition assistant...
        Query: {query}
        Respond with a JSON object..."""
    )
    parser = JsonOutputParser()
    decomposition_chain = prompt | classification_llm | parser
    result = decomposition_chain.invoke({"query": query})
    print(f"--- Sub-queries: {result['queries']} ---")
    return {"sub_queries": result['queries']}

def retrieve_multi_docs_node(state: AgentState) -> dict:
    print("---NODE: RETRIEVE_DOCS (MULTI)---")
    sub_queries = state["sub_queries"]
    all_docs = []
    for query in sub_queries:
        documents = retriever.invoke(query)
        all_docs.extend(documents)
    unique_docs_map = {doc.page_content: doc for doc in all_docs}
    unique_docs = list(unique_docs_map.values())
    print("\n--- RETRIEVED CONTEXT (MULTI) ---")
    if unique_docs:
        for i, doc in enumerate(unique_docs):
            print(f"DOC {i+1}: Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
    else: print("!!! No context retrieved. !!!")
    print("---------------------------\n")
    return {"context": unique_docs}

def generate_synthesized_answer_node(state: AgentState) -> dict:
    print("---NODE: GENERATE_ANSWER (SYNTHESIZED)---")
    query = state["rewritten_query"]
    context_docs = state["context"]
    context_str = format_docs_for_prompt(context_docs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant. Your task is to answer a comparative question based on the provided context. "
            "Synthesize the information from the context to form a comprehensive answer. "
            "If the context is insufficient, you *must* state that you do not have the information "
            "and recommend visiting the official Indian Institute of Information Technology, Design and Manufacturing, Jabalpur (IIITDM Jabalpur) website (https://www.iiitdmj.ac.in/) for more details."
        )),
        ("user", (
            "Here is the context I've gathered:\n{context}\n\n"
            "Now, please answer this original question:\n{query}"
        ))
    ])
    
    generation_chain = prompt | llm | StrOutputParser()
    answer = generation_chain.invoke({"context": context_str, "query": query})
    
    sources = []
    if context_docs:
        for i, doc in enumerate(context_docs):
            source_file = doc.metadata.get('source', 'N/A')
            source_name = source_file.split('/')[-1]
            page_num = doc.metadata.get('page', 'N/A')
            sources.append(f"  {i+1}. {source_name} (Page: {page_num})")
    
    if sources and "website" not in answer:
        pretty_answer = answer + "\n--- \n**Sources:**\n" + "\n".join(sources)
    else:
        pretty_answer = answer

    return {"messages": [AIMessage(content=pretty_answer)]}

def router(state: AgentState) -> Literal["conversational", "simple_rag", "comparative_rag"]:
    print(f"--- ROUTING TO: {state['query_type']} ---")
    return state["query_type"]

checkpointer = MemorySaver()

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("inject_system_prompt", inject_system_prompt)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("handle_chat", handle_chat_node)
    workflow.add_node("retrieve_docs", retrieve_docs_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("decompose_query", decompose_query_node)
    workflow.add_node("retrieve_multi_docs", retrieve_multi_docs_node)
    workflow.add_node("generate_synthesized_answer", generate_synthesized_answer_node)

    workflow.set_entry_point("inject_system_prompt")
    workflow.add_edge("inject_system_prompt", "rewrite_query")
    workflow.add_edge("rewrite_query", "classify_query")
    workflow.add_conditional_edges(
        "classify_query",
        router,
        {
            "conversational": "handle_chat",
            "simple_rag": "retrieve_docs",
            "comparative_rag": "decompose_query"
        }
    )
    workflow.add_edge("handle_chat", END)
    workflow.add_edge("retrieve_docs", "generate_answer")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("decompose_query", "retrieve_multi_docs")
    workflow.add_edge("retrieve_multi_docs", "generate_synthesized_answer")
    workflow.add_edge("generate_synthesized_answer", END)

    app = workflow.compile(checkpointer=checkpointer)
    return app

chatbot = build_graph() 

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test-direct-run-1"}} 
    print("\n--- Testing Direct Run ---")
    inputs = {"messages": [HumanMessage(content="What is the name of director?")]}
    for event in chatbot.stream(inputs, config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()