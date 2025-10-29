import streamlit as st
from agent import chatbot, classification_llm 
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
import uuid
import asyncio

def generate_thread_id():
    thread_id= uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id=uuid.uuid4()
    st.session_state['thread_id']=thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history']=[]

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        st.session_state['thread_titles'][thread_id]=f"New Chat {len(st.session_state['chat_threads'])}"

def load_conversation(thread_id):
   try:
        state= chatbot.get_state(config={'configurable' : {'thread_id': thread_id}})
        raw_messages = state.values.get('messages', []) if state else []
        return [msg for msg in raw_messages if isinstance(msg, BaseMessage)]
   except Exception as e:
       print(f"Error loading conversation for thread {thread_id}: {e}")
       return []

def generate_title(query):
    print("--- Generating Title ---")
    try:
        prompt = f"Summarize this query into a very short title (max 5 words): {query}"
        response = classification_llm.invoke(prompt)
        title = response.content.strip().strip('"')
        return title if title else "Chat"
    except Exception as e:
        print(f"Error generating title: {e}")
        return "Chat"

if 'message_history' not in st.session_state: st.session_state['message_history']=[]
if 'thread_id' not in st.session_state: st.session_state['thread_id']=generate_thread_id()
if 'chat_threads' not in st.session_state: st.session_state['chat_threads']=[]
if 'thread_titles' not in st.session_state: st.session_state['thread_titles']={}
add_thread(st.session_state['thread_id'])

st.sidebar.title("IIITDMJ Chatbot")
if st.sidebar.button("➕ New Chat"):
    reset_chat()
    st.rerun()
st.sidebar.header("My Conversations")
for thread_id in st.session_state['chat_threads'][::-1]:
    title=st.session_state['thread_titles'].get(thread_id,"Untitled Chat")
    if st.sidebar.button(title, key=f"thread_{thread_id}", use_container_width=True):
        st.session_state['thread_id']=thread_id
        messages= load_conversation(thread_id)
        temp_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage): continue
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})
        st.session_state['message_history'] = temp_messages
        st.rerun()

st.title("IIITDMJ College Assistant")
st.caption("This bot uses a local vector store and LangGraph to answer your questions.")

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        if message['role'] == 'assistant':
            st.markdown(f"<div style='font-size: 15px;'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(message['content'])

user_input=st.chat_input("Ask about IIITDMJ...")

if user_input:

    CONFIG={'configurable' : {'thread_id': st.session_state['thread_id']}}

    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message('assistant'):
        placeholder = st.empty()
        ai_message_content = ""

        try:
            print(f"\n--- Streaming response for Thread ID: {st.session_state['thread_id']} ---")

            async def stream_agent_events(stream_placeholder):
                local_ai_message_content_streamed = ""
                local_final_node_output = None
                local_final_node_name = ""

                async for event in chatbot.astream_events(
                    {'messages': [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    version="v1"
                ):
                    kind = event["event"]
                    name = event["name"]

                    if kind == "on_chat_model_stream":
                        if name in ("generate_answer", "generate_synthesized_answer", "handle_chat"):
                            chunk_content = event["data"]["chunk"].content
                            if chunk_content:
                                local_ai_message_content_streamed += chunk_content
                                stream_placeholder.markdown(f"<div style='font-size: 15px;'>{local_ai_message_content_streamed}▌</div>", unsafe_allow_html=True)

                    if kind == "on_chain_end":
                        if name in ("generate_answer", "generate_synthesized_answer", "handle_chat"):
                             if "output" in event.get("data", {}) and isinstance(event["data"]["output"], dict):
                                local_final_node_output = event["data"]["output"]
                                local_final_node_name = name
                                print(f"--- Captured final output from node: {name} ---")
                
                return local_ai_message_content_streamed, local_final_node_output, local_final_node_name


            streamed_content, final_output, final_name = asyncio.run(stream_agent_events(placeholder))

            if not streamed_content and final_output:
                print(f"--- Using fallback: No stream content captured. Using final output from {final_name}. ---")
                if "messages" in final_output and final_output["messages"]:
                    ai_message_content = final_output["messages"][-1].content
                    placeholder.markdown(f"<div style='font-size: 15px;'>{ai_message_content}</div>", unsafe_allow_html=True)
                else:
                    print(f"--- Fallback failed: Final output from {final_name} had unexpected format: {final_output} ---")
                    ai_message_content = "Sorry, I couldn't generate a response (fallback error)."
                    placeholder.markdown(ai_message_content) 

            elif streamed_content:
                 ai_message_content = streamed_content
                 placeholder.markdown(f"<div style='font-size: 15px;'>{ai_message_content}</div>", unsafe_allow_html=True)
            else:
                 print("--- Fallback failed: No stream content and no final output captured. ---")
                 ai_message_content = "Sorry, I couldn't generate a response (capture error)."
                 placeholder.markdown(ai_message_content)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            print(f"ERROR DURING STREAM/FALLBACK: {e}")
            ai_message_content = "Sorry, I encountered an error during execution."
            placeholder.markdown(ai_message_content)

    if not ai_message_content:
        ai_message_content = "Sorry, I couldn't generate a response."

    st.session_state['message_history'].append({'role':'assistant','content':ai_message_content})

    current_id=st.session_state['thread_id']
    current_title=st.session_state['thread_titles'].get(current_id,"New Chat")
    if current_title.startswith("New Chat") and len(st.session_state['message_history']) <= 2:
        summarized_title = generate_title(user_input)
        st.session_state['thread_titles'][current_id] = summarized_title
        st.rerun()