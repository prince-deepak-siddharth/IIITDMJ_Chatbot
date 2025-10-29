import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import RagasLLM

# We can use LangChain models with RAGAs
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.embeddings import LangchainEmbeddings

# --- 1. Set up RAGAs Models ---
# Use a strong model for judging
ragas_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
# Use the same embeddings as your retriever
lc_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- 2. Create a Test Dataset ---
# You MUST create a "ground truth" dataset to evaluate against.
# This is the most important part of evaluation.
test_questions = [
    "What are the prerequisites for the AI course?",
    "Who is the head of the Computer Science department?",
    "What is the college's policy on attendance?",
]
# You must manually find the correct answers from your data
ground_truth_answers = [
    "The prerequisites for the AI course (AI404) are CS101 and MATH203.",
    "Dr. Emily White is the head of the Computer Science department.",
    "Students are expected to maintain a minimum of 75% attendance in all courses.",
]
# For 'context_recall', you also need the ground_truth context
ground_truth_contexts = [
    ["Page 42: The AI course (AI404) requires CS101 and MATH203 as prerequisites."],
    ["Page 5: The faculty directory lists Dr. Emily White as the HOD for CS."],
    ["Page 12: The student handbook states: 'A minimum of 75% attendance is mandatory...'"],
]

# --- 3. Run Your Agent to Get Results ---
from agent import build_graph, HumanMessage

print("Running agent to collect evaluation data...")
app = build_graph()
results = []

for i, query in enumerate(test_questions):
    config = {"configurable": {"thread_id": f"eval-thread-{i}"}}
    inputs = {"messages": [HumanMessage(content=query)]}
    
    answer = ""
    context = ""
    
    # We run the graph to get the final state
    final_state = app.invoke(inputs, config)
    
    # Extract the answer and context
    answer = final_state["messages"][-1].content
    context = final_state.get("context", "") # Get context from the state
    
    results.append({
        "query": query,
        "ground_truth": ground_truth_answers[i],
        "answer": answer,
        "contexts": [context], # RAGAs expects context as a list of strings
    })

print("...Data collected.")

# --- 4. Prepare Dataset and Run RAGAs ---

# Convert the results to a Hugging Face Dataset
eval_dataset = Dataset.from_list(results)

# Define the metrics
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

# Run the evaluation
print("Running RAGAs evaluation...")
result = evaluate(
    eval_dataset,
    metrics=metrics,
    llm=ragas_llm,
    embeddings=lc_embeddings,
    raise_exceptions=False # Don't stop on a single bad run
)

print("--- EVALUATION COMPLETE ---")
print(result)

# You can also convert to a pandas DataFrame for a cleaner view
df = result.to_pandas()
print(df)
df.to_csv("rag_evaluation_results.csv")
print("Results saved to rag_evaluation_results.csv")