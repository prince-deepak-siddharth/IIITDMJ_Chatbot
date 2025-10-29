import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2" # Model for embeddings

embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

print(f"Loading documents from {DATA_PATH}...")
loader = DirectoryLoader(
    DATA_PATH,
    glob='*.pdf',         
    loader_cls=PyPDFLoader  
)
documents = loader.load()

if not documents:
    print("No PDF documents found. Make sure your PDFs are in the /data folder.")
    exit()

print(f"Loaded {len(documents)} PDF document(s).")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
docs = text_splitter.split_documents(documents)

print(f"Split into {len(docs)} chunks.")

print("Creating and saving FAISS vector store...")
db = FAISS.from_documents(docs, embeddings)
db.save_local(DB_FAISS_PATH)

print(f"Successfully created and saved FAISS index to {DB_FAISS_PATH}")