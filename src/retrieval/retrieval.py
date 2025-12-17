# src/retrieval/retrieval.py
import os
from pathlib import Path
from dotenv import load_dotenv
# FIX: Use the newer package to match ingest.py
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma # Or langchain_community.vectorstores if errors occur
# Note: If langchain_chroma isn't installed, revert to:
# from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Paths and constants
ROOT_DIR = Path(__file__).resolve().parents[2]
PERSIST_DIR = ROOT_DIR / "vector_db"
COLLECTION_NAME = "10k_filings"

def get_vectorstore():
    """
    Load the existing Chroma vector store with matching embedding settings.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Load from disk
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    return vectorstore

def get_retriever():
    """
    Build a retriever configured for MMR search with top-k=5.
    """
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5},
    )

if __name__ == "__main__":
    try:
        retriever = get_retriever()
        query = "What are the primary risk factors mentioned?"
        
        # FIX: Use .invoke() for LangChain v0.2+
        docs = retriever.invoke(query)
        
        print(f"Retrieved {len(docs)} documents.")
        if docs:
            first = docs[0]
            print(f"First document source: {first.metadata.get('source', 'N/A')}")
            print("-" * 20)
            print(f"First document content snippet:\n{first.page_content[:200]}...")
        else:
            print("No documents retrieved.")
    except Exception as e:
        print(f"An error occurred: {e}")
