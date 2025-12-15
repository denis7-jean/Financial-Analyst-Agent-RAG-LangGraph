# src/retrieval/retrieval.py

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load environment variables (expects OPENAI_API_KEY in .env)
load_dotenv()

# Paths and constants
ROOT_DIR = Path(__file__).resolve().parents[2]
PERSIST_DIR = ROOT_DIR / "vector_db"
COLLECTION_NAME = "10k_filings"


def get_vectorstore() -> Chroma:
    """
    Load the existing Chroma vector store with matching embedding settings.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
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
    retriever = get_retriever()
    query = "What are the primary risk factors mentioned?"
    docs = retriever.get_relevant_documents(query)

    print(f"Retrieved {len(docs)} documents.")
    if docs:
        first = docs[0]
        print(f"First document source: {first.metadata.get('source', 'N/A')}")
        print("First document content:")
        print(first.page_content)
    else:
        print("No documents retrieved.")
