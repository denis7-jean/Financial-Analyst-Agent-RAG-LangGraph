# src/ingestion/ingest.py
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


def ingest_pdf(
    pdf_path: str = "data/APPL_10k.pdf",
    persist_directory: str = "./vector_db",
    collection_name: str = "10k_filings",
):
    """
    Ingest a PDF into a Chroma vector store.
    """
    # Ensure environment variables (e.g., OPENAI_API_KEY) are loaded
    load_dotenv()

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    # 1) Load
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    # 2) Split
    # NOTE: Chunking may need to be tuned for financial tables to preserve row/column integrity.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    # 3) Embed
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Ensure persistence directory exists
    os.makedirs(persist_directory, exist_ok=True)

    # 4) Store in Chroma
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    # Persist to disk
    vectordb.persist()

    return len(chunks)


if __name__ == "__main__":
    added = ingest_pdf()
    print(f"Ingestion Complete! [{added}] chunks added to ChromaDB.")

