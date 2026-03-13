import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in test environments
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

try:
    from langchain.retrievers import EnsembleRetriever as LangChainEnsembleRetriever
except ImportError:  # pragma: no cover - fallback used in local tests
    LangChainEnsembleRetriever = None

try:
    from langchain_community.retrievers import BM25Retriever as LangChainBM25Retriever
except ImportError:  # pragma: no cover - fallback used in local tests
    LangChainBM25Retriever = None

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - lightweight fallback for local tests
    @dataclass
    class Document:
        page_content: str
        metadata: dict[str, Any] = field(default_factory=dict)


from src.config import CHROMA_PERSIST_DIR, CHUNKS_JSONL_PATH, EMBED_MODEL, GCS_BUCKET, VECTOR_DB_PREFIX


COLLECTION_NAME = "10k_filings"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def _document_key(document: Document) -> str:
    metadata = document.metadata or {}
    if metadata.get("chunk_id") is not None:
        return f"chunk:{metadata['chunk_id']}"
    return f"{metadata.get('source', 'unknown')}|{metadata.get('page', 'na')}|{hash(document.page_content)}"


def load_chunk_documents(chunks_jsonl_path: str | Path | None = None) -> list[Document]:
    path = Path(chunks_jsonl_path or CHUNKS_JSONL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {path}")

    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metadata = dict(record.get("metadata") or {})
            if metadata.get("chunk_id") is None and record.get("id") is not None:
                metadata["chunk_id"] = record["id"]
            documents.append(Document(page_content=record.get("text", ""), metadata=metadata))
    return documents


class LocalBM25Retriever:
    def __init__(self, documents: Sequence[Document], k: int = 6) -> None:
        self.documents = list(documents)
        self.k = k

    @classmethod
    def from_documents(cls, documents: Sequence[Document]) -> "LocalBM25Retriever":
        return cls(documents)

    def invoke(self, query: str) -> list[Document]:
        query_tokens = set(_tokenize(query))
        scored: list[tuple[int, int, Document]] = []
        for index, document in enumerate(self.documents):
            score = sum(1 for token in _tokenize(document.page_content) if token in query_tokens)
            if score > 0:
                scored.append((score, -index, document))
        scored.sort(reverse=True)
        return [document for _, _, document in scored[: self.k]]


class LocalEnsembleRetriever:
    def __init__(self, retrievers: Sequence[Any], weights: Sequence[float]) -> None:
        self.retrievers = list(retrievers)
        self.weights = list(weights)

    def invoke(self, query: str) -> list[Document]:
        fused_scores: dict[str, float] = {}
        documents: dict[str, Document] = {}

        for weight, retriever in zip(self.weights, self.retrievers):
            for rank, document in enumerate(retriever.invoke(query), start=1):
                key = _document_key(document)
                fused_scores[key] = fused_scores.get(key, 0.0) + (float(weight) / rank)
                documents[key] = document

        return [documents[key] for key, _ in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)]


def build_ensemble_retriever(
    retrievers: Sequence[Any],
    weights: Sequence[float],
    prefer_langchain: bool = True,
) -> Any:
    if prefer_langchain and LangChainEnsembleRetriever is not None:
        return LangChainEnsembleRetriever(retrievers=list(retrievers), weights=list(weights))
    return LocalEnsembleRetriever(retrievers=retrievers, weights=weights)


def get_vectorstore(persist_directory: str | Path | None = None):
    load_dotenv()

    persist_dir = Path(persist_directory or CHROMA_PERSIST_DIR)
    if not persist_dir.exists() or not any(persist_dir.iterdir()):
        persist_dir.mkdir(parents=True, exist_ok=True)
        from src.storage.gcs import download_prefix

        download_prefix(GCS_BUCKET, VECTOR_DB_PREFIX, str(persist_dir))

    from langchain_community.vectorstores import Chroma
    from langchain_google_vertexai import VertexAIEmbeddings

    embeddings = VertexAIEmbeddings(model=EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def get_dense_retriever(k: int = 6, persist_directory: str | Path | None = None):
    vectorstore = get_vectorstore(persist_directory=persist_directory)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k},
    )


def get_bm25_retriever(
    k: int = 6,
    chunks_jsonl_path: str | Path | None = None,
    prefer_langchain: bool = True,
) -> Any:
    documents = load_chunk_documents(chunks_jsonl_path=chunks_jsonl_path)
    if prefer_langchain and LangChainBM25Retriever is not None:
        retriever = LangChainBM25Retriever.from_documents(documents)
        retriever.k = k
        return retriever

    retriever = LocalBM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever


def get_hybrid_retriever(
    k: int = 6,
    weights: tuple[float, float] = (0.5, 0.5),
    persist_directory: str | Path | None = None,
    chunks_jsonl_path: str | Path | None = None,
    prefer_langchain: bool = True,
) -> Any:
    dense_retriever = get_dense_retriever(k=k, persist_directory=persist_directory)
    bm25_retriever = get_bm25_retriever(
        k=k,
        chunks_jsonl_path=chunks_jsonl_path,
        prefer_langchain=prefer_langchain,
    )
    return build_ensemble_retriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=weights,
        prefer_langchain=prefer_langchain,
    )


def hybrid_search(
    query: str,
    k: int = 6,
    weights: tuple[float, float] = (0.5, 0.5),
    persist_directory: str | Path | None = None,
    chunks_jsonl_path: str | Path | None = None,
    prefer_langchain: bool = True,
) -> list[Document]:
    retriever = get_hybrid_retriever(
        k=k,
        weights=weights,
        persist_directory=persist_directory,
        chunks_jsonl_path=chunks_jsonl_path,
        prefer_langchain=prefer_langchain,
    )
    return list(retriever.invoke(query))[:k]


if __name__ == "__main__":
    docs = hybrid_search("What are the primary risk factors mentioned?", k=5)
    print(f"Retrieved {len(docs)} documents.")
    for document in docs:
        metadata = document.metadata or {}
        print(
            f"- chunk_id={metadata.get('chunk_id')} | "
            f"page={metadata.get('page')} | "
            f"section={metadata.get('section')} | "
            f"source={metadata.get('source')}"
        )
