import json
import uuid
from pathlib import Path

from src.retrieval.retrieval import (
    Document,
    build_ensemble_retriever,
    get_bm25_retriever,
)


class StaticRetriever:
    def __init__(self, documents):
        self.documents = list(documents)

    def invoke(self, query: str):
        return list(self.documents)


def test_bm25_retriever_loads_structured_chunks_from_jsonl() -> None:
    scratch_dir = Path("tests") / ".tmp"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = scratch_dir / f"chunks_{uuid.uuid4().hex}.jsonl"
    rows = [
        {
            "id": 0,
            "text": "Revenue increased due to services growth.",
            "metadata": {
                "chunk_id": 0,
                "page": 22,
                "section": "Net Sales",
                "source": "Apple_2024_10k.pdf",
                "chunk_kind": "text",
            },
        },
        {
            "id": 1,
            "text": "<table><tr><td>2024</td><td>391,035</td></tr></table>",
            "metadata": {
                "chunk_id": 1,
                "page": 23,
                "section": "Net Sales",
                "source": "Apple_2024_10k.pdf",
                "chunk_kind": "table",
                "content_format": "html",
            },
        },
    ]
    try:
        chunks_path.write_text(
            "\n".join(json.dumps(row) for row in rows),
            encoding="utf-8",
        )

        retriever = get_bm25_retriever(k=2, chunks_jsonl_path=chunks_path, prefer_langchain=False)
        results = retriever.invoke("services revenue growth")

        assert len(results) == 1
        assert results[0].metadata["page"] == 22
        assert results[0].metadata["section"] == "Net Sales"
        assert results[0].metadata["source"] == "Apple_2024_10k.pdf"
    finally:
        chunks_path.unlink(missing_ok=True)


def test_ensemble_retriever_merges_dense_and_sparse_results_without_losing_metadata() -> None:
    dense_only = Document(
        page_content="Dense result about liquidity.",
        metadata={"chunk_id": 10, "page": 15, "section": "Liquidity", "source": "Apple_2024_10k.pdf"},
    )
    shared = Document(
        page_content="Shared result about net sales.",
        metadata={"chunk_id": 11, "page": 32, "section": "Net Sales", "source": "Apple_2024_10k.pdf"},
    )
    sparse_only = Document(
        page_content="Sparse result about risk factors.",
        metadata={"chunk_id": 12, "page": 18, "section": "Risk Factors", "source": "Apple_2024_10k.pdf"},
    )

    dense_retriever = StaticRetriever([dense_only, shared])
    sparse_retriever = StaticRetriever([shared, sparse_only])

    ensemble = build_ensemble_retriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5],
        prefer_langchain=False,
    )
    results = ensemble.invoke("sales risk")

    assert len(results) == 3
    assert {doc.metadata["chunk_id"] for doc in results} == {10, 11, 12}

    merged_shared = next(doc for doc in results if doc.metadata["chunk_id"] == 11)
    assert merged_shared.metadata["page"] == 32
    assert merged_shared.metadata["section"] == "Net Sales"

    sparse_result = next(doc for doc in results if doc.metadata["chunk_id"] == 12)
    assert sparse_result.metadata["page"] == 18
    assert sparse_result.metadata["section"] == "Risk Factors"
