# src/retrieval/hybrid.py
from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_core.documents import Document

from src.retrieval.retrieval import get_vectorstore
from src.config import CHROMA_PERSIST_DIR as PERSIST_DIR
from src.retrieval.bm25 import bm25_search


def _minmax_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _doc_key(doc: Document) -> str:
    """
    Stable key for merging bm25 + vector results.
    Prefer chunk_id (we add it during ingestion).
    Fallback to (source,page,content-hash) if needed.
    """
    md = doc.metadata or {}
    if "chunk_id" in md:
        return f"chunk:{md['chunk_id']}"
    src = md.get("source", "unknown")
    page = md.get("page", "na")
    return f"{src}#p{page}#{hash(doc.page_content)}"


def hybrid_search(
    query: str,
    k_vec: int = 10,
    k_bm25: int = 10,
    k_final: int = 8,
    w_bm25: float = 0.4,
    w_vec: float = 0.6,
) -> List[Document]:
    """
    Hybrid retrieval: BM25 + Vector with min-max normalized weighted fusion.
    Writes scores into Document.metadata:
      - score_bm25 (0..1)
      - score_vec  (0..1)
      - score_final (0..1-ish, weighted)
    Returns top k_final Documents.
    """
    # ---- Vector search ----
    vectorstore = get_vectorstore()

    # Chroma similarity_search_with_score returns (Document, distance)
    vec_pairs: List[Tuple[Document, float]] = vectorstore.similarity_search_with_score(query, k=k_vec)
    vec_docs = [d for d, _ in vec_pairs]
    vec_dists = [float(s) for _, s in vec_pairs]

    # Convert distance -> similarity-ish score: smaller dist => higher score
    # normalize distances then invert
    norm_dists = _minmax_normalize(vec_dists) if vec_dists else []
    vec_norm_scores = [1.0 - nd for nd in norm_dists]

    # ---- BM25 search ----
    bm25_pairs: List[Tuple[Document, float]] = bm25_search(query, PERSIST_DIR, k=k_bm25)
    bm_docs = [d for d, _ in bm25_pairs]
    bm_raw_scores = [float(s) for _, s in bm25_pairs]
    bm_norm_scores = _minmax_normalize(bm_raw_scores)

    # ---- Merge by key ----
    merged: Dict[str, dict] = {}

    # Add vector results
    for doc, score in zip(vec_docs, vec_norm_scores):
        key = _doc_key(doc)
        merged.setdefault(key, {"doc": doc, "bm25": 0.0, "vec": 0.0})
        merged[key]["doc"] = doc
        merged[key]["vec"] = max(merged[key]["vec"], float(score))

    # Add BM25 results
    for doc, score in zip(bm_docs, bm_norm_scores):
        key = _doc_key(doc)
        merged.setdefault(key, {"doc": doc, "bm25": 0.0, "vec": 0.0})
        if merged[key]["doc"] is None:
            merged[key]["doc"] = doc
        merged[key]["bm25"] = max(merged[key]["bm25"], float(score))

    # ---- Fuse + attach scores into metadata ----
    scored: List[Tuple[Document, float, float, float]] = []
    for item in merged.values():
        doc: Document = item["doc"]
        bm = float(item["bm25"])
        ve = float(item["vec"])
        fused = float(w_bm25 * bm + w_vec * ve)

        # Write scores into metadata (non-breaking; just adds fields)
        md = dict(doc.metadata or {})
        md["score_bm25"] = bm
        md["score_vec"] = ve
        md["score_final"] = fused
        doc.metadata = md

        scored.append((doc, bm, ve, fused))

    scored.sort(key=lambda x: x[3], reverse=True)
    return [d for d, _, _, _ in scored[:k_final]]
