# src/retrieval/bm25.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

_bm25_cache: dict[str, tuple[BM25Okapi, list[Document]]] = {}


def _tokenize(text: str) -> List[str]:
    # Simple, fast tokenizer for BM25
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def load_chunks_jsonl(chunks_jsonl_path: Path) -> List[Document]:
    """
    Load chunks.jsonl into a list of LangChain Documents.
    Expects each line: {"id": int, "text": str, "metadata": dict}
    """
    if not chunks_jsonl_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {chunks_jsonl_path}")

    docs: List[Document] = []
    with chunks_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            md = rec.get("metadata", {}) or {}
            # Ensure chunk_id exists for merging
            if "chunk_id" not in md and rec.get("id") is not None:
                md["chunk_id"] = rec["id"]
            docs.append(Document(page_content=text, metadata=md))
    return docs


def get_bm25_index(persist_dir: Path) -> Tuple[BM25Okapi, List[Document]]:
    """
    Build (or reuse cached) BM25 index from persist_dir/chunks.jsonl.
    """
    key = str(persist_dir.resolve())
    if key in _bm25_cache:
        return _bm25_cache[key]

    chunks_jsonl_path = persist_dir / "chunks.jsonl"
    docs = load_chunks_jsonl(chunks_jsonl_path)
    corpus_tokens = [_tokenize(d.page_content) for d in docs]
    bm25 = BM25Okapi(corpus_tokens)

    _bm25_cache[key] = (bm25, docs)
    return bm25, docs


def bm25_search(
    query: str,
    persist_dir: Path,
    k: int = 10,
) -> List[tuple[Document, float]]:
    """
    Return top-k BM25 results as (Document, raw_score).
    """
    bm25, docs = get_bm25_index(persist_dir)
    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)  # numpy array-like
    # Take top-k indices by score desc
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(docs[i], float(scores[i])) for i in top_idx]
