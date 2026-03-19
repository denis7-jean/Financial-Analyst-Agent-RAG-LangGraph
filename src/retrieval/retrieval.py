from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in test environments
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

try:
    from langsmith import traceable as _ls_traceable
except ImportError:  # pragma: no cover - optional in environments without langsmith
    def _ls_traceable(**kwargs):  # type: ignore[misc]
        def _decorator(fn):
            return fn
        return _decorator

try:
    from langchain.retrievers import EnsembleRetriever as LangChainEnsembleRetriever
except ImportError:  # pragma: no cover - optional dependency
    LangChainEnsembleRetriever = None

try:
    from langchain_community.retrievers import BM25Retriever as LangChainBM25Retriever
except ImportError:  # pragma: no cover - optional dependency
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
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
RRF_K = 20
DEFAULT_DENSE_K = 10
DEFAULT_SPARSE_K = 10
FINANCIAL_TERMS = (
    "net sales",
    "sales",
    "revenue",
    "income",
    "margin",
    "cash flow",
    "eps",
    "profit",
    "operating",
)
SECTION_HINTS = {
    "risk": ("risk factors",),
    "risk factors": ("risk factors",),
    "md&a": ("management's discussion", "management discussion"),
    "management discussion": ("management's discussion", "management discussion"),
    "operations": ("consolidated statements of operations", "statement of comprehensive income"),
    "income": ("consolidated statements of operations", "statement of comprehensive income"),
}
COMPANY_HINTS = {
    "apple": ("apple", "aapl"),
    "microsoft": ("microsoft", "msft"),
    "tesla": ("tesla", "tsla"),
    "alphabet": ("alphabet", "google", "googl"),
    "amazon": ("amazon", "amzn"),
    "meta": ("meta", "facebook", "meta platforms"),
    "nvidia": ("nvidia", "nvda"),
}

_CHUNK_CACHE: dict[str, list[Document]] = {}
_SPARSE_CACHE: dict[str, "LocalBM25Retriever"] = {}


@dataclass
class SearchCandidate:
    """Aggregated candidate state across retrievers and query expansions."""

    document: Document
    key: str
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    fused_score: float = 0.0
    rerank_boost: float = 0.0
    rerank_reasons: list[str] = field(default_factory=list)

    @property
    def final_score(self) -> float:
        return self.fused_score + self.rerank_boost


class LocalEnsembleRetriever:
    """Small local ensemble for compatibility tests and simple callers."""

    def __init__(self, retrievers: Sequence[Any], weights: Sequence[float]) -> None:
        self.retrievers = list(retrievers)
        self.weights = list(weights)

    def invoke(self, query: str) -> list[Document]:
        fused: dict[str, float] = {}
        documents: dict[str, Document] = {}
        for weight, retriever in zip(self.weights, self.retrievers):
            for rank, document in enumerate(retriever.invoke(query), start=1):
                key = _document_key(document)
                fused[key] = fused.get(key, 0.0) + (float(weight) / (rank + RRF_K))
                documents[key] = document
        return [documents[key] for key, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)]


class LocalBM25Retriever:
    """Lightweight BM25-style sparse retriever with IDF and length normalization."""

    def __init__(
        self,
        documents: Sequence[Document],
        k: int = 6,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = list(documents)
        self.k = k
        self.k1 = k1
        self.b = b
        self._tokenized_docs = [_tokenize(doc.page_content) for doc in self.documents]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokenized_docs]
        self._doc_lengths = [len(tokens) for tokens in self._tokenized_docs]
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 1.0
        )
        self._doc_frequencies = self._build_doc_frequencies(self._tokenized_docs)
        self._doc_count = len(self.documents)

    @classmethod
    def from_documents(cls, documents: Sequence[Document]) -> "LocalBM25Retriever":
        return cls(documents)

    @staticmethod
    def _build_doc_frequencies(tokenized_docs: Sequence[Sequence[str]]) -> Counter[str]:
        doc_frequencies: Counter[str] = Counter()
        for tokens in tokenized_docs:
            doc_frequencies.update(set(tokens))
        return doc_frequencies

    def _idf(self, token: str) -> float:
        document_frequency = self._doc_frequencies.get(token, 0)
        return math.log(1.0 + ((self._doc_count - document_frequency + 0.5) / (document_frequency + 0.5)))

    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        requested_k = k or self.k
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        query_terms = Counter(query_tokens)
        scored: list[tuple[float, int, Document]] = []
        for index, (document, term_counts, doc_length) in enumerate(
            zip(self.documents, self._term_frequencies, self._doc_lengths)
        ):
            score = 0.0
            for token in query_terms:
                term_frequency = term_counts.get(token, 0)
                if term_frequency == 0:
                    continue
                numerator = term_frequency * (self.k1 + 1.0)
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * (doc_length / max(self._avg_doc_length, 1.0))
                )
                score += self._idf(token) * (numerator / max(denominator, 1e-9))
            if score > 0.0:
                scored.append((score, -index, document))

        scored.sort(reverse=True)
        return [(document, float(score)) for score, _, document in scored[:requested_k]]

    def invoke(self, query: str) -> list[Document]:
        return [document for document, _ in self.search(query, k=self.k)]


class HybridRetriever:
    """Compatibility wrapper exposing invoke(query) over hybrid_search()."""

    def __init__(
        self,
        *,
        k: int = 6,
        weights: tuple[float, float] = (0.5, 0.5),
        persist_directory: str | Path | None = None,
        chunks_jsonl_path: str | Path | None = None,
        prefer_langchain: bool = True,
        dense_k: int | None = None,
        sparse_k: int | None = None,
        use_query_expansion: bool = True,
        use_rerank: bool = True,
    ) -> None:
        self.k = k
        self.weights = weights
        self.persist_directory = persist_directory
        self.chunks_jsonl_path = chunks_jsonl_path
        self.prefer_langchain = prefer_langchain
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.use_query_expansion = use_query_expansion
        self.use_rerank = use_rerank

    def invoke(self, query: str) -> list[Document]:
        return hybrid_search(
            query=query,
            k=self.k,
            weights=self.weights,
            persist_directory=self.persist_directory,
            chunks_jsonl_path=self.chunks_jsonl_path,
            prefer_langchain=self.prefer_langchain,
            dense_k=self.dense_k,
            sparse_k=self.sparse_k,
            use_query_expansion=self.use_query_expansion,
            use_rerank=self.use_rerank,
        )


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenizer used by sparse retrieval and heuristics."""
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def _extract_years(text: str) -> list[str]:
    return YEAR_RE.findall(text or "")


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def _normalize_punctuation(text: str) -> str:
    normalized = re.sub(r"[^\w\s]", " ", text or "")
    return _normalize_whitespace(normalized)


def _is_financial_query(query: str) -> bool:
    query_lower = (query or "").lower()
    if _extract_years(query_lower):
        return True
    return any(term in query_lower for term in FINANCIAL_TERMS)


def _extract_metric_terms(query: str) -> list[str]:
    query_lower = (query or "").lower()
    return [term for term in FINANCIAL_TERMS if term in query_lower]


def _extract_company_hints(query: str) -> list[str]:
    query_lower = (query or "").lower()
    hints: list[str] = []
    for company_aliases in COMPANY_HINTS.values():
        for alias in company_aliases:
            if alias in query_lower and alias not in hints:
                hints.append(alias)
    return hints


def _section_targets(query: str) -> list[str]:
    query_lower = (query or "").lower()
    targets: list[str] = []
    for trigger, sections in SECTION_HINTS.items():
        if trigger in query_lower:
            for section in sections:
                if section not in targets:
                    targets.append(section)
    return targets


def expand_query(query: str) -> list[str]:
    """Create conservative deterministic query variants for recall."""
    variants: list[str] = []

    def add_variant(value: str) -> None:
        normalized = _normalize_whitespace(value)
        if normalized and normalized not in variants:
            variants.append(normalized)

    add_variant(query)
    add_variant((query or "").lower())
    add_variant(_normalize_punctuation(query))

    if _is_financial_query(query):
        years = _extract_years(query)
        metric_terms = _extract_metric_terms(query)
        company_terms = _extract_company_hints(query)
        lower_query = (query or "").lower()
        if (
            years
            and metric_terms
            and "consolidated statements" not in lower_query
            and "statement of comprehensive income" not in lower_query
        ):
            keyword_variant = " ".join(
                [*company_terms[:1], *metric_terms[:2], *years, "consolidated statements of operations"]
            )
            add_variant(keyword_variant)

    return variants


def _stable_fallback_key(document: Document) -> str:
    metadata = document.metadata or {}
    source = metadata.get("source") or metadata.get("doc_source") or "unknown"
    source_path = metadata.get("source_path") or ""
    page = metadata.get("page") or metadata.get("page_start") or "na"
    section = metadata.get("section") or ""
    digest = hashlib.md5((document.page_content or "").encode("utf-8")).hexdigest()[:12]
    return f"{source}|{source_path}|{page}|{section}|{digest}"


def _document_key(document: Document) -> str:
    metadata = document.metadata or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id is not None:
        return f"chunk:{chunk_id}"
    return _stable_fallback_key(document)


def _candidate_identifier(candidate: SearchCandidate) -> str:
    metadata = candidate.document.metadata or {}
    chunk_id = metadata.get("chunk_id")
    return str(chunk_id) if chunk_id is not None else candidate.key


def load_chunk_documents(chunks_jsonl_path: str | Path | None = None) -> list[Document]:
    """Load documents from chunks.jsonl written by ingest.py."""
    path = Path(chunks_jsonl_path or CHUNKS_JSONL_PATH)
    cache_key = str(path.resolve())
    if cache_key in _CHUNK_CACHE:
        return list(_CHUNK_CACHE[cache_key])

    if not path.exists():
        raise FileNotFoundError(
            f"chunks.jsonl not found at {path}. Run ingestion first or set CHUNKS_JSONL_PATH correctly."
        )

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

    _CHUNK_CACHE[cache_key] = list(documents)
    return documents


def _get_local_sparse_retriever(chunks_jsonl_path: str | Path | None = None) -> LocalBM25Retriever:
    path = Path(chunks_jsonl_path or CHUNKS_JSONL_PATH)
    cache_key = str(path.resolve())
    if cache_key in _SPARSE_CACHE:
        return _SPARSE_CACHE[cache_key]

    retriever = LocalBM25Retriever.from_documents(load_chunk_documents(path))
    _SPARSE_CACHE[cache_key] = retriever
    return retriever


def build_ensemble_retriever(
    retrievers: Sequence[Any],
    weights: Sequence[float],
    prefer_langchain: bool = True,
) -> Any:
    """Compatibility helper for callers that want a simple ensemble retriever."""
    if prefer_langchain and LangChainEnsembleRetriever is not None:
        return LangChainEnsembleRetriever(retrievers=list(retrievers), weights=list(weights))
    return LocalEnsembleRetriever(retrievers=retrievers, weights=weights)


def get_vectorstore(persist_directory: str | Path | None = None):
    """Load the local Chroma vectorstore, downloading from GCS if needed."""
    load_dotenv()

    persist_dir = Path(persist_directory or CHROMA_PERSIST_DIR)
    if not persist_dir.exists() or not any(persist_dir.iterdir()):
        persist_dir.mkdir(parents=True, exist_ok=True)
        try:
            from src.storage.gcs import download_prefix
        except ImportError as exc:
            raise RuntimeError(
                f"Vectorstore directory {persist_dir} is empty and GCS download support is unavailable."
            ) from exc
        download_prefix(GCS_BUCKET, VECTOR_DB_PREFIX, str(persist_dir))

    try:
        from langchain_community.vectorstores import Chroma
        from langchain_google_vertexai import VertexAIEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "Dense retrieval requires langchain-community and langchain-google-vertexai."
        ) from exc

    embeddings = VertexAIEmbeddings(model=EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def get_dense_retriever(k: int = 6, persist_directory: str | Path | None = None):
    """Return a LangChain-compatible dense retriever."""
    vectorstore = get_vectorstore(persist_directory=persist_directory)
    try:
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k},
        )
    except Exception:
        return vectorstore.as_retriever(search_kwargs={"k": k})


def get_bm25_retriever(
    k: int = 6,
    chunks_jsonl_path: str | Path | None = None,
    prefer_langchain: bool = True,
) -> Any:
    """Return a sparse retriever, preferring LangChain BM25 when available."""
    documents = load_chunk_documents(chunks_jsonl_path=chunks_jsonl_path)
    if prefer_langchain and LangChainBM25Retriever is not None:
        retriever = LangChainBM25Retriever.from_documents(documents)
        retriever.k = k
        return retriever

    retriever = _get_local_sparse_retriever(chunks_jsonl_path=chunks_jsonl_path)
    retriever.k = k
    return retriever


def get_hybrid_retriever(
    k: int = 6,
    weights: tuple[float, float] = (0.5, 0.5),
    persist_directory: str | Path | None = None,
    chunks_jsonl_path: str | Path | None = None,
    prefer_langchain: bool = True,
    dense_k: int | None = None,
    sparse_k: int | None = None,
    use_query_expansion: bool = True,
    use_rerank: bool = True,
) -> HybridRetriever:
    """Return a wrapper retriever that exposes invoke(query) over hybrid_search()."""
    return HybridRetriever(
        k=k,
        weights=weights,
        persist_directory=persist_directory,
        chunks_jsonl_path=chunks_jsonl_path,
        prefer_langchain=prefer_langchain,
        dense_k=dense_k,
        sparse_k=sparse_k,
        use_query_expansion=use_query_expansion,
        use_rerank=use_rerank,
    )


def _normalize_dense_distance(distance: float | None) -> float | None:
    if distance is None:
        return None
    try:
        numeric = float(distance)
    except (TypeError, ValueError):
        return None
    return 1.0 / (1.0 + max(numeric, 0.0))


def _retrieve_dense_candidates(
    query: str,
    *,
    k: int,
    persist_directory: str | Path | None = None,
) -> list[tuple[Document, int, float | None]]:
    vectorstore = get_vectorstore(persist_directory=persist_directory)
    fetch_k = max(k * 2, k)
    similarity_scores: dict[str, float | None] = {}

    try:
        similarity_pairs = vectorstore.similarity_search_with_score(query, k=fetch_k)
        for document, raw_score in similarity_pairs:
            similarity_scores[_document_key(document)] = _normalize_dense_distance(raw_score)
    except Exception:
        similarity_pairs = []

    try:
        documents = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
    except Exception:
        if similarity_pairs:
            documents = [document for document, _ in similarity_pairs[:k]]
        else:
            try:
                documents = vectorstore.similarity_search(query, k=k)
            except Exception as exc:
                raise RuntimeError(f"Dense retrieval failed for query '{query}': {exc}") from exc

    ranked_results: list[tuple[Document, int, float | None]] = []
    for rank, document in enumerate(documents, start=1):
        score = similarity_scores.get(_document_key(document))
        ranked_results.append((document, rank, score))
    return ranked_results


def _retrieve_sparse_candidates(
    query: str,
    *,
    k: int,
    chunks_jsonl_path: str | Path | None = None,
) -> list[tuple[Document, int, float]]:
    retriever = _get_local_sparse_retriever(chunks_jsonl_path=chunks_jsonl_path)
    scored = retriever.search(query, k=k)
    return [(document, rank, score) for rank, (document, score) in enumerate(scored, start=1)]


def _apply_rrf(
    candidates: dict[str, SearchCandidate],
    *,
    results: Sequence[tuple[Document, int, float | None]],
    retriever_type: str,
    weight: float,
    expansion_weight: float,
) -> None:
    for document, rank, score in results:
        key = _document_key(document)
        candidate = candidates.get(key)
        if candidate is None:
            candidate = SearchCandidate(document=document, key=key)
            candidates[key] = candidate
        if retriever_type == "dense":
            if candidate.dense_rank is None or rank < candidate.dense_rank:
                candidate.dense_rank = rank
            if score is not None:
                candidate.dense_score = max(candidate.dense_score or 0.0, float(score))
        else:
            if candidate.sparse_rank is None or rank < candidate.sparse_rank:
                candidate.sparse_rank = rank
            if score is not None:
                candidate.sparse_score = max(candidate.sparse_score or 0.0, float(score))
        candidate.fused_score += (weight * expansion_weight) / (rank + RRF_K)


def _metadata_text(document: Document) -> str:
    metadata = document.metadata or {}
    values = [
        str(metadata.get("section", "")),
        str(metadata.get("chunk_kind", "")),
        str(metadata.get("element_type", "")),
        str(metadata.get("content_format", "")),
        str(metadata.get("source", metadata.get("doc_source", ""))),
    ]
    return " ".join(values).lower()


def _rerank_documents(candidates: Iterable[SearchCandidate], query: str) -> list[SearchCandidate]:
    """Apply gentle local boosts for years, metadata, and filing-specific signals."""
    query_lower = (query or "").lower()
    requested_years = _extract_years(query)
    financial_query = _is_financial_query(query)
    metric_terms = _extract_metric_terms(query)
    company_hints = _extract_company_hints(query)
    section_targets = _section_targets(query)

    reranked: list[SearchCandidate] = []
    for candidate in candidates:
        boost = 0.0
        reasons: list[str] = []
        text_lower = (candidate.document.page_content or "").lower()
        metadata_text = _metadata_text(candidate.document)
        metadata = candidate.document.metadata or {}

        matched_years = [year for year in requested_years if year in text_lower]
        if matched_years:
            increment = 0.04 * len(matched_years)
            boost += increment
            reasons.append(f"year_match:{','.join(matched_years)}(+{increment:.2f})")
        if requested_years and len(set(matched_years)) == len(set(requested_years)):
            boost += 0.08
            reasons.append("all_years_present(+0.08)")

        if financial_query and metadata.get("chunk_kind") == "table_aligned":
            boost += 0.08
            reasons.append("table_aligned(+0.08)")
        elif financial_query and "table" in metadata_text:
            boost += 0.03
            reasons.append("table_metadata(+0.03)")

        if financial_query and "consolidated statements of operations" in metadata_text:
            boost += 0.06
            reasons.append("statements_section(+0.06)")
        if "risk" in query_lower and "risk factors" in metadata_text:
            boost += 0.06
            reasons.append("risk_section(+0.06)")

        metric_overlap = [term for term in metric_terms if term in text_lower]
        if metric_overlap:
            increment = min(0.12, 0.03 * len(metric_overlap))
            boost += increment
            reasons.append(f"metric_overlap:{','.join(metric_overlap)}(+{increment:.2f})")

        if section_targets and any(target in metadata_text for target in section_targets):
            boost += 0.04
            reasons.append("section_target(+0.04)")

        if company_hints and any(hint in text_lower or hint in metadata_text for hint in company_hints):
            boost += 0.03
            reasons.append("company_hint(+0.03)")

        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        if page_start is not None and page_end is not None and page_start == page_end:
            boost += 0.02
            reasons.append("narrow_page_span(+0.02)")

        candidate.rerank_boost = boost
        candidate.rerank_reasons = reasons
        reranked.append(candidate)

    reranked.sort(key=lambda item: (item.final_score, item.fused_score), reverse=True)
    return reranked


def _dedupe_documents(candidates: Iterable[SearchCandidate]) -> list[SearchCandidate]:
    """Deduplicate by chunk_id or stable fallback key, keeping the best candidate."""
    deduped: dict[str, SearchCandidate] = {}
    for candidate in candidates:
        key = candidate.key
        current = deduped.get(key)
        if current is None or candidate.final_score > current.final_score:
            deduped[key] = candidate
    return list(deduped.values())


def _fuse_ranked_lists(
    query: str,
    *,
    weights: tuple[float, float],
    persist_directory: str | Path | None,
    chunks_jsonl_path: str | Path | None,
    dense_k: int,
    sparse_k: int,
    use_query_expansion: bool,
) -> tuple[list[SearchCandidate], dict[str, Any]]:
    expanded_queries = expand_query(query) if use_query_expansion else [_normalize_whitespace(query)]
    candidates: dict[str, SearchCandidate] = {}
    dense_error: str | None = None
    sparse_error: str | None = None
    raw_dense_count = 0
    raw_sparse_count = 0

    for expansion_index, expanded_query in enumerate(expanded_queries):
        expansion_weight = 1.0 if expansion_index == 0 else 0.9
        try:
            dense_results = _retrieve_dense_candidates(
                expanded_query,
                k=dense_k,
                persist_directory=persist_directory,
            )
            raw_dense_count += len(dense_results)
            _apply_rrf(
                candidates,
                results=dense_results,
                retriever_type="dense",
                weight=weights[0],
                expansion_weight=expansion_weight,
            )
        except Exception as exc:
            dense_error = str(exc)

        try:
            sparse_results = _retrieve_sparse_candidates(
                expanded_query,
                k=sparse_k,
                chunks_jsonl_path=chunks_jsonl_path,
            )
            raw_sparse_count += len(sparse_results)
            _apply_rrf(
                candidates,
                results=sparse_results,
                retriever_type="sparse",
                weight=weights[1],
                expansion_weight=expansion_weight,
            )
        except Exception as exc:
            sparse_error = str(exc)

    if not candidates:
        if dense_error and sparse_error:
            raise RuntimeError(
                f"Hybrid retrieval failed. Dense error: {dense_error}. Sparse error: {sparse_error}."
            )
        if dense_error:
            raise RuntimeError(f"Dense retrieval failed and sparse produced no candidates: {dense_error}")
        if sparse_error:
            raise RuntimeError(f"Sparse retrieval failed and dense produced no candidates: {sparse_error}")
        raise RuntimeError("Hybrid retrieval produced no candidates.")

    debug_info = {
        "expanded_queries": expanded_queries,
        "candidate_counts": {
            "dense": raw_dense_count,
            "sparse": raw_sparse_count,
            "fused": len(candidates),
            "final": 0,
        },
        "errors": {
            "dense": dense_error,
            "sparse": sparse_error,
        },
    }
    return list(candidates.values()), debug_info


@_ls_traceable(run_type="retriever", name="hybrid_search")
def hybrid_search(
    query: str,
    k: int = 6,
    weights: tuple[float, float] = (0.5, 0.5),
    persist_directory: str | Path | None = None,
    chunks_jsonl_path: str | Path | None = None,
    prefer_langchain: bool = True,  # kept for backwards compatibility
    dense_k: int | None = None,
    sparse_k: int | None = None,
    use_query_expansion: bool = True,
    use_rerank: bool = True,
    return_debug: bool = False,
) -> list[Document] | tuple[list[Document], dict[str, Any]]:
    """
    Multi-stage hybrid retrieval with conservative query expansion, RRF fusion, dedupe, and gentle reranking.

    Backwards compatible with existing callers:
        hybrid_search(query, k=6, weights=(0.5, 0.5))
    """
    del prefer_langchain  # compatibility-only; core pipeline is self-contained

    final_k = k
    dense_pool = dense_k or max(DEFAULT_DENSE_K, final_k * 2)
    sparse_pool = sparse_k or max(DEFAULT_SPARSE_K, final_k * 2)

    candidates, debug_info = _fuse_ranked_lists(
        query,
        weights=weights,
        persist_directory=persist_directory,
        chunks_jsonl_path=chunks_jsonl_path,
        dense_k=dense_pool,
        sparse_k=sparse_pool,
        use_query_expansion=use_query_expansion,
    )

    deduped = _dedupe_documents(candidates)
    ranked = _rerank_documents(deduped, query) if use_rerank else sorted(
        deduped, key=lambda item: item.fused_score, reverse=True
    )
    final_candidates = ranked[:final_k]
    documents = [candidate.document for candidate in final_candidates]

    debug_info["candidate_counts"]["final"] = len(documents)
    debug_info["document_diagnostics"] = [
        {
            "chunk_id": _candidate_identifier(candidate),
            "dense_rank": candidate.dense_rank,
            "dense_score": candidate.dense_score,
            "sparse_rank": candidate.sparse_rank,
            "sparse_score": candidate.sparse_score,
            "fused_score": candidate.fused_score,
            "rerank_boost_reason": "; ".join(candidate.rerank_reasons),
        }
        for candidate in final_candidates
    ]

    if return_debug:
        return documents, debug_info
    return documents


def expand_neighbor_context(
    docs: list[Document],
    all_chunks: list[Document] | None = None,
    chunks_jsonl_path: str | Path | None = None,
    window_pages: int = 1,
    max_neighbors_per_doc: int = 2,
) -> list[Document]:
    """Supplement top results with adjacent narrative chunks from the same section/page.

    Returns original docs first, then unique neighbors appended.
    Never raises — returns original docs unchanged on any error.
    """
    try:
        if all_chunks is None:
            all_chunks = load_chunk_documents(chunks_jsonl_path)
        if not all_chunks or not docs:
            return list(docs)

        existing_keys = {_document_key(d) for d in docs}
        neighbors: list[Document] = []

        for doc in docs:
            doc_meta = doc.metadata or {}
            doc_source = doc_meta.get("source") or doc_meta.get("doc_source")
            doc_section = doc_meta.get("section")
            doc_page = doc_meta.get("page")
            added_for_doc = 0

            for candidate in all_chunks:
                if added_for_doc >= max_neighbors_per_doc:
                    break
                c_meta = candidate.metadata or {}

                # Only narrative neighbors
                if c_meta.get("chunk_kind") not in ("text", None):
                    continue

                c_source = c_meta.get("source") or c_meta.get("doc_source")
                if c_source != doc_source:
                    continue

                c_section = c_meta.get("section")
                c_page = c_meta.get("page")

                # Must share section or be on an adjacent page
                section_match = (doc_section and c_section and c_section == doc_section)
                page_match = False
                if doc_page is not None and c_page is not None:
                    try:
                        page_match = abs(int(c_page) - int(doc_page)) <= window_pages
                    except (TypeError, ValueError):
                        pass

                if not section_match and not page_match:
                    continue

                c_key = _document_key(candidate)
                if c_key in existing_keys:
                    continue

                existing_keys.add(c_key)
                neighbors.append(candidate)
                added_for_doc += 1

        return list(docs) + neighbors
    except Exception:
        return list(docs)


if __name__ == "__main__":
    sample_query = "What were Apple's total net sales in 2024 and 2023?"
    docs, debug = hybrid_search(sample_query, k=5, return_debug=True)

    print("\n==============================")
    print("QUERY:", sample_query)
    print("==============================")
    print("Expanded queries:", debug["expanded_queries"])
    print("Candidate counts:", debug["candidate_counts"])

    for index, document in enumerate(docs, start=1):
        metadata = document.metadata or {}
        print("\n==============================")
        print("Rank:", index)
        print("Chunk ID:", metadata.get("chunk_id"))
        print("Page:", metadata.get("page"))
        print("Section:", metadata.get("section"))
        print("Source:", metadata.get("source"))
        print("\n----- CHUNK TEXT -----\n")
        print(document.page_content[:800])
        print("\n----------------------")
