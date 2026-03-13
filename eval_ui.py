import json
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional for local UI bootstrapping
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

from src.config import CHUNKS_JSONL_PATH, INGEST_SUMMARY_PATH, PARSED_ELEMENTS_PATH
from src.retrieval.retrieval import hybrid_search


load_dotenv()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def render_content(body: str, content_format: str) -> None:
    if content_format == "html":
        st.markdown(body, unsafe_allow_html=True)
        with st.expander("Show raw HTML"):
            st.code(body, language="html")
        return
    st.text(body)


def render_retrieval_card(index: int, content: str, metadata: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown(f"**Rank {index}**")
        st.caption(
            f"chunk_kind: {metadata.get('chunk_kind', 'N/A')} | "
            f"page: {metadata.get('page', 'N/A')} | "
            f"section: {metadata.get('section', 'N/A')} | "
            f"source: {metadata.get('source', metadata.get('doc_source', 'N/A'))}"
        )
        render_content(content, metadata.get("content_format", "text"))
        with st.expander("Metadata"):
            st.json(metadata)


st.set_page_config(page_title="RAG Eval UI", layout="wide")
st.title("RAG Artifact and Retrieval Inspector")

summary_path = Path(INGEST_SUMMARY_PATH)
parsed_path = Path(PARSED_ELEMENTS_PATH)
chunks_path = Path(CHUNKS_JSONL_PATH)

with st.sidebar:
    st.subheader("Artifact Paths")
    st.code(str(summary_path))
    st.code(str(parsed_path))
    st.code(str(chunks_path))

summary = load_summary(summary_path)
parsed_elements = load_jsonl(parsed_path)
chunks = load_jsonl(chunks_path)

if not parsed_elements and not chunks:
    st.warning("No ingestion artifacts found. Run `python src/ingestion/ingest.py` first.")
    st.stop()

if summary:
    col1, col2, col3 = st.columns(3)
    col1.metric("PDFs", summary.get("pdf_count", 0))
    col2.metric("Parsed Elements", summary.get("parsed_element_count", 0))
    col3.metric("Chunks", summary.get("chunk_count", 0))

source_options = sorted(
    {
        row.get("source")
        or row.get("metadata", {}).get("source")
        or row.get("metadata", {}).get("doc_source")
        for row in [*parsed_elements, *chunks]
        if row
    }
)
selected_source = st.selectbox("Source Document", ["All"] + source_options)

element_types = sorted({row.get("element_type") for row in parsed_elements if row.get("element_type")})
selected_element_types = st.multiselect(
    "Parsed Element Types",
    element_types,
    default=element_types,
)

chunk_types = sorted(
    {
        row.get("metadata", {}).get("chunk_kind")
        for row in chunks
        if row.get("metadata", {}).get("chunk_kind")
    }
)
selected_chunk_types = st.multiselect(
    "Chunk Types",
    chunk_types,
    default=chunk_types,
)

artifact_query = st.text_input("Artifact Text Filter")


def matches_source(row: dict[str, Any]) -> bool:
    if selected_source == "All":
        return True
    source = row.get("source") or row.get("metadata", {}).get("source") or row.get("metadata", {}).get("doc_source")
    return source == selected_source


def matches_query(row: dict[str, Any], body_key: str) -> bool:
    if not artifact_query:
        return True
    haystack = " ".join(
        [
            str(row.get(body_key, "")),
            str(row.get("text", "")),
            json.dumps(row.get("metadata", {}), ensure_ascii=False),
        ]
    ).lower()
    return artifact_query.lower() in haystack


filtered_elements = [
    row
    for row in parsed_elements
    if matches_source(row)
    and matches_query(row, "content")
    and (not selected_element_types or row.get("element_type") in selected_element_types)
]
filtered_chunks = [
    row
    for row in chunks
    if matches_source(row)
    and matches_query(row, "text")
    and (
        not selected_chunk_types
        or row.get("metadata", {}).get("chunk_kind") in selected_chunk_types
    )
]

tab_elements, tab_chunks, tab_retrieval = st.tabs(
    ["Parsed Elements", "Chunks", "Retrieval & Traceability Testing"]
)

with tab_elements:
    st.caption(f"{len(filtered_elements)} parsed elements")
    for row in filtered_elements:
        title = (
            f"{row.get('element_type', 'Element')} | "
            f"page {row.get('page_number', 'N/A')} | "
            f"section {row.get('section', 'N/A')}"
        )
        with st.expander(title):
            st.json(
                {
                    "element_id": row.get("element_id"),
                    "source": row.get("source"),
                    "source_path": row.get("source_path"),
                    "content_format": row.get("content_format"),
                    "metadata": row.get("metadata", {}),
                }
            )
            render_content(row.get("content", ""), row.get("content_format", "text"))

with tab_chunks:
    st.caption(f"{len(filtered_chunks)} chunks")
    for row in filtered_chunks:
        metadata = row.get("metadata", {})
        title = (
            f"chunk {metadata.get('chunk_id', 'N/A')} | "
            f"{metadata.get('chunk_kind', 'unknown')} | "
            f"page {metadata.get('page', 'N/A')} | "
            f"section {metadata.get('section', 'N/A')}"
        )
        with st.expander(title):
            st.json(metadata)
            render_content(row.get("text", ""), metadata.get("content_format", "text"))

with tab_retrieval:
    st.subheader("Hybrid Search")
    retrieval_query = st.text_input(
        "Query",
        placeholder="What were Apple's net sales in 2024?",
        key="retrieval_query",
    )
    top_k = st.slider("Top-K", min_value=1, max_value=12, value=6, key="retrieval_top_k")

    if st.button("Run Hybrid Retriever", type="primary"):
        if not retrieval_query.strip():
            st.warning("Enter a query to test retrieval.")
        else:
            try:
                retrieved_docs = hybrid_search(retrieval_query, k=top_k)
                st.caption(f"{len(retrieved_docs)} retrieved chunks")
                for index, document in enumerate(retrieved_docs, start=1):
                    render_retrieval_card(index, document.page_content, document.metadata or {})
            except Exception as exc:
                st.error(f"Hybrid retrieval failed: {exc}")
