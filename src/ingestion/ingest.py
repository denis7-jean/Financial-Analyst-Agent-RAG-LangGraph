import json
import re
from io import StringIO
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional for unit tests
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - lightweight fallback for local tests
    @dataclass
    class Document:
        page_content: str
        metadata: dict[str, Any] = field(default_factory=dict)

    class RecursiveCharacterTextSplitter:
        def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            separators: Optional[Sequence[str]] = None,
        ) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def create_documents(
            self,
            texts: Sequence[str],
            metadatas: Optional[Sequence[dict[str, Any]]] = None,
        ) -> List[Document]:
            docs: List[Document] = []
            metadatas = metadatas or [{} for _ in texts]
            for text, metadata in zip(texts, metadatas):
                if len(text) <= self.chunk_size:
                    docs.append(Document(page_content=text, metadata=dict(metadata)))
                    continue

                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    docs.append(Document(page_content=text[start:end], metadata=dict(metadata)))
                    if end >= len(text):
                        break
                    start = max(end - self.chunk_overlap, start + 1)
            return docs

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - runtime dependency for aligned table parsing
    raise ImportError(
        "pandas is required for aligned HTML table parsing in ingestion. "
        "Install with `pip install pandas lxml`."
    ) from exc

from src.config import (
    CHROMA_PERSIST_DIR,
    CHUNKS_JSONL_PATH,
    DATA_DIR,
    EMBED_MODEL,
    GCS_BUCKET,
    INGEST_SUMMARY_PATH,
    PARSED_ELEMENTS_PATH,
    UPLOAD_VECTOR_DB_TO_GCS,
    VECTOR_DB_PREFIX,
)


TABLE_ELEMENT_TYPES = {"Table"}
SECTION_ELEMENT_TYPES = {"Title", "Header"}
TEXT_ELEMENT_TYPES = {
    "CompositeElement",
    "NarrativeText",
    "ListItem",
    "Text",
    "UncategorizedText",
}
EMBEDDING_BATCH_SIZE = 10


@dataclass
class ParsedElement:
    source: str
    source_path: str
    element_type: str
    text: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    content: Optional[str] = None
    content_format: str = "text"
    element_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "source": self.source,
            "source_path": self.source_path,
            "element_type": self.element_type,
            "page_number": self.page_number,
            "section": self.section,
            "text": self.text,
            "content": self.content or self.text,
            "content_format": self.content_format,
            "metadata": self.metadata,
        }


def _get_value(container: Any, key: str, default: Any = None) -> Any:
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ").strip()
    return text


def _convert_html_table_to_aligned_text(html_str: str, section_title: str) -> str:
    if not html_str:
        return ""

    try:
        dfs = pd.read_html(StringIO(html_str))
    except Exception:
        fallback_section = section_title or "Unknown"
        return (
            "[TABLE ALIGNMENT]\n"
            f"Table Section: {fallback_section}\n"
            "Parsing failed. Raw HTML follows:\n"
            f"{html_str}"
        )

    if not dfs:
        return html_str

    df = dfs[0].fillna("")
    if hasattr(df.columns, "to_flat_index"):
        flat_columns = df.columns.to_flat_index()
        headers = [
            " ".join(str(part).strip() for part in column if str(part).strip() and str(part) != "nan")
            if isinstance(column, tuple)
            else str(column).strip()
            for column in flat_columns
        ]
    else:
        headers = [str(column).strip() for column in df.columns]

    headers = [header or f"Column {idx}" for idx, header in enumerate(headers)]
    lines = [
        "[TABLE ALIGNMENT]",
        f"Table Section: {section_title or 'Unknown'}",
        "Headers: | " + " | ".join(headers) + " |",
        "-" * max(50, len(" | ".join(headers)) + 12),
    ]

    for row_index, row in enumerate(df.itertuples(index=False, name=None), start=1):
        row_values = [str(value).strip() if value is not None else "" for value in row]
        lines.append(f"Row {row_index}: | " + " | ".join(row_values) + " |")

    return "\n".join(lines)


_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _extract_table_metadata(html_str: str) -> dict[str, Any]:
    """Extract structural metadata from an HTML table for downstream retrieval."""
    if not html_str:
        return {}
    try:
        dfs = pd.read_html(StringIO(html_str))
    except Exception:
        return {}
    if not dfs:
        return {}

    df = dfs[0].fillna("")

    if hasattr(df.columns, "to_flat_index"):
        flat_columns = df.columns.to_flat_index()
        headers = [
            " ".join(str(part).strip() for part in col if str(part).strip() and str(part) != "nan")
            if isinstance(col, tuple) else str(col).strip()
            for col in flat_columns
        ]
    else:
        headers = [str(col).strip() for col in df.columns]
    headers = [h for h in headers if h]

    year_headers = sorted({y for h in headers for y in _YEAR_RE.findall(h)})

    first_column_candidates: list[str] = []
    if len(df.columns) > 0:
        for val in df.iloc[:, 0]:
            s = str(val).strip()
            if s and s.lower() != "nan":
                first_column_candidates.append(s)

    return {
        "table_headers": headers,
        "year_headers": year_headers,
        "first_column_candidates": first_column_candidates,
        "table_row_count": len(df),
    }


def _extract_table_rows(
    html_str: str,
    chunk_id: int,
    source: str,
    page: Optional[int],
    section: Optional[str],
) -> List[dict[str, Any]]:
    """Extract row-level records from an HTML table for rule-assisted matching."""
    if not html_str:
        return []
    try:
        dfs = pd.read_html(StringIO(html_str))
    except Exception:
        return []
    if not dfs:
        return []

    df = dfs[0].fillna("")

    if hasattr(df.columns, "to_flat_index"):
        flat_columns = df.columns.to_flat_index()
        headers = [
            " ".join(str(part).strip() for part in col if str(part).strip() and str(part) != "nan")
            if isinstance(col, tuple) else str(col).strip()
            for col in flat_columns
        ]
    else:
        headers = [str(col).strip() for col in df.columns]

    year_headers = sorted({y for h in headers for y in _YEAR_RE.findall(h)})

    rows: List[dict[str, Any]] = []
    for row_index, row in enumerate(df.itertuples(index=False, name=None)):
        values = [str(v).strip() for v in row]
        first_col = values[0] if values else ""

        if not first_col:
            continue
        # Skip sub-headers: all caps with no digits
        if first_col == first_col.upper() and not any(c.isdigit() for c in first_col):
            continue

        row_label_normalized = " ".join(first_col.lower().split())
        rows.append({
            "chunk_id": chunk_id,
            "source": source,
            "page": page,
            "section": section,
            "row_label": first_col,
            "row_label_normalized": row_label_normalized,
            "headers": headers,
            "year_headers": year_headers,
            "values": values,
            "row_index": row_index,
        })

    return rows


def _normalize_element(raw_element: Any, source_path: Path, index: int, current_section: Optional[str]) -> ParsedElement:
    metadata = _get_value(raw_element, "metadata", {}) or {}
    element_type = (
        _get_value(raw_element, "category")
        or _get_value(raw_element, "element_type")
        or raw_element.__class__.__name__
    )
    text = _clean_text(_get_value(raw_element, "text"))
    if not text:
        text = _clean_text(raw_element)

    page_number = _get_value(metadata, "page_number")
    section = current_section
    if element_type in SECTION_ELEMENT_TYPES and text:
        section = text

    table_html = _clean_text(_get_value(metadata, "text_as_html"))
    content_format = "html" if element_type in TABLE_ELEMENT_TYPES and table_html else "text"
    content = table_html if content_format == "html" else text

    serialized_metadata = {
        "page_number": page_number,
        "filename": _get_value(metadata, "filename"),
        "category_depth": _get_value(metadata, "category_depth"),
        "parent_id": _get_value(metadata, "parent_id"),
    }
    if table_html:
        serialized_metadata["table_html"] = table_html

    return ParsedElement(
        source=source_path.name,
        source_path=str(source_path),
        element_type=element_type,
        text=text,
        page_number=page_number,
        section=section,
        content=content,
        content_format=content_format,
        element_id=f"{source_path.stem}:{index}",
        metadata={k: v for k, v in serialized_metadata.items() if v not in (None, "")},
    )


def normalize_parsed_elements(raw_elements: Sequence[Any], source_path: str | Path) -> List[ParsedElement]:
    source = Path(source_path)
    normalized: List[ParsedElement] = []
    current_section: Optional[str] = None

    for index, raw_element in enumerate(raw_elements):
        element = _normalize_element(raw_element, source, index=index, current_section=current_section)
        normalized.append(element)
        if element.section:
            current_section = element.section

    return normalized


def parse_pdf_elements(pdf_path: str | Path) -> List[ParsedElement]:
    pdf = Path(pdf_path)
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:
        raise ImportError(
            "unstructured is required for hi-res PDF ingestion. "
            "Install with `pip install \"unstructured[pdf]\"`."
        ) from exc

    raw_elements = partition_pdf(
        filename=str(pdf),
        strategy="hi_res",
        infer_table_structure=True,
        include_page_breaks=False,
    )
    return normalize_parsed_elements(raw_elements, source_path=pdf)


def build_chunks(
    parsed_elements: Sequence[ParsedElement],
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: List[Document] = []
    text_buffer: List[ParsedElement] = []
    current_section: Optional[str] = None

    def flush_text_buffer() -> None:
        nonlocal text_buffer
        if not text_buffer:
            return

        body = "\n\n".join(element.text for element in text_buffer if element.text)
        if not body:
            text_buffer = []
            return

        pages = [element.page_number for element in text_buffer if element.page_number is not None]
        section = next((element.section for element in text_buffer if element.section), current_section)
        if section:
            body = f"{section}\n\n{body}"

        base_metadata = {
            "source": text_buffer[0].source,
            "doc_source": text_buffer[0].source,
            "source_path": text_buffer[0].source_path,
            "page": min(pages) if pages else None,
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "section": section,
            "element_type": "CompositeText",
            "content_format": "text",
            "chunk_kind": "text",
            "element_count": len(text_buffer),
        }

        for index, doc in enumerate(splitter.create_documents([body], metadatas=[base_metadata])):
            doc.metadata["chunk_index_in_group"] = index
            chunks.append(doc)

        text_buffer = []

    for element in parsed_elements:
        if element.section:
            current_section = element.section

        if element.element_type in SECTION_ELEMENT_TYPES:
            flush_text_buffer()
            continue

        if element.element_type in TABLE_ELEMENT_TYPES:
            flush_text_buffer()
            table_content = element.content or element.text
            if not table_content:
                continue
            table_html = element.metadata.get("table_html")
            aligned_text = (
                _convert_html_table_to_aligned_text(
                    table_html,
                    element.section or current_section or "",
                )
                if table_html
                else table_content
            )
            table_meta = _extract_table_metadata(table_html) if table_html else {}
            # Convert empty lists to None for ChromaDB compatibility
            table_meta = {k: (v if v else None) for k, v in table_meta.items()}
            chunks.append(
                Document(
                    page_content=aligned_text,
                    metadata={
                        "source": element.source,
                        "doc_source": element.source,
                        "source_path": element.source_path,
                        "page": element.page_number,
                        "page_start": element.page_number,
                        "page_end": element.page_number,
                        "section": element.section or current_section,
                        "element_type": element.element_type,
                        "content_format": element.content_format,
                        "chunk_kind": "table_aligned",
                        "element_count": 1,
                        "table_html": table_html,
                        **table_meta,
                    },
                )
            )
            continue

        if element.element_type in TEXT_ELEMENT_TYPES or element.text:
            text_buffer.append(element)

    flush_text_buffer()
    return assign_chunk_ids(chunks)


def assign_chunk_ids(chunks: Sequence[Document]) -> List[Document]:
    assigned: List[Document] = []
    for chunk_id, document in enumerate(chunks):
        document.metadata = dict(document.metadata or {})
        document.metadata["chunk_id"] = chunk_id
        assigned.append(document)
    return assigned


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_parsed_elements(elements: Sequence[ParsedElement], output_path: Path) -> None:
    _write_jsonl(output_path, (element.to_record() for element in elements))


def _write_chunks_jsonl(chunks: Sequence[Document], output_path: Path) -> None:
    records = (
        {
            "id": document.metadata.get("chunk_id"),
            "text": document.page_content,
            "metadata": document.metadata,
        }
        for document in chunks
    )
    _write_jsonl(output_path, records)


def _write_summary(
    summary_path: Path,
    pdf_files: Sequence[Path],
    parsed_elements: Sequence[ParsedElement],
    chunks: Sequence[Document],
    persist_directory: Path,
    chunks_jsonl_path: Path,
    parsed_elements_path: Path,
    table_rows_path: Optional[Path] = None,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "pdf_count": len(pdf_files),
        "sources": [pdf.name for pdf in pdf_files],
        "parsed_element_count": len(parsed_elements),
        "chunk_count": len(chunks),
        "persist_directory": str(persist_directory),
        "chunks_jsonl_path": str(chunks_jsonl_path),
        "parsed_elements_path": str(parsed_elements_path),
    }
    if table_rows_path is not None:
        summary["table_rows_path"] = str(table_rows_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _batched_documents(documents: Sequence[Document], batch_size: int = EMBEDDING_BATCH_SIZE) -> Iterable[Sequence[Document]]:
    for start in range(0, len(documents), batch_size):
        yield documents[start:start + batch_size]


def ingest_pdf_files(
    pdf_files: Sequence[str | Path],
    persist_directory: str | Path | None = None,
    collection_name: str = "10k_filings",
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    chunks_jsonl_path: str | Path | None = None,
    parsed_elements_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> int:
    load_dotenv()

    pdf_paths = [Path(pdf) for pdf in pdf_files]
    missing = [str(pdf) for pdf in pdf_paths if not pdf.exists()]
    if missing:
        raise FileNotFoundError(f"PDF files not found: {missing}")

    persist_dir = Path(persist_directory or CHROMA_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = Path(chunks_jsonl_path or CHUNKS_JSONL_PATH)
    parsed_path = Path(parsed_elements_path or PARSED_ELEMENTS_PATH)
    summary_output_path = Path(summary_path or INGEST_SUMMARY_PATH)

    all_parsed_elements: List[ParsedElement] = []
    all_chunks: List[Document] = []

    for pdf_path in pdf_paths:
        elements = parse_pdf_elements(pdf_path)
        all_parsed_elements.extend(elements)
        all_chunks.extend(build_chunks(elements, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    all_chunks = assign_chunk_ids(all_chunks)

    _write_parsed_elements(all_parsed_elements, parsed_path)
    _write_chunks_jsonl(all_chunks, chunks_path)

    # Generate row-level table artifact
    table_rows_records: List[dict[str, Any]] = []
    for chunk in all_chunks:
        meta = chunk.metadata or {}
        if meta.get("chunk_kind") == "table_aligned" and meta.get("table_html"):
            table_rows_records.extend(_extract_table_rows(
                html_str=meta["table_html"],
                chunk_id=meta.get("chunk_id"),
                source=meta.get("source", ""),
                page=meta.get("page"),
                section=meta.get("section"),
            ))
    table_rows_jsonl_path = persist_dir / "table_rows.jsonl"
    _write_jsonl(table_rows_jsonl_path, table_rows_records)

    compatibility_chunks_path = persist_dir / "chunks.jsonl"
    if compatibility_chunks_path.resolve() != chunks_path.resolve():
        _write_chunks_jsonl(all_chunks, compatibility_chunks_path)

    from langchain_community.vectorstores import Chroma
    from langchain_google_vertexai import VertexAIEmbeddings

    embeddings = VertexAIEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    for batch in _batched_documents(all_chunks):
        vectorstore.add_documents(
            documents=list(batch),
            ids=[str(document.metadata["chunk_id"]) for document in batch],
        )

    vectorstore.persist()

    _write_summary(
        summary_path=summary_output_path,
        pdf_files=pdf_paths,
        parsed_elements=all_parsed_elements,
        chunks=all_chunks,
        persist_directory=persist_dir,
        chunks_jsonl_path=chunks_path,
        parsed_elements_path=parsed_path,
        table_rows_path=table_rows_jsonl_path,
    )

    if UPLOAD_VECTOR_DB_TO_GCS and persist_dir.exists():
        from src.storage.gcs import upload_dir

        upload_dir(str(persist_dir), GCS_BUCKET, VECTOR_DB_PREFIX)
        print(f"Uploaded vector DB to gs://{GCS_BUCKET}/{VECTOR_DB_PREFIX}/")

    return len(all_chunks)


def ingest_data_folder(
    data_dir: str | Path | None = None,
    persist_directory: str | Path | None = None,
    collection_name: str = "10k_filings",
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    chunks_jsonl_path: str | Path | None = None,
    parsed_elements_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> int:
    load_dotenv()

    data_path = Path(data_dir or DATA_DIR)
    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path.resolve()}")

    pdf_files = sorted(data_path.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found under: {data_path.resolve()}")

    return ingest_pdf_files(
        pdf_files=pdf_files,
        persist_directory=persist_directory,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunks_jsonl_path=chunks_jsonl_path,
        parsed_elements_path=parsed_elements_path,
        summary_path=summary_path,
    )


def ingest_pdf(
    pdf_path: str | Path,
    persist_directory: str | Path | None = None,
    collection_name: str = "10k_filings",
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    chunks_jsonl_path: str | Path | None = None,
    parsed_elements_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> int:
    return ingest_pdf_files(
        pdf_files=[pdf_path],
        persist_directory=persist_directory,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunks_jsonl_path=chunks_jsonl_path,
        parsed_elements_path=parsed_elements_path,
        summary_path=summary_path,
    )


if __name__ == "__main__":
    added = ingest_data_folder()
    print(f"Ingestion complete: {added} chunks written to ChromaDB and JSONL artifacts.")
