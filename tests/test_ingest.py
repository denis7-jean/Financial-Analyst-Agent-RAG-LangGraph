from dataclasses import dataclass, field

from src.ingestion.ingest import build_chunks, normalize_parsed_elements


TABLE_HTML = """
<table>
  <thead>
    <tr><th>Year</th><th>Net sales</th><th>Gross margin</th></tr>
  </thead>
  <tbody>
    <tr><td>2024</td><td>391,035</td><td>180,683</td></tr>
    <tr><td>2023</td><td>383,285</td><td>169,148</td></tr>
  </tbody>
</table>
""".strip()


@dataclass
class FakeElement:
    category: str
    text: str = ""
    metadata: dict = field(default_factory=dict)


def test_table_chunk_is_preserved_as_html_without_truncation() -> None:
    raw_elements = [
        FakeElement(category="Title", text="Net Sales"),
        FakeElement(
            category="NarrativeText",
            text="Apple reported higher services revenue in fiscal 2024.",
            metadata={"page_number": 32},
        ),
        FakeElement(
            category="Table",
            text="Condensed financial table",
            metadata={"page_number": 33, "text_as_html": TABLE_HTML},
        ),
    ]

    parsed_elements = normalize_parsed_elements(raw_elements, source_path="Apple_2024_10k.pdf")
    chunks = build_chunks(parsed_elements, chunk_size=300, chunk_overlap=0)

    table_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_kind") == "table_aligned"]
    assert len(table_chunks) == 1

    table_chunk = table_chunks[0]
    assert table_chunk.page_content.startswith("[TABLE ALIGNMENT]")
    assert "Net Sales" in table_chunk.page_content
    assert "391035" in table_chunk.page_content
    assert table_chunk.metadata["content_format"] == "html"
    assert table_chunk.metadata["element_type"] == "Table"
    assert table_chunk.metadata["section"] == "Net Sales"


def test_chunk_metadata_is_assigned_with_sequential_chunk_ids() -> None:
    raw_elements = [
        FakeElement(category="Title", text="Risk Factors", metadata={"page_number": 12}),
        FakeElement(
            category="NarrativeText",
            text="Supply chain constraints and foreign exchange can affect margins.",
            metadata={"page_number": 12},
        ),
        FakeElement(
            category="Table",
            text="Risk scoring table",
            metadata={"page_number": 13, "text_as_html": TABLE_HTML},
        ),
        FakeElement(
            category="NarrativeText",
            text="Management monitors geographic concentration and vendor risk.",
            metadata={"page_number": 14},
        ),
    ]

    parsed_elements = normalize_parsed_elements(raw_elements, source_path="Apple_2024_10k.pdf")
    chunks = build_chunks(parsed_elements, chunk_size=500, chunk_overlap=0)

    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(range(len(chunks)))

    first_text_chunk = next(chunk for chunk in chunks if chunk.metadata["chunk_kind"] == "text")
    assert first_text_chunk.metadata["source"] == "Apple_2024_10k.pdf"
    assert first_text_chunk.metadata["doc_source"] == "Apple_2024_10k.pdf"
    assert first_text_chunk.metadata["section"] == "Risk Factors"
    assert first_text_chunk.metadata["page_start"] == 12
    assert first_text_chunk.metadata["page_end"] == 12

    table_chunk = next(chunk for chunk in chunks if chunk.metadata["chunk_kind"] == "table_aligned")
    assert table_chunk.metadata["page"] == 13
    assert table_chunk.metadata["chunk_id"] == 1
