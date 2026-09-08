from ingestion.metadata import MetadataExtractor
from ingestion.pdf_loader import RawDocument


def test_prefers_embedded_title_and_author():
    raw = RawDocument(
        document_id="d1",
        source_filename="paper.pdf",
        raw_text="Ignored First Line\nJohn Smith\nAbstract...",
        page_count=3,
        raw_metadata={"title": "Attention Is All You Need", "author": "John Smith, Jane Doe"},
    )
    metadata = MetadataExtractor().extract(raw)
    assert metadata.title == "Attention Is All You Need"
    assert metadata.authors == ["John Smith", "Jane Doe"]
    assert metadata.page_count == 3


def test_falls_back_to_first_line_when_no_embedded_title():
    raw = RawDocument(
        document_id="d2",
        source_filename="paper.pdf",
        raw_text="A Novel Approach To Something Interesting\nSome Author Name\nAbstract text follows.",
        page_count=1,
        raw_metadata={},
    )
    metadata = MetadataExtractor().extract(raw)
    assert metadata.title == "A Novel Approach To Something Interesting"
