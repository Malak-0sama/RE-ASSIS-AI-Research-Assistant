from chunking.chunker import Chunker, RecursiveCharacterTextSplitter
from config.settings import ChunkingConfig
from ingestion.metadata import PaperMetadata


def test_splitter_respects_chunk_size():
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10, separators=["\n\n", " ", ""])
    text = "word " * 100
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    # Small tolerance: overlap re-prepends up to `chunk_overlap` characters.
    assert all(len(c) <= 60 for c in chunks)


def test_splitter_rejects_invalid_overlap():
    try:
        RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=10, separators=[""])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_chunker_produces_chunk_objects():
    config = ChunkingConfig(chunk_size=100, chunk_overlap=20, separators=["\n\n", " ", ""])
    chunker = Chunker(config)
    metadata = PaperMetadata(
        title="Test Paper",
        authors=["A. Author"],
        page_count=1,
        source_filename="test.pdf",
        document_id="doc123",
    )
    text = "Sentence one. " * 50
    chunks = chunker.chunk_document(text, document_id="doc123", metadata=metadata)

    assert len(chunks) > 1
    assert all(c.document_id == "doc123" for c in chunks)
    assert all(c.metadata["title"] == "Test Paper" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
