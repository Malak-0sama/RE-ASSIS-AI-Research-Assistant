"""
PDF ingestion using PyMuPDF (fitz).

Responsible ONLY for turning a PDF file into a `RawDocument`: full text,
page count, and the raw fitz-extracted metadata dictionary. Cleaning and
structured metadata extraction are deliberately handled by other modules
(`ingestion.cleaner`, `ingestion.metadata`) to respect single responsibility.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import fitz  # PyMuPDF
from pydantic import BaseModel, Field

from utils.helpers import stable_hash
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF cannot be loaded or parsed."""


class EmptyPDFError(PDFLoadError):
    """Raised when a PDF contains no extractable text."""


class RawDocument(BaseModel):
    """Result of loading a PDF: unprocessed text plus low-level metadata."""

    document_id: str
    source_filename: str
    raw_text: str
    page_count: int
    raw_metadata: dict = Field(default_factory=dict)


class PDFLoader:
    """Loads a PDF file from disk or bytes and extracts its raw text."""

    def load(self, file_path: Union[str, Path]) -> RawDocument:
        """Load a PDF from a filesystem path."""
        path = Path(file_path)
        if not path.exists():
            raise PDFLoadError(f"PDF file not found: {path}")
        return self._load_from_bytes(path.read_bytes(), source_filename=path.name)

    def load_from_bytes(self, data: bytes, source_filename: str) -> RawDocument:
        """Load a PDF from raw bytes (e.g. a Streamlit `UploadedFile`)."""
        return self._load_from_bytes(data, source_filename=source_filename)

    def _load_from_bytes(self, data: bytes, source_filename: str) -> RawDocument:
        if not data:
            raise EmptyPDFError(f"'{source_filename}' is empty (0 bytes).")

        logger.info(f"Loading PDF '{source_filename}' ({len(data)} bytes)")

        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:  # PyMuPDF raises generic exceptions on bad files
            raise PDFLoadError(f"Failed to open '{source_filename}' as a PDF: {exc}") from exc

        try:
            if doc.page_count == 0:
                raise EmptyPDFError(f"'{source_filename}' contains zero pages.")

            page_texts = []
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                page_texts.append(page.get_text("text"))

            full_text = "\n".join(page_texts).strip()

            if not full_text:
                raise EmptyPDFError(
                    f"'{source_filename}' contains no extractable text "
                    "(it may be a scanned/image-only PDF)."
                )

            raw_metadata = dict(doc.metadata or {})
            document_id = stable_hash(source_filename, str(len(data)), full_text[:500])

            logger.info(
                f"Loaded '{source_filename}': {doc.page_count} pages, "
                f"{len(full_text)} characters"
            )

            return RawDocument(
                document_id=document_id,
                source_filename=source_filename,
                raw_text=full_text,
                page_count=doc.page_count,
                raw_metadata=raw_metadata,
            )
        finally:
            doc.close()
