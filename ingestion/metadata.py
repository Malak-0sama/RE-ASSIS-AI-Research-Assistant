"""
Structured metadata extraction (title, authors, page count) for a loaded PDF.
"""
from __future__ import annotations

import re
from typing import List

from pydantic import BaseModel, Field

from ingestion.pdf_loader import RawDocument
from utils.logger import get_logger

logger = get_logger(__name__)

_AUTHOR_SPLIT_PATTERN = re.compile(r"\s*(?:,|;|\band\b)\s*", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\S+@\S+")


class PaperMetadata(BaseModel):
    """Structured, human-readable metadata for a research paper."""

    title: str
    authors: List[str] = Field(default_factory=list)
    page_count: int
    source_filename: str
    document_id: str


class MetadataExtractor:
    """Derives `PaperMetadata` from a `RawDocument`.

    Strategy: prefer PDF-embedded metadata (from the document's info
    dictionary), and fall back to heuristics over the first page of text
    when embedded metadata is missing or unreliable (common for papers
    exported from LaTeX, where title/author fields are often blank).
    """

    def extract(self, raw_document: RawDocument) -> PaperMetadata:
        title = self._extract_title(raw_document)
        authors = self._extract_authors(raw_document)

        metadata = PaperMetadata(
            title=title,
            authors=authors,
            page_count=raw_document.page_count,
            source_filename=raw_document.source_filename,
            document_id=raw_document.document_id,
        )
        logger.info(f"Extracted metadata: title='{metadata.title}', authors={metadata.authors}")
        return metadata

    def _extract_title(self, raw_document: RawDocument) -> str:
        embedded_title = (raw_document.raw_metadata.get("title") or "").strip()
        if embedded_title and len(embedded_title) > 3:
            return embedded_title

        for line in self._first_page_lines(raw_document.raw_text, limit=5):
            candidate = line.strip()
            if 8 <= len(candidate) <= 200 and not _EMAIL_PATTERN.search(candidate):
                return candidate

        return raw_document.source_filename

    def _extract_authors(self, raw_document: RawDocument) -> List[str]:
        embedded_author = (raw_document.raw_metadata.get("author") or "").strip()
        if embedded_author:
            parts = [p.strip() for p in _AUTHOR_SPLIT_PATTERN.split(embedded_author) if p.strip()]
            if parts:
                return parts

        first_lines = self._first_page_lines(raw_document.raw_text, limit=8)
        for line in first_lines[1:]:
            candidate = line.strip()
            if self._looks_like_author_line(candidate):
                return [p.strip() for p in _AUTHOR_SPLIT_PATTERN.split(candidate) if p.strip()]

        return []

    @staticmethod
    def _first_page_lines(text: str, limit: int) -> List[str]:
        lines = [line for line in text.split("\n") if line.strip()]
        return lines[:limit]

    @staticmethod
    def _looks_like_author_line(line: str) -> bool:
        if not line or len(line) > 200:
            return False
        if _EMAIL_PATTERN.search(line):
            return True
        # Heuristic: short line, comma-separated, mostly capitalized words.
        words = line.replace(",", " ").replace(" and ", " ").split()
        if not (2 <= len(words) <= 20):
            return False
        capitalized_ratio = sum(1 for w in words if w[:1].isupper()) / len(words)
        return capitalized_ratio > 0.6
