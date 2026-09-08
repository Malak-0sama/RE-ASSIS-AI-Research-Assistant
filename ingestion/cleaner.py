"""
Text cleaning for extracted PDF content.

Removes common PDF extraction artifacts (running headers/footers, page
numbers, excessive whitespace) while preserving paragraph structure, since
downstream chunking relies on meaningful paragraph boundaries.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)

_PAGE_NUMBER_PATTERN = re.compile(r"^\s*(page\s+)?\d{1,4}\s*(of\s+\d{1,4})?\s*$", re.IGNORECASE)
_MULTI_BLANK_LINES = re.compile(r"\n\s*\n\s*\n+")
_MULTI_SPACES = re.compile(r"[ \t]{2,}")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")


class TextCleaner:
    """Cleans raw extracted PDF text into readable, paragraph-preserving text."""

    def __init__(self, header_footer_min_repeats: int = 3) -> None:
        # A line repeated across many pages is almost certainly a running
        # header/footer rather than article content.
        self._header_footer_min_repeats = header_footer_min_repeats

    def clean(self, text: str) -> str:
        logger.debug("Cleaning extracted text")

        # Rejoin words that PDF extraction split across a line-wrap hyphen.
        text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)

        lines = text.split("\n")
        lines = self._strip_repeated_headers_footers(lines)
        lines = [line for line in lines if not _PAGE_NUMBER_PATTERN.match(line)]

        cleaned = "\n".join(lines)
        cleaned = _MULTI_SPACES.sub(" ", cleaned)
        cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
        cleaned = _MULTI_BLANK_LINES.sub("\n\n", cleaned)

        cleaned = cleaned.strip()
        logger.debug(f"Cleaned text length: {len(cleaned)} characters")
        return cleaned

    def _strip_repeated_headers_footers(self, lines: List[str]) -> List[str]:
        normalized = [line.strip() for line in lines]
        counts = Counter(line for line in normalized if line and len(line) < 120)
        repeated = {
            line
            for line, count in counts.items()
            if count >= self._header_footer_min_repeats
        }
        if repeated:
            logger.debug(f"Removing {len(repeated)} repeated header/footer line(s)")
        return [line for line, norm in zip(lines, normalized) if norm not in repeated]
