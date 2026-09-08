"""
Recursive character-based text chunking.

A hand-rolled equivalent of a recursive character text splitter: splits on
an ordered list of separators, progressively falling back to finer-grained
separators for any piece still larger than `chunk_size`, then stitches
pieces back together into overlapping windows. Implemented manually (no
LangChain dependency) per project requirements.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from config.settings import ChunkingConfig
from ingestion.metadata import PaperMetadata
from utils.helpers import stable_hash
from utils.logger import get_logger

logger = get_logger(__name__)


class Chunk(BaseModel):
    """A single, embeddable slice of a document."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    metadata: dict = Field(default_factory=dict)


class RecursiveCharacterTextSplitter:
    """Splits text recursively on a priority list of separators.

    The algorithm tries the first separator; any resulting segment still
    longer than `chunk_size` is recursively split using the next separator
    in the list, until segments fit (or separators are exhausted, in which
    case a hard character split is used). Adjacent segments are then merged
    up to `chunk_size` and emitted with `chunk_overlap` characters of
    trailing context repeated into the next chunk.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int, separators: List[str]) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [""]

    def split_text(self, text: str) -> List[str]:
        raw_segments = self._recursive_split(text, self.separators)
        return self._merge_with_overlap(raw_segments)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            return self._hard_split(text)

        separator, *remaining_separators = separators
        parts = text.split(separator) if separator else list(text)

        segments: List[str] = []
        for part in parts:
            piece = part if not separator else part + separator
            if len(piece) <= self.chunk_size:
                if piece.strip():
                    segments.append(piece)
            else:
                segments.extend(self._recursive_split(piece, remaining_separators))
        return segments

    def _hard_split(self, text: str) -> List[str]:
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size)
            if text[i : i + self.chunk_size].strip()
        ]

    def _merge_with_overlap(self, segments: List[str]) -> List[str]:
        if not segments:
            return []

        merged: List[str] = []
        current = ""

        for segment in segments:
            candidate = current + segment
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    merged.append(current.strip())
                overlap_tail = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                current = overlap_tail + segment

        if current.strip():
            merged.append(current.strip())

        return merged


class Chunker:
    """High-level chunking service used by the ingestion pipeline."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
        )

    def chunk_document(self, text: str, document_id: str, metadata: PaperMetadata) -> List[Chunk]:
        raw_chunks = self._splitter.split_text(text)
        logger.info(f"Split document '{document_id}' into {len(raw_chunks)} chunks")

        chunks: List[Chunk] = []
        for index, chunk_text in enumerate(raw_chunks):
            chunk_id = stable_hash(document_id, str(index), chunk_text[:100])
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    chunk_index=index,
                    text=chunk_text,
                    metadata={
                        "title": metadata.title,
                        "source_filename": metadata.source_filename,
                        "chunk_index": index,
                    },
                )
            )
        return chunks
