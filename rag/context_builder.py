"""
Builds a single, LLM-ready context string from retrieved chunks.
"""
from __future__ import annotations

from typing import List

from vectordb.chroma_service import RetrievedChunk


class ContextBuilder:
    """Formats retrieved chunks into a numbered, source-attributed context block."""

    def build(self, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return ""

        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            segment_hint = chunk.metadata.get("chunk_index")
            source_tag = f"[Source {i}" + (f", segment {segment_hint}" if segment_hint is not None else "") + "]"
            blocks.append(f"{source_tag}\n{chunk.text.strip()}")

        return "\n\n".join(blocks)
