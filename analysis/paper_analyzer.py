"""
Full-paper analysis pipeline: retrieve every chunk for the ingested paper ->
build one large context -> single structured analysis prompt -> LLM.

Deliberately mirrors `rag/qa_pipeline.py`'s shape but does NOT touch it or
duplicate its retrieval logic — it reuses the same `VectorStoreBase`,
`ContextBuilder`, and `ProviderRouter` instances the chat feature already
uses, just fetching all chunks instead of a similarity-ranked subset.
"""
from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel

from llm.provider_router import ProviderRouter
from prompts.paper_analysis_prompt import build_paper_analysis_prompt
from rag.context_builder import ContextBuilder
from utils.logger import get_logger
from vectordb.chroma_service import RetrievedChunk, VectorStoreBase

logger = get_logger(__name__)


class PaperAnalysisResult(BaseModel):
    """Everything the UI needs to render one full-paper analysis."""

    analysis: str
    provider_used: Optional[str] = None
    latency_seconds: float
    from_cache: bool = False
    chunk_count: int


class PaperAnalyzer:
    """Produces a full structured analysis of the currently ingested paper."""

    def __init__(
        self,
        vector_store: VectorStoreBase,
        context_builder: ContextBuilder,
        provider_router: ProviderRouter,
    ) -> None:
        self._vector_store = vector_store
        self._context_builder = context_builder
        self._provider_router = provider_router

    def analyze(self, temperature: float, max_tokens: int) -> PaperAnalysisResult:
        start = time.perf_counter()

        raw_chunks = self._vector_store.get_document_chunks()
        logger.info(f"Analyzing paper using {len(raw_chunks)} chunk(s)")

        # Reuse ContextBuilder as-is: it just needs a RetrievedChunk-shaped
        # list, so every chunk is wrapped with a nominal max similarity
        # score (there is no query here, so relevance ranking doesn't apply).
        chunks = [
            RetrievedChunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                metadata=c["metadata"],
                similarity_score=1.0,
            )
            for c in raw_chunks
        ]
        context = self._context_builder.build(chunks)

        if not context:
            logger.warning("No chunks available for analysis.")
            return PaperAnalysisResult(
                analysis="No content is available for this paper yet. Please upload a PDF first.",
                provider_used=None,
                latency_seconds=time.perf_counter() - start,
                chunk_count=0,
            )

        prompt = build_paper_analysis_prompt(context=context)

        generation = self._provider_router.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        elapsed = time.perf_counter() - start
        logger.info(f"Paper analysis completed in {elapsed:.2f}s via provider='{generation.provider_name}'")

        return PaperAnalysisResult(
            analysis=generation.text,
            provider_used=generation.provider_name,
            latency_seconds=elapsed,
            from_cache=generation.from_cache,
            chunk_count=len(chunks),
        )
