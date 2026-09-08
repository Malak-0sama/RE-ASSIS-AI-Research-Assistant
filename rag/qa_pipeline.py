"""
End-to-end RAG question-answering pipeline:
question -> retriever -> context builder -> prompt -> LLM -> answer.
"""
from __future__ import annotations

import time
from typing import List, Optional

from pydantic import BaseModel

from llm.provider_router import ProviderRouter
from prompts.qa_prompt import FALLBACK_ANSWER, build_qa_prompt
from rag.context_builder import ContextBuilder
from rag.retriever import Retriever
from utils.logger import get_logger
from vectordb.chroma_service import RetrievedChunk

logger = get_logger(__name__)


class QAResult(BaseModel):
    """Everything the UI needs to render one answered question."""

    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    provider_used: Optional[str] = None
    latency_seconds: float
    from_cache: bool = False


class QAPipeline:
    """Orchestrates the full RAG flow for a single question."""

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        provider_router: ProviderRouter,
    ) -> None:
        self._retriever = retriever
        self._context_builder = context_builder
        self._provider_router = provider_router

    def answer(
        self,
        question: str,
        top_k: int,
        temperature: float,
        max_tokens: int,
        score_threshold: Optional[float] = None,
    ) -> QAResult:
        start = time.perf_counter()

        logger.info(f"Building prompt for question: '{question[:80]}'")
        chunks = self._retriever.retrieve(question, top_k=top_k, score_threshold=score_threshold)
        context = self._context_builder.build(chunks)

        if not context:
            logger.warning("No relevant chunks retrieved; skipping LLM call.")
            return QAResult(
                question=question,
                answer=FALLBACK_ANSWER,
                retrieved_chunks=[],
                provider_used=None,
                latency_seconds=time.perf_counter() - start,
            )

        prompt = build_qa_prompt(context=context, question=question)

        generation = self._provider_router.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        elapsed = time.perf_counter() - start
        logger.info(f"Answered question in {elapsed:.2f}s via provider='{generation.provider_name}'")

        return QAResult(
            question=question,
            answer=generation.text,
            retrieved_chunks=chunks,
            provider_used=generation.provider_name,
            latency_seconds=elapsed,
            from_cache=generation.from_cache,
        )
