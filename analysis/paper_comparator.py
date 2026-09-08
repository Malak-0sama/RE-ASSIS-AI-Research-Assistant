"""
Paper comparison module.

Compares two research papers using the existing ProviderRouter.
This module DOES NOT call any provider directly.
It only builds prompts and delegates generation to ProviderRouter.
"""

from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel

from llm.provider_router import ProviderRouter
from rag.context_builder import ContextBuilder
from vectordb.chroma_service import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)


class PaperComparisonResult(BaseModel):
    comparison: str
    provider_used: Optional[str] = None
    latency_seconds: float
    from_cache: bool = False
    paper_a_chunks: int
    paper_b_chunks: int


class PaperComparator:

    def __init__(
        self,
        context_builder: ContextBuilder,
        provider_router: ProviderRouter,
    ):
        self._context_builder = context_builder
        self._provider_router = provider_router

    def compare(
        self,
        paper_a_chunks: list[RetrievedChunk],
        paper_b_chunks: list[RetrievedChunk],
        temperature: float,
        max_tokens: int,
    ) -> PaperComparisonResult:

        start = time.perf_counter()

        context_a = self._context_builder.build(paper_a_chunks)
        context_b = self._context_builder.build(paper_b_chunks)

        prompt = self._build_prompt(context_a, context_b)

        generation = self._provider_router.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        elapsed = time.perf_counter() - start

        logger.info(
            f"Paper comparison completed in {elapsed:.2f}s using '{generation.provider_name}'"
        )

        return PaperComparisonResult(
            comparison=generation.text,
            provider_used=generation.provider_name,
            latency_seconds=elapsed,
            from_cache=generation.from_cache,
            paper_a_chunks=len(paper_a_chunks),
            paper_b_chunks=len(paper_b_chunks),
        )

    def _build_prompt(self, paper_a: str, paper_b: str) -> str:

        return f"""
You are an expert AI Research Scientist.

Compare the following two research papers.

Paper A:
------------------------
{paper_a}

Paper B:
------------------------
{paper_b}

Produce the answer in EXACTLY the following sections.

# 1. Research Problem
You are an expert AI Research Scientist.

The following two documents are expected to be research papers.

First determine whether BOTH uploaded documents are actually research papers.

If one or both documents are NOT research papers:

- Do NOT fabricate information.
- Clearly explain that the uploaded files are not research papers.
- Briefly describe what each document appears to be.
- Stop.

Only if BOTH documents are research papers, continue with the comparison below.

Paper A:
...
Paper B:
...

# 2. Objectives
Compare the objectives.

# 3. Methodology
Compare the proposed methodology.

# 4. Dataset
Compare datasets.

# 5. Models
Compare architectures/models.

# 6. Results
Compare reported results.

# 7. Advantages
List strengths of each paper.

# 8. Limitations
List weaknesses of each paper.

# 9. Main Contributions
Summarize the contribution of each paper.

# 10. Research Gap
Identify missing research opportunities that neither paper fully solves.

# 11. Future Work
Suggest future work combining strengths of both papers.

Return the answer in clean Markdown.
"""