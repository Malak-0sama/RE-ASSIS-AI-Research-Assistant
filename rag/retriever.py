"""
Semantic retriever: question -> query embedding -> top-K similar chunks.
"""
from __future__ import annotations

from typing import List, Optional

from embeddings.embedding_service import EmbeddingService
from utils.logger import get_logger
from vectordb.chroma_service import RetrievedChunk, VectorStoreBase

logger = get_logger(__name__)


class Retriever:
    """Retrieves the most relevant chunks for a natural-language question."""

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStoreBase) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def retrieve(
        self, question: str, top_k: int, score_threshold: Optional[float] = None
    ) -> List[RetrievedChunk]:
        logger.info(f"Retrieving top-{top_k} chunks for question: '{question[:80]}'")
        query_embedding = self._embedding_service.embed_query(question)
        results = self._vector_store.query(query_embedding, top_k=top_k)

        if score_threshold is not None:
            before = len(results)
            results = [r for r in results if r.similarity_score >= score_threshold]
            logger.debug(f"Score threshold {score_threshold} filtered {before - len(results)} chunk(s)")

        return results
