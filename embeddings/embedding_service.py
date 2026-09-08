"""
Embedding generation via Sentence Transformers.

Wraps a `SentenceTransformer` model behind a small, swappable service so
the rest of the codebase never depends on the embedding library directly.
"""
from __future__ import annotations

from typing import List

from config.settings import EmbeddingConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerationError(Exception):
    """Raised when embedding generation fails."""


class EmbeddingService:
    """Generates dense vector embeddings for documents and queries."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model = None  # Lazy-loaded: the model is large and slow to init.

    def _get_model(self):
        if self._model is None:
            logger.info(f"Loading embedding model '{self._config.model_name}' on {self._config.device}")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingGenerationError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
            try:
                self._model = SentenceTransformer(self._config.model_name, device=self._config.device)
            except Exception as exc:
                raise EmbeddingGenerationError(
                    f"Failed to load embedding model '{self._config.model_name}': {exc}"
                ) from exc
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        try:
            vectors = model.encode(
                texts,
                batch_size=self._config.batch_size,
                normalize_embeddings=self._config.normalize_embeddings,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingGenerationError(f"Failed to embed {len(texts)} document(s): {exc}") from exc
        logger.debug(f"Generated {len(texts)} document embedding(s)")
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise EmbeddingGenerationError("Cannot embed an empty query.")
        model = self._get_model()
        try:
            vector = model.encode(
                text,
                normalize_embeddings=self._config.normalize_embeddings,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingGenerationError(f"Failed to embed query: {exc}") from exc
        return vector.tolist()
