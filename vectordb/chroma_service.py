"""
Vector storage backed by ChromaDB.

`VectorStoreBase` is an abstract interface so an alternative backend (e.g.
FAISS) can be added later without touching any calling code — only a new
implementation of this interface plus a factory change would be required.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from pydantic import BaseModel

from chunking.chunker import Chunk
from config.settings import VectorDBConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when a vector database operation fails."""


class RetrievedChunk(BaseModel):
    """A chunk returned from similarity search, enriched with its score."""

    chunk_id: str
    text: str
    metadata: dict
    similarity_score: float


class VectorStoreBase(ABC):
    """Abstract vector store interface. Implement this to add new backends."""

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        ...

    @abstractmethod
    def query(self, embedding: List[float], top_k: int) -> List[RetrievedChunk]:
        ...

    @abstractmethod
    def collection_exists(self) -> bool:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


class ChromaService(VectorStoreBase):
    """ChromaDB-backed implementation of `VectorStoreBase`.

    Each research paper gets its own collection (named from its document
    id) so switching papers never mixes embeddings from different sources.
    """

    def __init__(self, config: VectorDBConfig, document_id: str) -> None:
        self._config = config
        self._document_id = document_id
        self._collection_name = f"{config.collection_prefix}_{document_id}"[:63]
        self._client = self._build_client()
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _build_client(self):
        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreError("chromadb is not installed. Run: pip install chromadb") from exc

        persist_dir = Path(self._config.persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)
        try:
            return chromadb.PersistentClient(path=str(persist_dir))
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize ChromaDB at '{persist_dir}': {exc}") from exc

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunk/embedding count mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
        if not chunks:
            return
        try:
            self._collection.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=embeddings,
                documents=[c.text for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to store {len(chunks)} chunk(s) in ChromaDB: {exc}") from exc
        logger.info(f"Stored {len(chunks)} chunk(s) in collection '{self._collection_name}'")

    def query(self, embedding: List[float], top_k: int) -> List[RetrievedChunk]:
        try:
            result = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB query failed: {exc}") from exc

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        retrieved: List[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            # Chroma's cosine space returns a distance; similarity = 1 - distance.
            similarity = 1.0 - distance
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata or {},
                    similarity_score=round(similarity, 4),
                )
            )
        logger.debug(f"Retrieved {len(retrieved)} chunk(s) for query")
        return retrieved

    def get_document_chunks(self) -> List[dict]:
        """Return every chunk stored for the current paper, ordered by chunk index.

        Unlike `query`, this does no similarity search — it's used by
        paper-level operations (e.g. full-paper analysis) that need the
        entire document rather than a similarity-ranked subset.
        """
        try:
            result = self._collection.get()
        except Exception as exc:
            raise VectorStoreError(f"Failed to fetch document chunks from ChromaDB: {exc}") from exc

        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        chunks = [
            {"chunk_id": chunk_id, "text": text, "metadata": metadata or {}}
            for chunk_id, text, metadata in zip(ids, documents, metadatas)
        ]
        chunks.sort(key=lambda c: c["metadata"].get("chunk_index", 0))
        logger.debug(f"Fetched {len(chunks)} chunk(s) for full-document analysis")
        return chunks

    def collection_exists(self) -> bool:
        return self._collection.count() > 0

    def clear(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to clear collection '{self._collection_name}': {exc}") from exc
