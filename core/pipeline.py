"""
Top-level orchestrator wiring together ingestion, chunking, embeddings,
vector storage, retrieval, and the LLM router into a single facade used
directly by the Streamlit UI.
"""
from __future__ import annotations

from typing import Optional

from analysis.paper_analyzer import PaperAnalysisResult, PaperAnalyzer
from analysis.paper_comparator import (
    PaperComparator,
    PaperComparisonResult,
)
from analysis.paper_recommender import (
    PaperRecommender,
    RecommendedPaper,
)
from chunking.chunker import Chunker
from config.settings import Settings
from embeddings.embedding_service import EmbeddingService
from ingestion.cleaner import TextCleaner
from ingestion.metadata import MetadataExtractor, PaperMetadata
from ingestion.pdf_loader import PDFLoader
from llm.cache import ResponseCache
from llm.provider_router import ProviderRouter
from llm.providers.gemini_provider import GeminiProvider
from llm.providers.groq_provider import GroqProvider
from llm.providers.huggingface_provider import HuggingFaceProvider
from rag.context_builder import ContextBuilder
from rag.qa_pipeline import QAPipeline, QAResult
from rag.retriever import Retriever
from utils.logger import get_logger
from vectordb.chroma_service import ChromaService, VectorStoreBase
from vectordb.chroma_service import RetrievedChunk

logger = get_logger(__name__)


class ResearchAssistantPipeline:
    """Facade that owns the full lifecycle: ingest a PDF, then answer questions about it."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pdf_loader = PDFLoader()
        self._cleaner = TextCleaner()
        self._metadata_extractor = MetadataExtractor()
        self._chunker = Chunker(settings.app.chunking)
        self._embedding_service = EmbeddingService(settings.app.embedding)
        self._cache = ResponseCache(settings.app.cache)

        self._current_document_id: Optional[str] = None
        self._current_metadata: Optional[PaperMetadata] = None
        self._qa_pipeline: Optional[QAPipeline] = None

        # Kept so `analyze_paper()` can reuse the exact same vector store,
        # context builder, and provider router the chat pipeline uses,
        # without duplicating retrieval or LLM-calling logic.
        self._vector_store: Optional[VectorStoreBase] = None
        self._context_builder: Optional[ContextBuilder] = None
        self._provider_router: Optional[ProviderRouter] = None
        self._paper_b_vector_store: Optional[VectorStoreBase] = None
        self._paper_b_metadata: Optional[PaperMetadata] = None
        self._paper_recommender = PaperRecommender()

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def current_metadata(self) -> Optional[PaperMetadata]:
        return self._current_metadata

    def _build_providers(self) -> dict:
        llm_cfg = self._settings.app.llm
        keys = self._settings.api_keys
        providers = {}

        if keys.groq_api_key:
            providers["groq"] = GroqProvider(
                keys.groq_api_key,
                llm_cfg.groq_model,
            )

        if keys.gemini_api_key:
            providers["gemini"] = GeminiProvider(
                keys.gemini_api_key,
                llm_cfg.gemini_model,
            )

        if keys.huggingface_api_key:
            providers["huggingface"] = HuggingFaceProvider(
                keys.huggingface_api_key,
                llm_cfg.huggingface_model,
            )



        return providers

    def ingest_pdf(self, file_bytes: bytes, filename: str) -> PaperMetadata:
        """Run the full ingestion pipeline for one uploaded PDF: load, clean,
        extract metadata, chunk, embed, and index into the vector store.
        """
        logger.info(f"Starting ingestion for '{filename}'")

        raw_document = self._pdf_loader.load_from_bytes(file_bytes, source_filename=filename)
        cleaned_text = self._cleaner.clean(raw_document.raw_text)
        metadata = self._metadata_extractor.extract(raw_document)

        chunks = self._chunker.chunk_document(cleaned_text, raw_document.document_id, metadata)
        embeddings = self._embedding_service.embed_documents([c.text for c in chunks])

        vector_store = ChromaService(self._settings.app.vectordb, document_id=raw_document.document_id)
        vector_store.clear()
        vector_store.add_chunks(chunks, embeddings)

        retriever = Retriever(self._embedding_service, vector_store)
        context_builder = ContextBuilder()
        provider_router = ProviderRouter(
            providers=self._build_providers(),
            llm_config=self._settings.app.llm,
            cache=self._cache,
        )
        self._qa_pipeline = QAPipeline(retriever, context_builder, provider_router)

        self._current_document_id = raw_document.document_id
        self._current_metadata = metadata
        self._vector_store = vector_store
        self._context_builder = context_builder
        self._provider_router = provider_router

        logger.info(f"Ingestion complete for '{filename}': {len(chunks)} chunk(s) indexed")
        return metadata

    def ingest_second_pdf(self, file_bytes: bytes, filename: str) -> PaperMetadata:
        """
        Ingest the second paper used for comparison.
        """

        logger.info(f"Starting ingestion for second paper '{filename}'")

        raw_document = self._pdf_loader.load_from_bytes(
            file_bytes,
            source_filename=filename,
        )

        cleaned_text = self._cleaner.clean(raw_document.raw_text)

        metadata = self._metadata_extractor.extract(raw_document)

        chunks = self._chunker.chunk_document(
            cleaned_text,
            raw_document.document_id,
            metadata,
        )

        embeddings = self._embedding_service.embed_documents(
            [c.text for c in chunks]
        )

        vector_store = ChromaService(
            self._settings.app.vectordb,
            document_id=raw_document.document_id,
        )

        vector_store.clear()

        vector_store.add_chunks(
            chunks,
            embeddings,
        )
        if self._paper_b_vector_store is not None:
            self._paper_b_vector_store.clear()

        self._paper_b_vector_store = vector_store
        self._paper_b_metadata = metadata

        logger.info(
            f"Second paper indexed successfully ({len(chunks)} chunks)"
        )

        return metadata

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> QAResult:
        if self._qa_pipeline is None:
            raise RuntimeError("No document has been ingested yet. Upload a PDF first.")

        cfg = self._settings.app
        return self._qa_pipeline.answer(
            question=question,
            top_k=top_k if top_k is not None else cfg.retrieval.top_k,
            temperature=temperature if temperature is not None else cfg.llm.temperature,
            max_tokens=max_tokens if max_tokens is not None else cfg.llm.max_tokens,
            score_threshold=cfg.retrieval.score_threshold,
        )

    def analyze_paper(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> PaperAnalysisResult:
        """Run a full, automatic analysis of the currently ingested paper.

        Reuses the same vector store, ContextBuilder, and ProviderRouter as
        `ask()` — it does not create its own retrieval or LLM-calling logic.
        """
        if self._vector_store is None or self._context_builder is None or self._provider_router is None:
            raise RuntimeError("No document has been ingested yet. Upload a PDF first.")

        cfg = self._settings.app
        analyzer = PaperAnalyzer(self._vector_store, self._context_builder, self._provider_router)
        return analyzer.analyze(
            temperature=temperature if temperature is not None else cfg.llm.temperature,
            # Full-paper analysis produces much longer output than a single
            # QA answer, so it needs more headroom than the chat default.
            max_tokens=max_tokens if max_tokens is not None else max(cfg.llm.max_tokens, 3000),
        )

    def compare_papers(
            self,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
    ) -> PaperComparisonResult:

        if (
                self._vector_store is None
                or self._paper_b_vector_store is None
                or self._provider_router is None
                or self._context_builder is None
        ):
            raise RuntimeError(
                "Please upload two papers first."
            )

        cfg = self._settings.app

        comparator = PaperComparator(
            self._context_builder,
            self._provider_router,
        )

        paper_a = sorted(
            self._vector_store.get_document_chunks(),
            key=lambda x: x["metadata"]["chunk_index"],
        )

        paper_b = sorted(
            self._paper_b_vector_store.get_document_chunks(),
            key=lambda x: x["metadata"]["chunk_index"],
        )



        chunks_a = [
            RetrievedChunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                metadata=c["metadata"],
                similarity_score=1.0,
            )
            for c in paper_a
        ]

        chunks_b = [
            RetrievedChunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                metadata=c["metadata"],
                similarity_score=1.0,
            )
            for c in paper_b
        ]

        return comparator.compare(
            paper_a_chunks=chunks_a,
            paper_b_chunks=chunks_b,
            temperature=temperature
            if temperature is not None
            else cfg.llm.temperature,
            max_tokens=max_tokens
            if max_tokens is not None
            else max(cfg.llm.max_tokens, 4000),
        )

    def recommend_papers(
            self,
            query: str,
            limit: int = 5,
    ) -> list[RecommendedPaper]:

        return self._paper_recommender.recommend(
            query=query,
            limit=limit,
        )
