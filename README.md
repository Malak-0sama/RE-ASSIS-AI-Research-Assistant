# AI Research Scientist — Milestone 1: Chat with Research Paper

A production-grade Retrieval-Augmented Generation (RAG) application that lets
you upload a research paper (PDF) and ask questions about it in a chat
interface. Answers are grounded strictly in the paper's content, with
multi-provider LLM fallback, response caching, and full observability.

This is **Milestone 1** of the AI Research Scientist project. Summarization,
paper comparison, related-paper discovery, knowledge graphs, research-gap
analysis, and future-work generation are intentionally out of scope and will
arrive in later milestones.

## Features

- Upload a single PDF research paper
- Text extraction (PyMuPDF) with header/footer/page-number cleaning
- Title & author metadata extraction
- Configurable recursive-character chunking
- Embeddings via Sentence Transformers (`BAAI/bge-m3` by default)
- Vector storage & semantic retrieval via ChromaDB
- Context-grounded RAG question answering
- Five-provider LLM router (Groq → OpenAI → Gemini → HuggingFace → Ollama)
  with automatic retry + fallback
- TTL response cache
- Structured logging (Loguru)
- Fully config-driven (`.env` + `config/config.yaml`) — nothing hardcoded
- Clean, professional Streamlit UI with a debug panel

## Architecture

The codebase follows Clean Architecture / SOLID principles: each layer has a
single responsibility and depends only on the layers beneath it.

```
PDF Upload
    │
    ▼
ingestion/  →  pdf_loader → cleaner → metadata
    │
    ▼
chunking/   →  chunker (RecursiveCharacterTextSplitter)
    │
    ▼
embeddings/ →  embedding_service (Sentence Transformers)
    │
    ▼
vectordb/   →  chroma_service (ChromaDB)
    │
    ▼
rag/        →  retriever → context_builder → qa_pipeline
    │
    ▼
llm/        →  provider_router → [groq | openai | gemini | huggingface | ollama]
    │
    ▼
core/pipeline.py  (facade orchestrating everything above)
    │
    ▼
app.py  (Streamlit UI)
```

## Project Structure

```
research_ai/
├── app.py                      # Streamlit entry point
├── requirements.txt
├── README.md
├── .env.example
│
├── config/
│   ├── settings.py             # Pydantic settings (env + YAML merge)
│   └── config.yaml             # All non-secret configuration
│
├── core/
│   └── pipeline.py             # Top-level orchestrator facade
│
├── ingestion/
│   ├── pdf_loader.py           # PyMuPDF text extraction
│   ├── cleaner.py              # Header/footer/whitespace cleaning
│   └── metadata.py             # Title/author extraction
│
├── chunking/
│   └── chunker.py              # RecursiveCharacterTextSplitter
│
├── embeddings/
│   └── embedding_service.py    # Sentence Transformers wrapper
│
├── vectordb/
│   └── chroma_service.py       # ChromaDB store (FAISS-ready interface)
│
├── rag/
│   ├── retriever.py            # Question → top-K chunks
│   ├── context_builder.py      # Chunks → formatted context
│   └── qa_pipeline.py          # Full RAG orchestration
│
├── llm/
│   ├── base_provider.py        # Abstract provider interface
│   ├── provider_router.py      # Priority routing + retry + fallback
│   ├── cache.py                # TTL response cache
│   └── providers/
│       ├── groq_provider.py
│       ├── openai_provider.py
│       ├── gemini_provider.py
│       ├── huggingface_provider.py
│       └── ollama_provider.py
│
├── prompts/
│   └── qa_prompt.py            # Centralized prompt templates
│
├── utils/
│   ├── logger.py                # Loguru setup
│   └── helpers.py                # Hashing, timing, misc helpers
│
└── tests/
    ├── test_cleaner.py
    ├── test_chunker.py
    ├── test_cache.py
    ├── test_qa_prompt.py
    └── test_metadata.py
```

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Fill in whichever provider keys you have in `.env`. You don't need all five —
the router will skip any provider whose key is missing and fall back to the
next one in `config/config.yaml` under `llm.provider_priority`. Ollama needs
no key but requires a locally running Ollama server
(`ollama serve`, with your chosen model pulled).

### 3. (Optional) Adjust configuration

All tunable values — chunk size/overlap, top-K, temperature, max tokens,
timeouts, provider priority, cache TTL, embedding model, logging — live in
`config/config.yaml`. Nothing is hardcoded elsewhere in the codebase.

### 4. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`), upload a
PDF from the sidebar, and start asking questions once indexing finishes.

## Running Tests

```bash
pytest tests/ -v
```

The included tests cover text cleaning, chunking, metadata heuristics,
prompt construction, and caching — the modules that don't require downloading
ML models or hitting live network services, so they run instantly offline.

## How a Question Gets Answered

1. Your question is embedded with the same model used for the paper's chunks.
2. ChromaDB returns the top-K most similar chunks (configurable).
3. The chunks are formatted into a numbered, source-attributed context block.
4. A prompt instructing the LLM to answer **only** from that context is sent
   to the highest-priority available provider.
5. If a provider fails, it's retried once, then the router automatically
   falls back to the next provider in the priority list.
6. If nothing relevant was retrieved, or the paper genuinely doesn't contain
   the answer, you'll get: *"I couldn't find this information in the
   uploaded paper."*
7. Identical questions (same prompt + parameters) are served from cache
   instead of re-querying a provider.

## Notes

- Each uploaded paper gets its own ChromaDB collection, so switching papers
  never mixes context between documents.
- `vectordb/chroma_service.py` implements an abstract `VectorStoreBase` so a
  FAISS backend can be dropped in later without touching any calling code.
- Debug mode in the sidebar exposes the retrieved chunks, similarity scores,
  provider used, response time, and cache status for every answer.
