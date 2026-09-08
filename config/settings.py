"""
Centralized configuration management for the AI Research Assistant.

Combines environment-based secrets (API keys) loaded via pydantic-settings
with structured, non-secret configuration loaded from config.yaml. The
resulting `Settings` object is the single source of truth for every
configurable value in the system. No other module should hardcode a value
that belongs here.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
DEFAULT_CONFIG_YAML_PATH = CONFIG_DIR / "config.yaml"


class ApiKeys(BaseSettings):
    """Secrets loaded exclusively from environment variables / .env file.

    Kept in a dedicated BaseSettings subclass (instead of mixed in with the
    YAML-driven config) so secrets are never accidentally serialized,
    logged, or written back to disk alongside ordinary configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, alias="HUGGINGFACE_API_KEY")



class LLMConfig(BaseModel):
    provider_priority: List[str] = Field(
        default_factory=lambda: ["huggingface","groq",  "gemini" ]
    )
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_seconds: float = 30.0
    max_retries_per_provider: int = 1

    huggingface_model: str = "deepseek-ai/DeepSeek-V4-Flash"
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-1.5-flash"




class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    normalize_embeddings: bool = True
    batch_size: int = 32


class ChunkingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 150
    separators: List[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])


class RetrievalConfig(BaseModel):
    top_k: int = 5
    score_threshold: Optional[float] = None


class VectorDBConfig(BaseModel):
    persist_directory: str = "./data/chroma"
    collection_prefix: str = "paper"


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_seconds: int = 3600
    max_size: int = 256


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "./logs"
    rotation: str = "10 MB"
    retention: str = "7 days"


class AppConfig(BaseModel):
    """Root configuration object assembled from config.yaml."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vectordb: VectorDBConfig = Field(default_factory=VectorDBConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class Settings(BaseModel):
    """Unified settings: YAML-driven config plus environment secrets."""

    app: AppConfig
    api_keys: ApiKeys

    model_config = {"arbitrary_types_allowed": True}


def _load_yaml_config(path: Path) -> AppConfig:
    if not path.exists():
        return AppConfig()
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig(**raw)


@functools.lru_cache(maxsize=1)
def get_settings(config_path: Optional[str] = None) -> Settings:
    """Return a cached, process-wide Settings instance.

    `lru_cache` guarantees the YAML file is parsed once per process while
    still letting tests bypass the cache via `get_settings.cache_clear()`.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_YAML_PATH
    app_config = _load_yaml_config(path)
    api_keys = ApiKeys()
    return Settings(app=app_config, api_keys=api_keys)
