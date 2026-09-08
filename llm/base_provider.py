"""
Abstract base class every LLM provider must implement, plus shared response
and error types used by the provider router and cache.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from pydantic import BaseModel


class ProviderError(Exception):
    """Raised when a provider fails to generate a response."""

    def __init__(self, provider_name: str, message: str) -> None:
        self.provider_name = provider_name
        self.message = message
        super().__init__(f"[{provider_name}] {message}")


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request exceeds the configured timeout."""


class ProviderAuthError(ProviderError):
    """Raised when a provider rejects the request due to missing/invalid credentials."""


class LLMResponse(BaseModel):
    """A normalized response returned by any provider (or the cache)."""

    text: str
    provider_name: str
    latency_seconds: float
    from_cache: bool = False


class BaseProvider(ABC):
    """Common interface implemented by every LLM provider integration."""

    name: str

    @abstractmethod
    def generate(self, prompt: str, temperature: float, max_tokens: int, timeout: float) -> str:
        """Generate a complete response for `prompt`. Raises ProviderError on failure."""
        ...

    @abstractmethod
    def stream(self, prompt: str, temperature: float, max_tokens: int, timeout: float) -> Iterator[str]:
        """Yield response text incrementally. Raises ProviderError on failure."""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the provider is reachable and configured correctly."""
        ...
