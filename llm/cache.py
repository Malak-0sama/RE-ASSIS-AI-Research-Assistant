"""
TTL response cache keyed by prompt + generation parameters.
"""
from __future__ import annotations

from typing import Optional

from cachetools import TTLCache

from config.settings import CacheConfig
from utils.helpers import stable_hash
from utils.logger import get_logger

logger = get_logger(__name__)


class ResponseCache:
    """Thin wrapper around `cachetools.TTLCache` for LLM responses."""

    def __init__(self, config: CacheConfig) -> None:
        self._enabled = config.enabled
        self._cache: TTLCache = TTLCache(maxsize=config.max_size, ttl=config.ttl_seconds)

    def _key(self, prompt: str, temperature: float, max_tokens: int) -> str:
        return stable_hash(prompt, f"{temperature:.3f}", str(max_tokens))

    def get(self, prompt: str, temperature: float, max_tokens: int) -> Optional[str]:
        if not self._enabled:
            return None
        key = self._key(prompt, temperature, max_tokens)
        value = self._cache.get(key)
        if value is not None:
            logger.debug("Cache hit for prompt")
        return value

    def set(self, prompt: str, temperature: float, max_tokens: int, response: str) -> None:
        if not self._enabled:
            return
        key = self._key(prompt, temperature, max_tokens)
        self._cache[key] = response

    def clear(self) -> None:
        self._cache.clear()
