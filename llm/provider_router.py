"""
Multi-provider LLM router with priority ordering, single-retry-then-fallback
behavior, and response caching.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from config.settings import LLMConfig
from llm.base_provider import BaseProvider, LLMResponse, ProviderError
from llm.cache import ResponseCache
from utils.logger import get_logger

logger = get_logger(__name__)


class AllProvidersFailedError(Exception):
    """Raised when every configured provider fails to answer a request."""


class ProviderRouter:
    """Routes generation requests across providers in priority order."""

    def __init__(
        self,
        providers: Dict[str, BaseProvider],
        llm_config: LLMConfig,
        cache: ResponseCache,
    ) -> None:
        self._providers = providers
        self._llm_config = llm_config
        self._cache = cache

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> LLMResponse:
        start = time.perf_counter()

        cached = self._cache.get(prompt, temperature, max_tokens)
        if cached is not None:
            logger.info("Serving response from cache")
            return LLMResponse(
                text=cached,
                provider_name="cache",
                latency_seconds=time.perf_counter() - start,
                from_cache=True,
            )

        errors: List[str] = []

        for provider_name in self._llm_config.provider_priority:
            provider = self._providers.get(provider_name)

            if provider is None:
                logger.warning(f"Provider '{provider_name}' is not registered; skipping")
                continue

            text = self._try_provider(
                provider,
                prompt,
                temperature,
                max_tokens,
                errors,
            )

            if text is not None:
                self._cache.set(prompt, temperature, max_tokens, text)

                latency = time.perf_counter() - start

                logger.info(
                    f"Provider '{provider_name}' succeeded in {latency:.2f}s"
                )

                return LLMResponse(
                    text=text,
                    provider_name=provider_name,
                    latency_seconds=latency,
                    from_cache=False,
                )

        raise AllProvidersFailedError(
            "All configured providers failed:\n" + "\n".join(errors)
        )

    def _try_provider(
        self,
        provider: BaseProvider,
        prompt: str,
        temperature: float,
        max_tokens: int,
        errors: List[str],
    ) -> Optional[str]:

        attempts = 1 + self._llm_config.max_retries_per_provider

        fatal_keywords = [
            "401",
            "403",
            "404",
            "413",
            "unauthorized",
            "forbidden",
            "not configured",
            "api_key",
            "api key",
            "invalid api key",
            "context length",
            "request too large",
        ]

        for attempt in range(1, attempts + 1):

            attempt_start = time.perf_counter()

            try:
                text = provider.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self._llm_config.timeout_seconds,
                )

                latency = time.perf_counter() - attempt_start

                logger.info(
                    f"[{provider.name}] attempt {attempt}/{attempts} "
                    f"succeeded in {latency:.2f}s"
                )

                return text

            except ProviderError as exc:

                latency = time.perf_counter() - attempt_start

                message = (
                    f"[{provider.name}] attempt {attempt}/{attempts} "
                    f"failed after {latency:.2f}s: {exc.message}"
                )

                logger.warning(message)
                errors.append(message)


                if any(k in exc.message.lower() for k in fatal_keywords):
                    logger.info(
                        f"[{provider.name}] fatal error detected. "
                        f"Skipping retries and moving to next provider."
                    )
                    break

        return None