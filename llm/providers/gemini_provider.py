"""
Google Gemini provider (Generative Language API).
"""
from __future__ import annotations

from typing import Iterator, Optional

import httpx

from llm.base_provider import BaseProvider, ProviderAuthError, ProviderError, ProviderTimeoutError
from utils.logger import get_logger

logger = get_logger(__name__)

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, api_key: Optional[str], model: str) -> None:
        self._api_key = api_key
        self._model = model

    def _url(self, action: str) -> str:
        if not self._api_key:
            raise ProviderAuthError(self.name, "GEMINI_API_KEY is not configured.")
        return f"{_GEMINI_BASE_URL}/{self._model}:{action}?key={self._api_key}"

    def _payload(self, prompt: str, temperature: float, max_tokens: int) -> dict:
        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

    def generate(self, prompt: str, temperature: float, max_tokens: int, timeout: float) -> str:
        try:
            response = httpx.post(
                self._url("generateContent"),
                json=self._payload(prompt, temperature, max_tokens),
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self.name, f"Request timed out after {timeout}s") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"Network error: {exc}") from exc

        if response.status_code in (401, 403):
            raise ProviderAuthError(self.name, "Invalid or missing API key.")
        if response.status_code >= 400:
            raise ProviderError(self.name, f"HTTP {response.status_code}: {response.text[:300]}")

        try:
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(self.name, f"Unexpected response shape: {exc}") from exc

    def stream(self, prompt: str, temperature: float, max_tokens: int, timeout: float) -> Iterator[str]:
        try:
            with httpx.stream(
                "POST",
                self._url("streamGenerateContent"),
                json=self._payload(prompt, temperature, max_tokens),
                timeout=timeout,
            ) as response:
                if response.status_code >= 400:
                    raise ProviderError(self.name, f"HTTP {response.status_code}")
                for line in response.iter_lines():
                    if line.strip():
                        yield line
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self.name, f"Stream timed out after {timeout}s") from exc

    def health_check(self) -> bool:
        return bool(self._api_key)
