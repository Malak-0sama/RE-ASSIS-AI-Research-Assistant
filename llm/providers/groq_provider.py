"""
Groq provider (OpenAI-compatible chat completions API).
"""
from __future__ import annotations

from typing import Iterator, Optional

import httpx

from llm.base_provider import BaseProvider, ProviderAuthError, ProviderError, ProviderTimeoutError
from utils.logger import get_logger

logger = get_logger(__name__)

_GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(BaseProvider):
    name = "groq"

    def __init__(self, api_key: Optional[str], model: str) -> None:
        self._api_key = api_key
        self._model = model

    def _headers(self) -> dict:
        if not self._api_key:
            raise ProviderAuthError(self.name, "GROQ_API_KEY is not configured.")
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def generate(self, prompt: str, temperature: float, max_tokens: int, timeout: float) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            response = httpx.post(
                _GROQ_CHAT_COMPLETIONS_URL, headers=self._headers(), json=payload, timeout=timeout
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self.name, f"Request timed out after {timeout}s") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"Network error: {exc}") from exc

        if response.status_code == 401:
            raise ProviderAuthError(self.name, "Invalid or missing API key.")
        if response.status_code >= 400:
            raise ProviderError(self.name, f"HTTP {response.status_code}: {response.text[:300]}")

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(self.name, f"Unexpected response shape: {exc}") from exc

    def stream(self, prompt: str, temperature: float, max_tokens: int, timeout: float) -> Iterator[str]:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        try:
            with httpx.stream(
                "POST", _GROQ_CHAT_COMPLETIONS_URL, headers=self._headers(), json=payload, timeout=timeout
            ) as response:
                if response.status_code >= 400:
                    raise ProviderError(self.name, f"HTTP {response.status_code}")
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line.removeprefix("data:").strip()
                    if chunk == "[DONE]":
                        break
                    yield chunk
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self.name, f"Stream timed out after {timeout}s") from exc

    def health_check(self) -> bool:
        return bool(self._api_key)
