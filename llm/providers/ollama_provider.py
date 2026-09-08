"""
Local Ollama provider — no API key required.
"""
from __future__ import annotations

from typing import Iterator

import httpx

from llm.base_provider import BaseProvider, ProviderError, ProviderTimeoutError
from utils.logger import get_logger

logger = get_logger(__name__)


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    def generate(self, prompt: str, temperature: float, max_tokens: int, timeout: float) -> str:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            response = httpx.post(f"{self._base_url}/api/generate", json=payload, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self.name, f"Request timed out after {timeout}s") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"Cannot reach Ollama at {self._base_url}: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(self.name, f"HTTP {response.status_code}: {response.text[:300]}")

        try:
            return response.json()["response"]
        except (KeyError, ValueError) as exc:
            raise ProviderError(self.name, f"Unexpected response shape: {exc}") from exc

    def stream(self, prompt: str, temperature: float, max_tokens: int, timeout: float) -> Iterator[str]:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            with httpx.stream(
                "POST", f"{self._base_url}/api/generate", json=payload, timeout=timeout
            ) as response:
                if response.status_code >= 400:
                    raise ProviderError(self.name, f"HTTP {response.status_code}")
                for line in response.iter_lines():
                    if line.strip():
                        yield line
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self.name, f"Stream timed out after {timeout}s") from exc

    def health_check(self) -> bool:
        try:
            response = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False
