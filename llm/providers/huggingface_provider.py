"""
Hugging Face Inference API provider.
"""
from __future__ import annotations

from typing import Iterator, Optional

import httpx

from llm.base_provider import BaseProvider, ProviderAuthError, ProviderError, ProviderTimeoutError
from utils.logger import get_logger

logger = get_logger(__name__)

_HF_INFERENCE_BASE_URL = "https://api-inference.huggingface.co/models"


class HuggingFaceProvider(BaseProvider):
    name = "huggingface"

    def __init__(self, api_key: Optional[str], model: str) -> None:
        self._api_key = api_key
        self._model = model

    def _headers(self) -> dict:
        if not self._api_key:
            raise ProviderAuthError(self.name, "HUGGINGFACE_API_KEY is not configured.")
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def generate(self, prompt: str, temperature: float, max_tokens: int, timeout: float) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": max(temperature, 0.01),
                "max_new_tokens": max_tokens,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }
        url = f"{_HF_INFERENCE_BASE_URL}/{self._model}"
        try:
            response = httpx.post(url, headers=self._headers(), json=payload, timeout=timeout)
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
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "")
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]
            raise ProviderError(self.name, f"Unexpected response shape: {data}")
        except ValueError as exc:
            raise ProviderError(self.name, f"Invalid JSON response: {exc}") from exc

    def stream(self, prompt: str, temperature: float, max_tokens: int, timeout: float) -> Iterator[str]:
        # The free Inference API does not reliably support token streaming
        # for arbitrary models, so we fall back to yielding the full result.
        yield self.generate(prompt, temperature, max_tokens, timeout)

    def health_check(self) -> bool:
        return bool(self._api_key)
