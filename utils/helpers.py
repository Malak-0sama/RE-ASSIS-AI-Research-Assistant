"""
Small, dependency-light utility functions shared across modules.
"""
from __future__ import annotations

import hashlib
import time
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def timed(func: F) -> F:
    """Decorator that records the wrapped call's wall-clock duration.

    The elapsed time (in seconds) of the most recent call is exposed via
    the `last_call_seconds` attribute on the wrapped function, keeping the
    original call signature and return value untouched for callers.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        wrapper.last_call_seconds = time.perf_counter() - start  # type: ignore[attr-defined]
        return result

    wrapper.last_call_seconds = 0.0  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def stable_hash(*parts: str) -> str:
    """Produce a deterministic SHA-256 hex digest from arbitrary string parts.

    Used for cache keys and deterministic document/chunk identifiers so the
    same input always maps to the same id across runs and processes.
    """
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x1f")  # unit separator, avoids accidental collisions
    return hasher.hexdigest()


def truncate(text: str, max_chars: int = 200) -> str:
    """Truncate text for safe, compact logging."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def safe_filename(name: str) -> str:
    """Strip characters that are unsafe for filesystem paths."""
    keep = "-_.() "
    return "".join(c for c in name if c.isalnum() or c in keep).strip() or "document"
