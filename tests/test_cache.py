from config.settings import CacheConfig
from llm.cache import ResponseCache


def test_cache_hit_and_miss():
    cache = ResponseCache(CacheConfig(enabled=True, ttl_seconds=60, max_size=10))
    assert cache.get("prompt", 0.2, 100) is None

    cache.set("prompt", 0.2, 100, "answer")
    assert cache.get("prompt", 0.2, 100) == "answer"


def test_cache_disabled_never_hits():
    cache = ResponseCache(CacheConfig(enabled=False, ttl_seconds=60, max_size=10))
    cache.set("prompt", 0.2, 100, "answer")
    assert cache.get("prompt", 0.2, 100) is None


def test_cache_distinguishes_parameters():
    cache = ResponseCache(CacheConfig(enabled=True, ttl_seconds=60, max_size=10))
    cache.set("prompt", 0.2, 100, "answer-a")
    cache.set("prompt", 0.9, 100, "answer-b")
    assert cache.get("prompt", 0.2, 100) == "answer-a"
    assert cache.get("prompt", 0.9, 100) == "answer-b"
