"""Tests for ResponseCache (exact + semantic matching)."""

import os
import tempfile

from agent_memory.cache import ResponseCache, CacheResult
from agent_memory.embedding import HashEmbedding
from agent_memory.store import SQLiteStore
from agent_memory.types import ContentType


def _temp_setup() -> tuple[ResponseCache, SQLiteStore, str]:
    """Create a temporary cache + store for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    embedding = HashEmbedding(dimension=64)
    store = SQLiteStore(db_path=path, embedding_engine=embedding)
    cache = ResponseCache(db=store, embedding_engine=embedding)
    return cache, store, path


def test_cache_miss_on_empty():
    """Empty cache should miss."""
    cache, store, path = _temp_setup()
    
    result = cache.get([{"role": "user", "content": "Hello"}])
    
    assert not result.hit
    assert result.hit_type.value == "none"
    
    store.close()
    os.unlink(path)


def test_exact_match_after_store():
    """Storing and retrieving the same messages should hit."""
    cache, store, path = _temp_setup()
    
    messages = [{"role": "user", "content": "What is Python?"}]
    response = "Python is a programming language."
    
    # Store
    cache.store(messages, response, model="gpt-4", cost_usd=0.01)
    
    # Retrieve exact same messages
    result = cache.get(messages)
    
    assert result.hit
    assert result.hit_type.value == "client_exact"
    assert result.response == response
    
    store.close()
    os.unlink(path)


def test_exact_match_despite_key_order():
    """Key order in messages shouldn't matter."""
    cache, store, path = _temp_setup()
    
    messages1 = [{"role": "user", "content": "Hello"}]
    messages2 = [{"content": "Hello", "role": "user"}]  # Different key order
    
    cache.store(messages1, "Hi there!", model="gpt-4")
    
    result = cache.get(messages2)
    assert result.hit
    assert result.response == "Hi there!"
    
    store.close()
    os.unlink(path)


def test_different_messages_miss():
    """Different content should miss."""
    cache, store, path = _temp_setup()
    
    cache.store(
        [{"role": "user", "content": "What is Python?"}],
        "Python is a programming language.",
        model="gpt-4",
    )
    
    result = cache.get([{"role": "user", "content": "What is Rust?"}])
    
    assert not result.hit
    
    store.close()
    os.unlink(path)


def test_cost_saved_on_hit():
    """Cache hits should report cost saved."""
    cache, store, path = _temp_setup()
    
    messages = [{"role": "user", "content": "Hello"}]
    cost = 0.05
    
    cache.store(messages, "Hi!", model="gpt-4", cost_usd=cost)
    result = cache.get(messages)
    
    assert result.hit
    assert result.cost_saved == cost
    
    store.close()
    os.unlink(path)


def test_system_prompt_volatile_normalization():
    """System prompt changes (volatile) shouldn't affect cache."""
    cache, store, path = _temp_setup()
    
    # First request
    messages1 = [
        {"role": "system", "content": "Tokens used: 1234. Answer concisely."},
        {"role": "user", "content": "Hello"},
    ]
    cache.store(messages1, "Hi!", model="gpt-4")
    
    # Second request with different volatile values
    messages2 = [
        {"role": "system", "content": "Tokens used: 5678. Cache rate: 90%. Answer concisely."},
        {"role": "user", "content": "Hello"},
    ]
    result = cache.get(messages2)
    
    # Should hit despite different token counts / cache rates
    assert result.hit
    
    store.close()
    os.unlink(path)
