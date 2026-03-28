"""Tests for SQLiteStore (warm memory tier)."""

import os
import tempfile
import time

from agent_memory.store import SQLiteStore
from agent_memory.types import (
    CacheHitType,
    ContentType,
    CostEvent,
    MemoryEntry,
    MemoryTier,
    SourceType,
    TaskType,
)


def _temp_store() -> tuple[SQLiteStore, str]:
    """Create a temporary store for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = SQLiteStore(db_path=path, embedding_engine=None)
    return store, path


def test_store_and_retrieve():
    """Basic store and retrieve by hash."""
    store, path = _temp_store()
    
    entry = MemoryEntry(
        text="Hello, world!",
        tier=MemoryTier.WARM,
        content_type=ContentType.CONVERSATION,
        session_id="test-session",
        source_type=SourceType.USER,
    )
    
    store.store(entry)
    
    # Retrieve by hash
    found = store.get_by_hash(entry.text_hash)
    assert found is not None
    assert found.text == "Hello, world!"
    assert found.session_id == "test-session"
    
    store.close()
    os.unlink(path)


def test_store_batch():
    """Batch storage should work."""
    store, path = _temp_store()
    
    entries = [
        MemoryEntry(
            text=f"Message {i}",
            tier=MemoryTier.WARM,
            content_type=ContentType.CONVERSATION,
            session_id="test-session",
            source_type=SourceType.USER,
        )
        for i in range(10)
    ]
    
    ids = store.store_batch(entries)
    assert len(ids) == 10
    
    # All should be retrievable
    for entry in entries:
        found = store.get_by_hash(entry.text_hash)
        assert found is not None
    
    store.close()
    os.unlink(path)


def test_hash_collision_different_entries():
    """Different texts should have different hashes."""
    store, path = _temp_store()
    
    e1 = MemoryEntry(
        text="Hello",
        tier=MemoryTier.WARM,
        content_type=ContentType.CONVERSATION,
        session_id="s1",
        source_type=SourceType.USER,
    )
    e2 = MemoryEntry(
        text="World",
        tier=MemoryTier.WARM,
        content_type=ContentType.CONVERSATION,
        session_id="s1",
        source_type=SourceType.USER,
    )
    
    store.store(e1)
    store.store(e2)
    
    assert e1.text_hash != e2.text_hash
    assert store.get_by_hash(e1.text_hash).text == "Hello"
    assert store.get_by_hash(e2.text_hash).text == "World"
    
    store.close()
    os.unlink(path)


def test_get_by_id():
    """Retrieve by ID should work."""
    store, path = _temp_store()
    
    entry = MemoryEntry(
        text="Test",
        tier=MemoryTier.WARM,
        content_type=ContentType.CONVERSATION,
        session_id="s1",
        source_type=SourceType.USER,
    )
    
    store.store(entry)
    
    found = store.get_by_id(entry.id)
    assert found is not None
    assert found.id == entry.id
    
    store.close()
    os.unlink(path)


def test_session_entries():
    """Session filtering should work."""
    store, path = _temp_store()
    
    for i in range(5):
        store.store(MemoryEntry(
            text=f"Session A msg {i}",
            tier=MemoryTier.WARM,
            content_type=ContentType.CONVERSATION,
            session_id="session-a",
            source_type=SourceType.USER,
        ))
    
    for i in range(3):
        store.store(MemoryEntry(
            text=f"Session B msg {i}",
            tier=MemoryTier.WARM,
            content_type=ContentType.CONVERSATION,
            session_id="session-b",
            source_type=SourceType.USER,
        ))
    
    a_entries = store.get_session_entries("session-a", limit=10)
    b_entries = store.get_session_entries("session-b", limit=10)
    
    assert len(a_entries) == 5
    assert len(b_entries) == 3
    
    store.close()
    os.unlink(path)


def test_tool_cache():
    """Tool cache should work with TTL."""
    store, path = _temp_store()
    
    cache_key = "abc123"
    
    # Store
    store.store_tool_result(
        cache_key=cache_key,
        tool_name="read_file",
        tool_args_hash="def456",
        result="File contents here",
        session_id="s1",
        ttl=3600,
    )
    
    # Retrieve
    result = store.get_tool_result(cache_key)
    assert result == "File contents here"
    
    # Expired should return None
    store.store_tool_result(
        cache_key="expired-key",
        tool_name="read_file",
        tool_args_hash="ghi789",
        result="Expired data",
        ttl=-1,  # Already expired
    )
    
    expired = store.get_tool_result("expired-key")
    assert expired is None
    
    store.close()
    os.unlink(path)


def test_tool_cache_cleanup():
    """Expired tool cache entries should be cleaned up."""
    store, path = _temp_store()
    
    store.store_tool_result("key1", "tool", "h1", "result1", ttl=3600)
    store.store_tool_result("key2", "tool", "h2", "result2", ttl=-1)
    store.store_tool_result("key3", "tool", "h3", "result3", ttl=-1)
    
    removed = store.cleanup_expired_tool_cache()
    assert removed == 2
    
    assert store.get_tool_result("key1") == "result1"
    assert store.get_tool_result("key2") is None
    assert store.get_tool_result("key3") is None
    
    store.close()
    os.unlink(path)


def test_cost_tracking():
    """Cost events should be trackable."""
    store, path = _temp_store()
    
    store.record_cost(CostEvent(
        timestamp=int(time.time()),
        session_id="s1",
        event_type="llm_call",
        model="gpt-4",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.01,
        cache_hit_type=CacheHitType.NONE,
    ))
    
    store.record_cost(CostEvent(
        timestamp=int(time.time()),
        session_id="s1",
        event_type="llm_call",
        model="gpt-4",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.01,
        cache_hit_type=CacheHitType.CLIENT_EXACT,
    ))
    
    cost = store.get_session_cost("s1")
    assert cost["total_calls"] == 2
    assert cost["total_cost"] == 0.02
    
    store.close()
    os.unlink(path)


def test_cache_stats():
    """Cache hit stats should accumulate."""
    store, path = _temp_store()
    
    store.record_cache_hit(CacheHitType.CLIENT_EXACT)
    store.record_cache_hit(CacheHitType.CLIENT_EXACT)
    store.record_cache_hit(CacheHitType.CLIENT_SEMANTIC)
    store.record_cache_hit(CacheHitType.NONE)
    
    stats = store.get_stats()
    assert stats.exact_hits == 2
    assert stats.semantic_hits == 1
    assert stats.misses == 1
    assert stats.total_requests == 4
    
    store.close()
    os.unlink(path)


def test_eviction_lru():
    """LRU eviction should remove oldest entries."""
    store, path = _temp_store()
    store.max_entries = 5
    
    # Add 10 entries
    for i in range(10):
        store.store(MemoryEntry(
            text=f"Entry {i}",
            tier=MemoryTier.WARM,
            content_type=ContentType.CONVERSATION,
            session_id="s1",
            source_type=SourceType.USER,
        ))
    
    stats = store.get_stats()
    assert stats.total_entries <= 5
    
    store.close()
    os.unlink(path)


def test_explicit_eviction():
    """Explicit eviction by policy."""
    store, path = _temp_store()
    
    for i in range(10):
        store.store(MemoryEntry(
            text=f"Entry {i}",
            tier=MemoryTier.WARM,
            content_type=ContentType.CONVERSATION,
            session_id="s1",
            source_type=SourceType.USER,
            importance=0.5 + (i * 0.05),  # Later entries have higher importance
        ))
    
    # Evict 3 lowest importance
    removed = store.evict("lowest_importance", count=3)
    assert removed == 3
    
    stats = store.get_stats()
    assert stats.total_entries == 7
    
    store.close()
    os.unlink(path)


def test_compact():
    """Compaction should remove expired entries."""
    store, path = _temp_store()
    
    store.store(MemoryEntry(
        text="Normal entry",
        tier=MemoryTier.WARM,
        content_type=ContentType.CONVERSATION,
        session_id="s1",
        source_type=SourceType.USER,
    ))
    
    store.store(MemoryEntry(
        text="Expired entry",
        tier=MemoryTier.WARM,
        content_type=ContentType.CONVERSATION,
        session_id="s1",
        source_type=SourceType.USER,
        expires_at=int(time.time()) - 1000,  # Already expired
    ))
    
    report = store.compact()
    assert report["expired_removed"] >= 1
    
    store.close()
    os.unlink(path)
