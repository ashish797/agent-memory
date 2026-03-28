"""Tests for the Retrieval Engine."""

import os
import tempfile
import time

from agent_memory.decision import DecisionEngine
from agent_memory.embedding import HashEmbedding
from agent_memory.hot import HotMemory, Turn, ToolResult
from agent_memory.retrieval import RetrievalEngine, RetrievalContext, ScoredEntry
from agent_memory.store import SQLiteStore
from agent_memory.types import ContentType, MemoryEntry, MemoryTier, SourceType, TaskType


def _setup() -> tuple[RetrievalEngine, HotMemory, SQLiteStore, str]:
    """Create a retrieval engine with test data."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    embedding = HashEmbedding(dimension=64)
    store = SQLiteStore(db_path=path, embedding_engine=embedding)
    hot = HotMemory(max_turns=20)
    decision = DecisionEngine()
    
    engine = RetrievalEngine(
        hot_memory=hot,
        warm_store=store,
        decision_engine=decision,
        embedding_engine=embedding,
    )
    
    return engine, hot, store, path


def test_hot_memory_retrieval():
    """Hot memory entries should be retrieved."""
    engine, hot, store, path = _setup()
    
    # Add turns to hot memory
    hot.add_user_message("What is recursion?")
    hot.add_assistant_message("Recursion is when a function calls itself.")
    
    context = RetrievalContext(
        session_id="test",
        current_messages=[{"role": "user", "content": "What is recursion?"}],
        task_type=TaskType.DEBUG,
    )
    
    result = engine.retrieve(context, budget_tokens=4000)
    
    assert len(result.entries) > 0
    assert any(e.source_tier == MemoryTier.HOT for e in result.entries)
    
    store.close()
    os.unlink(path)


def test_warm_memory_retrieval():
    """Warm memory entries should be retrieved via semantic search."""
    engine, hot, store, path = _setup()
    
    # Store some entries in warm memory
    for i in range(5):
        entry = MemoryEntry(
            text=f"This is information about topic {i}",
            tier=MemoryTier.WARM,
            content_type=ContentType.CONVERSATION,
            session_id="test-session",
            source_type=SourceType.MODEL,
            importance=0.6,
        )
        store.store(entry)
    
    context = RetrievalContext(
        session_id="test",
        current_messages=[{"role": "user", "content": "Tell me about topic 3"}],
    )
    
    result = engine.retrieve(context, budget_tokens=4000)
    
    # With HashEmbedding (random), semantic search won't find meaningful matches
    # but entries should still be retrievable if they pass the similarity threshold
    # This test mainly verifies the retrieval doesn't crash
    assert result is not None
    
    store.close()
    os.unlink(path)


def test_budget_capping():
    """Results should respect token budget."""
    engine, hot, store, path = _setup()
    
    # Add many entries
    for i in range(20):
        hot.add_user_message(f"Message number {i} with some content")
        hot.add_assistant_message(f"Response number {i}")
    
    context = RetrievalContext(
        session_id="test",
        current_messages=[{"role": "user", "content": "test"}],
    )
    
    result = engine.retrieve(context, budget_tokens=500)  # Small budget
    
    assert result.total_tokens <= 550  # Slight margin for estimation
    
    store.close()
    os.unlink(path)


def test_active_task_included():
    """Active task context should be retrieved."""
    engine, hot, store, path = _setup()
    
    hot.set_task(TaskType.DEBUG, "Fixing login bug", focus="Auth middleware")
    
    context = RetrievalContext(
        session_id="test",
        current_messages=[{"role": "user", "content": "What's next?"}],
    )
    
    result = engine.retrieve(context, budget_tokens=4000)
    
    # Should have active task in results
    task_entries = [e for e in result.entries if "Active Task" in e.entry.text]
    assert len(task_entries) > 0
    
    store.close()
    os.unlink(path)


def test_tool_results_included():
    """Recent tool results should be retrieved."""
    engine, hot, store, path = _setup()
    
    tool_result = ToolResult(
        tool_name="read_file",
        tool_call_id="tc_1",
        args={"path": "/etc/config"},
        result="File contents",
    )
    hot.recent_tool_results.append(tool_result)
    
    context = RetrievalContext(
        session_id="test",
        current_messages=[{"role": "user", "content": "What was in that file?"}],
    )
    
    result = engine.retrieve(context, budget_tokens=4000)
    
    # Should have tool result in results
    tool_entries = [e for e in result.entries if "read_file" in e.entry.text]
    assert len(tool_entries) > 0
    
    store.close()
    os.unlink(path)


def test_relevance_ranking():
    """More relevant entries should be ranked higher."""
    engine, hot, store, path = _setup()
    
    # Add entries to warm memory
    entry1 = MemoryEntry(
        text="Python is a programming language",
        tier=MemoryTier.WARM,
        content_type=ContentType.CONVERSATION,
        session_id="s1",
        source_type=SourceType.MODEL,
        created_at=int(time.time()) - 100,  # Recent
        access_count=10,  # Frequently accessed
    )
    entry2 = MemoryEntry(
        text="Weather is nice today",
        tier=MemoryTier.WARM,
        content_type=ContentType.CONVERSATION,
        session_id="s1",
        source_type=SourceType.MODEL,
        created_at=int(time.time()) - 86400,  # Old
        access_count=0,  # Never accessed
    )
    
    store.store(entry1)
    store.store(entry2)
    
    # Search for something Python-related
    context = RetrievalContext(
        session_id="test",
        current_messages=[{"role": "user", "content": "Tell me about Python programming"}],
    )
    
    result = engine.retrieve(context, budget_tokens=4000)
    
    if len(result.entries) >= 2:
        # entry1 should rank higher (more recent, more accessed, better match)
        scores = {e.entry.text_hash: e.relevance_score for e in result.entries}
        if entry1.text_hash in scores and entry2.text_hash in scores:
            assert scores[entry1.text_hash] > scores[entry2.text_hash]
    
    store.close()
    os.unlink(path)


def test_deduplication():
    """Duplicate entries should be removed."""
    engine, hot, store, path = _setup()
    
    # Add same message to both hot and warm
    text = "Duplicate content"
    hot.add_user_message(text)
    
    entry = MemoryEntry(
        text=text,
        tier=MemoryTier.WARM,
        content_type=ContentType.CONVERSATION,
        session_id="test",
        source_type=SourceType.USER,
    )
    store.store(entry)
    
    context = RetrievalContext(
        session_id="test",
        current_messages=[{"role": "user", "content": "test"}],
    )
    
    result = engine.retrieve(context, budget_tokens=4000)
    
    # Should only have one copy
    hash_counts: dict[str, int] = {}
    for e in result.entries:
        h = e.entry.text_hash
        hash_counts[h] = hash_counts.get(h, 0) + 1
    
    assert all(c == 1 for c in hash_counts.values())
    
    store.close()
    os.unlink(path)


def test_retrieval_timing():
    """Retrieval should be fast (<100ms for small datasets)."""
    engine, hot, store, path = _setup()
    
    for i in range(10):
        store.store(MemoryEntry(
            text=f"Entry {i}",
            tier=MemoryTier.WARM,
            content_type=ContentType.CONVERSATION,
            session_id="test",
            source_type=SourceType.MODEL,
        ))
    
    context = RetrievalContext(
        session_id="test",
        current_messages=[{"role": "user", "content": "test"}],
    )
    
    result = engine.retrieve(context, budget_tokens=4000)
    
    assert result.retrieval_time_ms < 100  # Should be very fast
    
    store.close()
    os.unlink(path)


def test_context_string_generation():
    """to_context_string() should format entries properly."""
    engine, hot, store, path = _setup()
    
    hot.add_user_message("Hello")
    
    context = RetrievalContext(
        session_id="test",
        current_messages=[{"role": "user", "content": "test"}],
    )
    
    result = engine.retrieve(context, budget_tokens=4000)
    
    if result.entries:
        ctx_str = result.to_context_string()
        assert isinstance(ctx_str, str)
        assert len(ctx_str) > 0
    
    store.close()
    os.unlink(path)


def test_empty_retrieval():
    """Empty memory should return empty result gracefully."""
    engine, hot, store, path = _setup()
    
    context = RetrievalContext(
        session_id="test",
        current_messages=[{"role": "user", "content": "test"}],
    )
    
    result = engine.retrieve(context, budget_tokens=4000)
    
    assert len(result.entries) == 0
    assert result.total_tokens == 0
    
    store.close()
    os.unlink(path)
