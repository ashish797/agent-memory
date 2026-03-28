"""Integration test: full agent memory flow."""

import os
import tempfile

from agent_memory import (
    SQLiteStore, HotMemory, ResponseCache,
    MemoryEntry, MemoryTier, ContentType, SourceType, TaskType,
)
from agent_memory.embedding import HashEmbedding
from agent_memory.normalizer import normalize_messages


def test_full_agent_flow():
    """
    Simulate a full agent conversation with memory:
    1. User asks a question
    2. Agent responds (cache miss)
    3. User asks similar question
    4. Agent hits cache (semantic or exact)
    5. Tool is called, result cached
    6. Tool called again with same args — cache hit
    """
    # Setup
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    embedding = HashEmbedding(dimension=64)
    store = SQLiteStore(db_path=path, embedding_engine=embedding)
    cache = ResponseCache(db=store, embedding_engine=embedding, session_id="test-session")
    hot = HotMemory(max_turns=20)
    
    # ── Turn 1: Cache miss ──────────────────────────────────────────
    messages1 = [
        {"role": "system", "content": "You are a helpful assistant. Tokens used: 0."},
        {"role": "user", "content": "What is recursion?"},
    ]
    
    result = cache.get(messages1)
    assert not result.hit  # First time, no cache
    
    # Agent responds
    response1 = "Recursion is when a function calls itself."
    cache.store(messages1, response1, model="gpt-4", cost_usd=0.02)
    hot.add_user_message("What is recursion?")
    hot.add_assistant_message(response1, model="gpt-4", cost=0.02)
    
    # ── Turn 2: Exact match ─────────────────────────────────────────
    # User asks same question (exact match after normalization)
    messages2 = [
        {"role": "system", "content": "You are a helpful assistant. Tokens used: 50."},
        {"role": "user", "content": "What is recursion?"},
    ]
    
    result = cache.get(messages2)
    assert result.hit
    assert result.hit_type.value == "client_exact"
    assert result.response == response1
    
    # Record the cache hit
    store.record_cache_hit(result.hit_type)
    
    # ── Turn 3: Tool call, cache result ─────────────────────────────
    from agent_memory.hot import ToolResult
    
    tool_result = ToolResult(
        tool_name="read_file",
        tool_call_id="tc_1",
        args={"path": "/etc/config.yaml"},
        result="api_key: secret123\ndatabase: postgres",
    )
    hot.recent_tool_results.append(tool_result)
    
    # Cache the tool result
    store.store_tool_result(
        cache_key=tool_result.cache_key(),
        tool_name="read_file",
        tool_args_hash="hash123",
        result=tool_result.result,
        session_id="test-session",
        ttl=3600,
    )
    
    # Same tool call again — should hit cache
    found = hot.get_tool_result("read_file", {"path": "/etc/config.yaml"})
    assert found is not None
    assert found.result == "api_key: secret123\ndatabase: postgres"
    
    # Also check via store
    cached_result = store.get_tool_result(tool_result.cache_key())
    assert cached_result == tool_result.result
    
    # ── Turn 4: Different question, miss ────────────────────────────
    messages3 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is sorting?"},
    ]
    
    result = cache.get(messages3)
    assert not result.hit  # Different question
    
    # ── Stats Check ─────────────────────────────────────────────────
    stats = store.get_stats()
    assert stats.exact_hits >= 1
    assert stats.total_requests >= 2  # We recorded at least 2 cache checks
    
    # Hot memory stats
    hot_stats = hot.stats()
    assert hot_stats["turns_stored"] >= 2  # We added 2 explicit turns
    assert hot_stats["tool_results_stored"] >= 1
    
    # Cost tracking
    store.record_cost_event = lambda e: store.record_cost(e)
    
    # Cleanup
    store.close()
    os.unlink(path)
    
    print("✅ Full agent flow test passed")


if __name__ == "__main__":
    test_full_agent_flow()
