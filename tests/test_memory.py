"""Tests for the main AgentMemory SDK interface."""

import os
import tempfile

from agent_memory.memory import AgentMemory, MemoryConfig, CacheCheckResult, TurnResult
from agent_memory.types import ContentType, TaskType


def _setup() -> tuple[AgentMemory, str]:
    """Create a test memory instance."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    config = MemoryConfig(
        db_path=path,
        embedding_provider="hash",  # No external deps
    )
    memory = AgentMemory(config)
    memory.start_session("test-session")
    
    return memory, path


def test_session_lifecycle():
    """Session start and tracking."""
    memory, path = _setup()
    
    assert memory.session_id == "test-session"
    assert memory.session_cost == 0.0
    assert memory.turn_count == 0
    
    memory.close()
    os.unlink(path)


def test_cache_miss_first_time():
    """First request should miss cache."""
    memory, path = _setup()
    
    messages = [
        {"role": "system", "content": "You are helpful. Tokens used: 0."},
        {"role": "user", "content": "What is Python?"},
    ]
    
    result = memory.check_cache(messages)
    
    assert not result.hit
    assert result.hit_type.value == "none"
    
    memory.close()
    os.unlink(path)


def test_cache_hit_after_storing():
    """Cache hit after recording a turn."""
    memory, path = _setup()
    
    messages = [
        {"role": "system", "content": "You are helpful. Tokens used: 0."},
        {"role": "user", "content": "What is Python?"},
    ]
    
    # Record a turn
    memory.record_turn(
        messages=messages,
        response="Python is a programming language.",
        model="gpt-4",
        cost_usd=0.02,
    )
    
    # Check cache again
    result = memory.check_cache(messages)
    
    assert result.hit
    assert "Python" in result.response
    
    memory.close()
    os.unlink(path)


def test_record_turn_updates_stats():
    """Recording turns should update session stats."""
    memory, path = _setup()
    
    messages = [{"role": "user", "content": "Hello"}]
    
    memory.record_turn(messages, "Hi!", model="gpt-4", cost_usd=0.01)
    memory.record_turn(messages, "Hi again!", model="gpt-4", cost_usd=0.015)
    
    assert memory.turn_count == 2
    assert abs(memory.session_cost - 0.025) < 0.001
    
    memory.close()
    os.unlink(path)


def test_tool_cache_roundtrip():
    """Tool results should cache and retrieve."""
    memory, path = _setup()
    
    # Store tool result
    memory.record_tool_result(
        tool_name="read_file",
        args={"path": "/etc/config.yaml"},
        result="key: value",
        ttl=3600,
    )
    
    # Retrieve
    cached = memory.check_tool_cache("read_file", {"path": "/etc/config.yaml"})
    assert cached == "key: value"
    
    # Different args should miss
    not_found = memory.check_tool_cache("read_file", {"path": "/etc/other.yaml"})
    assert not_found is None
    
    memory.close()
    os.unlink(path)


def test_retrieve_context():
    """Retrieval should return context from hot memory."""
    memory, path = _setup()
    
    # Add some turns to hot memory
    memory.record_turn(
        [{"role": "user", "content": "What is recursion?"}],
        "Recursion is when a function calls itself.",
        model="gpt-4",
        cost_usd=0.01,
    )
    
    # Retrieve for similar query
    result = memory.retrieve([{"role": "user", "content": "Tell me more about recursion"}])
    
    assert len(result.entries) > 0
    assert result.total_tokens > 0
    
    memory.close()
    os.unlink(path)


def test_augmented_messages():
    """Augmented messages should include memory context."""
    memory, path = _setup()
    
    memory.record_turn(
        [{"role": "user", "content": "I love Python"}],
        "Great choice!",
        model="gpt-4",
    )
    
    messages = [{"role": "user", "content": "What did I say I love?"}]
    augmented = memory.get_augmented_messages(messages)
    
    # Should have memory context prepended
    assert len(augmented) > len(messages)
    assert augmented[-1]["content"] == messages[0]["content"]  # Original message preserved
    
    memory.close()
    os.unlink(path)


def test_model_routing():
    """Model routing should work."""
    memory, path = _setup()
    
    # Simple task → cheap
    route = memory.route_to_model(task_description="What is 2+2?")
    assert route.route == "cheap"
    
    # Complex task → advanced
    route = memory.route_to_model(
        task_description="Analyze the architecture and design a comprehensive microservices solution",
        task_type=TaskType.DESIGN,
    )
    assert route.route == "advanced"
    
    memory.close()
    os.unlink(path)


def test_task_management():
    """Task tracking should work."""
    memory, path = _setup()
    
    memory.set_task(TaskType.DEBUG, "Fix login bug", focus="Auth middleware")
    assert memory.hot.active_task is not None
    assert memory.hot.active_task.task_type == TaskType.DEBUG
    
    memory.complete_step("Found the bug")
    assert len(memory.hot.active_task.steps_completed) == 1
    
    memory.close()
    os.unlink(path)


def test_preferences():
    """Session preferences should work."""
    memory, path = _setup()
    
    memory.set_preference("language", "Python")
    assert memory.get_preference("language") == "Python"
    assert memory.get_preference("missing", "default") == "default"
    
    memory.close()
    os.unlink(path)


def test_clear_session():
    """Clearing session should reset state."""
    memory, path = _setup()
    
    memory.record_turn(
        [{"role": "user", "content": "Hello"}],
        "Hi!",
        model="gpt-4",
        cost_usd=0.01,
    )
    
    memory.clear_session()
    
    assert memory.session_cost == 0.0
    assert memory.turn_count == 0
    assert len(memory.hot.recent_turns) == 0
    
    memory.close()
    os.unlink(path)


def test_stats():
    """Stats should be comprehensive."""
    memory, path = _setup()
    
    memory.record_turn(
        [{"role": "user", "content": "Test"}],
        "Response",
        model="gpt-4",
        cost_usd=0.01,
    )
    
    stats = memory.get_stats()
    
    assert "session" in stats
    assert "hot_memory" in stats
    assert "warm_memory" in stats
    assert stats["session"]["turns"] == 1
    assert abs(stats["session"]["cost_usd"] - 0.01) < 0.001
    
    memory.close()
    os.unlink(path)


def test_context_manager():
    """AgentMemory should work as context manager."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    config = MemoryConfig(db_path=path, embedding_provider="hash")
    
    with AgentMemory(config) as memory:
        memory.start_session("ctx-test")
        memory.record_turn(
            [{"role": "user", "content": "Hello"}],
            "Hi!",
            model="gpt-4",
        )
    
    os.unlink(path)


def test_volatile_normalization_in_cache():
    """System prompt volatile changes shouldn't affect cache."""
    memory, path = _setup()
    
    messages1 = [
        {"role": "system", "content": "Tokens used: 100. Be helpful."},
        {"role": "user", "content": "Hello"},
    ]
    messages2 = [
        {"role": "system", "content": "Tokens used: 500. Cache: 90%. Be helpful."},
        {"role": "user", "content": "Hello"},
    ]
    
    memory.record_turn(messages1, "Hi!", model="gpt-4")
    
    result = memory.check_cache(messages2)
    assert result.hit
    
    memory.close()
    os.unlink(path)
