"""Tests for Hot Memory (ring buffer)."""

import time

from agent_memory.hot import HotMemory, Turn, ToolCall, ToolResult, TaskContext
from agent_memory.types import TaskType


def test_add_turn():
    """Basic turn storage."""
    hot = HotMemory(max_turns=10)
    hot.add_user_message("Hello")
    
    assert len(hot.recent_turns) == 1
    assert hot.recent_turns[0].role == "user"
    assert hot.recent_turns[0].content == "Hello"


def test_ring_buffer_respects_max():
    """Ring buffer should evict oldest entries."""
    hot = HotMemory(max_turns=5)
    
    for i in range(10):
        hot.add_user_message(f"Message {i}")
    
    assert len(hot.recent_turns) == 5
    assert hot.recent_turns[0].content == "Message 5"
    assert hot.recent_turns[-1].content == "Message 9"


def test_get_recent_messages():
    """Should return messages as dicts."""
    hot = HotMemory(max_turns=10)
    hot.add_user_message("Hi")
    hot.add_assistant_message("Hello!", model="gpt-4", cost=0.01)
    
    msgs = hot.get_recent_messages()
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "Hi"}
    assert msgs[1] == {"role": "assistant", "content": "Hello!"}


def test_get_recent_messages_limit():
    """Limit should work."""
    hot = HotMemory(max_turns=10)
    for i in range(5):
        hot.add_user_message(f"Msg {i}")
    
    msgs = hot.get_recent_messages(limit=3)
    assert len(msgs) == 3
    assert msgs[0]["content"] == "Msg 2"


def test_tool_result_caching():
    """Tool results should be storable and retrievable."""
    hot = HotMemory(max_tool_results=10)
    
    result = ToolResult(
        tool_name="read_file",
        tool_call_id="tc_1",
        args={"path": "/etc/hosts"},
        result="127.0.0.1 localhost",
    )
    hot.recent_tool_results.append(result)
    
    # Retrieve by name and args
    found = hot.get_tool_result("read_file", {"path": "/etc/hosts"})
    assert found is not None
    assert found.result == "127.0.0.1 localhost"
    
    # Different args should miss
    not_found = hot.get_tool_result("read_file", {"path": "/etc/passwd"})
    assert not_found is None


def test_task_context():
    """Active task tracking."""
    hot = HotMemory()
    
    assert hot.active_task is None
    
    hot.set_task(TaskType.DEBUG, "Fixing login bug", focus="Auth middleware")
    
    assert hot.active_task is not None
    assert hot.active_task.task_type == TaskType.DEBUG
    assert hot.active_task.description == "Fixing login bug"
    
    hot.complete_step("Found error in auth middleware")
    hot.complete_step("Added null check")
    
    assert len(hot.active_task.steps_completed) == 2


def test_user_preferences():
    """Session preferences."""
    hot = HotMemory()
    
    hot.set_preference("style", "concise")
    hot.set_preference("language", "Python")
    
    assert hot.get_preference("style") == "concise"
    assert hot.get_preference("language") == "Python"
    assert hot.get_preference("missing", "default") == "default"


def test_session_cost_accumulation():
    """Cost should accumulate across turns."""
    hot = HotMemory()
    
    hot.add_assistant_message("Response 1", model="gpt-4", cost=0.01)
    hot.add_assistant_message("Response 2", model="gpt-4", cost=0.02)
    hot.add_assistant_message("Response 3", model="gpt-4", cost=0.005)
    
    assert abs(hot.session_cost - 0.035) < 0.001


def test_estimate_hot_tokens():
    """Token estimation should be reasonable."""
    hot = HotMemory(max_turns=100)
    
    hot.add_user_message("Hello world")  # ~12 chars / 4 = 3 tokens
    hot.add_assistant_message("This is a longer response with more content")
    
    tokens = hot.estimate_hot_tokens()
    assert tokens > 0
    assert tokens < 100  # Should be small for these short messages


def test_assemble_context():
    """Context assembly should include task + preferences + turns."""
    hot = HotMemory(max_tokens=4000)
    
    hot.set_task(TaskType.DEBUG, "Fixing the bug", focus="Line 42")
    hot.set_preference("style", "terse")
    hot.add_user_message("What's wrong?")
    hot.add_assistant_message("Checking...")
    
    context = hot.assemble_context()
    
    assert "Fixing the bug" in context
    assert "style: terse" in context
    assert "What's wrong?" in context
    assert "Checking..." in context


def test_clear():
    """Clear should reset everything."""
    hot = HotMemory()
    hot.add_user_message("Hello")
    hot.set_task(TaskType.DEBUG, "Bug")
    hot.set_preference("x", "y")
    hot.session_cost = 0.5
    
    hot.clear()
    
    assert len(hot.recent_turns) == 0
    assert hot.active_task is None
    assert len(hot.user_preferences) == 0
    assert hot.session_cost == 0.0


def test_stats():
    """Stats should be accurate."""
    hot = HotMemory()
    hot.add_user_message("Hello")
    hot.add_assistant_message("Hi!", cost=0.01)
    
    stats = hot.stats()
    assert stats["turns_stored"] == 2
    assert stats["session_cost_usd"] > 0
    assert stats["has_active_task"] is False
