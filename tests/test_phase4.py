"""Tests for Phase 4: Proxy, Sync, Adapters."""

import os
import tempfile

from agent_memory.adapters import CachedLLM, AgentMemoryTools
from agent_memory.memory import AgentMemory, MemoryConfig
from agent_memory.sync import CloudSync, SyncConfig, SyncStore, SyncEntry
from agent_memory.types import ContentType, MemoryEntry, MemoryTier, SourceType


# ── CachedLLM Tests ─────────────────────────────────────────────────

def test_cached_llm_basic():
    """CachedLLM should cache responses."""
    call_count = 0
    
    def mock_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Response to: {prompt}"
    
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    cached = CachedLLM(mock_llm, model="test-model", db_path=path)
    
    # First call
    r1 = cached("Hello")
    assert call_count == 1
    assert "Hello" in r1
    
    # Same prompt — cache hit
    r2 = cached("Hello")
    assert call_count == 1  # No additional call
    assert r2 == r1
    
    # Different prompt — cache miss
    r3 = cached("World")
    assert call_count == 2
    
    cached.close()
    os.unlink(path)


def test_cached_llm_different_prompts():
    """Different prompts should not share cache."""
    def mock_llm(prompt: str) -> str:
        return f"Answer: {prompt}"
    
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    cached = CachedLLM(mock_llm, model="test", db_path=path)
    
    r1 = cached("What is Python?")
    r2 = cached("What is Rust?")
    
    assert r1 != r2
    
    cached.close()
    os.unlink(path)


# ── AgentMemoryTools Tests ──────────────────────────────────────────

def test_agent_memory_tools():
    """AgentMemoryTools should provide callable tools."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    config = MemoryConfig(db_path=path, embedding_provider="hash")
    memory = AgentMemory(config)
    memory.start_session("test")
    
    # Store something
    memory.record_turn(
        [{"role": "user", "content": "I prefer Python"}],
        "Got it, I'll use Python.",
        model="gpt-4",
    )
    
    tools = AgentMemoryTools(memory)
    
    # Search
    result = tools.search_memory("Python")
    assert isinstance(result, str)
    
    # Save
    result = tools.save_memory("Remember to use type hints", "preference")
    assert "Saved" in result
    
    # Stats
    stats = tools.get_stats()
    assert "Session:" in stats
    assert "Cache:" in stats
    
    # Tool list format
    tool_list = tools.as_tool_list()
    assert len(tool_list) >= 3
    assert tool_list[0]["function"]["name"] == "search_memory"
    
    memory.close()
    os.unlink(path)


# ── SyncStore Tests ─────────────────────────────────────────────────

def test_sync_store():
    """SyncStore should track sync state."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    store = SyncStore(path)
    
    # Initially no sync time
    assert store.get_last_sync_time() == 0
    
    # Set sync time
    store.set_last_sync_time(1234567890)
    assert store.get_last_sync_time() == 1234567890
    
    # Log sync events
    store.log_sync("push", 10, "success", duration_ms=150)
    store.log_sync("pull", 5, "success", duration_ms=200)
    
    history = store.get_sync_history()
    assert len(history) == 2
    assert history[0]["direction"] == "pull"  # Most recent first
    
    store.close()
    os.unlink(path)


# ── SyncEntry Tests ─────────────────────────────────────────────────

def test_sync_entry_serialization():
    """SyncEntry should serialize and deserialize."""
    entry = MemoryEntry(
        text="Test content",
        tier=MemoryTier.WARM,
        content_type=ContentType.INSIGHT,
        session_id="s1",
        source_type=SourceType.MODEL,
    )
    
    sync_entry = SyncEntry(
        entry=entry,
        device_id="device-1",
        version=2,
    )
    
    # Serialize
    data = sync_entry.to_dict()
    assert data["device_id"] == "device-1"
    assert data["version"] == 2
    assert "entry" in data
    
    # Deserialize
    restored = SyncEntry.from_dict(data)
    assert restored.device_id == "device-1"
    assert restored.version == 2
    assert restored.entry.text == "Test content"
    assert restored.entry.text_hash == entry.text_hash


def test_sync_entry_hash():
    """Same content should produce same sync hash."""
    entry = MemoryEntry(
        text="Same content",
        tier=MemoryTier.WARM,
        content_type=ContentType.CONVERSATION,
        session_id="s1",
        source_type=SourceType.USER,
    )
    
    se1 = SyncEntry(entry=entry, device_id="d1")
    se2 = SyncEntry(entry=entry, device_id="d2")
    
    assert se1.sync_hash == se2.sync_hash


# ── CloudSync Tests ─────────────────────────────────────────────────

def test_cloud_sync_disabled():
    """Disabled sync should return early."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    config = SyncConfig(enabled=False)
    sync = CloudSync(config, SyncStore(path))
    
    result = sync.push()
    assert result["status"] == "disabled"
    
    result = sync.pull()
    assert result["status"] == "disabled"
    
    status = sync.get_status()
    assert not status["enabled"]
    
    os.unlink(path)


def test_cloud_sync_status():
    """Sync status should reflect configuration."""
    config = SyncConfig(
        enabled=True,
        provider="http",
        device_id="test-device",
        sync_interval=600,
    )
    
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    sync = CloudSync(config, SyncStore(path))
    
    status = sync.get_status()
    assert status["enabled"]
    assert status["provider"] == "http"
    assert status["device_id"] == "test-device"
    assert status["config"]["sync_interval"] == 600
    
    os.unlink(path)


# ── Proxy Tests ─────────────────────────────────────────────────────

def test_proxy_estimate_cost():
    """Cost estimation should work for known models."""
    from agent_memory.proxy import _estimate_cost
    
    # GPT-4: $30/1M input, $60/1M output
    cost = _estimate_cost("gpt-4", 1000, 500)
    assert cost > 0
    assert cost == (1000/1_000_000)*30 + (500/1_000_000)*60
    
    # GPT-4o-mini: $0.15/1M input, $0.60/1M output
    cost = _estimate_cost("gpt-4o-mini", 1000, 500)
    assert cost < 0.1  # Very cheap (~$0.00045)


def test_proxy_extract_messages():
    """Message extraction should work."""
    from agent_memory.proxy import _extract_messages, _extract_model
    
    body = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hello"},
        ],
    }
    
    assert len(_extract_messages(body)) == 1
    assert _extract_model(body) == "gpt-4"


def test_proxy_config():
    """ProxyConfig should have sensible defaults."""
    from agent_memory.proxy import ProxyConfig
    
    config = ProxyConfig()
    assert config.port == 8080
    assert config.upstream == "https://api.openai.com/v1"
    assert config.embedding_provider == "hash"
