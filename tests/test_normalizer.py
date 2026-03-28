"""Tests for prompt normalization."""

import hashlib
import json

from agent_memory.normalizer import (
    normalize_messages,
    normalize_text,
    _strip_volatile,
    _collapse_whitespace,
    DEFAULT_VOLATILE_PATTERNS,
)


def test_exact_same_messages_hash_to_same():
    """Same messages should produce the same normalized output."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
    ]
    norm1 = normalize_messages(messages)
    norm2 = normalize_messages(messages)
    assert norm1 == norm2
    assert hashlib.sha256(norm1.encode()).hexdigest() == hashlib.sha256(norm2.encode()).hexdigest()


def test_volatile_patterns_stripped():
    """Volatile patterns should be replaced with [VOLATILE]."""
    messages = [
        {"role": "system", "content": "Tokens used: 1234. Cache hit rate: 87%. Current time: 2026-03-28T10:00:00"},
        {"role": "user", "content": "Hello"},
    ]
    norm = normalize_messages(messages)
    
    # These should NOT appear in normalized output
    assert "1234" not in norm  # tokens_used value stripped
    assert "87%" not in norm   # cache_hit_rate value stripped
    assert "2026-03-28" not in norm  # timestamp stripped
    assert "[VOLATILE]" in norm


def test_user_messages_not_stripped():
    """User messages should NOT have volatile patterns stripped."""
    messages = [
        {"role": "user", "content": "Tokens used: 1234. Cache hit rate: 87%."},
    ]
    norm = normalize_messages(messages)
    
    # User content should be preserved exactly
    data = json.loads(norm)
    assert data[0]["content"] == "Tokens used: 1234. Cache hit rate: 87%."


def test_whitespace_collapsed():
    """Multiple spaces/newlines should be collapsed."""
    messages = [
        {"role": "user", "content": "Hello     world\n\n\n\nTest"},
    ]
    norm = normalize_messages(messages, strip_whitespace=True)
    data = json.loads(norm)
    assert data[0]["content"] == "Hello world\n\nTest"


def test_order_independent():
    """Ordering of keys shouldn't matter."""
    messages1 = [{"content": "Hello", "role": "user"}]
    messages2 = [{"role": "user", "content": "Hello"}]
    
    norm1 = normalize_messages(messages1)
    norm2 = normalize_messages(messages2)
    assert norm1 == norm2


def test_normalize_text():
    """Plain text normalization."""
    text = "Tokens used: 1234.   Multiple   spaces.\n\n\n\nToo many newlines."
    norm = normalize_text(text)
    assert "1234" not in norm
    assert "[VOLATILE]" in norm
    assert "Multiple spaces." in norm
    assert "Too many newlines." in norm
    # Newlines are collapsed by the whitespace normalization
    assert "\n\n" not in norm


def test_custom_volatile_patterns():
    """Custom patterns should work."""
    messages = [
        {"role": "system", "content": "Request ID: abc-123. Session: xyz-789."},
    ]
    custom_patterns = [r'Request ID: \w+', r'Session: \w+']
    norm = normalize_messages(messages, volatile_patterns=custom_patterns)
    
    assert "abc-123" not in norm
    assert "xyz-789" not in norm
    assert "[VOLATILE]" in norm


def test_empty_messages():
    """Empty messages list should return valid JSON."""
    norm = normalize_messages([])
    assert norm == "[]"


def test_non_dict_messages_skipped():
    """Non-dict messages should be skipped gracefully."""
    messages = [
        "not a dict",
        {"role": "user", "content": "Hello"},
        42,
        {"role": "assistant", "content": "Hi there!"},
    ]
    norm = normalize_messages(messages)
    data = json.loads(norm)
    assert len(data) == 2
    assert data[0]["role"] == "user"
    assert data[1]["role"] == "assistant"
