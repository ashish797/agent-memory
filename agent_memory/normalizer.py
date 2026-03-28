"""
Prompt normalization for consistent cache keys.

Strips volatile sections from system prompts so that usage stats,
timestamps, and counters don't invalidate the cache every turn.
"""

from __future__ import annotations

import json
import re
from typing import Optional


# Default patterns to strip before hashing/embedding.
# These change every turn and would break cache if included.
DEFAULT_VOLATILE_PATTERNS: list[str] = [
    r'[Tt]okens?\s*used:?\s*\d+',
    r'[Cc]ache\s*hit\s*rate:?\s*[\d.]+%',
    r'[Cc]ache\s*rate:?\s*[\d.]+%',
    r'[Cc]ache(?:\s*hit)?(?:\s*[Rr]ate)?:?\s*[\d.]+%',
    r'[Cc]urrent time:?\s*[^\n]+',
    r'[Cc]ontext\s*length:?\s*\d+/\d+',
    r'usageLine:?\s*[^\n]+',
    r'cacheLine:?\s*[^\n]+',
    r'contextLine:?\s*[^\n]+',
    r'[Tt]imestamp:?\s*\d+',
    r'started_at:?\s*\d+',
    r'request_?id:?\s*[\w-]+',
    r'session_?counter:?\s*\d+',
    r'turn_?number:?\s*\d+',
    # Any line starting with a timestamp-like pattern
    r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}.*$',
]


def normalize_messages(
    messages: list[dict],
    volatile_patterns: Optional[list[str]] = None,
    strip_whitespace: bool = True,
) -> str:
    """
    Normalize a list of messages into a consistent string for hashing/embedding.
    
    Strips volatile patterns, sorts keys for determinism, and optionally
    strips extra whitespace.
    
    Args:
        messages: List of {"role": str, "content": str, ...} dicts.
        volatile_patterns: Custom patterns to strip. Defaults to DEFAULT_VOLATILE_PATTERNS.
        strip_whitespace: Collapse multiple spaces/newlines.
    
    Returns:
        Normalized JSON string.
    """
    patterns = volatile_patterns if volatile_patterns is not None else DEFAULT_VOLATILE_PATTERNS
    
    filtered = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        
        role = msg.get("role", "")
        content = str(msg.get("content", ""))
        
        if role == "system":
            content = _strip_volatile(content, patterns)
        
        if strip_whitespace:
            content = _collapse_whitespace(content)
        
        filtered.append({
            "role": role,
            "content": content,
        })
    
    # Sort keys for deterministic JSON
    return json.dumps(filtered, sort_keys=True, ensure_ascii=False)


def normalize_text(text: str, volatile_patterns: Optional[list[str]] = None) -> str:
    """
    Normalize a plain text string (for single-prompt cache keys).
    """
    patterns = volatile_patterns if volatile_patterns is not None else DEFAULT_VOLATILE_PATTERNS
    text = _strip_volatile(text, patterns)
    text = _collapse_whitespace(text)
    return text.strip()


def _strip_volatile(text: str, patterns: list[str]) -> str:
    """Remove volatile patterns from text, replacing with single [VOLATILE]."""
    result = text
    for pattern in patterns:
        result = re.sub(pattern, ' __VOLATILE__ ', result, flags=re.MULTILINE)
    
    # Remove any punctuation that's now orphaned next to __VOLATILE__
    result = re.sub(r'\.\s*__VOLATILE__', ' __VOLATILE__', result)
    result = re.sub(r'__VOLATILE__\s*\.', ' __VOLATILE__', result)
    
    # Collapse consecutive volatile markers into one
    result = re.sub(r'(\s*__VOLATILE__\s*)+', ' [VOLATILE] ', result)
    
    # Normalize whitespace
    result = re.sub(r'\s+', ' ', result)
    
    return result.strip()


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace into single space."""
    # Preserve newlines but collapse runs
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text
