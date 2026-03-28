"""Core types for the Agent Memory System."""

from __future__ import annotations

import enum
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Enums ────────────────────────────────────────────────────────────

class MemoryTier(str, enum.Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class ContentType(str, enum.Enum):
    CONVERSATION = "conversation"
    TOOL_RESULT = "tool_result"
    DECISION = "decision"
    INSIGHT = "insight"
    LOG = "log"
    PREFERENCE = "preference"


class SourceType(str, enum.Enum):
    USER = "user"
    MODEL = "model"
    TOOL = "tool"
    SYSTEM = "system"
    INFERRED = "inferred"


class TaskType(str, enum.Enum):
    DEBUG = "debug"
    WRITE = "write"
    SEARCH = "search"
    DESIGN = "design"
    CODE = "code"
    RESEARCH = "research"
    TRIAGE = "triage"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class EvictionPolicy(str, enum.Enum):
    LRU = "lru"
    LFU = "lfu"
    LOWEST_IMPORTANCE = "lowest_importance"
    EXPIRED = "expired"
    SIZE_BASED = "size_based"


class CacheHitType(str, enum.Enum):
    NONE = "none"
    SERVER = "server"
    CLIENT_EXACT = "client_exact"
    CLIENT_SEMANTIC = "client_semantic"
    CLIENT_TRIAGE = "client_triage"


# ── Data Classes ─────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single memory entry stored in any tier."""
    
    text: str
    tier: MemoryTier
    content_type: ContentType
    session_id: str
    source_type: SourceType = SourceType.SYSTEM
    text_hash_override: Optional[str] = None  # Set this to override the computed hash
    
    id: str = field(default_factory=lambda: _uuid())
    embedding: Optional[list[float]] = None
    source_detail: Optional[str] = None  # model name, tool name, etc.
    task_type: Optional[TaskType] = None
    model_used: Optional[str] = None
    cost_usd: float = 0.0
    
    created_at: int = field(default_factory=lambda: int(time.time()))
    last_accessed_at: int = field(default_factory=lambda: int(time.time()))
    access_count: int = 0
    expires_at: Optional[int] = None
    importance: float = 0.5
    
    parent_id: Optional[str] = None
    tool_call_id: Optional[str] = None
    
    @property
    def text_hash(self) -> str:
        if self.text_hash_override:
            return self.text_hash_override
        return hashlib.sha256(self.text.encode()).hexdigest()
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return int(time.time()) > self.expires_at
    
    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed_at = int(time.time())
        self.access_count += 1
    
    def estimate_tokens(self) -> int:
        """Rough token estimate (1 token ≈ 4 chars)."""
        return len(self.text) // 4
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tier": self.tier.value,
            "content_type": self.content_type.value,
            "text": self.text,
            "source_type": self.source_type.value,
            "source_detail": self.source_detail,
            "session_id": self.session_id,
            "task_type": self.task_type.value if self.task_type else None,
            "model_used": self.model_used,
            "cost_usd": self.cost_usd,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "expires_at": self.expires_at,
            "importance": self.importance,
            "parent_id": self.parent_id,
            "tool_call_id": self.tool_call_id,
            "text_hash": self.text_hash,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> MemoryEntry:
        return cls(
            id=data["id"],
            tier=MemoryTier(data["tier"]),
            content_type=ContentType(data["content_type"]),
            text=data["text"],
            session_id=data["session_id"],
            source_type=SourceType(data["source_type"]),
            source_detail=data.get("source_detail"),
            task_type=TaskType(data["task_type"]) if data.get("task_type") else None,
            model_used=data.get("model_used"),
            cost_usd=data.get("cost_usd", 0.0),
            created_at=data["created_at"],
            last_accessed_at=data.get("last_accessed_at", data["created_at"]),
            access_count=data.get("access_count", 0),
            expires_at=data.get("expires_at"),
            importance=data.get("importance", 0.5),
            parent_id=data.get("parent_id"),
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class ScoredEntry:
    """A memory entry with relevance scoring."""
    
    entry: MemoryEntry
    relevance_score: float
    source_tier: MemoryTier
    retrieval_reason: str = ""


@dataclass
class CostEvent:
    """A single cost event for tracking."""
    
    timestamp: int
    session_id: str
    event_type: str  # 'llm_call' | 'embedding' | 'triage'
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_hit_type: CacheHitType = CacheHitType.NONE
    task_type: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "event_type": self.event_type,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "cache_hit_type": self.cache_hit_type.value,
            "task_type": self.task_type,
        }


@dataclass
class ToolCacheEntry:
    """A cached tool result."""
    
    cache_key: str
    tool_name: str
    tool_args_hash: str
    result: str
    result_hash: str
    session_id: Optional[str] = None
    created_at: int = field(default_factory=lambda: int(time.time()))
    expires_at: int = 0
    hit_count: int = 0
    last_hit_at: Optional[int] = None


@dataclass
class MemoryStats:
    """Aggregate memory system statistics."""
    
    total_entries: int = 0
    total_size_bytes: int = 0
    hot_entries: int = 0
    warm_entries: int = 0
    cold_entries: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    triage_hits: int = 0
    misses: int = 0
    total_cost_saved: float = 0.0
    total_requests: int = 0
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.exact_hits + self.semantic_hits + self.triage_hits) / self.total_requests
    
    def to_dict(self) -> dict:
        return {
            "total_entries": self.total_entries,
            "total_size_bytes": self.total_size_bytes,
            "hot_entries": self.hot_entries,
            "warm_entries": self.warm_entries,
            "cold_entries": self.cold_entries,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "triage_hits": self.triage_hits,
            "misses": self.misses,
            "total_cost_saved": self.total_cost_saved,
            "total_requests": self.total_requests,
            "hit_rate": f"{self.hit_rate:.1%}",
        }


# ── Helpers ──────────────────────────────────────────────────────────

def _uuid() -> str:
    """Generate a simple UUID without uuid dependency."""
    import random
    import struct
    t = struct.pack('>Q', int(time.time() * 1000))
    r = struct.pack('>Q', random.getrandbits(64))
    hex_str = (t + r).hex()
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"
