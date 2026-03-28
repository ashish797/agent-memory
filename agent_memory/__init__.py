"""
Agent Memory System — Memory hierarchy with cost-aware routing.

Tiers:
  - Hot:  In-memory session state (ring buffer)
  - Warm: SQLite + FAISS semantic storage
  - Cold: Compressed file archival (future)

Phase 1: Hot + Warm tiers with exact + semantic match.
"""

from agent_memory.store import SQLiteStore
from agent_memory.hot import HotMemory, Turn, ToolResult
from agent_memory.cache import ResponseCache
from agent_memory.embedding import EmbeddingEngine, LocalEmbedding
from agent_memory.normalizer import normalize_messages, DEFAULT_VOLATILE_PATTERNS
from agent_memory.decision import DecisionEngine, score_importance, check_freshness, route_to_model
from agent_memory.retrieval import RetrievalEngine, RetrievalContext, RetrievalResult
from agent_memory.memory import AgentMemory, MemoryConfig
from agent_memory.types import (
    MemoryEntry, MemoryTier, ContentType, SourceType, TaskType,
    EvictionPolicy, CostEvent, CacheHitType,
)

__version__ = "0.1.0-alpha"

__all__ = [
    "SQLiteStore",
    "HotMemory", "Turn", "ToolResult",
    "ResponseCache",
    "EmbeddingEngine", "LocalEmbedding",
    "normalize_messages",
    "DecisionEngine", "score_importance", "check_freshness", "route_to_model",
    "RetrievalEngine", "RetrievalContext", "RetrievalResult",
    "MemoryEntry", "MemoryTier", "ContentType", "SourceType", "TaskType",
    "EvictionPolicy", "CostEvent", "CacheHitType",
]
