"""
Agent Memory System — Main SDK interface.

Unified interface for the complete memory system:
- Hot memory (session ring buffer)
- Warm memory (SQLite + FAISS semantic storage)
- Response cache (exact + semantic match)
- Decision engine (importance, freshness, routing)
- Retrieval engine (multi-tier scoring)
- Cost tracking
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agent_memory.cache import ResponseCache
from agent_memory.decision import (
    DecisionEngine,
    FreshnessResult,
    ModelRoute,
    RememberDecision,
    score_importance,
)
from agent_memory.embedding import EmbeddingEngine, get_embedding_engine
from agent_memory.hot import HotMemory, TaskContext, ToolResult, Turn
from agent_memory.retrieval import RetrievalEngine, RetrievalContext, RetrievalResult
from agent_memory.store import SQLiteStore
from agent_memory.types import (
    CacheHitType,
    ContentType,
    CostEvent,
    MemoryEntry,
    MemoryStats,
    MemoryTier,
    SourceType,
    TaskType,
)
from agent_memory.normalizer import normalize_messages, DEFAULT_VOLATILE_PATTERNS

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for the memory system."""
    
    # Storage
    db_path: str = "~/.openclaw/agent-memory/cache.db"
    max_entries: int = 100_000
    max_size_mb: int = 500
    
    # Hot memory
    hot_max_turns: int = 50
    hot_max_tool_results: int = 20
    hot_max_tokens: int = 4000
    
    # Embedding
    embedding_provider: str = "local"  # local | hash | openai
    embedding_model: Optional[str] = None
    
    # Cache thresholds
    hit_threshold: float = 0.95
    gray_zone_low: float = 0.80
    
    # Triage
    triage_model: str = "gpt-4o-mini"
    
    # Budget
    budget_usd: float = 10.0
    cheap_models: list[str] = field(default_factory=lambda: ["gpt-4o-mini", "claude-haiku"])
    standard_models: list[str] = field(default_factory=lambda: ["gpt-4o", "claude-sonnet"])
    advanced_models: list[str] = field(default_factory=lambda: ["gpt-4", "claude-opus"])
    
    # Volatile patterns to strip from system prompts
    volatile_patterns: list[str] = field(default_factory=lambda: DEFAULT_VOLATILE_PATTERNS)
    
    # TTLs
    default_ttl: int = 86400  # 24 hours
    eviction_policy: str = "lru"  # lru | lfu | lowest_importance


@dataclass
class CacheCheckResult:
    """Result of checking the cache."""
    hit: bool
    hit_type: CacheHitType
    response: Optional[str] = None
    entry_id: Optional[str] = None
    similarity: float = 0.0
    freshness: Optional[FreshnessResult] = None
    model_route: Optional[ModelRoute] = None
    latency_ms: float = 0.0
    cost_saved: float = 0.0


@dataclass
class TurnResult:
    """Result of recording a completed turn."""
    stored: bool
    remember_decision: RememberDecision
    entry_id: Optional[str] = None
    model_route: Optional[ModelRoute] = None


class AgentMemory:
    """
    Agent Memory System — Main interface.
    
    Usage:
        memory = AgentMemory(MemoryConfig(db_path="~/.agent-memory.db"))
        
        # Check cache before LLM call
        result = memory.check_cache(messages, model="gpt-4")
        if result.hit:
            return result.response  # Skip LLM
        
        # Make LLM call...
        response = call_llm(messages)
        
        # Record the turn
        memory.record_turn(messages, response, model="gpt-4", cost=0.01)
        
        # Get retrieval context for next turn
        context = memory.retrieve(messages)
        augmented_messages = context.to_messages() + messages
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        
        # Initialize embedding engine
        self.embedding = get_embedding_engine(
            provider=self.config.embedding_provider,
            model=self.config.embedding_model,
        )
        
        # Initialize stores
        self.warm_store = SQLiteStore(
            db_path=self.config.db_path,
            embedding_engine=self.embedding,
            max_entries=self.config.max_entries,
        )
        
        # Initialize hot memory
        self.hot = HotMemory(
            max_turns=self.config.hot_max_turns,
            max_tool_results=self.config.hot_max_tool_results,
            max_tokens=self.config.hot_max_tokens,
        )
        
        # Initialize decision engine
        self.decision = DecisionEngine(
            budget_usd=self.config.budget_usd,
            cheap_models=self.config.cheap_models,
            standard_models=self.config.standard_models,
            advanced_models=self.config.advanced_models,
        )
        
        # Initialize cache
        self.cache = ResponseCache(
            db=self.warm_store,
            embedding_engine=self.embedding,
            hit_threshold=self.config.hit_threshold,
            gray_zone_low=self.config.gray_zone_low,
        )
        
        # Initialize retrieval engine
        self.retrieval = RetrievalEngine(
            hot_memory=self.hot,
            warm_store=self.warm_store,
            decision_engine=self.decision,
            embedding_engine=self.embedding,
        )
        
        # Session tracking
        self.session_id: str = ""
        self.session_cost: float = 0.0
        self.turn_count: int = 0
    
    def start_session(self, session_id: str = "") -> None:
        """Start a new session."""
        self.session_id = session_id or f"session-{int(time.time())}"
        self.hot.session_id = self.session_id
        self.cache.session_id = self.session_id
        self.session_cost = 0.0
        self.turn_count = 0
        logger.info(f"Session started: {self.session_id}")
    
    # ── Cache Operations ────────────────────────────────────────────
    
    def check_cache(
        self,
        messages: list[dict],
        model: str = "",
    ) -> CacheCheckResult:
        """
        Check if we have a cached response for these messages.
        
        Returns CacheCheckResult with hit/miss and metadata.
        """
        start = time.monotonic()
        
        # Layer 0-1: Response cache (exact + semantic)
        cache_result = self.cache.get(messages, model=model)
        
        if cache_result.hit:
            self.warm_store.record_cache_hit(cache_result.hit_type)
            elapsed = (time.monotonic() - start) * 1000
            
            return CacheCheckResult(
                hit=True,
                hit_type=cache_result.hit_type,
                response=cache_result.response,
                entry_id=cache_result.entry_id,
                similarity=cache_result.similarity,
                latency_ms=elapsed,
                cost_saved=cache_result.cost_saved,
            )
        
        # Layer 2: Check retrieval engine for cached answer with freshness
        entry, freshness = self.retrieval.retrieve_for_cache_check(messages)
        
        if entry and freshness.is_fresh:
            self.warm_store.record_cache_hit(CacheHitType.CLIENT_SEMANTIC)
            elapsed = (time.monotonic() - start) * 1000
            
            return CacheCheckResult(
                hit=True,
                hit_type=CacheHitType.CLIENT_SEMANTIC,
                response=entry.text,
                entry_id=entry.id,
                similarity=0.9,  # Approximate
                freshness=freshness,
                latency_ms=elapsed,
                cost_saved=entry.cost_usd,
            )
        
        # Cache miss
        self.warm_store.record_cache_hit(CacheHitType.NONE)
        elapsed = (time.monotonic() - start) * 1000
        
        return CacheCheckResult(
            hit=False,
            hit_type=CacheHitType.NONE,
            freshness=freshness if entry else None,
            latency_ms=elapsed,
        )
    
    # ── Turn Recording ──────────────────────────────────────────────
    
    def record_turn(
        self,
        messages: list[dict],
        response: str,
        model: str = "",
        cost_usd: float = 0.0,
        content_type: ContentType = ContentType.CONVERSATION,
        task_type: Optional[TaskType] = None,
        cache_hit: bool = False,
    ) -> TurnResult:
        """
        Record a completed turn (LLM response).
        
        Decides whether to store, where to store, and updates hot memory.
        """
        self.turn_count += 1
        self.session_cost += cost_usd
        
        # Decide if we should remember
        remember = self.decision.should_remember(
            text=response,
            content_type=content_type,
            is_new_info=True,  # Simplified — production would check similarity
        )
        
        entry_id = None
        if remember.remember:
            # Store in warm memory
            entry_id = self.cache.store(
                messages=messages,
                response=response,
                model=model,
                session_id=self.session_id,
                cost_usd=cost_usd,
                content_type=content_type,
                task_type=task_type,
            )
        
        # Always update hot memory
        self.hot.add_user_message(
            next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        )
        self.hot.add_assistant_message(
            content=response,
            model=model,
            cost=cost_usd,
            cache_hit=cache_hit,
        )
        
        # Record cost event
        self.warm_store.record_cost(CostEvent(
            timestamp=int(time.time()),
            session_id=self.session_id,
            event_type="llm_call",
            model=model,
            input_tokens=len(str(messages)) // 4,  # Rough estimate
            output_tokens=len(response) // 4,
            cost_usd=cost_usd,
            cache_hit_type=CacheHitType.CLIENT_EXACT if cache_hit else CacheHitType.NONE,
            task_type=task_type.value if task_type else None,
        ))
        
        # Route for next turn
        route = self.decision.route_to_model(
            session_cost=self.session_cost,
        )
        
        return TurnResult(
            stored=remember.remember,
            remember_decision=remember,
            entry_id=entry_id,
            model_route=route,
        )
    
    # ── Tool Operations ─────────────────────────────────────────────
    
    def check_tool_cache(self, tool_name: str, args: dict) -> Optional[str]:
        """Check if we have a cached result for this tool call."""
        # Check hot memory first
        hot_result = self.hot.get_tool_result(tool_name, args)
        if hot_result:
            return hot_result.result
        
        # Check warm store
        import hashlib
        import json
        args_str = json.dumps(args, sort_keys=True)
        cache_key = hashlib.sha256(f"{tool_name}:{args_str}".encode()).hexdigest()
        
        return self.warm_store.get_tool_result(cache_key)
    
    def record_tool_result(
        self,
        tool_name: str,
        args: dict,
        result: str,
        ttl: int = 3600,
    ) -> None:
        """Record a tool result in cache."""
        tool_result = ToolResult(
            tool_name=tool_name,
            tool_call_id=f"tc-{int(time.time())}",
            args=args,
            result=result,
        )
        
        # Add to hot memory
        self.hot.recent_tool_results.append(tool_result)
        
        # Store in warm cache
        import hashlib
        import json
        args_str = json.dumps(args, sort_keys=True)
        cache_key = hashlib.sha256(f"{tool_name}:{args_str}".encode()).hexdigest()
        args_hash = hashlib.sha256(args_str.encode()).hexdigest()
        
        self.warm_store.store_tool_result(
            cache_key=cache_key,
            tool_name=tool_name,
            tool_args_hash=args_hash,
            result=result,
            session_id=self.session_id,
            ttl=ttl,
        )
    
    # ── Retrieval ───────────────────────────────────────────────────
    
    def retrieve(
        self,
        messages: list[dict],
        task_type: TaskType = TaskType.UNKNOWN,
        budget_tokens: int = 4000,
    ) -> RetrievalResult:
        """Retrieve relevant memory context for the current messages."""
        context = RetrievalContext(
            session_id=self.session_id,
            current_messages=messages,
            task_type=task_type,
        )
        
        return self.retrieval.retrieve(context, budget_tokens=budget_tokens)
    
    def get_augmented_messages(
        self,
        messages: list[dict],
        task_type: TaskType = TaskType.UNKNOWN,
        budget_tokens: int = 4000,
    ) -> list[dict]:
        """
        Get messages augmented with relevant memory context.
        
        Prepends memory context as system messages before the conversation.
        """
        retrieval_result = self.retrieve(messages, task_type, budget_tokens)
        
        if not retrieval_result.entries:
            return messages
        
        # Build memory context messages
        memory_messages = [
            {
                "role": "system",
                "content": f"[Memory Context]\n{retrieval_result.to_context_string()}",
            }
        ]
        
        return memory_messages + messages
    
    # ── Model Routing ───────────────────────────────────────────────
    
    def route_to_model(
        self,
        task_description: str = "",
        task_type: TaskType = TaskType.UNKNOWN,
        cached_messages: Optional[list[dict]] = None,
    ) -> ModelRoute:
        """Determine which model to use for this task."""
        cached_entry = None
        freshness = None
        
        if cached_messages:
            cached_entry, freshness = self.retrieval.retrieve_for_cache_check(cached_messages)
        
        return self.decision.route_to_model(
            task_description=task_description,
            task_type=task_type,
            cached_entry=cached_entry,
            cached_freshness=freshness,
            session_cost=self.session_cost,
        )
    
    # ── Task Management ─────────────────────────────────────────────
    
    def set_task(self, task_type: TaskType, description: str, focus: str = "") -> None:
        """Set the current active task."""
        self.hot.set_task(task_type, description, focus)
    
    def complete_step(self, step: str) -> None:
        """Mark a task step as completed."""
        self.hot.complete_step(step)
    
    # ── Session Management ──────────────────────────────────────────
    
    def set_preference(self, key: str, value: str) -> None:
        """Set a session preference."""
        self.hot.set_preference(key, value)
    
    def get_preference(self, key: str, default: str = "") -> str:
        """Get a session preference."""
        return self.hot.get_preference(key, default)
    
    # ── Maintenance ─────────────────────────────────────────────────
    
    def compact(self) -> dict:
        """
        Run memory compaction.
        
        Suggests and applies compaction actions based on decision engine.
        """
        # Get all warm entries
        conn = self.warm_store._get_conn()
        rows = conn.execute("SELECT id FROM memory_entries ORDER BY last_accessed_at").fetchall()
        entry_ids = [r["id"] for r in rows]
        
        # Get full entries for analysis (sample for efficiency)
        entries = []
        for eid in entry_ids[:1000]:  # Sample first 1000
            entry = self.warm_store.get_by_id(eid)
            if entry:
                entries.append(entry)
        
        # Get compaction suggestions
        actions = self.decision.suggest_compaction(entries)
        
        # Apply drop actions
        dropped = 0
        for action in actions:
            if action.action == "drop":
                for eid in action.entry_ids:
                    conn.execute("DELETE FROM memory_entries WHERE id = ?", (eid,))
                    dropped += 1
        
        conn.commit()
        
        # Cleanup expired tool cache
        tool_cleaned = self.warm_store.cleanup_expired_tool_cache()
        
        return {
            "entries_dropped": dropped,
            "tool_cache_cleaned": tool_cleaned,
            "actions_suggested": len(actions),
        }
    
    def get_stats(self) -> dict:
        """Get comprehensive memory system statistics."""
        warm_stats = self.warm_store.get_stats()
        hot_stats = self.hot.stats()
        session_cost = self.warm_store.get_session_cost(self.session_id)
        
        return {
            "session": {
                "id": self.session_id,
                "turns": self.turn_count,
                "cost_usd": round(self.session_cost, 4),
            },
            "hot_memory": hot_stats,
            "warm_memory": warm_stats.to_dict(),
            "session_cost": session_cost,
        }
    
    # ── Lifecycle ───────────────────────────────────────────────────
    
    def clear_session(self) -> None:
        """Clear hot memory and session state."""
        self.hot.clear()
        self.session_cost = 0.0
        self.turn_count = 0
    
    def close(self) -> None:
        """Close the memory system and release resources."""
        self.warm_store.close()
        logger.info("Memory system closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
