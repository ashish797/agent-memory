"""
Retrieval Engine — Multi-tier memory retrieval with scoring.

Merges results from Hot (session) and Warm (semantic) tiers into a
unified, ranked context within a token budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from agent_memory.decision import DecisionEngine, FreshnessResult, score_importance
from agent_memory.hot import HotMemory
from agent_memory.store import SQLiteStore
from agent_memory.types import (
    ContentType,
    MemoryEntry,
    MemoryTier,
    SourceType,
    TaskType,
)


@dataclass
class ScoredEntry:
    """A memory entry with relevance scoring."""
    entry: MemoryEntry
    relevance_score: float
    source_tier: MemoryTier
    retrieval_reason: str = ""


@dataclass
class RetrievalContext:
    """Context for a retrieval request."""
    session_id: str
    current_messages: list[dict]
    task_type: TaskType = TaskType.UNKNOWN
    task_description: str = ""
    intent: Optional[str] = None  # "debugging" | "writing" | etc.


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    entries: list[ScoredEntry]
    total_tokens: int
    sources: dict[str, int]  # tier name → count
    retrieval_time_ms: float
    
    def to_context_string(self) -> str:
        """Format entries as a context string for prompt injection."""
        parts = []
        for scored in self.entries:
            entry = scored.entry
            tag = f"[{entry.content_type.value} | {entry.source_type.value}]"
            if scored.retrieval_reason:
                tag += f" ({scored.retrieval_reason})"
            parts.append(f"{tag}\n{entry.text}")
        return "\n\n".join(parts)
    
    def to_messages(self) -> list[dict]:
        """Format entries as message dicts (for conversation context)."""
        return [
            {
                "role": "system",
                "content": f"[Memory Context - {entry.content_type.value}]\n{entry.text}",
            }
            for entry in [s.entry for s in self.entries]
        ]


# ── Scoring Functions ────────────────────────────────────────────────


def compute_relevance_score(
    entry: MemoryEntry,
    context: RetrievalContext,
    embedding_similarity: float = 0.0,
) -> float:
    """
    Multi-factor relevance scoring for memory retrieval.
    
    Weights:
    - Semantic similarity: 40% (if available)
    - Recency: 20%
    - Access frequency: 15%
    - Task match: 15%
    - Importance: 10%
    """
    now = time.time()
    
    # Factor 1: Semantic similarity (40%)
    # If no embedding similarity, redistribute to recency
    if embedding_similarity > 0:
        sim_score = embedding_similarity
        sim_weight = 0.40
    else:
        sim_score = 0.0
        sim_weight = 0.0
    
    # Factor 2: Recency (20%)
    age_hours = (now - entry.created_at) / 3600
    recency_score = _exponential_decay(age_hours, half_life=168)  # 1 week half-life
    recency_weight = 0.20 + (sim_weight - 0.40) * -1 if embedding_similarity == 0 else 0.20
    
    # Factor 3: Access frequency (15%)
    freq_score = _log_scale(entry.access_count, max_val=100)
    freq_weight = 0.15
    
    # Factor 4: Task match (15%)
    task_score = 1.0 if entry.task_type == context.task_type else 0.3
    task_weight = 0.15
    
    # Factor 5: Importance (10%)
    importance_score = entry.importance
    importance_weight = 0.10
    
    total = (
        sim_weight * sim_score +
        recency_weight * recency_score +
        freq_weight * freq_score +
        task_weight * task_score +
        importance_weight * importance_score
    )
    
    return min(1.0, total)


def _exponential_decay(age_hours: float, half_life: float) -> float:
    """Exponential decay based on half-life."""
    if half_life <= 0:
        return 0.0
    return 2 ** (-age_hours / half_life)


def _log_scale(value: int, max_val: int) -> float:
    """Logarithmic scaling to 0-1 range."""
    if value <= 0:
        return 0.0
    import math
    return min(1.0, math.log(1 + value) / math.log(1 + max_val))


# ── Retrieval Engine ─────────────────────────────────────────────────


class RetrievalEngine:
    """
    Multi-tier retrieval engine.
    
    Retrieves from Hot (session) and Warm (semantic) tiers,
    scores and merges results, returns top-K within token budget.
    """
    
    def __init__(
        self,
        hot_memory: HotMemory,
        warm_store: SQLiteStore,
        decision_engine: Optional[DecisionEngine] = None,
        embedding_engine=None,
    ):
        self.hot = hot_memory
        self.warm = warm_store
        self.decision = decision_engine or DecisionEngine()
        self.embedding = embedding_engine
    
    def retrieve(
        self,
        context: RetrievalContext,
        budget_tokens: int = 4000,
        top_k: int = 20,
        min_relevance: float = 0.1,
    ) -> RetrievalResult:
        """
        Retrieve relevant memories for the given context.
        
        Sources:
        1. Hot memory: recent turns (always included, up to budget)
        2. Warm memory: semantic search on FAISS index
        3. Tool results: recent tool outputs from hot memory
        
        Returns scored, deduplicated, budget-capped results.
        """
        start = time.monotonic()
        scored_entries: list[ScoredEntry] = []
        
        # ── Hot Memory Retrieval ────────────────────────────────────
        # Recent turns are always relevant for current context
        recent_turns = list(self.hot.recent_turns)
        for turn in recent_turns[-10:]:  # Last 10 turns
            entry = MemoryEntry(
                text=f"[{turn.role}] {turn.content}",
                tier=MemoryTier.HOT,
                content_type=ContentType.CONVERSATION,
                session_id=self.hot.session_id or context.session_id,
                source_type=SourceType.USER if turn.role == "user" else SourceType.MODEL,
                source_detail=turn.model_used,
                cost_usd=turn.cost_usd,
                created_at=turn.timestamp,
                last_accessed_at=turn.timestamp,
                access_count=1,
            )
            
            # Hot memory gets a recency boost
            relevance = compute_relevance_score(
                entry, context, embedding_similarity=0.0
            )
            relevance = min(1.0, relevance + 0.2)  # Recency boost for hot memory
            
            if relevance >= min_relevance:
                scored_entries.append(ScoredEntry(
                    entry=entry,
                    relevance_score=relevance,
                    source_tier=MemoryTier.HOT,
                    retrieval_reason="recent_turn",
                ))
        
        # Active task context
        if self.hot.active_task:
            task_entry = MemoryEntry(
                text=f"[Active Task: {self.hot.active_task.task_type.value}] "
                     f"{self.hot.active_task.description}",
                tier=MemoryTier.HOT,
                content_type=ContentType.CONVERSATION,
                session_id=self.hot.session_id or context.session_id,
                source_type=SourceType.SYSTEM,
                importance=0.7,
            )
            scored_entries.append(ScoredEntry(
                entry=task_entry,
                relevance_score=0.9,
                source_tier=MemoryTier.HOT,
                retrieval_reason="active_task",
            ))
        
        # Recent tool results
        for tool_result in list(self.hot.recent_tool_results)[-5:]:
            tool_entry = MemoryEntry(
                text=f"[Tool: {tool_result.tool_name}]\n{tool_result.result}",
                tier=MemoryTier.HOT,
                content_type=ContentType.TOOL_RESULT,
                session_id=self.hot.session_id or context.session_id,
                source_type=SourceType.TOOL,
                source_detail=tool_result.tool_name,
                cost_usd=tool_result.cost_usd,
                created_at=tool_result.timestamp,
            )
            scored_entries.append(ScoredEntry(
                entry=tool_entry,
                relevance_score=0.7,
                source_tier=MemoryTier.HOT,
                retrieval_reason="recent_tool_result",
            ))
        
        # ── Warm Memory Retrieval (Semantic) ────────────────────────
        if self.embedding is not None:
            # Extract last user message for embedding
            last_user_msg = ""
            for msg in reversed(context.current_messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        last_user_msg = content
                    break
            
            if last_user_msg:
                query_embedding = self.embedding.embed(last_user_msg)
                similar = self.warm.search_similar(
                    query_embedding,
                    top_k=top_k,
                    min_similarity=0.5,  # Lower threshold, relevance score handles rest
                )
                
                for entry, similarity in similar:
                    # Check freshness
                    freshness = self.decision.check_freshness(entry)
                    
                    relevance = compute_relevance_score(
                        entry, context, embedding_similarity=similarity
                    )
                    
                    # Penalize stale entries
                    if not freshness.is_fresh:
                        relevance *= 0.5
                    
                    if relevance >= min_relevance:
                        scored_entries.append(ScoredEntry(
                            entry=entry,
                            relevance_score=relevance,
                            source_tier=MemoryTier.WARM,
                            retrieval_reason=f"semantic ({similarity:.2f})",
                        ))
        
        # ── Deduplication ───────────────────────────────────────────
        # Remove duplicates (same text_hash)
        seen_hashes: set[str] = set()
        deduped: list[ScoredEntry] = []
        for scored in scored_entries:
            h = scored.entry.text_hash
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped.append(scored)
        
        # ── Ranking ─────────────────────────────────────────────────
        deduped.sort(key=lambda s: s.relevance_score, reverse=True)
        
        # ── Budget Capping ──────────────────────────────────────────
        total_tokens = 0
        budget_entries: list[ScoredEntry] = []
        
        for scored in deduped:
            entry_tokens = scored.entry.estimate_tokens()
            if total_tokens + entry_tokens <= budget_tokens:
                budget_entries.append(scored)
                total_tokens += entry_tokens
            else:
                break
        
        elapsed_ms = (time.monotonic() - start) * 1000
        
        # Count sources
        sources: dict[str, int] = {}
        for scored in budget_entries:
            tier = scored.source_tier.value
            sources[tier] = sources.get(tier, 0) + 1
        
        return RetrievalResult(
            entries=budget_entries,
            total_tokens=total_tokens,
            sources=sources,
            retrieval_time_ms=elapsed_ms,
        )
    
    def retrieve_for_cache_check(
        self,
        query_messages: list[dict],
    ) -> tuple[Optional[MemoryEntry], FreshnessResult]:
        """
        Check if we have a cached answer for these messages.
        
        Returns (entry, freshness) or (None, stale result).
        """
        import hashlib
        from agent_memory.normalizer import normalize_messages
        
        normalized = normalize_messages(query_messages)
        text_hash = hashlib.sha256(normalized.encode()).hexdigest()
        
        # Exact match
        entry = self.warm.get_by_hash(text_hash)
        if entry:
            freshness = self.decision.check_freshness(entry)
            return entry, freshness
        
        # Semantic match
        if self.embedding:
            last_user_msg = ""
            for msg in reversed(query_messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        last_user_msg = content
                    break
            
            if last_user_msg:
                query_emb = self.embedding.embed(last_user_msg)
                similar = self.warm.search_similar(query_emb, top_k=1, min_similarity=0.90)
                if similar:
                    entry, sim = similar[0]
                    freshness = self.decision.check_freshness(entry)
                    return entry, freshness
        
        return None, FreshnessResult(is_fresh=False, reason="No match found")
