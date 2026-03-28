"""
Decision Engine — Intelligence layer for the memory system.

Decides:
- What to remember from this turn?
- Which model to route to?
- Is a cached answer still fresh?
- How important is this memory entry?
- What compaction actions to take?
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Protocol

from agent_memory.types import (
    CacheHitType,
    ContentType,
    MemoryEntry,
    MemoryTier,
    SourceType,
    TaskType,
)


# ── Result Types ─────────────────────────────────────────────────────


@dataclass
class RememberDecision:
    """Should this turn's output be remembered?"""
    remember: bool
    reason: str = ""
    tier: MemoryTier = MemoryTier.WARM
    importance: float = 0.5


@dataclass
class ModelRoute:
    """Which model to route to?"""
    route: str  # 'cached' | 'cheap' | 'standard' | 'advanced'
    model: Optional[str] = None
    reason: str = ""
    confidence: float = 0.0


@dataclass
class FreshnessResult:
    """Is a cached entry still valid?"""
    is_fresh: bool
    confidence: float = 1.0
    reason: str = ""


@dataclass
class CompactionAction:
    """Suggested compaction action for an entry."""
    action: str  # 'keep' | 'merge' | 'summarize' | 'drop'
    entry_ids: list[str]
    reason: str = ""


# ── Importance Scoring ───────────────────────────────────────────────


def score_importance(
    entry: MemoryEntry,
    access_history: Optional[list[int]] = None,
    reference_count: int = 0,
) -> float:
    """
    Score how important this entry is for long-term retention.
    
    Higher = less likely to be evicted.
    Range: 0.0 - 1.0
    
    Factors:
    - Content type (decisions/preferences are important)
    - Access frequency
    - References from other entries
    - Cost to produce
    - Age vs. activity
    """
    score = 0.5  # Base score
    
    # Content type signals
    type_scores = {
        ContentType.DECISION: 0.3,
        ContentType.PREFERENCE: 0.25,
        ContentType.INSIGHT: 0.2,
        ContentType.CONVERSATION: 0.0,
        ContentType.TOOL_RESULT: 0.05,
        ContentType.LOG: -0.1,
    }
    score += type_scores.get(entry.content_type, 0.0)
    
    # Access frequency (log scale, caps at ~100)
    if entry.access_count > 0:
        score += min(0.15, math.log(1 + entry.access_count) / math.log(1 + 100) * 0.15)
    
    # Referenced by other entries
    if reference_count > 0:
        score += min(0.1, reference_count * 0.02)
    
    # Expensive to produce = probably important
    if entry.cost_usd > 0.1:
        score += 0.1
    elif entry.cost_usd > 0.01:
        score += 0.05
    
    # Age penalty: never accessed after creation = unimportant
    age_hours = (time.time() - entry.created_at) / 3600
    if entry.access_count == 0 and age_hours > 24:
        score -= 0.2
    
    # User/explicit signals
    if entry.source_type == SourceType.USER:
        score += 0.05  # User messages are slightly more important
    
    return max(0.0, min(1.0, score))


# ── Freshness Checking ──────────────────────────────────────────────


# Default TTLs by content type (seconds)
DEFAULT_TTLS: dict[ContentType, Optional[int]] = {
    ContentType.CONVERSATION: None,     # Never expires
    ContentType.TOOL_RESULT: 3600,      # 1 hour
    ContentType.DECISION: 86400 * 30,   # 30 days
    ContentType.INSIGHT: 86400 * 7,     # 7 days
    ContentType.LOG: 86400,             # 1 day
    ContentType.PREFERENCE: None,       # Never expires
}

# Tool-specific TTL overrides
TOOL_TTLS: dict[str, int] = {
    "web_fetch": 1800,      # 30 min
    "read_file": 300,       # 5 min
    "write_file": 60,       # 1 min (file might change)
    "execute_code": 0,      # Never cache
    "memory_search": 300,   # 5 min
    "database_query": 60,   # 1 min
    "api_call": 300,        # 5 min
    "shell_command": 300,   # 5 min
}


def check_freshness(
    entry: MemoryEntry,
    content_type_ttl: Optional[dict[ContentType, int]] = None,
    tool_ttl: Optional[dict[str, int]] = None,
) -> FreshnessResult:
    """
    Determine if a cached entry is still valid.
    
    Checks:
    1. Content type TTL
    2. Tool-specific TTL (if applicable)
    3. Explicit expiry
    """
    ttls = content_type_ttl or DEFAULT_TTLS
    tool_ttls = tool_ttl or TOOL_TTLS
    
    # Explicit expiry
    if entry.expires_at:
        if time.time() > entry.expires_at:
            return FreshnessResult(
                is_fresh=False,
                confidence=1.0,
                reason=f"Explicitly expired at {entry.expires_at}",
            )
    
    # Tool-specific TTL
    if entry.content_type == ContentType.TOOL_RESULT and entry.source_detail:
        ttl = tool_ttls.get(entry.source_detail)
        if ttl is not None:
            age = time.time() - entry.created_at
            if age >= ttl:
                return FreshnessResult(
                    is_fresh=False,
                    confidence=0.9,
                    reason=f"Tool '{entry.source_detail}' TTL expired ({age:.0f}s > {ttl}s)",
                )
            elif age > ttl * 0.8:
                # Nearing expiry — flag but allow use
                return FreshnessResult(
                    is_fresh=True,
                    confidence=0.7,
                    reason=f"Tool '{entry.source_detail}' nearing TTL expiry",
                )
    
    # Content type TTL
    ttl = ttls.get(entry.content_type)
    if ttl is not None:
        age = time.time() - entry.created_at
        if age >= ttl:
            return FreshnessResult(
                is_fresh=False,
                confidence=0.85,
                reason=f"Content type TTL expired ({age:.0f}s > {ttl}s)",
            )
    
    return FreshnessResult(is_fresh=True, confidence=1.0, reason="Fresh")


# ── Model Routing ────────────────────────────────────────────────────


def route_to_model(
    task_description: str = "",
    task_type: TaskType = TaskType.UNKNOWN,
    cached_entry: Optional[MemoryEntry] = None,
    cached_freshness: Optional[FreshnessResult] = None,
    session_cost: float = 0.0,
    budget_usd: float = 10.0,
    cheap_models: Optional[list[str]] = None,
    standard_models: Optional[list[str]] = None,
    advanced_models: Optional[list[str]] = None,
) -> ModelRoute:
    """
    Route to the right model based on task + memory context.
    
    Priority:
    1. Use cached answer (if fresh + high confidence)
    2. Use cheap model (simple tasks, over budget)
    3. Use standard model (normal tasks)
    4. Use advanced model (complex tasks)
    """
    cheap = cheap_models or ["gpt-4o-mini", "claude-haiku", "mistral-small"]
    standard = standard_models or ["gpt-4o", "claude-sonnet", "mistral-medium"]
    advanced = advanced_models or ["gpt-4", "claude-opus", "mistral-large"]
    
    # Check cached answer
    if cached_entry and cached_freshness:
        if cached_freshness.is_fresh and cached_freshness.confidence >= 0.8:
            return ModelRoute(
                route="cached",
                model=cached_entry.model_used,
                reason=f"Cached answer available (freshness: {cached_freshness.confidence:.0%})",
                confidence=cached_freshness.confidence,
            )
    
    # Budget check — if over 80% of budget, use cheap model
    budget_ratio = session_cost / budget_usd if budget_usd > 0 else 0
    if budget_ratio > 0.8:
        return ModelRoute(
            route="cheap",
            model=cheap[0] if cheap else None,
            reason=f"Over budget threshold ({budget_ratio:.0%} of ${budget_usd})",
            confidence=0.7,
        )
    
    # Task complexity heuristics
    complexity = _estimate_complexity(task_description, task_type)
    
    if complexity <= 0.3:
        return ModelRoute(
            route="cheap",
            model=cheap[0] if cheap else None,
            reason=f"Simple task (complexity: {complexity:.2f})",
            confidence=0.8,
        )
    elif complexity <= 0.7:
        return ModelRoute(
            route="standard",
            model=standard[0] if standard else None,
            reason=f"Moderate task (complexity: {complexity:.2f})",
            confidence=0.8,
        )
    else:
        return ModelRoute(
            route="advanced",
            model=advanced[0] if advanced else None,
            reason=f"Complex task (complexity: {complexity:.2f})",
            confidence=0.8,
        )


def _estimate_complexity(task_description: str, task_type: TaskType) -> float:
    """
    Estimate task complexity on 0.0-1.0 scale.
    
    This is a simple heuristic — production version would use a classifier.
    """
    score = 0.5  # Base: moderate
    
    # Task type heuristics
    simple_types = {TaskType.TRIAGE, TaskType.MAINTENANCE}
    complex_types = {TaskType.DEBUG, TaskType.DESIGN, TaskType.RESEARCH}
    
    if task_type in simple_types:
        score -= 0.3
    elif task_type in complex_types:
        score += 0.3
    
    # Description heuristics
    desc_lower = task_description.lower()
    
    # Simple indicators
    simple_keywords = ["what is", "define", "list", "count", "yes or no", "true or false"]
    for kw in simple_keywords:
        if kw in desc_lower:
            score -= 0.15
            break
    
    # Complex indicators
    complex_keywords = ["why", "analyze", "compare", "design", "architect", "explain", "debug"]
    for kw in complex_keywords:
        if kw in desc_lower:
            score += 0.15
            break
    
    # Length heuristic — longer = usually more complex
    if len(task_description) > 500:
        score += 0.1
    elif len(task_description) < 50:
        score -= 0.1
    
    return max(0.0, min(1.0, score))


# ── Remember Decision ────────────────────────────────────────────────


def should_remember(
    text: str,
    content_type: ContentType,
    is_new_info: bool = True,
    is_repeat_with_new_answer: bool = False,
    explicit_save_signal: bool = False,
    current_memory_similarity: float = 0.0,
) -> RememberDecision:
    """
    Decide if something is worth storing in memory.
    
    Signals that favor remembering:
    - Explicit user request ("remember this", "note that")
    - Decisions or conclusions
    - New information (not already in memory)
    - Repeated question with different/better answer
    - High-cost generation (already paid for it)
    
    Signals that favor forgetting:
    - Already have very similar content (similarity > 0.95)
    - Trivial/chitchat content
    - Error messages / failed tool calls
    """
    # Explicit signal is strongest
    if explicit_save_signal:
        return RememberDecision(
            remember=True,
            reason="Explicit save signal",
            tier=MemoryTier.WARM,
            importance=0.8,
        )
    
    # Decision/conclusion — always remember
    if content_type == ContentType.DECISION:
        return RememberDecision(
            remember=True,
            reason="Decision/conclusion",
            tier=MemoryTier.WARM,
            importance=0.85,
        )
    
    # Already have very similar content
    if current_memory_similarity > 0.95 and not is_repeat_with_new_answer:
        return RememberDecision(
            remember=False,
            reason=f"Already have similar content (similarity: {current_memory_similarity:.2f})",
        )
    
    # New information is worth keeping
    if is_new_info:
        return RememberDecision(
            remember=True,
            reason="New information",
            tier=MemoryTier.WARM,
            importance=0.6,
        )
    
    # Repeat with new answer — useful to keep
    if is_repeat_with_new_answer:
        return RememberDecision(
            remember=True,
            reason="Repeated question with new answer",
            tier=MemoryTier.WARM,
            importance=0.5,
        )
    
    # Default: don't store trivial content
    return RememberDecision(
        remember=False,
        reason="Not significant enough to store",
    )


# ── Compaction Suggestions ──────────────────────────────────────────


def suggest_compaction(
    entries: list[MemoryEntry],
    max_entries: int = 1000,
    max_age_days: int = 30,
) -> list[CompactionAction]:
    """
    Suggest compaction actions to manage memory size.
    
    Strategies:
    1. Drop expired entries
    2. Drop low-importance, never-accessed entries
    3. Merge similar entries
    4. Summarize old conversation threads
    """
    actions: list[CompactionAction] = []
    now = time.time()
    
    # 1. Drop expired
    expired = [e for e in entries if e.is_expired]
    if expired:
        actions.append(CompactionAction(
            action="drop",
            entry_ids=[e.id for e in expired],
            reason=f"{len(expired)} expired entries",
        ))
    
    # 2. Drop low-importance, never-accessed, old
    droppable = [
        e for e in entries
        if e.importance < 0.2
        and e.access_count == 0
        and (now - e.created_at) > max_age_days * 86400
        and e not in expired
    ]
    if droppable:
        actions.append(CompactionAction(
            action="drop",
            entry_ids=[e.id for e in droppable],
            reason=f"{len(droppable)} low-importance, unaccessed entries older than {max_age_days}d",
        ))
    
    # 3. Find similar entries for merging (by content type + session)
    by_type_session: dict[tuple, list[MemoryEntry]] = {}
    for e in entries:
        key = (e.content_type, e.session_id)
        by_type_session.setdefault(key, []).append(e)
    
    for key, group in by_type_session.items():
        if len(group) > 10:  # Only merge if there are many
            # Find entries with same importance tier (within 0.1)
            importance_groups: dict[int, list[MemoryEntry]] = {}
            for e in group:
                bucket = int(e.importance * 10)
                importance_groups.setdefault(bucket, []).append(e)
            
            for bucket, bucket_entries in importance_groups.items():
                if len(bucket_entries) > 5:
                    # Suggest merging oldest 80% into summary
                    sorted_entries = sorted(bucket_entries, key=lambda e: e.created_at)
                    merge_count = int(len(sorted_entries) * 0.8)
                    if merge_count > 1:
                        actions.append(CompactionAction(
                            action="merge",
                            entry_ids=[e.id for e in sorted_entries[:merge_count]],
                            reason=f"Merge {merge_count} similar {key[0].value} entries",
                        ))
    
    return actions


# ── Decision Engine (Combined Interface) ─────────────────────────────


class DecisionEngine:
    """
    Combined decision engine implementing all memory decisions.
    
    This is the main interface for the memory system's intelligence layer.
    """
    
    def __init__(
        self,
        budget_usd: float = 10.0,
        cheap_models: Optional[list[str]] = None,
        standard_models: Optional[list[str]] = None,
        advanced_models: Optional[list[str]] = None,
    ):
        self.budget_usd = budget_usd
        self.cheap_models = cheap_models
        self.standard_models = standard_models
        self.advanced_models = advanced_models
    
    def score_importance(self, entry: MemoryEntry, **kwargs) -> float:
        """Score entry importance."""
        return score_importance(entry, **kwargs)
    
    def check_freshness(self, entry: MemoryEntry, **kwargs) -> FreshnessResult:
        """Check if entry is still fresh."""
        return check_freshness(entry, **kwargs)
    
    def route_to_model(
        self,
        task_description: str = "",
        task_type: TaskType = TaskType.UNKNOWN,
        cached_entry: Optional[MemoryEntry] = None,
        cached_freshness: Optional[FreshnessResult] = None,
        session_cost: float = 0.0,
    ) -> ModelRoute:
        """Route to appropriate model."""
        return route_to_model(
            task_description=task_description,
            task_type=task_type,
            cached_entry=cached_entry,
            cached_freshness=cached_freshness,
            session_cost=session_cost,
            budget_usd=self.budget_usd,
            cheap_models=self.cheap_models,
            standard_models=self.standard_models,
            advanced_models=self.advanced_models,
        )
    
    def should_remember(self, **kwargs) -> RememberDecision:
        """Decide if content should be remembered."""
        return should_remember(**kwargs)
    
    def suggest_compaction(self, entries: list[MemoryEntry], **kwargs) -> list[CompactionAction]:
        """Suggest compaction actions."""
        return suggest_compaction(entries, **kwargs)
