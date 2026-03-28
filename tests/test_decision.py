"""Tests for the Decision Engine."""

import time

from agent_memory.decision import (
    DecisionEngine,
    score_importance,
    check_freshness,
    route_to_model,
    should_remember,
    suggest_compaction,
    FreshnessResult,
    ModelRoute,
    RememberDecision,
)
from agent_memory.types import (
    ContentType,
    MemoryEntry,
    MemoryTier,
    SourceType,
    TaskType,
)


def _make_entry(
    content_type: ContentType = ContentType.CONVERSATION,
    importance: float = 0.5,
    access_count: int = 0,
    cost: float = 0.0,
    age_seconds: int = 0,
    source_type: SourceType = SourceType.MODEL,
    source_detail: str = "gpt-4",
) -> MemoryEntry:
    """Helper to create test entries."""
    now = int(time.time()) - age_seconds
    return MemoryEntry(
        text="Test entry",
        tier=MemoryTier.WARM,
        content_type=content_type,
        session_id="test-session",
        source_type=source_type,
        source_detail=source_detail,
        importance=importance,
        access_count=access_count,
        cost_usd=cost,
        created_at=now,
        last_accessed_at=now,
    )


# ── Importance Scoring ───────────────────────────────────────────────

def test_decisions_are_important():
    entry = _make_entry(content_type=ContentType.DECISION)
    assert score_importance(entry) >= 0.7


def test_preferences_are_important():
    entry = _make_entry(content_type=ContentType.PREFERENCE)
    assert score_importance(entry) >= 0.7


def test_logs_are_unimportant():
    entry = _make_entry(content_type=ContentType.LOG)
    assert score_importance(entry) < 0.5


def test_expensive_entries_are_important():
    entry = _make_entry(cost=0.15)
    assert score_importance(entry) >= 0.6


def test_accessed_entries_are_important():
    entry = _make_entry(access_count=20)
    assert score_importance(entry) >= 0.59


def test_old_unaccessed_are_unimportant():
    entry = _make_entry(age_seconds=86400 * 2, access_count=0)  # 2 days, never accessed
    assert score_importance(entry) < 0.5


def test_importance_bounds():
    """Importance should always be 0.0-1.0."""
    # Extreme inputs
    very_bad = _make_entry(content_type=ContentType.LOG, age_seconds=86400 * 60, access_count=0)
    very_good = _make_entry(content_type=ContentType.DECISION, cost=1.0, access_count=100)
    
    assert 0.0 <= score_importance(very_bad) <= 1.0
    assert 0.0 <= score_importance(very_good) <= 1.0


# ── Freshness Checking ──────────────────────────────────────────────

def test_conversation_never_expires():
    entry = _make_entry(content_type=ContentType.CONVERSATION, age_seconds=86400 * 365)
    result = check_freshness(entry)
    assert result.is_fresh


def test_tool_result_expires():
    entry = _make_entry(
        content_type=ContentType.TOOL_RESULT,
        source_detail="web_fetch",
        age_seconds=8600,  # > 1 hour
    )
    result = check_freshness(entry)
    assert not result.is_fresh


def test_tool_result_fresh_within_ttl():
    entry = _make_entry(
        content_type=ContentType.TOOL_RESULT,
        source_detail="web_fetch",
        age_seconds=1700,  # 28 min (well within 1 hour / 30 min TTL)
    )
    result = check_freshness(entry)
    assert result.is_fresh


def test_decisions_long_ttl():
    entry = _make_entry(
        content_type=ContentType.DECISION,
        age_seconds=86400 * 10,  # 10 days (within 30 day TTL)
    )
    result = check_freshness(entry)
    assert result.is_fresh


def test_decisions_expired():
    entry = _make_entry(
        content_type=ContentType.DECISION,
        age_seconds=86400 * 40,  # 40 days (beyond 30 day TTL)
    )
    result = check_freshness(entry)
    assert not result.is_fresh


def test_explicit_expiry():
    entry = _make_entry()
    entry.expires_at = int(time.time()) - 1000  # Expired
    result = check_freshness(entry)
    assert not result.is_fresh


# ── Model Routing ────────────────────────────────────────────────────

def test_cached_route_when_fresh():
    entry = _make_entry()
    freshness = FreshnessResult(is_fresh=True, confidence=0.95)
    
    route = route_to_model(cached_entry=entry, cached_freshness=freshness)
    assert route.route == "cached"


def test_cheap_route_for_simple_tasks():
    route = route_to_model(task_description="What is Python?")
    assert route.route == "cheap"


def test_advanced_route_for_complex_tasks():
    route = route_to_model(
        task_description="Analyze the architecture and design a comprehensive solution for microservices communication",
        task_type=TaskType.DESIGN,
    )
    assert route.route == "advanced"


def test_budget_overrides_to_cheap():
    route = route_to_model(
        task_description="Design a complex system",
        task_type=TaskType.DESIGN,
        session_cost=9.0,
        budget_usd=10.0,
    )
    assert route.route == "cheap"
    assert "budget" in route.reason.lower()


def test_triage_type_is_cheap():
    route = route_to_model(task_type=TaskType.TRIAGE)
    assert route.route == "cheap"


def test_maintenance_type_is_cheap():
    route = route_to_model(task_type=TaskType.MAINTENANCE)
    assert route.route == "cheap"


# ── Should Remember ─────────────────────────────────────────────────

def test_explicit_save_signal():
    result = should_remember(
        text="Some text",
        content_type=ContentType.CONVERSATION,
        explicit_save_signal=True,
    )
    assert result.remember
    assert "explicit" in result.reason.lower()


def test_decisions_always_remembered():
    result = should_remember(
        text="Let's use Python for this",
        content_type=ContentType.DECISION,
    )
    assert result.remember


def test_duplicate_content_not_remembered():
    result = should_remember(
        text="Some text",
        content_type=ContentType.CONVERSATION,
        current_memory_similarity=0.98,
    )
    assert not result.remember


def test_new_info_remembered():
    result = should_remember(
        text="New information",
        content_type=ContentType.CONVERSATION,
        is_new_info=True,
    )
    assert result.remember


def test_trivial_content_not_remembered():
    result = should_remember(
        text="ok",
        content_type=ContentType.CONVERSATION,
        is_new_info=False,
    )
    assert not result.remember


# ── Compaction Suggestions ──────────────────────────────────────────

def test_expired_entries_marked_for_drop():
    entries = [
        _make_entry(),  # Not expired
        _make_entry(),  # Not expired
    ]
    entries[0].expires_at = int(time.time()) - 1000  # Expired
    entries[1].expires_at = int(time.time()) + 1000  # Not expired
    
    actions = suggest_compaction(entries)
    
    drop_actions = [a for a in actions if a.action == "drop" and "expired" in a.reason]
    assert len(drop_actions) >= 1
    assert entries[0].id in drop_actions[0].entry_ids


def test_low_importance_old_entries_dropped():
    entries = [
        _make_entry(importance=0.1, access_count=0, age_seconds=86400 * 35),
        _make_entry(importance=0.8),  # Important, kept
    ]
    
    actions = suggest_compaction(entries)
    
    drop_actions = [a for a in actions if a.action == "drop" and "low-importance" in a.reason]
    assert len(drop_actions) >= 1


# ── DecisionEngine (Combined) ────────────────────────────────────────

def test_decision_engine_init():
    engine = DecisionEngine(budget_usd=5.0)
    assert engine.budget_usd == 5.0


def test_decision_engine_methods():
    engine = DecisionEngine()
    
    entry = _make_entry(content_type=ContentType.DECISION)
    
    # Should have all methods
    assert engine.score_importance(entry) > 0.5
    assert isinstance(engine.check_freshness(entry), FreshnessResult)
    assert isinstance(engine.route_to_model(), ModelRoute)
    assert isinstance(engine.should_remember(
        text="test", content_type=ContentType.DECISION
    ), RememberDecision)
