"""
Response Cache — Multi-layer cache with exact + semantic matching.

Ties together:
- Layer 0: Exact match (hash lookup) — $0, <1ms
- Layer 1: Semantic match (FAISS) — ~$0, ~5ms
- Layer 2: Triage gate (cheap model verification) — ~$0.00015, ~1s
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Optional

from agent_memory.embedding import EmbeddingEngine, HashEmbedding
from agent_memory.normalizer import normalize_messages
from agent_memory.store import SQLiteStore
from agent_memory.types import (
    CacheHitType,
    ContentType,
    CostEvent,
    MemoryEntry,
    MemoryTier,
    SourceType,
)


@dataclass
class CacheResult:
    """Result of a cache lookup."""
    
    hit: bool
    hit_type: CacheHitType
    response: Optional[str] = None
    entry_id: Optional[str] = None
    similarity: float = 0.0
    latency_ms: float = 0.0
    cost_saved: float = 0.0


class ResponseCache:
    """
    Multi-layer response cache for LLM calls.
    
    Layers:
    0. Exact match — SHA-256 hash of normalized prompt
    1. Semantic match — FAISS nearest neighbor on embeddings
    2. Triage gate — cheap model verification (gray zone)
    
    Usage:
        cache = ResponseCache(store, embedding_engine)
        
        # Check cache before LLM call
        result = cache.get(messages, model="gpt-4", session_id="s123")
        if result.hit:
            return result.response  # Skip LLM entirely
        
        # After LLM call, store result
        response = llm.complete(messages)
        cache.store(messages, response, model="gpt-4", session_id="s123")
    """
    
    def __init__(
        self,
        db: SQLiteStore,
        embedding_engine: Optional[EmbeddingEngine] = None,
        hit_threshold: float = 0.95,
        gray_zone_low: float = 0.80,
        gray_zone_high: float = 0.95,
        session_id: str = "",
    ):
        self.db = db
        self.embedding_engine = embedding_engine or HashEmbedding()
        self.hit_threshold = hit_threshold
        self.gray_zone_low = gray_zone_low
        self.gray_zone_high = gray_zone_high
        self.session_id = session_id
    
    def get(
        self,
        messages: list[dict],
        model: str = "",
        session_id: Optional[str] = None,
    ) -> CacheResult:
        """
        Check cache for a matching response.
        
        Returns CacheResult with hit/miss and the response if hit.
        """
        start = time.monotonic()
        sid = session_id or self.session_id
        
        # Normalize prompt
        normalized = normalize_messages(messages)
        prompt_hash = hashlib.sha256(normalized.encode()).hexdigest()
        
        # ── Layer 0: Exact Match ────────────────────────────────────
        entry = self.db.get_by_hash(prompt_hash)
        if entry:
            elapsed = (time.monotonic() - start) * 1000
            self.db.record_cache_hit(CacheHitType.CLIENT_EXACT)
            return CacheResult(
                hit=True,
                hit_type=CacheHitType.CLIENT_EXACT,
                response=entry.text,
                entry_id=entry.id,
                similarity=1.0,
                latency_ms=elapsed,
                cost_saved=entry.cost_usd,
            )
        
        # ── Layer 1: Semantic Match ─────────────────────────────────
        if self.embedding_engine is not None:
            # Embed the last user message (not full prompt)
            last_user_msg = _extract_last_user_message(messages)
            if last_user_msg:
                query_embedding = self.embedding_engine.embed(last_user_msg)
                similar = self.db.search_similar(
                    query_embedding,
                    top_k=1,
                    min_similarity=self.gray_zone_low,
                )
                
                if similar:
                    best_entry, similarity = similar[0]
                    
                    if similarity >= self.hit_threshold:
                        # High confidence — return cached response
                        elapsed = (time.monotonic() - start) * 1000
                        self.db.record_cache_hit(CacheHitType.CLIENT_SEMANTIC)
                        return CacheResult(
                            hit=True,
                            hit_type=CacheHitType.CLIENT_SEMANTIC,
                            response=best_entry.text,
                            entry_id=best_entry.id,
                            similarity=similarity,
                            latency_ms=elapsed,
                            cost_saved=best_entry.cost_usd,
                        )
                    
                    elif similarity >= self.gray_zone_low:
                        # Gray zone — needs triage verification
                        # For Phase 1: return as miss (triage in Phase 2)
                        pass
        
        # ── Layer 2: Miss ───────────────────────────────────────────
        elapsed = (time.monotonic() - start) * 1000
        self.db.record_cache_hit(CacheHitType.NONE)
        return CacheResult(
            hit=False,
            hit_type=CacheHitType.NONE,
            latency_ms=elapsed,
        )
    
    def store(
        self,
        messages: list[dict],
        response: str,
        model: str = "",
        session_id: Optional[str] = None,
        cost_usd: float = 0.0,
        content_type: ContentType = ContentType.CONVERSATION,
        task_type=None,
    ) -> str:
        """
        Store a response in the cache.
        
        Returns the entry ID.
        """
        sid = session_id or self.session_id
        normalized = normalize_messages(messages)
        
        # Embed for semantic search
        last_user_msg = _extract_last_user_message(messages)
        embedding = None
        if last_user_msg and self.embedding_engine:
            embedding = self.embedding_engine.embed(last_user_msg)
        
        entry = MemoryEntry(
            text=response,
            tier=MemoryTier.WARM,
            content_type=content_type,
            session_id=sid,
            source_type=SourceType.MODEL,
            source_detail=model,
            model_used=model,
            cost_usd=cost_usd,
            embedding=embedding,
            task_type=task_type,
        )
        
        # Override text_hash with normalized prompt hash for exact match
        entry.text_hash_override = hashlib.sha256(normalized.encode()).hexdigest()
        
        return self.db.store(entry)
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = self.db.get_stats()
        return stats.to_dict()


def _extract_last_user_message(messages: list[dict]) -> Optional[str]:
    """Extract the last user message content for embedding."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Handle multimodal content — extract text parts
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                if text_parts:
                    return " ".join(text_parts)
    return None
