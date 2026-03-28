"""
Warm Memory Store — SQLite + FAISS.

Storage layer for warm memory tier.
Handles exact match (hash lookup) and semantic match (vector search).
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from agent_memory.types import (
    CacheHitType,
    ContentType,
    CostEvent,
    EvictionPolicy,
    MemoryEntry,
    MemoryStats,
    MemoryTier,
    SourceType,
    TaskType,
    ToolCacheEntry,
)


# ── Schema ───────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_entries (
    id TEXT PRIMARY KEY,
    tier TEXT NOT NULL,
    content_type TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB,
    source_type TEXT NOT NULL,
    source_detail TEXT,
    session_id TEXT NOT NULL,
    task_type TEXT,
    model_used TEXT,
    cost_usd REAL DEFAULT 0,
    created_at INTEGER NOT NULL,
    last_accessed_at INTEGER NOT NULL,
    access_count INTEGER DEFAULT 0,
    expires_at INTEGER,
    importance REAL DEFAULT 0.5,
    parent_id TEXT,
    tool_call_id TEXT,
    text_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_tier ON memory_entries(tier);
CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_entries(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_hash ON memory_entries(text_hash);
CREATE INDEX IF NOT EXISTS idx_memory_created ON memory_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_memory_accessed ON memory_entries(last_accessed_at);
CREATE INDEX IF NOT EXISTS idx_memory_importance ON memory_entries(importance);
CREATE INDEX IF NOT EXISTS idx_memory_content_type ON memory_entries(content_type);

CREATE TABLE IF NOT EXISTS tool_cache (
    cache_key TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    tool_args_hash TEXT NOT NULL,
    result TEXT NOT NULL,
    result_hash TEXT NOT NULL,
    session_id TEXT,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    hit_count INTEGER DEFAULT 0,
    last_hit_at INTEGER
);

CREATE INDEX IF NOT EXISTS idx_tool_cache_expiry ON tool_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_tool_cache_tool ON tool_cache(tool_name);

CREATE TABLE IF NOT EXISTS cost_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd REAL NOT NULL,
    cache_hit_type TEXT,
    task_type TEXT
);

CREATE INDEX IF NOT EXISTS idx_cost_session ON cost_events(session_id);
CREATE INDEX IF NOT EXISTS idx_cost_time ON cost_events(timestamp);

CREATE TABLE IF NOT EXISTS cache_stats (
    key TEXT PRIMARY KEY,
    value INTEGER DEFAULT 0
);

INSERT OR IGNORE INTO cache_stats (key, value) VALUES ('exact_hits', 0);
INSERT OR IGNORE INTO cache_stats (key, value) VALUES ('semantic_hits', 0);
INSERT OR IGNORE INTO cache_stats (key, value) VALUES ('triage_hits', 0);
INSERT OR IGNORE INTO cache_stats (key, value) VALUES ('misses', 0);
INSERT OR IGNORE INTO cache_stats (key, value) VALUES ('total_requests', 0);
"""


class SQLiteStore:
    """
    SQLite-backed warm memory store with FAISS vector index.
    
    Features:
    - Exact match via SHA-256 text hash
    - Semantic match via FAISS nearest neighbor search
    - Tool result cache with TTL
    - Cost event tracking
    - LRU/LFU eviction
    """
    
    def __init__(
        self,
        db_path: str = "~/.openclaw/agent-memory/cache.db",
        embedding_engine=None,
        max_entries: int = 100_000,
    ):
        self.db_path = Path(os.path.expanduser(db_path))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_engine = embedding_engine
        self.max_entries = max_entries
        
        self._conn: Optional[sqlite3.Connection] = None
        self._faiss_index = None
        self._id_map: dict[int, str] = {}  # FAISS index → memory_entry id
        self._next_faiss_id: int = 0
    
    # ── Connection Management ────────────────────────────────────────
    
    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=-64000")  # 64MB
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()
            self._load_faiss_index()
        return self._conn
    
    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
        self._faiss_index = None
    
    # ── FAISS Index ──────────────────────────────────────────────────
    
    def _load_faiss_index(self) -> None:
        """Load FAISS index from existing entries (on startup)."""
        if self.embedding_engine is None:
            return
        
        try:
            import faiss
            
            dim = self.embedding_engine.dimension
            self._faiss_index = faiss.IndexFlatIP(dim)  # Inner product (cosine if normalized)
            
            # Load existing embeddings
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT id, embedding FROM memory_entries WHERE embedding IS NOT NULL AND tier = 'warm'"
            ).fetchall()
            
            if rows:
                embeddings = []
                for row in rows:
                    embedding = _blob_to_vec(row["embedding"])
                    if embedding:
                        embeddings.append(embedding)
                        self._id_map[self._next_faiss_id] = row["id"]
                        self._next_faiss_id += 1
                
                if embeddings:
                    vectors = np.array(embeddings, dtype=np.float32)
                    self._faiss_index.add(vectors)
            
            print(f"[Memory] FAISS index loaded: {self._faiss_index.ntotal} vectors")
        except ImportError:
            print("[Memory] faiss-cpu not installed, semantic search disabled")
            self._faiss_index = None
    
    # ── Store Operations ─────────────────────────────────────────────
    
    def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry. Returns the entry ID."""
        conn = self._get_conn()
        
        embedding_blob = None
        if entry.embedding:
            embedding_blob = _vec_to_blob(entry.embedding)
        elif self.embedding_engine and entry.tier == MemoryTier.WARM:
            embedding = self.embedding_engine.embed(entry.text)
            entry.embedding = embedding
            embedding_blob = _vec_to_blob(embedding)
        
        conn.execute(
            """INSERT OR REPLACE INTO memory_entries 
               (id, tier, content_type, text, embedding, source_type, source_detail,
                session_id, task_type, model_used, cost_usd, created_at, last_accessed_at,
                access_count, expires_at, importance, parent_id, tool_call_id, text_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id, entry.tier.value, entry.content_type.value, entry.text,
                embedding_blob, entry.source_type.value, entry.source_detail,
                entry.session_id, entry.task_type.value if entry.task_type else None,
                entry.model_used, entry.cost_usd, entry.created_at, entry.last_accessed_at,
                entry.access_count, entry.expires_at, entry.importance,
                entry.parent_id, entry.tool_call_id, entry.text_hash,
            ),
        )
        conn.commit()
        
        # Add to FAISS index
        if self._faiss_index is not None and entry.embedding:
            self._faiss_index.add(np.array([entry.embedding], dtype=np.float32))
            self._id_map[self._next_faiss_id] = entry.id
            self._next_faiss_id += 1
        
        # Check if we need to evict
        self._maybe_evict()
        
        return entry.id
    
    def store_batch(self, entries: list[MemoryEntry]) -> list[str]:
        """Store multiple entries in a single transaction."""
        conn = self._get_conn()
        ids = []
        
        with conn:
            for entry in entries:
                embedding_blob = None
                if entry.embedding:
                    embedding_blob = _vec_to_blob(entry.embedding)
                elif self.embedding_engine and entry.tier == MemoryTier.WARM:
                    embedding = self.embedding_engine.embed(entry.text)
                    entry.embedding = embedding
                    embedding_blob = _vec_to_blob(embedding)
                
                conn.execute(
                    """INSERT OR REPLACE INTO memory_entries 
                       (id, tier, content_type, text, embedding, source_type, source_detail,
                        session_id, task_type, model_used, cost_usd, created_at, last_accessed_at,
                        access_count, expires_at, importance, parent_id, tool_call_id, text_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry.id, entry.tier.value, entry.content_type.value, entry.text,
                        embedding_blob, entry.source_type.value, entry.source_detail,
                        entry.session_id, entry.task_type.value if entry.task_type else None,
                        entry.model_used, entry.cost_usd, entry.created_at, entry.last_accessed_at,
                        entry.access_count, entry.expires_at, entry.importance,
                        entry.parent_id, entry.tool_call_id, entry.text_hash,
                    ),
                )
                ids.append(entry.id)
                
                if self._faiss_index is not None and entry.embedding:
                    self._faiss_index.add(np.array([entry.embedding], dtype=np.float32))
                    self._id_map[self._next_faiss_id] = entry.id
                    self._next_faiss_id += 1
        
        self._maybe_evict()
        return ids
    
    # ── Exact Match ──────────────────────────────────────────────────
    
    def get_by_hash(self, text_hash: str) -> Optional[MemoryEntry]:
        """Look up entry by text hash (exact match)."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM memory_entries WHERE text_hash = ? AND (expires_at IS NULL OR expires_at > ?)",
            (text_hash, int(time.time())),
        ).fetchone()
        
        if row:
            entry = _row_to_entry(row)
            # Update access metadata
            conn.execute(
                "UPDATE memory_entries SET last_accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
                (int(time.time()), entry.id),
            )
            conn.commit()
            return entry
        return None
    
    # ── Semantic Match ───────────────────────────────────────────────
    
    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 1,
        min_similarity: float = 0.80,
    ) -> list[tuple[MemoryEntry, float]]:
        """Search for similar entries using FAISS."""
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []
        
        query = np.array([query_embedding], dtype=np.float32)
        distances, indices = self._faiss_index.search(query, min(top_k, self._faiss_index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if dist < min_similarity:
                continue
            
            entry_id = self._id_map.get(idx)
            if entry_id:
                entry = self.get_by_id(entry_id)
                if entry and not entry.is_expired:
                    results.append((entry, float(dist)))
        
        return results
    
    def get_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get an entry by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM memory_entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
        
        if row:
            entry = _row_to_entry(row)
            conn.execute(
                "UPDATE memory_entries SET last_accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
                (int(time.time()), entry.id),
            )
            conn.commit()
            return entry
        return None
    
    # ── Session Retrieval ────────────────────────────────────────────
    
    def get_session_entries(
        self, 
        session_id: str, 
        limit: int = 20,
        content_type: Optional[ContentType] = None,
    ) -> list[MemoryEntry]:
        """Get recent entries for a session."""
        conn = self._get_conn()
        query = "SELECT * FROM memory_entries WHERE session_id = ?"
        params: list = [session_id]
        
        if content_type:
            query += " AND content_type = ?"
            params.append(content_type.value)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        rows = conn.execute(query, params).fetchall()
        return [_row_to_entry(r) for r in rows]
    
    # ── Tool Cache ───────────────────────────────────────────────────
    
    def get_tool_result(self, cache_key: str) -> Optional[str]:
        """Get a cached tool result. Returns None if expired or missing."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT result FROM tool_cache WHERE cache_key = ? AND expires_at > ?",
            (cache_key, int(time.time())),
        ).fetchone()
        
        if row:
            conn.execute(
                "UPDATE tool_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE cache_key = ?",
                (int(time.time()), cache_key),
            )
            conn.commit()
            return row["result"]
        return None
    
    def store_tool_result(
        self,
        cache_key: str,
        tool_name: str,
        tool_args_hash: str,
        result: str,
        session_id: Optional[str] = None,
        ttl: int = 3600,
    ) -> None:
        """Cache a tool result."""
        import hashlib
        conn = self._get_conn()
        result_hash = hashlib.sha256(result.encode()).hexdigest()
        
        conn.execute(
            """INSERT OR REPLACE INTO tool_cache 
               (cache_key, tool_name, tool_args_hash, result, result_hash, 
                session_id, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cache_key, tool_name, tool_args_hash, result, result_hash,
                session_id, int(time.time()), int(time.time()) + ttl,
            ),
        )
        conn.commit()
    
    def cleanup_expired_tool_cache(self) -> int:
        """Remove expired tool cache entries. Returns count removed."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM tool_cache WHERE expires_at <= ?",
            (int(time.time()),),
        )
        conn.commit()
        return cursor.rowcount
    
    # ── Cost Tracking ────────────────────────────────────────────────
    
    def record_cost(self, event: CostEvent) -> None:
        """Record a cost event."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO cost_events 
               (timestamp, session_id, event_type, model, input_tokens, output_tokens, 
                cost_usd, cache_hit_type, task_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.timestamp, event.session_id, event.event_type, event.model,
                event.input_tokens, event.output_tokens, event.cost_usd,
                event.cache_hit_type.value, event.task_type,
            ),
        )
        conn.commit()
    
    def get_session_cost(self, session_id: str) -> dict:
        """Get cost summary for a session."""
        conn = self._get_conn()
        row = conn.execute(
            """SELECT 
                COUNT(*) as total_calls,
                COALESCE(SUM(cost_usd), 0) as total_cost,
                COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                COALESCE(SUM(output_tokens), 0) as total_output_tokens
               FROM cost_events WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
        
        return dict(row) if row else {
            "total_calls": 0, "total_cost": 0.0,
            "total_input_tokens": 0, "total_output_tokens": 0,
        }
    
    # ── Stats ────────────────────────────────────────────────────────
    
    def _update_stat(self, key: str, amount: int = 1) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE cache_stats SET value = value + ? WHERE key = ?",
            (amount, key),
        )
        conn.commit()
    
    def record_cache_hit(self, hit_type: CacheHitType) -> None:
        """Record a cache hit event."""
        conn = self._get_conn()
        conn.execute("UPDATE cache_stats SET value = value + 1 WHERE key = 'total_requests'", ())
        
        if hit_type == CacheHitType.CLIENT_EXACT:
            conn.execute("UPDATE cache_stats SET value = value + 1 WHERE key = 'exact_hits'", ())
        elif hit_type == CacheHitType.CLIENT_SEMANTIC:
            conn.execute("UPDATE cache_stats SET value = value + 1 WHERE key = 'semantic_hits'", ())
        elif hit_type == CacheHitType.CLIENT_TRIAGE:
            conn.execute("UPDATE cache_stats SET value = value + 1 WHERE key = 'triage_hits'", ())
        else:
            conn.execute("UPDATE cache_stats SET value = value + 1 WHERE key = 'misses'", ())
        
        conn.commit()
    
    def get_stats(self) -> MemoryStats:
        """Get aggregate statistics."""
        conn = self._get_conn()
        
        # Cache stats
        stats_rows = conn.execute("SELECT key, value FROM cache_stats").fetchall()
        stats_dict = {r["key"]: r["value"] for r in stats_rows}
        
        # Entry counts
        entry_count = conn.execute("SELECT COUNT(*) as c FROM memory_entries").fetchone()["c"]
        tier_counts = conn.execute(
            "SELECT tier, COUNT(*) as c FROM memory_entries GROUP BY tier"
        ).fetchall()
        tier_dict = {r["tier"]: r["c"] for r in tier_counts}
        
        # Cost saved (estimated)
        cost_saved = 0.0
        for row in conn.execute(
            "SELECT cache_hit_type, SUM(cost_usd) as saved FROM cost_events WHERE cache_hit_type != 'none' GROUP BY cache_hit_type"
        ).fetchall():
            cost_saved += row["saved"] or 0
        
        return MemoryStats(
            total_entries=entry_count,
            hot_entries=0,  # Hot memory is managed separately
            warm_entries=tier_dict.get("warm", 0),
            cold_entries=tier_dict.get("cold", 0),
            exact_hits=stats_dict.get("exact_hits", 0),
            semantic_hits=stats_dict.get("semantic_hits", 0),
            triage_hits=stats_dict.get("triage_hits", 0),
            misses=stats_dict.get("misses", 0),
            total_cost_saved=cost_saved,
            total_requests=stats_dict.get("total_requests", 0),
        )
    
    # ── Eviction ─────────────────────────────────────────────────────
    
    def _maybe_evict(self) -> None:
        """Evict entries if we've exceeded max_entries."""
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) as c FROM memory_entries").fetchone()["c"]
        
        if count > self.max_entries:
            to_remove = count - self.max_entries
            conn.execute(
                """DELETE FROM memory_entries WHERE id IN (
                    SELECT id FROM memory_entries 
                    ORDER BY last_accessed_at ASC 
                    LIMIT ?
                )""",
                (to_remove,),
            )
            conn.commit()
    
    def evict(self, policy: EvictionPolicy = EvictionPolicy.LRU, count: int = 100) -> int:
        """Manually evict entries."""
        conn = self._get_conn()
        
        if policy == EvictionPolicy.LRU:
            order = "last_accessed_at ASC"
        elif policy == EvictionPolicy.LFU:
            order = "access_count ASC"
        elif policy == EvictionPolicy.LOWEST_IMPORTANCE:
            order = "importance ASC"
        elif policy == EvictionPolicy.EXPIRED:
            conn.execute(
                "DELETE FROM memory_entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (int(time.time()),),
            )
            conn.commit()
            return conn.execute("SELECT changes()").fetchone()[0]
        else:
            order = "last_accessed_at ASC"
        
        cursor = conn.execute(
            f"DELETE FROM memory_entries WHERE id IN (SELECT id FROM memory_entries ORDER BY {order} LIMIT ?)",
            (count,),
        )
        conn.commit()
        return cursor.rowcount
    
    def compact(self) -> dict:
        """Remove expired entries and optimize database."""
        conn = self._get_conn()
        
        # Remove expired
        expired = conn.execute(
            "DELETE FROM memory_entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (int(time.time()),),
        ).rowcount
        
        # Remove expired tool cache
        tool_expired = self.cleanup_expired_tool_cache()
        
        # VACUUM (can be slow on large DBs)
        # conn.execute("VACUUM")
        
        conn.commit()
        
        return {
            "expired_removed": expired,
            "tool_cache_removed": tool_expired,
        }


# ── Helpers ──────────────────────────────────────────────────────────

def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
    """Convert a SQLite row to a MemoryEntry."""
    return MemoryEntry(
        id=row["id"],
        tier=MemoryTier(row["tier"]),
        content_type=ContentType(row["content_type"]),
        text=row["text"],
        source_type=SourceType(row["source_type"]),
        source_detail=row["source_detail"],
        session_id=row["session_id"],
        task_type=TaskType(row["task_type"]) if row["task_type"] else None,
        model_used=row["model_used"],
        cost_usd=row["cost_usd"],
        created_at=row["created_at"],
        last_accessed_at=row["last_accessed_at"],
        access_count=row["access_count"],
        expires_at=row["expires_at"],
        importance=row["importance"],
        parent_id=row["parent_id"],
        tool_call_id=row["tool_call_id"],
        embedding=_blob_to_vec(row["embedding"]),
    )


def _vec_to_vec(list_vec: list[float]) -> list[float]:
    """Ensure embedding is a list of floats."""
    return [float(x) for x in list_vec]


def _vec_to_blob(vec: list[float]) -> bytes:
    """Convert embedding vector to binary blob for SQLite storage."""
    return struct.pack(f"{len(vec)}f", *vec)


def _blob_to_vec(blob: Optional[bytes]) -> Optional[list[float]]:
    """Convert binary blob back to embedding vector."""
    if blob is None:
        return None
    count = len(blob) // 4  # float32 = 4 bytes
    return list(struct.unpack(f"{count}f", blob))
