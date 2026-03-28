"""
Cloud Sync — Multi-device memory synchronization.

Syncs memory entries across devices using a simple HTTP API.
Compatible with any S3-compatible storage or custom sync server.

Architecture:
- Each device has a local SQLite database
- Sync pushes local changes to cloud, pulls remote changes
- Conflict resolution: latest timestamp wins
- All entries have device_id + version for tracking
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from agent_memory.types import ContentType, MemoryEntry, MemoryTier, SourceType

logger = logging.getLogger(__name__)


@dataclass
class SyncConfig:
    """Cloud sync configuration."""
    enabled: bool = False
    provider: str = "s3"  # s3 | http | custom
    bucket: str = ""
    prefix: str = "agent-memory/"
    region: str = "us-east-1"
    
    # For HTTP/custom providers
    api_url: str = ""
    api_key: str = ""
    
    # Sync settings
    sync_interval: int = 300  # seconds
    max_sync_batch: int = 100
    compress: bool = True
    
    # Device identification
    device_id: str = ""
    
    def __post_init__(self):
        if not self.device_id:
            import platform
            self.device_id = f"{platform.node()}-{platform.system()}"


@dataclass
class SyncEntry:
    """A memory entry with sync metadata."""
    entry: MemoryEntry
    device_id: str
    version: int = 1
    last_modified: int = 0
    sync_hash: str = ""
    
    def __post_init__(self):
        if self.last_modified == 0:
            self.last_modified = int(time.time())
        if not self.sync_hash:
            self.sync_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute hash of entry content for change detection."""
        raw = json.dumps(self.entry.to_dict(), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        return {
            "entry": self.entry.to_dict(),
            "device_id": self.device_id,
            "version": self.version,
            "last_modified": self.last_modified,
            "sync_hash": self.sync_hash,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> SyncEntry:
        return cls(
            entry=MemoryEntry.from_dict(data["entry"]),
            device_id=data["device_id"],
            version=data.get("version", 1),
            last_modified=data.get("last_modified", 0),
            sync_hash=data.get("sync_hash", ""),
        )


@dataclass
class SyncState:
    """Tracks sync state for a device."""
    device_id: str
    last_sync: int = 0
    entries_synced: int = 0
    entries_received: int = 0
    conflicts_resolved: int = 0
    last_error: str = ""
    
    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "last_sync": self.last_sync,
            "entries_synced": self.entries_synced,
            "entries_received": self.entries_received,
            "conflicts_resolved": self.conflicts_resolved,
            "last_error": self.last_error,
        }


class SyncStore:
    """
    Manages sync state in a local SQLite table.
    
    Extends the main memory store with sync tracking.
    """
    
    SYNC_SCHEMA = """
    CREATE TABLE IF NOT EXISTS sync_state (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    
    CREATE TABLE IF NOT EXISTS sync_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        direction TEXT NOT NULL,  -- 'push' | 'pull'
        entries_count INTEGER DEFAULT 0,
        status TEXT NOT NULL,      -- 'success' | 'error'
        error TEXT,
        duration_ms INTEGER
    );
    
    CREATE INDEX IF NOT EXISTS idx_sync_log_ts ON sync_log(timestamp);
    """
    
    def __init__(self, db_path: str):
        import sqlite3
        from pathlib import Path
        
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SYNC_SCHEMA)
        self._conn.commit()
    
    def get_last_sync_time(self) -> int:
        """Get timestamp of last successful sync."""
        row = self._conn.execute(
            "SELECT value FROM sync_state WHERE key = 'last_sync'"
        ).fetchone()
        return int(row["value"]) if row else 0
    
    def set_last_sync_time(self, timestamp: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sync_state (key, value) VALUES ('last_sync', ?)",
            (str(timestamp),),
        )
        self._conn.commit()
    
    def log_sync(self, direction: str, entries_count: int, status: str, 
                 error: str = "", duration_ms: int = 0) -> None:
        self._conn.execute(
            """INSERT INTO sync_log (timestamp, direction, entries_count, status, error, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (int(time.time()), direction, entries_count, status, error, duration_ms),
        )
        self._conn.commit()
    
    def get_sync_history(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM sync_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    
    def close(self):
        self._conn.close()


class CloudSync:
    """
    Manages cloud synchronization of memory entries.
    
    Supports S3-compatible storage and custom HTTP APIs.
    """
    
    def __init__(self, config: SyncConfig, store):
        self.config = config
        self.store = store
        self.state = SyncState(device_id=config.device_id)
    
    def push(self, since_timestamp: int = 0) -> dict:
        """
        Push local entries to cloud.
        
        Returns sync report.
        """
        if not self.config.enabled:
            return {"status": "disabled"}
        
        start = time.monotonic()
        
        try:
            # Get entries modified since last sync
            conn = self.store._get_conn()
            rows = conn.execute(
                """SELECT * FROM memory_entries 
                   WHERE last_accessed_at > ? 
                   ORDER BY last_accessed_at 
                   LIMIT ?""",
                (since_timestamp, self.config.max_sync_batch),
            ).fetchall()
            
            entries = []
            for row in rows:
                entry = MemoryEntry(
                    id=row["id"],
                    tier=MemoryTier(row["tier"]),
                    content_type=ContentType(row["content_type"]),
                    text=row["text"],
                    session_id=row["session_id"],
                    source_type=SourceType(row["source_type"]),
                    created_at=row["created_at"],
                    last_accessed_at=row["last_accessed_at"],
                    importance=row["importance"],
                )
                sync_entry = SyncEntry(
                    entry=entry,
                    device_id=self.config.device_id,
                    last_modified=row["last_accessed_at"],
                )
                entries.append(sync_entry)
            
            if not entries:
                return {"status": "no_changes", "entries_pushed": 0}
            
            # Serialize
            payload = json.dumps([e.to_dict() for e in entries])
            
            # Upload based on provider
            if self.config.provider == "s3":
                self._push_s3(payload)
            elif self.config.provider == "http":
                self._push_http(payload)
            
            duration = (time.monotonic() - start) * 1000
            self.state.entries_synced += len(entries)
            
            # Log
            self.store.log_sync("push", len(entries), "success", duration_ms=int(duration))
            
            return {
                "status": "success",
                "entries_pushed": len(entries),
                "duration_ms": round(duration, 2),
            }
            
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            error_msg = str(e)
            self.state.last_error = error_msg
            self.store.log_sync("push", 0, "error", error=error_msg, duration_ms=int(duration))
            
            return {
                "status": "error",
                "error": error_msg,
                "duration_ms": round(duration, 2),
            }
    
    def pull(self, since_timestamp: int = 0) -> dict:
        """
        Pull remote entries from cloud.
        
        Returns sync report with entries received.
        """
        if not self.config.enabled:
            return {"status": "disabled"}
        
        start = time.monotonic()
        
        try:
            # Download from cloud
            if self.config.provider == "s3":
                payload = self._pull_s3()
            elif self.config.provider == "http":
                payload = self._pull_http()
            else:
                return {"status": "unsupported_provider"}
            
            if not payload:
                return {"status": "no_changes", "entries_received": 0}
            
            # Deserialize
            remote_entries = [SyncEntry.from_dict(d) for d in json.loads(payload)]
            
            # Merge: latest timestamp wins
            conn = self.store._get_conn()
            merged = 0
            conflicts = 0
            
            for sync_entry in remote_entries:
                # Check if we have this entry
                existing = conn.execute(
                    "SELECT id, last_accessed_at FROM memory_entries WHERE id = ?",
                    (sync_entry.entry.id,),
                ).fetchone()
                
                if existing:
                    # Conflict: use latest
                    if sync_entry.last_modified > existing["last_accessed_at"]:
                        # Remote is newer — update
                        conn.execute(
                            """UPDATE memory_entries 
                               SET text = ?, importance = ?, last_accessed_at = ?
                               WHERE id = ?""",
                            (sync_entry.entry.text, sync_entry.entry.importance,
                             sync_entry.last_modified, sync_entry.entry.id),
                        )
                        merged += 1
                    else:
                        conflicts += 1
                else:
                    # New entry — insert
                    conn.execute(
                        """INSERT INTO memory_entries 
                           (id, tier, content_type, text, session_id, source_type,
                            created_at, last_accessed_at, importance)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (sync_entry.entry.id, sync_entry.entry.tier.value,
                         sync_entry.entry.content_type.value, sync_entry.entry.text,
                         sync_entry.entry.session_id, sync_entry.entry.source_type.value,
                         sync_entry.entry.created_at, sync_entry.last_modified,
                         sync_entry.entry.importance),
                    )
                    merged += 1
            
            conn.commit()
            
            duration = (time.monotonic() - start) * 1000
            self.state.entries_received += merged
            self.state.last_sync = int(time.time())
            
            self.store.log_sync("pull", len(remote_entries), "success", duration_ms=int(duration))
            
            return {
                "status": "success",
                "entries_received": len(remote_entries),
                "merged": merged,
                "conflicts": conflicts,
                "duration_ms": round(duration, 2),
            }
            
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            error_msg = str(e)
            self.state.last_error = error_msg
            self.store.log_sync("pull", 0, "error", error=error_msg, duration_ms=int(duration))
            
            return {
                "status": "error",
                "error": error_msg,
                "duration_ms": round(duration, 2),
            }
    
    def _push_s3(self, payload: str) -> None:
        """Push payload to S3."""
        import boto3
        
        s3 = boto3.client("s3", region_name=self.config.region)
        key = f"{self.config.prefix}entries/{self.config.device_id}.json"
        
        s3.put_object(
            Bucket=self.config.bucket,
            Key=key,
            Body=payload.encode(),
            ContentType="application/json",
        )
    
    def _pull_s3(self) -> Optional[str]:
        """Pull entries from S3."""
        import boto3
        
        s3 = boto3.client("s3", region_name=self.config.region)
        key = f"{self.config.prefix}entries/{self.config.device_id}.json"
        
        try:
            response = s3.get_object(Bucket=self.config.bucket, Key=key)
            return response["Body"].read().decode()
        except s3.exceptions.NoSuchKey:
            return None
    
    def _push_http(self, payload: str) -> None:
        """Push payload to HTTP sync server."""
        import urllib.request
        
        req = urllib.request.Request(
            f"{self.config.api_url}/sync/push",
            data=payload.encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                raise Exception(f"Push failed: {resp.status}")
    
    def _pull_http(self) -> Optional[str]:
        """Pull entries from HTTP sync server."""
        import urllib.request
        
        since = self.store.get_last_sync_time()
        req = urllib.request.Request(
            f"{self.config.api_url}/sync/pull?since={since}&device={self.config.device_id}",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
        )
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 204:
                return None
            return resp.read().decode()
    
    def get_status(self) -> dict:
        """Get current sync status."""
        return {
            "enabled": self.config.enabled,
            "provider": self.config.provider,
            "device_id": self.config.device_id,
            "state": self.state.to_dict(),
            "config": {
                "sync_interval": self.config.sync_interval,
                "max_batch": self.config.max_sync_batch,
            },
        }
