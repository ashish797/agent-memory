#!/usr/bin/env python3
"""
Agent Memory CLI — Manage and inspect the agent memory system.

Usage:
    python -m agent_memory.cli stats [--db PATH]
    python -m agent_memory.cli compact [--db PATH]
    python -m agent_memory.cli search QUERY [--db PATH] [--top-k 5]
    python -m agent_memory.cli clear [--db PATH] [--hot | --warm | --all]
    python -m agent_memory.cli export [--db PATH] [--format json|csv]
    python -m agent_memory.cli info [--db PATH]
"""

import argparse
import json
import sys
from pathlib import Path


def cmd_stats(args):
    """Show memory system statistics."""
    from agent_memory import SQLiteStore
    
    store = SQLiteStore(db_path=args.db)
    stats = store.get_stats()
    session_cost = store.get_session_cost(args.session or "")
    
    print("=== Agent Memory Stats ===\n")
    print(f"Database: {args.db}")
    print(f"\nEntries:")
    print(f"  Total: {stats.total_entries}")
    print(f"  Warm:  {stats.warm_entries}")
    print(f"  Cold:  {stats.cold_entries}")
    
    print(f"\nCache Performance:")
    print(f"  Requests:     {stats.total_requests}")
    print(f"  Exact hits:   {stats.exact_hits}")
    print(f"  Semantic hits: {stats.semantic_hits}")
    print(f"  Triage hits:  {stats.triage_hits}")
    print(f"  Misses:       {stats.misses}")
    print(f"  Hit rate:     {stats.hit_rate:.1%}")
    print(f"  Cost saved:   ${stats.total_cost_saved:.4f}")
    
    if session_cost["total_calls"] > 0:
        print(f"\nSession ({args.session or 'all'}):")
        print(f"  Calls:    {session_cost['total_calls']}")
        print(f"  Cost:     ${session_cost['total_cost']:.4f}")
        print(f"  Input:    {session_cost['total_input_tokens']:,} tokens")
        print(f"  Output:   {session_cost['total_output_tokens']:,} tokens")
    
    store.close()


def cmd_compact(args):
    """Run memory compaction."""
    from agent_memory import SQLiteStore
    
    store = SQLiteStore(db_path=args.db)
    
    print("Running compaction...")
    report = store.compact()
    
    print(f"\nCompaction complete:")
    print(f"  Expired entries removed: {report['expired_removed']}")
    print(f"  Tool cache cleaned: {report['tool_cache_removed']}")
    
    store.close()


def cmd_search(args):
    """Search memory entries."""
    from agent_memory import SQLiteStore
    from agent_memory.embedding import get_embedding_engine
    
    store = SQLiteStore(db_path=args.db)
    embedding = get_embedding_engine(provider="hash")  # Fast for search
    
    query_embedding = embedding.embed(args.query)
    results = store.search_similar(query_embedding, top_k=args.top_k, min_similarity=0.0)
    
    print(f"=== Search: '{args.query}' ===\n")
    
    if not results:
        print("No results found.")
        return
    
    for i, (entry, similarity) in enumerate(results, 1):
        print(f"{i}. [similarity: {similarity:.3f}] [{entry.content_type.value}]")
        print(f"   {entry.text[:200]}{'...' if len(entry.text) > 200 else ''}")
        print(f"   Created: {entry.created_at} | Hits: {entry.access_count} | Importance: {entry.importance:.2f}")
        print()
    
    store.close()


def cmd_clear(args):
    """Clear memory entries."""
    from agent_memory import SQLiteStore
    
    store = SQLiteStore(db_path=args.db)
    conn = store._get_conn()
    
    if args.warm or args.all:
        count = conn.execute("DELETE FROM memory_entries").rowcount
        print(f"Cleared {count} warm memory entries")
    
    if args.tools or args.all:
        count = conn.execute("DELETE FROM tool_cache").rowcount
        print(f"Cleared {count} tool cache entries")
    
    if args.stats or args.all:
        count = conn.execute("DELETE FROM cost_events").rowcount
        print(f"Cleared {count} cost events")
    
    if args.all:
        conn.execute("UPDATE cache_stats SET value = 0")
        print("Reset all stats")
    
    conn.commit()
    store.close()


def cmd_info(args):
    """Show database info."""
    from agent_memory import SQLiteStore
    
    db_path = Path(args.db).expanduser()
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return
    
    size_mb = db_path.stat().st_size / (1024 * 1024)
    
    store = SQLiteStore(db_path=args.db)
    conn = store._get_conn()
    
    # Table sizes
    tables = {}
    for table in ["memory_entries", "tool_cache", "cost_events", "cache_stats"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        tables[table] = count
    
    print(f"=== Agent Memory Info ===\n")
    print(f"Path:     {db_path}")
    print(f"Size:     {size_mb:.2f} MB")
    print(f"\nTables:")
    for table, count in tables.items():
        print(f"  {table}: {count:,} rows")
    
    store.close()


def cmd_export(args):
    """Export memory entries."""
    from agent_memory import SQLiteStore
    
    store = SQLiteStore(db_path=args.db)
    conn = store._get_conn()
    
    rows = conn.execute(
        "SELECT id, tier, content_type, text, session_id, created_at, importance, access_count "
        "FROM memory_entries ORDER BY created_at DESC"
    ).fetchall()
    
    entries = [
        {
            "id": r["id"],
            "tier": r["tier"],
            "content_type": r["content_type"],
            "text": r["text"][:500],  # Truncate for export
            "session_id": r["session_id"],
            "created_at": r["created_at"],
            "importance": r["importance"],
            "access_count": r["access_count"],
        }
        for r in rows
    ]
    
    if args.format == "json":
        print(json.dumps(entries, indent=2))
    elif args.format == "csv":
        print("id,tier,content_type,session_id,created_at,importance,access_count,text")
        for e in entries:
            text = e["text"].replace('"', '""')
            print(f'{e["id"]},{e["tier"]},{e["content_type"]},{e["session_id"]},{e["created_at"]},{e["importance"]},{e["access_count"]},"{text}"')
    
    store.close()


def main():
    parser = argparse.ArgumentParser(description="Agent Memory CLI")
    parser.add_argument("--db", default="~/.openclaw/agent-memory/cache.db", help="Database path")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # stats
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--session", default="", help="Session ID")
    
    # compact
    subparsers.add_parser("compact", help="Run compaction")
    
    # search
    search_parser = subparsers.add_parser("search", help="Search memory")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    
    # clear
    clear_parser = subparsers.add_parser("clear", help="Clear memory")
    clear_parser.add_argument("--warm", action="store_true", help="Clear warm memory")
    clear_parser.add_argument("--tools", action="store_true", help="Clear tool cache")
    clear_parser.add_argument("--stats", action="store_true", help="Clear stats")
    clear_parser.add_argument("--all", action="store_true", help="Clear everything")
    
    # info
    subparsers.add_parser("info", help="Database info")
    
    # export
    export_parser = subparsers.add_parser("export", help="Export memory")
    export_parser.add_argument("--format", choices=["json", "csv"], default="json", help="Export format")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    commands = {
        "stats": cmd_stats,
        "compact": cmd_compact,
        "search": cmd_search,
        "clear": cmd_clear,
        "info": cmd_info,
        "export": cmd_export,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
