# agent-memory

**Agent Memory System** — Memory hierarchy with cost-aware routing for LLM agents.

![Tests](https://img.shields.io/badge/tests-92/92-brightgreen)

## The Problem

LLM agents waste 99% of compute on redundant context. Every turn re-sends the full system prompt, conversation history, and tool definitions. Tool calls break cache. Memory grows unbounded. Costs spiral.

## The Solution

A three-tier memory system with intelligent routing:

```
┌──────────────────────────────────────────┐
│  HOT  (Session)  — Ring buffer, <1ms     │
│  WARM (Semantic) — SQLite + FAISS, ~5ms  │
│  COLD (Archival) — Compressed files      │
│                                          │
│  + Decision Engine (importance, fresh)   │
│  + Cost-Aware Routing (cheap/standard)   │
│  + Prompt Normalization (volatile strip) │
└──────────────────────────────────────────┘
```

## Quick Start

```python
from agent_memory import AgentMemory, MemoryConfig

# Initialize
memory = AgentMemory(MemoryConfig(db_path="~/.agent-memory.db"))
memory.start_session("my-session")

# Check cache before LLM call
result = memory.check_cache(messages)
if result.hit:
    response = result.response  # Skip LLM entirely
else:
    # Make LLM call...
    response = call_llm(messages)
    
    # Record the turn (auto-caches, updates hot memory)
    memory.record_turn(messages, response, model="gpt-4", cost_usd=0.01)

# Get memory-augmented context for next turn
augmented = memory.get_augmented_messages(messages)

# Clean up
memory.close()
```

## Features

### 4-Layer Cache
1. **Exact match** (SHA-256 hash) — $0, <1ms
2. **Semantic match** (FAISS + embeddings) — $0, ~5ms  
3. **Triage gate** (cheap model verification) — ~$0.00015
4. **Full inference** (normal LLM call)

### Prompt Normalization
Strips volatile patterns (timestamps, usage stats, cache counters) before hashing. Same question with different system prompt stats = cache hit.

### Tool Result Cache
Caches tool outputs by args hash. Tool-specific TTLs:
- `web_fetch`: 30 min
- `read_file`: 5 min
- `execute_code`: never cached

### Decision Engine
- **Importance scoring** — 6 factors (content type, access frequency, references, cost, age, user signal)
- **Freshness checking** — Per-content-type TTLs, tool-specific expiry
- **Model routing** — 4 routes: cached → cheap → standard → advanced. Budget-aware.
- **Compaction** — Auto-suggests merge/drop actions

### Retrieval Engine
Multi-factor relevance scoring: 40% semantic + 20% recency + 15% frequency + 15% task match + 10% importance. Merges hot + warm tiers with budget capping.

## Architecture

```
agent_memory/
├── types.py         # MemoryEntry, enums, stats
├── normalizer.py    # Prompt normalization, volatile stripping
├── embedding.py     # Local (MiniLM), Hash, OpenAI embeddings
├── store.py         # SQLite + FAISS warm store, tool cache, cost tracking
├── hot.py           # Hot memory ring buffer (turns, tools, tasks, prefs)
├── cache.py         # Response cache (exact + semantic layers)
├── decision.py      # Importance, freshness, routing, remember decisions
├── retrieval.py     # Multi-tier retrieval engine
├── memory.py        # Main SDK (AgentMemory class)
├── plugin.py        # OpenClaw plugin integration
└── cli.py           # CLI management tool
```

## CLI

```bash
# View stats
python -m agent_memory.cli stats --db ~/.openclaw/agent-memory/cache.db

# Search memory
python -m agent_memory.cli search "what is recursion" --top-k 5

# Run compaction
python -m agent_memory.cli compact

# Export data
python -m agent_memory.cli export --format json > memory.json

# Database info
python -m agent_memory.cli info
```

## OpenClaw Integration

```yaml
# OpenClaw plugin config
plugins:
  - name: agent-memory
    path: /path/to/agent-memory
    config:
      db_path: ~/.openclaw/agent-memory/cache.db
      budget_usd: 10.0
      embedding_provider: local
```

Hooks: `pre_agent_turn`, `post_agent_turn`, `pre_tool_call`, `post_tool_call`, `on_session_start`, `on_session_end`

## Test Coverage

```
tests/test_normalizer.py   (9 tests)  — Prompt normalization
tests/test_hot.py          (12 tests) — Hot memory
tests/test_store.py        (12 tests) — SQLite + FAISS store
tests/test_cache.py        (6 tests)  — Response cache
tests/test_decision.py     (28 tests) — Decision engine
tests/test_retrieval.py    (10 tests) — Retrieval engine
tests/test_memory.py       (14 tests) — Main SDK
tests/test_integration.py  (1 test)   — Full agent flow
──────────────────────────────────────
Total:                      92 tests   ALL PASSING ✅
```

## Status

- ✅ Phase 1: Foundation (store, hot memory, exact/semantic cache)
- ✅ Phase 2: Intelligence (decision engine, retrieval engine)
- ✅ Phase 3: Production (SDK, OpenClaw plugin, CLI, monitoring)
- 📋 Phase 4: Expansion (proxy mode, cloud option, framework SDKs)

## License

MIT
