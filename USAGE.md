# Agent Memory — Usage Guide

## Quick Start (3 lines)

```python
from agent_memory import AgentMemory, MemoryConfig

memory = AgentMemory(MemoryConfig(db_path="~/.agent-memory.db"))
memory.start_session("my-session")

# Before every LLM call:
result = memory.check_cache(messages)
if result.hit:
    return result.response  # $0, instant

# After LLM responds:
memory.record_turn(messages, response, model="gpt-4", cost_usd=0.01)
```

---

## Use Case 1: Simple Caching

Wrap your existing LLM call with cache checking:

```python
from agent_memory import AgentMemory, MemoryConfig

memory = AgentMemory(MemoryConfig(db_path="~/.agent-memory.db"))
memory.start_session("chat-123")

def ask_llm(messages):
    # Check cache first
    result = memory.check_cache(messages)
    if result.hit:
        print(f"Cache hit! ({result.hit_type.value})")
        return result.response
    
    # Cache miss — call real LLM
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    ).choices[0].message.content
    
    # Store in cache
    memory.record_turn(messages, response, model="gpt-4o", cost_usd=0.02)
    
    return response
```

---

## Use Case 2: Memory-Augmented Agent

Your agent gets smarter over time — it remembers previous answers:

```python
from agent_memory import AgentMemory, MemoryConfig, TaskType

memory = AgentMemory(MemoryConfig(db_path="~/.agent-memory.db"))
memory.start_session("dev-session")

def agent_turn(user_message):
    messages = [{"role": "user", "content": user_message}]
    
    # Get relevant memories and prepend to prompt
    augmented = memory.get_augmented_messages(
        messages, 
        task_type=TaskType.DEBUG,
        budget_tokens=2000,  # Cap memory at 2K tokens
    )
    
    response = call_llm(augmented)
    memory.record_turn(messages, response, model="gpt-4o", task_type=TaskType.DEBUG)
    
    return response

# First time: "What causes segmentation faults?"
# → Full LLM call, stores answer

# Later: "What did we learn about segfaults?"
# → Returns cached insight, $0 cost
```

---

## Use Case 3: Tool Result Caching

Don't re-run expensive tools:

```python
# Before running a tool:
cached = memory.check_tool_cache("web_fetch", {"url": "https://api.example.com/data"})
if cached:
    result = cached  # Skip the actual HTTP call
else:
    result = requests.get("https://api.example.com/data").text
    memory.record_tool_result("web_fetch", {"url": "https://api.example.com/data"}, result, ttl=1800)
```

---

## Use Case 4: Cost-Aware Model Routing

Let the decision engine pick the right model:

```python
route = memory.route_to_model(
    task_description="What is 2+2?",
    task_type=TaskType.TRIAGE,
)

if route.route == "cheap":
    response = call_cheap_model(messages)  # gpt-4o-mini
elif route.route == "standard":
    response = call_standard_model(messages)  # gpt-4o
elif route.route == "advanced":
    response = call_advanced_model(messages)  # gpt-4

print(f"Routed to: {route.route} — {route.reason}")
```

---

## Use Case 5: OpenAI Proxy (Zero Code Changes)

Run the proxy and point your client at it:

```bash
# Start the proxy
python -m agent_memory.proxy --port 8080 --upstream https://api.openai.com/v1

# Or with OpenAI-compatible providers:
python -m agent_memory.proxy --port 8080 --upstream https://api.deepseek.com/v1
python -m agent_memory.proxy --port 8080 --upstream https://openrouter.ai/api/v1
```

```python
# Change only the base_url
import openai
openai.base_url = "http://localhost:8080/v1"
openai.api_key = "your-real-key"  # Passed through to upstream

# Everything else stays the same — caching is transparent
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
# Check headers: response._headers['x-cache'] → 'HIT' or 'MISS'
```

---

## Use Case 6: OpenClaw Plugin

Add to your OpenClaw config:

```yaml
# ~/.openclaw/config.yaml
plugins:
  - name: agent-memory
    path: /path/to/agent-memory
    config:
      db_path: ~/.openclaw/agent-memory/cache.db
      budget_usd: 10.0
      embedding_provider: local  # local | hash | openai
      hot_max_tokens: 4000
```

Automatic hooks:
- `pre_agent_turn` — checks cache, returns cached response if hit
- `post_agent_turn` — stores response, updates cost tracking
- `pre_tool_call` — checks tool cache
- `post_tool_call` — stores tool result

---

## Use Case 7: LangChain Integration

```python
from langchain.llms import OpenAI
from agent_memory.adapters import LangChainMemory, LangChainToolWrapper

# Wrap LLM
base_llm = OpenAI(model="gpt-4")
llm = LangChainMemory(base_llm, db_path="~/.langchain-memory.db")

# Use normally — caching is automatic
response = llm.invoke("What is Python?")

# Wrap tools
from langchain.tools import WikipediaQueryRun
wiki_tool = WikipediaQueryRun()
cached_wiki = LangChainToolWrapper(wiki_tool, memory=memory)

result = cached_wiki.run("Python programming")  # Cached
```

---

## Use Case 8: CLI Management

```bash
# View stats
python -m agent_memory.cli stats
# === Agent Memory Stats ===
# Entries: 15234
# Cache hit rate: 62.3%
# Cost saved: $14.27

# Search memory
python -m agent_memory.cli search "python debugging tips" --top-k 10

# Run compaction
python -m agent_memory.cli compact

# Export to JSON
python -m agent_memory.cli export --format json > memory-backup.json

# Database info
python -m agent_memory.cli info
```

---

## Use Case 9: Multi-Device Sync

```python
from agent_memory.sync import CloudSync, SyncConfig, SyncStore

# Configure sync (S3)
config = SyncConfig(
    enabled=True,
    provider="s3",
    bucket="my-agent-memory",
    region="us-east-1",
    device_id="laptop-1",
)

store = SyncStore("~/.openclaw/agent-memory/cache.db")
sync = CloudSync(config, store)

# Push local changes to cloud
sync.push()

# Pull remote changes from other devices
result = sync.pull()
print(f"Received {result['entries_received']} entries")

# Check sync status
print(sync.get_status())
```

---

## Configuration Reference

```python
from agent_memory import MemoryConfig

config = MemoryConfig(
    # Storage
    db_path="~/.agent-memory/cache.db",
    max_entries=100_000,
    
    # Hot memory (session)
    hot_max_turns=50,
    hot_max_tokens=4000,
    
    # Embedding (local | hash | openai)
    embedding_provider="local",
    
    # Cache thresholds
    hit_threshold=0.95,      # Semantic similarity for exact hit
    gray_zone_low=0.80,      # Below this = clear miss
    
    # Budget
    budget_usd=10.0,
    cheap_models=["gpt-4o-mini"],
    standard_models=["gpt-4o"],
    advanced_models=["gpt-4"],
)

memory = AgentMemory(config)
```

---

## How Caching Works (What Happens Under the Hood)

```
User: "What is recursion?"
         │
         ▼
┌─ Layer 0: Exact Match ─────────────────────────┐
│  Hash(normalized_messages) → SQLite lookup      │
│  Result: MISS (first time)                      │
└─────────────────────────────────────────────────┘
         │ MISS
         ▼
┌─ Layer 1: Semantic Match ──────────────────────┐
│  Embed("What is recursion?") → FAISS search    │
│  Result: MISS (no similar stored)               │
└─────────────────────────────────────────────────┘
         │ MISS
         ▼
┌─ Layer 2: Full Inference ──────────────────────┐
│  Call gpt-4o → "Recursion is when a function   │
│  calls itself..."                               │
│  Store response in cache for future             │
└─────────────────────────────────────────────────┘

---
Later:
User: "What is recursion?"  (same question, different system prompt stats)
         │
         ▼
┌─ Layer 0: Exact Match ─────────────────────────┐
│  System prompt: "Tokens used: 500" → [VOLATILE]│
│  Normalized = same as before → MATCH!           │
│  Return cached response. $0 cost, <1ms.         │
└─────────────────────────────────────────────────┘
```
