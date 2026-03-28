/**
 * Agent Memory — OpenClaw Context Engine Plugin
 * 
 * Node.js wrapper for the Python agent-memory system.
 * Provides caching, memory retrieval, and tool caching for OpenClaw agents.
 * 
 * Requires: agent-memory Python package (pip install -e .)
 */

const { execSync } = require('child_process');
const path = require('path');

// Per-session memory instances
const sessionMemory = new Map();

// Plugin config
let config = {
  db_path: '~/.openclaw/agent-memory/cache.db',
  embedding_provider: 'hash',
  budget_usd: 10.0,
  debug: false,
};

// Ensure Python package is importable
const AGENT_MEMORY_PATH = path.resolve(__dirname);
process.env.PYTHONPATH = (process.env.PYTHONPATH || '') + ':' + AGENT_MEMORY_PATH;

function log(...args) {
  if (config.debug) {
    console.log('[agent-memory]', ...args);
  }
}

function runPython(code) {
  try {
    const result = execSync(`python3 -c '${code.replace(/'/g, "'\\''")}'`, {
      encoding: 'utf-8',
      timeout: 30000,
      env: { ...process.env, PYTHONPATH: process.env.PYTHONPATH },
    });
    return result.trim();
  } catch (err) {
    log('Python error:', err.message);
    return null;
  }
}

function getSessionState(sessionId) {
  if (!sessionMemory.has(sessionId)) {
    const dbPath = config.db_path.replace('~', process.env.HOME || '/root');
    
    // Initialize Python memory
    const pyCode = `
from agent_memory import AgentMemory, MemoryConfig
import json

memory = AgentMemory(MemoryConfig(
    db_path="${dbPath}",
    embedding_provider="${config.embedding_provider}",
    budget_usd=${config.budget_usd},
))
memory.start_session("${sessionId}")
print("READY")
`;
    const result = runPython(pyCode);
    if (result === 'READY') {
      sessionMemory.set(sessionId, { active: true });
      log(`Session initialized: ${sessionId}`);
    }
  }
  return sessionMemory.get(sessionId);
}

/**
 * Plugin hooks for OpenClaw context engine
 */
const plugin = {
  /**
   * Initialize the plugin with config
   */
  init(pluginConfig) {
    config = { ...config, ...pluginConfig };
    log('Initialized with config:', JSON.stringify(config));
  },

  /**
   * Pre-agent-turn hook: Check cache before LLM call
   * Returns cached response if found, null otherwise
   */
  async preAgentTurn(context) {
    const sessionId = context.sessionId || 'default';
    getSessionState(sessionId);
    
    const messages = context.messages || [];
    if (messages.length === 0) return null;
    
    // Hash the normalized messages for exact match
    const messagesJson = JSON.stringify(messages);
    const dbPath = config.db_path.replace('~', process.env.HOME || '/root');
    
    const pyCode = `
from agent_memory import AgentMemory, MemoryConfig
from agent_memory.normalizer import normalize_messages
import hashlib, json

messages = json.loads("""${messagesJson.replace(/"/g, '\\"')}""")
memory = AgentMemory(MemoryConfig(db_path="${dbPath}", embedding_provider="${config.embedding_provider}"))
memory.start_session("${sessionId}")

result = memory.check_cache(messages)
if result.hit:
    print(json.dumps({"hit": True, "type": result.hit_type.value, "response": result.response, "saved": result.cost_saved}))
else:
    print(json.dumps({"hit": False}))
memory.close()
`;
    
    const result = runPython(pyCode);
    if (result) {
      try {
        const parsed = JSON.parse(result);
        if (parsed.hit) {
          log(`Cache hit (${parsed.type}) — saved $${parsed.saved}`);
          return {
            response: parsed.response,
            cacheHit: true,
            cacheHitType: parsed.type,
          };
        }
      } catch (e) {
        log('Parse error:', e.message);
      }
    }
    
    return null;
  },

  /**
   * Post-agent-turn hook: Store response in cache
   */
  async postAgentTurn(context) {
    if (context.cacheHit) return; // Already cached
    
    const sessionId = context.sessionId || 'default';
    const messages = context.messages || [];
    const response = context.response || '';
    const model = context.model || 'unknown';
    const cost = context.costUsd || 0;
    
    if (!response || messages.length === 0) return;
    
    const messagesJson = JSON.stringify(messages);
    const responseEscaped = response.replace(/"/g, '\\"').replace(/\n/g, '\\n');
    const dbPath = config.db_path.replace('~', process.env.HOME || '/root');
    
    const pyCode = `
from agent_memory import AgentMemory, MemoryConfig
import json

messages = json.loads("""${messagesJson.replace(/"/g, '\\"')}""")
memory = AgentMemory(MemoryConfig(db_path="${dbPath}", embedding_provider="${config.embedding_provider}"))
memory.start_session("${sessionId}")

memory.record_turn(messages, """${responseEscaped}""", model="${model}", cost_usd=${cost})
print("STORED")
memory.close()
`;
    
    runPython(pyCode);
    log(`Stored response (${model}, ${response.length} chars)`);
  },

  /**
   * Pre-tool-call hook: Check tool cache
   */
  async preToolCall(context) {
    const toolName = context.toolName || '';
    const toolArgs = context.args || {};
    
    if (!toolName) return null;
    
    const dbPath = config.db_path.replace('~', process.env.HOME || '/root');
    const argsJson = JSON.stringify(toolArgs).replace(/"/g, '\\"');
    
    const pyCode = `
from agent_memory import AgentMemory, MemoryConfig
import json

memory = AgentMemory(MemoryConfig(db_path="${dbPath}", embedding_provider="${config.embedding_provider}"))
memory.start_session("tool-session")

result = memory.check_tool_cache("${toolName}", json.loads("""${argsJson}"""))
if result:
    print(json.dumps({"cached": True, "result": result}))
else:
    print(json.dumps({"cached": False}))
memory.close()
`;
    
    const result = runPython(pyCode);
    if (result) {
      try {
        const parsed = JSON.parse(result);
        if (parsed.cached) {
          log(`Tool cache hit: ${toolName}`);
          return { result: parsed.result, cacheHit: true };
        }
      } catch (e) {
        log('Tool cache parse error:', e.message);
      }
    }
    
    return null;
  },

  /**
   * Post-tool-call hook: Store tool result
   */
  async postToolCall(context) {
    const toolName = context.toolName || '';
    const toolArgs = context.args || {};
    const result = context.result || '';
    const ttl = context.ttl || 3600;
    
    if (!toolName || !result) return;
    
    const dbPath = config.db_path.replace('~', process.env.HOME || '/root');
    const argsJson = JSON.stringify(toolArgs).replace(/"/g, '\\"');
    const resultEscaped = result.replace(/"/g, '\\"').substring(0, 10000); // Cap at 10K chars
    
    const pyCode = `
from agent_memory import AgentMemory, MemoryConfig
import json

memory = AgentMemory(MemoryConfig(db_path="${dbPath}", embedding_provider="${config.embedding_provider}"))
memory.start_session("tool-session")

memory.record_tool_result("${toolName}", json.loads("""${argsJson}"""), """${resultEscaped}""", ttl=${ttl})
print("STORED")
memory.close()
`;
    
    runPython(pyCode);
    log(`Stored tool result: ${toolName}`);
  },

  /**
   * Get memory context for prompt augmentation
   */
  async getContext(sessionId, query, budgetTokens = 2000) {
    const dbPath = config.db_path.replace('~', process.env.HOME || '/root');
    const queryEscaped = query.replace(/"/g, '\\"');
    
    const pyCode = `
from agent_memory import AgentMemory, MemoryConfig
import json

memory = AgentMemory(MemoryConfig(db_path="${dbPath}", embedding_provider="${config.embedding_provider}"))
memory.start_session("${sessionId}")

messages = [{"role": "user", "content": """${queryEscaped}"""}]
result = memory.retrieve(messages, budget_tokens=${budgetTokens})

print(json.dumps({
    "entries": len(result.entries),
    "tokens": result.total_tokens,
    "context": result.to_context_string(),
    "sources": result.sources,
    "time_ms": result.retrieval_time_ms,
}))
memory.close()
`;
    
    const result = runPython(pyCode);
    if (result) {
      try {
        return JSON.parse(result);
      } catch (e) {
        return { entries: 0, tokens: 0, context: '', sources: {} };
      }
    }
    return { entries: 0, tokens: 0, context: '', sources: {} };
  },

  /**
   * Get plugin stats
   */
  async getStats() {
    const dbPath = config.db_path.replace('~', process.env.HOME || '/root');
    
    const pyCode = `
from agent_memory import SQLiteStore

store = SQLiteStore(db_path="${dbPath}")
stats = store.get_stats()

print(json.dumps({
    "total_entries": stats.total_entries,
    "exact_hits": stats.exact_hits,
    "semantic_hits": stats.semantic_hits,
    "misses": stats.misses,
    "hit_rate": f"{stats.hit_rate:.1%}",
    "cost_saved": round(stats.total_cost_saved, 4),
}))
store.close()
`;
    
    const result = runPython(pyCode);
    if (result) {
      try {
        return JSON.parse(result);
      } catch (e) {
        return { error: 'Failed to parse stats' };
      }
    }
    return { error: 'No result' };
  },
};

module.exports = plugin;
