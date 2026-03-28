"""
Framework Adapters — Integration with popular agent frameworks.

Supported:
- LangChain: MemoryLLMWrapper, caching LLM class
- Generic: Any LLM wrapper with before/after hooks
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# LangChain Adapter
# ═══════════════════════════════════════════════════════════════════

class LangChainMemory:
    """
    LangChain integration for Agent Memory.
    
    Wraps any LangChain LLM with caching, memory retrieval, and cost tracking.
    
    Usage:
        from langchain.llms import OpenAI
        from agent_memory.adapters import LangChainMemory
        
        base_llm = OpenAI(model="gpt-4")
        llm = LangChainMemory(base_llm, db_path="~/.agent-memory.db")
        
        # Normal LangChain usage — caching is transparent
        response = llm.invoke("What is Python?")
        response2 = llm.invoke("What is Python?")  # Cache hit!
    """
    
    def __init__(
        self,
        llm: Any,
        db_path: str = "~/.openclaw/agent-memory/langchain.db",
        embedding_provider: str = "hash",
        enable_memory: bool = True,
        budget_usd: float = 10.0,
    ):
        self.llm = llm
        self._memory = None
        self._db_path = db_path
        self._embedding_provider = embedding_provider
        self._enable_memory = enable_memory
        self._budget_usd = budget_usd
    
    @property
    def memory(self):
        if self._memory is None:
            from agent_memory.memory import AgentMemory, MemoryConfig
            self._memory = AgentMemory(MemoryConfig(
                db_path=self._db_path,
                embedding_provider=self._embedding_provider,
                budget_usd=self._budget_usd,
            ))
            self._memory.start_session("langchain")
        return self._memory
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke the LLM with caching."""
        messages = [{"role": "user", "content": prompt}]
        
        # Check cache
        if self._enable_memory:
            result = self.memory.check_cache(messages)
            if result.hit:
                logger.debug(f"LangChain cache hit")
                return result.response
        
        # Cache miss — call base LLM
        start = time.monotonic()
        response = self.llm.invoke(prompt, **kwargs)
        latency = time.monotonic() - start
        
        # Store in cache
        if self._enable_memory and response:
            self.memory.record_turn(
                messages=messages,
                response=response,
                model=getattr(self.llm, 'model_name', 'unknown'),
                cost_usd=0.0,  # LangChain doesn't expose cost directly
            )
        
        return response
    
    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """Async invoke with caching."""
        messages = [{"role": "user", "content": prompt}]
        
        if self._enable_memory:
            result = self.memory.check_cache(messages)
            if result.hit:
                return result.response
        
        response = await self.llm.ainvoke(prompt, **kwargs)
        
        if self._enable_memory and response:
            self.memory.record_turn(messages=messages, response=response, model="unknown")
        
        return response
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped LLM."""
        return getattr(self.llm, name)


class LangChainToolWrapper:
    """
    Wraps a LangChain tool with result caching.
    
    Usage:
        from langchain.tools import WikipediaQueryRun
        from agent_memory.adapters import LangChainToolWrapper
        
        tool = WikipediaQueryRun()
        cached_tool = LangChainToolWrapper(tool, memory=agent_memory)
        
        result = cached_tool.run("Python")  # Cached
        result2 = cached_tool.run("Python")  # Cache hit!
    """
    
    def __init__(self, tool: Any, memory: Any = None, ttl: int = 3600):
        self.tool = tool
        self.memory = memory
        self.ttl = ttl
    
    @property
    def name(self) -> str:
        return getattr(self.tool, 'name', 'unknown')
    
    @property
    def description(self) -> str:
        return getattr(self.tool, 'description', '')
    
    def run(self, tool_input: Union[str, dict], **kwargs) -> str:
        """Run the tool with caching."""
        args = {"input": tool_input} if isinstance(tool_input, str) else tool_input
        
        if self.memory:
            cached = self.memory.check_tool_cache(self.name, args)
            if cached:
                logger.debug(f"Tool cache hit: {self.name}")
                return cached
        
        result = self.tool.run(tool_input, **kwargs)
        
        if self.memory:
            self.memory.record_tool_result(self.name, args, str(result), ttl=self.ttl)
        
        return result
    
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# Generic Adapter
# ═══════════════════════════════════════════════════════════════════

class CachedLLM:
    """
    Generic LLM wrapper with memory caching.
    
    Works with any LLM callable that takes a prompt and returns a string.
    
    Usage:
        def my_llm(prompt: str) -> str:
            return openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content
        
        cached = CachedLLM(my_llm, model="gpt-4")
        response = cached("What is Python?")  # Miss
        response2 = cached("What is Python?")  # Hit
    """
    
    def __init__(
        self,
        llm_callable: Callable[[str], str],
        model: str = "unknown",
        db_path: str = "~/.openclaw/agent-memory/cached-llm.db",
        embedding_provider: str = "hash",
    ):
        self.llm_callable = llm_callable
        self.model = model
        self._db_path = db_path
        self._embedding = embedding_provider
        self._memory = None
    
    @property
    def memory(self):
        if self._memory is None:
            from agent_memory.memory import AgentMemory, MemoryConfig
            self._memory = AgentMemory(MemoryConfig(
                db_path=self._db_path,
                embedding_provider=self._embedding,
            ))
            self._memory.start_session("cached-llm")
        return self._memory
    
    def __call__(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        
        result = self.memory.check_cache(messages, model=self.model)
        if result.hit:
            return result.response
        
        response = self.llm_callable(prompt, **kwargs)
        
        if response:
            self.memory.record_turn(
                messages=messages,
                response=response,
                model=self.model,
            )
        
        return response
    
    def close(self):
        if self._memory:
            self._memory.close()


class AgentMemoryTools:
    """
    Utility tools for agent frameworks.
    
    Provides callable tools that agents can use:
    - search_memory: Search past conversations
    - save_memory: Explicitly save information
    - get_stats: Get memory/cache statistics
    """
    
    def __init__(self, memory: Any):
        self.memory = memory
    
    def search_memory(self, query: str, limit: int = 5) -> str:
        """Search memory for relevant entries."""
        messages = [{"role": "user", "content": query}]
        result = self.memory.retrieve(messages)
        
        if not result.entries:
            return "No relevant memories found."
        
        parts = []
        for i, scored in enumerate(result.entries[:limit], 1):
            entry = scored.entry
            parts.append(f"{i}. [{entry.content_type.value}] {entry.text[:200]}")
        
        return "\n".join(parts)
    
    def save_memory(self, content: str, category: str = "insight") -> str:
        """Explicitly save information to memory."""
        from agent_memory.types import ContentType
        
        content_type_map = {
            "insight": ContentType.INSIGHT,
            "decision": ContentType.DECISION,
            "preference": ContentType.PREFERENCE,
        }
        
        content_type = content_type_map.get(category, ContentType.INSIGHT)
        
        entry_id = self.memory.cache.store(
            messages=[{"role": "user", "content": content}],
            response=content,
            content_type=content_type,
        )
        
        return f"Saved to memory (ID: {entry_id[:8]}...)"
    
    def get_stats(self) -> str:
        """Get memory system statistics."""
        stats = self.memory.get_stats()
        
        warm = stats.get("warm_memory", {})
        session = stats.get("session", {})
        
        return (
            f"Session: {session.get('turns', 0)} turns, "
            f"${session.get('cost_usd', 0):.4f} spent\n"
            f"Cache: {warm.get('total_requests', 0)} requests, "
            f"{warm.get('hit_rate', '0%')} hit rate\n"
            f"Entries: {warm.get('total_entries', 0)} stored"
        )
    
    def as_tool_list(self) -> list[dict]:
        """Return tools in a format compatible with tool-calling LLMs."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Search past conversations and saved information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "description": "Max results", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "save_memory",
                    "description": "Save information for future reference",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "Content to save"},
                            "category": {"type": "string", "enum": ["insight", "decision", "preference"]},
                        },
                        "required": ["content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_stats",
                    "description": "Get memory system statistics",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
