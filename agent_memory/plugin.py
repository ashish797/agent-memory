#!/usr/bin/env python3
"""
OpenClaw Plugin for Agent Memory System.

Hooks into OpenClaw's agent loop to provide:
- Client-side response caching (exact + semantic)
- Tool result caching
- Multi-tier memory retrieval
- Cost-aware model routing
- Budget tracking

Usage:
    # In OpenClaw plugin configuration:
    plugins:
      - name: agent-memory
        path: /path/to/agent-memory
        config:
          db_path: ~/.openclaw/agent-memory/cache.db
          budget_usd: 10.0
          embedding_provider: local
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class PluginConfig:
    """OpenClaw plugin configuration."""
    db_path: str = "~/.openclaw/agent-memory/cache.db"
    budget_usd: float = 10.0
    embedding_provider: str = "local"
    embedding_model: Optional[str] = None
    hot_max_tokens: int = 4000
    enabled: bool = True


class OpenClawPlugin:
    """
    OpenClaw plugin for agent memory.
    
    Integrates with OpenClaw's hook system:
    - pre_agent_turn: Check cache, retrieve memory context
    - post_agent_turn: Store response, update cost tracking
    - pre_tool_call: Check tool cache
    - post_tool_call: Store tool result
    """
    
    def __init__(self, config: Optional[PluginConfig] = None):
        self.config = config or PluginConfig()
        self._memory = None  # Lazy init
        self._enabled = self.config.enabled
    
    @property
    def memory(self):
        """Lazy-initialize the memory system."""
        if self._memory is None:
            from agent_memory.memory import AgentMemory, MemoryConfig
            
            mem_config = MemoryConfig(
                db_path=self.config.db_path,
                budget_usd=self.config.budget_usd,
                embedding_provider=self.config.embedding_provider,
                embedding_model=self.config.embedding_model,
                hot_max_tokens=self.config.hot_max_tokens,
            )
            self._memory = AgentMemory(mem_config)
        return self._memory
    
    def register(self, hooks: dict[str, Callable]) -> None:
        """
        Register hooks with OpenClaw.
        
        Called by OpenClaw during plugin initialization.
        """
        if not self._enabled:
            logger.info("Agent memory plugin disabled")
            return
        
        hooks["pre_agent_turn"] = self.pre_agent_turn
        hooks["post_agent_turn"] = self.post_agent_turn
        hooks["pre_tool_call"] = self.pre_tool_call
        hooks["post_tool_call"] = self.post_tool_call
        hooks["on_session_start"] = self.on_session_start
        hooks["on_session_end"] = self.on_session_end
        
        logger.info("Agent memory plugin registered")
    
    def on_session_start(self, session_id: str = "", **kwargs) -> None:
        """Called when a new session starts."""
        self.memory.start_session(session_id)
        logger.info(f"Memory session started: {session_id}")
    
    def on_session_end(self, **kwargs) -> None:
        """Called when a session ends."""
        stats = self.memory.get_stats()
        logger.info(f"Memory session ended. Cost: ${stats['session']['cost_usd']:.4f}, "
                    f"Turns: {stats['session']['turns']}, "
                    f"Cache hit rate: {stats['warm_memory'].get('hit_rate', 'N/A')}")
        self.memory.close()
    
    def pre_agent_turn(
        self,
        messages: list[dict],
        model: str = "",
        **kwargs,
    ) -> Optional[dict]:
        """
        Pre-hook: Check cache and retrieve memory context.
        
        Returns:
            Dict with 'response' key if cache hit (skip LLM).
            None if cache miss (continue with normal flow).
        """
        # Check cache
        result = self.memory.check_cache(messages, model=model)
        
        if result.hit:
            logger.debug(f"Cache hit ({result.hit_type.value}, saved ${result.cost_saved:.4f})")
            return {
                "response": result.response,
                "cache_hit": True,
                "cache_hit_type": result.hit_type.value,
                "latency_ms": result.latency_ms,
            }
        
        # Cache miss — check if we should route to a cheaper model
        route = self.memory.route_to_model(cached_messages=messages)
        
        if route.route == "cached" and route.confidence > 0.7:
            # Decision engine says we have a fresh cached answer
            entry, freshness = self.memory.retrieval.retrieve_for_cache_check(messages)
            if entry and freshness.is_fresh:
                return {
                    "response": entry.text,
                    "cache_hit": True,
                    "cache_hit_type": "decision_engine",
                    "freshness_confidence": freshness.confidence,
                }
        
        # No cache hit — continue with LLM
        return None
    
    def post_agent_turn(
        self,
        messages: list[dict],
        response: str,
        model: str = "",
        cost_usd: float = 0.0,
        cache_hit: bool = False,
        **kwargs,
    ) -> None:
        """
        Post-hook: Store response and update tracking.
        """
        if cache_hit:
            # Already cached, just update hot memory
            self.memory.hot.add_assistant_message(
                content=response,
                model=model,
                cost=0.0,  # No cost for cache hit
                cache_hit=True,
            )
        else:
            # Record the turn
            result = self.memory.record_turn(
                messages=messages,
                response=response,
                model=model,
                cost_usd=cost_usd,
            )
            
            if result.stored:
                logger.debug(f"Stored in memory (tier: {result.remember_decision.tier.value})")
    
    def pre_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        **kwargs,
    ) -> Optional[str]:
        """
        Pre-hook: Check tool cache.
        
        Returns cached result if available, None otherwise.
        """
        cached = self.memory.check_tool_cache(tool_name, tool_args)
        if cached:
            logger.debug(f"Tool cache hit: {tool_name}")
            return cached
        return None
    
    def post_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        result: str,
        ttl: int = 3600,
        **kwargs,
    ) -> None:
        """Post-hook: Store tool result."""
        self.memory.record_tool_result(
            tool_name=tool_name,
            args=tool_args,
            result=result,
            ttl=ttl,
        )
    
    def get_context(self, messages: list[dict], budget_tokens: int = 4000) -> str:
        """
        Get memory context for augmenting prompts.
        
        Returns formatted context string.
        """
        result = self.memory.retrieve(messages, budget_tokens=budget_tokens)
        return result.to_context_string()
    
    def get_stats(self) -> dict:
        """Get memory system stats."""
        return self.memory.get_stats()


# ── Plugin Entry Point ──────────────────────────────────────────────

def create_plugin(config: Optional[dict] = None) -> OpenClawPlugin:
    """
    Create the plugin instance.
    
    This is the entry point called by OpenClaw's plugin loader.
    """
    plugin_config = PluginConfig(**(config or {}))
    return OpenClawPlugin(plugin_config)


# Allow direct testing
if __name__ == "__main__":
    plugin = create_plugin({"db_path": "/tmp/test-memory.db"})
    plugin.memory.start_session("test-session")
    
    # Simulate a turn
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
    ]
    
    # First call — cache miss
    result = plugin.pre_agent_turn(messages)
    print(f"Pre-turn result: {result}")
    
    # Simulate LLM response
    if not result:
        plugin.post_agent_turn(
            messages=messages,
            response="Python is a programming language.",
            model="gpt-4",
            cost_usd=0.01,
        )
    
    # Second call — cache hit
    result2 = plugin.pre_agent_turn(messages)
    print(f"Pre-turn result (2nd): {result2}")
    
    # Stats
    print(f"\nStats: {plugin.get_stats()}")
    
    plugin.memory.close()
