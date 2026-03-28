"""
Hot Memory — In-memory session state (ring buffer).

Stores the most recent turns, tool results, and active task context.
Fastest tier: <1ms access, no disk I/O.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from agent_memory.types import TaskType


@dataclass
class ToolResult:
    """A tool execution result."""
    
    tool_name: str
    tool_call_id: str
    args: dict
    result: str
    timestamp: int = field(default_factory=lambda: int(time.time()))
    cost_usd: float = 0.0
    
    def cache_key(self) -> str:
        """Deterministic cache key for tool result lookup."""
        import hashlib
        import json
        args_str = json.dumps(self.args, sort_keys=True)
        raw = f"{self.tool_name}:{args_str}"
        return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class ToolCall:
    """A tool call made by the model."""
    
    tool_name: str
    tool_call_id: str
    args: dict


@dataclass
class Turn:
    """A single conversation turn."""
    
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    timestamp: int = field(default_factory=lambda: int(time.time()))
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    model_used: Optional[str] = None
    cost_usd: float = 0.0
    cache_hit: bool = False
    
    def estimate_tokens(self) -> int:
        return len(self.content) // 4


@dataclass
class TaskContext:
    """Active task tracking."""
    
    task_type: TaskType
    description: str
    started_at: int = field(default_factory=lambda: int(time.time()))
    steps_completed: list[str] = field(default_factory=list)
    current_focus: str = ""


class HotMemory:
    """
    In-memory ring buffer for session state.
    
    Stores recent turns and tool results.
    Bounded size to prevent unbounded memory growth.
    """
    
    def __init__(
        self,
        max_turns: int = 50,
        max_tool_results: int = 20,
        max_tokens: int = 4000,
    ):
        self.max_turns = max_turns
        self.max_tool_results = max_tool_results
        self.max_tokens = max_tokens
        
        self.recent_turns: deque[Turn] = deque(maxlen=max_turns)
        self.recent_tool_results: deque[ToolResult] = deque(maxlen=max_tool_results)
        self.active_task: Optional[TaskContext] = None
        self.user_preferences: dict[str, str] = {}
        self.session_cost: float = 0.0
        self.session_id: str = ""
    
    def add_turn(self, turn: Turn) -> None:
        """Add a turn to the ring buffer."""
        self.recent_turns.append(turn)
        self.session_cost += turn.cost_usd
        
        # Capture tool results from this turn
        for result in turn.tool_results:
            self.recent_tool_results.append(result)
    
    def add_user_message(self, content: str) -> Turn:
        """Convenience: create and add a user turn."""
        turn = Turn(role="user", content=content)
        self.add_turn(turn)
        return turn
    
    def add_assistant_message(
        self, 
        content: str, 
        model: str = "", 
        cost: float = 0.0,
        tool_calls: Optional[list[ToolCall]] = None,
        cache_hit: bool = False,
    ) -> Turn:
        """Convenience: create and add an assistant turn."""
        turn = Turn(
            role="assistant",
            content=content,
            model_used=model,
            cost_usd=cost,
            tool_calls=tool_calls or [],
            cache_hit=cache_hit,
        )
        self.add_turn(turn)
        return turn
    
    def get_recent_messages(self, limit: Optional[int] = None) -> list[dict]:
        """
        Get recent turns as OpenAI-style message dicts.
        Used for constructing prompts.
        """
        turns = list(self.recent_turns)
        if limit:
            turns = turns[-limit:]
        
        messages = []
        for turn in turns:
            msg = {"role": turn.role, "content": turn.content}
            messages.append(msg)
        return messages
    
    def get_tool_result(self, tool_name: str, args: dict) -> Optional[ToolResult]:
        """Look up a recent tool result by name and args."""
        import hashlib
        import json
        args_str = json.dumps(args, sort_keys=True)
        raw = f"{tool_name}:{args_str}"
        target_hash = hashlib.sha256(raw.encode()).hexdigest()
        
        for result in self.recent_tool_results:
            if result.cache_key() == target_hash:
                return result
        return None
    
    def set_task(self, task_type: TaskType, description: str, focus: str = "") -> None:
        """Set the active task context."""
        self.active_task = TaskContext(
            task_type=task_type,
            description=description,
            current_focus=focus,
        )
    
    def complete_step(self, step_description: str) -> None:
        """Mark a task step as completed."""
        if self.active_task:
            self.active_task.steps_completed.append(step_description)
    
    def set_preference(self, key: str, value: str) -> None:
        """Store a user preference for this session."""
        self.user_preferences[key] = value
    
    def get_preference(self, key: str, default: str = "") -> str:
        """Get a user preference."""
        return self.user_preferences.get(key, default)
    
    def estimate_hot_tokens(self) -> int:
        """Estimate total tokens in hot memory."""
        total = 0
        for turn in self.recent_turns:
            total += turn.estimate_tokens()
        return total
    
    def assemble_context(self) -> str:
        """
        Assemble hot memory into a context string.
        Respects max_tokens budget.
        """
        parts = []
        token_budget = self.max_tokens
        
        # Include active task if set
        if self.active_task:
            task_text = (
                f"[Active Task: {self.active_task.task_type.value}] "
                f"{self.active_task.description}"
            )
            if self.active_task.current_focus:
                task_text += f"\nCurrent focus: {self.active_task.current_focus}"
            if self.active_task.steps_completed:
                task_text += f"\nCompleted: {', '.join(self.active_task.steps_completed[-5:])}"
            
            task_tokens = len(task_text) // 4
            if task_tokens <= token_budget:
                parts.append(task_text)
                token_budget -= task_tokens
        
        # Include user preferences if any
        if self.user_preferences:
            pref_text = "[User Preferences]\n" + "\n".join(
                f"- {k}: {v}" for k, v in self.user_preferences.items()
            )
            pref_tokens = len(pref_text) // 4
            if pref_tokens <= token_budget:
                parts.append(pref_text)
                token_budget -= pref_tokens
        
        # Include recent turns (newest first in budget)
        recent = list(self.recent_turns)
        for turn in reversed(recent):
            turn_text = f"[{turn.role}] {turn.content}"
            turn_tokens = turn.estimate_tokens()
            
            if turn_tokens <= token_budget:
                parts.insert(0, turn_text)  # Prepend to maintain order
                token_budget -= turn_tokens
            else:
                break
        
        return "\n\n".join(parts)
    
    def clear(self) -> None:
        """Clear all hot memory."""
        self.recent_turns.clear()
        self.recent_tool_results.clear()
        self.active_task = None
        self.user_preferences.clear()
        self.session_cost = 0.0
    
    def stats(self) -> dict:
        """Get hot memory statistics."""
        return {
            "turns_stored": len(self.recent_turns),
            "tool_results_stored": len(self.recent_tool_results),
            "estimated_tokens": self.estimate_hot_tokens(),
            "session_cost_usd": round(self.session_cost, 4),
            "has_active_task": self.active_task is not None,
            "preferences_count": len(self.user_preferences),
        }
