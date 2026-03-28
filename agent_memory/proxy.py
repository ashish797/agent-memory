#!/usr/bin/env python3
"""
Agent Memory Proxy — Drop-in OpenAI-compatible proxy with memory caching.

Sits between your application and any LLM provider.
Transparently caches responses, serves from cache on repeat/similar queries.

Usage:
    python -m agent_memory.proxy --port 8080 --upstream https://api.openai.com/v1

    # Then point your client at the proxy:
    export OPENAI_BASE_URL=http://localhost:8080/v1
    export OPENAI_API_KEY=your-key  # Passed through to upstream
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProxyConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    upstream: str = "https://api.openai.com/v1"
    db_path: str = "~/.openclaw/agent-memory/proxy-cache.db"
    embedding_provider: str = "hash"  # Fast for proxy
    hit_threshold: float = 0.95
    gray_zone_low: float = 0.80
    log_level: str = "INFO"
    cors: bool = True
    rate_limit: int = 0  # 0 = no limit


def _extract_messages(body: dict) -> list[dict]:
    """Extract messages from a chat completion request."""
    return body.get("messages", [])


def _extract_model(body: dict) -> str:
    """Extract model from request."""
    return body.get("model", "unknown")


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost based on model (rough approximation)."""
    # Approximate pricing per 1M tokens
    prices = {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-opus": {"input": 15.0, "output": 75.0},
        "claude-sonnet": {"input": 3.0, "output": 15.0},
        "claude-haiku": {"input": 0.25, "output": 1.25},
    }
    
    # Find best match
    model_lower = model.lower()
    for key, price in prices.items():
        if key in model_lower:
            return (
                (input_tokens / 1_000_000) * price["input"] +
                (output_tokens / 1_000_000) * price["output"]
            )
    
    # Default: gpt-4o pricing
    return (
        (input_tokens / 1_000_000) * 5.0 +
        (output_tokens / 1_000_000) * 15.0
    )


class MemoryProxy:
    """
    OpenAI-compatible proxy with memory caching.
    
    Intercepts chat completion requests, checks cache, forwards to upstream
    if cache miss, stores response for future use.
    """
    
    def __init__(self, config: Optional[ProxyConfig] = None):
        self.config = config or ProxyConfig()
        self._memory = None
        self._request_count = 0
        self._cache_hits = 0
        self._total_saved = 0.0
        self._start_time = time.time()
    
    @property
    def memory(self):
        if self._memory is None:
            from agent_memory.memory import AgentMemory, MemoryConfig
            
            mem_config = MemoryConfig(
                db_path=self.config.db_path,
                embedding_provider=self.config.embedding_provider,
                hit_threshold=self.config.hit_threshold,
                gray_zone_low=self.config.gray_zone_low,
            )
            self._memory = AgentMemory(mem_config)
            self._memory.start_session("proxy")
        return self._memory
    
    async def handle_request(
        self,
        path: str,
        method: str,
        headers: dict,
        body: Optional[dict] = None,
    ) -> tuple[int, dict, dict]:
        """
        Handle an incoming request.
        
        Returns: (status_code, response_headers, response_body)
        """
        # Health check
        if path == "/health" or path == "/":
            return 200, {}, {
                "status": "ok",
                "cache_hits": self._cache_hits,
                "total_requests": self._request_count,
                "uptime_seconds": time.time() - self._start_time,
                "hit_rate": self._cache_hits / max(1, self._request_count),
            }
        
        # Stats endpoint
        if path == "/stats":
            stats = self.memory.get_stats()
            return 200, {}, {
                "proxy": {
                    "requests": self._request_count,
                    "cache_hits": self._cache_hits,
                    "hit_rate": self._cache_hits / max(1, self._request_count),
                    "total_saved": self._total_saved,
                },
                "memory": stats,
            }
        
        # Chat completions endpoint
        if path == "/chat/completions" and method == "POST":
            return await self._handle_chat_completions(body or {})
        
        # Models endpoint (pass through)
        if path == "/models":
            return await self._proxy_to_upstream("GET", "/models", headers)
        
        # Unknown endpoint — proxy through
        return await self._proxy_to_upstream(method, path, headers, body)
    
    async def _handle_chat_completions(self, body: dict) -> tuple[int, dict, dict]:
        """Handle a chat completion request with caching."""
        self._request_count += 1
        
        messages = _extract_messages(body)
        model = _extract_model(body)
        
        # Skip caching for streaming (not supported yet)
        stream = body.get("stream", False)
        
        if not stream:
            # Check cache
            result = self.memory.check_cache(messages, model=model)
            
            if result.hit:
                self._cache_hits += 1
                self._total_saved += result.cost_saved
                
                logger.info(f"Cache hit ({result.hit_type.value}) — saved ${result.cost_saved:.4f}")
                
                # Build cached response
                response = {
                    "id": f"chatcmpl-cache-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.response,
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "x-cache": "HIT",
                    "x-cache-type": result.hit_type.value,
                    "x-latency-ms": round(result.latency_ms, 2),
                }
                
                return 200, {"x-cache": "HIT"}, response
        
        # Cache miss — proxy to upstream
        status, resp_headers, response = await self._proxy_to_upstream(
            "POST", "/chat/completions", {}, body
        )
        
        if status == 200 and isinstance(response, dict):
            # Store in cache
            assistant_content = ""
            if "choices" in response and response["choices"]:
                msg = response["choices"][0].get("message", {})
                assistant_content = msg.get("content", "")
            
            if assistant_content:
                usage = response.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                cost = _estimate_cost(model, input_tokens, output_tokens)
                
                self.memory.record_turn(
                    messages=messages,
                    response=assistant_content,
                    model=model,
                    cost_usd=cost,
                )
                
                response["x-cache"] = "MISS"
        
        return status, resp_headers or {}, response
    
    async def _proxy_to_upstream(
        self,
        method: str,
        path: str,
        headers: dict,
        body: Optional[dict] = None,
    ) -> tuple[int, dict, dict]:
        """Proxy a request to the upstream provider."""
        import urllib.request
        import urllib.error
        
        url = f"{self.config.upstream}{path}"
        
        data = None
        if body:
            data = json.dumps(body).encode()
        
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                **{k: v for k, v in headers.items() if k.lower().startswith("authorization")},
            },
            method=method,
        )
        
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                response_body = json.loads(resp.read().decode())
                return resp.status, dict(resp.headers), response_body
        except urllib.error.HTTPError as e:
            return e.code, {}, {"error": str(e)}
        except Exception as e:
            return 502, {}, {"error": str(e)}


def run_server(config: Optional[ProxyConfig] = None):
    """Run the proxy server using asyncio."""
    import http.server
    import threading
    
    config = config or ProxyConfig()
    proxy = MemoryProxy(config)
    
    logging.basicConfig(level=getattr(logging, config.log_level))
    
    class ProxyHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self._handle("GET")
        
        def do_POST(self):
            self._handle("POST")
        
        def _handle(self, method):
            content_length = int(self.headers.get("Content-Length", 0))
            body_data = self.rfile.read(content_length) if content_length > 0 else b""
            
            body = None
            if body_data:
                try:
                    body = json.loads(body_data.decode())
                except json.JSONDecodeError:
                    body = {}
            
            headers = dict(self.headers)
            
            # Run async handler
            loop = asyncio.new_event_loop()
            try:
                status, resp_headers, response = loop.run_until_complete(
                    proxy.handle_request(
                        path=self.path,
                        method=method,
                        headers=headers,
                        body=body,
                    )
                )
            finally:
                loop.close()
            
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            
            for key, value in resp_headers.items():
                if key.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(key, value)
            
            if config.cors:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
            
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        
        def log_message(self, format, *args):
            logger.debug(format % args)
    
    server = http.server.HTTPServer((config.host, config.port), ProxyHandler)
    
    print(f"""
╔══════════════════════════════════════════════════╗
║  Agent Memory Proxy                              ║
║                                                  ║
║  Listening: http://{config.host}:{config.port}            ║
║  Upstream:  {config.upstream:<38s}║
║  Cache DB:  {config.db_path[:38]:<38s}║
║                                                  ║
║  Endpoints:                                      ║
║    POST /chat/completions  (cached)              ║
║    GET  /models            (proxied)             ║
║    GET  /health            (health check)        ║
║    GET  /stats             (cache stats)         ║
║                                                  ║
║  Usage:                                          ║
║    OPENAI_BASE_URL=http://localhost:{config.port}/v1  ║
╚══════════════════════════════════════════════════╝
""")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        proxy.memory.close()
        server.server_close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Memory Proxy")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument("--upstream", default="https://api.openai.com/v1", help="Upstream API URL")
    parser.add_argument("--db", default="~/.openclaw/agent-memory/proxy-cache.db", help="Cache DB path")
    parser.add_argument("--embedding", default="hash", choices=["hash", "local", "openai"], help="Embedding provider")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    config = ProxyConfig(
        host=args.host,
        port=args.port,
        upstream=args.upstream,
        db_path=args.db,
        embedding_provider=args.embedding,
        log_level=args.log_level,
    )
    
    run_server(config)
