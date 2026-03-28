"""
Embedding engine for semantic search.

Default: local model (all-MiniLM-L6-v2) — zero API cost.
Optional: OpenAI text-embedding-3-small for higher quality.
"""

from __future__ import annotations

import abc
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np


class EmbeddingEngine(abc.ABC):
    """Abstract interface for embedding models."""
    
    @abc.abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a float vector."""
        ...
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Override for batch efficiency."""
        return [self.embed(t) for t in texts]
    
    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vector."""
        ...


class LocalEmbedding(EmbeddingEngine):
    """
    Local embedding using sentence-transformers.
    
    Model: all-MiniLM-L6-v2 (384 dims, ~80MB, fast)
    Cost: $0 (runs locally)
    Latency: ~5-20ms per text on CPU
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None  # Lazy load
        self._dim = 384  # Known for MiniLM
    
    @property
    def dimension(self) -> int:
        return self._dim
    
    def _load(self):
        """Lazy load the model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embedding. "
                    "Install with: pip install sentence-transformers"
                )
    
    def embed(self, text: str) -> list[float]:
        self._load()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._load()
        embeddings = self._model.encode(
            texts, 
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            batch_size=64,
        )
        return [e.tolist() for e in embeddings]


class HashEmbedding(EmbeddingEngine):
    """
    Deterministic pseudo-embedding using content hashing.
    
    NOT semantically meaningful — uses character-level features.
    For testing only. No real similarity search possible.
    
    Use this as a fallback when sentence-transformers isn't installed.
    """
    
    def __init__(self, dimension: int = 384):
        self._dim = dimension
    
    @property
    def dimension(self) -> int:
        return self._dim
    
    def embed(self, text: str) -> list[float]:
        """Generate a deterministic but non-semantic embedding."""
        import random
        rng = random.Random(hashlib.sha256(text.encode()).digest())
        vec = [rng.gauss(0, 1) for _ in range(self._dim)]
        # L2 normalize
        norm = sum(x**2 for x in vec) ** 0.5
        return [x / norm for x in vec]


def get_embedding_engine(
    provider: str = "local",
    model: Optional[str] = None,
) -> EmbeddingEngine:
    """
    Factory function to get an embedding engine.
    
    Args:
        provider: "local" | "hash" | "openai"
        model: Model name override.
    
    Returns:
        EmbeddingEngine instance.
    """
    if provider == "local":
        return LocalEmbedding(model_name=model or "all-MiniLM-L6-v2")
    elif provider == "hash":
        return HashEmbedding()
    elif provider == "openai":
        return OpenAIEmbedding(model_name=model or "text-embedding-3-small")
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


class OpenAIEmbedding(EmbeddingEngine):
    """
    OpenAI embedding API.
    
    Model: text-embedding-3-small (1536 dims)
    Cost: ~$0.02/1M tokens
    Latency: ~200ms per call
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self._model_name = model_name
        self._dim = 1536
    
    @property
    def dimension(self) -> int:
        return self._dim
    
    def embed(self, text: str) -> list[float]:
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.embeddings.create(
                input=text,
                model=self._model_name,
            )
            return response.data[0].embedding
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI embedding. "
                "Install with: pip install openai"
            )
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.embeddings.create(
                input=texts,
                model=self._model_name,
            )
            return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI embedding. "
                "Install with: pip install openai"
            )
