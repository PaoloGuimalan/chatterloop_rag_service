from __future__ import annotations

from ..config import EmbeddingSettings
from .base import Embedder
from .cache import LRUEmbeddingCache
from .openai_embedder import EmbeddingError, OpenAIEmbedder, l2_normalise

__all__ = [
    "Embedder",
    "EmbeddingError",
    "LRUEmbeddingCache",
    "OpenAIEmbedder",
    "build_embedder",
    "l2_normalise",
]


def build_embedder(settings: EmbeddingSettings) -> Embedder:
    return OpenAIEmbedder(settings)
