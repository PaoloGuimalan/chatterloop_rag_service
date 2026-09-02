"""Pub/sub-driven RAG indexing and retrieval on Milvus."""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["RagService", "__version__"]


def __getattr__(name: str):
    # Lazy so `import rag_service` stays cheap and dependency-free for
    # tooling that only wants the version.
    if name == "RagService":
        from .service import RagService

        return RagService
    raise AttributeError(name)
