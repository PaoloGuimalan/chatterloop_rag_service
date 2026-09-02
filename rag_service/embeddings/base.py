from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors.

    Implementations must return L2-normalised vectors so that inner product and
    cosine similarity coincide - Milvus is configured for COSINE and mixing in
    unnormalised vectors silently skews every ranking.
    """

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str], tenant_id: str = "") -> list[list[float]]: ...

    def embed_query(self, text: str, tenant_id: str = "") -> list[float]: ...
