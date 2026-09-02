from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    """Second-stage precision filter.

    Fusion optimises recall over a wide candidate set; a cross-encoder reads
    each (query, passage) pair jointly and reorders for precision. The two
    stages are complementary - a bi-encoder must compress a passage into one
    vector before ever seeing the query, and that is where the nuance goes.
    """

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...


class NoopReranker:
    """Pass-through. Fusion order is already respectable; this is the default
    so the service has no heavyweight dependency out of the box."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        return chunks[:top_n]
