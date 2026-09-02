"""Core value objects.

Every retrieval result is normalised into `RetrievedChunk` before it leaves the
pipeline. Callers never touch raw vector-store metadata dicts, so a field that
exists on chat rows but not on document rows can't turn into a KeyError three
layers up.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Scope(StrEnum):
    """What a chunk is. Retrieval filters on this."""

    DOCUMENT = "doc"
    CHAT = "chat"


class Role(StrEnum):
    """Speaker of a chat chunk. `NONE` for document chunks."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    NONE = ""


def now_ms() -> int:
    return int(time.time() * 1000)


def deterministic_chunk_id(tenant_id: str, source_id: str, chunk_index: int) -> str:
    """Stable primary key so re-ingesting a document upserts instead of duplicating.

    Milvus upsert is keyed on the primary field. Deriving the key from
    (tenant, source, index) makes re-indexing idempotent for the common case
    where a document is edited and resubmitted.
    """
    raw = f"{tenant_id}\x00{source_id}\x00{chunk_index}"
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(slots=True)
class Chunk:
    """A unit of text ready to be embedded and stored."""

    tenant_id: str
    scope: Scope
    text: str
    source_id: str
    chunk_index: int = 0
    conversation_id: str = ""
    role: Role = Role.NONE
    title: str = ""
    created_at: int = field(default_factory=now_ms)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return deterministic_chunk_id(self.tenant_id, self.source_id, self.chunk_index)

    def embedding_text(self, prepend_title: bool = True) -> str:
        """Text as it should be embedded.

        The title prefix gives an otherwise context-free chunk ("Refunds are
        processed within 5 days") the surrounding subject it needs to match a
        query that names the document rather than its contents.
        """
        if prepend_title and self.title and self.scope is Scope.DOCUMENT:
            return f"{self.title}\n\n{self.text}"
        return self.text


@dataclass(slots=True)
class RetrievedChunk:
    """A search hit, normalised. Every field is always present."""

    chunk_id: str
    text: str
    scope: Scope
    score: float
    source_id: str = ""
    conversation_id: str = ""
    role: Role = Role.NONE
    title: str = ""
    chunk_index: int = 0
    created_at: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    # Only populated when MMR is enabled, which is the only consumer.
    dense: list[float] | None = None

    def to_message(self) -> dict[str, str]:
        """Render as an LLM chat message.

        Document chunks become system context; chat chunks keep their speaker so
        the model can tell "the customer said" from "we replied".
        """
        if self.scope is Scope.CHAT and self.role in (Role.USER, Role.ASSISTANT):
            return {"role": str(self.role), "content": self.text}
        return {"role": "system", "content": f"Context: {self.text}"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "scope": str(self.scope),
            "score": round(self.score, 6),
            "source_id": self.source_id,
            "conversation_id": self.conversation_id,
            "role": str(self.role),
            "title": self.title,
            "chunk_index": self.chunk_index,
            "created_at": self.created_at,
            "meta": self.meta,
        }


@dataclass(slots=True)
class RetrievalResult:
    query: str
    tenant_id: str
    conversation_id: str
    chunks: list[RetrievedChunk]
    took_ms: int = 0
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def as_messages(self) -> list[dict[str, str]]:
        return [c.to_message() for c in self.chunks]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "tenant_id": self.tenant_id,
            "conversation_id": self.conversation_id,
            "took_ms": self.took_ms,
            "chunks": [c.to_dict() for c in self.chunks],
        }
