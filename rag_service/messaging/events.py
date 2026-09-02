"""Bus message contracts.

The envelope is stable across transports so a Redis Streams deployment and a
Google Pub/Sub deployment speak the same language. Payloads are validated on
arrival: a malformed message is a permanent failure that belongs in the DLQ,
not a transient one worth retrying five times.

Deliberately absent: API keys. Credentials are configuration, not payload. The
previous pipeline threaded `organization.llm_api_key` through call arguments,
which is how keys end up in log lines and queue backlogs.
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..domain import Role


class EventType(StrEnum):
    DOCUMENT_INGEST = "document.ingest"
    DOCUMENT_DELETE = "document.delete"
    MESSAGE_INDEX = "message.index"
    CONVERSATION_DELETE = "conversation.delete"
    RETRIEVAL_REQUEST = "retrieval.request"
    RETRIEVAL_RESULT = "retrieval.result"


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Consumers dedupe on this. Producers must make it stable across retries -
    # a new id per publish attempt defeats the whole mechanism.
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    event_type: EventType
    tenant_id: str = Field(min_length=1, max_length=64)
    occurred_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tenant_id")
    @classmethod
    def _no_control_chars(cls, v: str) -> str:
        if "\x00" in v or v.strip() != v:
            raise ValueError("tenant_id must not contain null bytes or surrounding whitespace")
        return v


class DocumentIngest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_id: str = Field(min_length=1, max_length=128)
    text: str
    title: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: int | None = None


class DocumentDelete(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_id: str = Field(min_length=1, max_length=128)


class MessageIndex(BaseModel):
    model_config = ConfigDict(extra="ignore")

    conversation_id: str = Field(min_length=1, max_length=64)
    message_id: str = Field(min_length=1, max_length=128)
    text: str
    role: Role = Role.USER
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: int | None = None

    @field_validator("role", mode="before")
    @classmethod
    def _coerce_role(cls, v: object) -> object:
        """Accept the sending system's vocabulary.

        Neon calls them `text` and `ai_reply`; the model layer wants
        `user`/`assistant`. Translating at the boundary keeps that mapping in
        one place instead of scattered through handlers.
        """
        aliases = {
            "text": "user",
            "reply": "user",
            "ai_reply": "assistant",
            "agent": "assistant",
            "bot": "assistant",
        }
        if isinstance(v, str):
            return aliases.get(v, v)
        return v


class ConversationDelete(BaseModel):
    model_config = ConfigDict(extra="ignore")

    conversation_id: str = Field(min_length=1, max_length=64)


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1)
    conversation_id: str = ""
    top_k: int | None = Field(default=None, ge=1, le=100)
    # Absent means both scopes.
    scopes: list[str] | None = None
    include_recent_history: bool = True
    # Where to publish the answer. Falls back to the configured default stream.
    reply_to: str = ""
    correlation_id: str = ""


PAYLOAD_MODELS: dict[EventType, type[BaseModel]] = {
    EventType.DOCUMENT_INGEST: DocumentIngest,
    EventType.DOCUMENT_DELETE: DocumentDelete,
    EventType.MESSAGE_INDEX: MessageIndex,
    EventType.CONVERSATION_DELETE: ConversationDelete,
    EventType.RETRIEVAL_REQUEST: RetrievalRequest,
}


class InvalidEvent(ValueError):
    """Permanent parse/validation failure. Never worth retrying."""


def parse_event(raw: dict[str, Any]) -> tuple[Envelope, BaseModel]:
    try:
        envelope = Envelope.model_validate(raw)
    except Exception as exc:
        raise InvalidEvent(f"invalid envelope: {exc}") from exc

    model = PAYLOAD_MODELS.get(envelope.event_type)
    if model is None:
        raise InvalidEvent(f"no handler contract for event type {envelope.event_type!r}")

    try:
        payload = model.model_validate(envelope.payload)
    except Exception as exc:
        raise InvalidEvent(f"invalid payload for {envelope.event_type}: {exc}") from exc

    return envelope, payload
