"""Event handlers: one function per event type, no transport knowledge."""

from __future__ import annotations

import logging
from typing import Any, Callable

from pydantic import BaseModel

from .config import MessagingSettings
from .domain import Scope
from .messaging.base import Publisher
from .messaging.events import (
    ConversationDelete,
    DocumentDelete,
    DocumentIngest,
    Envelope,
    EventType,
    MessageIndex,
    RetrievalRequest,
)
from .pipeline import IngestionPipeline, RetrievalPipeline

logger = logging.getLogger(__name__)


class PermanentError(Exception):
    """The message will never succeed. Straight to the DLQ, no retries."""


class EventHandlers:
    def __init__(
        self,
        ingestion: IngestionPipeline,
        retrieval: RetrievalPipeline,
        publisher: Publisher,
        messaging: MessagingSettings,
    ) -> None:
        self.ingestion = ingestion
        self.retrieval = retrieval
        self.publisher = publisher
        self.messaging = messaging

        self._routes: dict[EventType, Callable[[Envelope, Any], dict[str, Any]]] = {
            EventType.DOCUMENT_INGEST: self.on_document_ingest,
            EventType.DOCUMENT_DELETE: self.on_document_delete,
            EventType.MESSAGE_INDEX: self.on_message_index,
            EventType.CONVERSATION_DELETE: self.on_conversation_delete,
            EventType.RETRIEVAL_REQUEST: self.on_retrieval_request,
        }

    def dispatch(self, envelope: Envelope, payload: BaseModel) -> dict[str, Any]:
        handler = self._routes.get(envelope.event_type)
        if handler is None:
            raise PermanentError(f"unroutable event type {envelope.event_type!r}")
        return handler(envelope, payload)

    # --------------------------------------------------------------- indexing

    def on_document_ingest(self, envelope: Envelope, payload: DocumentIngest) -> dict[str, Any]:
        written = self.ingestion.ingest_document(
            tenant_id=envelope.tenant_id,
            document_id=payload.document_id,
            text=payload.text,
            title=payload.title,
            meta=payload.meta,
            created_at=payload.created_at or envelope.occurred_at,
        )
        return {"chunks_written": written}

    def on_document_delete(self, envelope: Envelope, payload: DocumentDelete) -> dict[str, Any]:
        self.ingestion.delete_document(envelope.tenant_id, payload.document_id)
        return {"deleted": True}

    def on_message_index(self, envelope: Envelope, payload: MessageIndex) -> dict[str, Any]:
        written = self.ingestion.index_message(
            tenant_id=envelope.tenant_id,
            conversation_id=payload.conversation_id,
            message_id=payload.message_id,
            text=payload.text,
            role=payload.role,
            meta=payload.meta,
            created_at=payload.created_at or envelope.occurred_at,
        )
        return {"chunks_written": written}

    def on_conversation_delete(
        self, envelope: Envelope, payload: ConversationDelete
    ) -> dict[str, Any]:
        self.ingestion.delete_conversation(envelope.tenant_id, payload.conversation_id)
        return {"deleted": True}

    # -------------------------------------------------------------- retrieval

    def on_retrieval_request(
        self, envelope: Envelope, payload: RetrievalRequest
    ) -> dict[str, Any]:
        scopes: list[Scope] | None = None
        if payload.scopes:
            try:
                scopes = [Scope(s) for s in payload.scopes]
            except ValueError as exc:
                raise PermanentError(f"unknown scope in request: {exc}") from exc

        result = self.retrieval.retrieve(
            tenant_id=envelope.tenant_id,
            query=payload.query,
            conversation_id=payload.conversation_id,
            top_k=payload.top_k,
            scopes=scopes,
            include_recent_history=payload.include_recent_history,
        )

        destination = payload.reply_to or self.messaging.reply_stream_default
        if not destination:
            # Nowhere to send the answer. Retrying cannot create a destination.
            raise PermanentError("retrieval.request has no reply_to and no default reply stream")

        reply = {
            "event_id": f"{envelope.event_id}:result",
            "event_type": str(EventType.RETRIEVAL_RESULT),
            "tenant_id": envelope.tenant_id,
            "correlation_id": payload.correlation_id or envelope.event_id,
            "payload": {
                **result.to_dict(),
                # Ready to splice straight into a chat completion.
                "messages": result.as_messages(),
            },
        }
        self.publisher.publish(destination, reply)
        return {"returned": len(result.chunks), "reply_to": destination}
