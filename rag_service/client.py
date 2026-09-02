"""Producer-side helper.

Small and dependency-light on purpose: `redis` and the standard library, no
import of the pipeline. Django (or anything else) can `pip install
rag-service` purely for this, or copy the file - it is designed to be
readable enough to vendor.

The one rule callers must follow: `event_id` has to be *stable* across retries.
Deriving it from the thing being indexed (message id, document revision) is what
makes the consumer's deduplication work. A fresh uuid per publish attempt turns
every retry into new work.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any


class RagEventPublisher:
    def __init__(
        self,
        redis_url: str,
        stream: str = "rag.events",
        maxlen: int = 100_000,
    ) -> None:
        import redis

        self._redis = redis.Redis.from_url(redis_url)
        self._stream = stream
        self._maxlen = maxlen

    def _publish(
        self, event_type: str, tenant_id: str, payload: dict[str, Any], event_id: str
    ) -> str:
        envelope = {
            "event_id": event_id,
            "event_type": event_type,
            "tenant_id": tenant_id,
            "occurred_at": int(time.time() * 1000),
            "payload": payload,
        }
        self._redis.xadd(
            self._stream,
            {"data": json.dumps(envelope, default=str)},
            maxlen=self._maxlen,
            approximate=True,
        )
        return event_id

    # ------------------------------------------------------------- indexing

    def index_message(
        self,
        tenant_id: str,
        conversation_id: str,
        message_id: str,
        text: str,
        role: str = "user",
        created_at: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        # Keyed on the message: republishing the same message is a no-op.
        return self._publish(
            "message.index",
            tenant_id,
            {
                "conversation_id": conversation_id,
                "message_id": message_id,
                "text": text,
                "role": role,
                "created_at": created_at,
                "meta": meta or {},
            },
            event_id=f"msg:{message_id}",
        )

    def ingest_document(
        self,
        tenant_id: str,
        document_id: str,
        text: str,
        title: str = "",
        revision: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        # Include a revision so an *edited* document is a new event while a
        # duplicate publish of the same revision is not.
        suffix = revision or str(abs(hash(text)) % (10**12))
        return self._publish(
            "document.ingest",
            tenant_id,
            {"document_id": document_id, "text": text, "title": title, "meta": meta or {}},
            event_id=f"doc:{document_id}:{suffix}",
        )

    def delete_document(self, tenant_id: str, document_id: str) -> str:
        return self._publish(
            "document.delete",
            tenant_id,
            {"document_id": document_id},
            event_id=f"doc-del:{document_id}:{uuid.uuid4().hex[:8]}",
        )

    def delete_conversation(self, tenant_id: str, conversation_id: str) -> str:
        return self._publish(
            "conversation.delete",
            tenant_id,
            {"conversation_id": conversation_id},
            event_id=f"conv-del:{conversation_id}:{uuid.uuid4().hex[:8]}",
        )

    # ------------------------------------------------------------ retrieval

    def request_retrieval(
        self,
        tenant_id: str,
        query: str,
        conversation_id: str = "",
        top_k: int | None = None,
        reply_to: str = "",
        correlation_id: str = "",
    ) -> str:
        """Ask for context. The answer arrives on `reply_to`.

        Retrieval is request/response over a queue, so the caller needs to be
        prepared to wait or to correlate asynchronously - see
        `await_result` for the simple blocking case.
        """
        correlation_id = correlation_id or uuid.uuid4().hex
        return self._publish(
            "retrieval.request",
            tenant_id,
            {
                "query": query,
                "conversation_id": conversation_id,
                "top_k": top_k,
                "reply_to": reply_to,
                "correlation_id": correlation_id,
            },
            event_id=f"ret:{correlation_id}",
        )

    def await_result(
        self, reply_stream: str, correlation_id: str, timeout_seconds: float = 30.0
    ) -> dict[str, Any] | None:
        """Block until the matching result appears on the reply stream.

        Reads from the stream tail with a deadline. Fine for a request-scoped
        call in a web worker; for high volume, run one reader that dispatches by
        correlation_id instead of one blocking read per request.
        """
        deadline = time.monotonic() + timeout_seconds
        last_id = "$"
        while time.monotonic() < deadline:
            remaining_ms = max(100, int((deadline - time.monotonic()) * 1000))
            response = self._redis.xread({reply_stream: last_id}, count=16, block=remaining_ms)
            if not response:
                continue
            for _stream, entries in response:
                for entry_id, fields in entries:
                    last_id = entry_id
                    raw = fields.get(b"data") or fields.get("data")
                    if raw is None:
                        continue
                    event = json.loads(raw)
                    if event.get("correlation_id") == correlation_id:
                        return event
        return None

    def close(self) -> None:
        try:
            self._redis.close()
        except Exception:
            pass
