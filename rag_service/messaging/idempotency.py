"""At-least-once delivery guard.

Every bus worth using redelivers. Without a guard, a redelivered
`document.ingest` re-embeds the whole document - money spent for a no-op, since
the upsert is already idempotent by primary key. Retrieval requests are worse:
a duplicate publishes a second reply that the caller may treat as a new answer.

`SET NX EX` is the entire mechanism. The claim is taken *before* the handler
runs and released if the handler fails, so a genuine failure is still retried.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class IdempotencyGuard:
    def __init__(self, redis_client: object, ttl_seconds: int = 86_400, prefix: str = "rag:seen:"):
        self._redis = redis_client
        self.ttl = ttl_seconds
        self.prefix = prefix

    def claim(self, event_id: str) -> bool:
        """True if this worker now owns the event; False if already processed."""
        if not event_id:
            return True
        try:
            acquired = self._redis.set(  # type: ignore[attr-defined]
                f"{self.prefix}{event_id}", "1", nx=True, ex=self.ttl
            )
        except Exception as exc:
            # A dead dedupe store must not stop the pipeline. Duplicate work is
            # wasteful; dropped work is a data loss bug.
            logger.warning("idempotency check failed, processing anyway",
                           extra={"error": str(exc), "event_id": event_id})
            return True
        return bool(acquired)

    def release(self, event_id: str) -> None:
        """Give the claim back so a failed event can be retried."""
        if not event_id:
            return
        try:
            self._redis.delete(f"{self.prefix}{event_id}")  # type: ignore[attr-defined]
        except Exception:
            logger.debug("failed to release idempotency claim", exc_info=True)


class NullIdempotencyGuard:
    """Used when no Redis is configured (e.g. a pure GCP deployment without one)."""

    def claim(self, event_id: str) -> bool:
        return True

    def release(self, event_id: str) -> None:
        return None
