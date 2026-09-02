"""The worker: consume, dispatch, ack. No HTTP surface."""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field

from .chunking import TokenChunker, default_tokenizer
from .config import Settings, get_settings
from .embeddings import build_embedder
from .handlers import EventHandlers, PermanentError
from .logging_setup import configure_logging
from .messaging import build_consumer, build_publisher
from .messaging.base import Consumer, Publisher, RawMessage
from .messaging.events import InvalidEvent, parse_event
from .messaging.idempotency import IdempotencyGuard, NullIdempotencyGuard
from .pipeline import IngestionPipeline, RetrievalPipeline
from .rerank import build_reranker
from .store import MilvusStore

logger = logging.getLogger(__name__)


@dataclass
class Stats:
    processed: int = 0
    failed: int = 0
    dead_lettered: int = 0
    duplicates: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "processed": self.processed,
            "failed": self.failed,
            "dead_lettered": self.dead_lettered,
            "duplicates": self.duplicates,
            "uptime_s": round(time.monotonic() - self.started_at, 1),
        }


class RagService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.stats = Stats()

        embedder = build_embedder(self.settings.embedding)
        self.store = MilvusStore(self.settings.milvus, dim=embedder.dim)

        chunker = TokenChunker(
            tokenizer=default_tokenizer(self.settings.embedding.model),
            max_tokens=self.settings.chunking.max_tokens,
            overlap_tokens=self.settings.chunking.overlap_tokens,
            min_tokens=self.settings.chunking.min_tokens,
        )

        ingestion = IngestionPipeline(embedder, self.store, chunker, self.settings.chunking)
        retrieval = RetrievalPipeline(
            embedder, self.store, build_reranker(self.settings.retrieval), self.settings.retrieval
        )

        self.consumer: Consumer = build_consumer(self.settings.messaging)
        self.publisher: Publisher = build_publisher(self.settings.messaging)
        self.handlers = EventHandlers(
            ingestion, retrieval, self.publisher, self.settings.messaging
        )
        self.guard = self._build_guard()
        self._stopping = False

    def _build_guard(self):
        if self.settings.messaging.backend == "redis":
            import redis

            return IdempotencyGuard(
                redis.Redis.from_url(self.settings.messaging.redis_url),
                ttl_seconds=self.settings.messaging.idempotency_ttl_seconds,
            )
        # A GCP deployment may not run Redis. Pub/Sub's own exactly-once
        # delivery subscription option covers this case; otherwise duplicates
        # are re-embedded, which is wasteful but not corrupting.
        return NullIdempotencyGuard()

    # ------------------------------------------------------------------- run

    def install_signal_handlers(self) -> None:
        def handle(signum: int, _frame: object) -> None:
            logger.info("shutdown signal received", extra={"signal": signum})
            self.stop()

        signal.signal(signal.SIGTERM, handle)
        signal.signal(signal.SIGINT, handle)

    def stop(self) -> None:
        self._stopping = True
        self.consumer.stop()

    def run(self) -> None:
        self.store.ensure_collection()
        logger.info(
            "worker ready",
            extra={
                "backend": self.settings.messaging.backend,
                "collection": self.settings.milvus.collection,
                "model": self.settings.embedding.model,
                "dim": self.settings.embedding.dim,
                "rerank": self.settings.retrieval.rerank_provider,
            },
        )
        try:
            for message in self.consumer.consume():
                self.handle(message)
                if self._stopping:
                    break
        finally:
            self.shutdown()

    def handle(self, message: RawMessage) -> None:
        # 1. Parse. A message we can't understand will never become
        #    understandable, so it is parked immediately rather than retried.
        try:
            envelope, payload = parse_event(message.body)
        except InvalidEvent as exc:
            self.stats.dead_lettered += 1
            self.consumer.dead_letter(message, str(exc))
            return

        log_context = {
            "event_id": envelope.event_id,
            "event_type": str(envelope.event_type),
            "tenant_id": envelope.tenant_id,
            "delivery": message.delivery_count,
        }

        # 2. Poison-message cutoff, before doing any expensive work.
        if message.delivery_count > self.settings.messaging.max_deliveries:
            self.stats.dead_lettered += 1
            logger.error("max deliveries exceeded", extra=log_context)
            self.consumer.dead_letter(message, "max deliveries exceeded")
            return

        # 3. Deduplicate. Claimed before the handler runs, released on failure
        #    so a genuine error is still retried.
        if not self.guard.claim(envelope.event_id):
            self.stats.duplicates += 1
            logger.info("duplicate event ignored", extra=log_context)
            self.consumer.ack(message)
            return

        started = time.monotonic()
        try:
            result = self.handlers.dispatch(envelope, payload)
        except PermanentError as exc:
            self.stats.dead_lettered += 1
            logger.error("permanent failure", extra={**log_context, "error": str(exc)})
            self.consumer.dead_letter(message, str(exc))
        except Exception as exc:
            self.stats.failed += 1
            self.guard.release(envelope.event_id)
            logger.exception("handler failed, will retry", extra={**log_context, "error": str(exc)})
            self.consumer.nack(message)
        else:
            self.stats.processed += 1
            logger.info(
                "event handled",
                extra={
                    **log_context,
                    **result,
                    "took_ms": int((time.monotonic() - started) * 1000),
                },
            )
            self.consumer.ack(message)

    def shutdown(self) -> None:
        logger.info("shutting down", extra=self.stats.snapshot())
        for closable in (self.consumer, self.publisher, self.store):
            try:
                closable.close()
            except Exception:  # pragma: no cover - best effort
                logger.debug("error during shutdown", exc_info=True)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    service = RagService(settings)
    service.install_signal_handlers()
    service.run()
