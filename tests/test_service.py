"""Delivery semantics: ack, retry, dead-letter, dedupe."""

from __future__ import annotations

import pytest

from rag_service.config import MessagingSettings
from rag_service.handlers import PermanentError
from rag_service.messaging.base import RawMessage
from rag_service.messaging.idempotency import IdempotencyGuard
from rag_service.service import RagService, Stats


class FakeConsumer:
    def __init__(self):
        self.acked: list[str] = []
        self.nacked: list[str] = []
        self.dead: list[tuple[str, str]] = []

    def ack(self, m): self.acked.append(m.message_id)
    def nack(self, m): self.nacked.append(m.message_id)
    def dead_letter(self, m, reason): self.dead.append((m.message_id, reason))
    def stop(self): pass
    def close(self): pass


class FakeHandlers:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[str] = []

    def dispatch(self, envelope, payload):
        self.calls.append(envelope.event_id)
        if self.error:
            raise self.error
        return {"ok": True}


class FakeRedis:
    def __init__(self): self.store: dict[str, str] = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    def delete(self, key): self.store.pop(key, None)


def make_service(handlers: FakeHandlers, guard=None) -> RagService:
    # Bypass __init__: it builds a Milvus client, an embedder and a bus
    # connection, none of which this test is about.
    service = RagService.__new__(RagService)
    service.settings = type("S", (), {"messaging": MessagingSettings(max_deliveries=3)})()
    service.stats = Stats()
    service.consumer = FakeConsumer()
    service.handlers = handlers
    service.guard = guard or IdempotencyGuard(FakeRedis())
    service._stopping = False
    return service


def message(event_id="e1", event_type="document.delete", payload=None, delivery=1) -> RawMessage:
    return RawMessage(
        message_id=f"msg-{event_id}",
        body={
            "event_id": event_id,
            "event_type": event_type,
            "tenant_id": "org_1",
            "payload": payload if payload is not None else {"document_id": "d1"},
        },
        delivery_count=delivery,
    )


class TestHappyPath:
    def test_valid_message_is_dispatched_and_acked(self):
        service = make_service(FakeHandlers())
        service.handle(message())
        assert service.consumer.acked == ["msg-e1"]
        assert service.stats.processed == 1


class TestPermanentFailures:
    def test_unparseable_message_is_dead_lettered_immediately(self):
        service = make_service(FakeHandlers())
        service.handle(RawMessage(message_id="bad", body={"nonsense": True}))
        assert len(service.consumer.dead) == 1
        assert service.consumer.nacked == []

    def test_permanent_handler_error_is_dead_lettered(self):
        service = make_service(FakeHandlers(error=PermanentError("no reply_to")))
        service.handle(message())
        assert service.consumer.dead[0][1] == "no reply_to"
        assert service.stats.dead_lettered == 1

    def test_poison_message_stops_at_the_delivery_ceiling(self):
        service = make_service(FakeHandlers())
        service.handle(message(delivery=4))  # max_deliveries=3
        assert service.consumer.dead[0][1] == "max deliveries exceeded"
        # The handler must not run again for a message we are giving up on.
        assert service.handlers.calls == []


class TestTransientFailures:
    def test_unexpected_error_is_nacked_for_retry(self):
        service = make_service(FakeHandlers(error=RuntimeError("milvus timeout")))
        service.handle(message())
        assert service.consumer.nacked == ["msg-e1"]
        assert service.consumer.dead == []
        assert service.stats.failed == 1

    def test_failed_event_can_be_retried(self):
        # The idempotency claim must be released on failure, or the retry would
        # be swallowed as a duplicate and the work lost forever.
        guard = IdempotencyGuard(FakeRedis())
        failing = FakeHandlers(error=RuntimeError("boom"))
        service = make_service(failing, guard=guard)
        service.handle(message())

        succeeding = FakeHandlers()
        retry = make_service(succeeding, guard=guard)
        retry.handle(message(delivery=2))
        assert succeeding.calls == ["e1"]
        assert retry.consumer.acked == ["msg-e1"]


class TestIdempotency:
    def test_redelivery_of_a_processed_event_is_skipped(self):
        guard = IdempotencyGuard(FakeRedis())
        handlers = FakeHandlers()
        first = make_service(handlers, guard=guard)
        first.handle(message())

        second = make_service(handlers, guard=guard)
        second.handle(message(delivery=2))

        assert handlers.calls == ["e1"]  # dispatched exactly once
        assert second.consumer.acked == ["msg-e1"]  # but still acked
        assert second.stats.duplicates == 1

    def test_distinct_events_both_run(self):
        guard = IdempotencyGuard(FakeRedis())
        handlers = FakeHandlers()
        service = make_service(handlers, guard=guard)
        service.handle(message(event_id="e1"))
        service.handle(message(event_id="e2"))
        assert handlers.calls == ["e1", "e2"]

    def test_dedupe_store_failure_does_not_block_processing(self):
        class BrokenRedis:
            def set(self, *a, **kw): raise ConnectionError("redis down")
            def delete(self, *a, **kw): raise ConnectionError("redis down")

        handlers = FakeHandlers()
        service = make_service(handlers, guard=IdempotencyGuard(BrokenRedis()))
        service.handle(message())
        # Duplicate work beats dropped work.
        assert handlers.calls == ["e1"]
        assert service.consumer.acked == ["msg-e1"]


class TestStats:
    def test_snapshot_reports_counters(self):
        service = make_service(FakeHandlers())
        service.handle(message())
        snapshot = service.stats.snapshot()
        assert snapshot["processed"] == 1
        assert "uptime_s" in snapshot
