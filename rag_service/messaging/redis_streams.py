"""Redis Streams transport.

Streams rather than plain Redis pub/sub, which is fire-and-forget: a worker
restart during a deploy would silently drop every message published in that
window. Consumer groups give at-least-once delivery, per-consumer ownership,
and a pending-entries list that lets a live worker reclaim what a crashed one
was holding.

Chosen as the default because chatterloop already runs Redis for Celery, so
this adds a transport without adding infrastructure.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from typing import Any, Iterator

from .base import RawMessage

logger = logging.getLogger(__name__)


def default_consumer_name() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _decode_body(fields: dict[Any, Any]) -> dict[str, Any]:
    """Read the event out of a stream entry.

    Canonical form is a single `data` field holding JSON. A flat field map is
    also accepted so anything can publish here with redis-cli.
    """
    normalised = {
        (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
        for k, v in fields.items()
    }
    if "data" in normalised:
        return json.loads(normalised["data"])
    if "payload" in normalised and isinstance(normalised["payload"], str):
        normalised = {**normalised, "payload": json.loads(normalised["payload"])}
    return normalised


class RedisStreamsConsumer:
    def __init__(
        self,
        url: str,
        stream: str,
        group: str,
        consumer_name: str = "",
        dlq_stream: str = "",
        batch_size: int = 16,
        block_ms: int = 5_000,
        claim_min_idle_ms: int = 60_000,
        max_deliveries: int = 5,
        delete_on_ack: bool = True,
    ) -> None:
        import redis

        self.redis = redis.Redis.from_url(url)
        self.stream = stream
        self.group = group
        self.consumer = consumer_name or default_consumer_name()
        self.dlq_stream = dlq_stream or f"{stream}.dlq"
        self.batch_size = batch_size
        self.block_ms = block_ms
        self.claim_min_idle_ms = claim_min_idle_ms
        self.max_deliveries = max_deliveries
        # Safe while this stream has exactly one consumer group. If another
        # service ever reads the same stream, turn this off and trim by MAXLEN
        # instead, or acked entries will vanish from under it.
        self.delete_on_ack = delete_on_ack

        self._running = False
        self._claim_cursor = "0-0"

    def ensure_group(self) -> None:
        try:
            self.redis.xgroup_create(self.stream, self.group, id="0", mkstream=True)
            logger.info("created consumer group",
                        extra={"stream": self.stream, "group": self.group})
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    # ------------------------------------------------------------------ read

    def consume(self) -> Iterator[RawMessage]:
        self.ensure_group()
        self._running = True

        while self._running:
            # Rescue anything a dead worker was holding before taking new work,
            # otherwise those entries sit pending forever.
            for message in self._claim_stale():
                yield message
                if not self._running:
                    return

            try:
                response = self.redis.xreadgroup(
                    groupname=self.group,
                    consumername=self.consumer,
                    streams={self.stream: ">"},
                    count=self.batch_size,
                    block=self.block_ms,
                )
            except Exception as exc:
                logger.error("xreadgroup failed", extra={"error": str(exc)})
                continue

            if not response:
                continue

            for _stream, entries in response:
                for entry_id, fields in entries:
                    message = self._build(entry_id, fields, delivery_count=1)
                    if message is not None:
                        yield message
                    if not self._running:
                        return

    def _claim_stale(self) -> list[RawMessage]:
        try:
            cursor, entries, _ = self.redis.xautoclaim(
                name=self.stream,
                groupname=self.group,
                consumername=self.consumer,
                min_idle_time=self.claim_min_idle_ms,
                start_id=self._claim_cursor,
                count=self.batch_size,
            )
        except Exception as exc:
            logger.error("xautoclaim failed", extra={"error": str(exc)})
            return []

        self._claim_cursor = cursor.decode() if isinstance(cursor, bytes) else cursor
        if not entries:
            self._claim_cursor = "0-0"
            return []

        counts = self._delivery_counts([e[0] for e in entries])
        out: list[RawMessage] = []
        for entry_id, fields in entries:
            key = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
            message = self._build(entry_id, fields, delivery_count=counts.get(key, 1))
            if message is not None:
                out.append(message)
        if out:
            logger.info("reclaimed stale messages", extra={"count": len(out)})
        return out

    def _delivery_counts(self, entry_ids: list[Any]) -> dict[str, int]:
        """How many times each reclaimed entry has been delivered.

        Drives the poison-message cutoff: without it a handler that always
        throws would be retried until the end of time.
        """
        if not entry_ids:
            return {}
        ids = [e.decode() if isinstance(e, bytes) else e for e in entry_ids]
        try:
            pending = self.redis.xpending_range(
                name=self.stream,
                groupname=self.group,
                min=min(ids),
                max=max(ids),
                count=len(ids),
            )
        except Exception:
            return {}
        result: dict[str, int] = {}
        for item in pending:
            mid = item.get("message_id")
            mid = mid.decode() if isinstance(mid, bytes) else mid
            result[str(mid)] = int(item.get("times_delivered", 1))
        return result

    def _build(
        self, entry_id: Any, fields: dict[Any, Any], delivery_count: int
    ) -> RawMessage | None:
        key = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
        try:
            body = _decode_body(fields)
        except Exception as exc:
            # Unparseable bytes will never become parseable. Park it now.
            logger.error("undecodable stream entry, dead-lettering",
                         extra={"entry_id": key, "error": str(exc)})
            self._raw_dead_letter(key, fields, f"decode error: {exc}")
            return None
        return RawMessage(message_id=key, body=body, delivery_count=delivery_count, raw=fields)

    # ----------------------------------------------------------------- write

    def ack(self, message: RawMessage) -> None:
        try:
            self.redis.xack(self.stream, self.group, message.message_id)
            if self.delete_on_ack:
                self.redis.xdel(self.stream, message.message_id)
        except Exception as exc:
            logger.error("ack failed", extra={"entry_id": message.message_id, "error": str(exc)})

    def nack(self, message: RawMessage) -> None:
        # Leaving the entry unacked keeps it in the pending list, where
        # xautoclaim picks it up once it goes idle. No explicit requeue needed.
        logger.info(
            "message left pending for redelivery",
            extra={"entry_id": message.message_id, "delivery_count": message.delivery_count},
        )

    def dead_letter(self, message: RawMessage, reason: str) -> None:
        self._raw_dead_letter(message.message_id, message.raw or {}, reason, message.body)
        self.ack(message)

    def _raw_dead_letter(
        self,
        entry_id: str,
        fields: dict[Any, Any],
        reason: str,
        body: dict[str, Any] | None = None,
    ) -> None:
        try:
            payload = json.dumps(body) if body is not None else json.dumps(
                {k.decode() if isinstance(k, bytes) else k:
                 v.decode(errors="replace") if isinstance(v, bytes) else v
                 for k, v in fields.items()}
            )
            self.redis.xadd(
                self.dlq_stream,
                {"data": payload, "reason": reason, "origin_id": entry_id},
                maxlen=10_000,
                approximate=True,
            )
            logger.error("dead-lettered", extra={"entry_id": entry_id, "reason": reason})
        except Exception as exc:  # pragma: no cover
            logger.critical("could not write to DLQ, message will be lost",
                            extra={"entry_id": entry_id, "error": str(exc)})

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        try:
            self.redis.close()
        except Exception:  # pragma: no cover
            pass


class RedisStreamPublisher:
    def __init__(self, url: str, maxlen: int = 10_000) -> None:
        import redis

        self.redis = redis.Redis.from_url(url)
        self.maxlen = maxlen

    def publish(self, destination: str, payload: dict[str, Any]) -> None:
        self.redis.xadd(
            destination,
            {"data": json.dumps(payload, default=str)},
            maxlen=self.maxlen,
            approximate=True,
        )

    def close(self) -> None:
        try:
            self.redis.close()
        except Exception:  # pragma: no cover
            pass
