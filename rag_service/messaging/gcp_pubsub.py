"""Google Cloud Pub/Sub transport.

Synchronous pull rather than the streaming-pull callback API: the service loop
is a plain iterator over messages, and pulling keeps backpressure in our hands
instead of the client library's thread pool.

Redelivery, DLQ and retry policy are configured on the subscription itself
(`--max-delivery-attempts`, `--dead-letter-topic`), so `dead_letter()` here only
needs to record the reason and ack.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from .base import RawMessage

logger = logging.getLogger(__name__)


class GcpPubSubConsumer:
    def __init__(
        self,
        project: str,
        subscription: str,
        batch_size: int = 16,
        deadline_seconds: int = 60,
    ) -> None:
        from google.cloud import pubsub_v1

        if not project or not subscription:
            raise ValueError("BUS_GCP_PROJECT and BUS_GCP_SUBSCRIPTION are required")

        self._client = pubsub_v1.SubscriberClient()
        self._path = self._client.subscription_path(project, subscription)
        self.batch_size = batch_size
        self.deadline_seconds = deadline_seconds
        self._running = False

    def consume(self) -> Iterator[RawMessage]:
        self._running = True
        while self._running:
            try:
                response = self._client.pull(
                    request={"subscription": self._path, "max_messages": self.batch_size},
                    timeout=30.0,
                )
            except Exception as exc:
                if self._running:
                    logger.error("pubsub pull failed", extra={"error": str(exc)})
                continue

            for received in response.received_messages:
                try:
                    body = json.loads(received.message.data.decode("utf-8"))
                except Exception as exc:
                    logger.error("undecodable pubsub message", extra={"error": str(exc)})
                    self._ack_id(received.ack_id)
                    continue

                yield RawMessage(
                    message_id=received.message.message_id,
                    body=body,
                    delivery_count=int(getattr(received, "delivery_attempt", 0) or 1),
                    raw=received.ack_id,
                )
                if not self._running:
                    return

    def ack(self, message: RawMessage) -> None:
        self._ack_id(message.raw)

    def _ack_id(self, ack_id: Any) -> None:
        try:
            self._client.acknowledge(request={"subscription": self._path, "ack_ids": [ack_id]})
        except Exception as exc:
            logger.error("pubsub ack failed", extra={"error": str(exc)})

    def nack(self, message: RawMessage) -> None:
        # Zero deadline tells Pub/Sub to redeliver immediately.
        try:
            self._client.modify_ack_deadline(
                request={
                    "subscription": self._path,
                    "ack_ids": [message.raw],
                    "ack_deadline_seconds": 0,
                }
            )
        except Exception as exc:
            logger.error("pubsub nack failed", extra={"error": str(exc)})

    def dead_letter(self, message: RawMessage, reason: str) -> None:
        logger.error(
            "dead-lettering message",
            extra={"message_id": message.message_id, "reason": reason},
        )
        self.ack(message)

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover
            pass


class GcpPubSubPublisher:
    def __init__(self, project: str) -> None:
        from google.cloud import pubsub_v1

        self._client = pubsub_v1.PublisherClient()
        self._project = project

    def publish(self, destination: str, payload: dict[str, Any]) -> None:
        topic = self._client.topic_path(self._project, destination)
        future = self._client.publish(topic, json.dumps(payload, default=str).encode("utf-8"))
        future.result(timeout=30)

    def close(self) -> None:
        return None
