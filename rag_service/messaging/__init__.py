from __future__ import annotations

from ..config import MessagingSettings
from .base import Consumer, Publisher, RawMessage
from .events import (
    ConversationDelete,
    DocumentDelete,
    DocumentIngest,
    Envelope,
    EventType,
    InvalidEvent,
    MessageIndex,
    RetrievalRequest,
    parse_event,
)
from .idempotency import IdempotencyGuard, NullIdempotencyGuard

__all__ = [
    "Consumer",
    "ConversationDelete",
    "DocumentDelete",
    "DocumentIngest",
    "Envelope",
    "EventType",
    "IdempotencyGuard",
    "InvalidEvent",
    "MessageIndex",
    "NullIdempotencyGuard",
    "Publisher",
    "RawMessage",
    "RetrievalRequest",
    "build_consumer",
    "build_publisher",
    "parse_event",
]


def build_consumer(settings: MessagingSettings) -> Consumer:
    if settings.backend == "gcp":
        from .gcp_pubsub import GcpPubSubConsumer

        return GcpPubSubConsumer(
            project=settings.gcp_project,
            subscription=settings.gcp_subscription,
            batch_size=settings.batch_size,
        )

    from .redis_streams import RedisStreamsConsumer

    return RedisStreamsConsumer(
        url=settings.redis_url,
        stream=settings.stream,
        group=settings.group,
        consumer_name=settings.consumer_name,
        dlq_stream=settings.dlq_stream,
        batch_size=settings.batch_size,
        block_ms=settings.block_ms,
        claim_min_idle_ms=settings.claim_min_idle_ms,
        max_deliveries=settings.max_deliveries,
    )


def build_publisher(settings: MessagingSettings) -> Publisher:
    if settings.backend == "gcp":
        from .gcp_pubsub import GcpPubSubPublisher

        return GcpPubSubPublisher(project=settings.gcp_project)

    from .redis_streams import RedisStreamPublisher

    return RedisStreamPublisher(url=settings.redis_url)
