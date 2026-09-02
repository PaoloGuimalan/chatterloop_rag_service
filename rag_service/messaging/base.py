from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable


@dataclass(slots=True)
class RawMessage:
    """One delivery, before parsing."""

    message_id: str
    body: dict[str, Any]
    # Redis and Pub/Sub both expose this. Used to route poison messages to the
    # DLQ instead of retrying them forever.
    delivery_count: int = 1
    raw: Any = field(default=None, repr=False)


@runtime_checkable
class Consumer(Protocol):
    def consume(self) -> Iterator[RawMessage]:
        """Yield messages until `stop()` is called. Blocks when idle."""
        ...

    def ack(self, message: RawMessage) -> None: ...

    def nack(self, message: RawMessage) -> None:
        """Return a message for redelivery."""
        ...

    def dead_letter(self, message: RawMessage, reason: str) -> None:
        """Park a message that will never succeed, and stop redelivering it."""
        ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class Publisher(Protocol):
    def publish(self, destination: str, payload: dict[str, Any]) -> None: ...

    def close(self) -> None: ...
