"""SSE consumer for one entity's realtime channel.

WHAT CHANGED, AND WHY
---------------------
This used to subscribe to the platform's Redis directly, on the grounds that a
backend service had no reason to go through an HTTP bridge to receive frames
already sitting on a channel it could reach. That reasoning was sound while the
alternative was Node's browser-facing endpoint, which wanted a JWT in a URL
path.

It stopped being sound once `developer_service` existed. Subscribing directly
meant this pipeline held the platform's Redis credentials - a second secret,
with far more reach than the one credential it now carries, since a Redis
connection can read every entity's channel and write to any of them. Going
through the API means the stream is scoped to whoever the token belongs to, and
that scoping is enforced somewhere this service cannot bypass.

THE TRADE-OFF THAT DID NOT CHANGE
---------------------------------
The transport is still fire-and-forget. Frames published while the bot is
disconnected are gone; there is no replay, unlike the Streams transport used
for the RAG bus. That remains acceptable because the frames are notifications,
not the record - they say "something happened in this conversation" and carry
no message text - so a missed one is recovered by reading the API, not by
replaying the stream.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# The stream's own opening frame, not a platform event. Skipped rather than
# yielded so downstream sees only real frames.
READY_EVENT = "ready"


class EntityEventConsumer:
    """Streams `GET /v1/events` from developer_service, forever."""

    def __init__(
        self,
        base_url: str,
        token: str,
        reconnect_delay_seconds: float = 2.0,
        max_reconnect_delay_seconds: float = 30.0,
        read_timeout_seconds: float = 90.0,
    ) -> None:
        if not base_url:
            raise ValueError("PLATFORM_API_BASE_URL is required to stream events")
        if not token:
            raise ValueError("PLATFORM_TOKEN is required to stream events")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.reconnect_delay = reconnect_delay_seconds
        self.max_reconnect_delay = max_reconnect_delay_seconds
        # Comfortably longer than the server's heartbeat, so a quiet stream is
        # not mistaken for a dead one - but finite, so a connection that has
        # silently gone away is eventually noticed and remade.
        self.read_timeout = read_timeout_seconds
        self._running = False
        self._response: Any = None

    def consume(self) -> Iterator[dict[str, Any]]:
        """Yield raw platform envelopes forever, reconnecting on failure.

        Reconnects with backoff rather than dying: the server caps a single
        stream's lifetime, so a clean disconnect is EXPECTED roughly hourly and
        must read as routine rather than as an error.
        """
        self._running = True
        delay = self.reconnect_delay

        while self._running:
            try:
                for envelope in self._stream_once():
                    delay = self.reconnect_delay  # reset only on real traffic
                    yield envelope
                if self._running:
                    logger.info("event stream ended, reconnecting")
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    # A bad token or a missing scope will be exactly as bad on
                    # the next attempt. Retrying it turns a misconfiguration
                    # into a burst of traffic that looks like an attack.
                    logger.error(
                        "event stream rejected; not retrying",
                        extra={"status": exc.code},
                    )
                    return
                logger.error(
                    "event stream failed, reconnecting",
                    extra={"status": exc.code, "retry_in_s": delay},
                )
            except Exception as exc:
                if not self._running:
                    break
                logger.error(
                    "event stream failed, reconnecting",
                    extra={"error": str(exc), "retry_in_s": delay},
                )
            finally:
                self._close_response()

            if not self._running:
                break
            time.sleep(delay)
            delay = min(delay * 2, self.max_reconnect_delay)

    def _stream_once(self) -> Iterator[dict[str, Any]]:
        request = urllib.request.Request(
            f"{self.base_url}/v1/events",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "text/event-stream",
                # Any proxy that buffers would hold frames until its buffer
                # fills, which on a quiet stream means a mention arriving
                # minutes late.
                "Cache-Control": "no-cache",
                "User-Agent": "chatterloop-rag/1.0",
            },
        )
        self._response = urllib.request.urlopen(request, timeout=self.read_timeout)
        logger.info("subscribed to event stream", extra={"url": self.base_url})

        event_name = "message"
        for raw_line in self._response:
            if not self._running:
                break
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")

            if line.startswith(":"):
                # A comment - the server's keepalive. Proof of life, no payload.
                continue
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
                continue
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                if event_name == READY_EVENT:
                    event_name = "message"
                    continue
                envelope = self._decode(payload)
                event_name = "message"
                if envelope is not None:
                    yield envelope
                continue
            # A blank line terminates an event; anything else (id:, retry:) is
            # valid SSE this consumer has no use for.

    @staticmethod
    def _decode(data: str) -> dict[str, Any] | None:
        if not data:
            return None
        try:
            parsed = json.loads(data)
        except Exception as exc:
            # Another service's firehose. One bad frame is not our problem to
            # solve, and must not interrupt the ones after it.
            logger.warning("undecodable frame on event stream", extra={"error": str(exc)})
            return None
        return parsed if isinstance(parsed, dict) else None

    def stop(self) -> None:
        self._running = False
        self._close_response()

    def _close_response(self) -> None:
        if self._response is not None:
            try:
                self._response.close()
            except Exception:  # pragma: no cover
                pass
            self._response = None

    def close(self) -> None:
        self.stop()
