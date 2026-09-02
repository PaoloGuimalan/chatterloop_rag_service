"""The realtime seam: SSE frames in, platform envelopes out.

The parsing is small but the failure mode is not - a consumer that silently
drops frames looks exactly like a quiet platform, and the bot would just never
answer anyone.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from rag_service.chatterloop import consumer as consumer_module
from rag_service.chatterloop.consumer import EntityEventConsumer

TOKEN = "clt_" + "a" * 12 + "_" + "b" * 64


class FakeResponse:
    """Yields SSE lines the way urlopen's file object does."""

    def __init__(self, lines):
        self._lines = [line.encode("utf-8") for line in lines]
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


def envelope(event="messages_list", conversation_id="c1"):
    return {
        "logType": None,
        "pod": "pod-1",
        "event": event,
        "message": {
            "status": True,
            "auth": True,
            "onseen": False,
            "message": {"conversationID": conversation_id, "entityID": "e1"},
            "result": "",
        },
        "dateTime": "2024-01-01T00:00:00Z",
    }


def sse(*frames):
    """Render frames as an SSE body, the way developer_service writes them."""
    lines = ["event: ready\n", 'data: {"entity_id":"e1"}\n', "\n"]
    for name, payload in frames:
        lines += [f"event: {name}\n", f"data: {json.dumps(payload)}\n", "\n"]
    return lines


def _install(monkeypatch, subject, responses):
    """Serve each response in turn, then end the loop.

    `consume()` sets `_running = True` on entry and retries transport failures
    forever, which is correct in production and an infinite loop in a test. So
    the fake stops the consumer once the script is exhausted - that is what
    makes these terminate, and it is why the consumer is built before this is
    installed.
    """
    calls = {"n": 0, "requests": []}

    def fake_urlopen(request, timeout=None):
        calls["requests"].append(request)
        index = calls["n"]
        calls["n"] += 1
        if index >= len(responses):
            subject.stop()
            raise urllib.error.URLError("script exhausted")
        result = responses[index]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(consumer_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(consumer_module.time, "sleep", lambda _s: None)
    return calls


def _consumer():
    return EntityEventConsumer(base_url="https://api.example", token=TOKEN)


class TestConstruction:
    def test_a_base_url_is_required(self):
        with pytest.raises(ValueError):
            EntityEventConsumer(base_url="", token=TOKEN)

    def test_a_token_is_required(self):
        with pytest.raises(ValueError):
            EntityEventConsumer(base_url="https://x", token="")


class TestStreaming:
    def test_frames_are_yielded_as_platform_envelopes(self, monkeypatch):
        subject = _consumer()
        _install(monkeypatch, subject, [FakeResponse(sse(("messages_list", envelope())))])

        received = []
        for frame in subject.consume():
            received.append(frame)
            subject.stop()

        assert len(received) == 1
        assert received[0]["event"] == "messages_list"
        assert received[0]["message"]["message"]["conversationID"] == "c1"

    def test_the_ready_frame_is_not_yielded(self, monkeypatch):
        # It is the stream announcing itself, not a platform event. Passing it
        # downstream would look like an event with no type.
        subject = _consumer()
        _install(monkeypatch, subject, [FakeResponse(sse())])
        assert list(subject.consume()) == []

    def test_keepalive_comments_are_ignored(self, monkeypatch):
        body = [": keepalive\n", "\n"] + sse(("messages_list", envelope()))
        subject = _consumer()
        _install(monkeypatch, subject, [FakeResponse(body)])

        received = []
        for frame in subject.consume():
            received.append(frame)
            subject.stop()
        assert len(received) == 1

    def test_one_undecodable_frame_does_not_stop_the_ones_after_it(self, monkeypatch):
        body = sse() + [
            "event: messages_list\n", "data: {not json\n", "\n",
            "event: messages_list\n", f"data: {json.dumps(envelope())}\n", "\n",
        ]
        subject = _consumer()
        _install(monkeypatch, subject, [FakeResponse(body)])

        received = []
        for frame in subject.consume():
            received.append(frame)
            subject.stop()
        assert len(received) == 1

    def test_the_token_travels_as_a_bearer_header(self, monkeypatch):
        subject = _consumer()
        calls = _install(monkeypatch, subject, [FakeResponse(sse())])
        list(subject.consume())
        assert calls["requests"][0].get_header("Authorization") == f"Bearer {TOKEN}"
        assert TOKEN not in calls["requests"][0].full_url


class TestReconnection:
    def test_a_closed_stream_is_reconnected(self, monkeypatch):
        # The server caps a stream's lifetime, so a clean end is EXPECTED
        # roughly hourly and must be routine rather than fatal.
        first = FakeResponse(sse(("messages_list", envelope(conversation_id="c1"))))
        second = FakeResponse(sse(("messages_list", envelope(conversation_id="c2"))))
        subject = _consumer()
        _install(monkeypatch, subject, [first, second])

        seen = []
        for frame in subject.consume():
            seen.append(frame["message"]["message"]["conversationID"])
            if len(seen) == 2:
                subject.stop()
        assert seen == ["c1", "c2"]

    @pytest.mark.parametrize("status", [401, 403])
    def test_a_rejected_token_stops_rather_than_retrying(self, monkeypatch, status):
        # Retrying an auth failure turns one misconfigured token into a burst
        # of traffic that looks like an attack.
        error = urllib.error.HTTPError("http://x", status, "denied", {}, None)
        subject = _consumer()
        calls = _install(monkeypatch, subject, [error, FakeResponse(sse())])
        assert list(subject.consume()) == []
        assert calls["n"] == 1

    def test_a_transport_failure_is_retried(self, monkeypatch):
        subject = _consumer()
        calls = _install(monkeypatch, subject, [
            urllib.error.URLError("refused"),
            FakeResponse(sse(("messages_list", envelope()))),
        ])
        for _frame in subject.consume():
            subject.stop()
        assert calls["n"] == 2

    def test_the_response_is_closed_on_the_way_out(self, monkeypatch):
        response = FakeResponse(sse(("messages_list", envelope())))
        subject = _consumer()
        _install(monkeypatch, subject, [response])
        for _frame in subject.consume():
            subject.stop()
        assert response.closed
