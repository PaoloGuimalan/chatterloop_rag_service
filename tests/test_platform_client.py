"""The bot API client: what it sends, and what it does when the platform says no.

The retry policy is the load-bearing part. Retrying a 5xx is right; retrying a
401 turns one misconfigured token into a burst of traffic that looks like an
attack, and it will be exactly as rejected on the fourth attempt as the first.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from rag_service.chatterloop.platform import client as client_module
from rag_service.chatterloop.platform.client import (
    BotApiClient,
    PlatformAPIError,
    PlatformAuthError,
    PlatformTransientError,
)

TOKEN = "clt_" + "a" * 12 + "_" + "b" * 64


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code, body="denied"):
    return urllib.error.HTTPError(
        "http://x/y", code, "err", {}, io.BytesIO(body.encode("utf-8"))
    )


@pytest.fixture
def client():
    return BotApiClient(
        token=TOKEN,
        base_url="https://api.example",
        timeout=1.0,
        max_attempts=3,
    )


def _install(monkeypatch, behaviour):
    """Replace urlopen, recording every request it is handed."""
    seen = []

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        result = behaviour(len(seen))
        if isinstance(result, Exception):
            raise result
        return _Response(result)

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)
    return seen


class TestConstruction:
    def test_a_token_is_required(self):
        with pytest.raises(ValueError):
            BotApiClient(token="", base_url="https://x")

    def test_a_base_url_is_required(self):
        with pytest.raises(ValueError):
            BotApiClient(token=TOKEN, base_url="")

    def test_reads_and_writes_share_one_origin(self, client, monkeypatch):
        # One host, one credential: the write path is not a second base URL
        # that can drift out of step with the read path.
        seen = _install(monkeypatch, lambda n: json.dumps({"ok": True}))
        client.get("/v1/whoami")
        client.post("/v1/messages/send", {"content": "hi"})
        assert [r.full_url for r in seen] == [
            "https://api.example/v1/whoami",
            "https://api.example/v1/messages/send",
        ]


class TestRequestShape:
    def test_the_token_travels_as_a_bearer_header(self, client, monkeypatch):
        seen = _install(monkeypatch, lambda n: json.dumps({"ok": True}))
        client.get("/v1/whoami")
        [request] = seen
        assert request.get_header("Authorization") == f"Bearer {TOKEN}"

    def test_the_token_never_appears_in_the_url(self, client, monkeypatch):
        seen = _install(monkeypatch, lambda n: json.dumps({"ok": True}))
        client.get("/v1/whoami", {"limit": 5})
        assert TOKEN not in seen[0].full_url
        assert "limit=5" in seen[0].full_url

    def test_it_does_not_use_the_user_paths_header(self, client, monkeypatch):
        # x-access-token is the client credential. A bot token presented there
        # would be a credential on the wrong door.
        seen = _install(monkeypatch, lambda n: json.dumps({"ok": True}))
        client.get("/v1/whoami")
        assert seen[0].get_header("X-access-token") is None

    def test_a_post_sends_json(self, client, monkeypatch):
        seen = _install(monkeypatch, lambda n: json.dumps({"status": True}))
        client.post("/v1/messages/send", {"content": "hi"})
        [request] = seen
        assert request.method == "POST"
        assert json.loads(request.data.decode()) == {"content": "hi"}
        assert request.get_header("Content-type") == "application/json"


class TestFailureHandling:
    @pytest.mark.parametrize("code", [401, 403])
    def test_auth_failures_are_not_retried(self, client, monkeypatch, code):
        seen = _install(monkeypatch, lambda n: _http_error(code))
        with pytest.raises(PlatformAuthError):
            client.get("/v1/whoami")
        assert len(seen) == 1

    def test_server_errors_are_retried_to_the_attempt_limit(self, client, monkeypatch):
        seen = _install(monkeypatch, lambda n: _http_error(503))
        with pytest.raises(PlatformTransientError):
            client.get("/v1/whoami")
        assert len(seen) == 3

    def test_a_retry_can_succeed(self, client, monkeypatch):
        _install(
            monkeypatch,
            lambda n: _http_error(500) if n == 1 else json.dumps({"ok": True}),
        )
        assert client.get("/v1/whoami") == {"ok": True}

    def test_a_client_error_is_not_retried(self, client, monkeypatch):
        seen = _install(monkeypatch, lambda n: _http_error(404, "no such thing"))
        with pytest.raises(PlatformAPIError):
            client.get("/v1/conversations/nope/messages")
        assert len(seen) == 1

    def test_an_unreachable_host_is_transient(self, client, monkeypatch):
        seen = _install(monkeypatch, lambda n: urllib.error.URLError("refused"))
        with pytest.raises(PlatformTransientError):
            client.get("/v1/whoami")
        assert len(seen) == 3

    def test_non_json_is_an_error_not_a_silent_empty_result(self, client, monkeypatch):
        _install(monkeypatch, lambda n: "<html>502 Bad Gateway</html>")
        with pytest.raises(PlatformAPIError):
            client.get("/v1/whoami")

    def test_a_json_array_is_refused(self, client, monkeypatch):
        # Every endpoint returns an object. A list means something upstream is
        # answering that is not the bot API.
        _install(monkeypatch, lambda n: "[1, 2, 3]")
        with pytest.raises(PlatformAPIError):
            client.get("/v1/whoami")

    def test_an_error_message_does_not_leak_the_query_string(self, client, monkeypatch):
        _install(monkeypatch, lambda n: _http_error(500))
        with pytest.raises(PlatformTransientError) as excinfo:
            client.get("/v1/mentions/comments", {"limit": 5})
        assert "limit=5" not in str(excinfo.value)
