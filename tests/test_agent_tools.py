"""Tool execution: the three request shapes, and that a failure never raises."""

from __future__ import annotations

import io
import json
import urllib.error

from rag_service.agent import tools as tools_module
from rag_service.agent.tools import call_tool, to_openai_tool
from rag_service.config import ToolConfig


class _Response:
    def __init__(self, payload: str):
        self._payload = payload

    def read(self):
        return self._payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code: int, body: str = "denied"):
    return urllib.error.HTTPError(
        "http://x/y", code, "err", {}, io.BytesIO(body.encode("utf-8"))
    )


def _install(monkeypatch, behaviour):
    """Replace urlopen, recording every request it is handed."""
    seen = []

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        result = behaviour(request)
        if isinstance(result, Exception):
            raise result
        return _Response(result)

    monkeypatch.setattr(tools_module.urllib.request, "urlopen", fake_urlopen)
    return seen


def test_to_openai_tool_shape():
    tool = ToolConfig(
        name="lookup_order",
        description="Looks up an order by id.",
        parameters_schema={"type": "object", "properties": {"order_id": {"type": "string"}}},
        api_endpoint="https://api.example/orders/{order_id}",
        param_type="route",
    )
    schema = to_openai_tool(tool)
    assert schema == {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Looks up an order by id.",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
        },
    }


def test_query_param_call(monkeypatch):
    tool = ToolConfig(
        name="search",
        api_endpoint="https://api.example/search",
        http_method="GET",
        param_type="query",
    )
    seen = _install(monkeypatch, lambda req: json.dumps({"ok": True}))

    result = call_tool(tool, {"q": "widgets"}, timeout=5.0)

    assert result == {"ok": True}
    assert seen[0].full_url == "https://api.example/search?q=widgets"
    assert seen[0].get_method() == "GET"


def test_route_param_call(monkeypatch):
    tool = ToolConfig(
        name="lookup_order",
        api_endpoint="https://api.example/orders/{order_id}",
        http_method="GET",
        param_type="route",
    )
    seen = _install(monkeypatch, lambda req: json.dumps({"status": "shipped"}))

    result = call_tool(tool, {"order_id": "42"}, timeout=5.0)

    assert result == {"status": "shipped"}
    assert seen[0].full_url == "https://api.example/orders/42"


def test_body_param_call_sends_json(monkeypatch):
    tool = ToolConfig(
        name="create_ticket",
        api_endpoint="https://api.example/tickets",
        http_method="POST",
        param_type="body",
    )
    seen = _install(monkeypatch, lambda req: json.dumps({"id": "t_1"}))

    result = call_tool(tool, {"title": "Broken widget"}, timeout=5.0)

    assert result == {"id": "t_1"}
    request = seen[0]
    assert request.get_method() == "POST"
    assert json.loads(request.data) == {"title": "Broken widget"}
    assert request.get_header("Content-type") == "application/json"


def test_static_headers_are_sent(monkeypatch):
    tool = ToolConfig(
        name="secure_call",
        api_endpoint="https://api.example/secure",
        http_method="GET",
        param_type="query",
        headers={"Authorization": "Bearer static-token"},
    )
    seen = _install(monkeypatch, lambda req: "{}")

    call_tool(tool, {}, timeout=5.0)

    assert seen[0].get_header("Authorization") == "Bearer static-token"


def test_route_template_missing_argument_is_reported_not_raised():
    tool = ToolConfig(
        name="lookup_order",
        api_endpoint="https://api.example/orders/{order_id}",
        http_method="GET",
        param_type="route",
    )

    result = call_tool(tool, {}, timeout=5.0)

    assert "error" in result
    assert "lookup_order" in result["error"]


def test_http_error_is_reported_not_raised(monkeypatch):
    tool = ToolConfig(
        name="flaky",
        api_endpoint="https://api.example/flaky",
        http_method="GET",
        param_type="query",
    )
    _install(monkeypatch, lambda req: _http_error(500, "boom"))

    result = call_tool(tool, {}, timeout=5.0)

    assert result == {"error": "flaky failed (500): boom"}


def test_connection_failure_is_reported_not_raised(monkeypatch):
    tool = ToolConfig(
        name="unreachable",
        api_endpoint="https://api.example/x",
        http_method="GET",
        param_type="query",
    )
    _install(monkeypatch, lambda req: urllib.error.URLError("no route"))

    result = call_tool(tool, {}, timeout=5.0)

    assert "unreachable" in result["error"]


def test_non_json_response_is_returned_as_text(monkeypatch):
    tool = ToolConfig(
        name="plain",
        api_endpoint="https://api.example/x",
        http_method="GET",
        param_type="query",
    )
    _install(monkeypatch, lambda req: "not json")

    result = call_tool(tool, {}, timeout=5.0)

    assert result == "not json"


def test_empty_response_is_empty_dict(monkeypatch):
    tool = ToolConfig(
        name="empty",
        api_endpoint="https://api.example/x",
        http_method="POST",
        param_type="body",
    )
    _install(monkeypatch, lambda req: "")

    result = call_tool(tool, {}, timeout=5.0)

    assert result == {}


def test_tool_config_from_json_env(monkeypatch):
    """The exact env-parsing path CHATTERLOOP_TOOLS goes through."""
    from rag_service.config import ChatterloopSettings

    raw = json.dumps(
        [
            {
                "name": "weather",
                "description": "Current weather for a city.",
                "api_endpoint": "https://api.example/weather",
                "http_method": "GET",
                "param_type": "query",
            }
        ]
    )
    monkeypatch.setenv("CHATTERLOOP_TOOLS", raw)
    monkeypatch.setenv("CHATTERLOOP_TOOLS_ENABLED", "true")

    settings = ChatterloopSettings()

    assert settings.tools_enabled is True
    assert len(settings.tools) == 1
    assert settings.tools[0].name == "weather"
    assert settings.tools[0].is_enabled is True


def test_tool_config_disabled_by_default():
    from rag_service.config import ChatterloopSettings

    settings = ChatterloopSettings()

    assert settings.tools_enabled is False
    assert settings.tools == []
