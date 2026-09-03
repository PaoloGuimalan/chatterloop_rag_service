"""Executes one function call the model asked for.

Mirrors NeonCentralized_API's `api_call`/`trigger_function`
(llm/utils/function_calls.py) - same three request shapes (GET+query,
GET-or-POST+route-templated URL, POST+JSON body) and the same rule that a
failed call becomes a JSON error object handed back to the model, never an
exception that reaches the caller. A tool is something the model can be
WRONG about calling; it must never be something that can take the process
down.

Speaks the current OpenAI `tools`/`tool_choice` function-calling schema
(`{"type": "function", "function": {...}}`), not the `functions`/
`function_call` one Neon's code uses - that shape is deprecated on OpenAI's
side, and reintroducing it into new code would risk exactly the kind of
silent breakage this rewrite exists to rule out. The request/response
CONTRACT with the model is otherwise the same: name in, JSON arguments in,
one JSON result out per call.

urllib, not requests/httpx - same reasoning as
chatterloop/platform/client.py: this is a handful of GETs and POSTs, and
rag_service does not carry an HTTP client dependency for that.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..config import ToolConfig

logger = logging.getLogger(__name__)


def to_openai_tool(tool: ToolConfig) -> dict[str, Any]:
    """This tool's entry in the `tools=[...]` list passed to the API."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        },
    }


def call_tool(tool: ToolConfig, arguments: dict[str, Any], *, timeout: float) -> Any:
    """Runs one tool call over HTTP and returns something JSON-serialisable.

    Never raises. Every failure mode - a bad route template, a network
    error, a non-2xx response, a body that isn't JSON - becomes a value the
    model can read and react to (apologise, try different arguments, give up
    gracefully) rather than an exception unwinding out of a chat reply.
    """
    method = tool.http_method.upper()
    url = tool.api_endpoint
    headers = dict(tool.headers)
    body: bytes | None = None

    try:
        if tool.param_type == "route":
            # KeyError (a placeholder the model didn't fill) and
            # IndexError/ValueError (a malformed template) are argument
            # problems, not infra ones - caught below with everything else
            # that means "tell the model, don't crash the bot".
            url = url.format(**arguments)
        elif method == "GET":
            if arguments:
                url = f"{url}?{urllib.parse.urlencode(arguments)}"
        else:
            body = json.dumps(arguments).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")

        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")

    except urllib.error.HTTPError as exc:
        detail = _safe_body(exc)
        logger.warning(
            "tool call failed",
            extra={"tool": tool.name, "status": exc.code, "detail": detail},
        )
        return {"error": f"{tool.name} failed ({exc.code}): {detail}"}
    except urllib.error.URLError as exc:
        logger.warning("tool call unreachable", extra={"tool": tool.name, "error": str(exc.reason)})
        return {"error": f"{tool.name} unreachable: {exc.reason}"}
    except (TimeoutError, KeyError, IndexError, ValueError) as exc:
        logger.warning("tool call failed", extra={"tool": tool.name, "error": str(exc)})
        return {"error": f"{tool.name} failed: {exc}"}

    if not payload:
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # Same fallback Neon's api_call() uses: a tool need not return JSON,
        # only something the model can read.
        return payload


def _safe_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")[:500]
    except Exception:  # pragma: no cover - defensive
        return ""
