"""The HTTP client for chatterloop's developer API.

ONE HOST, ONE CREDENTIAL
------------------------
Everything this pipeline needs is on `developer_service`: conversation history,
comment mentions, sending a reply, and the realtime stream. It holds one
`entity_token` and talks to one origin - no database credentials, no Redis
credentials, and no second base URL to keep in step.

WHY urllib AND NOT httpx/requests
---------------------------------
Two GETs and a POST. Adding a dependency to a service that is otherwise built
on stdlib for this would be a poor trade, and this cut-over is already REMOVING
two heavier ones (pymongo, psycopg2). Retries come from `tenacity`, which the
service already depends on.

WHAT IS RETRIED, AND WHAT IS NOT
--------------------------------
Timeouts, connection failures and 5xx are retried - they are the platform
having a bad moment. 401 and 403 are not: a bad token or a missing scope will
be exactly as bad on the fourth attempt, and retrying an auth failure turns a
misconfiguration into a burst of traffic that looks like an attack.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class PlatformAPIError(RuntimeError):
    """The developer API could not answer."""


class PlatformAuthError(PlatformAPIError):
    """The token was rejected, or lacks the scope. Never retried."""


class PlatformTransientError(PlatformAPIError):
    """A timeout or 5xx. Retried."""


class BotApiClient:
    """Token-authenticated access to the developer API."""

    def __init__(
        self,
        token: str,
        base_url: str,
        timeout: float = 15.0,
        max_attempts: int = 3,
    ) -> None:
        if not token:
            raise ValueError("PLATFORM_TOKEN is required to reach the developer API")
        if not base_url:
            raise ValueError("PLATFORM_API_BASE_URL is required")
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_attempts = max_attempts

    # ------------------------------------------------------------ requests --

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        return self._request("GET", url)

    def post(self, path: str, body: dict[str, Any]) -> dict:
        return self._request("POST", f"{self.base_url}{path}", body=body)

    def _request(self, method: str, url: str, body: dict | None = None) -> dict:
        # The retry wrapper is built per call rather than as a decorator so
        # that max_attempts stays configurable per client instance.
        runner = retry(
            retry=retry_if_exception_type(PlatformTransientError),
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            reraise=True,
        )(self._request_once)
        return runner(method, url, body)

    def _request_once(self, method: str, url: str, body: dict | None) -> dict:
        data = None
        headers = {
            # NOT x-access-token: that is the user path's header, and a
            # credential that cannot be presented on the wrong door cannot be
            # accepted by it by mistake.
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "User-Agent": "chatterloop-rag/1.0",
        }
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = _safe_body(exc)
            if exc.code in (401, 403):
                raise PlatformAuthError(
                    f"{method} {_scrub(url)} rejected ({exc.code}): {detail}"
                ) from exc
            if exc.code >= 500:
                raise PlatformTransientError(
                    f"{method} {_scrub(url)} failed ({exc.code}): {detail}"
                ) from exc
            raise PlatformAPIError(
                f"{method} {_scrub(url)} failed ({exc.code}): {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise PlatformTransientError(
                f"{method} {_scrub(url)} unreachable: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise PlatformTransientError(f"{method} {_scrub(url)} timed out") from exc

        if not payload:
            return {}
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise PlatformAPIError(
                f"{method} {_scrub(url)} returned non-JSON: {payload[:200]}"
            ) from exc
        if not isinstance(parsed, dict):
            raise PlatformAPIError(
                f"{method} {_scrub(url)} returned {type(parsed).__name__}, expected object"
            )
        return parsed

    # ------------------------------------------------------------- helpers --

    def whoami(self) -> dict:
        """Identity and scopes for this token.

        Worth calling once at startup: it turns "the bot is silently answering
        nothing" into a loud, immediate failure when a token is wrong, expired
        or missing a scope.
        """
        return self.get("/v1/whoami")


def _safe_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8")[:300]
    except Exception:  # pragma: no cover - defensive
        return "<unreadable>"


def _scrub(url: str) -> str:
    """URLs are logged; a token must never ride in one, but be sure."""
    return url.split("?")[0]
