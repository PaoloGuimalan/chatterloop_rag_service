"""Milvus boolean-expression construction.

Milvus filters are a string DSL, which means unescaped values are an injection
surface exactly like SQL. A tenant id containing a quote could otherwise close
the literal and widen the filter - which in a multi-tenant vector store means
reading another organisation's data. Every value goes through `quote()`.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..domain import Scope


class FilterError(ValueError):
    pass


def quote(value: str) -> str:
    """Render a Python string as a safe Milvus string literal."""
    if not isinstance(value, str):
        raise FilterError(f"expected str, got {type(value).__name__}")
    if "\x00" in value:
        raise FilterError("null bytes are not permitted in filter values")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_filter(
    tenant_id: str,
    scopes: Iterable[Scope] | None = None,
    conversation_id: str | None = None,
    source_ids: Iterable[str] | None = None,
) -> str:
    """Compose the retrieval filter.

    `tenant_id` is mandatory and always the first conjunct - there is no code
    path that produces a filter without it. That is the whole tenant-isolation
    guarantee, so it is enforced here rather than left to callers.

    Chat chunks are additionally scoped to a single conversation; document
    chunks are shared across the tenant. Expressed as:

        tenant == T and ( scope == "doc" or (scope == "chat" and conv == C) )
    """
    if not tenant_id or not tenant_id.strip():
        raise FilterError("tenant_id is required: refusing to build an unscoped filter")

    clauses = [f"tenant_id == {quote(tenant_id)}"]

    scope_list = list(scopes) if scopes is not None else [Scope.DOCUMENT, Scope.CHAT]
    if not scope_list:
        raise FilterError("at least one scope is required")

    parts: list[str] = []
    for scope in scope_list:
        if scope is Scope.CHAT:
            if conversation_id:
                parts.append(
                    f'(scope == "chat" and conversation_id == {quote(conversation_id)})'
                )
            else:
                # No conversation given: chat history is not tenant-wide
                # readable, so this scope contributes nothing rather than
                # everything.
                continue
        else:
            parts.append(f"scope == {quote(str(scope))}")

    if not parts:
        raise FilterError(
            "resulting filter would match nothing; pass conversation_id with the chat scope"
        )

    clauses.append(parts[0] if len(parts) == 1 else "(" + " or ".join(parts) + ")")

    if source_ids is not None:
        ids = [quote(s) for s in source_ids]
        if not ids:
            raise FilterError("source_ids was provided but empty")
        clauses.append(f"source_id in [{', '.join(ids)}]")

    return " and ".join(clauses)


def delete_filter(
    tenant_id: str,
    source_id: str | None = None,
    conversation_id: str | None = None,
) -> str:
    """Filter for removal. Also tenant-locked, for the same reason."""
    if not tenant_id or not tenant_id.strip():
        raise FilterError("tenant_id is required for deletion")
    if not source_id and not conversation_id:
        raise FilterError("refusing to delete an entire tenant: pass source_id or conversation_id")

    clauses = [f"tenant_id == {quote(tenant_id)}"]
    if source_id:
        clauses.append(f"source_id == {quote(source_id)}")
    if conversation_id:
        clauses.append(f"conversation_id == {quote(conversation_id)}")
    return " and ".join(clauses)
