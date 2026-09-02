"""The two read seams, implemented against the platform's developer API.

These satisfy `MessageFetcher` and `MentionFetcher` from ports.py.

WHAT MOVED SERVER-SIDE IN THE CUT-OVER
--------------------------------------
Three things this module used to do itself now happen inside the endpoint,
because they are the owning service's knowledge and not a client's:

  * `messageDate` normalisation. The column holds either a BSON date or an
    embedded `{date, time}` depending on who wrote the row, and knowing which
    is not a fact a consumer should have to carry.
  * Handle resolution. Turning a sender entity into "@ana" meant a second
    query against `user_account` - and, done here, it silently resolved only
    users, leaving realms and bots blank.
  * Comment text. A mention arrives as a notification but its words live in
    Postgres, so resolving one used to span two datastores.

WHAT MOVED IS ALSO WHAT MADE THIS SAFE
--------------------------------------
`fetch_comment_mentions` no longer takes an entity id. It used to, and it had
to, because the query was ours to write - which meant the pipeline could in
principle read anyone's notifications. The endpoint scopes to the token's own
entity and takes no such parameter, so that is now unrepresentable rather than
merely not done.

Both fetchers stay tolerant: a row in a shape this service does not recognise
costs one skipped item, never the whole batch.
"""

from __future__ import annotations

import logging

from ..ports import PendingMention, PlatformMessage
from .client import BotApiClient, PlatformAPIError, PlatformAuthError

logger = logging.getLogger(__name__)


class ApiMessageFetcher:
    """Recent conversation history, from `GET /v1/conversations/.../messages`."""

    def __init__(self, client: BotApiClient) -> None:
        self.client = client

    def fetch_recent(self, conversation_id: str, limit: int) -> list[PlatformMessage]:
        if not conversation_id:
            return []
        try:
            payload = self.client.get(
                f"/v1/conversations/{conversation_id}/messages",
                {"limit": max(1, limit)},
            )
        except PlatformAuthError as exc:
            # Distinct from a transport failure: this is a configuration
            # problem that will not fix itself, and it should read that way in
            # the logs rather than as a flaky read.
            logger.error(
                "not permitted to read this conversation",
                extra={"conversation_id": conversation_id, "error": str(exc)},
            )
            return []
        except PlatformAPIError as exc:
            logger.error(
                "could not read conversation history",
                extra={"conversation_id": conversation_id, "error": str(exc)},
            )
            return []

        messages: list[PlatformMessage] = []
        for row in payload.get("messages") or []:
            if not isinstance(row, dict):
                continue
            message_id = row.get("message_id")
            sender = row.get("sender_entity_id")
            content = row.get("content")
            if not message_id or not sender or not isinstance(content, str):
                continue
            messages.append(
                PlatformMessage(
                    message_id=str(message_id),
                    conversation_id=str(
                        row.get("conversation_id") or conversation_id
                    ),
                    sender_entity_id=str(sender),
                    content=content,
                    created_at=_int(row.get("created_at")),
                    message_type=str(row.get("message_type") or "text"),
                    sender_handle=str(row.get("sender_handle") or ""),
                )
            )
        return messages


class ApiMentionFetcher:
    """Comment mentions, from `GET /v1/mentions/comments`.

    Takes no entity id - see the module docstring. The endpoint answers for
    whoever the token belongs to and nobody else.
    """

    def __init__(self, client: BotApiClient) -> None:
        self.client = client

    def fetch_comment_mentions(self, limit: int) -> list[PendingMention]:
        try:
            payload = self.client.get(
                "/v1/mentions/comments", {"limit": max(1, limit)}
            )
        except PlatformAuthError as exc:
            logger.error(
                "not permitted to read mentions", extra={"error": str(exc)}
            )
            return []
        except PlatformAPIError as exc:
            logger.error("could not read comment mentions", extra={"error": str(exc)})
            return []

        pending: list[PendingMention] = []
        for row in payload.get("mentions") or []:
            if not isinstance(row, dict):
                continue
            comment_id = row.get("comment_id")
            text = row.get("text")
            if not comment_id or not isinstance(text, str) or not text.strip():
                # The endpoint already drops textless mentions; this is the
                # belt to that braces, because an unanswerable trigger reaching
                # the policy is worse than one dropped twice.
                continue
            pending.append(
                PendingMention(
                    comment_id=str(comment_id),
                    post_id=str(row.get("post_id") or ""),
                    author_entity_id=str(row.get("author_entity_id") or ""),
                    text=text,
                    author_handle=str(row.get("author_handle") or ""),
                    created_at=_int(row.get("created_at")),
                )
            )
        return pending


def _int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
