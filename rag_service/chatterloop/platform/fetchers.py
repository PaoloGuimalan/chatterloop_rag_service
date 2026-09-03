"""The read seams, implemented against the platform's developer API.

These satisfy `MessageFetcher` and `MentionFetcher` from ports.py - two reads
each, one for being named and one for being replied to.

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
    """Conversation reads, from the two `/v1/conversations/...` routes."""

    def __init__(self, client: BotApiClient) -> None:
        self.client = client

    def fetch_recent(self, conversation_id: str, limit: int) -> list[PlatformMessage]:
        return self._fetch(
            conversation_id, "messages", limit, "could not read conversation history"
        )

    def fetch_replies_to_me(
        self, conversation_id: str, limit: int
    ) -> list[PlatformMessage]:
        """Messages here that reply to one the bot wrote.

        Whose replies is NOT a parameter - the endpoint answers for the token's
        own entity. That is what keeps "reply to me without naming me" from
        widening into "reply to anyone", and it is enforced somewhere this
        service cannot reach.
        """
        return self._fetch(
            conversation_id, "replies", limit, "could not read conversation replies"
        )

    def conversation_type(self, conversation_id: str) -> str:
        """ "single" for a DM, something else otherwise - empty if unresolvable.

        Reuses the SAME endpoint `fetch_recent` does, not a dedicated one:
        `GET /v1/conversations/{id}/messages` already resolves and returns
        `conversation_type` for the membership check alone, before it even
        reads a message (LoadConversation, in developer_service's
        internal/platform/messages.go). `limit=1` asks for the one field this
        needs without paying for `history_window` messages just to read it -
        the OPPOSITE of the reply probe (`/replies`), which deliberately does
        NOT resolve this (its own comment: "a read that answers nothing is
        worth not making" - conversation_type is exactly that, on that path).
        """
        if not conversation_id:
            return ""
        try:
            payload = self.client.get(
                f"/v1/conversations/{conversation_id}/messages", {"limit": 1}
            )
        except PlatformAuthError as exc:
            logger.error(
                "not permitted to read this conversation",
                extra={"conversation_id": conversation_id, "error": str(exc)},
            )
            return ""
        except PlatformAPIError as exc:
            logger.error(
                "could not read conversation type",
                extra={"conversation_id": conversation_id, "error": str(exc)},
            )
            return ""
        value = payload.get("conversation_type")
        return value if isinstance(value, str) else ""

    def _fetch(
        self, conversation_id: str, route: str, limit: int, failure: str
    ) -> list[PlatformMessage]:
        if not conversation_id:
            return []
        try:
            payload = self.client.get(
                f"/v1/conversations/{conversation_id}/{route}",
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
                failure,
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
                    is_reply=bool(row.get("is_reply")),
                    replying_to=str(row.get("replying_to") or ""),
                    replying_to_sender_entity_id=str(
                        row.get("replying_to_sender_entity_id") or ""
                    ),
                    replying_to_sender_handle=str(
                        row.get("replying_to_sender_handle") or ""
                    ),
                )
            )
        return messages


class ApiMentionFetcher:
    """Comment activity addressed to the bot, from the two notification routes.

    Neither takes an entity id - see the module docstring. They answer for
    whoever the token belongs to and nobody else.
    """

    def __init__(self, client: BotApiClient) -> None:
        self.client = client

    def fetch_comment_mentions(self, limit: int) -> list[PendingMention]:
        return self._fetch(
            "/v1/mentions/comments", "mentions", "mention", limit,
            "could not read comment mentions",
        )

    def fetch_comment_replies(self, limit: int) -> list[PendingMention]:
        """Comments replying to one the bot wrote.

        A separate route rather than a flag, because server-side these are a
        different notification type with a different rule for what counts:
        Django files "replied to your comment" and "commented on your post"
        under one type, and only the first is an answer to something the bot
        said. The endpoint does that separation; this fetcher only labels.
        """
        return self._fetch(
            "/v1/comments/replies", "replies", "reply", limit,
            "could not read comment replies",
        )

    def _fetch(
        self, path: str, key: str, kind: str, limit: int, failure: str
    ) -> list[PendingMention]:
        try:
            payload = self.client.get(path, {"limit": max(1, limit)})
        except PlatformAuthError as exc:
            logger.error(
                "not permitted to read notifications", extra={"error": str(exc)}
            )
            return []
        except PlatformAPIError as exc:
            logger.error(failure, extra={"error": str(exc)})
            return []

        pending: list[PendingMention] = []
        for row in payload.get(key) or []:
            if not isinstance(row, dict):
                continue
            comment_id = row.get("comment_id")
            text = row.get("text")
            if not comment_id or not isinstance(text, str) or not text.strip():
                # The endpoint already drops textless rows; this is the belt to
                # that braces, because an unanswerable trigger reaching the
                # policy is worse than one dropped twice.
                continue
            pending.append(
                PendingMention(
                    comment_id=str(comment_id),
                    post_id=str(row.get("post_id") or ""),
                    author_entity_id=str(row.get("author_entity_id") or ""),
                    text=text,
                    author_handle=str(row.get("author_handle") or ""),
                    created_at=_int(row.get("created_at")),
                    # The endpoint labels every row; the route it came from is
                    # the fallback, so an older API still yields the right kind.
                    kind=str(row.get("kind") or kind),
                )
            )
        return pending


def _int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
