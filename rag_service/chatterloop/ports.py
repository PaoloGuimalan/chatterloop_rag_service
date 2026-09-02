"""The seams between this pipeline and the platform's services.

The realtime frames tell the bot that something happened but not what was said
- `messages_list` carries no message text and `notifications` carries no
subject at all - so answering needs two reads and one write against
chatterloop.

Those three operations are declared here as protocols. `chatterloop/platform/`
implements them against developer_service; the inert implementations at the
bottom of this module remain the default, so a pipeline with no platform
configured still connects, gates, and records what it would have said.

WHERE THEY BIND

All on developer_service, one origin authenticated by one entity_token.

  fetch messages   GET  /v1/conversations/{id}/messages
                   scope `messages.read`. Refuses a conversation the token's
                   entity is not a participant of - 404, not 403, so an
                   outsider cannot tell an existing conversation from one that
                   never existed. Normalises `messageDate`, which is a BSON
                   date or an embedded `{date, time}` depending on who wrote
                   the row, so a consumer never has to know that.

  fetch mentions   GET  /v1/mentions/comments
                   scope `notifications.read`. Takes no entity id: it answers
                   for whoever the token belongs to. Comment text lives in
                   Postgres while the mention itself is a Mongo notification,
                   and the endpoint joins them so this service does not reach
                   two stores.

  send a reply     POST /v1/messages/send
                   scope `messages.send`. Performs the same fan-out the
                   platform's own route does - un-archiving, conversation
                   preview, a realtime frame per recipient, chat score, push
                   with a distinct payload for mentions - so a bot's message
                   behaves like a person's.

                   `replyingTo` is a bare messageID string, not an object: the
                   webapp arms a reply with
                   `setisReplying({isReply: true, replyingTo: cnvs.messageID})`
                   (messenger/partials/ContentHandler.tsx) and renders the
                   quoted preview by finding `messageID == replyingTo`
                   (ConversationV2.tsx). The schema types the column as Mixed,
                   so nothing server-side would reject a different shape - it
                   would just render as an empty quote.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PlatformMessage:
    """A message as the platform stores it."""

    message_id: str
    conversation_id: str
    sender_entity_id: str
    content: str
    created_at: int = 0
    message_type: str = "text"
    sender_handle: str = ""


@dataclass(slots=True)
class PendingMention:
    """A comment mention discovered in the notification store."""

    comment_id: str
    post_id: str
    author_entity_id: str
    text: str = ""
    author_handle: str = ""
    created_at: int = 0


@runtime_checkable
class MessageFetcher(Protocol):
    def fetch_recent(self, conversation_id: str, limit: int) -> list[PlatformMessage]:
        """Most recent messages in a conversation, oldest first."""
        ...


@runtime_checkable
class MentionFetcher(Protocol):
    def fetch_comment_mentions(self, limit: int) -> list[PendingMention]:
        """Unhandled `comment_mention` notifications for this bot."""
        ...


@runtime_checkable
class Responder(Protocol):
    def reply_to_conversation(
        self, conversation_id: str, text: str, reply_to_message_id: str = ""
    ) -> bool:
        """Send a message, threaded under `reply_to_message_id` when given.

        An implementation should set `isReply: true` and `replyingTo:
        <message_id>` on the outgoing JWT. Empty means an unthreaded message -
        representable so a reply is still possible if the addressing message
        could not be pinned down, though the bot does not currently reach that
        state: a trigger without a message id never resolves.
        """
        ...

    def reply_to_comment(self, post_id: str, comment_id: str, text: str) -> bool:
        """Reply to the comment that mentioned us, on its post."""
        ...


class NullMessageFetcher:
    """Returns nothing, so every message mention resolves to unanswerable.

    Not a stub that pretends: a bot wired to this reads every event, logs every
    mention, and stays silent - which is the correct behaviour for a pipeline
    whose read path is not connected yet.
    """

    def fetch_recent(self, conversation_id: str, limit: int) -> list[PlatformMessage]:
        logger.debug(
            "no message fetcher configured; mention cannot be resolved",
            extra={"conversation_id": conversation_id},
        )
        return []


class NullMentionFetcher:
    def fetch_comment_mentions(self, limit: int) -> list[PendingMention]:
        return []


@dataclass
class RecordingResponder:
    """Logs what it would have sent, and keeps it for inspection.

    The default outbound. Every reply the bot decides to make is fully
    generated and recorded, so the pipeline can be evaluated end to end -
    including the quality of what it would say - before it is given the
    ability to actually say it.
    """

    sent: list[dict[str, Any]] = field(default_factory=list)

    def reply_to_conversation(
        self, conversation_id: str, text: str, reply_to_message_id: str = ""
    ) -> bool:
        self.sent.append({"kind": "conversation", "conversation_id": conversation_id,
                          "text": text, "reply_to_message_id": reply_to_message_id})
        logger.info(
            "WOULD REPLY (threaded)" if reply_to_message_id else "WOULD REPLY",
            extra={
                "conversation_id": conversation_id,
                "replying_to": reply_to_message_id,
                "text": text,
            },
        )
        return True

    def reply_to_comment(self, post_id: str, comment_id: str, text: str) -> bool:
        self.sent.append({"kind": "comment", "post_id": post_id,
                          "comment_id": comment_id, "text": text})
        logger.info(
            "WOULD REPLY (comment)",
            extra={"post_id": post_id, "comment_id": comment_id, "text": text},
        )
        return True
