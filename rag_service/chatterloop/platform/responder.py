"""The write seam - now connected.

WHAT CHANGED
------------
This module used to exist to explain why a bot could not send a message. The
reasons were real: `jwtchecker` needs a `user_account` row and a live device
session, which a bot entity structurally has neither of, and the route takes
its body as a JWT signed with the platform's `JWT_SECRET`, which this service
should never hold.

Both are now solved platform-side rather than worked around here:

  * `POST /v1/messages/send` authenticates with an `entity_token` instead
    of a user session, so no account and no device are involved.
  * The route takes plain JSON. The pipeline never holds the platform's
    JWT_SECRET - a secret that could mint any user's session - only a token
    that does exactly one thing.

WHAT DID NOT CHANGE
-------------------
The endpoint performs the same fan-out the platform's own send route does, so a
bot's message gets the same six side effects a person's does - the conversation is un-archived for
every participant, the preview updates, a realtime frame goes out per recipient,
the chat score moves, link previews resolve, and push notifications fire with a
distinct payload for mentions. Nothing here reimplements any of that, which is
why sending is a POST and not a database write.

Membership is still enforced server-side: the scope grants the CAPABILITY to
send, and the conversation decides where. A token cannot post into a
conversation its entity is not part of.

Two of the six are NOT performed for API-sent messages - link previews and
content tagging. See the developer_service README; a reply containing a URL
renders as a bare link rather than a preview card.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from .client import BotApiClient, PlatformAPIError

logger = logging.getLogger(__name__)


def build_send_body(
    conversation_id: str,
    content: str,
    conversation_type: str,
    reply_to_message_id: str = "",
    pending_id: str = "",
    message_type: str = "text",
) -> dict[str, Any]:
    """The JSON body `POST /v1/messages/send` accepts.

    Mirrors the fields the route reads and nothing else - `receivers` is
    deliberately absent because the route derives it server-side from the
    conversation rather than trusting the sender. That is what stops a token
    from addressing people who are not in the conversation.

    Threading is `isReply` plus `replyingTo`, and `replyingTo` is a bare
    messageID string: the webapp arms a reply with
    `setisReplying({isReply: true, replyingTo: cnvs.messageID})` and renders the
    quote by matching `messageID == replyingTo`. The Mongo column is `Mixed`, so
    a wrong shape would not be rejected - it would render as an empty quote.
    """
    if not conversation_id:
        raise ValueError("conversation_id is required")
    if not content or not content.strip():
        raise ValueError("refusing to send an empty message")

    return {
        "conversationID": str(conversation_id),
        # The client's optimistic-update key. The route echoes it back so a
        # sender can reconcile; generated here because nothing upstream has one.
        "pendingID": pending_id or uuid.uuid4().hex,
        "content": content.strip(),
        "isReply": bool(reply_to_message_id),
        "replyingTo": str(reply_to_message_id or ""),
        "messageType": message_type,
        "conversationType": conversation_type,
    }


class HttpResponder:
    """Sends through the platform's developer API."""

    def __init__(
        self,
        client: BotApiClient,
        conversation_type: str = "",
    ) -> None:
        self.client = client
        # EMPTY by default, deliberately. This used to default to "single",
        # which was a claim the bot had no business making: it replies into
        # conversations it did not create and whose type it is not told. One
        # such reply into a real group rewrote that conversation's type and the
        # UI began rendering a group as a DM.
        #
        # The API now derives the type from the conversation or its realm and
        # ignores what a sender claims, so the honest value here is "no
        # opinion". Pass a real type per call only when the caller genuinely
        # knows it.
        self.conversation_type = conversation_type

    def reply_to_conversation(
        self,
        conversation_id: str,
        text: str,
        reply_to_message_id: str = "",
        conversation_type: str = "",
    ) -> bool:
        body = build_send_body(
            conversation_id=conversation_id,
            content=text,
            conversation_type=conversation_type or self.conversation_type,
            reply_to_message_id=reply_to_message_id,
        )
        try:
            self.client.post("/v1/messages/send", body)
        except PlatformAPIError as exc:
            # False rather than raising: a failed reply is one unanswered
            # mention, and the consumer loop should carry on to the next one
            # rather than die on a single bad send.
            logger.error(
                "reply failed",
                extra={
                    "conversation_id": conversation_id,
                    "replying_to": reply_to_message_id,
                    "error": str(exc),
                },
            )
            return False

        logger.info(
            "replied",
            extra={
                "conversation_id": conversation_id,
                "replying_to": reply_to_message_id,
            },
        )
        return True

    def reply_to_comment(self, post_id: str, comment_id: str, text: str) -> bool:
        """Not implemented - and deliberately still not guessed at.

        Commenting is a Django newsfeed surface, not this Node route. Creating
        a comment flattens two-level threads (a reply to a reply re-parents to
        the top-level ancestor and mentions its author instead of nesting), and
        fans out its own mention notifications. None of that is shared with the
        message route, and there is no bot endpoint for it yet.

        This was unimplemented before the API cut-over too, so nothing here
        regressed - what changed is that the reason is now "that endpoint does
        not exist" rather than "a bot cannot authenticate at all".
        """
        raise NotImplementedError(
            "no bot endpoint for creating comments yet: comment creation is a "
            "Django newsfeed surface with its own thread-flattening and mention "
            "fan-out, and shares none of the message route's shape. Message "
            "replies work; comment replies need POST /v1/comments."
        )
