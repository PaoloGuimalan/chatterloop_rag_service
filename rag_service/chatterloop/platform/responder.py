"""The write seams - both now connected.

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

COMMENTS, WHICH USED TO BE THE HALF THAT DID NOT WORK
-----------------------------------------------------
`reply_to_comment` raised until `POST /v1/comments` existed, so a bot could be
addressed in a comment thread - by mention or by reply - and had no way to
answer. It now posts through that route, which owns the parts that made this
hard to do from a client: two-level thread flattening (a reply to a reply
re-parents to its top-level ancestor), the reply/post-comment notification, and
the mention fan-out.

One side effect it does NOT perform: a hashtag in a comment does not tag the
parent post. Reproducing that means widening the interest taxonomy from a fifth
implementation of a normaliser whose failure mode is a silent duplicate row.
Named in the developer_service README rather than papered over.
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


def build_comment_body(
    post_id: str,
    text: str,
    parent_id: str = "",
) -> dict[str, Any]:
    """The JSON body `POST /v1/comments` accepts.

    Three fields, and deliberately no fourth. There is no author - the endpoint
    takes it from the token - and no attachment, because a URL accepted from a
    caller is a different trust question from a file uploaded through the
    platform's own path, and the bot has nothing to attach.

    `parentID` empty means a top-level comment on the post. Non-empty means a
    reply to that comment, which is the only form the bot actually uses: it
    comments because it was addressed, never unprompted.
    """
    if not post_id:
        raise ValueError("post_id is required")
    if not text or not text.strip():
        raise ValueError("refusing to post an empty comment")

    return {
        "postID": str(post_id),
        "parentID": str(parent_id or ""),
        "text": text.strip(),
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
        """Post a comment in reply to the one that addressed us.

        THIS USED TO RAISE. The reason it did was real - commenting is a Django
        newsfeed surface with its own two-level thread flattening and its own
        mention fan-out, sharing none of the message route's shape - and it was
        the reason a bot could be addressed in a comment thread and had no way
        to speak in it. `POST /v1/comments` is that endpoint, and it does the
        flattening and the fan-out server-side rather than here.

        `parentID` is the comment being ANSWERED, not where the row will be
        stored. Replying to a reply re-parents to that reply's top-level
        ancestor, so the two differ exactly one level down; the endpoint returns
        both so the difference is visible rather than surprising. Nothing here
        tries to predict it - a client computing a parent id would be
        reimplementing the rule it is calling the endpoint to avoid.
        """
        body = build_comment_body(post_id=post_id, text=text, parent_id=comment_id)
        try:
            self.client.post("/v1/comments", body)
        except PlatformAPIError as exc:
            # False rather than raising, for the same reason a failed message
            # send returns False: one unanswered mention, not a dead consumer
            # loop. The policy does not charge a failed send against the
            # conversation's budget, so this is retried on the next trigger.
            logger.error(
                "comment reply failed",
                extra={
                    "post_id": post_id,
                    "replying_to": comment_id,
                    "error": str(exc),
                },
            )
            return False

        logger.info(
            "commented", extra={"post_id": post_id, "replying_to": comment_id}
        )
        return True
