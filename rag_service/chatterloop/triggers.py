"""Normalised "something addressed me" events.

Four routes converge here so that the policy and the responder only ever see
one shape:

    MESSAGE  + MENTION   messages_list frame with a non-null `mentioner`
    MESSAGE  + REPLY     a message threaded under one of the bot's own
    COMMENT  + MENTION   an unaddressed `notifications` ping, resolved against
                         the comment-mention store
    COMMENT  + REPLY     the same ping, resolved against the comment-reply store

The SOURCE says which surface it happened on and therefore how to answer it.
The REASON says why the bot is entitled to answer at all, which is the product
rule: a mention is an invitation to speak, a reply is a conversation already in
progress. Both are recorded on the trigger because "why did it answer that?"
is a question the logs have to be able to settle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TriggerSource(StrEnum):
    MESSAGE = "message"
    COMMENT = "comment"


class TriggerReason(StrEnum):
    """Why this counts as being addressed.

    MENTION is the original rule and still the only way to START a thread with
    the bot. REPLY covers a message or comment aimed directly at something the
    bot itself said, where re-typing the handle every turn would be the kind of
    ceremony no human conversation has.

    Deliberately NOT a third value for "replied to somebody else in a thread
    the bot is in". That is ordinary conversation between other people, and the
    bot has no business in it.
    """

    MENTION = "mention"
    REPLY = "reply"


@dataclass(slots=True)
class Trigger:
    """Something the bot may want to answer."""

    source: TriggerSource
    # Who addressed us.
    author_entity_id: str
    author_handle: str = ""
    reason: TriggerReason = TriggerReason.MENTION

    # Messages: the conversation. Comments: empty - a comment lives on a post.
    conversation_id: str = ""
    # The message that addressed us. The bot answers as a threaded reply to it,
    # so this is what ends up in `replyingTo` on the outgoing message.
    message_id: str = ""
    # Comments: the post and the comment doing the addressing. Taken from the
    # notification's target_id / target_anchor.
    post_id: str = ""
    comment_id: str = ""

    # The text that addressed us, once fetched. Empty until resolved: the
    # `messages_list` frame carries no content, so this is filled in by a
    # fetcher port.
    text: str = ""
    # `text` with the bot's own @handle removed - what actually gets embedded.
    query: str = ""

    realm_name: str | None = None
    is_single: bool = False
    occurred_at: str = ""
    # Stable key for deduplication. Derived from whatever identifies the
    # underlying object, so a redelivered frame collapses onto the same key -
    # and so does the SAME message arriving once as a mention and once as a
    # reply, which is exactly what a "@bot, and also replying to you" message
    # does.
    dedupe_key: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        """Whether the trigger carries enough to answer.

        A trigger with no text is something we know happened but cannot read -
        the fetcher either failed or has not run. Answering one would mean
        generating a reply to an unknown question.
        """
        return bool(self.query.strip())
