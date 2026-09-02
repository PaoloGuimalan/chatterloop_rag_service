"""Normalised "something addressed me" events.

Message mentions and comment mentions arrive by completely different routes -
one is a self-describing `messages_list` frame, the other is an unaddressed
`notifications` ping that has to be resolved against the notification store.
Both converge here so the policy and the responder only ever see one shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MentionSource(StrEnum):
    MESSAGE = "message"
    COMMENT = "comment"


@dataclass(slots=True)
class MentionTrigger:
    """A mention the bot may want to answer."""

    source: MentionSource
    # Who mentioned us.
    author_entity_id: str
    author_handle: str = ""

    # Messages: the conversation. Comments: empty - a comment lives on a post.
    conversation_id: str = ""
    # The message that mentioned us. The bot answers as a threaded reply to it,
    # so this is what ends up in `replyingTo` on the outgoing message.
    message_id: str = ""
    # Comments: the post and the comment doing the mentioning. Taken from the
    # notification's target_id / target_anchor.
    post_id: str = ""
    comment_id: str = ""

    # The text that addressed us, once fetched. Empty until resolved: neither
    # frame carries content, so this is filled in by a fetcher port.
    text: str = ""
    # `text` with the bot's own @handle removed - what actually gets embedded.
    query: str = ""

    realm_name: str | None = None
    is_single: bool = False
    occurred_at: str = ""
    # Stable key for deduplication. Derived from whatever identifies the
    # underlying object, so a redelivered frame collapses onto the same key.
    dedupe_key: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        """Whether the trigger carries enough to answer.

        A trigger with no text is a mention we know happened but cannot read -
        the fetcher either failed or has not run. Answering one would mean
        generating a reply to an unknown question.
        """
        return bool(self.query.strip())
