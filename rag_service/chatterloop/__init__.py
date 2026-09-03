"""Acting as a user on the chatterloop platform.

Subscribes to the bot entity's realtime channel and replies only when
addressed: named with an @handle, or replied to directly in a message or a
comment. Everything else on the channel is read and dropped.
"""

from .bot import BotStats, ChatterloopBot
from .consumer import EntityEventConsumer
from .frames import Envelope, Frame, MessagesListPayload, Mentioner
from .identity import BotIdentity, conversation_tenant
from .mentions import (
    MENTION_PATTERN,
    extract_handles,
    is_addressed_to,
    normalise_handle,
    strip_mentions,
)
from .policy import AddressedOnlyPolicy, Decision, Verdict
from .ports import (
    MentionFetcher,
    MessageFetcher,
    NullMentionFetcher,
    NullMessageFetcher,
    PendingMention,
    PlatformMessage,
    RecordingResponder,
    Responder,
)
from .replies import OpenAIReplyGenerator, ReplyGenerator, StubReplyGenerator
from .triggers import Trigger, TriggerReason, TriggerSource

__all__ = [
    "AddressedOnlyPolicy",
    "BotIdentity",
    "BotStats",
    "ChatterloopBot",
    "Decision",
    "Envelope",
    "EntityEventConsumer",
    "Frame",
    "MENTION_PATTERN",
    "MentionFetcher",
    "Mentioner",
    "MessageFetcher",
    "MessagesListPayload",
    "NullMentionFetcher",
    "NullMessageFetcher",
    "OpenAIReplyGenerator",
    "PendingMention",
    "PlatformMessage",
    "RecordingResponder",
    "ReplyGenerator",
    "Responder",
    "StubReplyGenerator",
    "Trigger",
    "TriggerReason",
    "TriggerSource",
    "Verdict",
    "conversation_tenant",
    "extract_handles",
    "is_addressed_to",
    "normalise_handle",
    "strip_mentions",
]
