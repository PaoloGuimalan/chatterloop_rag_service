"""Acting as a user on the chatterloop platform.

Subscribes to the bot entity's realtime channel, and - initially - replies only
when explicitly mentioned in a message or a comment.
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
from .policy import Decision, MentionOnlyPolicy, Verdict
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
from .triggers import MentionSource, MentionTrigger

__all__ = [
    "BotIdentity",
    "BotStats",
    "ChatterloopBot",
    "Decision",
    "Envelope",
    "EntityEventConsumer",
    "Frame",
    "MENTION_PATTERN",
    "MentionFetcher",
    "MentionOnlyPolicy",
    "MentionSource",
    "MentionTrigger",
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
    "Verdict",
    "conversation_tenant",
    "extract_handles",
    "is_addressed_to",
    "normalise_handle",
    "strip_mentions",
]
