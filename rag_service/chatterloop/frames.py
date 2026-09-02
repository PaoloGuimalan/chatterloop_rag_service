"""Typed models of the platform's realtime frames.

The bot subscribes to the same Redis channel the browser's SSE stream is fed
from - `events_<entity_id>` - so these are the frames `webapp/src/reusables/
hooks/sse.ts` handles, one layer lower.

The envelope written by `publish(channel, event, message)` in
server/reusables/redis/pubsub.js is:

    {"logType": null, "pod": "...", "event": "<name>",
     "message": <frame>, "dateTime": "..."}

and the SSE bridge forwards only `data.message` as the frame body
(`res.sse(data.event, data.message)`). So what the webapp calls
`parsedresponse` is the `message` field here - which is why `Frame` below has
a `message` of its own nested inside it. That double nesting is confusing but
it is what is on the wire.

Only the frames the bot acts on are modelled. Everything else - the call
signalling, mediasoup transport plumbing, typing indicators - is consumed and
dropped, but is listed in IGNORED_EVENTS so that "unknown event" stays a real
signal rather than routine noise.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Mentioner(BaseModel):
    """Who mentioned us, as built by the /sendMessage handler.

    Present on a `messages_list` frame only when THIS recipient was mentioned:
    the server resolves handles against the conversation's member list and
    passes `isMentioned ? mentioner : null` per receiver. That per-recipient
    resolution is why the bot can trust this field instead of re-parsing.
    """

    model_config = ConfigDict(extra="ignore")

    entityID: str = ""
    # Arrives with the "@" already on it (`@${handle}`).
    username: str = ""
    realmName: str | None = None
    isSingle: bool = False


class MessagesListPayload(BaseModel):
    """Body of a `messages_list` frame.

    Note what is absent: the message text, and the message id. This event is a
    ping telling a client to refetch, not a carrier of content - the webapp
    responds by calling InitConversationListRequest and re-rendering the
    conversation. The bot has to do the same, which is why replying needs a
    MessageFetcher port.
    """

    model_config = ConfigDict(extra="ignore")

    conversationID: str = ""
    # The ACTING entity that sent the message. Compared against the bot's own
    # entity id for loop prevention.
    entityID: str = ""
    mentioner: Mentioner | None = None
    # Present instead of a normal delivery when a message was deleted.
    deletedMessageID: str | None = None


class Frame(BaseModel):
    """The SSE frame body - `parsedresponse` in sse.ts."""

    model_config = ConfigDict(extra="ignore")

    status: bool = False
    auth: bool = False
    onseen: bool = False
    # A dict for messages_list, a plain string for notifications. The platform
    # overloads this field, so it is kept loose and narrowed per event type.
    message: Any = None
    result: Any = ""


class Envelope(BaseModel):
    """What is actually published on the Redis channel."""

    model_config = ConfigDict(extra="ignore")

    event: str = ""
    pod: str = ""
    dateTime: str = ""
    message: Frame = Field(default_factory=Frame)


# Events the bot reacts to.
EVENT_MESSAGES_LIST = "messages_list"
EVENT_NOTIFICATIONS = "notifications"
EVENT_NOTIFICATIONS_RELOAD = "notifications_reload"

HANDLED_EVENTS = frozenset(
    {EVENT_MESSAGES_LIST, EVENT_NOTIFICATIONS, EVENT_NOTIFICATIONS_RELOAD}
)

# Consumed and dropped without comment. Enumerated from sse.ts so that an event
# the platform adds later shows up in the logs as genuinely unknown rather than
# being lost in the call-signalling noise.
IGNORED_EVENTS = frozenset(
    {
        "coordinates_broadcast",
        "profile_relationship_updated",
        "istyping_broadcast",
        "incomingcall",
        "callreject",
        "contactslist",
        "active_users",
        "voice-joined",
        "join-room-response",
        "create-transport-response",
        "transport-connect-response",
        "produce-response",
        "new_producer",
        "participant-joined",
        "participant-left",
        "update_participants",
        "participant-status",
        "producer-closed",
        "consume-response",
        "consume-transport-error",
        "consume-error",
        "conference_requests_changed",
        "conference_members_changed",
        "conference_access_changed",
        "server_channels_changed",
        "realm_membership_changed",
        "removed_user_notif",
    }
)


def parse_envelope(raw: dict[str, Any]) -> Envelope:
    return Envelope.model_validate(raw)


def parse_messages_list(frame: Frame) -> MessagesListPayload | None:
    """Narrow a `messages_list` frame body.

    Returns None rather than raising on a shape we don't recognise: this is a
    firehose of another service's events, and one malformed frame must not stop
    the bot from reading the next one.
    """
    if not isinstance(frame.message, dict):
        return None
    try:
        return MessagesListPayload.model_validate(frame.message)
    except Exception:
        return None
