"""Who the bot is, and how its world maps onto the RAG index."""

from __future__ import annotations

from dataclasses import dataclass, field

from .mentions import normalise_handle


@dataclass(frozen=True, slots=True)
class BotIdentity:
    """The bot's entity, as the platform sees it.

    A bot acts through an Entity exactly like a person does - see
    services/user_service/bot/models.py: "a bot is exactly as capable as its
    entity is and no more". So `entity_id` is the only field that matters
    functionally; the handles are for reading text.
    """

    entity_id: str
    handle: str
    # Additional handles that should also count as addressing the bot. Useful
    # while a handle is being migrated, or for a short alias.
    aliases: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("bot entity_id is required")
        if not self.handle:
            raise ValueError("bot handle is required")

    @property
    def channel(self) -> str:
        """The Redis channel carrying this entity's realtime events.

        Same channel the SSE endpoint subscribes to:
        `listen(events_${entity_id}, res)` in server/routes/users/index.js.
        """
        return f"events_{self.entity_id}"

    @property
    def handles(self) -> set[str]:
        return {normalise_handle(self.handle)} | {
            normalise_handle(a) for a in self.aliases if a
        }

    def is_self(self, entity_id: str | None) -> bool:
        """Whether an entity id is the bot itself.

        The single most important check in the whole adapter: a bot that treats
        its own messages as input will answer itself forever.
        """
        return bool(entity_id) and str(entity_id) == self.entity_id


def conversation_tenant(conversation_id: str) -> str:
    """Map a conversation onto a RAG tenant (a Milvus partition key).

    One conversation, one tenant. That is a stricter boundary than the platform
    itself draws - a realm's members can all read a realm channel - but the
    event the bot acts on does not carry a realm id. `messages_list` gives us
    `conversationID` and, at most, a `realmName` *string* on the mentioner. A
    display name is not an identifier: two realms can share one, and renaming a
    realm would silently repartition its history.

    Conversation-scoped is therefore the only boundary that can be drawn
    correctly from the data available, and it errs the safe way - context never
    crosses a conversation. Widening to realm scope is a real improvement, but
    it needs `realm_id` on the frame first, which is a platform change.
    """
    if not conversation_id:
        raise ValueError("conversation_id is required to derive a tenant")
    return f"conv:{conversation_id}"
