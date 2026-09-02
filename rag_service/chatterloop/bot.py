"""The bot: realtime frame in, reply out.

    events_<entity_id>  (Redis pub/sub)
        |
        v
    route by event name
        |
        +-- messages_list, mentioner != null ----> MentionTrigger
        +-- notifications ----------------------> poll notification store
        |                                          for comment_mention rows
        v
    resolve text (MessageFetcher / MentionFetcher)
        |
        v
    policy.evaluate  -> IGNORE (with a reason) or RESPOND
        |
        v
    index the conversation, retrieve context, generate, reply

INDEXING IS LAZY, AND THAT IS FORCED BY THE EVENT SHAPE. `messages_list`
carries `conversationID`, the sender, and a mentioner - but not the message
text and not even a message id. There is nothing to index at the moment a
message arrives. So the bot fetches recent history when it is addressed and
indexes that; upserts are keyed on (tenant, source, chunk_index), so
re-indexing the same window is idempotent and costs only the embedding of
messages it has not seen before, which the content-hash cache then absorbs.

Eager indexing would be better - the bot would answer from a warm index
instead of paying a fetch on the critical path - but it needs either message
content on the frame or a periodic backfill. Both are platform changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..domain import Role, Scope
from ..pipeline import IngestionPipeline, RetrievalPipeline
from .frames import (
    EVENT_MESSAGES_LIST,
    EVENT_NOTIFICATIONS,
    EVENT_NOTIFICATIONS_RELOAD,
    IGNORED_EVENTS,
    parse_envelope,
    parse_messages_list,
)
from .identity import BotIdentity, conversation_tenant
from .mentions import normalise_handle, strip_mentions
from .policy import MentionOnlyPolicy
from .ports import MentionFetcher, MessageFetcher, PlatformMessage, Responder
from .replies import ReplyGenerator
from .triggers import MentionSource, MentionTrigger

logger = logging.getLogger(__name__)


@dataclass
class BotStats:
    frames_seen: int = 0
    mentions_seen: int = 0
    replied: int = 0
    ignored: int = 0
    errors: int = 0
    ignore_reasons: dict[str, int] = field(default_factory=dict)

    def note_ignored(self, reason: str) -> None:
        self.ignored += 1
        self.ignore_reasons[reason] = self.ignore_reasons.get(reason, 0) + 1

    def snapshot(self) -> dict[str, object]:
        return {
            "frames_seen": self.frames_seen,
            "mentions_seen": self.mentions_seen,
            "replied": self.replied,
            "ignored": self.ignored,
            "errors": self.errors,
            "ignore_reasons": dict(self.ignore_reasons),
        }


class ChatterloopBot:
    def __init__(
        self,
        identity: BotIdentity,
        policy: MentionOnlyPolicy,
        ingestion: IngestionPipeline,
        retrieval: RetrievalPipeline,
        generator: ReplyGenerator,
        responder: Responder,
        message_fetcher: MessageFetcher,
        mention_fetcher: MentionFetcher,
        history_window: int = 40,
        top_k: int = 8,
    ) -> None:
        self.identity = identity
        self.policy = policy
        self.ingestion = ingestion
        self.retrieval = retrieval
        self.generator = generator
        self.responder = responder
        self.message_fetcher = message_fetcher
        self.mention_fetcher = mention_fetcher
        self.history_window = history_window
        self.top_k = top_k
        self.stats = BotStats()

    # ------------------------------------------------------------- routing

    def handle_envelope(self, raw: dict) -> None:
        self.stats.frames_seen += 1
        try:
            envelope = parse_envelope(raw)
        except Exception as exc:
            logger.warning("unparseable envelope", extra={"error": str(exc)})
            return

        event = envelope.event
        if event in IGNORED_EVENTS:
            return

        frame = envelope.message
        # The webapp checks both on every handler; an unauthenticated or failed
        # frame carries no usable payload.
        if not (frame.auth and frame.status):
            return

        if event == EVENT_MESSAGES_LIST:
            self._on_messages_list(envelope)
        elif event in (EVENT_NOTIFICATIONS, EVENT_NOTIFICATIONS_RELOAD):
            self._on_notifications()
        else:
            logger.info("unhandled event on entity channel", extra={"event": event})

    # ------------------------------------------------------------- messages

    def _on_messages_list(self, envelope) -> None:
        payload = parse_messages_list(envelope.message)
        if payload is None:
            return

        # A deletion ping, not a delivery.
        if payload.deletedMessageID:
            return

        # THE GATE. `mentioner` is non-null only when the server resolved a
        # mention against this recipient specifically - see the per-receiver
        # `isMentioned ? mentioner : null` in the /sendMessage handler. No
        # mention, no reply, and nothing else in this method runs.
        if payload.mentioner is None:
            return

        # Loop prevention runs before the fetch, not after, so the bot never
        # spends an API call on its own message.
        if self.identity.is_self(payload.entityID):
            self.stats.note_ignored("author is the bot itself")
            return

        self.stats.mentions_seen += 1

        trigger = MentionTrigger(
            source=MentionSource.MESSAGE,
            author_entity_id=payload.entityID,
            author_handle=normalise_handle(payload.mentioner.username),
            conversation_id=payload.conversationID,
            realm_name=payload.mentioner.realmName,
            is_single=payload.mentioner.isSingle,
            occurred_at=envelope.dateTime,
        )

        try:
            self._resolve_message_trigger(trigger)
            self._act(trigger)
        except Exception as exc:
            self.stats.errors += 1
            logger.exception(
                "failed handling message mention",
                extra={"conversation_id": trigger.conversation_id, "error": str(exc)},
            )

    def _resolve_message_trigger(self, trigger: MentionTrigger) -> None:
        """Fetch the conversation, index it, and pull out the addressing message."""
        history = self.message_fetcher.fetch_recent(
            trigger.conversation_id, self.history_window
        )
        if not history:
            return

        self._index(trigger.conversation_id, history)

        # The addressing message is the most recent one from the mentioner that
        # actually names us. Scanning back rather than taking the last message
        # outright: between the mention being sent and this fetch completing,
        # other people may have spoken.
        for message in reversed(history):
            if self.identity.is_self(message.sender_entity_id):
                continue
            if str(message.sender_entity_id) != str(trigger.author_entity_id):
                continue
            handles = set(self.identity.handles)
            from .mentions import is_addressed_to

            if not is_addressed_to(message.content, handles):
                continue
            trigger.text = message.content
            # Strip our own @handle before embedding: "@assistant what did we
            # decide about pricing" is a question about pricing, and leaving
            # the address in drags both retrieval legs toward the bot's name -
            # the one term guaranteed to be irrelevant.
            trigger.query = strip_mentions(message.content, handles)
            # The reply is threaded under this message, so the answer stays
            # attached to the question. In a busy realm channel an unthreaded
            # reply arrives detached from what it answers, and by the time the
            # bot has fetched, retrieved and generated, several other messages
            # may have landed in between.
            trigger.message_id = message.message_id
            trigger.dedupe_key = f"msg:{message.message_id}"
            return

        # The platform said we were mentioned but the fetched window does not
        # show it. Most likely the message landed after the window, or the
        # fetcher is not wired up. Either way there is no question to answer.
        logger.info(
            "mention reported but not found in fetched history",
            extra={"conversation_id": trigger.conversation_id},
        )

    def _index(self, conversation_id: str, messages: list[PlatformMessage]) -> None:
        tenant = conversation_tenant(conversation_id)
        for message in messages:
            if not message.content.strip():
                continue
            role = (
                Role.ASSISTANT
                if self.identity.is_self(message.sender_entity_id)
                else Role.USER
            )
            self.ingestion.index_message(
                tenant_id=tenant,
                conversation_id=conversation_id,
                message_id=message.message_id,
                text=message.content,
                role=role,
                created_at=message.created_at or None,
                meta={"sender_entity_id": message.sender_entity_id,
                      "sender_handle": message.sender_handle},
            )

    # ------------------------------------------------------------- comments

    def _on_notifications(self) -> None:
        """Resolve comment mentions from the notification store.

        The `notifications` frame is unaddressed - it carries a status, a
        human-readable sentence, and nothing that identifies a subject. sse.ts
        says so in as many words, and responds by refetching. Matching on the
        sentence text would be matching on a UI string, so the notification
        store is treated as the source of truth and this frame purely as a
        hint to go and read it.
        """
        try:
            pending = self.mention_fetcher.fetch_comment_mentions(limit=20)
        except Exception as exc:
            self.stats.errors += 1
            logger.exception("failed fetching comment mentions", extra={"error": str(exc)})
            return

        for mention in pending:
            if self.identity.is_self(mention.author_entity_id):
                continue
            self.stats.mentions_seen += 1
            handles = set(self.identity.handles)
            trigger = MentionTrigger(
                source=MentionSource.COMMENT,
                author_entity_id=mention.author_entity_id,
                author_handle=normalise_handle(mention.author_handle),
                post_id=mention.post_id,
                comment_id=mention.comment_id,
                text=mention.text,
                query=strip_mentions(mention.text, handles),
                dedupe_key=f"comment:{mention.comment_id}",
            )
            try:
                self._act(trigger)
            except Exception as exc:
                self.stats.errors += 1
                logger.exception(
                    "failed handling comment mention",
                    extra={"comment_id": mention.comment_id, "error": str(exc)},
                )

    # --------------------------------------------------------------- acting

    def _act(self, trigger: MentionTrigger) -> None:
        decision = self.policy.evaluate(trigger)
        # Recorded whatever the verdict: a redelivery of something already
        # judged should not be judged again.
        self.policy.record_seen(trigger.dedupe_key)

        if not decision.should_respond:
            self.stats.note_ignored(decision.reason)
            logger.info(
                "not replying",
                extra={
                    "reason": decision.reason,
                    "source": str(trigger.source),
                    "conversation_id": trigger.conversation_id,
                    "comment_id": trigger.comment_id,
                },
            )
            return

        context = self.retrieval.retrieve(
            tenant_id=self._tenant_for(trigger),
            query=trigger.query,
            conversation_id=trigger.conversation_id,
            top_k=self.top_k,
            scopes=[Scope.DOCUMENT, Scope.CHAT],
        )

        reply = self.generator.generate(trigger.query, context).strip()
        if not reply:
            self.stats.note_ignored("generator produced an empty reply")
            return

        if trigger.source is MentionSource.COMMENT:
            sent = self.responder.reply_to_comment(
                trigger.post_id, trigger.comment_id, reply
            )
        else:
            sent = self.responder.reply_to_conversation(
                trigger.conversation_id, reply, trigger.message_id
            )

        if not sent:
            # Deliberately not counted against the conversation's budget: a
            # broken outbound path must not rate-limit the bot into silence.
            self.stats.errors += 1
            logger.error("reply could not be delivered",
                         extra={"conversation_id": trigger.conversation_id})
            return

        self.policy.record_reply(trigger)
        self.stats.replied += 1
        logger.info(
            "replied",
            extra={
                "source": str(trigger.source),
                "conversation_id": trigger.conversation_id,
                "replying_to": trigger.message_id,
                "comment_id": trigger.comment_id,
                "context_chunks": len(context.chunks),
            },
        )

    def _tenant_for(self, trigger: MentionTrigger) -> str:
        if trigger.conversation_id:
            return conversation_tenant(trigger.conversation_id)
        # A comment lives on a post, not a conversation. Same reasoning as
        # conversation_tenant: the narrowest boundary the data supports.
        return f"post:{trigger.post_id}"
