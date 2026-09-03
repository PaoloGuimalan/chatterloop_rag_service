"""The bot: realtime frame in, reply out.

    events_<entity_id>  (Redis pub/sub)
        |
        v
    route by event name
        |
        +-- messages_list, mentioner != null ----> MENTION trigger
        |
        +-- messages_list, mentioner == null ----> probe /conversations/{id}/replies
        |                                          for a message threaded under
        |                                          one of ours -> REPLY trigger
        |
        +-- notifications ----------------------> poll the notification store,
        |                                          comment mentions AND comment
        |                                          replies
        v
    resolve text (MessageFetcher / MentionFetcher)
        |
        v
    policy.evaluate  -> IGNORE (with a reason) or RESPOND
        |
        v
    index the conversation, retrieve context, generate, reply

TWO WAYS IN, AND WHY THE SECOND COSTS A READ. A mention is self-describing:
`messages_list` carries a non-null `mentioner` only for recipients the server
resolved a mention against, so the bot knows it was addressed from the frame
alone. A reply is not. The frame carries the conversation, the sender and that
mentioner field - no message id, and no `replyingTo` - so "was that a reply to
me?" cannot be answered without reading.

It is read through a route that answers exactly that question
(`/v1/conversations/{id}/replies`, scoped to the token's own entity) rather
than by fetching history and inspecting it here. The difference is a
usually-empty result against a full history window on every message in every
conversation the bot belongs to, and it is also the difference between "replies
to me" and "replies to anyone" being a server-side fact rather than a
client-side intention.

LIVE EVENTS ONLY. The bot answers what happens while it is listening, and
nothing else. That is not a detail of the implementation, it is the rule:

  * a realtime FRAME is inherently live - pub/sub has no replay, so a frame
    published while the bot was down is simply gone, and there is nothing to
    guard against;
  * everything the bot READS to resolve a frame is not. Conversation replies
    come back as a window, and comment notifications are DURABLE - unread rows
    wait indefinitely. Both accumulate while the bot is offline, and a bot that
    answered everything it found would come back from a restart and work
    through the backlog: one reply per conversation from the reply probe, and
    up to a full page of comment mentions in a single burst, each one a real
    message to a real person about something they said hours ago.

So every candidate resolved from a READ is checked against `started_at_ms` and
dropped if it predates this process. The cost is real and accepted: a mention
that arrives during a deploy is never answered, because a notification the bot
did not see live is one it will never see. That is the trade the alternative
does not offer - the alternative is a bot whose restart is a broadcast.

An unreadable timestamp counts as STALE, never fresh. Silence on a row we
cannot date is recoverable; a burst is not.

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
import time
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
from .mentions import is_addressed_to, normalise_handle, strip_mentions
from .policy import AddressedOnlyPolicy
from .ports import MentionFetcher, MessageFetcher, PlatformMessage, Responder
from .replies import ReplyGenerator
from .triggers import Trigger, TriggerReason, TriggerSource

logger = logging.getLogger(__name__)


@dataclass
class BotStats:
    frames_seen: int = 0
    triggers_seen: int = 0
    replied: int = 0
    ignored: int = 0
    errors: int = 0
    ignore_reasons: dict[str, int] = field(default_factory=dict)
    # How the bot came to be addressed, by reason. Mentions and replies fail in
    # different ways - a mention that resolves to nothing is a fetch problem, a
    # reply that never fires is a probe problem - so one combined count would
    # hide whichever half is broken.
    trigger_reasons: dict[str, int] = field(default_factory=dict)

    def note_ignored(self, reason: str) -> None:
        self.ignored += 1
        self.ignore_reasons[reason] = self.ignore_reasons.get(reason, 0) + 1

    def note_trigger(self, reason: str) -> None:
        self.triggers_seen += 1
        self.trigger_reasons[reason] = self.trigger_reasons.get(reason, 0) + 1

    def snapshot(self) -> dict[str, object]:
        return {
            "frames_seen": self.frames_seen,
            "triggers_seen": self.triggers_seen,
            "replied": self.replied,
            "ignored": self.ignored,
            "errors": self.errors,
            "ignore_reasons": dict(self.ignore_reasons),
            "trigger_reasons": dict(self.trigger_reasons),
        }


class ChatterloopBot:
    def __init__(
        self,
        identity: BotIdentity,
        policy: AddressedOnlyPolicy,
        ingestion: IngestionPipeline,
        retrieval: RetrievalPipeline,
        generator: ReplyGenerator,
        responder: Responder,
        message_fetcher: MessageFetcher,
        mention_fetcher: MentionFetcher,
        history_window: int = 40,
        top_k: int = 8,
        answer_replies: bool = True,
        answer_dms: bool = True,
        reply_probe_window: int = 25,
        only_live_events: bool = True,
        started_at_ms: int | None = None,
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
        # The kill switch for the whole unnamed-reply path. Off, the bot is
        # exactly what it was before: mention-only, one early return, and no
        # read at all on a message that did not name it.
        self.answer_replies = answer_replies
        # The kill switch for treating EVERY message in a DM as addressed, no
        # @handle or reply-threading needed. Off, a DM is judged exactly like
        # a group: mention, then (if answer_replies) the reply probe - which
        # means a plain "hey" in a DM goes unanswered, same as it always did.
        self.answer_dms = answer_dms
        # How many recent replies the probe looks at. Bounds the work per
        # frame, not the conversation - see RepliesTo in developer_service.
        self.reply_probe_window = reply_probe_window
        # conversation_id -> conversation_type, resolved once and kept
        # forever: the platform never changes a conversation's type after
        # creation, so this is one read per conversation over the bot's whole
        # lifetime, not one per unaddressed message. Only definitive results
        # are cached - see _is_dm - so a transient read failure retries on
        # the next message instead of wrongly answering "not a DM" forever.
        self._conversation_types: dict[str, str] = {}
        # The watermark. Anything resolved from a READ that predates this is
        # backlog, not traffic - see the module docstring.
        self.only_live_events = only_live_events
        self.started_at_ms = (
            int(time.time() * 1000) if started_at_ms is None else started_at_ms
        )
        self.stats = BotStats()

    def _is_live(self, created_at_ms: int) -> bool:
        """Whether something read from a store happened on this process's watch.

        A zero or missing timestamp is NOT live. Every path that calls this is
        one where being wrong means sending a real message about something
        somebody said hours ago, so an undatable row is refused rather than
        given the benefit of the doubt.
        """
        if not self.only_live_events:
            return True
        return bool(created_at_ms) and created_at_ms >= self.started_at_ms

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

        # Loop prevention, moved AHEAD of the mention gate now that a frame
        # without a mention is no longer free. The bot never spends an API call
        # on its own message, and this is what keeps the reply path bounded:
        # the bot's own answers are threaded under somebody else's message, and
        # a bot that read them back would sustain a conversation with itself
        # that needs no @handle at all.
        #
        # Silent rather than counted as an ignore. Every message the bot sends
        # comes back to it on its own channel, so counting these would bury the
        # ignore reasons that mean something under one that never does.
        if self.identity.is_self(payload.entityID):
            return

        # `mentioner` is non-null only when the server resolved a mention
        # against this recipient specifically - see the per-receiver
        # `isMentioned ? mentioner : null` in the /sendMessage handler. It is
        # authoritative, so a frame carrying one needs no read to classify.
        if payload.mentioner is not None:
            trigger = Trigger(
                source=TriggerSource.MESSAGE,
                reason=TriggerReason.MENTION,
                author_entity_id=payload.entityID,
                author_handle=normalise_handle(payload.mentioner.username),
                conversation_id=payload.conversationID,
                realm_name=payload.mentioner.realmName,
                is_single=payload.mentioner.isSingle,
                occurred_at=envelope.dateTime,
            )
            self.stats.note_trigger(str(trigger.reason))
            self._guard(trigger, self._resolve_mention_trigger)
            return

        # No mention on the frame. In a DM that is not evidence of anything -
        # the bot is one of exactly two participants, so a message with no
        # @handle is still aimed at it, the same way typing into any other 1:1
        # chat is. Checked BEFORE the reply probe below: a DM needs neither an
        # @mention nor reply-threading, so there is nothing for that probe to
        # add here, only a read to skip.
        if self.answer_dms and self._is_dm(payload.conversationID):
            trigger = Trigger(
                source=TriggerSource.MESSAGE,
                reason=TriggerReason.DM,
                author_entity_id=payload.entityID,
                conversation_id=payload.conversationID,
                is_single=True,
                occurred_at=envelope.dateTime,
            )
            self.stats.note_trigger(str(trigger.reason))
            self._guard(trigger, self._resolve_dm_trigger)
            return

        # Not a DM either. The message may still be a direct reply to
        # something the bot said, which the frame cannot tell us - so this is
        # where the extra read lives, and where it stops if the feature is
        # off.
        if not self.answer_replies:
            return

        self._probe_for_reply(payload, envelope)

    def _is_dm(self, conversation_id: str) -> bool:
        """Whether this conversation has exactly two participants: the bot,
        and whoever it is talking to.

        Cached forever once known - see `_conversation_types` in __init__. A
        cache miss costs one small read (`MessageFetcher.conversation_type`,
        `limit=1`); after that, every future message in the SAME conversation
        is free. A GROUP the bot is freshly added to still pays this once, on
        its first unaddressed message - there is no way around that, since
        nothing on the realtime frame says what kind of conversation it is
        (see frames.py's MessagesListPayload) - but every message after the
        first is not this method's problem any more.
        """
        cached = self._conversation_types.get(conversation_id)
        if cached is not None:
            return cached == "single"

        # Guarded the same way _probe_for_reply guards its own fetch: a read
        # against the platform can fail for reasons that have nothing to do
        # with this frame, and one bad conversation must not stop the bot
        # reading the next one. Unguarded, this exception would propagate out
        # of _on_messages_list uncaught (handle_envelope wraps nothing around
        # it) and take the whole process down over a single flaky read.
        try:
            conversation_type = self.message_fetcher.conversation_type(conversation_id)
        except Exception as exc:
            self.stats.errors += 1
            logger.exception(
                "failed resolving conversation type",
                extra={"conversation_id": conversation_id, "error": str(exc)},
            )
            return False

        if not conversation_type:
            # Unresolvable - no fetcher configured, or the read failed. Do NOT
            # cache a guess (a transient failure must not become a permanent
            # misclassification), and default to "not a DM": falling through
            # to the reply probe is the safe direction to be wrong in, since
            # an unnamed message there still gets a chance to be answered
            # instead of this method silently claiming DM status it cannot
            # back up.
            return False

        self._conversation_types[conversation_id] = conversation_type
        return conversation_type == "single"

    def _probe_for_reply(self, payload, envelope) -> None:
        """Ask the API whether that message was threaded under one of ours."""
        try:
            replies = self.message_fetcher.fetch_replies_to_me(
                payload.conversationID, self.reply_probe_window
            )
        except Exception as exc:
            self.stats.errors += 1
            logger.exception(
                "failed probing for replies",
                extra={"conversation_id": payload.conversationID, "error": str(exc)},
            )
            return

        message = self._newest_unhandled_reply(replies, payload.entityID)
        if message is None:
            return

        trigger = Trigger(
            source=TriggerSource.MESSAGE,
            reason=TriggerReason.REPLY,
            author_entity_id=message.sender_entity_id,
            author_handle=normalise_handle(message.sender_handle),
            conversation_id=payload.conversationID,
            occurred_at=envelope.dateTime,
        )
        self.stats.note_trigger(str(trigger.reason))
        self._guard(trigger, lambda t: self._resolve_reply_trigger(t, message))

    def _newest_unhandled_reply(
        self, replies: list[PlatformMessage], author_entity_id: str
    ) -> PlatformMessage | None:
        """The reply this frame is about, or None.

        Scanned newest-first and narrowed to the frame's own sender, for the
        same reason the mention path scans back rather than taking the last
        message outright: between the message being sent and this probe
        completing, other people may have spoken.

        Already-handled keys are skipped HERE rather than left to the policy,
        because the probe returns a window. Every new message in a busy
        conversation re-offers the replies already answered, and letting those
        through would mean paying for a history fetch to resolve something that
        is about to be ignored.

        The window is also why the liveness check lives here. The probe is
        happy to hand back a reply from yesterday; on a freshly started process
        the dedupe set is empty, so without this the first frame in a
        conversation would answer that instead of what just arrived.
        """
        for message in reversed(replies):
            if self.identity.is_self(message.sender_entity_id):
                continue
            if str(message.sender_entity_id) != str(author_entity_id):
                continue
            if self.policy.has_seen(f"msg:{message.message_id}"):
                continue
            if not self._is_live(message.created_at):
                # Recorded, so the next frame in this conversation does not
                # reconsider the same stale reply and log it again.
                self.policy.record_seen(f"msg:{message.message_id}")
                self.stats.note_ignored("reply predates this process")
                continue
            return message
        return None

    def _guard(self, trigger: Trigger, resolve) -> None:
        """Resolve then act, with one place for the failure to be contained.

        A read against the platform can fail for reasons that have nothing to
        do with this frame, and one bad frame must not stop the bot reading the
        next one.
        """
        try:
            resolve(trigger)
            self._act(trigger)
        except Exception as exc:
            self.stats.errors += 1
            logger.exception(
                "failed handling trigger",
                extra={
                    "reason": str(trigger.reason),
                    "conversation_id": trigger.conversation_id,
                    "error": str(exc),
                },
            )

    def _resolve_mention_trigger(self, trigger: Trigger) -> None:
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
        handles = set(self.identity.handles)
        for message in reversed(history):
            if self.identity.is_self(message.sender_entity_id):
                continue
            if str(message.sender_entity_id) != str(trigger.author_entity_id):
                continue
            if not is_addressed_to(message.content, handles):
                continue
            self._attach(trigger, message)
            return

        # The platform said we were mentioned but the fetched window does not
        # show it. Most likely the message landed after the window, or the
        # fetcher is not wired up. Either way there is no question to answer.
        logger.info(
            "mention reported but not found in fetched history",
            extra={"conversation_id": trigger.conversation_id},
        )

    def _resolve_dm_trigger(self, trigger: Trigger) -> None:
        """Fetch the conversation, index it, and take the newest message from
        the other side - unlike a mention, with no addressing check at all.

        A single conversation has exactly two participants, so "not the bot"
        and "the person who sent the triggering message" are the same set -
        there is no group of other people `is_addressed_to` would need to
        rule out. Scanning back rather than taking the last message outright
        for the same reason `_resolve_mention_trigger` does: between the
        frame arriving and this fetch completing, the other side may have
        sent something newer, and that is the thing actually worth answering.
        """
        history = self.message_fetcher.fetch_recent(
            trigger.conversation_id, self.history_window
        )
        if not history:
            return

        self._index(trigger.conversation_id, history)

        for message in reversed(history):
            if self.identity.is_self(message.sender_entity_id):
                continue
            trigger.author_handle = normalise_handle(message.sender_handle)
            self._attach(trigger, message)
            return

        # Every message in the fetched window was the bot's own - it has
        # nothing to answer yet, not a failure. Distinct from the mention
        # case's log line above: that one means "the API and this fetch
        # disagree", which this does not.

    def _resolve_reply_trigger(self, trigger: Trigger, message: PlatformMessage) -> None:
        """Attach the already-identified reply, then index around it.

        The message is in hand before this runs - the probe returned it - so
        unlike the mention path there is nothing to search for. The history
        fetch is purely for the index, and it happens AFTER the trigger is
        attached so that a history read failing still leaves an answerable
        question rather than a silent drop.
        """
        self._attach(trigger, message)

        history = self.message_fetcher.fetch_recent(
            trigger.conversation_id, self.history_window
        )
        if history:
            self._index(trigger.conversation_id, history)

    def _attach(self, trigger: Trigger, message: PlatformMessage) -> None:
        """Bind a resolved message onto a trigger."""
        handles = set(self.identity.handles)
        trigger.text = message.content
        # Strip our own @handle before embedding: "@assistant what did we
        # decide about pricing" is a question about pricing, and leaving the
        # address in drags both retrieval legs toward the bot's name - the one
        # term guaranteed to be irrelevant. A no-op on the reply path, where
        # usually there is no handle to strip, which is the point of it.
        trigger.query = strip_mentions(message.content, handles)
        # The reply is threaded under this message, so the answer stays
        # attached to the question. In a busy realm channel an unthreaded reply
        # arrives detached from what it answers, and by the time the bot has
        # fetched, retrieved and generated, several other messages may have
        # landed in between.
        trigger.message_id = message.message_id
        # Keyed on the message, NOT on how it reached us. A message that both
        # names the bot and replies to it arrives twice - once per path - and
        # deserves one answer.
        trigger.dedupe_key = f"msg:{message.message_id}"

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
        """Resolve comment mentions AND comment replies from the notification store.

        The `notifications` frame is unaddressed - it carries a status, a
        human-readable sentence, and nothing that identifies a subject. sse.ts
        says so in as many words, and responds by refetching. Matching on the
        sentence text would be matching on a UI string, so the notification
        store is treated as the source of truth and this frame purely as a
        hint to go and read it.

        That same shapelessness is why the reply half needs no new frame: one
        ping already means "go and look", and there are now two places to look.
        """
        pending = self._pending_comments()

        for mention in pending:
            if self.identity.is_self(mention.author_entity_id):
                continue
            # No post, no answer. A notification whose target could not be
            # resolved names a comment we cannot reply on - and its tenant
            # would be the bare string "post:", which is every such comment
            # sharing one partition. Dropped here rather than failing at the
            # send, where it reads as a broken write path instead of an
            # unanswerable row.
            if not mention.post_id:
                self.stats.note_ignored("comment mention has no post")
                continue
            # THE BACKLOG GUARD. Unread notifications are durable and pile up
            # while the bot is down, and this loop answers every row it is
            # given - so without this one ping after a restart fires a reply
            # per accumulated mention, to people who wrote them hours ago.
            if not self._is_live(mention.created_at):
                self.policy.record_seen(f"comment:{mention.comment_id}")
                self.stats.note_ignored("comment predates this process")
                continue
            reason = (
                TriggerReason.REPLY
                if mention.kind == TriggerReason.REPLY
                else TriggerReason.MENTION
            )
            handles = set(self.identity.handles)
            trigger = Trigger(
                source=TriggerSource.COMMENT,
                reason=reason,
                author_entity_id=mention.author_entity_id,
                author_handle=normalise_handle(mention.author_handle),
                post_id=mention.post_id,
                comment_id=mention.comment_id,
                text=mention.text,
                query=strip_mentions(mention.text, handles),
                dedupe_key=f"comment:{mention.comment_id}",
            )
            self.stats.note_trigger(str(trigger.reason))
            try:
                self._act(trigger)
            except Exception as exc:
                self.stats.errors += 1
                logger.exception(
                    "failed handling comment trigger",
                    extra={"comment_id": mention.comment_id, "error": str(exc)},
                )

    def _pending_comments(self) -> list:
        """Both notification reads, merged and deduplicated by comment id.

        Each read is contained on its own: the reply store being unreachable
        must not also cost us the mentions, which worked perfectly well before
        it existed.

        Mentions are read first and win the deduplication. The two lists should
        not overlap - Django skips the mention notification for anyone it has
        already pinged for the same comment - but if they ever do, the mention
        is the older, better-tested path.
        """
        pending = []
        seen: set[str] = set()

        reads = [("comment mentions", self.mention_fetcher.fetch_comment_mentions)]
        if self.answer_replies:
            reads.append(("comment replies", self.mention_fetcher.fetch_comment_replies))

        for label, read in reads:
            try:
                rows = read(limit=20)
            except Exception as exc:
                self.stats.errors += 1
                logger.exception(
                    "failed fetching %s", label, extra={"error": str(exc)}
                )
                continue
            for row in rows:
                if row.comment_id in seen:
                    continue
                seen.add(row.comment_id)
                pending.append(row)
        return pending

    # --------------------------------------------------------------- acting

    def _act(self, trigger: Trigger) -> None:
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
                    "trigger": str(trigger.reason),
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

        if trigger.source is TriggerSource.COMMENT:
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
                "trigger": str(trigger.reason),
                "conversation_id": trigger.conversation_id,
                "replying_to": trigger.message_id,
                "comment_id": trigger.comment_id,
                "context_chunks": len(context.chunks),
            },
        )

    def _tenant_for(self, trigger: Trigger) -> str:
        if trigger.conversation_id:
            return conversation_tenant(trigger.conversation_id)
        # A comment lives on a post, not a conversation. Same reasoning as
        # conversation_tenant: the narrowest boundary the data supports.
        return f"post:{trigger.post_id}"
