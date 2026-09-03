"""End to end: a realtime frame goes in, a reply comes out (or doesn't).

Frames below are byte-for-byte the shape server/reusables/redis/pubsub.js
publishes - envelope, then the SSE frame body under `message`, then the
`messages_list` payload under `message.message`. That double nesting is real
and is the thing most likely to be got wrong, so the tests carry it verbatim.
"""

from __future__ import annotations

import pytest

from rag_service.chatterloop.bot import ChatterloopBot
from rag_service.chatterloop.identity import BotIdentity, conversation_tenant
from rag_service.chatterloop.policy import AddressedOnlyPolicy
from rag_service.chatterloop.ports import PendingMention, PlatformMessage, RecordingResponder
from rag_service.domain import RetrievalResult, RetrievedChunk, Role, Scope

BOT = BotIdentity(entity_id="bot-1", handle="assistant")


def frame(event="messages_list", *, mentioner=True, sender="human-1",
          conversation="conv-1", deleted=None, auth=True, status=True):
    payload = {
        "conversationID": conversation,
        "entityID": sender,
        "mentioner": (
            {"entityID": sender, "username": "@ana", "realmName": "Design",
             "isSingle": False}
            if mentioner else None
        ),
    }
    if deleted:
        payload["deletedMessageID"] = deleted
    return {
        "logType": None,
        "pod": "podless",
        "event": event,
        "dateTime": "2026-09-01T10:00:00",
        "message": {"status": status, "auth": auth, "onseen": False,
                    "message": payload, "result": ""},
    }


class FakeFetcher:
    def __init__(self, messages=None, replies=None):
        self.messages = messages if messages is not None else [
            PlatformMessage("m1", "conv-1", "human-1", "we agreed on tiered pricing", 1000),
            PlatformMessage("m2", "conv-1", "human-1",
                            "@assistant what did we decide about pricing?", 2000),
        ]
        # What GET /v1/conversations/{id}/replies would answer: only messages
        # threaded under one of the BOT's own. The server decides that, so a
        # fake that returned anything else would be testing a shape the bot
        # cannot receive.
        self.replies = replies or []
        self.calls: list[str] = []
        self.reply_calls: list[str] = []

    def fetch_recent(self, conversation_id, limit):
        self.calls.append(conversation_id)
        return list(self.messages)

    def fetch_replies_to_me(self, conversation_id, limit):
        self.reply_calls.append(conversation_id)
        return list(self.replies)


class FakeMentionFetcher:
    def __init__(self, pending=None, replies=None):
        self.pending = pending or []
        self.replies = replies or []

    def fetch_comment_mentions(self, limit):
        return list(self.pending)

    def fetch_comment_replies(self, limit):
        return list(self.replies)


class FakeIngestion:
    def __init__(self):
        self.indexed: list[dict] = []

    def index_message(self, **kw):
        self.indexed.append(kw)
        return 1


class FakeRetrieval:
    def __init__(self):
        self.calls: list[dict] = []

    def retrieve(self, **kw):
        self.calls.append(kw)
        return RetrievalResult(
            query=kw["query"], tenant_id=kw["tenant_id"],
            conversation_id=kw.get("conversation_id", ""),
            chunks=[RetrievedChunk("c1", "we agreed on tiered pricing",
                                   Scope.CHAT, 0.9, role=Role.USER)],
        )


class FakeGenerator:
    def __init__(self, reply="Tiered pricing."):
        self.reply = reply
        self.calls: list[str] = []

    def generate(self, question, context):
        self.calls.append(question)
        return self.reply


@pytest.fixture
def bot():
    responder = RecordingResponder()
    b = ChatterloopBot(
        identity=BOT,
        policy=AddressedOnlyPolicy(BOT, cooldown_seconds=0.0),
        ingestion=FakeIngestion(),
        retrieval=FakeRetrieval(),
        generator=FakeGenerator(),
        responder=responder,
        message_fetcher=FakeFetcher(),
        mention_fetcher=FakeMentionFetcher(),
    )
    return b


class TestTheGate:
    def test_a_mention_produces_a_reply(self, bot):
        bot.handle_envelope(frame())
        assert bot.responder.sent == [
            {
                "kind": "conversation",
                "conversation_id": "conv-1",
                "text": "Tiered pricing.",
                # Threaded under the message that did the mentioning.
                "reply_to_message_id": "m2",
            }
        ]
        assert bot.stats.replied == 1

    def test_an_unmentioned_message_costs_one_probe_and_nothing_more(self, bot):
        bot.handle_envelope(frame(mentioner=False))
        assert bot.responder.sent == []
        # One cheap "did that reply to me?" question, answered no. The
        # expensive read - history, for indexing - never happens.
        assert bot.message_fetcher.reply_calls == ["conv-1"]
        assert bot.message_fetcher.calls == []
        assert bot.stats.triggers_seen == 0

    def test_an_unmentioned_message_costs_nothing_when_replies_are_off(self, bot):
        bot.answer_replies = False
        bot.handle_envelope(frame(mentioner=False))
        assert bot.message_fetcher.reply_calls == []
        assert bot.message_fetcher.calls == []

    def test_its_own_message_is_ignored_before_any_fetch(self, bot):
        bot.handle_envelope(frame(sender="bot-1"))
        assert bot.responder.sent == []
        assert bot.message_fetcher.calls == []

    def test_a_deletion_ping_is_ignored(self, bot):
        bot.handle_envelope(frame(deleted="m9"))
        assert bot.responder.sent == []

    def test_unauthenticated_frames_are_dropped(self, bot):
        bot.handle_envelope(frame(auth=False))
        bot.handle_envelope(frame(status=False))
        assert bot.responder.sent == []

    def test_call_signalling_noise_is_dropped_silently(self, bot):
        for event in ("istyping_broadcast", "incomingcall", "new_producer",
                      "active_users", "contactslist"):
            bot.handle_envelope(frame(event=event))
        assert bot.responder.sent == []
        assert bot.stats.frames_seen == 5


class TestThreading:
    """Replies hang off the message that asked.

    An unthreaded answer in a busy realm channel arrives detached from its
    question - and the bot is slow by construction (fetch, index, retrieve,
    generate), so other messages will usually have landed in between.
    """

    def test_reply_targets_the_mentioning_message_not_the_latest(self):
        # m2 mentions us; m3 arrives afterwards from someone else. The reply
        # must attach to m2.
        fetcher = FakeFetcher([
            PlatformMessage("m1", "conv-1", "human-1", "we agreed on tiered pricing", 1000),
            PlatformMessage("m2", "conv-1", "human-1",
                            "@assistant what did we decide about pricing?", 2000),
            PlatformMessage("m3", "conv-1", "human-9", "unrelated chatter", 3000),
        ])
        b = ChatterloopBot(
            identity=BOT, policy=AddressedOnlyPolicy(BOT, cooldown_seconds=0.0),
            started_at_ms=0,
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=fetcher, mention_fetcher=FakeMentionFetcher(),
        )
        b.handle_envelope(frame())
        assert b.responder.sent[0]["reply_to_message_id"] == "m2"

    def test_reply_targets_the_mentioners_message_not_someone_elses(self):
        # Two people mention the bot; the frame names human-1 as the sender,
        # so the reply belongs on human-1's message.
        fetcher = FakeFetcher([
            PlatformMessage("m1", "conv-1", "human-2", "@assistant hello", 1000),
            PlatformMessage("m2", "conv-1", "human-1", "@assistant pricing?", 2000),
        ])
        b = ChatterloopBot(
            identity=BOT, policy=AddressedOnlyPolicy(BOT, cooldown_seconds=0.0),
            started_at_ms=0,
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=fetcher, mention_fetcher=FakeMentionFetcher(),
        )
        b.handle_envelope(frame(sender="human-1"))
        assert b.responder.sent[0]["reply_to_message_id"] == "m2"

    def test_the_most_recent_matching_message_wins(self):
        # The same person mentioned us twice; answer the latest.
        fetcher = FakeFetcher([
            PlatformMessage("m1", "conv-1", "human-1", "@assistant one", 1000),
            PlatformMessage("m2", "conv-1", "human-1", "@assistant two", 2000),
        ])
        b = ChatterloopBot(
            identity=BOT, policy=AddressedOnlyPolicy(BOT, cooldown_seconds=0.0),
            started_at_ms=0,
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=fetcher, mention_fetcher=FakeMentionFetcher(),
        )
        b.handle_envelope(frame())
        assert b.responder.sent[0]["reply_to_message_id"] == "m2"

    def test_the_bots_own_message_is_never_a_reply_target(self):
        # The bot's own prior reply also contains no address, but guard the
        # ordering explicitly: scanning back must skip our own messages.
        fetcher = FakeFetcher([
            PlatformMessage("m1", "conv-1", "human-1", "@assistant pricing?", 1000),
            PlatformMessage("m2", "conv-1", "bot-1", "@assistant was mentioned", 2000),
        ])
        b = ChatterloopBot(
            identity=BOT, policy=AddressedOnlyPolicy(BOT, cooldown_seconds=0.0),
            started_at_ms=0,
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=fetcher, mention_fetcher=FakeMentionFetcher(),
        )
        b.handle_envelope(frame())
        assert b.responder.sent[0]["reply_to_message_id"] == "m1"

    def test_the_dedupe_key_tracks_the_same_message(self):
        b = ChatterloopBot(
            identity=BOT, policy=AddressedOnlyPolicy(BOT, cooldown_seconds=0.0),
            started_at_ms=0,
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=FakeFetcher(), mention_fetcher=FakeMentionFetcher(),
        )
        b.handle_envelope(frame())
        # Same message, redelivered - one reply, threaded once.
        b.handle_envelope(frame())
        assert len(b.responder.sent) == 1


class TestQueryConstruction:
    def test_the_bots_own_handle_is_stripped_before_retrieval(self, bot):
        bot.handle_envelope(frame())
        assert bot.retrieval.calls[0]["query"] == "what did we decide about pricing?"

    def test_retrieval_is_scoped_to_the_conversation_tenant(self, bot):
        bot.handle_envelope(frame())
        call = bot.retrieval.calls[0]
        assert call["tenant_id"] == conversation_tenant("conv-1")
        assert call["conversation_id"] == "conv-1"

    def test_history_is_indexed_before_retrieval(self, bot):
        bot.handle_envelope(frame())
        assert len(bot.ingestion.indexed) == 2
        assert {r["message_id"] for r in bot.ingestion.indexed} == {"m1", "m2"}

    def test_the_bots_own_past_messages_index_as_assistant(self):
        fetcher = FakeFetcher([
            PlatformMessage("m1", "conv-1", "bot-1", "earlier answer", 1000),
            PlatformMessage("m2", "conv-1", "human-1", "@assistant and now?", 2000),
        ])
        b = ChatterloopBot(
            identity=BOT, policy=AddressedOnlyPolicy(BOT, cooldown_seconds=0.0),
            started_at_ms=0,
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=fetcher, mention_fetcher=FakeMentionFetcher(),
        )
        b.handle_envelope(frame())
        roles = {r["message_id"]: r["role"] for r in b.ingestion.indexed}
        assert roles["m1"] is Role.ASSISTANT
        assert roles["m2"] is Role.USER


class TestUnresolvableMentions:
    def test_no_fetchable_history_means_silence(self):
        b = ChatterloopBot(
            identity=BOT, policy=AddressedOnlyPolicy(BOT, cooldown_seconds=0.0),
            started_at_ms=0,
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=FakeFetcher([]), mention_fetcher=FakeMentionFetcher(),
        )
        b.handle_envelope(frame())
        # The default wiring: reads the event, cannot read the message, says
        # nothing. Correct behaviour for an unwired read path.
        assert b.responder.sent == []
        assert b.stats.triggers_seen == 1

    def test_a_mention_not_present_in_history_is_not_answered(self):
        b = ChatterloopBot(
            identity=BOT, policy=AddressedOnlyPolicy(BOT, cooldown_seconds=0.0),
            started_at_ms=0,
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=FakeFetcher([
                PlatformMessage("m1", "conv-1", "human-1", "no address here", 1000),
            ]),
            mention_fetcher=FakeMentionFetcher(),
        )
        b.handle_envelope(frame())
        assert b.responder.sent == []

    def test_an_empty_generation_is_not_sent(self):
        b = ChatterloopBot(
            identity=BOT, policy=AddressedOnlyPolicy(BOT, cooldown_seconds=0.0),
            started_at_ms=0,
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(reply="   "), responder=RecordingResponder(),
            message_fetcher=FakeFetcher(), mention_fetcher=FakeMentionFetcher(),
        )
        b.handle_envelope(frame())
        assert b.responder.sent == []


class TestRedelivery:
    def test_the_same_message_is_answered_once(self, bot):
        bot.handle_envelope(frame())
        bot.handle_envelope(frame())
        assert len(bot.responder.sent) == 1
        assert bot.stats.ignore_reasons.get("already handled") == 1


class TestComments:
    def test_a_comment_mention_is_answered(self):
        b = ChatterloopBot(
            identity=BOT, policy=AddressedOnlyPolicy(BOT, cooldown_seconds=0.0),
            started_at_ms=0,
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=FakeFetcher(),
            mention_fetcher=FakeMentionFetcher([
                PendingMention(comment_id="cm1", post_id="p1",
                               author_entity_id="human-2",
                               text="@assistant is this still accurate?", created_at=1)
            ]),
        )
        b.handle_envelope(frame(event="notifications"))
        assert b.responder.sent == [
            {"kind": "comment", "post_id": "p1", "comment_id": "cm1",
             "text": "Tiered pricing."}
        ]

    def test_the_bots_own_comment_is_ignored(self):
        b = ChatterloopBot(
            identity=BOT, policy=AddressedOnlyPolicy(BOT, cooldown_seconds=0.0),
            started_at_ms=0,
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=FakeFetcher(),
            mention_fetcher=FakeMentionFetcher([
                PendingMention(comment_id="cm1", post_id="p1",
                               author_entity_id="bot-1", text="@assistant hi", created_at=1)
            ]),
        )
        b.handle_envelope(frame(event="notifications"))
        assert b.responder.sent == []

    def test_a_notifications_frame_with_nothing_pending_is_silent(self, bot):
        bot.handle_envelope(frame(event="notifications"))
        assert bot.responder.sent == []


class TestResilience:
    def test_a_malformed_envelope_does_not_raise(self, bot):
        bot.handle_envelope({"nonsense": True})
        bot.handle_envelope({"event": "messages_list", "message": "not a dict"})
        assert bot.responder.sent == []

    def test_a_fetcher_failure_is_contained(self, bot):
        class Exploding:
            def fetch_recent(self, conversation_id, limit):
                raise RuntimeError("service down")

        bot.message_fetcher = Exploding()
        bot.handle_envelope(frame())
        assert bot.stats.errors == 1
        assert bot.responder.sent == []
        # And the bot keeps working afterwards.
        bot.message_fetcher = FakeFetcher()
        bot.handle_envelope(frame(conversation="conv-2"))
        assert len(bot.responder.sent) == 1


class TestReplyingWithoutAHandle:
    """The second way in: a message threaded under one the bot wrote.

    The frame cannot say so - a `messages_list` payload carries no message id
    and no `replyingTo` - so every test here goes through the probe, and the
    probe's fake returns only what the server would: replies to the BOT's own
    messages, never to anyone else's.
    """

    def _bot(self, replies, history=None, **kw):
        return ChatterloopBot(
            identity=BOT, policy=AddressedOnlyPolicy(BOT, cooldown_seconds=0.0),
            started_at_ms=0,
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=FakeFetcher(messages=history, replies=replies),
            mention_fetcher=FakeMentionFetcher(), **kw,
        )

    def test_a_reply_to_the_bot_is_answered_with_no_mention_anywhere(self):
        b = self._bot(
            replies=[PlatformMessage("m3", "conv-1", "human-1", "and the second one?",
                                     3000, is_reply=True, replying_to="bot-m1",
                                     replying_to_sender_entity_id="bot-1")],
            history=[
                PlatformMessage("bot-m1", "conv-1", "bot-1", "tiered pricing", 2000),
                PlatformMessage("m3", "conv-1", "human-1", "and the second one?", 3000),
            ],
        )
        b.handle_envelope(frame(mentioner=False))
        assert b.responder.sent == [{
            "kind": "conversation", "conversation_id": "conv-1",
            "text": "Tiered pricing.",
            # Threaded under the reply, exactly as a mention would be.
            "reply_to_message_id": "m3",
        }]

    def test_the_query_is_the_message_itself(self):
        b = self._bot(replies=[
            PlatformMessage("m3", "conv-1", "human-1", "and the second one?", 3000,
                            is_reply=True, replying_to="bot-m1",
                            replying_to_sender_entity_id="bot-1")])
        b.handle_envelope(frame(mentioner=False))
        # Nothing to strip. strip_mentions runs anyway, and must not eat words.
        assert b.retrieval.calls[0]["query"] == "and the second one?"

    def test_history_is_still_indexed_on_the_reply_path(self):
        b = self._bot(replies=[
            PlatformMessage("m3", "conv-1", "human-1", "and the second one?", 3000,
                            is_reply=True, replying_to="bot-m1",
                            replying_to_sender_entity_id="bot-1")])
        b.handle_envelope(frame(mentioner=False))
        # Indexing is lazy either way: being addressed is what pays for it.
        assert {r["message_id"] for r in b.ingestion.indexed} == {"m1", "m2"}

    def test_a_reply_from_someone_other_than_the_frames_sender_is_not_taken(self):
        # human-9's reply is real but this frame is about human-1's message.
        # Answering the wrong one would thread the reply under the wrong turn.
        b = self._bot(replies=[
            PlatformMessage("m3", "conv-1", "human-9", "unrelated follow-up", 3000,
                            is_reply=True, replying_to="bot-m1",
                            replying_to_sender_entity_id="bot-1")])
        b.handle_envelope(frame(mentioner=False, sender="human-1"))
        assert b.responder.sent == []

    def test_the_newest_reply_from_that_sender_wins(self):
        b = self._bot(replies=[
            PlatformMessage("m3", "conv-1", "human-1", "first follow-up", 3000,
                            is_reply=True, replying_to="bot-m1",
                            replying_to_sender_entity_id="bot-1"),
            PlatformMessage("m4", "conv-1", "human-1", "second follow-up", 4000,
                            is_reply=True, replying_to="bot-m1",
                            replying_to_sender_entity_id="bot-1"),
        ])
        b.handle_envelope(frame(mentioner=False))
        assert b.responder.sent[0]["reply_to_message_id"] == "m4"

    def test_the_bots_own_reply_is_never_taken_as_a_trigger(self):
        # The unbounded loop: the bot's answers are themselves replies, and one
        # read back as input needs no @handle to sustain a thread with itself.
        b = self._bot(replies=[
            PlatformMessage("m4", "conv-1", "bot-1", "an earlier answer", 4000,
                            is_reply=True, replying_to="bot-m1",
                            replying_to_sender_entity_id="bot-1")])
        b.handle_envelope(frame(mentioner=False, sender="bot-1"))
        assert b.responder.sent == []
        # Not even probed: the frame is our own message.
        assert b.message_fetcher.reply_calls == []

    def test_an_already_answered_reply_costs_no_history_fetch(self):
        # The probe returns a window, so every later message in the
        # conversation re-offers replies already answered. Those must be
        # dropped before the expensive read, not after.
        reply = PlatformMessage("m3", "conv-1", "human-1", "and the second one?",
                                3000, is_reply=True, replying_to="bot-m1",
                                replying_to_sender_entity_id="bot-1")
        b = self._bot(replies=[reply])
        b.handle_envelope(frame(mentioner=False))
        assert len(b.responder.sent) == 1

        fetches = len(b.message_fetcher.calls)
        b.handle_envelope(frame(mentioner=False))
        assert len(b.responder.sent) == 1
        assert len(b.message_fetcher.calls) == fetches

    def test_a_reply_is_counted_under_its_own_reason(self):
        b = self._bot(replies=[
            PlatformMessage("m3", "conv-1", "human-1", "and the second one?", 3000,
                            is_reply=True, replying_to="bot-m1",
                            replying_to_sender_entity_id="bot-1")])
        b.handle_envelope(frame(mentioner=False))
        assert b.stats.trigger_reasons == {"reply": 1}

    def test_a_message_that_mentions_and_replies_is_answered_once(self):
        # The mention path claims it (the frame is authoritative), and the
        # dedupe key is the message rather than the route, so the probe on a
        # later frame cannot answer it a second time.
        b = self._bot(
            replies=[PlatformMessage("m2", "conv-1", "human-1",
                                     "@assistant what did we decide about pricing?",
                                     2000, is_reply=True, replying_to="bot-m1",
                                     replying_to_sender_entity_id="bot-1")])
        b.handle_envelope(frame())
        b.handle_envelope(frame(mentioner=False))
        assert len(b.responder.sent) == 1

    def test_a_failing_probe_is_contained(self):
        class Exploding:
            def fetch_recent(self, conversation_id, limit):
                return []

            def fetch_replies_to_me(self, conversation_id, limit):
                raise RuntimeError("service down")

        b = self._bot(replies=[])
        b.message_fetcher = Exploding()
        b.handle_envelope(frame(mentioner=False))
        assert b.stats.errors == 1
        assert b.responder.sent == []

    def test_the_feature_can_be_turned_off_entirely(self):
        b = self._bot(
            replies=[PlatformMessage("m3", "conv-1", "human-1", "and the second one?",
                                     3000, is_reply=True, replying_to="bot-m1",
                                     replying_to_sender_entity_id="bot-1")],
            answer_replies=False,
        )
        b.handle_envelope(frame(mentioner=False))
        assert b.responder.sent == []
        # Off means genuinely off: mention-only, and an unmentioned message
        # costs no read at all.
        assert b.message_fetcher.reply_calls == []

    def test_mentions_still_work_when_replies_are_off(self):
        b = self._bot(replies=[], answer_replies=False)
        b.handle_envelope(frame())
        assert len(b.responder.sent) == 1


class TestCommentReplies:
    def _bot(self, pending=None, replies=None):
        return ChatterloopBot(
            identity=BOT, policy=AddressedOnlyPolicy(BOT, cooldown_seconds=0.0),
            started_at_ms=0,
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=FakeFetcher(),
            mention_fetcher=FakeMentionFetcher(pending=pending, replies=replies),
        )

    def test_a_reply_to_the_bots_comment_is_answered(self):
        b = self._bot(replies=[
            PendingMention(comment_id="cm2", post_id="p1",
                           author_entity_id="human-2",
                           text="and what about the other one?", kind="reply", created_at=1)
        ])
        b.handle_envelope(frame(event="notifications"))
        assert b.responder.sent == [
            {"kind": "comment", "post_id": "p1", "comment_id": "cm2",
             "text": "Tiered pricing."}
        ]
        assert b.stats.trigger_reasons == {"reply": 1}

    def test_mentions_and_replies_are_both_collected_from_one_ping(self):
        # The notifications frame is unaddressed - it means "go and look" - so
        # there is nothing to route on and both stores are read.
        b = self._bot(
            pending=[PendingMention(comment_id="cm1", post_id="p1",
                                    author_entity_id="human-2",
                                    text="@assistant still accurate?", created_at=1)],
            replies=[PendingMention(comment_id="cm2", post_id="p1",
                                    author_entity_id="human-3",
                                    text="and the other one?", kind="reply", created_at=1)],
        )
        b.handle_envelope(frame(event="notifications"))
        assert [s["comment_id"] for s in b.responder.sent] == ["cm1", "cm2"]

    def test_one_comment_appearing_in_both_lists_is_answered_once(self):
        row = PendingMention(comment_id="cm1", post_id="p1",
                             author_entity_id="human-2", text="@assistant hello", created_at=1)
        b = self._bot(pending=[row], replies=[row])
        b.handle_envelope(frame(event="notifications"))
        assert len(b.responder.sent) == 1

    def test_a_failing_reply_read_does_not_cost_the_mentions(self):
        class HalfBroken:
            def fetch_comment_mentions(self, limit):
                return [PendingMention(comment_id="cm1", post_id="p1",
                                       author_entity_id="human-2",
                                       text="@assistant still accurate?", created_at=1)]

            def fetch_comment_replies(self, limit):
                raise RuntimeError("service down")

        b = self._bot()
        b.mention_fetcher = HalfBroken()
        b.handle_envelope(frame(event="notifications"))
        # The path that worked before this feature existed still works.
        assert len(b.responder.sent) == 1
        assert b.stats.errors == 1

    def test_the_bots_own_reply_to_its_own_comment_is_ignored(self):
        b = self._bot(replies=[
            PendingMention(comment_id="cm2", post_id="p1",
                           author_entity_id="bot-1", text="following up",
                           kind="reply", created_at=1)
        ])
        b.handle_envelope(frame(event="notifications"))
        assert b.responder.sent == []

    def test_a_comment_with_no_post_is_dropped_before_the_send(self):
        # target.supportingID is where the post id comes from, and a
        # notification written without a target has none. Such a row cannot be
        # replied on at all, and its tenant would be the bare string "post:" -
        # every unanswerable comment sharing one partition.
        b = self._bot(replies=[
            PendingMention(comment_id="cm2", post_id="",
                           author_entity_id="human-2",
                           text="and the other one?", kind="reply", created_at=1)
        ])
        b.handle_envelope(frame(event="notifications"))
        assert b.responder.sent == []
        assert b.stats.ignore_reasons.get("comment mention has no post") == 1

    def test_the_comment_answered_is_the_one_passed_as_the_parent(self):
        # The bot names what it is ANSWERING and does not try to predict where
        # the row lands: replying to a reply re-parents to the top-level
        # ancestor server-side, and computing that here would reimplement the
        # rule the endpoint exists to own.
        b = self._bot(replies=[
            PendingMention(comment_id="a-reply", post_id="p1",
                           author_entity_id="human-2",
                           text="and the other one?", kind="reply", created_at=1)
        ])
        b.handle_envelope(frame(event="notifications"))
        assert b.responder.sent[0]["comment_id"] == "a-reply"


class TestLiveEventsOnly:
    """The bot answers what happened on its watch, and nothing else.

    Frames need no guarding - pub/sub has no replay, so one published while the
    bot was down is gone. What needs guarding is everything the bot READS to
    resolve a frame: the reply probe returns a window, and comment
    notifications are durable and accumulate indefinitely. Both are backlog
    after a restart, and answering backlog means messaging real people about
    things they said hours ago.
    """

    NOW = 1_000_000_000_000
    BEFORE = NOW - 60_000
    AFTER = NOW + 60_000

    def _bot(self, replies=None, pending=None, comment_replies=None, **kw):
        kw.setdefault("started_at_ms", self.NOW)
        return ChatterloopBot(
            identity=BOT, policy=AddressedOnlyPolicy(BOT, cooldown_seconds=0.0),
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=FakeFetcher(replies=replies),
            mention_fetcher=FakeMentionFetcher(pending=pending, replies=comment_replies),
            **kw,
        )

    def _reply(self, created_at, message_id="m3"):
        return PlatformMessage(
            message_id, "conv-1", "human-1", "and the second one?", created_at,
            is_reply=True, replying_to="bot-m1",
            replying_to_sender_entity_id="bot-1",
        )

    # ------------------------------------------------------------- messages

    def test_a_reply_from_before_the_bot_started_is_not_answered(self):
        b = self._bot(replies=[self._reply(self.BEFORE)])
        b.handle_envelope(frame(mentioner=False))
        assert b.responder.sent == []
        assert b.stats.ignore_reasons.get("reply predates this process") == 1

    def test_a_reply_from_after_the_bot_started_is_answered(self):
        b = self._bot(replies=[self._reply(self.AFTER)])
        b.handle_envelope(frame(mentioner=False))
        assert len(b.responder.sent) == 1

    def test_a_stale_reply_costs_no_history_fetch(self):
        # The point of checking before resolving: a stale candidate must not
        # pay for the 40-message read that indexing needs.
        b = self._bot(replies=[self._reply(self.BEFORE)])
        b.handle_envelope(frame(mentioner=False))
        assert b.message_fetcher.calls == []

    def test_a_stale_reply_is_not_reconsidered_on_every_frame(self):
        # It stays in the probe's window forever. Recording it as seen is what
        # stops each subsequent message in the conversation re-judging and
        # re-logging the same row.
        b = self._bot(replies=[self._reply(self.BEFORE)])
        b.handle_envelope(frame(mentioner=False))
        b.handle_envelope(frame(mentioner=False))
        b.handle_envelope(frame(mentioner=False))
        assert b.stats.ignore_reasons.get("reply predates this process") == 1

    def test_a_reply_with_no_timestamp_is_treated_as_stale(self):
        # Undatable means unanswerable. Being wrong here sends a real message
        # about something somebody said hours ago, so an unreadable row gets
        # silence rather than the benefit of the doubt.
        b = self._bot(replies=[self._reply(0)])
        b.handle_envelope(frame(mentioner=False))
        assert b.responder.sent == []

    def test_a_fresh_reply_still_wins_when_a_stale_one_sits_behind_it(self):
        b = self._bot(replies=[
            self._reply(self.BEFORE, message_id="old"),
            self._reply(self.AFTER, message_id="new"),
        ])
        b.handle_envelope(frame(mentioner=False))
        assert b.responder.sent[0]["reply_to_message_id"] == "new"

    # ------------------------------------------------------------- comments

    def test_a_comment_from_before_the_bot_started_is_not_answered(self):
        b = self._bot(pending=[
            PendingMention(comment_id="cm1", post_id="p1",
                           author_entity_id="human-2",
                           text="@assistant still accurate?",
                           created_at=self.BEFORE)
        ])
        b.handle_envelope(frame(event="notifications"))
        assert b.responder.sent == []
        assert b.stats.ignore_reasons.get("comment predates this process") == 1

    def test_a_comment_from_after_the_bot_started_is_answered(self):
        b = self._bot(pending=[
            PendingMention(comment_id="cm1", post_id="p1",
                           author_entity_id="human-2",
                           text="@assistant still accurate?",
                           created_at=self.AFTER)
        ])
        b.handle_envelope(frame(event="notifications"))
        assert len(b.responder.sent) == 1

    def test_an_offline_backlog_does_not_become_a_burst(self):
        # THE FAILURE THIS RULE EXISTS FOR. Notifications are durable, and
        # _on_notifications answers every row it is handed - so one ping after
        # a restart used to fire a reply per accumulated mention. The
        # per-conversation cooldown does not help: its scope is the post id, so
        # twenty mentions across twenty posts are twenty separate scopes.
        backlog = [
            PendingMention(comment_id=f"cm{i}", post_id=f"p{i}",
                           author_entity_id="human-2",
                           text="@assistant hello", created_at=self.BEFORE)
            for i in range(20)
        ]
        b = self._bot(pending=backlog)
        b.handle_envelope(frame(event="notifications"))
        assert b.responder.sent == []
        assert b.stats.ignore_reasons.get("comment predates this process") == 20

    def test_one_live_comment_among_a_backlog_is_still_answered(self):
        rows = [
            PendingMention(comment_id=f"cm{i}", post_id=f"p{i}",
                           author_entity_id="human-2",
                           text="@assistant hello", created_at=self.BEFORE)
            for i in range(5)
        ]
        rows.append(PendingMention(comment_id="live", post_id="p9",
                                   author_entity_id="human-2",
                                   text="@assistant and now?",
                                   created_at=self.AFTER))
        b = self._bot(pending=rows)
        b.handle_envelope(frame(event="notifications"))
        assert [s["comment_id"] for s in b.responder.sent] == ["live"]

    def test_a_comment_with_no_timestamp_is_treated_as_stale(self):
        b = self._bot(pending=[
            PendingMention(comment_id="cm1", post_id="p1",
                           author_entity_id="human-2",
                           text="@assistant hello", created_at=0)
        ])
        b.handle_envelope(frame(event="notifications"))
        assert b.responder.sent == []

    # ------------------------------------------------------------ the switch

    def test_the_rule_can_be_turned_off_for_a_deliberate_catch_up(self):
        b = self._bot(replies=[self._reply(self.BEFORE)], only_live_events=False)
        b.handle_envelope(frame(mentioner=False))
        assert len(b.responder.sent) == 1

    def test_off_also_lets_the_comment_backlog_through(self):
        b = self._bot(
            pending=[PendingMention(comment_id="cm1", post_id="p1",
                                    author_entity_id="human-2",
                                    text="@assistant hello",
                                    created_at=self.BEFORE)],
            only_live_events=False,
        )
        b.handle_envelope(frame(event="notifications"))
        assert len(b.responder.sent) == 1

    # ------------------------------------------------------- the mention path

    def test_a_message_mention_needs_no_timestamp(self):
        # The mention path is gated by the FRAME's own mentioner field, which
        # only exists for a message that arrived while the bot was listening.
        # It is therefore live by construction, and is deliberately not subject
        # to this rule - the fixture's history carries created_at values well
        # before started_at_ms and must still be answered.
        b = self._bot()
        b.handle_envelope(frame())
        assert len(b.responder.sent) == 1
