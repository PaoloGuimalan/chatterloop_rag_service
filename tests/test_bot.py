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
from rag_service.chatterloop.policy import MentionOnlyPolicy
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
    def __init__(self, messages=None):
        self.messages = messages if messages is not None else [
            PlatformMessage("m1", "conv-1", "human-1", "we agreed on tiered pricing", 1000),
            PlatformMessage("m2", "conv-1", "human-1",
                            "@assistant what did we decide about pricing?", 2000),
        ]
        self.calls: list[str] = []

    def fetch_recent(self, conversation_id, limit):
        self.calls.append(conversation_id)
        return list(self.messages)


class FakeMentionFetcher:
    def __init__(self, pending=None):
        self.pending = pending or []

    def fetch_comment_mentions(self, limit):
        return list(self.pending)


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
        policy=MentionOnlyPolicy(BOT, cooldown_seconds=0.0),
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

    def test_an_unmentioned_message_is_ignored_entirely(self, bot):
        bot.handle_envelope(frame(mentioner=False))
        assert bot.responder.sent == []
        # Not even fetched: no mention, nothing happens at all.
        assert bot.message_fetcher.calls == []
        assert bot.stats.mentions_seen == 0

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
            identity=BOT, policy=MentionOnlyPolicy(BOT, cooldown_seconds=0.0),
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
            identity=BOT, policy=MentionOnlyPolicy(BOT, cooldown_seconds=0.0),
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
            identity=BOT, policy=MentionOnlyPolicy(BOT, cooldown_seconds=0.0),
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
            identity=BOT, policy=MentionOnlyPolicy(BOT, cooldown_seconds=0.0),
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=fetcher, mention_fetcher=FakeMentionFetcher(),
        )
        b.handle_envelope(frame())
        assert b.responder.sent[0]["reply_to_message_id"] == "m1"

    def test_the_dedupe_key_tracks_the_same_message(self):
        b = ChatterloopBot(
            identity=BOT, policy=MentionOnlyPolicy(BOT, cooldown_seconds=0.0),
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
            identity=BOT, policy=MentionOnlyPolicy(BOT, cooldown_seconds=0.0),
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
            identity=BOT, policy=MentionOnlyPolicy(BOT, cooldown_seconds=0.0),
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=FakeFetcher([]), mention_fetcher=FakeMentionFetcher(),
        )
        b.handle_envelope(frame())
        # The default wiring: reads the event, cannot read the message, says
        # nothing. Correct behaviour for an unwired read path.
        assert b.responder.sent == []
        assert b.stats.mentions_seen == 1

    def test_a_mention_not_present_in_history_is_not_answered(self):
        b = ChatterloopBot(
            identity=BOT, policy=MentionOnlyPolicy(BOT, cooldown_seconds=0.0),
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
            identity=BOT, policy=MentionOnlyPolicy(BOT, cooldown_seconds=0.0),
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
            identity=BOT, policy=MentionOnlyPolicy(BOT, cooldown_seconds=0.0),
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=FakeFetcher(),
            mention_fetcher=FakeMentionFetcher([
                PendingMention(comment_id="cm1", post_id="p1",
                               author_entity_id="human-2",
                               text="@assistant is this still accurate?")
            ]),
        )
        b.handle_envelope(frame(event="notifications"))
        assert b.responder.sent == [
            {"kind": "comment", "post_id": "p1", "comment_id": "cm1",
             "text": "Tiered pricing."}
        ]

    def test_the_bots_own_comment_is_ignored(self):
        b = ChatterloopBot(
            identity=BOT, policy=MentionOnlyPolicy(BOT, cooldown_seconds=0.0),
            ingestion=FakeIngestion(), retrieval=FakeRetrieval(),
            generator=FakeGenerator(), responder=RecordingResponder(),
            message_fetcher=FakeFetcher(),
            mention_fetcher=FakeMentionFetcher([
                PendingMention(comment_id="cm1", post_id="p1",
                               author_entity_id="bot-1", text="@assistant hi")
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
