"""When the bot stays quiet.

Silence is the default. Each test names one reason.
"""

from __future__ import annotations

import pytest

from rag_service.chatterloop.identity import BotIdentity
from rag_service.chatterloop.policy import MentionOnlyPolicy, Verdict
from rag_service.chatterloop.triggers import MentionSource, MentionTrigger

BOT = BotIdentity(entity_id="bot-1", handle="assistant")


def trigger(**kw) -> MentionTrigger:
    defaults = dict(
        source=MentionSource.MESSAGE,
        author_entity_id="human-1",
        conversation_id="conv-1",
        text="@assistant what did we decide?",
        query="what did we decide?",
        dedupe_key="msg:1",
    )
    defaults.update(kw)
    return MentionTrigger(**defaults)


@pytest.fixture
def policy() -> MentionOnlyPolicy:
    return MentionOnlyPolicy(BOT, cooldown_seconds=5.0, max_replies_per_hour=3)


class TestRespond:
    def test_a_plain_mention_gets_a_reply(self, policy):
        assert policy.evaluate(trigger()).should_respond


class TestLoopPrevention:
    def test_never_replies_to_itself(self, policy):
        # The unbounded failure: a self-reply is itself a message.
        decision = policy.evaluate(trigger(author_entity_id="bot-1"))
        assert decision.verdict is Verdict.IGNORE
        assert "itself" in decision.reason

    def test_never_replies_to_an_ignored_entity(self):
        policy = MentionOnlyPolicy(BOT, ignore_entity_ids=frozenset({"other-bot"}))
        decision = policy.evaluate(trigger(author_entity_id="other-bot"))
        assert decision.verdict is Verdict.IGNORE
        assert "ignore list" in decision.reason

    def test_self_check_precedes_everything_else(self, policy):
        # Even a well-formed, novel, in-budget trigger loses to being our own.
        assert not policy.evaluate(
            trigger(author_entity_id="bot-1", dedupe_key="msg:novel")
        ).should_respond


class TestDeduplication:
    def test_same_trigger_twice_replies_once(self, policy):
        first = trigger()
        assert policy.evaluate(first).should_respond
        policy.record_seen(first.dedupe_key)
        assert not policy.evaluate(trigger()).should_respond

    def test_different_triggers_both_reply(self, policy):
        assert policy.evaluate(trigger(dedupe_key="msg:1")).should_respond
        policy.record_seen("msg:1")
        policy.record_reply(trigger(), now=0.0)
        assert policy.evaluate(
            trigger(dedupe_key="msg:2", conversation_id="conv-2")
        ).should_respond

    def test_ignored_triggers_are_recorded_too(self, policy):
        # Re-judging is not free once a fetch is involved.
        policy.record_seen("msg:x")
        assert "already handled" in policy.evaluate(trigger(dedupe_key="msg:x")).reason


class TestResolution:
    def test_unresolved_mention_is_not_answered(self, policy):
        # A mention we know happened but cannot read is not a question.
        decision = policy.evaluate(trigger(text="", query=""))
        assert decision.verdict is Verdict.IGNORE
        assert "resolve" in decision.reason

    def test_whitespace_only_query_counts_as_unresolved(self, policy):
        # What a bare "@assistant" becomes after stripping the address.
        assert not policy.evaluate(trigger(query="   ")).should_respond


class TestCooldown:
    def test_second_mention_within_cooldown_is_dropped(self, policy):
        policy.record_reply(trigger(), now=100.0)
        decision = policy.evaluate(trigger(dedupe_key="msg:2"), now=102.0)
        assert "cooldown" in decision.reason

    def test_after_the_cooldown_it_answers_again(self, policy):
        policy.record_reply(trigger(), now=100.0)
        assert policy.evaluate(trigger(dedupe_key="msg:2"), now=106.0).should_respond

    def test_cooldown_is_per_conversation(self, policy):
        policy.record_reply(trigger(conversation_id="conv-1"), now=100.0)
        # A different conversation is a different audience.
        assert policy.evaluate(
            trigger(conversation_id="conv-2", dedupe_key="msg:2"), now=101.0
        ).should_respond


class TestHourlyCeiling:
    def test_ceiling_stops_further_replies(self, policy):
        for i in range(3):  # max_replies_per_hour=3
            policy.record_reply(trigger(), now=float(i))
        decision = policy.evaluate(trigger(dedupe_key="msg:9"), now=100.0)
        assert "hourly reply limit" in decision.reason

    def test_the_window_rolls_forward(self, policy):
        for i in range(3):
            policy.record_reply(trigger(), now=float(i))
        # More than an hour later the old replies fall out of the window.
        assert policy.evaluate(trigger(dedupe_key="msg:9"), now=4000.0).should_respond


class TestFailedSends:
    def test_a_reply_that_was_never_sent_costs_no_budget(self, policy):
        # record_reply is called after a successful send, not on decision - a
        # broken outbound path must not rate-limit the bot into silence.
        for _ in range(5):
            assert policy.evaluate(trigger(dedupe_key="fresh")).should_respond
