"""Mention parsing, and its parity with the platform.

The cases below are the ones the two existing implementations document as
mattering. If this file and either of them disagree, the bot's idea of "I was
mentioned" has drifted from the platform's, and mentions will be missed or
invented.
"""

from __future__ import annotations

import pytest

from rag_service.chatterloop.mentions import (
    MAX_MENTIONS,
    extract_handles,
    is_addressed_to,
    normalise_handle,
    strip_mentions,
)


class TestExtraction:
    def test_simple_mention(self):
        assert extract_handles("hey @ana can you look") == ["ana"]

    def test_mention_at_start_of_string(self):
        # The (?:^|\s) alternation exists for exactly this.
        assert extract_handles("@ana hello") == ["ana"]

    def test_multiple_mentions_preserve_order(self):
        assert extract_handles("@ana and @ben") == ["ana", "ben"]

    def test_case_is_folded(self):
        # "@Ana" and "@ana" must not spend two of the mention budget on one
        # person.
        assert extract_handles("@Ana @ana") == ["ana"]

    def test_email_is_not_a_mention(self):
        # The single most important negative case: the leading (?:^|\s) is
        # what stops "you@example.com" from mentioning @example.
        assert extract_handles("write to you@example.com") == []

    def test_trailing_dot_yields_both_forms(self):
        # The character class includes "." and is greedy, so "@ana." captures
        # "ana." - both existing implementations emit the stripped form too.
        assert extract_handles("thanks @ana.") == ["ana.", "ana"]

    @pytest.mark.parametrize("punct", [",", "!", "?", ";", ":"])
    def test_trailing_punctuation_terminates_the_handle(self, punct):
        assert extract_handles(f"hi @ana{punct} bye") == ["ana"]

    def test_handles_may_contain_dots_underscores_hyphens(self):
        assert extract_handles("@a.b_c-d ok") == ["a.b_c-d"]

    def test_over_long_handle_matches_nothing(self):
        # Over-long input matches nothing rather than being truncated to a
        # different handle - the safe direction.
        assert extract_handles("@" + "a" * 31 + " hi") == []

    def test_handle_at_exactly_thirty_characters_matches(self):
        handle = "a" * 30
        assert extract_handles(f"@{handle} hi") == [handle]

    def test_empty_and_none_are_safe(self):
        assert extract_handles("") == []
        assert extract_handles(None) == []  # type: ignore[arg-type]

    def test_bare_at_is_not_a_mention(self):
        assert extract_handles("hey @ there") == []

    def test_mention_budget_is_capped(self):
        text = " ".join(f"@user{i}" for i in range(40))
        assert len(extract_handles(text)) <= MAX_MENTIONS + 1


class TestNormalise:
    @pytest.mark.parametrize(
        "raw,expected",
        [("@Ana", "ana"), ("ana", "ana"), ("  @Ana  ", "ana"), ("", ""), (None, "")],
    )
    def test_forms_converge(self, raw, expected):
        # `mentioner.username` arrives as "@ana" while Account.username is
        # stored bare, so every comparison has to go through this.
        assert normalise_handle(raw) == expected


class TestIsAddressedTo:
    def test_matches_the_bot_handle(self):
        assert is_addressed_to("hey @assistant help", {"assistant"})

    def test_accepts_an_at_prefixed_handle(self):
        assert is_addressed_to("hey @assistant help", {"@assistant"})

    def test_ignores_other_peoples_mentions(self):
        assert not is_addressed_to("hey @ana help", {"assistant"})

    def test_case_insensitive(self):
        assert is_addressed_to("hey @Assistant", {"assistant"})

    def test_alias_matches(self):
        assert is_addressed_to("@bot hi", {"assistant", "bot"})

    def test_empty_inputs(self):
        assert not is_addressed_to("", {"assistant"})
        assert not is_addressed_to("@assistant", set())


class TestStripMentions:
    def test_own_handle_is_removed(self):
        # The address is not part of the question, and leaving it in drags
        # both retrieval legs toward the bot's own name.
        assert strip_mentions("@assistant what did we decide?", {"assistant"}) == (
            "what did we decide?"
        )

    def test_mid_sentence_address_is_removed(self):
        assert strip_mentions("hey @assistant what now", {"assistant"}) == "hey what now"

    def test_other_mentions_are_content_and_stay(self):
        assert strip_mentions("@assistant ask @ana about it", {"assistant"}) == (
            "ask @ana about it"
        )

    def test_trailing_dot_form_is_also_stripped(self):
        assert strip_mentions("thanks @assistant.", {"assistant"}) == "thanks"

    def test_no_handles_returns_trimmed_text(self):
        assert strip_mentions("  hello  ", set()) == "hello"

    def test_empty_text(self):
        assert strip_mentions("", {"assistant"}) == ""

    def test_message_that_is_only_an_address_becomes_empty(self):
        # Which the policy then rejects as unresolvable, rather than the bot
        # answering a question with no content.
        assert strip_mentions("@assistant", {"assistant"}) == ""
