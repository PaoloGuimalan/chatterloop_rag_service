"""The two read seams, now over HTTP.

What these assert is mostly what they asserted before the cut-over - a
malformed row costs one item and not the batch, a dead platform yields nothing
rather than raising - because those properties are the point and should have
survived the change of transport.

What is new is the last test in each class: the pipeline can no longer name
whose data it wants. That used to be a discipline; it is now a missing
parameter.
"""

from __future__ import annotations

import inspect

import pytest

from rag_service.chatterloop.platform.client import (
    PlatformAPIError,
    PlatformAuthError,
)
from rag_service.chatterloop.platform.fetchers import (
    ApiMentionFetcher,
    ApiMessageFetcher,
)


class FakeClient:
    """Stands in for BotApiClient, recording what was asked for."""

    def __init__(self, payload=None, error=None):
        self.payload = payload if payload is not None else {}
        self.error = error
        self.calls = []

    def get(self, path, params=None):
        self.calls.append((path, params))
        if self.error is not None:
            raise self.error
        return self.payload


def _message(**overrides):
    row = {
        "message_id": "m1",
        "conversation_id": "c1",
        "sender_entity_id": "e1",
        "content": "hello",
        "created_at": 1700000000000,
        "message_type": "text",
        "sender_handle": "ana",
    }
    row.update(overrides)
    return row


class TestApiMessageFetcher:
    def test_maps_rows_to_platform_messages(self):
        client = FakeClient({"messages": [_message()]})
        [message] = ApiMessageFetcher(client).fetch_recent("c1", 10)
        assert message.message_id == "m1"
        assert message.sender_entity_id == "e1"
        assert message.content == "hello"
        assert message.created_at == 1700000000000
        assert message.sender_handle == "ana"

    def test_handles_come_from_the_server_not_a_second_query(self):
        client = FakeClient({"messages": [_message(sender_handle="neon-systems")]})
        [message] = ApiMessageFetcher(client).fetch_recent("c1", 10)
        # One call total: the endpoint resolves handles for users, realms and
        # bots alike, which the old two-store version could not do.
        assert len(client.calls) == 1
        assert message.sender_handle == "neon-systems"

    @pytest.mark.parametrize(
        "bad",
        [
            {"message_id": ""},
            {"sender_entity_id": ""},
            {"content": None},
            {"content": {"url": "x.png"}},
        ],
    )
    def test_unusable_rows_are_skipped_not_fatal(self, bad):
        client = FakeClient({"messages": [_message(**bad), _message(message_id="m2")]})
        messages = ApiMessageFetcher(client).fetch_recent("c1", 10)
        assert [m.message_id for m in messages] == ["m2"]

    def test_a_non_dict_row_is_skipped(self):
        client = FakeClient({"messages": ["not-a-row", _message()]})
        assert len(ApiMessageFetcher(client).fetch_recent("c1", 10)) == 1

    def test_an_unreachable_platform_yields_nothing_rather_than_raising(self):
        client = FakeClient(error=PlatformAPIError("boom"))
        assert ApiMessageFetcher(client).fetch_recent("c1", 10) == []

    def test_a_rejected_token_yields_nothing_rather_than_raising(self):
        client = FakeClient(error=PlatformAuthError("401"))
        assert ApiMessageFetcher(client).fetch_recent("c1", 10) == []

    def test_an_empty_conversation_id_never_reaches_the_network(self):
        client = FakeClient({"messages": [_message()]})
        assert ApiMessageFetcher(client).fetch_recent("", 10) == []
        assert client.calls == []

    def test_limit_is_passed_through_and_floored_at_one(self):
        client = FakeClient({"messages": []})
        ApiMessageFetcher(client).fetch_recent("c1", 0)
        assert client.calls[0][1] == {"limit": 1}

    def test_a_missing_created_at_sorts_to_the_beginning_not_the_end(self):
        client = FakeClient({"messages": [_message(created_at="nonsense")]})
        [message] = ApiMessageFetcher(client).fetch_recent("c1", 10)
        # 0, deliberately: an unparseable timestamp must never make an old
        # message look like the newest one.
        assert message.created_at == 0


def _mention(**overrides):
    row = {
        "comment_id": "cm1",
        "post_id": "p1",
        "author_entity_id": "e9",
        "text": "@bot what is this",
        "author_handle": "ana",
    }
    row.update(overrides)
    return row


class TestApiMentionFetcher:
    def test_maps_rows_to_pending_mentions(self):
        client = FakeClient({"mentions": [_mention()]})
        [mention] = ApiMentionFetcher(client).fetch_comment_mentions(10)
        assert mention.comment_id == "cm1"
        assert mention.post_id == "p1"
        assert mention.text == "@bot what is this"
        assert mention.author_handle == "ana"

    @pytest.mark.parametrize("bad", [{"text": ""}, {"text": "   "}, {"text": None}])
    def test_a_mention_with_no_readable_text_is_dropped(self, bad):
        client = FakeClient({"mentions": [_mention(**bad)]})
        assert ApiMentionFetcher(client).fetch_comment_mentions(10) == []

    def test_a_mention_without_a_comment_id_is_skipped(self):
        client = FakeClient({"mentions": [_mention(comment_id=""), _mention()]})
        assert len(ApiMentionFetcher(client).fetch_comment_mentions(10)) == 1

    def test_a_dead_platform_yields_nothing(self):
        client = FakeClient(error=PlatformAPIError("boom"))
        assert ApiMentionFetcher(client).fetch_comment_mentions(10) == []

    def test_it_cannot_ask_for_another_entitys_mentions(self):
        # The old fetcher REQUIRED an entity id, because the Mongo query was
        # ours to write - so reading someone else's notifications was one
        # argument away. The endpoint scopes to the token's own entity and
        # takes no such parameter, which makes that unrepresentable rather
        # than merely not done.
        signature = inspect.signature(ApiMentionFetcher.fetch_comment_mentions)
        assert list(signature.parameters) == ["self", "limit"]

        constructor = inspect.signature(ApiMentionFetcher.__init__)
        assert list(constructor.parameters) == ["self", "client"]
