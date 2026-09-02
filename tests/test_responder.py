"""The write seam.

`build_send_body` is unchanged and so are its tests: the fields the platform
wants did not move, only how they get there (plain JSON to the bot API, which
re-signs it server-side, instead of a JWT this service would have had to sign
with a secret it should never hold).

What is new is HttpResponder actually sending, and the properties that keep one
bad send from taking the loop down with it.
"""

from __future__ import annotations

import pytest

from rag_service.chatterloop.platform.client import (
    PlatformAPIError,
    PlatformAuthError,
)
from rag_service.chatterloop.platform.responder import HttpResponder, build_send_body


class TestBuildSendBody:
    def test_threaded_reply_sets_both_fields(self):
        body = build_send_body("conv-1", "the answer", "group", reply_to_message_id="m2")
        assert body["isReply"] is True
        # A bare messageID string, not an object: the webapp renders the quoted
        # preview by matching `messageID == replyingTo`.
        assert body["replyingTo"] == "m2"

    def test_unthreaded_message_is_representable(self):
        body = build_send_body("conv-1", "hello", "single")
        assert body["isReply"] is False
        assert body["replyingTo"] == ""

    def test_carries_exactly_the_fields_the_route_reads(self):
        body = build_send_body("conv-1", "x", "group", reply_to_message_id="m1")
        assert set(body) == {
            "conversationID",
            "pendingID",
            "content",
            "isReply",
            "replyingTo",
            "messageType",
            "conversationType",
        }

    def test_receivers_are_never_sent(self):
        # The route derives them from the conversation rather than trusting the
        # sender, which is what stops a token addressing people not in it.
        assert "receivers" not in build_send_body("conv-1", "x", "group")

    def test_a_pending_id_is_generated_when_absent(self):
        a = build_send_body("conv-1", "x", "group")["pendingID"]
        b = build_send_body("conv-1", "x", "group")["pendingID"]
        assert a and b and a != b

    def test_an_explicit_pending_id_is_kept(self):
        assert build_send_body("c", "x", "group", pending_id="p9")["pendingID"] == "p9"

    def test_content_is_trimmed(self):
        assert build_send_body("c", "  hi  ", "group")["content"] == "hi"

    @pytest.mark.parametrize("bad", ["", "   ", "\n"])
    def test_empty_content_is_refused(self, bad):
        with pytest.raises(ValueError):
            build_send_body("conv-1", bad, "group")

    def test_conversation_id_is_required(self):
        with pytest.raises(ValueError):
            build_send_body("", "hi", "group")


class FakeClient:
    def __init__(self, error=None):
        self.error = error
        self.posts = []

    def post(self, path, body):
        self.posts.append((path, body))
        if self.error is not None:
            raise self.error
        return {"status": True}


class TestHttpResponder:
    def test_a_reply_posts_to_the_bot_write_endpoint(self):
        client = FakeClient()
        assert HttpResponder(client).reply_to_conversation("c1", "hi") is True
        [(path, body)] = client.posts
        assert path == "/v1/messages/send"
        assert body["conversationID"] == "c1"
        assert body["content"] == "hi"

    def test_threading_survives_the_round_trip(self):
        client = FakeClient()
        HttpResponder(client).reply_to_conversation("c1", "hi", reply_to_message_id="m7")
        assert client.posts[0][1]["replyingTo"] == "m7"
        assert client.posts[0][1]["isReply"] is True

    def test_the_per_call_conversation_type_wins_over_the_default(self):
        client = FakeClient()
        responder = HttpResponder(client, conversation_type="single")
        responder.reply_to_conversation("c1", "hi", conversation_type="group")
        # A group message sent as "single" gets the wrong push title, so the
        # caller's knowledge must beat the constructor's fallback.
        assert client.posts[0][1]["conversationType"] == "group"

    @pytest.mark.parametrize(
        "error", [PlatformAPIError("boom"), PlatformAuthError("403")]
    )
    def test_a_failed_send_returns_false_rather_than_raising(self, error):
        # One unanswered mention, not a dead consumer loop.
        client = FakeClient(error=error)
        assert HttpResponder(client).reply_to_conversation("c1", "hi") is False

    def test_an_empty_reply_is_refused_before_the_network(self):
        client = FakeClient()
        with pytest.raises(ValueError):
            HttpResponder(client).reply_to_conversation("c1", "   ")
        assert client.posts == []

    def test_commenting_is_still_unimplemented_and_says_why(self):
        # Unimplemented before the cut-over too, so nothing regressed - but the
        # reason changed from "a bot cannot authenticate" to "that endpoint
        # does not exist yet", and the message should reflect that.
        with pytest.raises(NotImplementedError) as excinfo:
            HttpResponder(FakeClient()).reply_to_comment("p1", "cm1", "hi")
        assert "comments" in str(excinfo.value)
