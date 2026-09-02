from __future__ import annotations

import pytest

from rag_service.domain import Role
from rag_service.messaging.events import EventType, InvalidEvent, parse_event


def envelope(event_type: str, payload: dict, **kw) -> dict:
    return {"event_type": event_type, "tenant_id": "org_1", "payload": payload, **kw}


class TestEnvelope:
    def test_event_id_is_generated_when_absent(self):
        env, _ = parse_event(envelope("document.delete", {"document_id": "d1"}))
        assert env.event_id

    def test_tenant_id_is_mandatory(self):
        with pytest.raises(InvalidEvent):
            parse_event({"event_type": "document.delete", "payload": {"document_id": "d"}})

    def test_unknown_event_type_is_permanent(self):
        with pytest.raises(InvalidEvent):
            parse_event(envelope("document.explode", {}))

    def test_unknown_top_level_field_is_rejected(self):
        # extra="forbid" catches producers drifting from the contract.
        with pytest.raises(InvalidEvent):
            parse_event(envelope("document.delete", {"document_id": "d"}, surprise=1))


class TestMessageIndex:
    @pytest.mark.parametrize(
        "incoming,expected",
        [
            ("text", Role.USER),
            ("reply", Role.USER),
            ("ai_reply", Role.ASSISTANT),
            ("agent", Role.ASSISTANT),
            ("user", Role.USER),
            ("assistant", Role.ASSISTANT),
        ],
    )
    def test_neon_message_types_are_translated(self, incoming, expected):
        _, payload = parse_event(
            envelope(
                "message.index",
                {"conversation_id": "c1", "message_id": "m1", "text": "hi", "role": incoming},
            )
        )
        assert payload.role is expected

    def test_missing_conversation_is_rejected(self):
        with pytest.raises(InvalidEvent):
            parse_event(envelope("message.index", {"message_id": "m1", "text": "hi"}))

    def test_unknown_role_is_rejected(self):
        with pytest.raises(InvalidEvent):
            parse_event(
                envelope(
                    "message.index",
                    {"conversation_id": "c", "message_id": "m", "text": "x", "role": "wizard"},
                )
            )


class TestRetrievalRequest:
    def test_minimal_request(self):
        env, payload = parse_event(envelope("retrieval.request", {"query": "where is my order"}))
        assert env.event_type is EventType.RETRIEVAL_REQUEST
        assert payload.top_k is None
        assert payload.include_recent_history is True

    def test_empty_query_is_rejected(self):
        with pytest.raises(InvalidEvent):
            parse_event(envelope("retrieval.request", {"query": ""}))

    def test_top_k_upper_bound(self):
        with pytest.raises(InvalidEvent):
            parse_event(envelope("retrieval.request", {"query": "x", "top_k": 5000}))

    def test_unknown_payload_fields_are_ignored(self):
        # Payloads are lenient so producers can add fields without a lockstep
        # deploy; envelopes are strict because they are the routing contract.
        _, payload = parse_event(
            envelope("retrieval.request", {"query": "x", "future_field": True})
        )
        assert payload.query == "x"


class TestDocumentIngest:
    def test_defaults(self):
        _, payload = parse_event(
            envelope("document.ingest", {"document_id": "d1", "text": "body"})
        )
        assert payload.title == ""
        assert payload.meta == {}

    def test_document_id_required(self):
        with pytest.raises(InvalidEvent):
            parse_event(envelope("document.ingest", {"text": "body"}))
