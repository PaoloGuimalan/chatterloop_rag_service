"""The tenant-isolation guarantee lives in these tests."""

from __future__ import annotations

import pytest

from rag_service.domain import Scope
from rag_service.store.filters import FilterError, build_filter, delete_filter, quote


class TestQuoting:
    def test_plain_value(self):
        assert quote("org_123") == '"org_123"'

    def test_embedded_quote_is_escaped(self):
        assert quote('a"b') == '"a\\"b"'

    def test_backslash_is_escaped_before_quote(self):
        # Order matters: escaping the quote first would leave the backslash
        # free to escape our escape.
        assert quote("a\\b") == '"a\\\\b"'

    def test_injection_attempt_stays_inside_the_literal(self):
        hostile = 'x" or tenant_id != "'
        expr = build_filter(hostile, [Scope.DOCUMENT])

        # The hostile quote survives only in escaped form, so it cannot close
        # the literal and append a predicate.
        assert '\\"' in expr
        # Removing the escape sequences leaves exactly the four real delimiters:
        # two around the tenant value, two around the scope value. Any breakout
        # would show up as extra unescaped quotes.
        assert expr.replace('\\"', "").count('"') == 4
        # And the predicate structure is exactly what we built.
        assert expr.count("tenant_id ==") == 1
        assert expr.count(" and ") == 1

    def test_null_byte_rejected(self):
        with pytest.raises(FilterError):
            quote("a\x00b")


class TestBuildFilter:
    def test_tenant_is_always_the_first_conjunct(self):
        expr = build_filter("org_1", [Scope.DOCUMENT])
        assert expr.startswith('tenant_id == "org_1"')

    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_missing_tenant_is_refused(self, bad):
        with pytest.raises(FilterError, match="tenant_id is required"):
            build_filter(bad, [Scope.DOCUMENT])  # type: ignore[arg-type]

    def test_chat_scope_requires_a_conversation(self):
        # Chat history must never be tenant-wide readable.
        with pytest.raises(FilterError, match="match nothing"):
            build_filter("org_1", [Scope.CHAT])

    def test_chat_scope_is_pinned_to_one_conversation(self):
        expr = build_filter("org_1", [Scope.CHAT], conversation_id="conv_9")
        assert 'conversation_id == "conv_9"' in expr
        assert 'scope == "chat"' in expr

    def test_both_scopes_are_disjoined(self):
        expr = build_filter("org_1", [Scope.DOCUMENT, Scope.CHAT], conversation_id="conv_9")
        assert expr.startswith('tenant_id == "org_1" and (')
        assert " or " in expr

    def test_documents_only_when_no_conversation_given(self):
        expr = build_filter("org_1", [Scope.DOCUMENT, Scope.CHAT])
        assert 'scope == "doc"' in expr
        assert "conversation_id" not in expr

    def test_source_id_restriction(self):
        expr = build_filter("org_1", [Scope.DOCUMENT], source_ids=["a", "b"])
        assert 'source_id in ["a", "b"]' in expr

    def test_empty_scope_list_is_refused(self):
        with pytest.raises(FilterError):
            build_filter("org_1", [])


class TestDeleteFilter:
    def test_refuses_to_wipe_a_whole_tenant(self):
        with pytest.raises(FilterError, match="refusing to delete"):
            delete_filter("org_1")

    def test_by_source(self):
        assert delete_filter("org_1", source_id="doc_1") == (
            'tenant_id == "org_1" and source_id == "doc_1"'
        )

    def test_by_conversation(self):
        assert delete_filter("org_1", conversation_id="c_1") == (
            'tenant_id == "org_1" and conversation_id == "c_1"'
        )

    def test_tenant_still_required(self):
        with pytest.raises(FilterError):
            delete_filter("", source_id="doc_1")
