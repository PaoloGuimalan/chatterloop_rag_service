"""Live Milvus integration tests.

NOT RUN as part of the unit suite, and NOT YET EXECUTED by the author - the
development machine ran out of disk before the Milvus image could be pulled.
Treat these as a checklist to run first, not as passing tests.

    make up                        # milvus + redis
    MILVUS_LIVE=1 pytest tests/test_milvus_integration.py -v

They exercise the parts unit tests cannot: that the schema is accepted, that the
BM25 Function actually populates the sparse field, that hybrid search fuses two
legs, and - most importantly - that the partition key really isolates tenants.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid

import pytest

from rag_service.config import MilvusSettings, RetrievalSettings
from rag_service.domain import Chunk, Role, Scope
from rag_service.store import MilvusStore, build_filter, delete_filter

pytestmark = pytest.mark.skipif(
    os.getenv("MILVUS_LIVE") != "1", reason="set MILVUS_LIVE=1 with Milvus running"
)

DIM = 64


def fake_vector(text: str) -> list[float]:
    """Deterministic pseudo-embedding.

    Lets the integration suite run with no OpenAI key. Similar strings do not
    produce similar vectors, so these tests assert on plumbing (does a hit come
    back, is it the right tenant's) rather than on semantic ranking quality.
    """
    digest = hashlib.blake2b(text.encode(), digest_size=DIM * 2).digest()
    raw = [(digest[i] - 128) / 128.0 for i in range(DIM)]
    norm = sum(v * v for v in raw) ** 0.5 or 1.0
    return [v / norm for v in raw]


@pytest.fixture(scope="module")
def store():
    settings = MilvusSettings(
        uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
        collection=f"it_rag_{uuid.uuid4().hex[:8]}",
    )
    store = MilvusStore(settings, dim=DIM)
    store.ensure_collection()
    yield store
    try:
        store.client.drop_collection(settings.collection)
    finally:
        store.close()


@pytest.fixture(scope="module")
def seeded(store):
    chunks = [
        Chunk(tenant_id="org_a", scope=Scope.DOCUMENT, source_id="doc_a1", chunk_index=0,
              title="Refund Policy",
              text="Refunds are issued within five business days of approval."),
        Chunk(tenant_id="org_a", scope=Scope.DOCUMENT, source_id="doc_a2", chunk_index=0,
              title="Shipping", text="Order SKU-99213 ships from the Manila warehouse."),
        Chunk(tenant_id="org_a", scope=Scope.CHAT, source_id="msg_1", chunk_index=0,
              conversation_id="conv_1", role=Role.USER,
              text="Where is my order SKU-99213?"),
        Chunk(tenant_id="org_a", scope=Scope.CHAT, source_id="msg_2", chunk_index=0,
              conversation_id="conv_1", role=Role.ASSISTANT,
              text="It left the Manila warehouse yesterday."),
        # A different tenant with deliberately similar text.
        Chunk(tenant_id="org_b", scope=Scope.DOCUMENT, source_id="doc_b1", chunk_index=0,
              title="Refund Policy", text="Refunds are issued within five business days."),
    ]
    store.upsert(chunks, [fake_vector(c.text) for c in chunks])
    # Bounded consistency: give the write a moment to become visible.
    time.sleep(2)
    return chunks


class TestSchema:
    def test_collection_exists_with_both_vector_fields(self, store):
        desc = store.client.describe_collection(store.settings.collection)
        names = {f["name"] for f in desc["fields"]}
        assert {"dense", "sparse", "text", "tenant_id"} <= names

    def test_tenant_id_is_the_partition_key(self, store):
        desc = store.client.describe_collection(store.settings.collection)
        field = next(f for f in desc["fields"] if f["name"] == "tenant_id")
        assert field.get("is_partition_key") is True

    def test_bm25_function_is_registered(self, store):
        desc = store.client.describe_collection(store.settings.collection)
        functions = desc.get("functions", [])
        assert any(f.get("name") == "bm25_text_to_sparse" for f in functions)


class TestTenantIsolation:
    def test_search_never_crosses_tenants(self, store, seeded):
        results = store.hybrid_search(
            query_text="refunds business days",
            query_vector=fake_vector("refunds business days"),
            filter_expr=build_filter("org_a", [Scope.DOCUMENT]),
            limit=10,
            retrieval=RetrievalSettings(),
        )
        assert results
        assert all(r.source_id != "doc_b1" for r in results)

    def test_other_tenant_sees_only_its_own(self, store, seeded):
        results = store.hybrid_search(
            query_text="refunds business days",
            query_vector=fake_vector("refunds business days"),
            filter_expr=build_filter("org_b", [Scope.DOCUMENT]),
            limit=10,
            retrieval=RetrievalSettings(),
        )
        assert {r.source_id for r in results} == {"doc_b1"}


class TestHybridSearch:
    def test_lexical_leg_finds_an_exact_token(self, store, seeded):
        # The whole argument for hybrid: a random-vector "embedding" cannot
        # match this, so a hit proves BM25 is contributing.
        results = store.hybrid_search(
            query_text="SKU-99213",
            query_vector=fake_vector("completely unrelated text"),
            filter_expr=build_filter("org_a", [Scope.DOCUMENT]),
            limit=5,
            retrieval=RetrievalSettings(),
        )
        assert any("SKU-99213" in r.text for r in results)

    def test_chat_scope_is_pinned_to_the_conversation(self, store, seeded):
        results = store.hybrid_search(
            query_text="warehouse",
            query_vector=fake_vector("warehouse"),
            filter_expr=build_filter("org_a", [Scope.CHAT], conversation_id="conv_1"),
            limit=5,
            retrieval=RetrievalSettings(),
        )
        assert results
        assert all(r.conversation_id == "conv_1" for r in results)

    def test_roles_survive_the_round_trip(self, store, seeded):
        results = store.hybrid_search(
            query_text="warehouse yesterday",
            query_vector=fake_vector("warehouse yesterday"),
            filter_expr=build_filter("org_a", [Scope.CHAT], conversation_id="conv_1"),
            limit=5,
            retrieval=RetrievalSettings(),
        )
        assert {r.role for r in results} <= {Role.USER, Role.ASSISTANT}

    def test_vectors_are_returned_when_requested(self, store, seeded):
        results = store.hybrid_search(
            query_text="refunds",
            query_vector=fake_vector("refunds"),
            filter_expr=build_filter("org_a", [Scope.DOCUMENT]),
            limit=3,
            retrieval=RetrievalSettings(),
            with_vectors=True,
        )
        assert all(r.dense is not None and len(r.dense) == DIM for r in results)


class TestWrites:
    def test_upsert_replaces_rather_than_duplicates(self, store):
        chunk = Chunk(tenant_id="org_c", scope=Scope.DOCUMENT, source_id="doc_c1",
                      chunk_index=0, text="first version")
        store.upsert([chunk], [fake_vector("first")])
        revised = Chunk(tenant_id="org_c", scope=Scope.DOCUMENT, source_id="doc_c1",
                        chunk_index=0, text="second version")
        store.upsert([revised], [fake_vector("second")])
        time.sleep(2)
        assert store.count(build_filter("org_c", [Scope.DOCUMENT])) == 1

    def test_delete_by_source_is_scoped(self, store, seeded):
        store.delete(delete_filter("org_a", source_id="doc_a1"))
        time.sleep(2)
        remaining = store.count(build_filter("org_a", [Scope.DOCUMENT]))
        assert remaining >= 1  # doc_a2 survives
        assert store.count(build_filter("org_b", [Scope.DOCUMENT])) == 1


class TestDimensionGuard:
    def test_changing_dimension_fails_loudly(self, store):
        mismatched = MilvusStore(store.settings, dim=DIM + 8)
        with pytest.raises(RuntimeError, match="full reindex"):
            mismatched.ensure_collection()
