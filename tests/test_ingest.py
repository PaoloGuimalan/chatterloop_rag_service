from __future__ import annotations

import pytest

from rag_service.chunking import HeuristicTokenizer, TokenChunker
from rag_service.domain import Role, Scope
from rag_service.pipeline.ingest import IngestionPipeline


class FakeStore:
    def __init__(self):
        self.upserts: list[tuple] = []
        self.deletes: list[str] = []

    def upsert(self, chunks, vectors):
        self.upserts.append((list(chunks), list(vectors)))
        return len(chunks)

    def delete(self, filter_expr):
        self.deletes.append(filter_expr)


class FakeEmbedder:
    dim = 4

    def __init__(self):
        self.seen: list[list[str]] = []

    def embed_documents(self, texts, tenant_id=""):
        self.seen.append(list(texts))
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text, tenant_id=""):
        return [1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def pipeline(chunking_settings):
    store, embedder = FakeStore(), FakeEmbedder()
    chunker = TokenChunker(HeuristicTokenizer(), max_tokens=50, overlap_tokens=10, min_tokens=5)
    pipe = IngestionPipeline(embedder, store, chunker, chunking_settings)
    pipe.store_spy, pipe.embedder_spy = store, embedder  # type: ignore[attr-defined]
    return pipe


class TestDocuments:
    def test_old_version_is_deleted_before_writing(self, pipeline):
        pipeline.ingest_document("org_1", "doc_1", "some body text", title="Policy")
        assert pipeline.store_spy.deletes == ['tenant_id == "org_1" and source_id == "doc_1"']
        assert len(pipeline.store_spy.upserts) == 1

    def test_title_is_prepended_for_embedding_but_not_stored(self, pipeline):
        pipeline.ingest_document("org_1", "doc_1", "Refunds take five days.", title="Refunds")
        embedded = pipeline.embedder_spy.seen[0][0]
        stored = pipeline.store_spy.upserts[0][0][0]
        assert embedded.startswith("Refunds\n\n")
        assert stored.text == "Refunds take five days."

    def test_chunks_carry_document_scope_and_no_role(self, pipeline):
        pipeline.ingest_document("org_1", "doc_1", "body")
        chunk = pipeline.store_spy.upserts[0][0][0]
        assert chunk.scope is Scope.DOCUMENT
        assert chunk.role is Role.NONE

    def test_empty_document_deletes_and_writes_nothing(self, pipeline):
        assert pipeline.ingest_document("org_1", "doc_1", "   ") == 0
        assert pipeline.store_spy.deletes
        assert pipeline.store_spy.upserts == []

    def test_long_document_produces_indexed_chunks(self, pipeline):
        written = pipeline.ingest_document("org_1", "doc_1", " ".join(f"w{i}" for i in range(400)))
        chunks = pipeline.store_spy.upserts[0][0]
        assert written == len(chunks) > 1
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_chunk_ids_are_stable_across_reingest(self, pipeline):
        pipeline.ingest_document("org_1", "doc_1", "stable body")
        first = [c.chunk_id for c in pipeline.store_spy.upserts[0][0]]
        pipeline.ingest_document("org_1", "doc_1", "stable body")
        second = [c.chunk_id for c in pipeline.store_spy.upserts[1][0]]
        assert first == second

    def test_different_tenants_never_share_chunk_ids(self, pipeline):
        pipeline.ingest_document("org_1", "doc_1", "same text")
        pipeline.ingest_document("org_2", "doc_1", "same text")
        a = pipeline.store_spy.upserts[0][0][0].chunk_id
        b = pipeline.store_spy.upserts[1][0][0].chunk_id
        assert a != b

    @pytest.mark.parametrize("tenant,doc", [("", "d"), ("org", "")])
    def test_required_identifiers(self, pipeline, tenant, doc):
        with pytest.raises(ValueError):
            pipeline.ingest_document(tenant, doc, "text")


class TestMessages:
    def test_short_message_is_one_chunk(self, pipeline):
        assert pipeline.index_message("org_1", "c1", "m1", "hello there") == 1

    def test_message_keeps_its_role_and_conversation(self, pipeline):
        pipeline.index_message("org_1", "c1", "m1", "hi", role=Role.ASSISTANT)
        chunk = pipeline.store_spy.upserts[0][0][0]
        assert chunk.role is Role.ASSISTANT
        assert chunk.conversation_id == "c1"
        assert chunk.scope is Scope.CHAT

    def test_title_is_not_prepended_to_chat(self, pipeline):
        pipeline.index_message("org_1", "c1", "m1", "hello")
        assert pipeline.embedder_spy.seen[0][0] == "hello"

    def test_blank_message_is_skipped_entirely(self, pipeline):
        assert pipeline.index_message("org_1", "c1", "m1", "   ") == 0
        assert pipeline.store_spy.upserts == []

    def test_very_long_message_is_chunked(self, pipeline):
        assert pipeline.index_message(
            "org_1", "c1", "m1", " ".join(f"w{i}" for i in range(400))
        ) > 1

    def test_indexing_a_message_does_not_delete(self, pipeline):
        # Messages are immutable; deleting by source_id first would be wasted
        # round trips on the hot path.
        pipeline.index_message("org_1", "c1", "m1", "hello")
        assert pipeline.store_spy.deletes == []


class TestDeletion:
    def test_delete_conversation_is_tenant_scoped(self, pipeline):
        pipeline.delete_conversation("org_1", "c1")
        assert pipeline.store_spy.deletes == [
            'tenant_id == "org_1" and conversation_id == "c1"'
        ]

    def test_delete_document_is_tenant_scoped(self, pipeline):
        pipeline.delete_document("org_1", "d1")
        assert 'tenant_id == "org_1"' in pipeline.store_spy.deletes[0]
