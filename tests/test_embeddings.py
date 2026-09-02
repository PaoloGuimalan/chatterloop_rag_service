from __future__ import annotations

import math

import pytest

from rag_service.chunking import HeuristicTokenizer
from rag_service.embeddings.cache import content_key
from rag_service.embeddings.openai_embedder import (
    EmbeddingError,
    OpenAIEmbedder,
    l2_normalise,
)


class FakeEmbeddingItem:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class FakeResponse:
    def __init__(self, data): self.data = data


class FakeEmbeddings:
    def __init__(self, owner): self._owner = owner

    def create(self, *, input, model, **kwargs):
        self._owner.calls.append({"input": list(input), "model": model, **kwargs})
        # Deliberately return items out of order to prove the embedder sorts.
        items = [FakeEmbeddingItem(i, [float(len(t)), 1.0, 0.0] + [0.0] * 1533)
                 for i, t in enumerate(input)]
        return FakeResponse(list(reversed(items)))


class FakeClient:
    def __init__(self):
        self.calls: list[dict] = []
        self.embeddings = FakeEmbeddings(self)


@pytest.fixture
def embedder(embedding_settings):
    emb = OpenAIEmbedder(embedding_settings, tokenizer=HeuristicTokenizer())
    fake = FakeClient()
    emb._clients[embedding_settings.api_key] = fake
    emb.fake = fake  # type: ignore[attr-defined]
    return emb


class TestNormalisation:
    def test_unit_length(self):
        out = l2_normalise([3.0, 4.0])
        assert math.isclose(math.sqrt(sum(v * v for v in out)), 1.0)

    def test_zero_vector_is_untouched(self):
        assert l2_normalise([0.0, 0.0]) == [0.0, 0.0]

    def test_embedder_returns_unit_vectors(self, embedder):
        for vec in embedder.embed_documents(["alpha", "beta"]):
            assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-6)


class TestOrdering:
    def test_results_realign_with_inputs(self, embedder):
        # The fake returns reversed data; index-sorting must undo that, or every
        # chunk silently gets its neighbour's vector.
        vectors = embedder.embed_documents(["a", "bbbb", "cc"])
        first_components = [v[0] for v in vectors]
        assert first_components[0] < first_components[1]
        assert first_components[2] < first_components[1]


class TestBatching:
    def test_respects_batch_size(self, embedder):
        embedder.embed_documents([f"text-{i}" for i in range(7)])
        # batch_size=3 -> 3 + 3 + 1
        assert [len(c["input"]) for c in embedder.fake.calls] == [3, 3, 1]

    def test_token_budget_forces_smaller_batches(self, embedding_settings):
        embedding_settings.max_tokens_per_batch = 20
        emb = OpenAIEmbedder(embedding_settings, tokenizer=HeuristicTokenizer())
        emb._clients[embedding_settings.api_key] = FakeClient()
        emb.embed_documents(["x" * 200, "y" * 200, "z" * 200])
        assert all(len(c["input"]) == 1 for c in emb._clients[embedding_settings.api_key].calls)

    def test_empty_input_makes_no_calls(self, embedder):
        assert embedder.embed_documents([]) == []
        assert embedder.fake.calls == []


class TestCache:
    def test_repeat_text_is_not_re_embedded(self, embedder):
        embedder.embed_documents(["hello", "world"])
        embedder.embed_documents(["hello", "world"])
        # Second round is served entirely from cache.
        assert len(embedder.fake.calls) == 1
        assert embedder.cache.hit_rate > 0

    def test_partial_hit_only_sends_the_misses(self, embedder):
        embedder.embed_documents(["hello"])
        embedder.fake.calls.clear()
        embedder.embed_documents(["hello", "fresh"])
        assert embedder.fake.calls[0]["input"] == ["fresh"]

    def test_key_includes_model_and_dim(self):
        assert content_key("t", "model-a", 1536) != content_key("t", "model-b", 1536)
        assert content_key("t", "model-a", 1536) != content_key("t", "model-a", 1024)


class TestDimensions:
    def test_matryoshka_dimensions_are_requested_only_when_truncating(self, embedding_settings):
        embedding_settings.dim = 512
        emb = OpenAIEmbedder(embedding_settings, tokenizer=HeuristicTokenizer())
        fake = FakeClient()
        emb._clients[embedding_settings.api_key] = fake
        with pytest.raises(EmbeddingError, match="dimension mismatch"):
            # The fake still returns 1536-d vectors, which must be caught rather
            # than written into a 512-d collection.
            emb.embed_documents(["x"])
        assert fake.calls[0]["dimensions"] == 512

    def test_native_dim_sends_no_dimensions_arg(self, embedder):
        embedder.embed_documents(["x"])
        assert "dimensions" not in embedder.fake.calls[0]

    def test_missing_key_is_an_error(self, embedding_settings):
        embedding_settings.api_key = ""
        emb = OpenAIEmbedder(embedding_settings, tokenizer=HeuristicTokenizer())
        with pytest.raises(EmbeddingError, match="no embedding API key"):
            emb.embed_documents(["x"])

    def test_per_tenant_key_selection(self, embedding_settings):
        embedding_settings.tenant_keys = {"org_vip": "vip-key"}
        assert embedding_settings.key_for("org_vip") == "vip-key"
        assert embedding_settings.key_for("org_other") == "test-key"
