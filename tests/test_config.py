"""Configuration parsing.

The regression that motivated this file: `.env.example` documents several
optional settings as blank, and a blank value was being parsed as the empty
string rather than as "unset" - so a stock .env crashed the worker at startup
on `EMBEDDING_DIM=`.
"""

from __future__ import annotations

import pytest

from rag_service.config import EmbeddingSettings, MilvusSettings, RetrievalSettings


class TestEmptyEnvValues:
    def test_blank_optional_int_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_DIM", "")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
        assert EmbeddingSettings().dim == 1536

    def test_blank_optional_string_falls_back(self, monkeypatch):
        monkeypatch.setenv("MILVUS_TOKEN", "")
        assert MilvusSettings().token == ""

    def test_blank_rerank_key_is_tolerated(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_RERANK_API_KEY", "")
        assert RetrievalSettings().rerank_api_key == ""


class TestDimensionResolution:
    def test_native_dim_is_inferred_from_the_model(self):
        assert EmbeddingSettings(model="text-embedding-3-large").dim == 3072

    def test_explicit_dim_enables_matryoshka_truncation(self):
        assert EmbeddingSettings(model="text-embedding-3-large", dim=1024).dim == 1024

    def test_unknown_model_must_declare_its_dimension(self):
        with pytest.raises(ValueError, match="must be set explicitly"):
            EmbeddingSettings(model="some-private-model")

    def test_unknown_model_with_explicit_dim_is_fine(self):
        assert EmbeddingSettings(model="some-private-model", dim=768).dim == 768


class TestTenantKeys:
    def test_json_string_is_parsed(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_TENANT_KEYS", '{"org_a": "sk-a"}')
        assert EmbeddingSettings().key_for("org_a") == "sk-a"

    def test_blank_json_is_an_empty_map(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_TENANT_KEYS", "")
        assert EmbeddingSettings(api_key="fallback").key_for("anyone") == "fallback"
