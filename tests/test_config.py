"""Configuration parsing.

The regression that motivated this file: `.env.example` documents several
optional settings as blank, and a blank value was being parsed as the empty
string rather than as "unset" - so a stock .env crashed the worker at startup
on `EMBEDDING_DIM=`.
"""

from __future__ import annotations

import pytest

from rag_service.config import (
    ChatterloopSettings,
    EmbeddingSettings,
    MilvusSettings,
    RetrievalSettings,
)


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


class TestAgents:
    """CHATTERLOOP_AGENTS / CHATTERLOOP_ACTIVE_AGENT: one persona, chosen at
    deploy time, out of however many are defined."""

    def test_no_agents_is_the_default_and_needs_no_active_agent(self):
        settings = ChatterloopSettings()
        assert settings.agents == []
        assert settings.active_agent_config is None

    def test_active_agent_without_any_agents_configured_is_rejected(self):
        with pytest.raises(ValueError, match="CHATTERLOOP_AGENTS is empty"):
            ChatterloopSettings(active_agent="support")

    def test_agents_configured_but_no_active_agent_chosen_is_rejected(self):
        with pytest.raises(ValueError, match="CHATTERLOOP_ACTIVE_AGENT is not set"):
            ChatterloopSettings(agents=[{"id": "support"}])

    def test_active_agent_naming_an_unknown_id_is_rejected(self):
        with pytest.raises(ValueError, match="not an enabled agent id"):
            ChatterloopSettings(agents=[{"id": "support"}], active_agent="sales")

    def test_active_agent_naming_a_disabled_agent_is_rejected(self):
        with pytest.raises(ValueError, match="not an enabled agent id"):
            ChatterloopSettings(
                agents=[{"id": "support", "is_enabled": False}],
                active_agent="support",
            )

    def test_valid_active_agent_resolves(self):
        settings = ChatterloopSettings(
            agents=[
                {"id": "support", "system_prompt": "Be helpful."},
                {"id": "sales"},
            ],
            active_agent="support",
        )
        assert settings.active_agent_config.id == "support"
        assert settings.active_agent_config.system_prompt == "Be helpful."

    def test_agent_referencing_an_unknown_tool_is_rejected(self):
        with pytest.raises(ValueError, match="unknown tool id"):
            ChatterloopSettings(
                tools=[{"name": "weather", "api_endpoint": "https://x"}],
                agents=[{"id": "support", "tool_ids": ["not_a_real_tool"]}],
                active_agent="support",
            )

    def test_agent_referencing_a_known_tool_is_accepted(self):
        settings = ChatterloopSettings(
            tools=[{"name": "weather", "api_endpoint": "https://x"}],
            agents=[{"id": "support", "tool_ids": ["weather"]}],
            active_agent="support",
        )
        assert settings.active_agent_config.tool_ids == ["weather"]

    def test_agents_parse_from_json_env(self, monkeypatch):
        monkeypatch.setenv(
            "CHATTERLOOP_AGENTS",
            '[{"id": "support", "name": "Support"}]',
        )
        monkeypatch.setenv("CHATTERLOOP_ACTIVE_AGENT", "support")

        settings = ChatterloopSettings()

        assert len(settings.agents) == 1
        assert settings.agents[0].name == "Support"
        assert settings.active_agent_config.id == "support"

    def test_disabled_agents_do_not_block_startup_validation(self):
        # A retired persona left in the roster (is_enabled=False) is not
        # a live candidate, but its own bad tool reference should still be
        # caught - nothing here is exempt from validation just for being off.
        with pytest.raises(ValueError, match="unknown tool id"):
            ChatterloopSettings(
                agents=[
                    {"id": "old", "is_enabled": False, "tool_ids": ["ghost"]},
                    {"id": "support"},
                ],
                active_agent="support",
            )
