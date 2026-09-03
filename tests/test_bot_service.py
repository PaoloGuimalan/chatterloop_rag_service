"""build_generator: wiring an active agent's persona/model/tools into the
generator it builds - the one place config.ChatterloopSettings.agents
actually takes effect.
"""

from __future__ import annotations

from rag_service.bot_service import build_generator
from rag_service.chatterloop.replies import StubReplyGenerator, default_system_prompt
from rag_service.config import ChatterloopSettings, EmbeddingSettings, Settings


def _settings(**chatterloop_kwargs) -> Settings:
    chatterloop_kwargs.setdefault("enabled", True)
    chatterloop_kwargs.setdefault("bot_handle", "assistant")
    chatterloop_kwargs.setdefault("reply_generator", "openai")
    chatterloop_kwargs.setdefault("reply_api_key", "test-key")
    return Settings(
        chatterloop=ChatterloopSettings(**chatterloop_kwargs),
        embedding=EmbeddingSettings(api_key="fallback-key"),
    )


class TestNoAgentsConfigured:
    """Zero migration required: identical behaviour to before agents existed."""

    def test_persona_is_just_the_bot_default(self):
        settings = _settings(bot_handle="neon")
        gen = build_generator(settings)
        assert gen._system_prompt == default_system_prompt("neon")

    def test_model_is_the_flat_setting(self):
        settings = _settings(reply_model="gpt-4o")
        gen = build_generator(settings)
        assert gen._model == "gpt-4o"

    def test_tools_are_the_flat_registry_when_enabled(self):
        settings = _settings(
            tools_enabled=True,
            tools=[{"name": "weather", "api_endpoint": "https://x"}],
        )
        gen = build_generator(settings)
        assert set(gen._tools) == {"weather"}

    def test_tools_are_none_when_disabled(self):
        settings = _settings(
            tools_enabled=False,
            tools=[{"name": "weather", "api_endpoint": "https://x"}],
        )
        gen = build_generator(settings)
        assert gen._tools == {}
        assert gen._tool_schemas is None


class TestActiveAgent:
    def test_agent_system_prompt_is_appended_not_replaced(self):
        # The load-bearing claim: the built-in BACKGROUND-handling framing
        # survives even when an agent defines its own persona.
        settings = _settings(
            bot_handle="neon",
            agents=[{"id": "support", "system_prompt": "Be extra concise."}],
            active_agent="support",
        )
        gen = build_generator(settings)
        assert gen._system_prompt.startswith(default_system_prompt("neon"))
        assert "Be extra concise." in gen._system_prompt
        assert "BACKGROUND" in gen._system_prompt  # the guard, still present

    def test_agent_with_no_system_prompt_falls_back_to_the_default(self):
        settings = _settings(
            bot_handle="neon",
            agents=[{"id": "support"}],
            active_agent="support",
        )
        gen = build_generator(settings)
        assert gen._system_prompt == default_system_prompt("neon")

    def test_agent_model_override_wins(self):
        settings = _settings(
            reply_model="gpt-4o-mini",
            agents=[{"id": "support", "model": "gpt-4o"}],
            active_agent="support",
        )
        gen = build_generator(settings)
        assert gen._model == "gpt-4o"

    def test_agent_without_a_model_override_falls_back_to_the_flat_setting(self):
        settings = _settings(
            reply_model="gpt-4o-mini",
            agents=[{"id": "support"}],
            active_agent="support",
        )
        gen = build_generator(settings)
        assert gen._model == "gpt-4o-mini"

    def test_agent_tool_ids_narrow_the_registry(self):
        settings = _settings(
            tools_enabled=True,
            tools=[
                {"name": "weather", "api_endpoint": "https://x"},
                {"name": "billing", "api_endpoint": "https://y"},
            ],
            agents=[{"id": "support", "tool_ids": ["billing"]}],
            active_agent="support",
        )
        gen = build_generator(settings)
        assert set(gen._tools) == {"billing"}

    def test_tools_enabled_false_wins_even_with_an_active_agent(self):
        settings = _settings(
            tools_enabled=False,
            tools=[{"name": "weather", "api_endpoint": "https://x"}],
            agents=[{"id": "support", "tool_ids": ["weather"]}],
            active_agent="support",
        )
        gen = build_generator(settings)
        assert gen._tools == {}
        assert gen._tool_schemas is None

    def test_agent_with_no_tool_ids_gets_none_of_the_registry(self):
        # Explicit narrowing to nothing, not "inherit everything" - an agent
        # that names no tools should not accidentally get the whole registry.
        settings = _settings(
            tools_enabled=True,
            tools=[{"name": "weather", "api_endpoint": "https://x"}],
            agents=[{"id": "support"}],
            active_agent="support",
        )
        gen = build_generator(settings)
        assert gen._tools == {}


class TestStubUnaffected:
    def test_stub_ignores_agents_entirely(self):
        settings = _settings(
            reply_generator="stub",
            agents=[{"id": "support"}],
            active_agent="support",
        )
        assert isinstance(build_generator(settings), StubReplyGenerator)
