"""Reply generation: retry, tool-calling, and the provider factory.

Exercises the shared orchestration (ChatCompletionReplyGenerator) once
through OpenAIReplyGenerator, and separately proves GroqReplyGenerator /
build_reply_generator select a DIFFERENT SDK client and DIFFERENT exception
types - the actual claim being tested is "one vendor swap, zero changes to
the completion/tool-loop/retry logic", not "the loop works" twice over.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIConnectionError, AuthenticationError, RateLimitError

from rag_service.chatterloop.replies import (
    GROQ_PROVIDER,
    OPENAI_PROVIDER,
    PROVIDERS,
    GroqReplyGenerator,
    OpenAIReplyGenerator,
    StubReplyGenerator,
    build_reply_generator,
)
from rag_service.config import ToolConfig
from rag_service.domain import RetrievalResult, RetrievedChunk, Role, Scope


def _context(chunks=()):
    return RetrievalResult(query="hi", tenant_id="t1", conversation_id="c1", chunks=list(chunks))


def _api_error(cls, message="boom"):
    """Builds a real openai/groq exception without a live HTTP call.

    APIConnectionError takes a bare httpx.Request (there was never a
    response); the status-coded errors take an httpx.Response, which itself
    carries the request that produced it.
    """
    request = httpx.Request("POST", "https://api.example/v1/chat/completions")
    if cls is RateLimitError:
        response = httpx.Response(429, request=request, json={"error": {"message": message}})
        return cls(message, response=response, body=None)
    if cls is AuthenticationError:
        response = httpx.Response(401, request=request, json={"error": {"message": message}})
        return cls(message, response=response, body=None)
    return cls(message=message, request=request)


def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _response(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=_message(content, tool_calls))])


def _tool_call(call_id, name, arguments):
    call = MagicMock()
    call.id = call_id
    call.function.name = name
    call.function.arguments = arguments
    call.model_dump.return_value = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    return call


@pytest.fixture
def generator(monkeypatch):
    gen = OpenAIReplyGenerator(api_key="test-key", max_retries=3)
    fake_client = MagicMock()
    monkeypatch.setattr(gen, "_get_client", lambda: fake_client)
    return gen, fake_client


class TestNoTools:
    def test_plain_reply(self, generator):
        gen, client = generator
        client.chat.completions.create.return_value = _response(content=" hi there ")

        reply = gen.generate("hello", _context())

        assert reply == "hi there"
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    def test_background_is_fenced(self, generator):
        gen, client = generator
        client.chat.completions.create.return_value = _response(content="ok")
        chunk = RetrievedChunk("c1", "we agreed on tiered pricing", Scope.CHAT, 0.9, role=Role.USER)

        gen.generate("what did we agree on?", _context([chunk]))

        messages = client.chat.completions.create.call_args.kwargs["messages"]
        contents = [m["content"] for m in messages]
        assert "--- BACKGROUND (retrieved, may be irrelevant) ---" in contents
        assert "we agreed on tiered pricing" in contents
        assert "--- END BACKGROUND. Now reply to this message: ---" in contents
        assert messages[-1] == {"role": "user", "content": "what did we agree on?"}


class TestRetry:
    def test_transient_error_is_retried_then_succeeds(self, generator):
        gen, client = generator
        client.chat.completions.create.side_effect = [
            _api_error(APIConnectionError),
            _response(content="recovered"),
        ]

        reply = gen.generate("hello", _context())

        assert reply == "recovered"
        assert client.chat.completions.create.call_count == 2

    def test_exhausted_retries_returns_empty_not_raise(self, generator):
        gen, client = generator
        client.chat.completions.create.side_effect = _api_error(RateLimitError)

        reply = gen.generate("hello", _context())

        assert reply == ""
        assert client.chat.completions.create.call_count == 3  # max_retries=3

    def test_permanent_error_is_not_retried(self, generator):
        gen, client = generator
        client.chat.completions.create.side_effect = _api_error(AuthenticationError)

        reply = gen.generate("hello", _context())

        assert reply == ""
        assert client.chat.completions.create.call_count == 1


class TestToolCalling:
    @pytest.fixture
    def tool(self):
        return ToolConfig(
            name="get_weather",
            description="Current weather for a city.",
            parameters_schema={"type": "object", "properties": {"city": {"type": "string"}}},
            api_endpoint="https://api.example/weather",
            http_method="GET",
            param_type="query",
        )

    @pytest.fixture
    def generator_with_tool(self, monkeypatch, tool):
        gen = OpenAIReplyGenerator(api_key="test-key", tools=[tool], tool_max_iterations=2)
        fake_client = MagicMock()
        monkeypatch.setattr(gen, "_get_client", lambda: fake_client)
        return gen, fake_client

    def test_no_tools_configured_sends_no_tools_field(self, generator):
        gen, _client = generator
        assert gen._tool_schemas is None

    def test_tools_are_offered_when_configured(self, generator_with_tool):
        gen, client = generator_with_tool
        client.chat.completions.create.return_value = _response(content="sunny")

        gen.generate("weather in Manila?", _context())

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["tool_choice"] == "auto"
        assert kwargs["tools"][0]["function"]["name"] == "get_weather"

    def test_tool_call_is_executed_and_result_fed_back(self, monkeypatch, generator_with_tool):
        gen, client = generator_with_tool
        call = _tool_call("call_1", "get_weather", json.dumps({"city": "Manila"}))
        client.chat.completions.create.side_effect = [
            _response(tool_calls=[call]),
            _response(content="It's sunny in Manila."),
        ]
        monkeypatch.setattr(
            "rag_service.chatterloop.replies.call_tool",
            lambda tool, args, timeout: {"temp_c": 31, "condition": "sunny"},
        )

        reply = gen.generate("weather in Manila?", _context())

        assert reply == "It's sunny in Manila."
        assert client.chat.completions.create.call_count == 2
        second_call_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
        tool_message = next(m for m in second_call_messages if m.get("role") == "tool")
        assert tool_message["tool_call_id"] == "call_1"
        assert json.loads(tool_message["content"]) == {"temp_c": 31, "condition": "sunny"}

    def test_unknown_tool_name_is_reported_not_raised(self, generator_with_tool):
        gen, client = generator_with_tool
        call = _tool_call("call_1", "not_a_real_tool", "{}")
        client.chat.completions.create.side_effect = [
            _response(tool_calls=[call]),
            _response(content="couldn't find that"),
        ]

        reply = gen.generate("do something", _context())

        assert reply == "couldn't find that"
        second_call_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
        tool_message = next(m for m in second_call_messages if m.get("role") == "tool")
        assert "no such tool" in json.loads(tool_message["content"])["error"]

    def test_malformed_arguments_do_not_raise(self, generator_with_tool):
        gen, client = generator_with_tool
        call = _tool_call("call_1", "get_weather", "{not json")
        client.chat.completions.create.side_effect = [
            _response(tool_calls=[call]),
            _response(content="ok"),
        ]

        reply = gen.generate("weather?", _context())

        assert reply == "ok"  # did not raise on the bad JSON arguments

    def test_iteration_cap_forces_a_final_toolless_call(self, generator_with_tool):
        gen, client = generator_with_tool
        call = _tool_call("call_1", "get_weather", json.dumps({"city": "Manila"}))
        client.chat.completions.create.side_effect = [
            _response(tool_calls=[call]),  # iteration 1
            _response(tool_calls=[call]),  # iteration 2 (cap = 2)
            _response(content="giving up gracefully"),  # forced final call
        ]

        reply = gen.generate("weather?", _context())

        assert reply == "giving up gracefully"
        assert client.chat.completions.create.call_count == 3
        final_kwargs = client.chat.completions.create.call_args_list[2].kwargs
        assert "tools" not in final_kwargs


class TestProviderFactory:
    def test_build_reply_generator_openai(self):
        # The factory returns the SAME class regardless of vendor - that is
        # the actual "one config value, no code change" claim. What has to
        # differ is which provider it was built with.
        gen = build_reply_generator("openai", api_key="k")
        assert gen._provider is OPENAI_PROVIDER
        assert gen.generate  # duck-typed ReplyGenerator either way

    def test_build_reply_generator_groq(self):
        gen = build_reply_generator("groq", api_key="k")
        assert gen._provider is GROQ_PROVIDER

    def test_build_reply_generator_is_case_insensitive(self):
        gen = build_reply_generator("OpenAI", api_key="k")
        assert gen._provider is OPENAI_PROVIDER

    def test_named_subclasses_pin_their_provider(self):
        # OpenAIReplyGenerator/GroqReplyGenerator exist for direct
        # construction at a call site that already knows the vendor - not
        # used by the factory itself, but still expected to select the
        # right ChatProvider.
        assert OpenAIReplyGenerator(api_key="k")._provider is OPENAI_PROVIDER
        assert GroqReplyGenerator(api_key="k")._provider is GROQ_PROVIDER

    def test_unsupported_service_raises(self):
        with pytest.raises(ValueError, match="unsupported reply generator service"):
            build_reply_generator("anthropic", api_key="k")

    def test_openai_and_groq_build_different_sdk_clients(self):
        # groq is an OPTIONAL extra (pyproject.toml), exactly like cohere and
        # sentence-transformers - a plain `make install` (`.[dev]`) does not
        # pull it in, and this test's whole point is to actually construct a
        # real groq.Groq client. Skip cleanly rather than fail a suite run
        # that never asked for the groq provider.
        pytest.importorskip("groq")
        openai_gen = OpenAIReplyGenerator(api_key="k")
        groq_gen = GroqReplyGenerator(api_key="k")

        openai_client = openai_gen._get_client()
        groq_client = groq_gen._get_client()

        assert type(openai_client).__module__.startswith("openai")
        assert type(groq_client).__module__.startswith("groq")

    def test_openai_and_groq_use_their_own_exception_types(self):
        groq_sdk = pytest.importorskip("groq")
        import openai as openai_sdk

        assert OPENAI_PROVIDER.transient_exceptions()[0].__module__.startswith("openai")
        assert GROQ_PROVIDER.transient_exceptions()[0].__module__.startswith("groq")
        # And the two are genuinely distinct classes, not one shared base
        # silently making the "per-vendor" claim meaningless.
        assert OPENAI_PROVIDER.transient_exceptions() != GROQ_PROVIDER.transient_exceptions()
        assert openai_sdk.APIConnectionError is not groq_sdk.APIConnectionError

    def test_adding_a_provider_requires_no_generator_change(self):
        # The whole point of the registry: PROVIDERS is the single place a
        # new vendor gets added, and everything that dispatches on it
        # (build_reply_generator, bot_service.build_generator) reads this
        # dict rather than naming vendors individually.
        assert set(PROVIDERS) == {"openai", "groq"}
        assert PROVIDERS["openai"] is OPENAI_PROVIDER
        assert PROVIDERS["groq"] is GROQ_PROVIDER


def test_stub_generator_is_unaffected():
    """The stub path shares no code with the provider machinery above."""
    stub = StubReplyGenerator()
    reply = stub.generate("hello", _context())
    assert "context" in reply.lower()
