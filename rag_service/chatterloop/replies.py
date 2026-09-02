"""Turning retrieved context into something to say."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from ..domain import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a participant in a group chat, not a support agent. "
    "Someone has mentioned you by name and is asking you something. "
    "Answer using the conversation context provided; it is the record of what "
    "was actually said. If the context does not contain the answer, say so "
    "plainly in one sentence rather than guessing. "
    "Keep replies short - chat length, not essay length. Do not greet, do not "
    "sign off, and do not restate the question."
)


@runtime_checkable
class ReplyGenerator(Protocol):
    def generate(self, question: str, context: RetrievalResult) -> str: ...


class OpenAIReplyGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 400,
        temperature: float = 0.3,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("an API key is required to generate replies")
        self._api_key = api_key
        self._model = model
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, timeout=self._timeout)
        return self._client

    def generate(self, question: str, context: RetrievalResult) -> str:
        messages = [{"role": "system", "content": self._system_prompt}]
        # Retrieved chunks already render themselves as chat messages, with
        # chat turns keeping their original speaker - so the model can tell
        # "the customer said" from "we replied" rather than reading one
        # undifferentiated context blob.
        messages.extend(context.as_messages())
        messages.append({"role": "user", "content": question})

        response = self._get_client().chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return (response.choices[0].message.content or "").strip()


class StubReplyGenerator:
    """Reports what it retrieved instead of composing prose.

    Useful before an LLM key is wired up, and genuinely useful afterwards: it
    makes retrieval quality visible on its own, separately from generation
    quality, which is the harder of the two to debug once they are mixed.
    """

    def generate(self, question: str, context: RetrievalResult) -> str:
        if not context.chunks:
            return "I don't have any context for that yet."
        lines = [f"[retrieval-only] {len(context.chunks)} passage(s) for {question!r}:"]
        for chunk in context.chunks[:3]:
            snippet = chunk.text.replace("\n", " ")[:120]
            lines.append(f"  - ({chunk.scope}) {snippet}")
        return "\n".join(lines)
