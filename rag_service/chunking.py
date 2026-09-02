"""Token-aware recursive chunking.

Measuring chunks in characters (as the previous Pinecone pipeline did) is a
proxy for the thing that actually matters, and a bad one: 2000 characters of
dense prose is ~500 tokens, but 2000 characters of JSON or code is closer to
900. Chunks then silently overflow the reranker's input window and get
truncated mid-passage. Counting tokens directly removes the guesswork.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Descending order of "how much meaning is lost by splitting here".
SEPARATORS: tuple[str, ...] = (
    "\n\n",  # paragraph
    "\n",  # line
    ". ",  # sentence
    "; ",
    ", ",
    " ",  # word
    "",  # hard character split, last resort
)


@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


class HeuristicTokenizer:
    """Character-bucket approximation used when tiktoken is unavailable.

    Deliberately conservative (3.6 chars/token vs the ~4.0 English average) so
    chunks come out under the limit rather than over it. Fine for tests and
    offline runs; not what you want in production.
    """

    CHARS_PER_TOKEN = 3.6

    def encode(self, text: str) -> list[int]:
        n = max(1, int(len(text) / self.CHARS_PER_TOKEN)) if text else 0
        return list(range(n))

    def decode(self, tokens: list[int]) -> str:  # pragma: no cover - not round-trippable
        raise NotImplementedError("HeuristicTokenizer cannot decode; overlap uses text slicing")


class TiktokenTokenizer:
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        import tiktoken

        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)


def default_tokenizer(model: str = "text-embedding-3-small") -> Tokenizer:
    try:
        return TiktokenTokenizer(model)
    except Exception as exc:  # network failure on first BPE download, or missing dep
        logger.warning(
            "tiktoken unavailable, falling back to heuristic token counting",
            extra={"error": str(exc)},
        )
        return HeuristicTokenizer()


class TokenChunker:
    """Recursive splitter that packs semantic units up to a token budget."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        min_tokens: int = 24,
    ) -> None:
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def split(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        if self.count(text) <= self.max_tokens:
            return [text]

        segments = self._atomise(text, SEPARATORS)
        return self._pack(segments)

    def _atomise(self, text: str, separators: tuple[str, ...]) -> list[str]:
        """Break text down until every piece fits inside max_tokens."""
        if self.count(text) <= self.max_tokens:
            return [text] if text.strip() else []

        if not separators:
            return self._hard_split(text)

        sep, rest = separators[0], separators[1:]
        if sep == "":
            return self._hard_split(text)

        parts = [p for p in text.split(sep) if p.strip()]
        if len(parts) <= 1:
            # This separator doesn't occur; try the next one on the same text.
            return self._atomise(text, rest)

        out: list[str] = []
        for i, part in enumerate(parts):
            # Put the separator back so sentences keep their punctuation.
            piece = part + sep if i < len(parts) - 1 and sep.strip() else part
            out.extend(self._atomise(piece, rest))
        return out

    def _hard_split(self, text: str) -> list[str]:
        """Last resort: slice on token boundaries.

        Only reached by pathological input (a single unbroken token run longer
        than the budget - minified JS, a base64 blob).
        """
        tokens = self.tokenizer.encode(text)
        if isinstance(self.tokenizer, HeuristicTokenizer):
            width = int(self.max_tokens * HeuristicTokenizer.CHARS_PER_TOKEN)
            return [text[i : i + width] for i in range(0, len(text), width)]
        return [
            self.tokenizer.decode(tokens[i : i + self.max_tokens])
            for i in range(0, len(tokens), self.max_tokens)
        ]

    def _pack(self, segments: list[str]) -> list[str]:
        """Greedily fill chunks, carrying an overlap tail between them."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for seg in segments:
            seg_tokens = self.count(seg)
            if current and current_tokens + seg_tokens > self.max_tokens:
                chunks.append(self._join(current))
                carry = self._overlap_tail(chunks[-1])
                current = [carry] if carry else []
                current_tokens = self.count(carry) if carry else 0
            current.append(seg)
            current_tokens += seg_tokens

        if current:
            chunks.append(self._join(current))

        # A trailing fragment ("Thanks!") carries no retrievable meaning on its
        # own - fold it back into its predecessor instead of indexing noise.
        if len(chunks) > 1 and self.count(chunks[-1]) < self.min_tokens:
            tail = chunks.pop()
            chunks[-1] = self._join([chunks[-1], tail])

        return [c for c in chunks if c.strip()]

    def _overlap_tail(self, text: str) -> str:
        """Last `overlap_tokens` of a chunk, so context survives the boundary."""
        if self.overlap_tokens <= 0:
            return ""
        if isinstance(self.tokenizer, HeuristicTokenizer):
            width = int(self.overlap_tokens * HeuristicTokenizer.CHARS_PER_TOKEN)
            return text[-width:].lstrip()
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.overlap_tokens:
            return text
        return self.tokenizer.decode(tokens[-self.overlap_tokens :]).lstrip()

    @staticmethod
    def _join(parts: list[str]) -> str:
        joined = " ".join(p.strip() for p in parts if p.strip())
        return re.sub(r"[ \t]{2,}", " ", joined).strip()
