"""OpenAI embedding backend."""

from __future__ import annotations

import logging
import math
from typing import Iterator

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ..chunking import Tokenizer, default_tokenizer
from ..config import EmbeddingSettings
from .cache import LRUEmbeddingCache, content_key

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


def l2_normalise(vector: list[float]) -> list[float]:
    """Scale to unit length.

    OpenAI returns unit vectors at a model's native dimensionality, but *not*
    when you request Matryoshka truncation via `dimensions` - the truncated
    vector has to be renormalised or cosine scores drift. Doing it
    unconditionally costs nothing and removes a footgun.
    """
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


class OpenAIEmbedder:
    def __init__(
        self,
        settings: EmbeddingSettings,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.settings = settings
        self._dim = int(settings.dim or 0)
        if self._dim <= 0:
            raise ValueError("embedding dimension must be resolved before use")
        self.tokenizer = tokenizer or default_tokenizer(settings.model)
        self.cache = (
            LRUEmbeddingCache(settings.cache_max_entries) if settings.cache_enabled else None
        )
        self._clients: dict[str, object] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def _client(self, tenant_id: str):
        """One client per distinct API key.

        Keys come from configuration, never from the message payload - a bus
        message is the wrong place for a credential, and the previous pipeline
        passing `organization.llm_api_key` around through call arguments is how
        keys end up in logs.
        """
        key = self.settings.key_for(tenant_id)
        if not key:
            raise EmbeddingError(f"no embedding API key configured for tenant {tenant_id!r}")
        if key not in self._clients:
            from openai import OpenAI

            self._clients[key] = OpenAI(
                api_key=key,
                base_url=self.settings.base_url,
                timeout=self.settings.timeout_seconds,
                max_retries=0,  # tenacity owns retries so backoff is observable
            )
        return self._clients[key]

    def _truncate(self, text: str) -> str:
        """Clamp to the model's context window.

        A single oversized input fails the whole batch, so this is a hard
        guarantee rather than an assumption about upstream chunking.
        """
        limit = self.settings.max_tokens_per_input
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= limit:
            return text
        try:
            return self.tokenizer.decode(tokens[:limit])
        except NotImplementedError:
            # Heuristic tokenizer: fall back to a conservative character slice.
            return text[: int(limit * 3.6)]

    def _batches(self, texts: list[str]) -> Iterator[list[int]]:
        """Yield index batches bounded by both item count and token budget."""
        batch: list[int] = []
        batch_tokens = 0
        for i, text in enumerate(texts):
            n = len(self.tokenizer.encode(text))
            over_items = len(batch) >= self.settings.batch_size
            over_tokens = batch and batch_tokens + n > self.settings.max_tokens_per_batch
            if over_items or over_tokens:
                yield batch
                batch, batch_tokens = [], 0
            batch.append(i)
            batch_tokens += n
        if batch:
            yield batch

    def embed_documents(self, texts: list[str], tenant_id: str = "") -> list[list[float]]:
        if not texts:
            return []

        prepared = [self._truncate(t.replace("\n", " ").strip() or " ") for t in texts]
        out: list[list[float]] = [[] for _ in prepared]

        pending: list[int] = []
        for i, text in enumerate(prepared):
            if self.cache is not None:
                hit = self.cache.get(content_key(text, self.settings.model, self._dim))
                if hit is not None:
                    out[i] = hit
                    continue
            pending.append(i)

        for batch in self._batches([prepared[i] for i in pending]):
            # `batch` indexes into the pending sub-list, not `prepared`.
            absolute = [pending[j] for j in batch]
            vectors = self._call([prepared[i] for i in absolute], tenant_id)
            for i, vec in zip(absolute, vectors, strict=True):
                out[i] = vec
                if self.cache is not None:
                    self.cache.put(content_key(prepared[i], self.settings.model, self._dim), vec)

        missing = [i for i, v in enumerate(out) if not v]
        if missing:
            raise EmbeddingError(f"embedding provider returned no vector for indices {missing}")
        return out

    def embed_query(self, text: str, tenant_id: str = "") -> list[float]:
        # text-embedding-3-* is symmetric: queries and documents go through the
        # same encoder with no instruction prefix. If you swap in an asymmetric
        # model (E5, BGE, GTE), this is where the "query: " prefix belongs.
        return self.embed_documents([text], tenant_id)[0]

    def _call(self, texts: list[str], tenant_id: str) -> list[list[float]]:
        client = self._client(tenant_id)

        @retry(
            stop=stop_after_attempt(self.settings.max_retries),
            wait=wait_random_exponential(multiplier=1, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _do() -> list[list[float]]:
            kwargs: dict[str, object] = {"input": texts, "model": self.settings.model}
            # Only send `dimensions` when actually truncating; ada-002 rejects it.
            native = _native_dim(self.settings.model)
            if native is not None and self._dim != native:
                kwargs["dimensions"] = self._dim
            response = client.embeddings.create(**kwargs)  # type: ignore[attr-defined]
            # The API documents order preservation, but the cost of being wrong
            # here is silently mismatched vectors, so sort by index explicitly.
            ordered = sorted(response.data, key=lambda d: d.index)
            return [l2_normalise(list(d.embedding)) for d in ordered]

        vectors = _do()
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"expected {len(texts)} embeddings, provider returned {len(vectors)}"
            )
        bad = next((v for v in vectors if len(v) != self._dim), None)
        if bad is not None:
            raise EmbeddingError(
                f"dimension mismatch: collection expects {self._dim}, provider returned {len(bad)}"
            )
        return vectors


def _native_dim(model: str) -> int | None:
    from ..config import KNOWN_EMBEDDING_DIMS

    return KNOWN_EMBEDDING_DIMS.get(model)
