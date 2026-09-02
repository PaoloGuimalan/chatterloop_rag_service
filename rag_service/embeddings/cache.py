"""Content-addressed embedding cache.

Support conversations repeat themselves relentlessly - greetings, "any update
on this?", canned agent replies. Hashing the exact text and reusing the vector
takes a visible bite out of the embedding bill and out of p99 latency, at the
cost of one dict lookup.

In-process and per-worker on purpose: embeddings are cheap to recompute after a
restart, and a shared Redis cache would add a network hop to save a call we may
not have needed anyway. Promote it to Redis only if you measure a real hit-rate
loss from worker churn.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict


def content_key(text: str, model: str, dim: int) -> str:
    # The model and dim are part of the key: the same string embedded by a
    # different model is a different vector, and serving a stale one across a
    # model change would be a silent correctness bug.
    raw = f"{model}\x00{dim}\x00{text}".encode("utf-8")
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


class LRUEmbeddingCache:
    def __init__(self, max_entries: int = 10_000) -> None:
        self.max_entries = max_entries
        self._data: OrderedDict[str, list[float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> list[float] | None:
        vec = self._data.get(key)
        if vec is None:
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return vec

    def put(self, key: str, vector: list[float]) -> None:
        if self.max_entries <= 0:
            return
        self._data[key] = vector
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        return len(self._data)
