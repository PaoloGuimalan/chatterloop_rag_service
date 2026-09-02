"""Local cross-encoder reranking.

Use when data cannot leave your infrastructure. `BAAI/bge-reranker-v2-m3` is the
strong multilingual default; `cross-encoder/ms-marco-MiniLM-L-6-v2` is ~20x
smaller and CPU-viable if latency matters more than quality.

Pulls torch. Keep it out of slim worker images unless you need it.
"""

from __future__ import annotations

import logging

from ..domain import RetrievedChunk

logger = logging.getLogger(__name__)


class LocalCrossEncoderReranker:
    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        try:
            scores = self._model.predict([(query, c.text) for c in chunks])
        except Exception as exc:
            logger.warning("rerank failed, falling back to fusion order", extra={"error": str(exc)})
            return chunks[:top_n]

        for chunk, score in zip(chunks, scores, strict=True):
            chunk.score = float(score)
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:top_n]
