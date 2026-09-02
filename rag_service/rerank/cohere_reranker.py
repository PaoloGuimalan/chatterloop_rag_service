"""Hosted cross-encoder reranking.

Chosen as the recommended non-default because it delivers cross-encoder quality
without putting torch and a 2GB model into a worker image. One network hop per
query, on a candidate set of ~50, is a few tens of milliseconds.
"""

from __future__ import annotations

import logging

from ..domain import RetrievedChunk

logger = logging.getLogger(__name__)


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-v3.5", timeout: float = 20.0) -> None:
        if not api_key:
            raise ValueError("rerank_api_key is required for the cohere reranker")
        import cohere

        self._client = cohere.ClientV2(api_key=api_key, timeout=timeout)
        self._model = model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        try:
            response = self._client.rerank(
                model=self._model,
                query=query,
                documents=[c.text for c in chunks],
                top_n=min(top_n, len(chunks)),
            )
        except Exception as exc:
            # Reranking is a quality improvement, not a correctness
            # requirement. Degrading to fusion order beats failing the request.
            logger.warning("rerank failed, falling back to fusion order", extra={"error": str(exc)})
            return chunks[:top_n]

        out: list[RetrievedChunk] = []
        for result in response.results:
            chunk = chunks[result.index]
            chunk.score = float(result.relevance_score)
            out.append(chunk)
        return out
