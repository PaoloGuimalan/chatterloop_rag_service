from __future__ import annotations

import logging

from ..config import RetrievalSettings
from .base import NoopReranker, Reranker

logger = logging.getLogger(__name__)

__all__ = ["NoopReranker", "Reranker", "build_reranker"]


def build_reranker(settings: RetrievalSettings) -> Reranker:
    """Construct the configured reranker, degrading to pass-through.

    A missing optional dependency should not stop the worker from serving
    retrievals at slightly lower quality.
    """
    provider = settings.rerank_provider
    try:
        if provider == "cohere":
            from .cohere_reranker import CohereReranker

            return CohereReranker(
                api_key=settings.rerank_api_key,
                model=settings.rerank_model,
                timeout=settings.rerank_timeout_seconds,
            )
        if provider == "local":
            from .local_reranker import LocalCrossEncoderReranker

            return LocalCrossEncoderReranker(model=settings.rerank_model)
    except Exception as exc:
        logger.error(
            "reranker unavailable, continuing without reranking",
            extra={"provider": provider, "error": str(exc)},
        )
        return NoopReranker()

    return NoopReranker()
