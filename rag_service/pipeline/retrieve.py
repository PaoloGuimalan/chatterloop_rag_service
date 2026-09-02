"""Read path: query in, ranked context out.

    embed query
      -> hybrid search (dense HNSW + BM25, RRF-fused)   [recall]
      -> cross-encoder rerank                           [precision]
      -> MMR                                            [coverage]
      -> prepend recent turns                           [continuity]

Each stage narrows a wider set produced by the one before it. The overfetch
multiplier is the knob that matters: everything downstream can only reorder or
discard what hybrid search returned.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable

from ..config import RetrievalSettings
from ..domain import RetrievalResult, RetrievedChunk, Role, Scope
from ..embeddings.base import Embedder
from ..rerank.base import Reranker
from ..store import MilvusStore, build_filter, quote
from .diversity import mmr_select

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: MilvusStore,
        reranker: Reranker,
        settings: RetrievalSettings,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.settings = settings

    def retrieve(
        self,
        tenant_id: str,
        query: str,
        conversation_id: str = "",
        top_k: int | None = None,
        scopes: Iterable[Scope] | None = None,
        include_recent_history: bool = True,
    ) -> RetrievalResult:
        started = time.monotonic()
        top_k = top_k or self.settings.top_k
        query = (query or "").strip()

        if not tenant_id:
            raise ValueError("tenant_id is required")
        if not query:
            return RetrievalResult(query="", tenant_id=tenant_id, conversation_id=conversation_id,
                                   chunks=[])

        scope_list = list(scopes) if scopes is not None else [Scope.DOCUMENT, Scope.CHAT]
        filter_expr = build_filter(tenant_id, scope_list, conversation_id or None)

        fetch_k = min(top_k * self.settings.fetch_multiplier, self.settings.max_fetch)
        # MMR needs the candidate vectors; skip the bandwidth when it's off.
        want_vectors = self.settings.mmr_enabled

        query_vector = self.embedder.embed_query(query, tenant_id=tenant_id)

        candidates = self.store.hybrid_search(
            query_text=query,
            query_vector=query_vector,
            filter_expr=filter_expr,
            limit=fetch_k,
            retrieval=self.settings,
            with_vectors=want_vectors,
        )
        fused_count = len(candidates)

        if candidates:
            # Rerank a wider slice than the final answer so MMR still has room
            # to trade relevance for coverage afterwards.
            rerank_n = min(len(candidates), max(top_k * 2, top_k + 4))
            candidates = self.reranker.rerank(query, candidates, rerank_n)

        if self.settings.mmr_enabled and len(candidates) > top_k:
            candidates = mmr_select(
                query_vector, candidates, top_k, self.settings.mmr_lambda
            )
        else:
            candidates = candidates[:top_k]

        chunks = candidates
        if include_recent_history and conversation_id and self.settings.recent_history_turns:
            chunks = self._merge_recent(tenant_id, conversation_id, chunks)

        took_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "retrieval complete",
            extra={
                "tenant_id": tenant_id,
                "conversation_id": conversation_id,
                "fused": fused_count,
                "returned": len(chunks),
                "took_ms": took_ms,
            },
        )
        return RetrievalResult(
            query=query,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            chunks=chunks,
            took_ms=took_ms,
        )

    # ---------------------------------------------------------------- history

    def _merge_recent(
        self, tenant_id: str, conversation_id: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Put the last few turns in front of the similarity results.

        Similarity search has no notion of "what we were just talking about", so
        a follow-up like "and the second one?" retrieves nothing useful on its
        own. Recent turns are prepended unconditionally and deduplicated against
        the ranked set.
        """
        recent = self.recent_turns(tenant_id, conversation_id, self.settings.recent_history_turns)
        if not recent:
            return chunks
        seen = {c.chunk_id for c in recent}
        return recent + [c for c in chunks if c.chunk_id not in seen]

    def recent_turns(
        self, tenant_id: str, conversation_id: str, limit: int
    ) -> list[RetrievedChunk]:
        """Most recent chat chunks in a conversation, oldest first.

        Milvus `query` has no ORDER BY, so this pulls a bounded window and sorts
        client-side. Conversations are small enough for that to be cheap; if you
        ever need this over very long conversations, keep the ordering in
        Postgres and pass the turns in with the request instead.
        """
        window = max(limit * 8, 64)
        expr = (
            f"tenant_id == {quote(tenant_id)} "
            f'and scope == "chat" '
            f"and conversation_id == {quote(conversation_id)}"
        )
        try:
            rows = self.store.client.query(
                collection_name=self.store.settings.collection,
                filter=expr,
                output_fields=[
                    "chunk_id",
                    "text",
                    "scope",
                    "role",
                    "source_id",
                    "conversation_id",
                    "chunk_index",
                    "created_at",
                ],
                limit=window,
            )
        except Exception as exc:
            logger.warning(
                "recent history lookup failed, continuing without it",
                extra={"error": str(exc), "conversation_id": conversation_id},
            )
            return []

        rows.sort(key=lambda r: (int(r.get("created_at") or 0), int(r.get("chunk_index") or 0)))
        tail = rows[-limit:]
        return [
            RetrievedChunk(
                chunk_id=str(r.get("chunk_id") or ""),
                text=r.get("text") or "",
                scope=Scope.CHAT,
                score=0.0,
                source_id=r.get("source_id") or "",
                conversation_id=r.get("conversation_id") or "",
                role=_safe_role(r.get("role")),
                chunk_index=int(r.get("chunk_index") or 0),
                created_at=int(r.get("created_at") or 0),
            )
            for r in tail
        ]


def _safe_role(value: object) -> Role:
    try:
        return Role(value or "")
    except ValueError:
        return Role.NONE
