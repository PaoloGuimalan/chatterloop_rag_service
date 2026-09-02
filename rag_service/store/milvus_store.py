"""Milvus-backed vector store with hybrid dense + BM25 retrieval."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..config import MilvusSettings, RetrievalSettings
from ..domain import Chunk, RetrievedChunk, Role, Scope
from .schema import OUTPUT_FIELDS, build_index_params, build_schema

logger = logging.getLogger(__name__)


class MilvusStore:
    def __init__(self, settings: MilvusSettings, dim: int) -> None:
        self.settings = settings
        self.dim = dim
        self._client: Any | None = None

    # ---------------------------------------------------------------- client

    @property
    def client(self) -> Any:
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(
                uri=self.settings.uri,
                token=self.settings.token or None,
                db_name=self.settings.db_name,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - best effort on shutdown
                logger.debug("error closing milvus client", exc_info=True)
            self._client = None

    # ------------------------------------------------------------ lifecycle

    def ensure_collection(self) -> None:
        """Create the collection and indexes if absent, then load it.

        Safe to call from every worker on boot: creation races surface as an
        already-exists error from Milvus, which we treat as success.
        """
        name = self.settings.collection

        if not self.client.has_collection(name):
            logger.info("creating collection", extra={"collection": name, "dim": self.dim})
            try:
                self.client.create_collection(
                    collection_name=name,
                    schema=build_schema(self.settings, self.dim),
                    index_params=build_index_params(self.settings),
                    consistency_level=self.settings.consistency_level,
                    num_shards=self.settings.shards,
                )
            except Exception as exc:
                if not self.client.has_collection(name):
                    raise
                logger.info("collection created concurrently", extra={"error": str(exc)})

        self._assert_dim_matches(name)

        if self.settings.load_on_start:
            self.client.load_collection(name)
            logger.info("collection loaded", extra={"collection": name})

    def _assert_dim_matches(self, name: str) -> None:
        """Fail loudly on an embedding-model change.

        Writing 1024-d vectors into a 1536-d collection is rejected by Milvus,
        but the failure surfaces deep inside an insert. Checking at boot turns a
        confusing runtime error into an obvious startup error.
        """
        try:
            desc = self.client.describe_collection(name)
        except Exception:  # pragma: no cover - description is advisory
            return
        for field in desc.get("fields", []):
            if field.get("name") == "dense":
                actual = (field.get("params") or {}).get("dim")
                if actual and int(actual) != self.dim:
                    raise RuntimeError(
                        f"collection {name!r} has dense dim {actual}, embedder produces "
                        f"{self.dim}. Changing embedding model requires a new collection "
                        f"and a full reindex."
                    )

    # ---------------------------------------------------------------- writes

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        """Insert-or-replace chunks.

        Primary keys are deterministic (tenant, source, index), so re-ingesting
        an edited document overwrites its chunks rather than duplicating them.
        Note this leaves orphans if the new version has *fewer* chunks - the
        ingest pipeline deletes by source_id first to handle that.
        """
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")

        rows: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.dim:
                raise ValueError(f"vector dim {len(vector)} != collection dim {self.dim}")
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "tenant_id": chunk.tenant_id,
                    "scope": str(chunk.scope),
                    "conversation_id": chunk.conversation_id,
                    "source_id": chunk.source_id,
                    "chunk_index": chunk.chunk_index,
                    "role": str(chunk.role),
                    "title": chunk.title[:512],
                    "text": chunk.text[: self.settings.max_text_length],
                    "created_at": chunk.created_at,
                    "meta": chunk.meta or {},
                    "dense": list(vector),
                    # `sparse` is intentionally absent: the BM25 Function
                    # produces it server-side from `text`.
                }
            )

        self.client.upsert(collection_name=self.settings.collection, data=rows)
        return len(rows)

    def delete(self, filter_expr: str) -> None:
        self.client.delete(collection_name=self.settings.collection, filter=filter_expr)

    # --------------------------------------------------------------- reading

    def hybrid_search(
        self,
        query_text: str,
        query_vector: Sequence[float],
        filter_expr: str,
        limit: int,
        retrieval: RetrievalSettings,
        with_vectors: bool = False,
    ) -> list[RetrievedChunk]:
        """Run dense and lexical retrieval in parallel and fuse the rankings.

        Both legs apply the same tenant filter and each returns `limit`
        candidates; the fuser then interleaves them. Recall is set here - a
        passage neither leg surfaces cannot be recovered by reranking.
        """
        from pymilvus import AnnSearchRequest

        dense_req = AnnSearchRequest(
            data=[list(query_vector)],
            anns_field="dense",
            param={"metric_type": "COSINE", "params": {"ef": retrieval.hnsw_ef}},
            limit=limit,
            expr=filter_expr,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse",
            param={
                "metric_type": "BM25",
                "params": {"drop_ratio_search": retrieval.sparse_drop_ratio},
            },
            limit=limit,
            expr=filter_expr,
        )

        output_fields = list(OUTPUT_FIELDS) + (["dense"] if with_vectors else [])

        results = self.client.hybrid_search(
            collection_name=self.settings.collection,
            reqs=[dense_req, sparse_req],
            ranker=self._ranker(retrieval),
            limit=limit,
            output_fields=output_fields,
        )
        if not results:
            return []
        return [_to_chunk(hit) for hit in results[0]]

    @staticmethod
    def _ranker(retrieval: RetrievalSettings) -> Any:
        from pymilvus import RRFRanker, WeightedRanker

        if retrieval.fusion == "weighted":
            # Only meaningful if you normalise scores first; Milvus does this
            # per-leg. Use it when you have measured that one signal should
            # dominate on your corpus - otherwise RRF is the safer default.
            return WeightedRanker(retrieval.dense_weight, retrieval.sparse_weight)
        # Reciprocal Rank Fusion: score = sum over legs of 1 / (k + rank).
        # Rank-based, so cosine and BM25 never have to be made commensurable.
        return RRFRanker(retrieval.rrf_k)

    def count(self, filter_expr: str) -> int:
        rows = self.client.query(
            collection_name=self.settings.collection,
            filter=filter_expr,
            output_fields=["count(*)"],
        )
        return int(rows[0]["count(*)"]) if rows else 0


def _to_chunk(hit: dict[str, Any]) -> RetrievedChunk:
    """Normalise a Milvus hit.

    Fields are read with defaults throughout: a document row has no `role` and a
    chat row has no `title`, and neither absence should be able to raise.
    """
    entity: dict[str, Any] = hit.get("entity") or hit

    raw_scope = entity.get("scope") or Scope.DOCUMENT
    try:
        scope = Scope(raw_scope)
    except ValueError:
        scope = Scope.DOCUMENT

    raw_role = entity.get("role") or ""
    try:
        role = Role(raw_role)
    except ValueError:
        role = Role.NONE

    return RetrievedChunk(
        chunk_id=str(entity.get("chunk_id") or hit.get("id") or ""),
        text=entity.get("text") or "",
        scope=scope,
        score=float(hit.get("distance", hit.get("score", 0.0)) or 0.0),
        source_id=entity.get("source_id") or "",
        conversation_id=entity.get("conversation_id") or "",
        role=role,
        title=entity.get("title") or "",
        chunk_index=int(entity.get("chunk_index") or 0),
        created_at=int(entity.get("created_at") or 0),
        meta=entity.get("meta") or {},
        dense=list(entity["dense"]) if entity.get("dense") is not None else None,
    )
