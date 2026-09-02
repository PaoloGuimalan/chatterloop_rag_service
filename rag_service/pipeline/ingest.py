"""Write path: text in, vectors and lexical index out."""

from __future__ import annotations

import logging
from typing import Any

from ..chunking import TokenChunker
from ..config import ChunkingSettings
from ..domain import Chunk, Role, Scope, now_ms
from ..embeddings.base import Embedder
from ..store import MilvusStore, delete_filter

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: MilvusStore,
        chunker: TokenChunker,
        settings: ChunkingSettings,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.chunker = chunker
        self.settings = settings

    # -------------------------------------------------------------- documents

    def ingest_document(
        self,
        tenant_id: str,
        document_id: str,
        text: str,
        title: str = "",
        meta: dict[str, Any] | None = None,
        created_at: int | None = None,
    ) -> int:
        """Chunk, embed and store a document, replacing any previous version.

        The delete-then-write is not redundant with upsert: primary keys are
        (tenant, source, chunk_index), so a revision with fewer chunks would
        otherwise leave the tail of the old version behind as retrievable
        orphans.
        """
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if not document_id:
            raise ValueError("document_id is required")

        pieces = self.chunker.split(text)
        if not pieces:
            logger.info(
                "document produced no chunks, deleting any prior version",
                extra={"tenant_id": tenant_id, "document_id": document_id},
            )
            self.store.delete(delete_filter(tenant_id, source_id=document_id))
            return 0

        stamp = created_at or now_ms()
        chunks = [
            Chunk(
                tenant_id=tenant_id,
                scope=Scope.DOCUMENT,
                text=piece,
                source_id=document_id,
                chunk_index=i,
                title=title,
                role=Role.NONE,
                created_at=stamp,
                meta=meta or {},
            )
            for i, piece in enumerate(pieces)
        ]

        self.store.delete(delete_filter(tenant_id, source_id=document_id))
        written = self._embed_and_write(chunks, tenant_id)
        logger.info(
            "document indexed",
            extra={"tenant_id": tenant_id, "document_id": document_id, "chunks": written},
        )
        return written

    # ----------------------------------------------------------- chat messages

    def index_message(
        self,
        tenant_id: str,
        conversation_id: str,
        message_id: str,
        text: str,
        role: Role = Role.USER,
        meta: dict[str, Any] | None = None,
        created_at: int | None = None,
    ) -> int:
        """Index a single chat turn.

        A message is the atomic retrieval unit - splitting "yes, that worked"
        away from what it answers destroys the meaning of both halves. Only
        genuinely long messages (pasted logs, forwarded threads) get chunked.
        """
        if not tenant_id or not conversation_id or not message_id:
            raise ValueError("tenant_id, conversation_id and message_id are all required")

        text = (text or "").strip()
        if not text:
            return 0

        pieces = (
            [text]
            if self.chunker.count(text) <= self.chunker.max_tokens
            else self.chunker.split(text)
        )
        stamp = created_at or now_ms()

        chunks = [
            Chunk(
                tenant_id=tenant_id,
                scope=Scope.CHAT,
                text=piece,
                source_id=message_id,
                chunk_index=i,
                conversation_id=conversation_id,
                role=role,
                created_at=stamp,
                meta=meta or {},
            )
            for i, piece in enumerate(pieces)
        ]
        return self._embed_and_write(chunks, tenant_id)

    # --------------------------------------------------------------- deletion

    def delete_document(self, tenant_id: str, document_id: str) -> None:
        self.store.delete(delete_filter(tenant_id, source_id=document_id))
        logger.info(
            "document deleted", extra={"tenant_id": tenant_id, "document_id": document_id}
        )

    def delete_conversation(self, tenant_id: str, conversation_id: str) -> None:
        self.store.delete(delete_filter(tenant_id, conversation_id=conversation_id))
        logger.info(
            "conversation deleted",
            extra={"tenant_id": tenant_id, "conversation_id": conversation_id},
        )

    # ---------------------------------------------------------------- internal

    def _embed_and_write(self, chunks: list[Chunk], tenant_id: str) -> int:
        texts = [c.embedding_text(self.settings.prepend_title) for c in chunks]
        vectors = self.embedder.embed_documents(texts, tenant_id=tenant_id)
        return self.store.upsert(chunks, vectors)
