"""Operator CLI.

The service itself has no HTTP surface, so this is how you inspect and smoke
test it: create the collection, index something, run a query, watch the bus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .chunking import TokenChunker, default_tokenizer
from .config import get_settings
from .domain import Scope
from .embeddings import build_embedder
from .logging_setup import configure_logging
from .pipeline import IngestionPipeline, RetrievalPipeline
from .rerank import build_reranker
from .store import MilvusStore, build_filter


def _components():
    settings = get_settings()
    embedder = build_embedder(settings.embedding)
    store = MilvusStore(settings.milvus, dim=embedder.dim)
    chunker = TokenChunker(
        tokenizer=default_tokenizer(settings.embedding.model),
        max_tokens=settings.chunking.max_tokens,
        overlap_tokens=settings.chunking.overlap_tokens,
        min_tokens=settings.chunking.min_tokens,
    )
    return settings, embedder, store, chunker


def cmd_bootstrap(args: argparse.Namespace) -> int:
    settings, _embedder, store, _chunker = _components()
    store.ensure_collection()
    print(f"collection {settings.milvus.collection!r} ready at {settings.milvus.uri}")
    print(f"  dense: {settings.milvus.dense_index_type} / COSINE / dim={settings.embedding.dim}")
    print(f"  sparse: SPARSE_INVERTED_INDEX / BM25 / {settings.milvus.sparse_index_algo}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    settings, embedder, store, chunker = _components()
    store.ensure_collection()
    pipeline = IngestionPipeline(embedder, store, chunker, settings.chunking)

    path = Path(args.file)
    text = path.read_text(encoding="utf-8")
    written = pipeline.ingest_document(
        tenant_id=args.tenant,
        document_id=args.document_id or path.stem,
        text=text,
        title=args.title or path.stem,
    )
    print(f"indexed {written} chunk(s) from {path.name}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    settings, embedder, store, _chunker = _components()
    pipeline = RetrievalPipeline(
        embedder, store, build_reranker(settings.retrieval), settings.retrieval
    )
    scopes = [Scope(s) for s in args.scope] if args.scope else None
    result = pipeline.retrieve(
        tenant_id=args.tenant,
        query=args.query,
        conversation_id=args.conversation or "",
        top_k=args.top_k,
        scopes=scopes,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"{len(result.chunks)} result(s) in {result.took_ms}ms\n")
    for i, chunk in enumerate(result.chunks, 1):
        label = f"{chunk.scope}" + (f"/{chunk.role}" if chunk.role else "")
        snippet = chunk.text.replace("\n", " ")[:160]
        print(f"{i:2}. [{label}] score={chunk.score:.4f} src={chunk.source_id}")
        print(f"    {snippet}\n")
    return 0


def cmd_count(args: argparse.Namespace) -> int:
    settings, _embedder, store, _chunker = _components()
    expr = build_filter(args.tenant, [Scope.DOCUMENT])
    print(f"{store.count(expr)} document chunk(s) for tenant {args.tenant!r}")
    return 0


def cmd_emit(args: argparse.Namespace) -> int:
    """Publish a raw event onto the bus - for testing the consumer end to end."""
    settings = get_settings()
    from .messaging import build_publisher

    publisher = build_publisher(settings.messaging)
    payload = json.loads(args.payload)
    publisher.publish(
        settings.messaging.stream,
        {
            "event_id": args.event_id,
            "event_type": args.type,
            "tenant_id": args.tenant,
            "payload": payload,
        },
    )
    print(f"published {args.type} to {settings.messaging.stream}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-service-cli", description=__doc__)
    parser.add_argument("--log-level", default="WARNING")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("bootstrap", help="create the collection and indexes")
    p.set_defaults(func=cmd_bootstrap)

    p = sub.add_parser("ingest", help="index a local text file as a document")
    p.add_argument("file")
    p.add_argument("--tenant", required=True)
    p.add_argument("--document-id", default="")
    p.add_argument("--title", default="")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="run a hybrid retrieval")
    p.add_argument("query")
    p.add_argument("--tenant", required=True)
    p.add_argument("--conversation", default="")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--scope", action="append", choices=["doc", "chat"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("count", help="count a tenant's document chunks")
    p.add_argument("--tenant", required=True)
    p.set_defaults(func=cmd_count)

    p = sub.add_parser("emit", help="publish a raw event onto the bus")
    p.add_argument("--type", required=True)
    p.add_argument("--tenant", required=True)
    p.add_argument("--payload", required=True, help="JSON object")
    p.add_argument("--event-id", default="cli-test")
    p.set_defaults(func=cmd_emit)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, as_json=False)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
