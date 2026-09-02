"""Collection schema and index configuration.

Design notes that matter more than the code:

*Multi-tenancy.* `tenant_id` is a Milvus **partition key**. Milvus hashes it
into physical partitions and, given a `tenant_id == X` predicate, only searches
the partitions that can contain X. That gives isolation and a latency win
without the collection-per-tenant explosion (Milvus caps collections in the low
thousands; organisations are not similarly capped).

*Hybrid retrieval.* The collection carries two vector fields. `dense` holds the
embedding. `sparse` is generated **inside Milvus** by a BM25 Function over the
`text` field, so lexical vectors are never computed, shipped, or version-skewed
by the client - you insert text and Milvus maintains the term statistics. This
is the single biggest quality upgrade over a dense-only store: embeddings are
famously weak on exact tokens (order numbers, SKUs, error codes, surnames),
which is most of what a support conversation is made of.
"""

from __future__ import annotations

from typing import Any

from ..config import MilvusSettings

# Fields returned on every search. `dense` is requested separately and only
# when MMR needs it - shipping 1536 floats per candidate is not free.
OUTPUT_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "text",
    "scope",
    "conversation_id",
    "source_id",
    "role",
    "title",
    "chunk_index",
    "created_at",
    "meta",
)


def build_schema(settings: MilvusSettings, dim: int) -> Any:
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)

    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
    # Partition key. Every query predicate includes it; see store/filters.py.
    schema.add_field("tenant_id", DataType.VARCHAR, max_length=64, is_partition_key=True)
    schema.add_field("scope", DataType.VARCHAR, max_length=16)
    schema.add_field("conversation_id", DataType.VARCHAR, max_length=64)
    schema.add_field("source_id", DataType.VARCHAR, max_length=128)
    schema.add_field("chunk_index", DataType.INT64)
    # Always populated (empty string for documents) so downstream code can read
    # it unconditionally.
    schema.add_field("role", DataType.VARCHAR, max_length=16)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=settings.max_text_length,
        # Required for BM25: Milvus tokenises this field to build the sparse
        # vector.
        enable_analyzer=True,
        analyzer_params={"type": "english"},
    )
    schema.add_field("created_at", DataType.INT64)
    schema.add_field("meta", DataType.JSON, nullable=True)

    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

    # Server-side BM25. Never insert `sparse` yourself - Milvus owns it.
    schema.add_function(
        Function(
            name="bm25_text_to_sparse",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse"],
        )
    )
    return schema


def build_index_params(settings: MilvusSettings) -> Any:
    from pymilvus import MilvusClient

    index_params = MilvusClient.prepare_index_params()

    if settings.dense_index_type == "HNSW":
        # M controls graph degree (recall and memory); efConstruction controls
        # build-time search width (recall and build cost). 24/256 sits just
        # past the knee of the recall curve for 1536-d text embeddings.
        dense_params: dict[str, Any] = {
            "M": settings.hnsw_m,
            "efConstruction": settings.hnsw_ef_construction,
        }
    elif settings.dense_index_type == "IVF_FLAT":
        dense_params = {"nlist": settings.ivf_nlist}
    else:
        dense_params = {}

    index_params.add_index(
        field_name="dense",
        index_type=settings.dense_index_type,
        # COSINE, with unit-normalised vectors from the embedder. Equivalent to
        # IP here, but naming the intent stops anyone from inserting
        # unnormalised vectors later and quietly getting magnitude-biased
        # rankings.
        metric_type="COSINE",
        params=dense_params,
    )

    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "inverted_index_algo": settings.sparse_index_algo,
            "bm25_k1": settings.bm25_k1,
            "bm25_b": settings.bm25_b,
        },
    )
    return index_params
