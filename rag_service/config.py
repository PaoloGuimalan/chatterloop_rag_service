"""Environment-driven configuration.

Everything the service needs is read once at startup. Nothing here reads from
the message payload - in particular, credentials never travel over the bus.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Embedding models we know the native dimensionality of. `text-embedding-3-*`
# support Matryoshka truncation, so the configured dim may be smaller.
KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class MilvusSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MILVUS_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    uri: str = "http://localhost:19530"
    token: str = ""
    db_name: str = "default"
    collection: str = "rag_service"

    # HNSW is the right default for the 10k-10M vector range this service lives
    # in: in-memory, best recall-per-millisecond. Switch to DISKANN past ~10M
    # vectors per node, or IVF_FLAT if you are memory-bound and can accept lower
    # recall at equal latency.
    dense_index_type: Literal["HNSW", "IVF_FLAT", "DISKANN", "AUTOINDEX"] = "HNSW"
    hnsw_m: int = 24
    hnsw_ef_construction: int = 256
    ivf_nlist: int = 1024

    # DAAT_MAXSCORE beats DAAT_WAND on the longer, noisier queries that come out
    # of chat transcripts.
    sparse_index_algo: Literal["DAAT_MAXSCORE", "DAAT_WAND", "TAAT_NAIVE"] = "DAAT_MAXSCORE"

    # BM25 term-frequency saturation and length normalisation. Milvus defaults
    # (1.2 / 0.75) are the textbook values and hold up well on chat text.
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    # Analyzer for the BM25 text field. "english" adds stemming and stop-word
    # removal and is the right choice against a real Milvus.
    #
    # It is configurable because milvus-lite - the embedded, file-backed build
    # used for local development - supports only "standard" and "jieba", and
    # rejects collection creation outright otherwise. Without this the service
    # cannot run embedded at all.
    bm25_analyzer: str = "english"

    max_text_length: int = 16384
    shards: int = 1
    consistency_level: Literal["Strong", "Bounded", "Session", "Eventually"] = "Bounded"
    load_on_start: bool = True


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    model: str = "text-embedding-3-small"
    # Explicit dim lets you use Matryoshka truncation: text-embedding-3-large
    # truncated to 1024 outperforms 3-small at 1536 while costing a third of the
    # storage. Leave unset to use the model's native dimensionality.
    dim: int | None = None
    api_key: str = ""
    base_url: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 5

    # OpenAI accepts 2048 inputs / 300k tokens per request. Staying well under
    # both keeps p99 latency sane and makes retries cheap.
    batch_size: int = 128
    max_tokens_per_input: int = 8191
    max_tokens_per_batch: int = 120_000

    # Chat is extremely repetitive ("ok", "thanks", "any update?"). Hashing text
    # and caching the vector removes a large slice of the embedding bill.
    cache_enabled: bool = True
    cache_max_entries: int = 10_000

    # Optional per-tenant API keys as a JSON object: {"<tenant_id>": "sk-..."}.
    # The *model* is deliberately service-wide: mixing embedding models inside
    # one collection puts vectors in incomparable spaces and silently destroys
    # retrieval quality.
    tenant_keys: dict[str, str] = Field(default_factory=dict)

    @field_validator("tenant_keys", mode="before")
    @classmethod
    def _parse_tenant_keys(cls, v: object) -> object:
        if isinstance(v, str):
            return json.loads(v) if v.strip() else {}
        return v

    @model_validator(mode="after")
    def _resolve_dim(self) -> "EmbeddingSettings":
        if self.dim is None:
            native = KNOWN_EMBEDDING_DIMS.get(self.model)
            if native is None:
                raise ValueError(
                    f"EMBEDDING_DIM must be set explicitly for unknown model {self.model!r}"
                )
            object.__setattr__(self, "dim", native)
        return self

    def key_for(self, tenant_id: str) -> str:
        return self.tenant_keys.get(tenant_id) or self.api_key


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHUNK_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    # 512 tokens is the retrieval sweet spot: large enough to carry an idea,
    # small enough that a cross-encoder can score it without truncation and that
    # a hit points at a specific passage rather than a whole page.
    max_tokens: int = 512
    overlap_tokens: int = 64
    # Chunks below this are usually headings or list fragments - they pollute
    # results without carrying meaning.
    min_tokens: int = 24
    # Prepending the document title to every chunk before embedding is a cheap,
    # consistent recall win on documents whose sections don't restate context.
    prepend_title: bool = True


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    top_k: int = 8
    # Hybrid search overfetches, then rerank/MMR narrow back down. Recall is
    # only recoverable at this stage - anything the fusion misses is gone.
    fetch_multiplier: int = 6
    max_fetch: int = 120

    # Reciprocal Rank Fusion. Cosine similarity and BM25 scores are on
    # incomparable scales, so rank-based fusion needs no normalisation and no
    # per-corpus weight tuning. k=60 is the value from the original RRF paper
    # and is remarkably insensitive.
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    # Only used when fusion == "weighted": [dense, sparse].
    dense_weight: float = 0.7
    sparse_weight: float = 0.3

    hnsw_ef: int = 128
    # Drops the lowest-weight query terms from the sparse leg. Small values trade
    # a little recall for a lot of latency on long queries.
    sparse_drop_ratio: float = 0.2

    # Recent turns are prepended verbatim - RAG should never lose the immediate
    # thread of a conversation to a similarity cutoff.
    recent_history_turns: int = 4

    # Maximal Marginal Relevance. Chat corpora are full of near-duplicates; MMR
    # spends the context budget on distinct information instead of eight
    # rephrasings of the same answer.
    mmr_enabled: bool = True
    mmr_lambda: float = 0.7

    rerank_provider: Literal["none", "cohere", "local"] = "none"
    rerank_model: str = "rerank-v3.5"
    rerank_api_key: str = ""
    rerank_timeout_seconds: float = 20.0


class MessagingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BUS_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    backend: Literal["redis", "gcp"] = "redis"

    redis_url: str = "redis://localhost:6379/0"
    stream: str = "rag.events"
    group: str = "rag-workers"
    consumer_name: str = ""  # defaults to hostname:pid
    dlq_stream: str = "rag.events.dlq"
    reply_stream_default: str = "rag.results"

    gcp_project: str = ""
    gcp_subscription: str = ""
    gcp_reply_topic: str = ""

    batch_size: int = 16
    block_ms: int = 5_000
    # Messages pending longer than this are assumed orphaned by a dead worker
    # and reclaimed by a live one.
    claim_min_idle_ms: int = 60_000
    max_deliveries: int = 5

    # Pub/sub is at-least-once, and embedding twice costs money and corrupts
    # nothing but wastes quota. Dedupe on event_id for this window.
    idempotency_ttl_seconds: int = 86_400


class ChatterloopSettings(BaseSettings):
    """Acting as a user on the chatterloop platform."""

    model_config = SettingsConfigDict(
        env_prefix="CHATTERLOOP_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    enabled: bool = False

    # The bot's Entity id. A bot acts through an Entity exactly like a person
    # does, and this is the id whose realtime channel it subscribes to.
    bot_entity_id: str = ""
    bot_handle: str = ""
    bot_aliases: list[str] = Field(default_factory=list)

    # Addressed-only is the product rule: the bot answers when named, and when
    # somebody replies directly to something it said. Turning this off - i.e.
    # replying to unaddressed messages - is not implemented anywhere. The flag
    # exists so that "respond to everything" has to be a deliberate future
    # change rather than an accident.
    #
    # Keeps its original name because it is an environment variable that live
    # deployments already set, and the meaning it names is unchanged: only
    # what counts as "addressed" grew.
    respond_to_mentions_only: bool = True

    # Whether a direct reply to one of the bot's own messages or comments
    # counts as addressing it, with no @handle needed. Off, the bot is
    # mention-only exactly as before, and a message that does not name it costs
    # no read at all.
    answer_replies: bool = True

    # How many recent replies the probe looks at per conversation. Bounds the
    # work per frame; the server-side route bounds the scan the same way.
    reply_probe_window: int = 25

    # Answer ONLY what happened while this process was listening.
    #
    # Frames are live by construction - pub/sub has no replay. The reads that
    # resolve them are not: the reply probe returns a window, and comment
    # notifications are durable and accumulate indefinitely while the bot is
    # down. With this off, a restart works through that backlog and every
    # queued row becomes a real reply to a real person about something they
    # said hours ago.
    #
    # The cost, stated plainly: a comment mention that lands during a deploy is
    # never answered. A notification the bot did not see live is one it will
    # never see. Turn this off only for a deliberate catch-up run, and expect
    # the burst.
    only_live_events: bool = True

    # Entities that never get a reply. Put other bots here: two bots in one
    # realm that both answer when addressed will otherwise talk to each other
    # forever.
    ignore_entity_ids: list[str] = Field(default_factory=list)

    cooldown_seconds: float = 5.0
    max_replies_per_hour: int = 30

    # How much conversation to fetch and index when addressed.
    history_window: int = 40
    top_k: int = 8

    # "openai" | "stub". The stub reports what was retrieved instead of
    # composing prose, which keeps retrieval quality debuggable on its own.
    reply_generator: Literal["openai", "stub"] = "stub"
    reply_model: str = "gpt-4o-mini"
    reply_api_key: str = ""
    # OpenAI-compatible endpoint override, e.g.
    # https://api.groq.com/openai/v1 . Empty means api.openai.com.
    reply_base_url: str = ""
    reply_max_tokens: int = 400
    reply_temperature: float = 0.3

    @field_validator("bot_aliases", "ignore_entity_ids", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [part.strip() for part in v.split(",") if part.strip()]
        return v


class PlatformSettings(BaseSettings):
    """Access to chatterloop through developer_service.

    Was direct access to the platform's Mongo, Postgres and Redis; is now one
    credential against one origin. Still enabled separately from the bot
    itself, so the pipeline can run listen-only against no platform at all.
    """

    model_config = SettingsConfigDict(
        env_prefix="PLATFORM_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    enabled: bool = False

    # The single credential, issued by `manage.py issue_entity_token`. Shown
    # once at issue and unrecoverable afterwards; if it is lost, issue another
    # and revoke this one.
    token: str = ""

    # developer_service, which serves everything: conversation history,
    # comment mentions, sending a reply, and the realtime event stream.
    # Origin only - the client appends /v1/... itself.
    api_base_url: str = ""

    timeout_seconds: float = 15.0

    # Transient failures only (timeouts, 5xx). A 401 or 403 is never retried:
    # a bad token will be exactly as bad on the fourth attempt.
    max_attempts: int = 3


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    log_level: str = "INFO"
    log_json: bool = True
    shutdown_grace_seconds: float = 30.0

    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    messaging: MessagingSettings = Field(default_factory=MessagingSettings)
    chatterloop: ChatterloopSettings = Field(default_factory=ChatterloopSettings)
    platform: PlatformSettings = Field(default_factory=PlatformSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
