"""Environment-driven configuration.

Everything the service needs is read once at startup. Nothing here reads from
the message payload - in particular, credentials never travel over the bus.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
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


class ToolConfig(BaseModel):
    """One function the reply generator may call, mid-completion.

    Deliberately shaped after NeonCentralized_API's `Tool` model
    (llm/models.py: name, description, parameters_schema, headers_schema,
    api_endpoint, http_method, param_type) - that is the tested, working
    function-calling contract. What differs is where it lives: Neon reads
    Tool rows from Postgres; this reads a JSON array from CHATTERLOOP_TOOLS.
    There is no database here to read from, and there should not be one -
    rag_service is a standalone developer-API product sitting in front of
    chatterloop, not a module inside it, so it has no standing to ask the
    chatterloop system to store or serve tool config on its behalf. Every
    tool this bot can call is therefore defined in ITS OWN environment,
    per deployment - "environment-bounded" the same way every other
    credential and endpoint in this file already is.
    """

    name: str
    description: str = ""
    # JSON Schema for the arguments the model must supply. An empty object
    # schema is a valid "no arguments" tool, not a misconfiguration.
    parameters_schema: dict = Field(default_factory=lambda: {"type": "object", "properties": {}})
    api_endpoint: str
    http_method: Literal["GET", "POST"] = "POST"
    # "query" -> GET params; "route" -> `.format(**arguments)` into the URL
    # (e.g. "https://api.example/items/{item_id}"); "body" -> POST JSON.
    param_type: Literal["query", "route", "body"] = "body"
    # Static headers only (e.g. a bearer token for a third-party API this
    # tool calls). Never templated from the model's arguments - a tool that
    # could inject its OWN auth header from LLM-controlled input would let a
    # prompt injection reach past whatever that header authorizes.
    headers: dict[str, str] = Field(default_factory=dict)
    is_enabled: bool = True


class AgentConfig(BaseModel):
    """One persona this bot can run as.

    Neon's `Agent` + `Role` folded into one (llm/models.py: an Agent belongs
    to an Organization and has a Role; a Role carries a system_prompt and a
    tools m2m) - rag_service has no Organization for Role to sit across, so
    there is no reason to keep them as two objects here.

    This deployment still answers chatterloop as exactly ONE bot - one
    handle, one Entity id (`bot_handle`/`bot_entity_id` above) - and exactly
    ONE agent is ever ACTIVE for it, chosen by `active_agent` below. Defining
    several here is what makes swapping personas an env change (redeploy
    with a different `CHATTERLOOP_ACTIVE_AGENT`) rather than a config
    rewrite: a roster this deployment can BE, one at a time - not several
    agents answering the same mention at once. That per-mention routing is a
    different, larger feature this does not build; picking the active one is
    a deploy-time decision, not a runtime one.
    """

    id: str
    name: str = ""
    # APPENDED to the bot's built-in framing (default_system_prompt), never
    # in place of it. That framing is not just personality - it is what
    # tells the model retrieved BACKGROUND is memory, not live instructions,
    # which is the actual defence against a retrieved chunk hijacking the
    # reply. Neon's Role.system_prompt has no such structural text to
    # preserve (Neon has no retrieval-augmented BACKGROUND concept at all),
    # so replacing it outright was safe there. Doing the same here would
    # silently drop that guard the moment an agent defined its own prompt.
    system_prompt: str | None = None
    # None = use CHATTERLOOP_REPLY_MODEL. Unlike Neon, where the CALLER picks
    # a Model per request (messenger/views.py: `model_uuid` on the payload),
    # a passive group-chat bot has no such per-message selection surface, so
    # fixing the model per agent is the adaptation, not a literal mirror.
    model: str | None = None
    # References CHATTERLOOP_TOOLS entries by `name` - there is no separate
    # tool id, `name` already has to be unique (it is the function name the
    # model sees) so a second identifier would only be one more thing to
    # keep in sync.
    tool_ids: list[str] = Field(default_factory=list)
    is_enabled: bool = True


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

    # Whether EVERY message in a DM (single conversation) counts as addressed -
    # no @handle, no reply-threading. A single conversation has exactly two
    # participants, so there is no third party a message could instead be
    # about. Off, a DM is judged exactly like a group: mention, then (if
    # answer_replies) the reply probe - so a plain "hey" in a DM goes
    # unanswered, same as it always did before this existed.
    answer_dms: bool = True

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

    # "openai" | "groq" | "stub". The stub reports what was retrieved
    # instead of composing prose, which keeps retrieval quality debuggable
    # on its own. "openai"/"groq" select a `chatterloop.replies.ChatProvider`
    # by name (see PROVIDERS there) - adding a vendor is registering one more
    # provider and one more literal here, never rewriting the generator.
    reply_generator: Literal["openai", "groq", "stub"] = "stub"
    reply_model: str = "gpt-4o-mini"
    reply_api_key: str = ""
    # OpenAI-compatible endpoint override, e.g.
    # https://api.groq.com/openai/v1 . Empty means api.openai.com.
    reply_base_url: str = ""
    reply_max_tokens: int = 400
    reply_temperature: float = 0.3
    # NeonCentralized_API retries a failed completion 3 times before giving
    # up (messenger/views.py). Same count, same idea - a rate limit or a
    # dropped connection is the provider having a bad moment, not a reason to
    # go silent for the rest of the hour.
    reply_max_retries: int = 3

    # Function-calling. Off by default: a deployment with no tools configured
    # pays no extra request and behaves exactly as before this existed.
    tools_enabled: bool = False
    # JSON array of ToolConfig objects. See ToolConfig's docstring for why
    # this is env, not a database.
    tools: list[ToolConfig] = Field(default_factory=list)
    # Caps how many times the model may chain tool calls before this forces
    # a final answer with tools switched off. Without a cap, a model that
    # keeps asking for "just one more call" turns one mention into an
    # unbounded number of outbound requests.
    tool_max_iterations: int = 2
    tool_timeout_seconds: float = 10.0

    # JSON array of AgentConfig objects. Empty (the default) means this
    # deployment runs exactly as it did before agents existed: the bot's own
    # default persona, CHATTERLOOP_REPLY_MODEL, and the flat CHATTERLOOP_TOOLS
    # list - no migration required to adopt this file's newer settings.
    agents: list[AgentConfig] = Field(default_factory=list)
    # The id of the one AgentConfig this process runs as. Required (and
    # validated below) whenever `agents` is non-empty; meaningless, and
    # rejected, when it is not - an unused setting that happens to be set is
    # usually a deployment pointing at an agent it thinks is there and isn't.
    active_agent: str = ""

    @field_validator("bot_aliases", "ignore_entity_ids", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [part.strip() for part in v.split(",") if part.strip()]
        return v

    @field_validator("tools", "agents", mode="before")
    @classmethod
    def _parse_json_array(cls, v: object) -> object:
        if isinstance(v, str):
            return json.loads(v) if v.strip() else []
        return v

    @model_validator(mode="after")
    def _resolve_active_agent(self) -> "ChatterloopSettings":
        # Fail at startup, not on the first mention: a typo'd id or a
        # forgotten CHATTERLOOP_ACTIVE_AGENT should be a boot-time error the
        # deploy log shows immediately, not a bot that answers with its
        # default persona forever and nobody notices why.
        tool_names = {t.name for t in self.tools}
        for agent in self.agents:
            unknown = [tid for tid in agent.tool_ids if tid not in tool_names]
            if unknown:
                raise ValueError(
                    f"agent {agent.id!r} references unknown tool id(s) {unknown} - "
                    f"known tools: {sorted(tool_names) or '(none configured)'}"
                )

        if not self.agents:
            if self.active_agent:
                raise ValueError(
                    "CHATTERLOOP_ACTIVE_AGENT is set but CHATTERLOOP_AGENTS is empty"
                )
            return self

        enabled = {a.id: a for a in self.agents if a.is_enabled}
        if not self.active_agent:
            raise ValueError(
                "CHATTERLOOP_AGENTS is configured but CHATTERLOOP_ACTIVE_AGENT is not "
                f"set - choose one of: {sorted(enabled) or '(none enabled)'}"
            )
        if self.active_agent not in enabled:
            raise ValueError(
                f"CHATTERLOOP_ACTIVE_AGENT={self.active_agent!r} is not an enabled "
                f"agent id - choose one of: {sorted(enabled) or '(none enabled)'}"
            )
        return self

    @property
    def active_agent_config(self) -> AgentConfig | None:
        """The one AgentConfig this process runs as, or None with no agents defined."""
        if not self.agents:
            return None
        return next(a for a in self.agents if a.id == self.active_agent)


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
