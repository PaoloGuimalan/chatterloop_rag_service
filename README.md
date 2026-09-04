# rag-service

RAG indexing and retrieval for chatterloop systems, backed by **Milvus**.

No HTTP surface. This is a worker: it consumes events from a bus, maintains the
vector index, and answers retrieval requests by publishing results back onto the
bus. Scale it by running more replicas.

```
                    ┌──────────────────────────────────────────┐
  Neon / other      │  rag.events  (Redis Streams | GCP Pub/Sub)│
  chatterloop  ───► │  document.ingest   message.index          │
  services          │  document.delete   conversation.delete    │
                    │  retrieval.request                        │
                    └───────────────────┬──────────────────────┘
                                        │
                            ┌───────────▼───────────┐
                            │  rag-service          │  ← N replicas,
                            │  worker               │    one consumer group
                            └───────┬───────┬───────┘
                                    │       │
                     ┌──────────────▼─┐   ┌─▼──────────────┐
                     │ Milvus         │   │ OpenAI         │
                     │ dense + BM25   │   │ embeddings     │
                     └────────────────┘   └────────────────┘
                                    │
                            ┌───────▼───────────┐
                            │  rag.results      │  retrieval.result
                            └───────────────────┘
```

---

## What changed from the Pinecone pipeline

| | Before (`NeonCentralized_API/llm/services/rag.py`) | Now |
|---|---|---|
| Vector store | Pinecone, one shared index | Milvus, `tenant_id` as **partition key** |
| Tenant isolation | **None** — `{"type": "doc"}` matched every org's documents | Enforced in `build_filter()`; no code path produces an unscoped query |
| Retrieval | Dense only, then hosted rerank | **Hybrid** dense + BM25, RRF-fused, then rerank |
| Chunking | 2000 **characters**, 200 overlap | 512 **tokens**, 64 overlap, recursive on semantic boundaries |
| Result shape | Raw metadata dicts — `msg["msg_type"]` raised `KeyError` on doc hits | `RetrievedChunk`, every field always populated |
| Redundancy | None | MMR, with a lexical fallback |
| API keys | Passed per call from `organization.llm_api_key`, over the wire | Service configuration; never in a payload |
| Duplicate work | Celery retries re-embedded everything | `event_id` dedupe with a TTL |
| Failure handling | Retry 3×, then a canned message | Permanent vs transient split, DLQ, delivery ceiling |
| Startup | `CustomerServiceRAG()` at module import — network call per gunicorn worker | Explicit `ensure_collection()` at boot |

---

## The retrieval pipeline

```
  query
    │
    ├─► embed (text-embedding-3-small, L2-normalised)
    │
    ├─► hybrid search, top_k × 6 candidates          ← recall is set here
    │     ├── dense: HNSW / COSINE / ef=128
    │     └── sparse: BM25 / SPARSE_INVERTED_INDEX
    │     └── fused with RRF (k=60)
    │
    ├─► cross-encoder rerank → top_k × 2             ← precision
    │
    ├─► MMR (λ=0.7) → top_k                          ← coverage
    │
    └─► prepend last 4 conversation turns            ← continuity
```

**Why hybrid.** Dense embeddings are weak at exact tokens — order numbers,
SKUs, error codes, surnames — which is most of what a support conversation is
made of. BM25 catches them. Milvus generates the sparse vector server-side from
a `Function` over the `text` field, so lexical vectors are never computed,
shipped, or version-skewed by the client.

**Why RRF.** Cosine similarity and BM25 scores live on incomparable scales.
RRF is rank-based (`score = Σ 1/(k + rank)`), so it needs no normalisation and
no per-corpus weight tuning. `WeightedRanker` is available via
`RETRIEVAL_FUSION=weighted` once you have measured that one signal should
dominate on your data.

**Why overfetch.** Every stage after hybrid search can only reorder or discard.
A passage neither leg surfaces is gone; `RETRIEVAL_FETCH_MULTIPLIER` is the knob
that actually governs answer quality.

---

## Index configuration, and when to change it

| Setting | Default | Change it when |
|---|---|---|
| `MILVUS_DENSE_INDEX_TYPE` | `HNSW` | `DISKANN` past ~10M vectors/node; `IVF_FLAT` if memory-bound and you accept lower recall |
| `MILVUS_HNSW_M` | `24` | Higher (32–48) for better recall at more memory; lower (12–16) to shrink the graph |
| `MILVUS_HNSW_EF_CONSTRUCTION` | `256` | Higher = better graph, slower builds. Build-time only |
| `RETRIEVAL_HNSW_EF` | `128` | Query-time recall/latency dial. Raise first when recall is short |
| `MILVUS_SPARSE_INDEX_ALGO` | `DAAT_MAXSCORE` | `DAAT_WAND` for short keyword queries; MAXSCORE wins on long noisy ones |
| `RETRIEVAL_SPARSE_DROP_RATIO` | `0.2` | Raise to cut latency on long queries, at some recall |
| `CHUNK_MAX_TOKENS` | `512` | Smaller (256) for FAQ-shaped content; larger (1024) for narrative docs |
| `RETRIEVAL_MMR_LAMBDA` | `0.7` | Lower for more diversity, higher for pure relevance |

**Embedding model.** `text-embedding-3-small` at 1536 dimensions, matching what
the Pinecone pipeline used. A worthwhile upgrade: `text-embedding-3-large`
truncated to 1024 via `EMBEDDING_DIM=1024` outperforms 3-small at full width
while costing a third of the storage — the models are Matryoshka-trained, so
truncation is lossy in a controlled way. Vectors are re-normalised after
truncation, which OpenAI requires and which is easy to miss.

The embedding *model* is deliberately service-wide even though API *keys* can be
per-tenant. Mixing models inside one collection puts vectors in incomparable
spaces and silently destroys retrieval quality; `MilvusStore._assert_dim_matches`
turns the dimension half of that mistake into a startup error.

---

## Quickstart

```bash
make install          # venv + dev extras
cp .env.example .env  # set EMBEDDING_API_KEY
make up               # milvus + redis via docker compose
make bootstrap        # create the collection and indexes
make worker           # run the consumer in the foreground
```

**Self-hosted Milvus vs Zilliz Cloud** is one toggle, not a code change —
`MILVUS_DEPLOYMENT` in `.env` (`self_hosted` | `zilliz_cloud`), matching
`CHATTERLOOP_REPLY_GENERATOR`'s vendor-switch pattern above. `pymilvus`
already speaks to both identically through `MILVUS_URI`/`MILVUS_TOKEN`, so
this changes no code path — it only names which one you're pointed at and,
usefully, fails at startup rather than after a mysterious connection error if
you pick `zilliz_cloud` and forget `MILVUS_TOKEN` (Zilliz Cloud always
requires one).

It also drives `make up`: `.env`'s `COMPOSE_PROFILES=self-hosted` is what
actually brings up `etcd`/`minio`/`milvus` (they're tagged with that Compose
profile). Move to `zilliz_cloud` and comment that line out, and `make up`
starts only `redis` — there's nothing local left to run for Milvus once
Zilliz Cloud hosts it. See `.env.example` for both blocks side by side and
`docker-compose.yml`'s header comment for the mechanics.

Smoke test it without writing a producer:

```bash
./.venv/bin/python -m rag_service.cli ingest ./policy.txt --tenant org_1
./.venv/bin/python -m rag_service.cli search "how long do refunds take" --tenant org_1
```

---

## Event contracts

All events share one envelope. `event_id` **must be stable across retries** —
that is what makes deduplication work.

```json
{
  "event_id": "msg:9f3c...",
  "event_type": "message.index",
  "tenant_id": "org_abc",
  "occurred_at": 1767225600000,
  "payload": { }
}
```

| `event_type` | payload |
|---|---|
| `document.ingest` | `document_id`, `text`, `title?`, `meta?` |
| `document.delete` | `document_id` |
| `message.index` | `conversation_id`, `message_id`, `text`, `role?`, `created_at?` |
| `conversation.delete` | `conversation_id` |
| `retrieval.request` | `query`, `conversation_id?`, `top_k?`, `scopes?`, `reply_to?`, `correlation_id?` |

`role` accepts Neon's vocabulary directly — `text` and `reply` map to `user`,
`ai_reply` and `agent` map to `assistant`.

The envelope is strict (`extra="forbid"`) because it is the routing contract;
payloads are lenient (`extra="ignore"`) so producers can add fields without a
lockstep deploy.

### Publishing from Neon

```python
from rag_service.client import RagEventPublisher

rag = RagEventPublisher(settings.BUS_REDIS_URL)

# Replaces index_chat_message_task.delay(...)
rag.index_message(
    tenant_id=str(conversation.organization_id),
    conversation_id=str(conversation.conversation_id),
    message_id=str(message.message_id),
    text=message.content,
    role=message.message_type,       # "text" / "ai_reply" both understood
)

# Replaces rag.retrieve(...)
cid = uuid4().hex
rag.request_retrieval(
    tenant_id=str(conversation.organization_id),
    query=content,
    conversation_id=str(conversation.conversation_id),
    reply_to="rag.results",
    correlation_id=cid,
)
result = rag.await_result("rag.results", cid, timeout_seconds=10)
messages = result["payload"]["messages"] if result else []
```

`messages` is already in chat-completion shape. Note this is request/response
over a queue: `await_result` blocks, which is fine inside a request-scoped
worker but wants a single dispatching reader at volume.

---

## Delivery semantics

Every bus redelivers, so the worker is explicit about it:

- **Parse failure** → straight to the DLQ. A malformed message never becomes
  well-formed.
- **`PermanentError`** (unknown scope, no reply destination) → DLQ, no retries.
- **Anything else** (Milvus timeout, embedding 429) → nack, redelivered later.
- **Delivery ceiling** (`BUS_MAX_DELIVERIES=5`) → DLQ, checked *before* any
  expensive work.
- **Duplicates** → `SET NX EX` on `event_id`. The claim is taken before the
  handler runs and **released on failure**, so a genuine error is still retried.
- **Dedupe store down** → process anyway. Duplicate work is wasteful; dropped
  work is data loss.
- **Crashed worker** → `XAUTOCLAIM` lets a live replica reclaim entries that
  have been pending too long.

Inspect the dead-letter queue:

```bash
redis-cli XRANGE rag.events.dlq - + COUNT 20
```

---

## Testing

```bash
make test    # 109 unit tests, no Milvus and no network
```

Live Milvus tests are separate and **have not been run** — this machine ran out
of disk before the image finished pulling:

```bash
make up
MILVUS_LIVE=1 ./.venv/bin/pytest tests/test_milvus_integration.py -v
```

They cover what unit tests cannot: schema acceptance, the BM25 function
populating `sparse`, hybrid fusion returning from both legs, upsert-not-duplicate,
and — most importantly — that the partition key really isolates tenants. Run
these before trusting the service with real data.

---

## Acting as a user on chatterloop

A second entrypoint (`make bot` / `python -m rag_service.bot_service`) runs
the same pipeline as a **platform participant** rather than a bus consumer. It
subscribes to a bot entity's realtime channel and replies only when
**addressed** — named with an `@handle`, or replied to directly.

```
events_<bot_entity_id>   (SSE from developer_service, bridged off Redis pub/sub)
        │
        ├── messages_list ── from us? ──yes──> silence
        │                       │no
        │                       ├─ mentioner != null ───────────> MENTION
        │                       │
        │                       └─ else: GET .../replies ─────────> REPLY
        │                          (threaded under one of ours?)
        │                                     │
        ├── notifications ── poll BOTH stores ────────────> MENTION / REPLY
        │     (comment mentions, and replies to our comments)
        │                                     v
        │                                  Trigger
        │                                     │
        │                       fetch conversation, index it
        │                                     │
        │                             policy.evaluate()
        │                              │            │
        │                           IGNORE       RESPOND
        │                         (with reason)     │
        │                                           v
        └─────────────────────────────>  retrieve → generate → reply
```

### Three ways to be addressed

A bot that only answers `@handle` cannot hold a conversation — every turn has
to re-address it. So being **replied to** counts as well, and in a **DM**,
where the bot is one of exactly two participants, *every message* does:

| | how it starts | what the bot needs |
|---|---|---|
| **mention** | somebody types `@assistant` | nothing — the frame says so |
| **reply** | somebody replies to a message or comment the bot wrote | one cheap read |
| **DM** | any message, in a single conversation | one cheap read, cached forever after |

This is not a loosening. All three are explicit acts aimed at the bot. Whether
a reply's parent belongs to the bot is decided **server-side from the token's
own entity** — not here, and not by anything a message can claim — and a reply
to somebody *else* in a **group** thread the bot is in is still none of its
business. A DM is different in kind, not just degree: there is no group of
onlookers a message there could instead be small talk between, so requiring an
`@handle` on every turn of a 1:1 would be the exact ceremony the reply rule
above already removed from group threads — just for a case where it never
applied in the first place.

The realtime frame never says what kind of conversation a message landed in
(`messages_list` carries no such field — see `frames.py`), so the DM check
reads it from `GET /v1/conversations/{id}/messages` with `limit=1`, the SAME
endpoint the mention/reply paths already call for history, just asking for the
one field this needs. A conversation's type never changes once created, so
this is cached **per conversation, forever** after the first resolution — a
busy DM pays for this once over its whole lifetime, not once per message. A
group the bot is freshly added to still pays it once, on its first unaddressed
message, for the same reason: there is no way to know in advance.

`CHATTERLOOP_ANSWER_REPLIES=false` and `CHATTERLOOP_ANSWER_DMS=false` each turn
off their own path independently. Both false is mention-only, exactly as the
bot behaved before either existed, and a message that does not name it costs
no read at all.

### What the platform actually gives us

Three findings from `webapp/src/reusables/hooks/sse.ts` and the services behind
it shaped this, and they're worth knowing before changing anything:

**SSE is a browser bridge over Redis pub/sub.** `GET /u/sseNotifications/:token`
does nothing but `listen(events_${entity_id}, res)` and forward frames. A
backend bot subscribes to that channel directly — same events, no long-lived
HTTP connection through the proxies, no JWT in a URL path. The trade-off is that
Redis pub/sub has no replay: frames published while the bot is down are gone.
That's acceptable here because the platform's own database is the authoritative
record — a missed mention is recovered by reading the notification store, not by
replaying a bus.

**`messages_list` already tells us whether we were mentioned.** The `/sendMessage`
handler resolves handles against the conversation's member list and publishes
`isMentioned ? mentioner : null` *per recipient*. So `mentioner != null` is an
authoritative, server-side signal and the bot trusts it rather than re-deriving
it. A frame carrying one needs no read at all to classify.

**But it says nothing about replies, and cannot be made to.** The frame has no
message id and no `replyingTo`, and the publisher is the platform's own Node
route — a human replying to the bot never goes near `developer_service`, so
enriching the frame from this side would cover only the bot's own messages. The
only honest answer is to read, and the read is a purpose-built route
(`GET /v1/conversations/{id}/replies`) returning the messages here that are
threaded under one of **ours**, scoped to the token's entity. Usually empty, two
indexed lookups when it is not — against a full history window on every message
in every conversation, which is what asking the question client-side would cost.

One consequence worth knowing: the probe reads a *window* of recent replies, so
a reply that arrived while the bot was down is picked up on the next frame in
that conversation rather than being lost — the same "the database is the record,
the stream is only a hint" property that already covers missed mentions. It also
means a late answer is possible; the dedupe set and the per-conversation
cooldown bound it.

**But no frame carries content.** `messages_list` has `conversationID`, the
sender and the mentioner — no message text, not even a message id. The webapp
responds by refetching, and the bot has to as well. `notifications` is worse: it
names no subject at all (sse.ts says so in as many words), so comment mentions
are discovered by reading the notification store, where the rows carry
`target_id` (post) and `target_anchor` (comment). That same shapelessness is why
the reply half needed no new frame: one ping already means "go and look", and
there are now two places to look.

**Consequence: indexing is lazy, and that is forced.** There is nothing to index
at the moment a message arrives. The bot fetches recent history when addressed
and indexes that; upserts are keyed on `(tenant, source, chunk_index)` so
re-indexing the same window is idempotent, and the content-hash cache absorbs
the repeats. Eager indexing would be better but needs either content on the
frame or a periodic backfill — both platform changes.

### Mention parsing parity

Three services now parse `@mentions`, and they must agree exactly:

| | |
|---|---|
| `server/reusables/hooks/transformers.js` | `extractMentionUsernames()` |
| `newsfeed/services/comment_mentions.py` | `MENTION_PATTERN` |
| `rag_service/chatterloop/mentions.py` | this service |

The pattern is reproduced character for character, including the quirks: the
leading `(?:^|\s)` that stops `you@example.com` from mentioning `@example`, and
the greedy `.` in the class that makes `"@ana."` capture `"ana."` — which both
platform implementations handle by also emitting the dot-stripped form.

`tests/test_mention_parity.py` pins our pattern against a verbatim copy, and —
given a checkout — reads the regex out of both live source files and compares
the actual characters:

```bash
make parity   # defaults to ../.. - this service is a sibling of server/ and webapp/
```

### Why it stays quiet

Silence is the default; every reason is logged. Beyond being addressed at all:

- **Never replies to itself.** Checked before any fetch, and now before the
  mention check too, since a frame without a mention is no longer free. This
  matters more on the reply path than it ever did on the mention path: the
  bot's own answers are *themselves* replies, threaded under somebody else's
  message, so a bot that read them back would sustain a thread with itself that
  needs no `@handle` at all.
- **Never replies to entities on the ignore list** — put other bots there.
- **Deduplicates** on message/comment id — the object, not the route it arrived
  on, so a message that both mentions the bot and replies to it is answered
  once. The webapp reopens its SSE stream on navigation, so repeats are normal;
  the reply probe adds a second source of them, because it returns a window and
  re-offers what was already answered. Those are dropped *before* the history
  fetch, not after.
- **Won't answer something it can't read.** With the read path unwired, that is
  every message trigger: the bot reads events, decides correctly, and says
  nothing.
- **Per-conversation cooldown and hourly ceiling.** `record_reply` is called
  after a successful send, not on decision, so a broken outbound path cannot
  rate-limit the bot into silence. The ceiling is also what bounds the cost of a
  back-and-forth that no longer needs a handle to continue.

### Replies are threaded

The bot answers as a **threaded reply to the message that addressed it** —
`isReply: true`, `replyingTo: <messageID>` on the outgoing `/sendMessage` JWT.
`replyingTo` is a bare message id string, matching how the webapp arms a reply
(`setisReplying({isReply: true, replyingTo: cnvs.messageID})` in
`messenger/partials/ContentHandler.tsx`) and renders the quoted preview
(`messageID == isReplying.replyingTo` in `ConversationV2.tsx`). The Mongo column
is typed `Mixed`, so a wrong shape would not be rejected — it would just render
as an empty quote.

Threading matters more here than it does for a person. The bot is slow by
construction — fetch history, index it, retrieve, generate — so by the time it
speaks, other messages have usually landed. An unthreaded answer arrives
detached from its question.

The target is the most recent message **from the entity named in the frame**
that actually addresses the bot, scanning backwards and skipping the bot's own
messages. Not simply the newest message in the window: between the mention and
the fetch completing, anyone may have spoken. The reply path narrows the same
way — the probe may legitimately return replies from several people, and
answering one belonging to a different frame would thread the answer under the
wrong turn.

### Tenancy

One conversation, one Milvus partition (`conv:<conversationID>`). That is
stricter than the platform's own boundary — a realm's members can all read a
realm channel — but `messages_list` carries no realm *id*, only a `realmName`
string on the mentioner. A display name is not an identifier: two realms can
share one, and a rename would silently repartition history. Conversation scope
is the only boundary drawable from the data on the wire, and it errs safe.
Widening to realm scope needs `realm_id` on the frame.

### Wired to chatterloop's developer API

The pipeline reaches the platform only over HTTP, against **one origin with one
credential** — `developer_service`. It holds no database credentials, no Redis
credentials, and the dependency set contains no database drivers, so it cannot
touch chatterloop's stores even by mistake.

| | endpoint | scope |
|---|---|---|
| realtime frames | `GET /v1/events` (SSE) | `events.subscribe` |
| read messages | `GET /v1/conversations/{id}/messages` | `messages.read` |
| read replies to us | `GET /v1/conversations/{id}/replies` | `messages.read` |
| read comment mentions | `GET /v1/mentions/comments` | `notifications.read` |
| read comment replies | `GET /v1/comments/replies` | `notifications.read` |
| send a reply | `POST /v1/messages/send` | `messages.send` |
| post a comment | `POST /v1/comments` | `comments.create` |

Authorization is an **intersection**: a request is allowed only if the scope is
on the token *and* the owning entity has been explicitly granted the
permission. A leaked token can never exceed its entity, and revoking the grant
narrows every token that entity owns with no token edit.

Two properties are worth naming, because both used to be this service's problem
and are now structurally impossible:

- **It cannot read anyone else's data.** `fetch_comment_mentions` takes no
  entity id and the event stream takes no channel — both resolve to whoever the
  token belongs to, server-side. Reading someone else's notifications is not
  something this code can express. The same holds for the two reply reads,
  where the temptation is strongest: `fetch_replies_to_me` takes a conversation
  and a limit and nothing else, because "replies to me" one argument away from
  "replies to anyone" is not a boundary at all.
- **It cannot reach a conversation it is not in.** The messages endpoint
  refuses a conversation the entity is not a participant of, and returns 404
  rather than 403 so an outsider cannot tell an existing conversation from one
  that never existed.

Two things stay deliberately unconnected:

| | default | effect |
|---|---|---|
| `PLATFORM_ENABLED=false` | `Null*Fetcher` / `RecordingResponder` | connects to the bus, gates correctly, generates what it *would* say, records it |
| no `PLATFORM_TOKEN` | same | the client refuses to construct rather than failing per request |

All four paths now work end to end. Commenting used to be the half that did
not: `reply_to_comment` raised, so a comment trigger was detected, gated,
retrieved for, generated — and then dropped. `POST /v1/comments` closed that,
and it owns the parts that made commenting hard to do from a client: two-level
thread flattening, the reply notification, and the mention fan-out.

The bot passes the comment it is answering as `parentID` and does **not** try to
predict where the row will be stored. Replying to a reply re-parents to the
top-level ancestor server-side, and a client computing that itself would be
reimplementing the rule it is calling the endpoint to avoid.

One side effect the endpoint does not perform: a hashtag in a comment does not
tag the parent post. Reproducing it would mean widening the interest taxonomy
from a fifth implementation of a normaliser whose failure mode is a silent
duplicate row — see the `developer_service` README.

### Generating the reply: vendor-switchable, tool-calling, retried

`CHATTERLOOP_REPLY_GENERATOR` picks what turns retrieved context into prose:

| value | what runs |
|---|---|
| `stub` (default) | reports what was retrieved instead of composing anything — keeps retrieval quality debuggable separately from generation quality, and needs no key |
| `openai` | OpenAI, via the `openai` SDK |
| `groq` | Groq, via the `groq` SDK (`pip install -e ".[groq]"`) |

Both real vendors run through the **same** code —
`chatterloop.replies.ChatCompletionReplyGenerator` — parameterized by a small
`ChatProvider` (which SDK class to build, which of its exception types mean
"try again"). Switching vendors is changing `CHATTERLOOP_REPLY_GENERATOR`;
nothing else in the pipeline changes shape. This mirrors
`NeonCentralized_API`'s `LLMFactory.create(service, api_key, model)`
(`llm/services/llm_factory.py`) with one difference on purpose: Neon
hand-duplicates the completion/tool-loop logic once per vendor class
(`GroqService`, `OpenAIService`), so a fix made in one has to be remembered in
the other. Here that logic is written once; a new vendor is one `ChatProvider`
instance registered in `PROVIDERS`, not a new class.

Both `openai` and `groq` share the same failure handling. A completion that
fails with a connection error, a timeout, a rate limit, or a vendor 5xx is
retried up to `CHATTERLOOP_REPLY_MAX_RETRIES` times (default 3, matching
Neon's own retry count) with exponential backoff; a rejected key or a bad
request is never retried, since it will be exactly as rejected on the third
attempt. If every attempt fails, the bot logs the failure and **stays
quiet** on that one mention rather than crashing the process or posting a
public "sorry, something went wrong" — this bot is a member of the group
chat, not a support desk (see `default_system_prompt`), and an apology for
every transient hiccup reads as broken rather than busy. That is a deliberate
departure from Neon, which does post the apology.

#### Function-calling (`CHATTERLOOP_TOOLS`)

Set `CHATTERLOOP_TOOLS_ENABLED=true` and `CHATTERLOOP_TOOLS` (a JSON array) to
let the model call out to an HTTP endpoint mid-reply — the same capability
Neon's `Tool`/`trigger_function` gives its agents
(`llm/models.py`, `llm/utils/function_calls.py`), speaking the current OpenAI
`tools`/`tool_choice` schema rather than the deprecated `functions` one Neon's
code uses.

The load-bearing difference from Neon: **tools are defined in this
deployment's own environment, never a database.** rag_service is a standalone
developer-API product sitting in front of chatterloop through one
token-authenticated credential — it has no standing to ask the chatterloop
system to store or serve config on its behalf, the way an in-house Django app
can reach into its own Postgres for a `Tool` row. Every tool this bot can call
is therefore "environment-bounded" exactly like every other credential and
endpoint in this file already is: per deployment, in `.env`, dynamic without a
migration.

```
CHATTERLOOP_TOOLS=[{
  "name": "get_weather",
  "description": "Current weather for a city.",
  "parameters_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
  "api_endpoint": "https://api.example.com/weather",
  "http_method": "GET",
  "param_type": "query"
}]
```

Three `param_type`s, matching Neon's `Tool.param_type` exactly: `query` (GET
params), `route` (`.format(**arguments)` into the URL, e.g.
`".../orders/{order_id}"`), `body` (POST JSON). `headers` are static only —
never templated from the model's own arguments, so a tool cannot be tricked
by a prompt injection into forging its own auth header from LLM-controlled
input.

A tool call never raises. A bad route template, a network failure, a
non-2xx response — every failure mode becomes `{"error": "..."}`, handed back
to the model as a normal tool result, so it can apologise or try different
arguments instead of a reply crashing outright.

`CHATTERLOOP_TOOL_MAX_ITERATIONS` (default 2) caps how many times the model
may chain tool calls before the pipeline forces a final answer with tools
switched off — without a cap, a model that keeps asking for "just one more
call" turns one mention into an unbounded number of outbound requests.

#### Agents (`CHATTERLOOP_AGENTS`) — one bot, several personas it could be

This deployment still answers chatterloop as **exactly one bot** — one
`CHATTERLOOP_BOT_HANDLE`, one `CHATTERLOOP_BOT_ENTITY_ID`. `CHATTERLOOP_AGENTS`
does not change that; it does not make several agents answer one mention,
and it is not a router. What it is: a roster of personas (`config.AgentConfig`
— system prompt, model override, a subset of `CHATTERLOOP_TOOLS`) this
deployment could run as, with `CHATTERLOOP_ACTIVE_AGENT` picking exactly one
at deploy time. Swapping persona is changing that one value and redeploying,
not rewriting `CHATTERLOOP_TOOLS`/`CHATTERLOOP_REPLY_MODEL`/the prompt by
hand. That is the sense in which several agents are defined here at once —
a bench, not a committee.

This is Neon's `Agent` + `Role` folded into one (`llm/models.py`: an `Agent`
belongs to an `Organization` and has a `Role`; a `Role` carries the
`system_prompt` and a tools relation) — there is no `Organization` here for
`Role` to sit across, so no reason to keep them as two objects. What is
**not** mirrored from Neon on purpose: an agent's `system_prompt` is
**appended** to the bot's built-in framing (`default_system_prompt`), never
substituted for it. That framing is not just personality — it is the
instruction that tells the model retrieved BACKGROUND is memory, not live
instructions, which is the actual defence against a retrieved chunk
hijacking the reply. Neon's `Role.system_prompt` carries no such structural
text to preserve (Neon has no retrieval-augmented BACKGROUND concept at
all), so replacing it outright was safe there. Doing the same here would
silently drop that guard the moment an agent defined its own prompt.

Leave `CHATTERLOOP_AGENTS` empty (the default) and nothing changes: the bot
runs on its own default persona, `CHATTERLOOP_REPLY_MODEL`, and the flat
`CHATTERLOOP_TOOLS` list, exactly as before this existed.

```
CHATTERLOOP_AGENTS=[
  {"id": "support", "name": "Support", "system_prompt": "Be extra concise and practical.", "tool_ids": ["get_weather"]},
  {"id": "sales", "name": "Sales", "model": "gpt-4o"}
]
CHATTERLOOP_ACTIVE_AGENT=support
```

An agent's `tool_ids` reference `CHATTERLOOP_TOOLS` entries by `name` — there
is no separate tool id; `name` already has to be unique, since it is the
function name the model sees. Startup fails loudly, not the first mention,
if: `CHATTERLOOP_AGENTS` is set with no `CHATTERLOOP_ACTIVE_AGENT`;
`CHATTERLOOP_ACTIVE_AGENT` names an id that is missing, misspelled, or
disabled; or any agent (active or not) references a `tool_ids` entry
`CHATTERLOOP_TOOLS` does not define.

`CHATTERLOOP_TOOLS_ENABLED=false` still wins over everything, active agent
included — it is the one lever that kills function-calling deployment-wide
without touching `CHATTERLOOP_AGENTS` or `CHATTERLOOP_TOOLS` at all.

---

## Migrating from Pinecone

There is no vector migration path worth taking: the collections differ in
dimension handling, metadata shape, and tenancy model, and re-embedding is
cheap next to the cost of a subtly wrong index. Replay your source of truth:

1. `make bootstrap`
2. Publish `document.ingest` for every org document.
3. Publish `message.index` for historical messages you want retrievable —
   `Message.objects.filter(...).iterator()` straight into `RagEventPublisher`.
4. Verify per tenant: `cli count --tenant <org>` and a few `cli search` probes.
5. Cut Neon's retrieval over to `retrieval.request`.

Backfill and live traffic can run concurrently; upserts are keyed on
`(tenant_id, source_id, chunk_index)`, so a replay is idempotent.
