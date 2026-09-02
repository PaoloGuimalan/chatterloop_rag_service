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
subscribes to a bot entity's realtime channel and — initially — replies only
when explicitly mentioned.

```
events_<bot_entity_id>   (Redis pub/sub, the channel SSE is bridged from)
        │
        ├── messages_list ── mentioner != null? ──no──> silence
        │                            │yes
        ├── notifications ───────────┤ (poll the notification store for
        │                            │  type == "comment_mention")
        │                            v
        │                      MentionTrigger
        │                            │
        │                fetch conversation, index it
        │                            │
        │                    policy.evaluate()
        │                     │            │
        │                  IGNORE       RESPOND
        │                (with reason)     │
        │                                  v
        └──────────────────>  retrieve → generate → reply
```

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
it. That is the gate, and nothing downstream of it runs without it.

**But no frame carries content.** `messages_list` has `conversationID`, the
sender and the mentioner — no message text, not even a message id. The webapp
responds by refetching, and the bot has to as well. `notifications` is worse: it
names no subject at all (sse.ts says so in as many words), so comment mentions
are discovered by reading the notification store, where the rows carry
`target_id` (post) and `target_anchor` (comment).

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

Silence is the default; every reason is logged. Beyond the mention requirement:

- **Never replies to itself.** Checked before any fetch. A self-reply is itself
  a message, so this failure is unbounded rather than merely wasteful.
- **Never replies to entities on the ignore list** — put other bots there.
- **Deduplicates** on message/comment id. The webapp reopens its SSE stream on
  navigation, so repeats are normal.
- **Won't answer a mention it can't read.** With the read path unwired, that is
  every message mention: the bot reads events, decides correctly, and says
  nothing.
- **Per-conversation cooldown and hourly ceiling.** `record_reply` is called
  after a successful send, not on decision, so a broken outbound path cannot
  rate-limit the bot into silence.

### Replies are threaded

The bot answers as a **threaded reply to the message that mentioned it** —
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
the fetch completing, anyone may have spoken.

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
| read comment mentions | `GET /v1/mentions/comments` | `notifications.read` |
| send a reply | `POST /v1/messages/send` | `messages.send` |

Authorization is an **intersection**: a request is allowed only if the scope is
on the token *and* the owning entity has been explicitly granted the
permission. A leaked token can never exceed its entity, and revoking the grant
narrows every token that entity owns with no token edit.

Two properties are worth naming, because both used to be this service's problem
and are now structurally impossible:

- **It cannot read anyone else's data.** `fetch_comment_mentions` takes no
  entity id and the event stream takes no channel — both resolve to whoever the
  token belongs to, server-side. Reading someone else's notifications is not
  something this code can express.
- **It cannot reach a conversation it is not in.** The messages endpoint
  refuses a conversation the entity is not a participant of, and returns 404
  rather than 403 so an outsider cannot tell an existing conversation from one
  that never existed.

Two things stay deliberately unconnected:

| | default | effect |
|---|---|---|
| `PLATFORM_ENABLED=false` | `Null*Fetcher` / `RecordingResponder` | connects to the bus, gates correctly, generates what it *would* say, records it |
| no `PLATFORM_TOKEN` | same | the client refuses to construct rather than failing per request |

Replying to a **comment** mention is still unimplemented: comment creation is a
Django newsfeed surface with its own two-level thread flattening and mention
fan-out, and there is no developer-API endpoint for it yet. Message replies
work.

Set `CHATTERLOOP_REPLY_GENERATOR=openai` for real prose; the default `stub`
generator reports what was retrieved instead, which keeps retrieval quality
debuggable separately from generation quality.

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
