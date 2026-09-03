"""Runner for the chatterloop bot.

A second entrypoint alongside the bus worker in service.py. Same pipeline
underneath - same Milvus collection, same embedder, same retrieval - but fed
by the platform's realtime channel instead of the RAG bus, and gated on
mentions.

    python -m rag_service.bot_service
"""

from __future__ import annotations

import logging
import signal

from .chatterloop.bot import ChatterloopBot
from .chatterloop.consumer import EntityEventConsumer
from .chatterloop.identity import BotIdentity
from .chatterloop.policy import AddressedOnlyPolicy
from .chatterloop.ports import (
    NullMentionFetcher,
    NullMessageFetcher,
    RecordingResponder,
)
from .chatterloop.replies import (
    PROVIDERS,
    ReplyGenerator,
    StubReplyGenerator,
    build_reply_generator,
    default_system_prompt,
)
from .chatterloop.single_instance import SingleInstanceLock
from .chunking import TokenChunker, default_tokenizer
from .config import Settings, get_settings
from .embeddings import build_embedder
from .logging_setup import configure_logging
from .pipeline import IngestionPipeline, RetrievalPipeline
from .rerank import build_reranker
from .store import MilvusStore

logger = logging.getLogger(__name__)


def build_generator(settings: Settings) -> ReplyGenerator:
    cfg = settings.chatterloop
    # Any registered ChatProvider name works here unchanged - switching
    # CHATTERLOOP_REPLY_GENERATOR from "openai" to "groq" (or a future
    # vendor added to chatterloop.replies.PROVIDERS) needs no change in this
    # function. Only "stub" is special: it names no provider, it names the
    # absence of one.
    if cfg.reply_generator in PROVIDERS:
        key = cfg.reply_api_key or settings.embedding.api_key
        agent = cfg.active_agent_config

        # Addressed by name: the bot is a named participant, and a reply
        # reads wrong when the model does not know who it is. An active
        # agent's own system_prompt is APPENDED here, never substituted -
        # see AgentConfig's docstring for why that framing has to survive.
        system_prompt = default_system_prompt(cfg.bot_handle)
        if agent and agent.system_prompt:
            system_prompt = f"{system_prompt}\n\n{agent.system_prompt}"

        model = (agent.model if agent else None) or cfg.reply_model

        # tools_enabled is the master switch regardless of an active agent:
        # it lets an operator kill function-calling for a moment (an
        # incident, a bad tool) without editing CHATTERLOOP_AGENTS or
        # CHATTERLOOP_TOOLS. With an agent active, its own tool_ids narrow
        # the flat CHATTERLOOP_TOOLS list to what THIS persona may call -
        # config.py already rejected any id that does not resolve, so the
        # lookup below cannot KeyError.
        if not cfg.tools_enabled:
            tools = None
        elif agent:
            tools_by_name = {t.name: t for t in cfg.tools}
            tools = [tools_by_name[tool_id] for tool_id in agent.tool_ids]
        else:
            tools = cfg.tools

        try:
            return build_reply_generator(
                cfg.reply_generator,
                api_key=key,
                model=model,
                system_prompt=system_prompt,
                max_tokens=cfg.reply_max_tokens,
                temperature=cfg.reply_temperature,
                base_url=cfg.reply_base_url or None,
                max_retries=cfg.reply_max_retries,
                tools=tools,
                tool_max_iterations=cfg.tool_max_iterations,
                tool_timeout_seconds=cfg.tool_timeout_seconds,
            )
        except Exception as exc:
            # The reason goes in the MESSAGE, not an `extra`. This fallback is
            # silent by design - the bot keeps running - so the one line it
            # emits has to carry enough to diagnose it. A plain-text formatter
            # drops `extra`, which turned a one-word NameError into a long hunt.
            logger.error(
                "reply generator unavailable (%s: %s), falling back to "
                "retrieval-only output",
                type(exc).__name__,
                exc,
            )
    return StubReplyGenerator()


class ChatterloopBotService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        cfg = self.settings.chatterloop

        # Fails fast, before Milvus, before the embedder, before anything -
        # a duplicate launch should cost milliseconds, not several seconds of
        # setup work thrown away the moment it turns out a peer is already
        # running. See single_instance.py for WHY this exists: the dedup
        # further down (AddressedOnlyPolicy) cannot protect against a second
        # PROCESS, only a second delivery inside one.
        self._instance_lock = SingleInstanceLock(cfg.bot_entity_id)
        self._instance_lock.acquire()

        self.identity = BotIdentity(
            entity_id=cfg.bot_entity_id,
            handle=cfg.bot_handle,
            aliases=frozenset(cfg.bot_aliases),
        )

        embedder = build_embedder(self.settings.embedding)
        self.store = MilvusStore(self.settings.milvus, dim=embedder.dim)
        chunker = TokenChunker(
            tokenizer=default_tokenizer(self.settings.embedding.model),
            max_tokens=self.settings.chunking.max_tokens,
            overlap_tokens=self.settings.chunking.overlap_tokens,
            min_tokens=self.settings.chunking.min_tokens,
        )

        ingestion = IngestionPipeline(
            embedder, self.store, chunker, self.settings.chunking
        )
        retrieval = RetrievalPipeline(
            embedder,
            self.store,
            build_reranker(self.settings.retrieval),
            self.settings.retrieval,
        )

        policy = AddressedOnlyPolicy(
            identity=self.identity,
            cooldown_seconds=cfg.cooldown_seconds,
            max_replies_per_hour=cfg.max_replies_per_hour,
            ignore_entity_ids=frozenset(cfg.ignore_entity_ids),
        )

        # Reads, the write seam and the client that carries all three. One
        # credential authenticates every one of them.
        self._client = self._build_client()
        self.responder = self._build_responder()
        message_fetcher, mention_fetcher = self._build_fetchers()

        self.bot = ChatterloopBot(
            identity=self.identity,
            policy=policy,
            ingestion=ingestion,
            retrieval=retrieval,
            generator=build_generator(self.settings),
            responder=self.responder,
            message_fetcher=message_fetcher,
            mention_fetcher=mention_fetcher,
            history_window=cfg.history_window,
            top_k=cfg.top_k,
            answer_replies=cfg.answer_replies,
            answer_dms=cfg.answer_dms,
            reply_probe_window=cfg.reply_probe_window,
            only_live_events=cfg.only_live_events,
        )

        # The stream is scoped server-side to whoever the token belongs to,
        # so there is no channel to name here - and no way for this pipeline
        # to subscribe to somebody else's.
        platform_cfg = self.settings.platform
        self.consumer = EntityEventConsumer(
            base_url=platform_cfg.api_base_url, token=platform_cfg.token
        )
        self._stopping = False

    def _build_client(self):
        """The developer API client, or None when no platform is configured.

        Degrading rather than failing is deliberate and unchanged: a bot with
        no platform connection should still start, listen, and log what it
        would answer. That is a useful mode, and it is the one the tests run
        in.
        """
        cfg = self.settings.platform
        if not cfg.enabled:
            logger.info("platform access disabled - listen-only")
            return None

        try:
            from .chatterloop.platform import BotApiClient

            client = BotApiClient(
                token=cfg.token,
                base_url=cfg.api_base_url,
                timeout=cfg.timeout_seconds,
                max_attempts=cfg.max_attempts,
            )
        except Exception as exc:
            logger.error(
                "platform client unavailable, falling back to listen-only",
                extra={"error": str(exc)},
            )
            return None

        # One call at startup, so a wrong or expired token is a loud failure
        # here rather than a bot that silently answers nothing all day. Its
        # failure is not fatal - the fallbacks below still apply - but it is
        # the difference between a diagnosable log line and a mystery.
        try:
            who = client.whoami()
            logger.info(
                "platform access enabled",
                extra={
                    "entity_id": who.get("entity_id"),
                    "handle": who.get("handle"),
                    # scopes sits beside "token" in the response, not inside
                    # it - {"entity_id", "handle", "scopes", "token": {"id",
                    # "name"}}. Reading it off `token` always returned None,
                    # silently defeating the point of this log line: it
                    # exists so a token missing a scope it needs shows up
                    # here, at startup, rather than as an unexplained
                    # "reply could not be delivered" three hours later.
                    # Verified against a real GET /v1/whoami response before
                    # fixing this, not assumed from the client code alone.
                    "scopes": who.get("scopes"),
                },
            )
        except Exception as exc:
            logger.error(
                "bot token could not be verified - reads and replies will fail",
                extra={"error": str(exc)},
            )
        return client

    def _build_fetchers(self):
        """Real fetchers when a client exists, inert stubs otherwise."""
        if self._client is None:
            return NullMessageFetcher(), NullMentionFetcher()

        from .chatterloop.platform import ApiMentionFetcher, ApiMessageFetcher

        # Both read through the same client and the same token. The mention
        # fetcher takes no entity id: the endpoint answers for whoever the
        # token belongs to, so reading someone else's notifications is not
        # something this code can express.
        return ApiMessageFetcher(self._client), ApiMentionFetcher(self._client)

    def _build_responder(self):
        """The real responder when a platform is configured.

        Falls back to RecordingResponder - which fully generates every reply
        and logs it without sending - when there is none. That is still the
        right default for evaluating what the bot would say before giving it
        the ability to say it: set PLATFORM_ENABLED=false and the pipeline runs
        end to end against nothing.
        """
        if self._client is None:
            return RecordingResponder()

        from .chatterloop.platform import HttpResponder

        return HttpResponder(self._client)

    def _active_tool_names(self) -> list[str]:
        """What the running generator can actually call - for the startup log.

        Mirrors `build_generator`'s own narrowing: the active agent's
        `tool_ids`, or the full CHATTERLOOP_TOOLS registry with no agent
        configured. Kept in sync with that function rather than reading the
        generator's private state back out of it.
        """
        cfg = self.settings.chatterloop
        agent = cfg.active_agent_config
        if agent:
            return list(agent.tool_ids)
        return [t.name for t in cfg.tools if t.is_enabled]

    def install_signal_handlers(self) -> None:
        def handle(signum: int, _frame: object) -> None:
            logger.info("shutdown signal received", extra={"signal": signum})
            self.stop()

        signal.signal(signal.SIGTERM, handle)
        signal.signal(signal.SIGINT, handle)

    def stop(self) -> None:
        self._stopping = True
        self.consumer.stop()

    def run(self) -> None:
        if not self.settings.chatterloop.respond_to_mentions_only:
            raise NotImplementedError(
                "respond_to_mentions_only=False is not implemented. Replying to "
                "UNADDRESSED messages is a product decision with a cost and a "
                "blast radius, and it needs its own gating rules before the "
                "flag means anything. Note this is not the switch for answering "
                "a direct reply without an @handle - that is CHATTERLOOP_"
                "ANSWER_REPLIES, it is on by default, and a reply aimed at the "
                "bot is still the bot being addressed."
            )

        self.store.ensure_collection()
        logger.info(
            "chatterloop bot ready",
            extra={
                "entity_id": self.identity.entity_id,
                "handle": self.identity.handle,
                "channel": self.identity.channel,
                "responder": type(self.responder).__name__,
                # The config value, not type(...).__name__: build_reply_generator
                # returns the same ChatCompletionReplyGenerator class for every
                # vendor (that's the point - see chatterloop.replies.ChatProvider),
                # so the class name alone can no longer tell openai from groq.
                "generator": self.settings.chatterloop.reply_generator,
                # None = the flat CHATTERLOOP_TOOLS/CHATTERLOOP_AGENTS setup
                # never adopted the agent layer at all - distinct from "" or
                # an empty string, which would say an agent IS active but
                # happens to have no id, a state config.py already refuses.
                "active_agent": self.settings.chatterloop.active_agent or None,
                # 0 tools is a legitimate, common state - logged anyway so
                # "why didn't it call anything?" starts here, not in a
                # transcript. Reflects what THIS run actually has available -
                # an active agent's own tool_ids, not the full registry.
                "tools_enabled": self.settings.chatterloop.tools_enabled,
                "tools": sorted(self._active_tool_names())
                if self.settings.chatterloop.tools_enabled
                else [],
                "answer_replies": self.bot.answer_replies,
                "only_live_events": self.bot.only_live_events,
                # The watermark every read-resolved candidate is judged
                # against. Logged because "why did it ignore that?" is
                # unanswerable without knowing when the process started.
                "started_at_ms": self.bot.started_at_ms,
            },
        )
        try:
            for envelope in self.consumer.consume():
                self.bot.handle_envelope(envelope)
                if self._stopping:
                    break
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        logger.info("shutting down", extra=self.bot.stats.snapshot())
        # The API client holds no pooled connection to close - urllib opens
        # and closes per request - so the datastore handles that used to be
        # shut down here are simply gone.
        for closable in (self.consumer, self.store, self._instance_lock):
            if closable is None:
                continue
            try:
                closable.close()
            except Exception:  # pragma: no cover
                logger.debug("error during shutdown", exc_info=True)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    service = ChatterloopBotService(settings)
    service.install_signal_handlers()
    service.run()


if __name__ == "__main__":
    main()
