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
from .chatterloop.policy import MentionOnlyPolicy
from .chatterloop.ports import (
    NullMentionFetcher,
    NullMessageFetcher,
    RecordingResponder,
)
from .chatterloop.replies import (
    OpenAIReplyGenerator,
    ReplyGenerator,
    StubReplyGenerator,
)
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
    if cfg.reply_generator == "openai":
        key = cfg.reply_api_key or settings.embedding.api_key
        try:
            return OpenAIReplyGenerator(
                api_key=key,
                model=cfg.reply_model,
                max_tokens=cfg.reply_max_tokens,
                temperature=cfg.reply_temperature,
            )
        except Exception as exc:
            logger.error(
                "reply generator unavailable, falling back to retrieval-only output",
                extra={"error": str(exc)},
            )
    return StubReplyGenerator()


class ChatterloopBotService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        cfg = self.settings.chatterloop

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

        policy = MentionOnlyPolicy(
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
                    "scopes": (who.get("token") or {}).get("scopes"),
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
                "unaddressed messages is a product decision with a cost and a "
                "blast radius, and it needs its own gating rules before the "
                "flag means anything."
            )

        self.store.ensure_collection()
        logger.info(
            "chatterloop bot ready",
            extra={
                "entity_id": self.identity.entity_id,
                "handle": self.identity.handle,
                "channel": self.identity.channel,
                "responder": type(self.responder).__name__,
                "generator": type(self.bot.generator).__name__,
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
        for closable in (self.consumer, self.store):
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
