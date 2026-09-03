"""Should the bot say something?

Answering is the *exception*, not the default. The bot sees every event on its
entity's channel - every message in every conversation it belongs to - and
replies only when it was addressed. Each rule below is a separate reason to
stay silent, evaluated cheapest-first, and each returns a reason string so "why
didn't it answer?" is answerable from the logs.

THREE WAYS TO BE ADDRESSED, and the second and third are not a loosening of
the first:

  * an @mention, which is how a thread with the bot STARTS in a GROUP;
  * a direct reply to something the bot itself said, which is how one
    CONTINUES in a group. Requiring the handle on every turn is ceremony no
    human conversation has, and its absence was the single thing that made
    the bot read as a command line rather than a participant; and
  * any message at all, in a DM. A single conversation has exactly two
    participants, so there is no third party a message could instead be
    small talk between - every message from the other side is already aimed
    at the bot, the same way every line you type into a 1:1 chat is aimed at
    whoever else is in it. Neither an @mention nor reply-threading means
    anything extra there.

What has NOT changed is that all three are explicit acts aimed at the bot -
being A participant in a DM is what makes "any message" mean "addressed",
not a general loosening of the rule. A reply to somebody else's message in a
GROUP thread the bot happens to be in is still nothing to do with it, and
whether a reply's parent belongs to the bot is decided by the API from the
token's own entity - not here, and not by anything a message can claim.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from .identity import BotIdentity
from .triggers import Trigger, TriggerReason

logger = logging.getLogger(__name__)


class Verdict(StrEnum):
    RESPOND = "respond"
    IGNORE = "ignore"


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: Verdict
    reason: str

    @property
    def should_respond(self) -> bool:
        return self.verdict is Verdict.RESPOND


# Kept distinct so the logs say which rule let a reply through. "Addressed" is
# one word for two quite different situations, and the difference is the first
# thing anyone will want when a reply looks unwarranted.
RESPOND_MENTIONED = Decision(Verdict.RESPOND, "addressed by mention")
RESPOND_REPLIED_TO = Decision(Verdict.RESPOND, "direct reply to the bot")
RESPOND_DM = Decision(Verdict.RESPOND, "message in a direct conversation")

_RESPOND_BY_REASON = {
    TriggerReason.MENTION: RESPOND_MENTIONED,
    TriggerReason.REPLY: RESPOND_REPLIED_TO,
    TriggerReason.DM: RESPOND_DM,
}


class AddressedOnlyPolicy:
    """Reply only when addressed, with loop and flood protection.

    Being addressed - mentioned, or replied to directly - is the product rule.
    The rest is what stops a bot with that rule from still being a problem:
    bots that answer bots, bots that answer themselves, and bots that answer
    the same thing twice because the bus redelivered.
    """

    def __init__(
        self,
        identity: BotIdentity,
        cooldown_seconds: float = 5.0,
        max_replies_per_hour: int = 30,
        dedupe_window: int = 2048,
        ignore_entity_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.identity = identity
        self.cooldown_seconds = cooldown_seconds
        self.max_replies_per_hour = max_replies_per_hour
        # Other bots, or any entity that should never get a reply. Two bots in
        # one realm that both answer mentions will otherwise mention each other
        # until someone notices the bill.
        self.ignore_entity_ids = {str(e) for e in ignore_entity_ids}

        self._seen: deque[str] = deque(maxlen=dedupe_window)
        self._seen_set: set[str] = set()
        self._last_reply_at: dict[str, float] = {}
        self._reply_times: dict[str, deque[float]] = {}

    # ------------------------------------------------------------- decision

    def evaluate(self, trigger: Trigger, now: float | None = None) -> Decision:
        now = time.monotonic() if now is None else now
        scope = trigger.conversation_id or trigger.post_id or "global"

        # 1. Never answer yourself. Cheapest check and the one whose failure is
        #    unbounded - a self-reply is itself a message that mentions
        #    whoever it quotes, which produces another event. It matters more
        #    on the reply path than it ever did on the mention path: the bot's
        #    own answers are threaded under someone else's message, so a bot
        #    that mistook them for input would answer its own thread forever
        #    without a single @handle being typed.
        if self.identity.is_self(trigger.author_entity_id):
            return Decision(Verdict.IGNORE, "author is the bot itself")

        # 2. Never answer an entity on the ignore list (other bots).
        if str(trigger.author_entity_id) in self.ignore_entity_ids:
            return Decision(Verdict.IGNORE, "author is on the ignore list")

        # 3. At-least-once delivery plus a webapp that reconnects its stream on
        #    every navigation means repeats are normal, not exceptional. The
        #    reply probe adds a second source of them: it returns a WINDOW of
        #    recent replies, so every new message in a busy conversation
        #    re-offers the ones already answered.
        if trigger.dedupe_key and trigger.dedupe_key in self._seen_set:
            return Decision(Verdict.IGNORE, "already handled")

        # 4. Something we cannot read is not a question we can answer.
        if not trigger.is_resolved:
            return Decision(Verdict.IGNORE, "trigger text could not be resolved")

        # 5. Per-conversation cooldown. Someone typing "@bot" five times in a
        #    row wants one answer.
        last = self._last_reply_at.get(scope)
        if last is not None and (now - last) < self.cooldown_seconds:
            return Decision(Verdict.IGNORE, "within cooldown for this conversation")

        # 6. Hourly ceiling per conversation - the backstop for anything the
        #    rules above did not anticipate, and the one that bounds the cost
        #    of a back-and-forth that no longer needs a handle to continue.
        if self._recent_reply_count(scope, now) >= self.max_replies_per_hour:
            return Decision(Verdict.IGNORE, "hourly reply limit reached")

        return _RESPOND_BY_REASON[trigger.reason]

    # -------------------------------------------------------------- recording

    def has_seen(self, dedupe_key: str) -> bool:
        """Whether this key has already been judged.

        Public because the reply probe needs it BEFORE building a trigger. The
        probe returns a window, so most of what it offers on any given frame is
        old news; discovering that inside `evaluate` would mean paying for a
        history fetch to resolve something that is about to be ignored.
        """
        return bool(dedupe_key) and dedupe_key in self._seen_set

    def record_seen(self, dedupe_key: str) -> None:
        """Mark a trigger as handled, whatever the verdict was.

        Recorded for ignored triggers too: a redelivery of something already
        judged not worth answering should not be re-judged, and re-judging is
        not free once a fetch is involved.
        """
        if not dedupe_key or dedupe_key in self._seen_set:
            return
        if len(self._seen) == self._seen.maxlen:
            self._seen_set.discard(self._seen[0])
        self._seen.append(dedupe_key)
        self._seen_set.add(dedupe_key)

    def record_reply(self, trigger: Trigger, now: float | None = None) -> None:
        """Called after a reply is actually sent, not when it is decided.

        A reply that failed to send should not consume the conversation's
        budget - otherwise a broken outbound path silently rate-limits the bot
        into silence.
        """
        now = time.monotonic() if now is None else now
        scope = trigger.conversation_id or trigger.post_id or "global"
        self._last_reply_at[scope] = now
        self._reply_times.setdefault(scope, deque()).append(now)

    def _recent_reply_count(self, scope: str, now: float) -> int:
        times = self._reply_times.get(scope)
        if not times:
            return 0
        cutoff = now - 3600.0
        while times and times[0] < cutoff:
            times.popleft()
        return len(times)
