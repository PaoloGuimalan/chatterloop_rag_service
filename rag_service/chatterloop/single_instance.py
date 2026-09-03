"""Refuses to let a second instance of the SAME bot start.

WHY THIS EXISTS
----------------
The dedup this pipeline already does (`AddressedOnlyPolicy._seen_set`,
policy.py) lives in one process's memory, deliberately. That is correct for
what it protects against - a redelivered SSE frame, a reply probe re-offering
something already answered - but it cannot protect against a SECOND PROCESS:
two bot instances, each innocent on its own, each independently deciding to
answer the same mention, because neither knows the other exists.

That is not a hypothetical. It is what actually happened: two replies to the
same message, different `pendingID`s, different generated text - proof they
were two SEPARATE resolve-retrieve-generate-send runs, not one retried POST -
sometimes under 100ms apart. That gap rules out a slow human restarting the
bot by hand; it is the fingerprint of two processes racing on the same frame
the instant it arrived. Across a long dev session with many restarts, an old
process left running in a background terminal is an easy, ordinary mistake -
and a bigger dedupe set does not help a process that does not know a peer
exists in the first place.

HOW
---
Binding a TCP socket to 127.0.0.1 on a port derived from the bot's OWN
entity id, held for the life of the process. Binding is atomic at the OS
level - there is no window where two processes both believe they acquired
the lock, which is exactly the race a PID file has ("read the file, see it
looks stale, write my own PID" is three separate operations with a gap
between every pair of them). It needs no new dependency, and works
identically on every platform this ever runs on, dev laptop or VPC alike.

The port is DERIVED, not fixed, so two DIFFERENT bots on the same host never
collide with each other - only two instances of the SAME bot identity do,
which is the one case that is always wrong: this pipeline subscribes to one
entity's whole event stream, so there is no valid reason to ever run two
processes for it, unlike a stateless HTTP service where N replicas is normal.
"""

from __future__ import annotations

import hashlib
import logging
import socket

logger = logging.getLogger(__name__)

# Deliberately unregistered to anything this service or a typical dev
# machine runs, and out of the way of common local dev ports (Milvus, Redis,
# a frontend dev server).
_PORT_RANGE_START = 20000
_PORT_RANGE_SIZE = 10000


def port_for(bot_entity_id: str) -> int:
    """A stable port derived from the bot's own identity.

    Deterministic, not random: the SAME bot identity lands on the SAME port
    every time, on every machine - which is what lets a second instance of
    THAT bot collide with the first at bind time.
    """
    digest = hashlib.blake2b(bot_entity_id.encode("utf-8"), digest_size=2).digest()
    offset = int.from_bytes(digest, "big") % _PORT_RANGE_SIZE
    return _PORT_RANGE_START + offset


class AlreadyRunning(RuntimeError):
    """Another instance of this exact bot already holds the lock."""


class SingleInstanceLock:
    """Held for the life of the process; closing it frees the port."""

    def __init__(self, bot_entity_id: str, port: int | None = None) -> None:
        self.bot_entity_id = bot_entity_id
        self.port = port if port is not None else port_for(bot_entity_id)
        self._socket: socket.socket | None = None

    def acquire(self) -> None:
        """Raises `AlreadyRunning` if another instance already holds this port.

        Called as early as possible in startup - before Milvus, before the
        embedder, before subscribing to anything - so a duplicate launch
        fails in milliseconds with a clear reason, not after several seconds
        of setup work that all gets thrown away.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 127.0.0.1 only: this is a same-machine mutex, not a network
        # service. Binding every interface would needlessly open a port a
        # network scan could find, for no benefit - nothing is ever meant to
        # connect to it.
        try:
            sock.bind(("127.0.0.1", self.port))
        except OSError as exc:
            sock.close()
            raise AlreadyRunning(
                f"another instance of this bot (entity {self.bot_entity_id}) "
                f"appears to already be running - port {self.port}, derived "
                f"from its entity id, is already bound on this machine. "
                f"Stop that process before starting this one: running two at "
                f"once means every mention and reply gets answered twice."
            ) from exc
        # LISTEN, not just bind - the unambiguous "this port is mine" state,
        # not a bound-but-idle socket whose backlog behaviour can differ
        # across platforms and restarts.
        sock.listen(1)
        self._socket = sock
        logger.info(
            "single-instance lock acquired",
            extra={"bot_entity_id": self.bot_entity_id, "port": self.port},
        )

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        except OSError:  # pragma: no cover - best effort on shutdown
            logger.debug("error releasing single-instance lock", exc_info=True)
        self._socket = None
