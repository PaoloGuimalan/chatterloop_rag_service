"""The mutex that stops a second bot process from starting.

What actually motivated this: two replies to the same message, different
pendingIDs, different generated text, sometimes under 100ms apart - proof of
two independent processes racing on one SSE frame, which no amount of
in-process dedup (AddressedOnlyPolicy) can see coming.
"""

from __future__ import annotations

import pytest

from rag_service.chatterloop.single_instance import (
    AlreadyRunning,
    SingleInstanceLock,
    port_for,
)


class TestPortDerivation:
    def test_the_same_entity_id_always_gets_the_same_port(self):
        assert port_for("bot-1") == port_for("bot-1")

    def test_different_entity_ids_usually_get_different_ports(self):
        # Not a proof (a hash can collide), but with a 10,000-wide range two
        # arbitrary ids landing on the same port would be a bad sign either
        # way - this is the case that actually matters: two DIFFERENT bots
        # on one host must not contend for the same lock.
        assert port_for("bot-1") != port_for("bot-2")

    def test_the_port_is_in_the_documented_range(self):
        port = port_for("some-entity-id")
        assert 20000 <= port < 30000


class TestSingleInstanceLock:
    def test_a_second_lock_for_the_same_bot_is_refused(self):
        first = SingleInstanceLock("bot-1", port=0)
        first.acquire()
        try:
            # port=0 means "let the OS pick" for the FIRST bind - re-derive
            # the port it actually got so the second lock contends for the
            # SAME one, the same way two real launches of the same bot would
            # (both computing port_for("bot-1") independently).
            second = SingleInstanceLock("bot-1", port=first._socket.getsockname()[1])
            with pytest.raises(AlreadyRunning, match="bot-1"):
                second.acquire()
        finally:
            first.close()

    def test_the_port_is_free_again_after_closing(self):
        lock = SingleInstanceLock("bot-2", port=0)
        lock.acquire()
        port = lock._socket.getsockname()[1]
        lock.close()

        # If the port were still held, THIS acquire would raise.
        reacquired = SingleInstanceLock("bot-2", port=port)
        reacquired.acquire()
        reacquired.close()

    def test_different_bots_do_not_contend(self):
        a = SingleInstanceLock("bot-a", port=port_for("bot-a"))
        b = SingleInstanceLock("bot-b", port=port_for("bot-b"))
        a.acquire()
        try:
            b.acquire()  # must not raise
            b.close()
        finally:
            a.close()

    def test_closing_twice_is_harmless(self):
        lock = SingleInstanceLock("bot-3", port=0)
        lock.acquire()
        lock.close()
        lock.close()  # must not raise

    def test_closing_without_ever_acquiring_is_harmless(self):
        SingleInstanceLock("bot-4", port=0).close()

    def test_the_failure_message_says_what_to_do(self):
        first = SingleInstanceLock("bot-5", port=0)
        first.acquire()
        try:
            second = SingleInstanceLock("bot-5", port=first._socket.getsockname()[1])
            with pytest.raises(AlreadyRunning, match="[Ss]top that process"):
                second.acquire()
        finally:
            first.close()
