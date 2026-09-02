"""Drift detector for the mention regex.

Three services parse @mentions with what is supposed to be one pattern:

  * server/reusables/hooks/transformers.js       (Node, messages)
  * newsfeed/services/comment_mentions.py        (Django, comments)
  * rag_service/chatterloop/mentions.py      (this service)

A mention one of them counts and another does not is a bug in whichever
drifted. The first test below always runs and pins our pattern against a
verbatim copy. The second reads the *live* platform sources when they are
available and compares the actual characters, which is what catches a change
made over there.

Point it at a checkout:

    CHATTERLOOP_SOURCE=~/Documents/Projects/chatterloop pytest tests/test_mention_parity.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from rag_service.chatterloop.mentions import MENTION_PATTERN

# Copied verbatim from both platform implementations at the time of writing.
PLATFORM_PATTERN_SOURCE = r"(?:^|\s)@([A-Za-z0-9._-]{1,30})(?=$|\s|[.,!?;:])"

CASES = [
    "hey @ana can you look",
    "@ana hello",
    "@ana and @ben",
    "@Ana @ana",
    "write to you@example.com",
    "thanks @ana.",
    "hi @ana, bye",
    "hi @ana! bye",
    "@a.b_c-d ok",
    "@" + "a" * 31 + " hi",
    "@" + "a" * 30 + " hi",
    "hey @ there",
    "@assistant what did we decide?",
    "x@y @z",
    "  @ana",
    "@ana@ben",
    "tab\they @ana",
    "@ana\n@ben",
    "email:foo@bar.com and @real",
    "@ana; @ben: @cara?",
    "...@ana",
    "(@ana)",
    "@ana-b_c.d!",
    "@1234",
    "@_",
]


class TestPinnedPattern:
    def test_pattern_source_is_identical(self):
        assert MENTION_PATTERN.pattern == PLATFORM_PATTERN_SOURCE

    @pytest.mark.parametrize("text", CASES)
    def test_same_matches_as_the_platform(self, text):
        platform = re.compile(PLATFORM_PATTERN_SOURCE)
        assert MENTION_PATTERN.findall(text) == platform.findall(text)


def _platform_root() -> Path | None:
    raw = os.getenv("CHATTERLOOP_SOURCE")
    if not raw:
        return None
    root = Path(raw).expanduser()
    return root if root.is_dir() else None


live = pytest.mark.skipif(
    _platform_root() is None, reason="set CHATTERLOOP_SOURCE to a chatterloop checkout"
)


@live
class TestAgainstLiveSources:
    def test_node_regex_still_matches_ours(self):
        root = _platform_root()
        assert root is not None
        source = (root / "server/reusables/hooks/transformers.js").read_text()
        found = re.search(r"text\.matchAll\(\s*/(.+?)/g\s*\)", source, re.S)
        assert found, "extractMentionUsernames regex not found - the JS side moved"
        assert found.group(1) == PLATFORM_PATTERN_SOURCE

    def test_django_regex_still_matches_ours(self):
        root = _platform_root()
        assert root is not None
        source = (
            root / "services/user_service/newsfeed/services/comment_mentions.py"
        ).read_text()
        found = re.search(r'MENTION_PATTERN = re\.compile\(r"(.+?)"\)', source)
        assert found, "MENTION_PATTERN not found - the Django side moved"
        assert found.group(1) == PLATFORM_PATTERN_SOURCE
