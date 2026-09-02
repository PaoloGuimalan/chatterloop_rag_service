"""@mention parsing.

Ported character for character from the two implementations already in the
platform:

  * server/reusables/hooks/transformers.js  -> extractMentionUsernames()
  * services/user_service/newsfeed/services/comment_mentions.py

Those two are deliberately identical to each other, and this is the third copy.
A mention this file counts that the messenger does not - or vice versa - is a
bug in whichever drifted, so the pattern is reproduced verbatim rather than
"improved". The parity tests exist to catch drift in any direction.

Note what this file is NOT for. The platform already resolves handle -> entity
server-side and tells us the answer: `messages_list` carries a non-null
`mentioner` only for recipients who were actually mentioned. That is the
authoritative signal and the bot trusts it. This parser is for the things the
event does not answer: which handle was used, and what the message says once
the address is stripped off the front.
"""

from __future__ import annotations

import re

# Identical to MENTION_PATTERN in comment_mentions.py and to the inline regex
# in transformers.js.
#
# The leading (?:^|\s) is what stops "you@example.com" from mentioning
# @example. The character class includes "." and is greedy, so "thanks @ana."
# captures "ana." - the lookahead is already satisfied by end-of-string with
# the dot consumed. Both existing implementations handle that by also emitting
# the dot-stripped form; so does this one.
MENTION_PATTERN = re.compile(r"(?:^|\s)@([A-Za-z0-9._-]{1,30})(?=$|\s|[.,!?;:])")

# Same bound as MAX_MENTIONS_PER_COMMENT. Past this the input is a spam vector.
MAX_MENTIONS = 20


def extract_handles(text: str) -> list[str]:
    """Candidate handles in `text`, lowercased, deduplicated, order preserved.

    A handle ending in "." also yields its stripped form, so "thanks @ana."
    offers both "ana." and "ana". Only one can be a real handle, so resolution
    stays unambiguous.
    """
    if not text:
        return []

    seen: list[str] = []
    for raw_handle in MENTION_PATTERN.findall(text):
        handle = raw_handle.lower()
        for candidate in (handle, handle.rstrip(".")):
            if candidate and candidate not in seen:
                seen.append(candidate)
        if len(seen) >= MAX_MENTIONS:
            break
    return seen


def normalise_handle(handle: str) -> str:
    """Strip a leading @ and lowercase.

    The platform is inconsistent about which form it hands out - `mentioner.
    username` arrives as "@ana" while Account.username is stored bare - so
    every comparison goes through here rather than guessing at the call site.
    """
    return (handle or "").strip().lstrip("@").lower()


def is_addressed_to(text: str, handles: set[str]) -> bool:
    """Whether any of `handles` is mentioned in `text`.

    A fallback for surfaces where the platform has not already resolved the
    mention for us. Prefer the server's own signal where it exists.
    """
    if not text or not handles:
        return False
    wanted = {normalise_handle(h) for h in handles}
    return bool(wanted & set(extract_handles(text)))


def strip_mentions(text: str, handles: set[str]) -> str:
    """Remove the bot's own @handle from the text.

    "@assistant what did we decide about pricing?" is a question about pricing,
    not about the assistant. Leaving the address in skews both the dense
    embedding and the BM25 leg toward the bot's own name, which is the one term
    guaranteed to be irrelevant to the answer.

    Only the addressed handles are removed - mentions of *other* people are
    content and stay.
    """
    if not text:
        return ""
    wanted = {normalise_handle(h) for h in handles if h}
    if not wanted:
        return text.strip()

    def replace(match: re.Match[str]) -> str:
        handle = match.group(1).lower()
        if handle in wanted or handle.rstrip(".") in wanted:
            # Keep the leading separator so neighbouring words don't fuse.
            return match.group(0)[: match.start(1) - match.start(0) - 1]
        return match.group(0)

    stripped = MENTION_PATTERN.sub(replace, text)
    return re.sub(r"\s{2,}", " ", stripped).strip()
