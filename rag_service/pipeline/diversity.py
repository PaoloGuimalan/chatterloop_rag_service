"""Result diversification.

Support corpora are pathologically redundant: the same policy is restated in a
doc, quoted by an agent, and paraphrased across five tickets. Pure relevance
ranking happily spends the whole context budget on eight versions of one fact.
Maximal Marginal Relevance trades a little relevance for coverage, which is what
actually improves answers.

    MMR(d) = lambda * sim(q, d) - (1 - lambda) * max sim(d, s) for s already selected

lambda = 1.0 is plain relevance; 0.0 is pure novelty. 0.7 keeps relevance in
charge while still breaking up duplicate runs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..domain import RetrievedChunk

_WORD = re.compile(r"[a-z0-9']+")


def mmr_select(
    query_vector: Sequence[float],
    chunks: list[RetrievedChunk],
    top_k: int,
    lambda_mult: float = 0.7,
) -> list[RetrievedChunk]:
    """Greedy MMR over candidates carrying dense vectors.

    Falls back to lexical near-duplicate suppression when vectors weren't
    fetched, so the caller gets diversification either way.
    """
    if top_k <= 0 or not chunks:
        return []
    if len(chunks) <= top_k:
        return chunks
    if any(c.dense is None for c in chunks):
        return dedupe_near_duplicates(chunks, top_k)

    import numpy as np

    query = np.asarray(query_vector, dtype=np.float32)
    matrix = np.asarray([c.dense for c in chunks], dtype=np.float32)

    # Vectors are unit-normalised by the embedder, so a dot product *is* cosine
    # similarity. Normalise defensively anyway - a future embedder that forgets
    # would otherwise skew this silently.
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    query = query / (np.linalg.norm(query) + 1e-12)

    relevance = matrix @ query
    pairwise = matrix @ matrix.T

    selected: list[int] = [int(np.argmax(relevance))]
    remaining = set(range(len(chunks))) - set(selected)

    while len(selected) < top_k and remaining:
        candidates = list(remaining)
        redundancy = pairwise[np.ix_(candidates, selected)].max(axis=1)
        scores = lambda_mult * relevance[candidates] - (1.0 - lambda_mult) * redundancy
        winner = candidates[int(np.argmax(scores))]
        selected.append(winner)
        remaining.discard(winner)

    return [chunks[i] for i in selected]


def dedupe_near_duplicates(
    chunks: list[RetrievedChunk],
    top_k: int,
    threshold: float = 0.85,
) -> list[RetrievedChunk]:
    """Vector-free fallback: drop candidates with high token overlap.

    Jaccard over word sets catches verbatim and lightly-edited repeats, which is
    the bulk of chat redundancy, for no extra storage or bandwidth.
    """
    kept: list[RetrievedChunk] = []
    seen: list[set[str]] = []

    for chunk in chunks:
        tokens = set(_WORD.findall(chunk.text.lower()))
        if not tokens:
            continue
        if any(_jaccard(tokens, prev) >= threshold for prev in seen):
            continue
        kept.append(chunk)
        seen.append(tokens)
        if len(kept) >= top_k:
            break

    return kept or chunks[:top_k]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    return intersection / (len(a) + len(b) - intersection)
