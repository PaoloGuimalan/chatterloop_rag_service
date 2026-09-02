from __future__ import annotations

from rag_service.domain import RetrievedChunk, Scope
from rag_service.pipeline.diversity import dedupe_near_duplicates, mmr_select


def chunk(cid: str, text: str, score: float = 1.0, dense=None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, text=text, scope=Scope.DOCUMENT, score=score, dense=dense
    )


# A query distinct from every candidate. If the query vector equals a
# candidate, relevance(x) and similarity(x, a) become the same number and the
# MMR score degenerates to (2*lambda - 1) * relevance - which ranks by
# relevance alone and tests nothing.
QUERY = [1.0, 0.0, 0.0]
# relevance 0.90; sits closest to the query
A = [0.90, 0.4359, 0.0]
# relevance 0.88, but similarity to A is 0.999 - a near-duplicate
B = [0.88, 0.4750, 0.0]
# relevance 0.75, similarity to A only 0.675 - genuinely new information
C = [0.75, 0.0, 0.6614]


class TestMMR:
    def test_prefers_a_novel_result_over_a_near_duplicate(self):
        candidates = [
            chunk("a", "most relevant", dense=A),
            chunk("b", "near duplicate of a", dense=B),
            chunk("c", "different angle", dense=C),
        ]
        # b: 0.5*0.88 - 0.5*0.999 = -0.060
        # c: 0.5*0.75 - 0.5*0.675 = +0.038  <- wins on novelty
        picked = mmr_select(QUERY, candidates, top_k=2, lambda_mult=0.5)
        assert [c.chunk_id for c in picked] == ["a", "c"]

    def test_lambda_one_is_pure_relevance(self):
        candidates = [
            chunk("a", "a", dense=A),
            chunk("b", "b", dense=B),
            chunk("c", "c", dense=C),
        ]
        # Redundancy carries zero weight, so 0.88 beats 0.75.
        picked = mmr_select(QUERY, candidates, top_k=2, lambda_mult=1.0)
        assert [c.chunk_id for c in picked] == ["a", "b"]

    def test_lambda_zero_is_pure_novelty(self):
        candidates = [
            chunk("a", "a", dense=A),
            chunk("b", "b", dense=B),
            chunk("c", "c", dense=C),
        ]
        picked = mmr_select(QUERY, candidates, top_k=2, lambda_mult=0.0)
        assert [c.chunk_id for c in picked] == ["a", "c"]

    def test_returns_everything_when_candidates_fit(self):
        candidates = [chunk("a", "a", dense=[1.0, 0.0])]
        assert len(mmr_select([1.0, 0.0], candidates, top_k=5)) == 1

    def test_empty_input(self):
        assert mmr_select([1.0, 0.0], [], top_k=3) == []

    def test_falls_back_to_lexical_dedupe_without_vectors(self):
        candidates = [
            chunk("a", "the refund window is thirty days"),
            chunk("b", "the refund window is thirty days"),
            chunk("c", "shipping takes two business days"),
        ]
        picked = mmr_select([1.0, 0.0], candidates, top_k=2)
        assert [c.chunk_id for c in picked] == ["a", "c"]


class TestLexicalDedupe:
    def test_drops_verbatim_repeats(self):
        candidates = [
            chunk("a", "password reset link expires after one hour"),
            chunk("b", "password reset link expires after one hour"),
            chunk("c", "billing runs on the first of the month"),
        ]
        assert [c.chunk_id for c in dedupe_near_duplicates(candidates, 3)] == ["a", "c"]

    def test_keeps_distinct_text(self):
        candidates = [chunk("a", "alpha beta gamma"), chunk("b", "delta epsilon zeta")]
        assert len(dedupe_near_duplicates(candidates, 2)) == 2

    def test_never_returns_empty_when_input_is_non_empty(self):
        candidates = [chunk("a", "!!!"), chunk("b", "???")]
        assert dedupe_near_duplicates(candidates, 2)
