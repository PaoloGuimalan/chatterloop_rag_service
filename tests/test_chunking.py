from __future__ import annotations

import pytest

from rag_service.chunking import HeuristicTokenizer, TokenChunker


class TestSplitting:
    def test_short_text_is_one_chunk(self, chunker):
        assert chunker.split("A short sentence.") == ["A short sentence."]

    def test_empty_input(self, chunker):
        assert chunker.split("") == []
        assert chunker.split("   \n  ") == []

    def test_long_text_is_split(self, chunker):
        text = " ".join(f"word{i}" for i in range(400))
        chunks = chunker.split(text)
        assert len(chunks) > 1

    def test_no_chunk_greatly_exceeds_the_budget(self, chunker):
        text = ". ".join(f"Sentence number {i} carries some filler text" for i in range(120))
        for chunk in chunker.split(text):
            # Overlap is prepended after the budget check, so allow one
            # overlap's worth of slack.
            assert chunker.count(chunk) <= chunker.max_tokens + chunker.overlap_tokens

    def test_paragraph_boundaries_are_preferred(self, chunker):
        para = "x " * 60
        chunks = chunker.split(f"{para}\n\n{para}\n\n{para}")
        assert len(chunks) >= 3

    def test_unbreakable_run_is_hard_split(self, chunker):
        # No separators at all - the last-resort path.
        chunks = chunker.split("a" * 2000)
        assert len(chunks) > 1
        assert all(c for c in chunks)

    def test_no_content_is_dropped_for_simple_input(self, chunker):
        words = [f"w{i}" for i in range(300)]
        chunks = chunker.split(" ".join(words))
        joined = " ".join(chunks)
        # Overlap means words can repeat, but none may go missing.
        for word in words:
            assert word in joined

    def test_tiny_trailing_fragment_is_folded_back(self, chunker):
        text = " ".join(f"word{i}" for i in range(200)) + "\n\nok"
        chunks = chunker.split(text)
        assert chunks[-1] != "ok"
        assert "ok" in chunks[-1]


class TestOverlap:
    def test_consecutive_chunks_share_context(self):
        chunker = TokenChunker(HeuristicTokenizer(), max_tokens=40, overlap_tokens=12, min_tokens=2)
        chunks = chunker.split(" ".join(f"token{i}" for i in range(200)))
        assert len(chunks) > 2
        # The tail of one chunk should reappear at the head of the next.
        first_tail = chunks[0].split()[-1]
        assert first_tail in chunks[1]

    def test_zero_overlap_is_allowed(self):
        chunker = TokenChunker(HeuristicTokenizer(), max_tokens=40, overlap_tokens=0, min_tokens=2)
        assert len(chunker.split(" ".join(f"t{i}" for i in range(200)))) > 1

    def test_overlap_must_be_smaller_than_the_budget(self):
        with pytest.raises(ValueError):
            TokenChunker(HeuristicTokenizer(), max_tokens=20, overlap_tokens=20)


class TestHeuristicTokenizer:
    def test_counts_scale_with_length(self):
        tok = HeuristicTokenizer()
        assert len(tok.encode("a" * 360)) > len(tok.encode("a" * 36))

    def test_empty_string_is_zero_tokens(self):
        assert HeuristicTokenizer().encode("") == []
