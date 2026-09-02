from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from rag_service.chunking import HeuristicTokenizer, TokenChunker
from rag_service.config import ChunkingSettings, EmbeddingSettings


@pytest.fixture
def tokenizer() -> HeuristicTokenizer:
    # Deterministic and offline: tiktoken downloads its BPE table on first use,
    # which has no place in a unit test.
    return HeuristicTokenizer()


@pytest.fixture
def chunker(tokenizer: HeuristicTokenizer) -> TokenChunker:
    return TokenChunker(tokenizer, max_tokens=50, overlap_tokens=10, min_tokens=5)


@pytest.fixture
def embedding_settings() -> EmbeddingSettings:
    return EmbeddingSettings(
        model="text-embedding-3-small",
        api_key="test-key",
        batch_size=3,
        max_tokens_per_batch=10_000,
        cache_enabled=True,
    )


@pytest.fixture
def chunking_settings() -> ChunkingSettings:
    return ChunkingSettings()
