from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from pydantic_settings import BaseSettings

from rag_service import config as config_module
from rag_service.chunking import HeuristicTokenizer, TokenChunker
from rag_service.config import ChunkingSettings, EmbeddingSettings


@pytest.fixture(autouse=True)
def isolate_settings_from_dotenv(monkeypatch):
    """Keep the developer's own .env out of the unit suite.

    Every settings class declares `env_file=".env"`, which pydantic-settings
    resolves against the CWD at instantiation - so with a real .env present,
    `EmbeddingSettings(model="text-embedding-3-large")` silently inherits that
    file's EMBEDDING_DIM and the assertion about native dimensionality fails.

    This is not hypothetical tidiness. The README tells you to `cp .env.example
    .env`, so the suite passed only for developers who had not yet done the
    thing the README asks for first, and broke the moment anyone configured the
    service. Autouse because the failure is silent and applies to every test
    that constructs settings without naming every field.
    """
    for name in dir(config_module):
        candidate = getattr(config_module, name)
        if isinstance(candidate, type) and issubclass(candidate, BaseSettings):
            monkeypatch.setitem(candidate.model_config, "env_file", None)


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
