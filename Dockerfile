# --- build stage -----------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY rag_service ./rag_service
RUN pip install --no-cache-dir --prefix=/install .

# --- runtime stage ---------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 rag

COPY --from=builder /install /usr/local
COPY rag_service ./rag_service

# Warm the tiktoken BPE cache at build time. Without this the first document
# after every deploy pays a network download, and an air-gapped runtime silently
# falls back to heuristic token counting.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" || true

USER rag

# No ports: this is a bus consumer, not a server.
CMD ["python", "-m", "rag_service"]
