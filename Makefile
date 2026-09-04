.PHONY: help install test lint up down logs bootstrap worker bot parity clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## create the venv and install with dev extras
	python3 -m venv .venv
	./.venv/bin/pip install -e ".[dev]"

test: ## run the unit suite (no Milvus or network required)
	./.venv/bin/python -m pytest -q

lint: ## ruff check + format check
	./.venv/bin/ruff check rag_service tests
	./.venv/bin/ruff format --check rag_service tests

up: ## start redis, plus milvus+etcd+minio if COMPOSE_PROFILES=self-hosted (.env)
	docker compose up -d

down: ## stop everything, including containers left from a profile you've since switched off
	docker compose down --remove-orphans

logs: ## follow worker logs
	docker compose logs -f worker

bootstrap: ## create the collection and indexes
	./.venv/bin/python -m rag_service.cli bootstrap

worker: ## run the bus worker in the foreground
	./.venv/bin/python -m rag_service

bot: ## run the chatterloop mention bot in the foreground
	./.venv/bin/python -m rag_service.bot_service

parity: ## check the mention regex against a chatterloop checkout
	CHATTERLOOP_SOURCE=$${CHATTERLOOP_SOURCE:-../..} \
		./.venv/bin/python -m pytest tests/test_mention_parity.py -v

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ volumes
