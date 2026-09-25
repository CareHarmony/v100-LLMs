# Shared targets for the vLLM (1Cat SM70 fork) model containers.
# Each model directory has a compose.yaml + .env and includes this file.
# docker compose is the source of truth for how a container runs; these targets
# just wrap it. Override any .env value per command, e.g.:
#   make up GPUS=4,5 TP=2 PORT=8002

-include .env
export GPUS TP PORT MEM_UTIL MAX_MODEL_LEN MAX_NUM_SEQS EXTRA_ARGS IMAGE WHEEL_DIR DTYPE REVISION
export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)

COMPOSE := docker compose
# A model directory can set these before including this file.
READY_PATH ?= /v1/models
SETTINGS   ?= GPUS=$(GPUS) TP=$(or $(TP),<number of GPUS>) PORT=$(PORT) MEM_UTIL=$(MEM_UTIL) MAX_MODEL_LEN=$(MAX_MODEL_LEN)

.DEFAULT_GOAL := help
.PHONY: help build download up down restart logs status wait test config shell clean

help: ## Show targets and current settings
	@grep -hE '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '!seen[$$1]++ {printf "  %-8s %s\n", $$1, $$2}'
	@echo "  settings: $(SETTINGS)"

build: ## Build the image
	$(COMPOSE) build

download: ## Download DOWNLOADS (repo[@revision] ...) to /data/hf; set HF_TOKEN to avoid rate limits
	@docker image inspect $(IMAGE) >/dev/null 2>&1 || { echo "image $(IMAGE) not found; run 'make build' first"; exit 1; }
	@for spec in $(DOWNLOADS); do \
		repo=$${spec%@*}; rev=$${spec#*@}; [ "$$rev" = "$$spec" ] && rev=main; \
		echo "== $$repo @ $$rev"; \
		docker run --rm --user $(HOST_UID):$(HOST_GID) -e HF_HOME=/data/hf $(if $(HF_TOKEN),-e HF_TOKEN) \
			-v /data/hf:/data/hf --entrypoint hf $(IMAGE) \
			download "$$repo" --revision "$$rev" --max-workers 16 >/dev/null || exit $$?; \
		echo "   done"; \
	done

up: ## Start the server on GPUS, host port PORT
	@if [ -z "$$($(COMPOSE) ps -q)" ] && ss -ltnH "sport = :$(PORT)" | grep -q .; then \
		echo "host port $(PORT) is already in use:"; ss -ltnpH "sport = :$(PORT)" 2>/dev/null; \
		echo "pick another with: make up PORT=<port>"; exit 1; fi
	$(COMPOSE) up -d
	@echo "startup takes several minutes -- 'make wait' or 'make logs'"

down: ## Stop and remove the container
	$(COMPOSE) down

restart: down up ## Recreate the container (picks up changed settings)

logs: ## Follow container logs
	$(COMPOSE) logs -f --tail 200

status: ## Show container state and API health
	@$(COMPOSE) ps
	@printf 'GET /health on :%s -> %s\n' "$(PORT)" "$$(curl -s -m5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$(PORT)/health)"

wait: ## Block until the API answers (or the container dies)
	@cid=$$($(COMPOSE) ps -aq); [ -n "$$cid" ] || { echo "not running; 'make up' first"; exit 1; }; \
	until curl -sf -m5 -o /dev/null http://127.0.0.1:$(PORT)$(READY_PATH); do \
		state=$$(docker inspect -f '{{.State.Status}}' $$cid); \
		if [ "$$state" != running ]; then echo "container is '$$state'"; $(COMPOSE) logs --tail 30; exit 1; fi; \
		sleep 10; done; echo "serving on port $(PORT)"

ifndef CUSTOM_TEST
test: ## Send one chat request
	@curl -s http://127.0.0.1:$(PORT)/v1/chat/completions -H 'Content-Type: application/json' \
		-d '{"model":"$(MODEL)","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":800}' \
		| python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"].strip() if "choices" in d else d)'
endif

config: ## Print the compose file with all variables resolved
	$(COMPOSE) config

shell: ## Open a shell in the running container
	$(COMPOSE) exec $(SERVICE) bash

clean: down ## Remove the container and the image
	-docker rmi $(IMAGE)
