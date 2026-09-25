# Run a target in every model directory, e.g. 'make build', 'make up', 'make status'.
# Per-model: 'make -C qwen-27b up GPUS=4,5 TP=2' (see each directory's .env).
MODELS  := qwen-27b flash-next jevk5
TARGETS := build download up down restart status wait test config clean

# 1Cat-vLLM (SM70 fork) release wheel, installed unmodified into both images.
WHEEL_DIR     ?= /data/wheels
WHEEL_VERSION := 1.5.0
WHEEL_URL     := https://github.com/1CatAI/1Cat-vLLM/releases/download/v$(WHEEL_VERSION)
WHEEL         := 1cat_vllm-$(WHEEL_VERSION)-cp312-cp312-linux_x86_64.whl

.PHONY: $(TARGETS) help wheel
help:
	@echo "targets: wheel $(TARGETS)  (run in: $(MODELS))"
	@echo "per model: make -C <model> help"

wheel: ## Fetch the fork wheel into WHEEL_DIR and verify its SHA256
	mkdir -p $(WHEEL_DIR)
	cd $(WHEEL_DIR) && { [ -f $(WHEEL) ] || curl -fLO $(WHEEL_URL)/$(WHEEL); } && curl -fsLO $(WHEEL_URL)/SHA256SUMS \
		&& grep " $(WHEEL)$$" SHA256SUMS | sha256sum -c -

$(TARGETS):
	@for m in $(MODELS); do echo "== $$m: $@"; $(MAKE) --no-print-directory -C $$m $@ || exit $$?; done
