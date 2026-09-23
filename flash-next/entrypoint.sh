#!/usr/bin/env bash
# Builds the vllm serve command from environment variables (set via compose/.env).
# Flags follow the fork's validated Qwen3.8 Flash-Next route (PRs #345/#389/#415,
# docs/design/sm70_qwen38_flash_next_nvfp4.md).
set -euo pipefail

# TP defaults to the number of GPUs visible in the container.
: "${TP:=$(nvidia-smi -L | wc -l)}"
: "${MEM_UTIL:=0.90}"          # 0.95 OOMed in the PLE short-conv on a large (opencode) prefill
: "${MAX_MODEL_LEN:=262144}"
: "${MAX_NUM_SEQS:=4}"
: "${EXTRA_ARGS:=}"

# The fork's SM70 NVFP4 MoE kernel only supports these (nvfp4_sm70_moe.py _SUPPORTED_TP_SIZES);
# PP>1 is rejected by the model's PLE layer and DP flattens the MoE to TP=dp*tp.
case "$TP" in
  1|2|4) ;;
  *) echo "Flash-Next on 1Cat-vLLM supports TP=1, 2 or 4 only (got TP=$TP). Set GPUS to 1, 2 or 4 devices, or set TP." >&2; exit 2 ;;
esac

echo "serving RadixArk/Qwen3.8-Flash-Next-NVFP4: TP=$TP on $(nvidia-smi -L | wc -l) visible GPU(s), MEM_UTIL=$MEM_UTIL, MAX_MODEL_LEN=$MAX_MODEL_LEN"

# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
exec vllm serve "RadixArk/Qwen3.8-Flash-Next-NVFP4" \
  --served-model-name "RadixArk/Qwen3.8-Flash-Next-NVFP4" \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size "$TP" \
  --dtype half \
  --attention-backend FLASH_ATTN_V100 \
  --kv-cache-dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$MEM_UTIL" \
  --max-num-batched-tokens 4096 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 --port 8000 \
  $EXTRA_ARGS "$@"
