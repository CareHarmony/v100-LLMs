#!/usr/bin/env bash
# Builds the vllm serve command from environment variables (set via compose/.env).
set -euo pipefail

# TP defaults to the number of GPUs visible in the container.
: "${TP:=$(nvidia-smi -L | wc -l)}"
: "${MEM_UTIL:=0.90}"          # 0.95 left too little headroom for large prefills
: "${MAX_MODEL_LEN:=262144}"
: "${EXTRA_ARGS:=}"
: "${DRAFT_MODEL:=incoai/Qwen3.8-27B-DFlash2}"
: "${DRAFT_REV:=dedf8df68adfb1afeaf7b7480c0a0243108177b4}"   # revision pinned in the 1Cat README

echo "serving Qwen/Qwen3.8-27B: TP=$TP on $(nvidia-smi -L | wc -l) visible GPU(s), MEM_UTIL=$MEM_UTIL, MAX_MODEL_LEN=$MAX_MODEL_LEN"

# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
exec vllm serve "Qwen/Qwen3.8-27B" \
  --trust-remote-code \
  --tensor-parallel-size "$TP" \
  --dtype half \
  --attention-backend FLASH_ATTN_V100 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$MEM_UTIL" \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_MODEL\",\"revision\":\"$DRAFT_REV\",\"kv_cache_dtype\":\"auto\"}" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 --port 8000 \
  $EXTRA_ARGS "$@"
