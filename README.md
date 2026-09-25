# v100-LLMs

Docker images and Makefiles for serving modern Qwen models on **Tesla V100 (SM70)** GPUs. The chat
models use [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), a vLLM fork that treats Volta as a
first-class target, because upstream vLLM no longer runs them properly on SM70. JevK5, a decision
model, runs on its own PyTorch runtime ([below](#jevk5-typed-decisions)).

| Model | Directory | Default GPUs | Default port | Speculative decoding |
|---|---|---|---|---|
| [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (dense, fp16) | `qwen-27b/` | 4,5,6,7 | 8001 | DFlash2 ([`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)) |
| [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) (~180B MoE, 4-bit experts) | `flash-next/` | 0,1,2,3 | 8000 | MTP (4 tokens) |
| [`alibiserikbay/JevK5`](https://huggingface.co/alibiserikbay/JevK5) (Qwen3.5-4B decision model, fp16) | `jevk5/` | 2 | 8090 | — |

The two Qwen3.8 models expose an OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`,
`/health`). JevK5 serves `/v1/systemone` and `/health` instead.

## Host requirements

- NVIDIA driver with CUDA 12.8+ support (tested: 580.178.04), Docker, and the NVIDIA Container Toolkit.
- A large volume at `/data` (tested: 4 TB). The containers mount, at the same paths:
  - `/data/hf` for Hugging Face weights (Flash-Next ~127 GB, 27B ~56 GB, DFlash2 ~4 GB, JevK5 ~8.4 GB)
  - `/data/vllm` and `/data/cache` for compile caches (they make restarts faster)
- The fork wheel in `/data/wheels` (Qwen3.8 images only). Run `make wheel` to fetch it and check its SHA256.

## Quick start

```sh
make wheel                        # fetch + verify the 1Cat-vLLM 1.5.0 wheel
make build                        # build all images (the two vLLM images share layers, ~17 GB total)
HF_TOKEN=hf_... make download     # fetch weights to /data/hf (a token avoids slow anonymous rate limits)
make -C qwen-27b up && make -C qwen-27b wait && make -C qwen-27b test
```

Run any target across every model with `make <target>` at the top level. For one model, use
`make -C <model> <target>`. Run `make -C <model> help` for the list:

`build` · `download` · `up` · `down` · `restart` · `status` · `wait` · `test` · `logs` · `config` · `shell` · `clean`

## Settings

Each model's `.env` holds its defaults. `docker compose` and the Makefiles both read it. You can
override any value per command:

```sh
make -C qwen-27b up GPUS=4,5 TP=2 PORT=8002
# or plain compose, from inside the model directory:
GPUS=4,5 TP=2 PORT=8002 docker compose up -d
```

| Variable | Meaning |
|---|---|
| `GPUS` | Host GPU indices to use, as `nvidia-smi` numbers them (e.g. `4,5,6,7`) |
| `TP` | Tensor-parallel size. Leave empty to use every GPU in `GPUS` |
| `PORT` | Host port. The container always listens on 8000 inside |
| `MEM_UTIL` | `--gpu-memory-utilization` (default `0.90`, see below) |
| `MAX_MODEL_LEN` | Context length (default `262144`) |
| `MAX_NUM_SEQS` | Flash-Next only: max concurrent sequences (default `4`) |
| `EXTRA_ARGS` | Extra `vllm serve` flags, appended last so they win |
| `HF_TOKEN` | Hugging Face token for `make download` (taken from your environment) |

`make config` prints the compose file with every variable filled in.

JevK5's `.env` has `GPUS`, `PORT`, `DTYPE` and `REVISION` instead (see [JevK5](#jevk5-typed-decisions)).

The containers run as your UID/GID, so files on `/data` stay yours. They restart with
`unless-stopped`. `make up` refuses to start if `PORT` is already in use.

## Measured on 8× V100-SXM2-32GB

Single request, 512 output tokens:

| | Qwen3.8-27B + DFlash2 (TP4) | Flash-Next NVFP4 + MTP4 (TP4) |
|---|---|---|
| Code prompt | ~140 tok/s | ~91 tok/s |
| Essay prompt (forced 512 tokens) | ~74 tok/s | ~67 tok/s |
| Highly predictable text (counting) | ~204 tok/s | — |
| KV cache at `MEM_UTIL=0.90` | 492,604 tokens (1.88× a 262K request) | 289,340 tokens (1.10×) |
| Startup | ~8 min | ~9 min |

Speculative decoding speed depends on the prompt, because it depends on how many draft tokens
get accepted. The fork's headline ≈200–250 tok/s figures for the 27B use a 4-bit target model.
This repo serves the full-precision (fp16) 27B.

## Gotchas (all handled here, listed so you know why)

- **Flash-Next runs at TP 1, 2 or 4 only.** The fork's SM70 NVFP4 MoE kernel hard-codes
  `(1, 2, 4)` (`nvfp4_sm70_moe.py`). TP8 fails at startup, the model's PLE layer rejects PP>1, and
  in-engine DP flattens the MoE to TP=dp×tp (and DP+EP all-to-all is rejected). To use 8 GPUs, run
  two separate TP4 servers. The entrypoint rejects any other TP right away.
- **`MEM_UTIL` defaults to 0.90, not 0.95.** At 0.95 a large prefill (e.g. opencode's system
  prompt) ran out of memory in Flash-Next's PLE short-conv. That memory isn't counted by vLLM's
  startup profiling.
- **No nvcc in the images.** The fork's default GDN (linear-attention) prefill compiles itself at
  runtime with TileLang and needs a CUDA toolkit. `VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL=0` uses the
  wheel's prebuilt SM70 kernel instead.
- **No bf16 on V100**, so both models run with `--dtype half`. The Flash-Next KV cache stays fp16
  (`--kv-cache-dtype auto`), because its sparse attention rejects a quantized KV cache.
- **Flash-Next needs `--language-model-only`.** The fork has not validated the vision tower.
  It also pins ~12 GB of host RAM per rank for the PLE table, hence `memlock=-1`.
- The fork wheel is installed **unmodified**. The build checks it against the release `SHA256SUMS`.

## JevK5 (typed decisions)

[JevK5](https://huggingface.co/alibiserikbay/JevK5) is not a chat model. It reads a state and a
yes/no (`noul`), `choice` or `score` question, and returns a calibrated probability for every
option from one forward pass, with no generated tokens. The image runs the model's own runtime
([allebee/jevk5](https://github.com/allebee/jevk5) v0.2.2) behind its TypeSafe-style endpoint:

```sh
make -C jevk5 up && make -C jevk5 wait && make -C jevk5 test
curl -s localhost:8090/v1/systemone -d '{"state": "Order #7120 shows delivered to No. 17; the customer lives at No. 71.",
  "questions": {"what": {"type": "choice", "instructions": "What happened to the parcel?",
                         "criteria": ["delivered", "misdelivered", "unknown"]}}}'
```

It uses one GPU (~14 GB at fp16) and starts in under a minute. Requests are served one at a time.
Each question takes about 70 ms at ~150 tokens, 150 ms at 300–1,000 tokens and 0.9 s at 2–4k
tokens (the H100 figure in the model card is ~13 ms). Inputs over 16,384 tokens are refused.

- **fp16, not bf16.** V100 has no bf16, and the stock `jevk5-serve` always loads bf16, so
  `serve.py` loads the same runtime with `DTYPE` (default `float16`). On JevBench's 231 public
  items, fp16 and fp32 give the same top answer on all 231 (max |Δp| 0.004), and both match the
  model card's bf16 H100 results on 227. The other 4 are near-ties in bf16 (top option ≤ 0.5).
  fp32 is ~3× slower (p50 238 ms vs 84 ms).
- **cu126 PyTorch.** PyTorch's cu128+ wheels no longer include SM70 kernels.
- **No flash-linear-attention.** Qwen3.5's linear-attention layers use transformers' PyTorch
  fallback. The Triton kernels (`jevk5[fast]`) do run on V100, but they were slower here (p50
  188 ms) and add ~5 minutes of compilation at startup.
- **GPU 2 overlaps Flash-Next's default GPUs (0–3).** JevK5 needs ~14 GB, and Flash-Next leaves
  ~3 GB per GPU at `MEM_UTIL=0.90`. If both run, move one: `make -C jevk5 up GPUS=<free GPU>`.
- The weights are pinned to a commit (`REVISION`), and the container runs with `HF_HUB_OFFLINE=1`.
  Update `REVISION` in `.env` and `DOWNLOADS` in the Makefile together.

## Using with opencode

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "vllm-27b": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8001/v1", "apiKey": "none" },
      "models": {
        "Qwen/Qwen3.8-27B": { "tool_call": true, "reasoning": true, "limit": { "context": 262144, "output": 32768 } }
      }
    }
  }
}
```

Add Flash-Next the same way on its port, with the model id `RadixArk/Qwen3.8-Flash-Next-NVFP4`.
