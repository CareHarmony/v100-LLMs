# syntax=docker/dockerfile:1
# Qwen3.8-27B + DFlash2 speculative decoding on V100 (SM70) via the 1Cat-vLLM fork.
# Build with the fork wheel directory as a named context (compose/Makefile do this):
#   docker build --build-context wheels=/data/wheels .
FROM ubuntu:24.04

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /usr/local/bin/

# gcc + libc headers: Triton compiles its small launcher stubs at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_TORCH_BACKEND=cu128 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

ARG WHEEL=1cat_vllm-1.5.0-cp312-cp312-linux_x86_64.whl
RUN --mount=from=wheels,target=/wheels \
    --mount=type=cache,target=/root/.cache/uv \
    cd /wheels && grep " ${WHEEL}\$" SHA256SUMS | sha256sum -c - \
 && uv venv --python 3.12 /opt/venv \
 && uv pip install /wheels/${WHEEL}

# Weights and caches live on the host's /data volume, mounted at the same paths.
# No nvcc in the image (or on the host): use the wheel's prebuilt SM70 GDN prefill
# kernel instead of the TileLang JIT path.
ENV HOME=/tmp \
    HF_HOME=/data/hf \
    VLLM_CACHE_ROOT=/data/vllm \
    XDG_CACHE_HOME=/data/cache \
    TRITON_CACHE_DIR=/data/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/data/cache/torchinductor \
    FLASHINFER_WORKSPACE_BASE=/data/cache \
    VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL=0

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
