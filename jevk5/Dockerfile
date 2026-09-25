# syntax=docker/dockerfile:1
# JevK5 (Qwen3.5-4B typed-decision model) on V100 (SM70) via its own PyTorch runtime,
# which serves the TypeSafe-style /v1/systemone endpoint.
FROM ubuntu:24.04

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /usr/local/bin/

# gcc + libc headers: Triton (used by transformers' Qwen3.5 layers) compiles launcher stubs at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

# cu126: PyTorch's cu128+ wheels no longer ship SM70 kernels.
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_TORCH_BACKEND=cu126 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

ARG JEVK5_REF=v0.2.2
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.12 /opt/venv \
 && uv pip install torch "jevk5 @ git+https://github.com/allebee/jevk5@${JEVK5_REF}"

ENV HOME=/tmp \
    HF_HOME=/data/hf \
    HF_HUB_OFFLINE=1 \
    XDG_CACHE_HOME=/data/cache \
    TRITON_CACHE_DIR=/data/cache/triton \
    TOKENIZERS_PARALLELISM=false

COPY serve.py /usr/local/bin/serve.py
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
ENTRYPOINT ["python", "/usr/local/bin/serve.py"]
