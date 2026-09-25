"""Serve jevk5-serve's /v1/systemone handler with the model, revision and dtype from the environment.

jevk5-serve always loads bf16, which V100 lacks. Weights come from the pinned snapshot in
HF_HOME (make download fetches it); the container runs offline.
"""

import os
from http.server import ThreadingHTTPServer

import torch
from huggingface_hub import snapshot_download

from jevk5.runtime import JevK5
from jevk5.server import make_handler

MODEL = os.environ.get("MODEL", "alibiserikbay/JevK5")
REVISION = os.environ.get("REVISION") or None
DTYPE = os.environ.get("DTYPE", "float16")

path = snapshot_download(MODEL, revision=REVISION)
print(f"loading {MODEL} @ {REVISION or 'main'} as {DTYPE} on {torch.cuda.get_device_name()}", flush=True)
model = JevK5(path, dtype=getattr(torch, DTYPE))
server = ThreadingHTTPServer(("0.0.0.0", 8000), make_handler(model, MODEL))
print(f"serving {MODEL} on :8000/v1/systemone", flush=True)
server.serve_forever()
