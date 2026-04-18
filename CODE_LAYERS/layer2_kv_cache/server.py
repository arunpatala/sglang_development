"""
CODE_LAYERS / Layer 2 — Manual Decode Loop WITH KV Cache
=========================================================
Only model.py changes vs Layer 1. This file is identical except for the import
and the port/title. That is the point — KV cache is a model-level concern.

What changes vs Layer 1:
  - model.py: past_key_values reuse → O(1) attention per decode step
  - server.py: import KVCacheModel instead of NaiveModel (and port 8102)

What stays the same:
  - /generate API shape — benchmark.py results are directly comparable
  - Sequential handling

Run:
    python server.py
    python server.py --model Qwen/Qwen3-0.6B --port 8102
"""

import argparse
import logging

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from model import KVCacheModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Layer 2 — KV cache decode server")
parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model path or HF hub id")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8102)
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

kv_model = KVCacheModel(args.model)

# ---------------------------------------------------------------------------
# Request / Response schema  (identical to Layer 1)
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    messages: list[Message]
    max_new_tokens: int = 64
    temperature: float = 1.0

class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    ttft_ms: float
    tpot_ms: float

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

app = FastAPI(title="Layer 2 — KV Cache Decode")


@app.get("/health")
def health():
    return {"status": "ok", "layer": 2}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    last_content = req.messages[-1].content if req.messages else ""
    logger.info(f"START: {last_content[:60]!r}")

    messages = [m.model_dump() for m in req.messages]
    result = kv_model.generate(
        messages=messages,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )

    return GenerateResponse(**result)


@app.get("/stats")
def stats():
    import torch
    return {
        "gpu_memory_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"Starting Layer 2 server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
