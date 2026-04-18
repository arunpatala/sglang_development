"""
CODE_LAYERS / Layer 1 — Manual Decode Loop, No KV Cache
=========================================================
Same computation as Layer 0, but the autoregressive loop is written by us
in model.py instead of being hidden inside HuggingFace's model.generate().

What changes vs Layer 0:
  - model.py owns model loading + the decode loop
  - server.py is a thin HTTP wrapper — no inference logic here

What stays the same:
  - use_cache=False  →  same O(seq²) cost, same throughput numbers
  - /generate API shape is identical  →  benchmark.py is directly comparable
  - Sequential handling (one request at a time)

What this enables:
  - Layer 2 will change only model.py to add KV cache (past_key_values)
  - server.py will not change at all between Layer 1 and Layer 2

Run:
    python server.py
    python server.py --model Qwen/Qwen3-0.6B --port 8101
"""

import argparse
import logging

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from model import NaiveModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Layer 1 — manual decode server")
parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model path or HF hub id")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8101)
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

naive_model = NaiveModel(args.model)

# ---------------------------------------------------------------------------
# Request / Response schema  (identical to Layer 0)
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
    ttft_ms: float    # Time To First Token  — includes the prefill forward pass
    tpot_ms: float    # Time Per Output Token — avg of decode steps 1..N

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

app = FastAPI(title="Layer 1 — Manual Decode, No KV Cache")


@app.get("/health")
def health():
    return {"status": "ok", "layer": 1}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    last_content = req.messages[-1].content if req.messages else ""
    logger.info(f"START: {last_content[:60]!r}")

    messages = [m.model_dump() for m in req.messages]
    result = naive_model.generate(
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
    logger.info(f"Starting Layer 1 server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
