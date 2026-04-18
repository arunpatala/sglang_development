"""
CODE_LAYERS / Layer 3 — Static Batching Server
================================================
Adds a /generate_batch endpoint that processes B requests in one GPU pass.
/generate (single request) still works — it calls generate_batch internally.

Run:
    python server.py
    python server.py --model Qwen/Qwen3-0.6B --port 8103
"""

import argparse
import logging

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from model import BatchedKVCacheModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Layer 3 — static batching server")
parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8103)
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

batched_model = BatchedKVCacheModel(args.model)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    messages: list[Message]
    max_new_tokens: int = 128
    temperature: float = 1.0

class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    ttft_ms: float
    tpot_ms: float

class BatchGenerateRequest(BaseModel):
    """
    A batch of independent conversations processed together on the GPU.
    Each element of `batch` is a list of messages for one request.
    """
    batch: list[list[Message]]
    max_new_tokens: int = 128
    temperature: float = 1.0

class BatchGenerateResponse(BaseModel):
    results: list[GenerateResponse]
    batch_size: int
    total_output_tokens: int
    wall_time_ms: float
    output_throughput: float   # tok/s

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

app = FastAPI(title="Layer 3 — Static Batching")


@app.get("/health")
def health():
    return {"status": "ok", "layer": 3}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Single request — delegates to generate_batch with B=1."""
    messages = [m.model_dump() for m in req.messages]
    result = batched_model.generate(
        messages=messages,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    return GenerateResponse(**result)


@app.post("/generate_batch", response_model=BatchGenerateResponse)
def generate_batch(req: BatchGenerateRequest):
    """
    Process a batch of B requests in one GPU pass.
    Returns all B results plus aggregate throughput stats.
    """
    import time
    t0 = time.perf_counter()

    batch_messages = [
        [m.model_dump() for m in conv]
        for conv in req.batch
    ]

    results = batched_model.generate_batch(
        batch_messages=batch_messages,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )

    wall_ms = (time.perf_counter() - t0) * 1000
    total_out = sum(r["completion_tokens"] for r in results)

    return BatchGenerateResponse(
        results=[GenerateResponse(**r) for r in results],
        batch_size=len(results),
        total_output_tokens=total_out,
        wall_time_ms=round(wall_ms, 1),
        output_throughput=round(total_out / (wall_ms / 1000), 1),
    )


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
    logger.info(f"Starting Layer 3 server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
