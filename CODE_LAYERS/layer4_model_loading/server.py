"""
CODE_LAYERS / Layer 4 — Custom Model Server
============================================
Identical API to Layer 3.  The only server-visible change is that the
model is loaded via our own Qwen3ForCausalLM instead of AutoModelForCausalLM.

Run:
    python server.py                        # all values from config.yml
    python server.py --port 8200            # override just the port; rest from config.yml
    python server.py --model Qwen/Qwen3-0.6B --port 8104
"""

import argparse
import logging
import sys
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from model_runner import BatchedModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config  —  CLI args  >  config.yml  >  Python defaults
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", default=str(_HERE / "config.yml"))
_pre_args, _ = _pre.parse_known_args()

def _load_yaml(path: str) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

_cfg = _load_yaml(_pre_args.config)

parser = argparse.ArgumentParser(description="Layer 4 — custom model server")
parser.add_argument("--config",    default=str(_HERE / "config.yml"), help="Path to YAML config file")
parser.add_argument("--model",     default=_cfg.get("model",     "Qwen/Qwen3-0.6B"))
parser.add_argument("--host",      default=_cfg.get("host",      "0.0.0.0"))
parser.add_argument("--port",      type=int, default=_cfg.get("port", 8104))
parser.add_argument("--log-level", default=_cfg.get("log_level", "warning"), dest="log_level")
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

batched_model = BatchedModel(args.model)

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
    batch: list[list[Message]]
    max_new_tokens: int = 128
    temperature: float = 1.0

class BatchGenerateResponse(BaseModel):
    results: list[GenerateResponse]
    batch_size: int
    total_output_tokens: int
    wall_time_ms: float
    output_throughput: float

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

app = FastAPI(title="Layer 4 — Custom Model")


@app.get("/health")
def health():
    return {"status": "ok", "layer": 4}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    messages = [m.model_dump() for m in req.messages]
    result = batched_model.generate(
        messages=messages,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    return GenerateResponse(**result)


@app.post("/generate_batch", response_model=BatchGenerateResponse)
def generate_batch(req: BatchGenerateRequest):
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

    wall_ms  = (time.perf_counter() - t0) * 1000
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
    return {"gpu_memory_mb": round(torch.cuda.memory_allocated() / 1024**2, 1)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"Starting Layer 4 server on {args.host}:{args.port}  model={args.model}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
