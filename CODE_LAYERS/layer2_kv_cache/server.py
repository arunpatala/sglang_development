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
    python server.py --config config.yml          # explicit (same result)
    python server.py --port 8200                  # override one param; rest from config.yml
    python server.py --model Qwen/Qwen3-0.6B --port 8102
"""

import argparse
import logging
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

from model import KVCacheModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading  (mirrors sglang's ConfigArgumentMerger pattern)
# Precedence:  CLI args  >  config.yml  >  Python defaults below
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent


def load_config(path: str) -> dict:
    """Load a YAML config file and return its contents as a dict."""
    p = Path(path)
    if not p.exists():
        logger.warning(f"Config file not found: {p} — using CLI/default values only")
        return {}
    with open(p) as f:
        data = yaml.safe_load(f)
    return data or {}


# ---------------------------------------------------------------------------
# CLI — two-pass parse:
#   Pass 1: extract --config (or default) to load the YAML.
#   Pass 2: build the real parser with YAML values as defaults so CLI wins.
# ---------------------------------------------------------------------------

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", default=str(_HERE / "config.yml"))
_pre_args, _ = _pre.parse_known_args()

cfg = load_config(_pre_args.config)

parser = argparse.ArgumentParser(description="Layer 2 — KV cache decode server")
parser.add_argument("--config",         default=str(_HERE / "config.yml"),      help="Path to YAML config file")
parser.add_argument("--model",          default=cfg.get("model",         "Qwen/Qwen3-0.6B"), help="Model path or HF hub id")
parser.add_argument("--host",           default=cfg.get("host",          "0.0.0.0"))
parser.add_argument("--port",           default=cfg.get("port",          8102),               type=int)
parser.add_argument("--dtype",          default=cfg.get("dtype",         "bfloat16"),         help="Weight dtype (bfloat16 | float16 | float32)")
parser.add_argument("--device",         default=cfg.get("device",        "cuda"))
parser.add_argument("--log-level",      default=cfg.get("log_level",     "warning"),          help="Uvicorn log level")
parser.add_argument("--max-new-tokens", default=cfg.get("max_new_tokens", 64),               type=int,   help="Default generation length when request omits it")
parser.add_argument("--temperature",    default=cfg.get("temperature",   1.0),               type=float)

args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

logger.info(f"Config : {_pre_args.config}")
logger.info(f"Model  : {args.model}")
logger.info(f"Device : {args.device}  dtype={args.dtype}")

kv_model = KVCacheModel(args.model, dtype=args.dtype, device=args.device)

# ---------------------------------------------------------------------------
# Request / Response schema  (identical to Layer 1)
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    messages: list[Message]
    max_new_tokens: int = args.max_new_tokens
    temperature: float = args.temperature

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
        "dtype": args.dtype,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"Starting Layer 2 server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
