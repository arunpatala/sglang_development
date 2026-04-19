"""
Layer 7 — Server: async FastAPI + asyncio.Future bridge to the Scheduler.

The key pattern (mirrors SGLang's TokenizerManager):

    asyncio event loop (main thread)          Scheduler (background thread)
    ────────────────────────────────          ──────────────────────────────
    POST /v1/chat/completions
      → tokenize prompt
      → create asyncio.Future
      → create Req(future=future)
      → scheduler.add_request(req)     ──→   waiting_queue.put(req)
      → await future               ←──       loop.call_soon_threadsafe(
                                               future.set_result, result)
      → return result

This decouples HTTP handling (async, I/O-bound) from GPU computation
(sync, compute-bound) without them ever sharing a thread.

Configuration is read from config.yml (same directory as this file).
CLI args override individual fields:
    python server.py                        # all values from config.yml
    python server.py --port 8200            # override just the port
    python server.py --attn-backend sdpa    # switch to F.sdpa for all paths
"""

import argparse
import asyncio
import logging
import sys
import threading
import uuid
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from model.config import AttnBackend
from model_runner import ModelRunner
from request import Req
from scheduler import Scheduler
from tokenizer import Tokenizer

# ── Config ────────────────────────────────────────────────────────────────────

_CONFIG_FILE = Path(__file__).parent / "config.yml"

def _load_config(cli_overrides: dict) -> dict:
    """Load config.yml, then apply any CLI overrides on top."""
    with open(_CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)
    cfg.update({k: v for k, v in cli_overrides.items() if v is not None})
    return cfg

def _parse_args() -> dict:
    p = argparse.ArgumentParser(description="Layer 7 inference server")
    p.add_argument("--model",        type=str,   default=None)
    p.add_argument("--port",         type=int,   default=None)
    p.add_argument("--host",         type=str,   default=None)
    p.add_argument("--log-level",    type=str,   default=None, dest="log_level")
    p.add_argument("--attn-backend", type=str,   default=None, dest="attn_backend",
                   choices=["sdpa", "flashinfer"])
    p.add_argument("--max-running",  type=int,   default=None, dest="max_running")
    args = p.parse_args()
    return {k: v for k, v in vars(args).items() if v is not None}

_ARGS = _parse_args()
_CFG  = _load_config(_ARGS)

MODEL_PATH   = _CFG.get("model",       "Qwen/Qwen3-0.6B")
PORT         = int(_CFG.get("port",    8106))
HOST         = _CFG.get("host",        "0.0.0.0")
LOG_LEVEL    = _CFG.get("log_level",   "warning")
MAX_RUNNING  = int(_CFG.get("max_running", 16))

_backend_str = _CFG.get("attn_backend", "flashinfer")
ATTN_BACKEND = AttnBackend.FLASHINFER if _backend_str == "flashinfer" else AttnBackend.SDPA

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App state ─────────────────────────────────────────────────────────────────
app       = FastAPI(title="Layer 7 — Packed Batching")
scheduler: Scheduler = None


# ── Request / Response schemas ────────────────────────────────────────────────

class Message(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    model:       str   = MODEL_PATH
    messages:    list[Message]
    max_tokens:  int   = 128
    temperature: float = 0.0

class ChatResponse(BaseModel):
    text:              str
    prompt_tokens:     int
    completion_tokens: int
    ttft_ms:           float
    latency_ms:        float


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat(req_body: ChatRequest):
    tok: Tokenizer = scheduler.model_runner.tokenizer

    messages   = [m.model_dump() for m in req_body.messages]
    prompt_str = tok.apply_chat_template(messages)
    input_ids  = tok.encode(prompt_str, device="cpu")[0].tolist()

    loop   = asyncio.get_event_loop()
    future = loop.create_future()

    req = Req(
        rid            = uuid.uuid4().hex,
        input_ids      = input_ids,
        max_new_tokens = req_body.max_tokens,
        temperature    = req_body.temperature,
        future         = future,
    )

    scheduler.add_request(req)

    result = await future
    return ChatResponse(**result)


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "attn_backend": ATTN_BACKEND.value,
        "running":      len(scheduler._running),
        "waiting":      scheduler._waiting.qsize(),
    }


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global scheduler

    logger.info(
        f"Starting  model={MODEL_PATH}  port={PORT}  "
        f"backend={ATTN_BACKEND.value}  max_running={MAX_RUNNING}"
    )
    runner = ModelRunner(MODEL_PATH, attn_backend=ATTN_BACKEND)

    scheduler = Scheduler(runner, max_running_reqs=MAX_RUNNING)

    loop = asyncio.get_event_loop()
    t = threading.Thread(
        target=scheduler.run,
        args=(loop,),
        daemon=True,
        name="scheduler",
    )
    t.start()
    logger.info(f"Scheduler thread started  port={PORT}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL,
    )
