"""
Layer 5 — Server: async FastAPI + asyncio.Future bridge to the Scheduler.

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

Port: 8105
"""

import asyncio
import logging
import sys
import threading
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from model_runner import ModelRunner
from request import Req
from scheduler import Scheduler
from tokenizer import Tokenizer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = "Qwen/Qwen3-0.6B"
PORT            = 8105
MAX_RUNNING     = 16     # max simultaneous decode requests

# ── App state ─────────────────────────────────────────────────────────────────
app       = FastAPI(title="Layer 5 — Continuous Batching")
scheduler: Scheduler = None


# ── Request / Response schemas ────────────────────────────────────────────────

class Message(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    model:       str = MODEL_PATH
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

    # Format and tokenise the prompt
    messages   = [m.model_dump() for m in req_body.messages]
    prompt_str = tok.apply_chat_template(messages)
    input_ids  = tok.encode(prompt_str, device="cpu")[0].tolist()

    # Create a Future in the current asyncio loop
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

    # Block until the scheduler resolves the future
    result = await future
    return ChatResponse(**result)


@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "running":  len(scheduler._running),
        "waiting":  scheduler._waiting.qsize(),
    }


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global scheduler

    logger.info("Loading model …")
    runner = ModelRunner(MODEL_PATH)

    scheduler = Scheduler(runner, max_running_reqs=MAX_RUNNING)

    # Start the scheduler in a background daemon thread.
    # Pass the running asyncio event loop so the scheduler can call
    # loop.call_soon_threadsafe() to resolve futures.
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
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )
