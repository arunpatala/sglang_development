"""
CODE_LAYERS / Layer 0 — Naive HTTP Inference Server
=====================================================
The absolute minimum: one request at a time, no KV cache, no batching.
HuggingFace model.generate() called fresh for every request.

What this teaches:
  - The basic shape of an inference server
  - Why sequential serving is slow (measure it with test_client.py)
  - The baseline we will improve layer by layer

What is deliberately missing (to be added in later layers):
  - KV cache (added in layer1)
  - Batching (added in layer2)
  - Request queue / async scheduling (added in layer3)
  - Custom memory management (added in layer4)
  - Prefix caching (added in layer5)

Run:
    python server.py
    python server.py --model /path/to/model --port 8100

Then in another terminal:
    python test_client.py
"""

import argparse
import logging
import time

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_MODEL = (
    "/home/arun/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-0.6B/snapshots/"
    "c1899de289a04d12100db370d81485cdf75e47ca"
)

parser = argparse.ArgumentParser(description="Layer 0 — naive inference server")
parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path or HF hub id")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8100)
# parse_known_args so uvicorn's own args don't cause errors
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Load model once at startup (blocking — that is fine here)
# ---------------------------------------------------------------------------

logger.info(f"Loading model from {args.model} ...")
t_load = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    dtype=torch.bfloat16,
).to("cuda")
model.eval()

logger.info(f"Model loaded in {time.perf_counter() - t_load:.1f}s")
logger.info(f"GPU memory after load: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

# ---------------------------------------------------------------------------
# Request / Response schema
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str   # "system" | "user" | "assistant"
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

# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------

app = FastAPI(title="Layer 0 — Naive Inference Server")


@app.get("/health")
def health():
    return {"status": "ok", "layer": 0}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """
    The naive inference path:
      1. Apply chat template + tokenize the messages
      2. Call model.generate() — one call, one request, no reuse of any state
      3. Decode the output tokens back to text
      4. Return

    Notice:
    - Every call recomputes the full forward pass from scratch.
    - There is no KV cache — even though HuggingFace uses one internally inside
      model.generate(), it is created fresh each time and discarded after.
    - If two requests arrive at the same time, the second one blocks until the
      first is fully complete (Python's GIL + synchronous FastAPI route).
    """
    last_content = req.messages[-1].content if req.messages else ""
    t0 = time.perf_counter()
    logger.info(f"START: {last_content[:60]!r}")

    # Step 1 — Apply chat template then tokenize.
    # req.messages is already a list of {role, content} dicts — the same shape
    # as the OpenAI API. We pass it directly to apply_chat_template, exactly as
    # SGLang's OpenAIServingChat._process_messages() does in serving_chat.py.
    # add_generation_prompt=True appends <|im_start|>assistant\n so the model
    # knows to generate a reply rather than continue the conversation as a document.
    # enable_thinking=False suppresses Qwen3's <think>...</think> reasoning mode.
    messages = [m.model_dump() for m in req.messages]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to("cuda")

    prompt_tokens = input_ids.shape[1]
    logger.info(f"prompt_tokens={prompt_tokens}")

    # Step 2 — Generate
    # use_cache=True is the HuggingFace default: it still builds past_key_values
    # internally during generate(). We disable it here deliberately to show the
    # true naive baseline. Toggle use_cache=True to see the Layer 1 speedup
    # without even changing the server architecture.
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=req.max_new_tokens,
            do_sample=(req.temperature > 0 and req.temperature != 1.0),
            temperature=req.temperature if req.temperature != 1.0 else None,
            use_cache=False,  # <-- the defining choice of Layer 0
        )

    # Step 3 — Decode only the newly generated tokens (strip the prompt)
    new_ids = output_ids[0, prompt_tokens:]
    completion_tokens = len(new_ids)
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(
        f"DONE: completion_tokens={completion_tokens} "
        f"latency={latency_ms}ms "
        f"tok/s={round(completion_tokens / (latency_ms / 1000), 1)}"
    )

    return GenerateResponse(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
    )


@app.get("/stats")
def stats():
    return {
        "gpu_memory_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"Starting Layer 0 server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
