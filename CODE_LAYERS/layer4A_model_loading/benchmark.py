"""
CODE_LAYERS / Layer 4 — Custom Model Benchmark
===============================================
Same batch-size sweep as Layer 3.  Expected result: throughput numbers
should be near-identical to Layer 3 — the compute is the same, only the
model init and forward-pass wiring has changed.

Any significant difference (>5%) indicates a bug in our implementation.

Usage:
    python server.py          # in terminal 1
    python benchmark.py
    python benchmark.py --batch-sizes 1 4 8 16
"""

import argparse
import datetime
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_MODEL   = "Qwen/Qwen3-0.6B"
SHAREGPT_REPO_ID  = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"

parser = argparse.ArgumentParser(description="Layer 4 custom model benchmark")
parser.add_argument("--host",           default="localhost")
parser.add_argument("--port",           type=int, default=8104)
parser.add_argument("--layer",          type=int, default=4)
parser.add_argument("--num-requests",   type=int, default=20)
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument("--seed",           type=int, default=42)
parser.add_argument("--model",          default=DEFAULT_MODEL)
parser.add_argument("--dataset-path",   default="")
parser.add_argument(
    "--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 20],
)
args = parser.parse_args()

random.seed(args.seed)
BASE_URL = f"http://{args.host}:{args.port}"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def get_dataset_path(dataset_path: str) -> str:
    if dataset_path and Path(dataset_path).exists():
        return dataset_path
    print("Downloading ShareGPT dataset from HuggingFace Hub ...")
    old_offline = os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=SHAREGPT_REPO_ID,
            filename=SHAREGPT_FILENAME,
            repo_type="dataset",
        )
    finally:
        if old_offline is not None:
            os.environ["HF_HUB_OFFLINE"] = old_offline
    print(f"  cached at: {path}")
    return path


@dataclass
class BenchRequest:
    prompt: str
    prompt_tokens: int
    expected_output_tokens: int


def load_sharegpt_requests(dataset_path, num_requests, tokenizer, max_new_tokens, seed):
    with open(dataset_path) as f:
        raw = json.load(f)
    raw = [d for d in raw if len(d.get("conversations", d.get("conversation", []))) >= 2]
    rng = random.Random(seed)
    rng.shuffle(raw)
    collected = []
    for item in raw:
        if len(collected) >= num_requests:
            break
        convs = item.get("conversations", item.get("conversation", []))
        human_turn    = convs[0]["value"]
        assistant_ref = convs[1]["value"]
        prompt_ids = tokenizer.encode(human_turn)
        ref_ids    = tokenizer.encode(assistant_ref)
        if len(prompt_ids) < 4 or len(ref_ids) < 4:
            continue
        collected.append(BenchRequest(
            prompt=human_turn,
            prompt_tokens=len(prompt_ids),
            expected_output_tokens=min(len(ref_ids), max_new_tokens),
        ))
    if len(collected) < num_requests:
        print(f"WARNING: only {len(collected)} valid requests (wanted {num_requests})")
    return collected

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def wait_for_server(timeout: int = 30):
    print(f"Checking server at {BASE_URL}/health ...", end=" ", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"OK (layer={r.json().get('layer', '?')})")
                return
        except Exception:
            time.sleep(0.5)
    print("FAILED — is the server running?")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    batch_size: int
    ok: bool
    total_output_tokens: int
    wall_time_ms: float
    output_throughput: float
    avg_ttft_ms: float
    avg_tpot_ms: float
    error: str = ""


def run_batch_size(bench_requests: List[BenchRequest], batch_size: int) -> BatchResult:
    total_output = 0
    all_ttft, all_tpot = [], []
    t_wall = time.perf_counter()

    chunks = [
        bench_requests[i : i + batch_size]
        for i in range(0, len(bench_requests), batch_size)
    ]

    for chunk_idx, chunk in enumerate(chunks):
        payload = {
            "batch": [
                [{"role": "user", "content": req.prompt}] for req in chunk
            ],
            "max_new_tokens": max(req.expected_output_tokens for req in chunk),
            "temperature": 1.0,
        }
        try:
            resp = requests.post(f"{BASE_URL}/generate_batch", json=payload, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            total_output += data["total_output_tokens"]
            for r in data["results"]:
                all_ttft.append(r["ttft_ms"])
                all_tpot.append(r["tpot_ms"])
            print(
                f"    chunk {chunk_idx+1:2d}/{len(chunks)} "
                f"(B={len(chunk)}) → {data['total_output_tokens']} tokens "
                f"in {data['wall_time_ms']/1000:.1f}s "
                f"({data['output_throughput']:.1f} tok/s)"
            )
        except Exception as e:
            wall_ms = (time.perf_counter() - t_wall) * 1000
            return BatchResult(batch_size, False, 0, wall_ms, 0, 0, 0, error=str(e))

    wall_ms  = (time.perf_counter() - t_wall) * 1000
    throughput = total_output / (wall_ms / 1000) if wall_ms > 0 else 0
    avg_ttft = sum(all_ttft) / len(all_ttft) if all_ttft else 0
    avg_tpot = sum(all_tpot) / len(all_tpot) if all_tpot else 0
    return BatchResult(batch_size, True, total_output, wall_ms, throughput, avg_ttft, avg_tpot)

# ---------------------------------------------------------------------------
# Print / write
# ---------------------------------------------------------------------------

def print_sweep_results(results: List[BatchResult]):
    bar = "═" * 72
    print()
    print(f"╔{bar}╗")
    print(f"║  BATCH SIZE SWEEP RESULTS{' '*47}║")
    print(f"╠{bar}╣")
    print(f"║  {'Batch':>5}  {'Output tok':>10}  {'Wall time':>9}  "
          f"{'Tok/s':>8}  {'TTFT':>7}  {'TPOT':>7}  {'Status':<8}║")
    print(f"╠{bar}╣")
    best = max((r for r in results if r.ok), key=lambda r: r.output_throughput, default=None)
    for r in results:
        marker = " ← best" if r == best else ""
        status = "ok" if r.ok else "FAIL"
        print(
            f"║  {r.batch_size:>5}  {r.total_output_tokens:>10}  "
            f"{r.wall_time_ms/1000:>8.1f}s  {r.output_throughput:>7.1f}  "
            f"{r.avg_ttft_ms:>6.0f}ms  {r.avg_tpot_ms:>6.0f}ms  "
            f"{status:<8}{marker}║"
        )
    print(f"╚{bar}╝")
    print()


def write_benchmark_md(results: List[BatchResult], bench_requests: List[BenchRequest]):
    md_path = Path(__file__).parent / "benchmark.md"
    rows = "\n".join(
        f"| {r.batch_size} | {r.total_output_tokens} | "
        f"{r.wall_time_ms/1000:.1f}s | {r.output_throughput:.1f} | "
        f"{r.avg_ttft_ms:.0f} | {r.avg_tpot_ms:.0f} | "
        f"{'ok' if r.ok else 'FAIL'} |"
        for r in results
    )
    best = max((r for r in results if r.ok), key=lambda r: r.output_throughput, default=None)
    best_line = (
        f"Best batch size: **{best.batch_size}** → **{best.output_throughput:.1f} tok/s**"
        if best else "No successful results."
    )
    content = f"""\
# Layer 4 (Custom Model Init) — Benchmark Results

**Date:** {datetime.date.today()}
**Model:** Qwen/Qwen3-0.6B (weights loaded via safetensors, no AutoModelForCausalLM)
**Hardware:** NVIDIA GeForce RTX 4060 Ti
**Requests:** {len(bench_requests)} ShareGPT (seed={args.seed})
**Max new tokens:** {args.max_new_tokens}

---

## Batch Size Sweep

| Batch size | Output tokens | Wall time | Tok/s | Avg TTFT | Avg TPOT | Status |
|-----------|---------------|-----------|-------|----------|----------|--------|
{rows}

{best_line}

---

## What changed vs Layer 3

**Only the model init changed.** `AutoModelForCausalLM.from_pretrained` was
replaced with our own `Qwen3ForCausalLM.from_pretrained`:

- `config.py` reads `config.json` into a plain dataclass
- `norm.py`, `rope.py`, `mlp.py`, `attention.py`, `decoder_layer.py` implement
  the architecture as plain `nn.Module` subclasses
- `qwen3.py` glues them and provides `load_weights()` which streams from
  `model.safetensors` directly

**Throughput should be within ~5% of Layer 3** — the computation is identical.
Any deviation is noise or a bug in our forward-pass implementation.

## Extensibility unlocked

With the architecture in our hands we can now:
- Fuse `q_proj + k_proj + v_proj → qkv_proj` in `attention.py` + `load_weights`
- Fuse `gate_proj + up_proj → gate_up_proj` in `mlp.py` + `load_weights`
- Swap `F.scaled_dot_product_attention` for Flash Attention in `attention.py`
- Add tensor parallelism by replacing `nn.Linear` with column/row-parallel variants
"""
    md_path.write_text(content)
    print(f"Results written to {md_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print(" CODE_LAYERS / Layer 4 — Custom Model Benchmark")
    print(f" {args.num_requests} requests  |  batch sizes: {args.batch_sizes}  |  port {args.port}")
    print("=" * 72)
    print()

    wait_for_server()

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset_path = get_dataset_path(args.dataset_path)
    print(f"Sampling {args.num_requests} requests from ShareGPT ...")
    bench_requests = load_sharegpt_requests(
        dataset_path, args.num_requests, tokenizer, args.max_new_tokens, args.seed
    )
    avg_in  = sum(r.prompt_tokens for r in bench_requests) / len(bench_requests)
    avg_out = sum(r.expected_output_tokens for r in bench_requests) / len(bench_requests)
    print(f"  sampled {len(bench_requests)} requests")
    print(f"  avg prompt tokens : {avg_in:.0f}")
    print(f"  avg output tokens : {avg_out:.0f}")
    print()

    sweep_results = []
    for bs in args.batch_sizes:
        print(f"── Batch size {bs} ──")
        result = run_batch_size(bench_requests, batch_size=bs)
        sweep_results.append(result)
        if not result.ok:
            print(f"  FAILED: {result.error}")
        print()

    print_sweep_results(sweep_results)
    write_benchmark_md(sweep_results, bench_requests)


if __name__ == "__main__":
    main()
