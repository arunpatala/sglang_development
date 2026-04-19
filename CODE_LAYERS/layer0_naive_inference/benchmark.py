"""
CODE_LAYERS / benchmark.py
==========================
Shared throughput benchmark for all code layers.

Downloads the real ShareGPT dataset (anon8231489123/ShareGPT_Vicuna_unfiltered),
samples N conversations, sends them sequentially to a running layer server,
and reports throughput metrics.

Usage:
    # Start the layer server first, e.g.:
    #   python layer0/server.py

    # Then run the benchmark:
    python benchmark.py                         # layer 0, 20 requests, port 8100
    python benchmark.py --layer 1 --port 8101   # layer 1 on a different port
    python benchmark.py --num-requests 50 --max-new-tokens 128

The same benchmark, same dataset sample (fixed seed), same N — every time.
That makes results directly comparable across layers.
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests
import yaml
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# CLI  (mirrors server.py: config.yml sets defaults, CLI args override)
# ---------------------------------------------------------------------------

SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"

_HERE = Path(__file__).parent


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", default=str(_HERE / "config.yml"))
_pre_args, _ = _pre.parse_known_args()

cfg = _load_config(_pre_args.config)

parser = argparse.ArgumentParser(description="Layer benchmark — throughput measurement")
parser.add_argument("--config", default=str(_HERE / "config.yml"), help="Path to YAML config file")
parser.add_argument("--host",         default=cfg.get("host", "localhost"))
parser.add_argument("--port",         default=cfg.get("port", 8100),                            type=int, help="Layer server port")
parser.add_argument("--layer",        default=0,                                                 type=int, help="Layer number (label only)")
parser.add_argument("--num-requests", default=cfg.get("benchmark_num_requests", 20),            type=int, help="Number of ShareGPT conversations to sample")
parser.add_argument("--max-new-tokens", default=cfg.get("benchmark_max_new_tokens", 128),       type=int, help="Max tokens to generate per request (cap)")
parser.add_argument("--seed",         default=cfg.get("benchmark_seed", 42),                    type=int, help="Random seed for reproducible sampling")
parser.add_argument("--model",        default=cfg.get("model", "Qwen/Qwen3-0.6B"),              help="Model path (for tokenizer only)")
parser.add_argument("--dataset-path", default="",                                               help="Local path to ShareGPT JSON. Leave empty to auto-download.")
args = parser.parse_args()

random.seed(args.seed)

BASE_URL = f"http://{args.host}:{args.port}"

# ---------------------------------------------------------------------------
# Step 1 — Download / load ShareGPT
# ---------------------------------------------------------------------------

def get_dataset_path(dataset_path: str) -> str:
    """Return local path to ShareGPT JSON, downloading it if needed."""
    if dataset_path and Path(dataset_path).exists():
        return dataset_path

    # Use huggingface_hub to download and cache
    print(f"Downloading ShareGPT dataset from HuggingFace Hub ...")
    print(f"  repo_id : {SHAREGPT_REPO_ID}")
    print(f"  filename: {SHAREGPT_FILENAME}")

    # Temporarily unset HF_HUB_OFFLINE so the download succeeds
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
    prompt: str          # formatted string ready to send as a user message
    prompt_tokens: int   # approximate token count (measured by tokenizer)
    expected_output_tokens: int  # from the reference completion in ShareGPT


def load_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer,
    max_new_tokens: int,
    seed: int,
) -> List[BenchRequest]:
    """
    Sample `num_requests` single-turn conversations from ShareGPT.
    Returns a list of BenchRequest in a fixed, reproducible order.
    """
    with open(dataset_path) as f:
        raw = json.load(f)

    # Keep only conversations with at least 2 turns (human + assistant)
    raw = [
        d for d in raw
        if len(d.get("conversations", d.get("conversation", []))) >= 2
    ]

    # Deterministic shuffle with the given seed
    rng = random.Random(seed)
    rng.shuffle(raw)

    collected: List[BenchRequest] = []
    for item in raw:
        if len(collected) >= num_requests:
            break

        convs = item.get("conversations", item.get("conversation", []))
        human_turn   = convs[0]["value"]
        assistant_ref = convs[1]["value"]

        prompt_ids = tokenizer.encode(human_turn)
        ref_ids    = tokenizer.encode(assistant_ref)

        prompt_len = len(prompt_ids)
        ref_len    = len(ref_ids)

        # Skip too-short sequences
        if prompt_len < 4 or ref_len < 4:
            continue

        # Cap the requested output at max_new_tokens; use reference length otherwise
        output_len = min(ref_len, max_new_tokens)

        collected.append(BenchRequest(
            prompt=human_turn,
            prompt_tokens=prompt_len,
            expected_output_tokens=output_len,
        ))

    if len(collected) < num_requests:
        print(f"WARNING: only found {len(collected)} valid requests (wanted {num_requests})")

    return collected


# ---------------------------------------------------------------------------
# Step 2 — Send requests sequentially
# ---------------------------------------------------------------------------

@dataclass
class Result:
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    ok: bool
    error: str = ""


def send_request(req: BenchRequest, idx: int, total: int) -> Result:
    """Send one request to the layer server and return timing data."""
    payload = {
        "messages": [{"role": "user", "content": req.prompt}],
        "max_new_tokens": req.expected_output_tokens,
        "temperature": 1.0,
    }

    print(f"  [{idx+1:2d}/{total}] prompt_tokens={req.prompt_tokens:4d}  "
          f"max_new_tokens={req.expected_output_tokens:3d}  ... ", end="", flush=True)

    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{BASE_URL}/generate", json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"done in {latency_ms/1000:.1f}s  ({data['completion_tokens']} tokens)")
        return Result(
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            latency_ms=latency_ms,
            ok=True,
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"FAILED: {e}")
        return Result(
            prompt_tokens=req.prompt_tokens,
            completion_tokens=0,
            latency_ms=latency_ms,
            ok=False,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Step 3 — Health check
# ---------------------------------------------------------------------------

def wait_for_server(timeout: int = 30):
    print(f"Checking server at {BASE_URL}/health ...", end=" ", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                data = r.json()
                print(f"OK (layer={data.get('layer', '?')})")
                return
        except Exception:
            time.sleep(0.5)
    print("FAILED — is the server running?")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 4 — Print results
# ---------------------------------------------------------------------------

def print_results(results: List[Result], wall_time: float, layer: int, num_requests: int):
    ok = [r for r in results if r.ok]
    failed = len(results) - len(ok)

    total_input  = sum(r.prompt_tokens for r in ok)
    total_output = sum(r.completion_tokens for r in ok)
    total_tokens = total_input + total_output

    output_throughput = total_output / wall_time if wall_time > 0 else 0
    total_throughput  = total_tokens / wall_time if wall_time > 0 else 0
    req_rate          = len(ok) / wall_time if wall_time > 0 else 0
    avg_latency       = sum(r.latency_ms for r in ok) / len(ok) if ok else 0
    p50 = sorted(r.latency_ms for r in ok)[len(ok)//2] if ok else 0
    p99 = sorted(r.latency_ms for r in ok)[int(len(ok)*0.99)] if ok else 0

    bar = "═" * 62

    print()
    print(f"╔{bar}╗")
    print(f"║  BENCHMARK RESULTS{' '*43}║")
    print(f"╠{bar}╣")
    print(f"║  Layer              : {layer:<39}║")
    print(f"║  Requests sent      : {num_requests:<39}║")
    print(f"║  Successful         : {len(ok):<39}║")
    print(f"║  Failed             : {failed:<39}║")
    print(f"║  Dataset            : ShareGPT (seed={args.seed}){'':<24}║")
    print(f"║  Mode               : sequential{'':<28}║")
    print(f"╠{bar}╣")
    print(f"║  Total input tokens : {total_input:<39}║")
    print(f"║  Total output tokens: {total_output:<39}║")
    print(f"║  Total wall time    : {wall_time:.1f}s{'':<36}║")
    print(f"╠{bar}╣")
    print(f"║  Output throughput  : {output_throughput:.1f} tok/s{'':<32}║")
    print(f"║  Total throughput   : {total_throughput:.1f} tok/s{'':<32}║")
    print(f"║  Request rate       : {req_rate:.3f} req/s{'':<31}║")
    print(f"╠{bar}╣")
    print(f"║  Avg latency        : {avg_latency:.0f} ms{'':<35}║")
    print(f"║  p50 latency        : {p50:.0f} ms{'':<35}║")
    print(f"║  p99 latency        : {p99:.0f} ms{'':<35}║")
    print(f"╠{bar}╣")
    print(f"║  LAYER COMPARISON (update results.md after each layer){'':<7}║")
    print(f"║  Layer {layer} (this run): {output_throughput:.1f} tok/s output  "
          f"{total_throughput:.1f} tok/s total{'':<5}║")
    print(f"╚{bar}╝")
    print()

    print(f"# results.md entry:")  # copy-paste row for CODE_LAYERS/results.md
    print(f"| {layer} | sequential | {len(ok)}/{num_requests} | "
          f"{total_input} | {total_output} | {wall_time:.1f}s | "
          f"{output_throughput:.1f} | {total_throughput:.1f} | {avg_latency:.0f} | — | — |")

    return dict(
        layer=layer, num_requests=num_requests, ok=len(ok), failed=failed,
        total_input=total_input, total_output=total_output,
        wall_time=wall_time,
        output_throughput=output_throughput, total_throughput=total_throughput,
        req_rate=req_rate,
        avg_latency=avg_latency, p50=p50, p99=p99,
        seed=args.seed,
        results=results,
    )


# ---------------------------------------------------------------------------
# Write benchmark.md
# ---------------------------------------------------------------------------

def write_benchmark_md(stats: dict, bench_requests: List[BenchRequest]):
    """Write a benchmark.md file next to this script with the run results."""
    import datetime
    md_path = Path(__file__).parent / "benchmark.md"

    rows = []
    for i, (req, res) in enumerate(zip(bench_requests, stats["results"])):
        status = "ok" if res.ok else "FAIL"
        rows.append(
            f"| {i+1:2d} | {req.prompt_tokens:5d} | "
            f"{res.completion_tokens:6d} | {res.latency_ms/1000:5.1f}s | {status} |"
        )

    content = f"""\
# Layer {stats['layer']} — Benchmark Results

**Date:** {datetime.date.today()}  
**Model:** Qwen/Qwen3-0.6B  
**Hardware:** NVIDIA GeForce RTX 4060 Ti  
**Command:**
```bash
python benchmark.py --layer {stats['layer']} --port {args.port} --num-requests {stats['num_requests']}
```

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | ShareGPT (`anon8231489123/ShareGPT_Vicuna_unfiltered`) |
| Seed | {stats['seed']} |
| Requests | {stats['num_requests']} |
| Max new tokens | {args.max_new_tokens} (capped at reference completion length) |
| Mode | Sequential (one request at a time) |
| `use_cache` | `False` — Layer 0 defining constraint |

---

## Summary Metrics

| Metric | Value |
|--------|-------|
| Successful requests | {stats['ok']} / {stats['num_requests']} |
| Total input tokens | {stats['total_input']:,} |
| Total output tokens | {stats['total_output']:,} |
| Total wall time | {stats['wall_time']:.1f} s |
| **Output throughput** | **{stats['output_throughput']:.1f} tok/s** |
| Total throughput | {stats['total_throughput']:.1f} tok/s |
| Request rate | {stats['req_rate']:.3f} req/s |
| Avg latency | {stats['avg_latency']:.0f} ms |
| p50 latency | {stats['p50']:.0f} ms |
| p99 latency | {stats['p99']:.0f} ms |

---

## Per-Request Breakdown

| # | Prompt tokens | Output tokens | Latency | Status |
|---|--------------|---------------|---------|--------|
{chr(10).join(rows)}

---

## What the numbers reveal

**Latency scales with prompt length.**  
With `use_cache=False`, every decode step recomputes attention over the entire sequence — O(prompt_len) extra work per generated token.

**Output throughput is the baseline to beat: {stats['output_throughput']:.1f} tok/s.**  
Layer 1 writes the decode loop manually (same cost). Layer 2 adds KV cache and this number should jump significantly.
"""

    md_path.write_text(content)
    print(f"Results written to {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print(" CODE_LAYERS Benchmark")
    print(f" Layer {args.layer}  |  {args.num_requests} requests  |  port {args.port}")
    print("=" * 64)
    print()

    # Health check
    wait_for_server()

    # Load tokenizer (needed to count tokens for dataset sampling)
    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Download / load ShareGPT
    dataset_path = get_dataset_path(args.dataset_path)
    print(f"Sampling {args.num_requests} requests from ShareGPT ...")
    bench_requests = load_sharegpt_requests(
        dataset_path=dataset_path,
        num_requests=args.num_requests,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    print(f"  sampled {len(bench_requests)} requests")
    avg_in  = sum(r.prompt_tokens for r in bench_requests) / len(bench_requests)
    avg_out = sum(r.expected_output_tokens for r in bench_requests) / len(bench_requests)
    print(f"  avg prompt tokens : {avg_in:.0f}")
    print(f"  avg output tokens : {avg_out:.0f} (capped at {args.max_new_tokens})")
    print()

    # Run benchmark — sequential
    print(f"Sending {len(bench_requests)} requests sequentially ...")
    print()
    results: List[Result] = []
    t_start = time.perf_counter()
    for i, req in enumerate(bench_requests):
        result = send_request(req, i, len(bench_requests))
        results.append(result)
    wall_time = time.perf_counter() - t_start

    # Print summary and write benchmark.md
    stats = print_results(results, wall_time, layer=args.layer, num_requests=args.num_requests)
    write_benchmark_md(stats, bench_requests)


if __name__ == "__main__":
    main()
