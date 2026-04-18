"""
CODE_LAYERS / Layer 1 — Benchmark
===================================
Same dataset, same seed, same N as Layer 0 — results are directly comparable.

The only differences from layer0/benchmark.py:
  - Default port: 8101
  - Default layer label: 1

Usage:
    # Start the server first:
    #   python server.py

    # Then run:
    python benchmark.py
    python benchmark.py --num-requests 20 --max-new-tokens 128
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
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"

parser = argparse.ArgumentParser(description="Layer 1 benchmark — throughput measurement")
parser.add_argument("--host", default="localhost")
parser.add_argument("--port", type=int, default=8101, help="Layer server port")
parser.add_argument("--layer", type=int, default=1, help="Layer number (label only)")
parser.add_argument("--num-requests", type=int, default=20, help="Number of ShareGPT conversations to sample")
parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens to generate per request (cap)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path (for tokenizer only)")
parser.add_argument(
    "--dataset-path",
    default="",
    help="Local path to ShareGPT JSON. Leave empty to auto-download.",
)
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

    print(f"Downloading ShareGPT dataset from HuggingFace Hub ...")
    print(f"  repo_id : {SHAREGPT_REPO_ID}")
    print(f"  filename: {SHAREGPT_FILENAME}")

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


def load_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer,
    max_new_tokens: int,
    seed: int,
) -> List[BenchRequest]:
    with open(dataset_path) as f:
        raw = json.load(f)

    raw = [
        d for d in raw
        if len(d.get("conversations", d.get("conversation", []))) >= 2
    ]

    rng = random.Random(seed)
    rng.shuffle(raw)

    collected: List[BenchRequest] = []
    for item in raw:
        if len(collected) >= num_requests:
            break

        convs = item.get("conversations", item.get("conversation", []))
        human_turn    = convs[0]["value"]
        assistant_ref = convs[1]["value"]

        prompt_ids = tokenizer.encode(human_turn)
        ref_ids    = tokenizer.encode(assistant_ref)

        prompt_len = len(prompt_ids)
        ref_len    = len(ref_ids)

        if prompt_len < 4 or ref_len < 4:
            continue

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
    ttft_ms: float
    tpot_ms: float
    ok: bool
    error: str = ""


def send_request(req: BenchRequest, idx: int, total: int) -> Result:
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
        print(
            f"done in {latency_ms/1000:.1f}s  "
            f"({data['completion_tokens']} tokens  "
            f"ttft={data['ttft_ms']:.0f}ms  "
            f"tpot={data['tpot_ms']:.0f}ms)"
        )
        return Result(
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            latency_ms=latency_ms,
            ttft_ms=data["ttft_ms"],
            tpot_ms=data["tpot_ms"],
            ok=True,
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"FAILED: {e}")
        return Result(
            prompt_tokens=req.prompt_tokens,
            completion_tokens=0,
            latency_ms=latency_ms,
            ttft_ms=0.0,
            tpot_ms=0.0,
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

    avg_ttft = sum(r.ttft_ms for r in ok) / len(ok) if ok else 0
    p50_ttft = sorted(r.ttft_ms for r in ok)[len(ok)//2] if ok else 0
    p99_ttft = sorted(r.ttft_ms for r in ok)[int(len(ok)*0.99)] if ok else 0

    avg_tpot = sum(r.tpot_ms for r in ok) / len(ok) if ok else 0
    p50_tpot = sorted(r.tpot_ms for r in ok)[len(ok)//2] if ok else 0
    p99_tpot = sorted(r.tpot_ms for r in ok)[int(len(ok)*0.99)] if ok else 0

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
    print(f"║  Avg E2E latency    : {avg_latency:.0f} ms{'':<35}║")
    print(f"║  p50 E2E latency    : {p50:.0f} ms{'':<35}║")
    print(f"║  p99 E2E latency    : {p99:.0f} ms{'':<35}║")
    print(f"╠{bar}╣")
    print(f"║  TTFT avg           : {avg_ttft:.0f} ms{'':<35}║")
    print(f"║  TTFT p50           : {p50_ttft:.0f} ms{'':<35}║")
    print(f"║  TTFT p99           : {p99_ttft:.0f} ms{'':<35}║")
    print(f"╠{bar}╣")
    print(f"║  TPOT avg           : {avg_tpot:.0f} ms/tok{'':<32}║")
    print(f"║  TPOT p50           : {p50_tpot:.0f} ms/tok{'':<32}║")
    print(f"║  TPOT p99           : {p99_tpot:.0f} ms/tok{'':<32}║")
    print(f"╠{bar}╣")
    print(f"║  LAYER COMPARISON (update results.md after each layer){'':<7}║")
    print(f"║  Layer {layer} (this run): {output_throughput:.1f} tok/s  "
          f"ttft={avg_ttft:.0f}ms  tpot={avg_tpot:.0f}ms/tok{'':<8}║")
    print(f"╚{bar}╝")
    print()

    print(f"# results.md entry:")
    print(f"| {layer} | sequential | {len(ok)}/{num_requests} | "
          f"{total_input} | {total_output} | {wall_time:.1f}s | "
          f"{output_throughput:.1f} | {total_throughput:.1f} | "
          f"{avg_latency:.0f} | {avg_ttft:.0f} | {avg_tpot:.0f} |")

    return dict(
        layer=layer, num_requests=num_requests, ok=len(ok), failed=failed,
        total_input=total_input, total_output=total_output,
        wall_time=wall_time,
        output_throughput=output_throughput, total_throughput=total_throughput,
        req_rate=req_rate,
        avg_latency=avg_latency, p50=p50, p99=p99,
        avg_ttft=avg_ttft, p50_ttft=p50_ttft, p99_ttft=p99_ttft,
        avg_tpot=avg_tpot, p50_tpot=p50_tpot, p99_tpot=p99_tpot,
        seed=args.seed,
        results=results,
    )


# ---------------------------------------------------------------------------
# Write benchmark.md
# ---------------------------------------------------------------------------

def write_benchmark_md(stats: dict, bench_requests: List[BenchRequest]):
    """Write benchmark.md next to this script with the full run results."""
    import datetime
    md_path = Path(__file__).parent / "benchmark.md"

    rows = []
    for i, (req, res) in enumerate(zip(bench_requests, stats["results"])):
        status = "ok" if res.ok else "FAIL"
        rows.append(
            f"| {i+1:2d} | {req.prompt_tokens:5d} | "
            f"{res.completion_tokens:6d} | {res.latency_ms/1000:5.1f}s | "
            f"{res.ttft_ms:.0f} ms | {res.tpot_ms:.0f} ms | {status} |"
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
| Decode loop | Manual — `model.forward()` called per token, `use_cache=False` |

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
| Avg E2E latency | {stats['avg_latency']:.0f} ms |
| p50 E2E latency | {stats['p50']:.0f} ms |
| p99 E2E latency | {stats['p99']:.0f} ms |
| **Avg TTFT** | **{stats['avg_ttft']:.0f} ms** |
| p50 TTFT | {stats['p50_ttft']:.0f} ms |
| p99 TTFT | {stats['p99_ttft']:.0f} ms |
| **Avg TPOT** | **{stats['avg_tpot']:.0f} ms/tok** |
| p50 TPOT | {stats['p50_tpot']:.0f} ms/tok |
| p99 TPOT | {stats['p99_tpot']:.0f} ms/tok |

---

## Per-Request Breakdown

| # | Prompt tokens | Output tokens | Latency | TTFT | TPOT | Status |
|---|--------------|---------------|---------|------|------|--------|
{chr(10).join(rows)}

---

## What the numbers reveal

**TTFT scales with prompt length.**  
The first forward pass covers the entire prompt (prefill). Longer prompts = longer TTFT.
With no KV cache, TPOT also grows with sequence length since every decode step
re-reads the full growing sequence.

**Output throughput: {stats['output_throughput']:.1f} tok/s.**  
Layer 2 will add `past_key_values` — one change in `model.py` — and TPOT should
drop to near-constant regardless of prompt length.
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

    wait_for_server()

    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

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

    print(f"Sending {len(bench_requests)} requests sequentially ...")
    print()
    results: List[Result] = []
    t_start = time.perf_counter()
    for i, req in enumerate(bench_requests):
        result = send_request(req, i, len(bench_requests))
        results.append(result)
    wall_time = time.perf_counter() - t_start

    stats = print_results(results, wall_time, layer=args.layer, num_requests=args.num_requests)
    write_benchmark_md(stats, bench_requests)


if __name__ == "__main__":
    main()
