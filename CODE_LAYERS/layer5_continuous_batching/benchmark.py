"""
Layer 5 — Benchmark: constant-concurrency sliding window.

Instead of blasting all N requests at once, we maintain exactly
--concurrency (default 4) requests in-flight at all times.  As soon
as one request finishes, the next one is sent immediately.

This is the correct way to benchmark continuous batching:
  • The scheduler decode batch always has ~concurrency active requests.
  • New requests genuinely join mid-flight as old ones finish.
  • GPU stays maximally busy without over-saturating the waiting queue.

Pattern: asyncio.Semaphore(concurrency) — a coroutine can only enter
the "send" block when a slot is free.  asyncio.gather() fires all
coroutines but at most `concurrency` are inside the HTTP call at once.

Metrics collected:
  - Total wall-clock time for all N requests to finish
  - Output throughput  (output tokens / total time)
  - Total throughput   (input + output tokens / total time)
  - Per-request TTFT, latency (from response JSON)

Usage:
    python benchmark.py
    python benchmark.py --n-requests 20 --concurrency 4 --max-tokens 128
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp
from datasets import load_dataset

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--host",        default="http://localhost:8105")
parser.add_argument("--n-requests",  type=int,   default=20)
parser.add_argument("--concurrency", type=int,   default=4)
parser.add_argument("--max-tokens",  type=int,   default=128)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--output",      default="benchmark.md")
args = parser.parse_args()

URL = f"{args.host}/v1/chat/completions"


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_prompts(n: int) -> list[str]:
    ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered",
                      data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
                      split="train")
    prompts = []
    for row in ds:
        for conv in row.get("conversations", []):
            if conv.get("from") == "human" and conv.get("value", "").strip():
                prompts.append(conv["value"].strip())
                if len(prompts) == n:
                    return prompts
    return prompts


# ── Single request ────────────────────────────────────────────────────────────

async def send_request(
    session: aiohttp.ClientSession,
    prompt: str,
    idx: int,
) -> dict:
    payload = {
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  args.max_tokens,
        "temperature": args.temperature,
    }
    t0 = time.perf_counter()
    async with session.post(URL, json=payload) as resp:
        data = await resp.json()
    data["wall_ms"] = (time.perf_counter() - t0) * 1000
    data["idx"]     = idx
    return data


# ── Benchmark ─────────────────────────────────────────────────────────────────

async def run_benchmark(prompts: list[str], concurrency: int) -> list[dict]:
    """
    Sliding-window benchmark: keep exactly `concurrency` requests in-flight.

    asyncio.Semaphore(concurrency) ensures at most `concurrency` coroutines
    are inside the HTTP send/await at any moment.  When one finishes and
    releases the semaphore, the next queued coroutine immediately starts —
    so the server always sees ~concurrency active requests.
    """
    sem = asyncio.Semaphore(concurrency)

    async def send_with_slot(session, prompt, idx):
        async with sem:          # blocks until a slot is free
            return await send_request(session, prompt, idx)

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [send_with_slot(session, p, i) for i, p in enumerate(prompts)]
        t0      = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_s = time.perf_counter() - t0
    return results, total_s


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(results: list[dict], total_s: float, n: int) -> None:
    out_toks  = sum(r["completion_tokens"] for r in results)
    in_toks   = sum(r["prompt_tokens"]     for r in results)
    ttfts     = [r["ttft_ms"]   for r in results]
    latencies = [r["latency_ms"] for r in results]

    out_tps   = out_toks / total_s
    total_tps = (in_toks + out_toks) / total_s

    print()
    print("=" * 60)
    print(f" Layer 5 — Continuous Batching Benchmark")
    print(f" Requests         : {n} total  (concurrency={args.concurrency})")
    print(f" Max output tokens: {args.max_tokens}")
    print(f" Total wall time  : {total_s:.2f}s")
    print(f" Output throughput: {out_tps:.1f} tok/s")
    print(f" Total throughput : {total_tps:.1f} tok/s")
    print(f" TTFT  avg/p95    : {statistics.mean(ttfts):.0f}ms / "
          f"{sorted(ttfts)[int(0.95*len(ttfts))]:.0f}ms")
    print(f" Latency avg/p95  : {statistics.mean(latencies):.0f}ms / "
          f"{sorted(latencies)[int(0.95*len(latencies))]:.0f}ms")
    print("=" * 60)

    # Write markdown results
    md = Path(args.output)
    section = f"""
## Layer 5 — Continuous Batching

| Metric | Value |
|--------|-------|
| Requests (total / concurrency) | {n} / {args.concurrency} |
| Max output tokens | {args.max_tokens} |
| Total wall time | {total_s:.2f}s |
| Output throughput | {out_tps:.1f} tok/s |
| Total throughput | {total_tps:.1f} tok/s |
| TTFT avg / p95 | {statistics.mean(ttfts):.0f}ms / {sorted(ttfts)[int(0.95*len(ttfts))]:.0f}ms |
| Latency avg / p95 | {statistics.mean(latencies):.0f}ms / {sorted(latencies)[int(0.95*len(latencies))]:.0f}ms |
"""
    md.write_text(section.strip())
    print(f"\nResults written to {md.resolve()}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print(f"Loading {args.n_requests} prompts from ShareGPT …")
    prompts = load_prompts(args.n_requests)
    print(
        f"Loaded {len(prompts)} prompts.  "
        f"Sending with concurrency={args.concurrency} to {URL} …\n"
    )
    results, total_s = await run_benchmark(prompts, args.concurrency)
    report(results, total_s, len(prompts))


if __name__ == "__main__":
    asyncio.run(main())
