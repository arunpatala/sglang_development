"""
bench_server.py — SGLang server A/B benchmark: BF16 KV vs FP8 KV cache.

Launches two SGLang server instances sequentially (baseline then compare),
fires concurrent HTTP requests at each, and measures:
  - Time to first token (TTFT)
  - End-to-end request latency
  - Throughput (output tokens / second)
  - VRAM consumed by the KV pool (via /get_server_info)

⚠ IMPORTANT — model compatibility with FP8 KV cache:
  - Qwen2.5 models are BROKEN with FP8 KV (outlier attention heads, repeating output).
    Use --compare-kv-dtype auto (BF16) or switch to a Llama/Qwen3 model.
  - Qwen3 models (with per-head QK RMSNorm) are safe for FP8 KV experiments.
  - Llama-3.1-8B-Instruct-FP8 has embedded calibrated KV scales — use --kv-dtype auto.

Usage:
    # Qwen3 (safe for FP8 KV — the model used in this layer):
    python bench_server.py \\
        --model Qwen/Qwen3-1.7B \\
        --baseline-kv-dtype auto \\
        --compare-kv-dtype fp8_e4m3 \\
        --n-requests 24 --concurrency 4 --max-tokens 200

    # Llama with embedded KV scales (best accuracy):
    python bench_server.py \\
        --model meta-llama/Meta-Llama-3.1-8B-Instruct-FP8 \\
        --baseline-kv-dtype bf16 \\
        --compare-kv-dtype auto \\
        --n-requests 24 --concurrency 6 --max-tokens 200

    # Quick smoke test (no model download, uses tiny Qwen3-0.6B):
    python bench_server.py \\
        --model Qwen/Qwen3-0.6B \\
        --n-requests 8 --concurrency 2 --max-tokens 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from utils import BenchResult, get_prompts, print_comparison, write_markdown


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SGLang KV cache quantization A/B benchmark"
    )
    # Model
    p.add_argument("--model", default="Qwen/Qwen3-1.7B",
                   help="HuggingFace model ID or local path")
    p.add_argument("--quantization", default=None,
                   help="Weight quantization method (e.g. gptq_marlin, fp8)")

    # KV cache dtypes to compare
    p.add_argument("--baseline-kv-dtype", default="auto",
                   choices=["auto", "bf16", "bfloat16", "fp8_e4m3", "fp8_e5m2"],
                   help="KV cache dtype for baseline run (default: auto → BF16 for most models)")
    p.add_argument("--compare-kv-dtype",  default="fp8_e4m3",
                   choices=["auto", "bf16", "bfloat16", "fp8_e4m3", "fp8_e5m2"],
                   help="KV cache dtype for comparison run (default: fp8_e4m3)")

    # Server
    p.add_argument("--port",           type=int,   default=30080,
                   help="Port for SGLang server (default: 30080)")
    p.add_argument("--mem-fraction",   type=float, default=0.75,
                   help="--mem-fraction-static passed to SGLang (default: 0.75)")
    p.add_argument("--startup-timeout",type=int,   default=120,
                   help="Seconds to wait for server to start (default: 120)")

    # Benchmark
    p.add_argument("--n-requests",  type=int, default=24,
                   help="Total requests to send per run (default: 24)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Concurrent requests in flight (default: 4)")
    p.add_argument("--max-tokens",  type=int, default=200,
                   help="Max new tokens per request (default: 200)")
    p.add_argument("--prompt-kind", default="mixed",
                   choices=["short", "long", "mixed"],
                   help="Prompt length distribution (default: mixed)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (default: 0.0 = greedy)")

    # Output
    p.add_argument("--output", default="results.md",
                   help="Markdown output file (default: results.md)")
    p.add_argument("--no-compare", action="store_true",
                   help="Only run the baseline (useful for debugging)")

    return p.parse_args()


# ── Server management ─────────────────────────────────────────────────────────

def _build_server_cmd(args: argparse.Namespace, kv_dtype: str) -> List[str]:
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model",              args.model,
        "--kv-cache-dtype",     kv_dtype,
        "--mem-fraction-static",str(args.mem_fraction),
        "--port",               str(args.port),
        "--host",               "127.0.0.1",
        "--disable-radix-cache",          # disable prefix caching for fair A/B comparison
    ]
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    return cmd


def _server_url(args: argparse.Namespace) -> str:
    return f"http://127.0.0.1:{args.port}"


def start_server(
    args:     argparse.Namespace,
    kv_dtype: str,
    timeout:  int,
) -> subprocess.Popen:
    """Launch SGLang server as a subprocess and wait until healthy."""
    import urllib.request, urllib.error

    cmd = _build_server_cmd(args, kv_dtype)
    print(f"\n  Launching: {' '.join(cmd)}\n")

    log_path = Path(__file__).parent / f"server_{kv_dtype}.log"
    log_file = open(log_path, "w")

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # process group for clean kill
    )

    health_url = f"{_server_url(args)}/health"
    deadline   = time.time() + timeout
    dots = 0

    print(f"  Waiting for server (log → {log_path}) ", end="", flush=True)
    while time.time() < deadline:
        if proc.poll() is not None:
            log_file.close()
            print(f"\n  ERROR: Server exited early (rc={proc.returncode}). "
                  f"Check {log_path}")
            sys.exit(1)
        try:
            urllib.request.urlopen(health_url, timeout=2)
            print(f" ready ({time.time() - (deadline - timeout):.0f}s)")
            return proc
        except (urllib.error.URLError, OSError):
            time.sleep(2)
            dots += 1
            if dots % 15 == 0:
                print(".", end="", flush=True)

    log_file.close()
    stop_server(proc)
    print(f"\n  ERROR: Server did not become healthy in {timeout}s. Check {log_path}")
    sys.exit(1)


def stop_server(proc: subprocess.Popen) -> None:
    """Kill the server process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


# ── VRAM measurement ──────────────────────────────────────────────────────────

def measure_vram_mb() -> float:
    """Current GPU VRAM allocated in MB."""
    try:
        import torch
        return torch.cuda.memory_allocated() / 1024**2
    except Exception:
        return 0.0


def query_server_info(base_url: str) -> dict:
    """Query SGLang's /get_server_info for KV pool stats (if available)."""
    import urllib.request, urllib.error
    try:
        resp = urllib.request.urlopen(f"{base_url}/get_server_info", timeout=5)
        return json.loads(resp.read())
    except Exception:
        return {}


# ── Async HTTP benchmark ──────────────────────────────────────────────────────

async def _one_request(
    session,
    base_url:   str,
    prompt:     str,
    max_tokens: int,
    temperature: float,
    result:     BenchResult,
) -> None:
    """Send one chat completion request and record metrics."""
    import aiohttp

    payload = {
        "model":       "default",
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      True,     # streaming to measure TTFT separately
    }

    t_start = time.perf_counter()
    t_first = None
    output_tokens = 0

    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                print(f"\n  HTTP {resp.status}: {body[:200]}")
                result.errors += 1
                return

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    if delta.get("content"):
                        if t_first is None:
                            t_first = time.perf_counter()
                        output_tokens += 1
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    except Exception as e:
        print(f"\n  Request error: {e}")
        result.errors += 1
        return

    t_end = time.perf_counter()

    if t_first is None:
        result.errors += 1
        return

    result.ttft_ms.append((t_first - t_start) * 1000)
    result.latency_ms.append((t_end - t_start) * 1000)
    result.output_tokens.append(output_tokens)


async def run_benchmark(
    base_url:    str,
    prompts:     List[str],
    max_tokens:  int,
    temperature: float,
    concurrency: int,
    result:      BenchResult,
) -> None:
    """Send all prompts with bounded concurrency; fill in result in-place."""
    try:
        import aiohttp
    except ImportError:
        print("ERROR: aiohttp not installed. Run: pip install aiohttp")
        sys.exit(1)

    sem = asyncio.Semaphore(concurrency)

    async def guarded(session, prompt: str) -> None:
        async with sem:
            await _one_request(
                session, base_url, prompt, max_tokens, temperature, result
            )

    t_wall_start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [guarded(session, p) for p in prompts]
        await asyncio.gather(*tasks)
    result.wall_s = time.perf_counter() - t_wall_start


# ── Non-streaming fallback (if aiohttp streaming not available) ───────────────

async def _one_request_nonstream(
    session,
    base_url:    str,
    prompt:      str,
    max_tokens:  int,
    temperature: float,
    result:      BenchResult,
) -> None:
    """Non-streaming fallback — measures latency only (TTFT not available)."""
    import aiohttp

    payload = {
        "model":       "default",
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      False,
    }
    t_start = time.perf_counter()
    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            body = await resp.json()

        t_end = time.perf_counter()

        # SGLang's non-streaming response has usage.completion_tokens
        usage  = body.get("usage", {})
        n_toks = usage.get("completion_tokens", max_tokens)
        lat_ms = (t_end - t_start) * 1000

        # Estimate TTFT as (latency / output_tokens) for the first token
        ttft_ms_est = lat_ms / max(n_toks, 1)

        result.ttft_ms.append(ttft_ms_est)
        result.latency_ms.append(lat_ms)
        result.output_tokens.append(n_toks)

    except Exception as e:
        print(f"\n  Request error: {e}")
        result.errors += 1


async def run_benchmark_nonstream(
    base_url:    str,
    prompts:     List[str],
    max_tokens:  int,
    temperature: float,
    concurrency: int,
    result:      BenchResult,
) -> None:
    try:
        import aiohttp
    except ImportError:
        print("ERROR: aiohttp not installed. Run: pip install aiohttp")
        sys.exit(1)

    sem = asyncio.Semaphore(concurrency)

    async def guarded(session, prompt: str) -> None:
        async with sem:
            await _one_request_nonstream(
                session, base_url, prompt, max_tokens, temperature, result
            )

    t_wall_start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [guarded(session, p) for p in prompts]
        await asyncio.gather(*tasks)
    result.wall_s = time.perf_counter() - t_wall_start


# ── Single run ────────────────────────────────────────────────────────────────

def run_one(
    args:      argparse.Namespace,
    kv_dtype:  str,
    label:     str,
    prompts:   List[str],
) -> BenchResult:
    result = BenchResult(
        label       = label,
        kv_dtype    = kv_dtype,
        n_requests  = len(prompts),
        max_tokens  = args.max_tokens,
        concurrency = args.concurrency,
    )

    print(f"\n{'─'*60}")
    print(f"  Run: {label}  (kv_dtype={kv_dtype})")
    print(f"{'─'*60}")

    proc = start_server(args, kv_dtype, args.startup_timeout)

    try:
        base_url = _server_url(args)

        # Short warmup — 2 requests to prime CUDA graphs / compilation
        print(f"  Warming up … ", end="", flush=True)
        warmup_result = BenchResult(
            label="warmup", kv_dtype=kv_dtype,
            n_requests=2, max_tokens=20, concurrency=1,
        )
        asyncio.run(run_benchmark_nonstream(
            base_url, prompts[:2], 20, args.temperature, 1, warmup_result
        ))
        print("done")

        # VRAM before
        info_before = query_server_info(base_url)
        result.vram_before_mb = measure_vram_mb()

        # Main benchmark
        print(f"  Sending {len(prompts)} requests (concurrency={args.concurrency}) … ")
        t0 = time.time()

        asyncio.run(run_benchmark(
            base_url, prompts, args.max_tokens, args.temperature,
            args.concurrency, result
        ))

        elapsed = time.time() - t0
        result.vram_after_mb = measure_vram_mb()
        info_after = query_server_info(base_url)

        n_ok = len(result.ttft_ms)
        print(
            f"  Done in {elapsed:.1f}s  |  "
            f"{n_ok}/{len(prompts)} succeeded  |  "
            f"{result.throughput_tps:.1f} tok/s  |  "
            f"TTFT avg {result.ttft_avg:.0f}ms"
        )
        if result.errors > 0:
            print(f"  WARNING: {result.errors} requests failed (see server log)")

    finally:
        print(f"  Stopping server … ", end="", flush=True)
        stop_server(proc)
        print("stopped")
        time.sleep(3)   # let GPU memory fully free before next run

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    prompts = get_prompts(args.n_requests, args.prompt_kind)

    print(f"\n{'═'*60}")
    print(f"  Layer 18 — KV Cache Quantization Server Benchmark")
    print(f"  Model      : {args.model}")
    print(f"  Baseline   : kv_dtype={args.baseline_kv_dtype}")
    print(f"  Compare    : kv_dtype={args.compare_kv_dtype}")
    print(f"  Requests   : {args.n_requests}  |  Concurrency: {args.concurrency}")
    print(f"  Max tokens : {args.max_tokens}  |  mem-fraction: {args.mem_fraction}")
    print(f"{'═'*60}")

    # Warn about Qwen2.5 + FP8 KV issue
    if "qwen2.5" in args.model.lower() and args.compare_kv_dtype in ("fp8_e4m3", "fp8_e5m2"):
        print(
            "\n  ⚠ WARNING: Qwen2.5 has known outlier attention heads that cause\n"
            "    repetitive/broken output with FP8 KV cache (uncalibrated scale=1.0).\n"
            "    The comparison run may show degraded quality.\n"
            "    Consider: --model Qwen/Qwen3-1.7B  OR  --compare-kv-dtype auto\n"
        )

    # ── Baseline run ──
    baseline = run_one(
        args,
        kv_dtype = args.baseline_kv_dtype,
        label    = f"baseline ({args.baseline_kv_dtype})",
        prompts  = prompts,
    )

    if args.no_compare:
        print(f"\n  Baseline only (--no-compare). Throughput: {baseline.throughput_tps:.1f} tok/s")
        return

    # ── Comparison run ──
    compare = run_one(
        args,
        kv_dtype = args.compare_kv_dtype,
        label    = f"compare ({args.compare_kv_dtype})",
        prompts  = prompts,
    )

    # ── Report ──
    print_comparison(baseline, compare)

    out_path = Path(__file__).parent / args.output
    write_markdown(baseline, compare, str(out_path))

    # ── Interpretation guide ──
    print(
        "\n  INTERPRETATION GUIDE\n"
        "  ─────────────────────────────────────────────────────────\n"
        "  FP8 KV is FASTER than BF16 only when KV cache bandwidth is the bottleneck.\n"
        "  For a 7B model at low concurrency / short context: compute-bound → roughly equal.\n"
        "  For a 7B model at high concurrency / long context: KV-bound → FP8 wins (fewer HBM reads).\n\n"
        "  WHAT TO DO NEXT:\n"
        "    • Re-run with --concurrency 8 --max-tokens 400 to stress the KV pool harder.\n"
        "    • Re-run with --n-requests 48 to see the KV pool fill up and evictions begin.\n"
        "    • If FP8 is slower: likely compute-bound at this batch size; try a larger model.\n"
        "    • If output quality degrades in FP8: model architecture incompatibility.\n"
    )


if __name__ == "__main__":
    main()
