"""
Layer 13 — Benchmark: speculative decoding vs. target-only greedy decode.

Runs a fixed set of prompts through two paths:
  1. Target-only:  standard greedy decode using Qwen/Qwen3-1.7B alone.
  2. Spec-decode:  SpecRunner with Qwen/Qwen3-0.6B draft + Qwen/Qwen3-1.7B target,
                   N draft tokens per verify step.

Metrics collected:
  - Tokens / second  (output tokens / wall time)
  - Time to first token (TTFT) in ms
  - Acceptance rate   (for spec decode)
  - Tokens per spec step  (theoretical max = N+1)
  - Speedup ratio  (spec tps / baseline tps)

Usage:
    python benchmark.py
    python benchmark.py --n-prompts 10 --max-tokens 100 --n-drafts 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).parent))

from model_runner import ModelRunner, DTYPE
from request import Req, ReqStatus
from spec_runner import SpecRunner
from tokenizer import Tokenizer

logging.basicConfig(
    level  = logging.WARNING,   # suppress INFO during benchmark
    format = "%(levelname)s %(name)s — %(message)s",
)

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Layer 13 speculative decoding benchmark"
)
parser.add_argument("--target",     default="Qwen/Qwen3-1.7B",
                    help="Target model path or HF id")
parser.add_argument("--draft",      default="Qwen/Qwen3-0.6B",
                    help="Draft model path or HF id")
parser.add_argument("--n-prompts",  type=int, default=8,
                    help="Number of prompts to benchmark")
parser.add_argument("--max-tokens", type=int, default=100,
                    help="Max new tokens per prompt")
parser.add_argument("--n-drafts",   type=int, default=5,
                    help="N: draft tokens per spec step")
parser.add_argument("--page-size",  type=int, default=16,
                    help="KV page size")
parser.add_argument("--output",     default="benchmark.md",
                    help="Markdown output file")
args = parser.parse_args()


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPTS = [
    "The capital of France is",
    "The theory of relativity states that",
    "In machine learning, gradient descent is",
    "Python is a programming language that was",
    "The human brain contains approximately",
    "Water boils at one hundred degrees Celsius because",
    "The largest planet in our solar system is",
    "Photosynthesis is the process by which",
    "The history of artificial intelligence began in",
    "Deep neural networks are powerful because",
    "The speed of light in a vacuum is approximately",
    "Transformers in NLP work by using attention to",
]


def get_prompts(n: int) -> List[str]:
    return (PROMPTS * ((n // len(PROMPTS)) + 1))[:n]


# ── Req factory ───────────────────────────────────────────────────────────────

def _make_req(rid: str, input_ids: List[int], max_new_tokens: int) -> Req:
    loop = asyncio.new_event_loop()
    return Req(
        rid            = rid,
        input_ids      = input_ids,
        max_new_tokens = max_new_tokens,
        temperature    = 0.0,
        future         = loop.create_future(),
    )


# ── Baseline: target-only greedy ─────────────────────────────────────────────

def _encode(tok: Tokenizer, prompt: str) -> List[int]:
    """Tokenise prompt → plain Python list of int token IDs."""
    return tok.encode(prompt)[0].tolist()   # [1, L] tensor → [L] → list[int]


def run_baseline(
    runner:   ModelRunner,
    tok:      Tokenizer,
    prompts:  List[str],
    max_toks: int,
) -> dict:
    """Run all prompts through the target model with standard greedy decode."""
    ttfts:   List[float] = []
    n_out:   List[int]   = []

    t_wall_start = time.perf_counter()

    for i, prompt in enumerate(prompts):
        ids = _encode(tok, prompt)
        req = _make_req(f"base_{i}", ids, max_toks)
        req.fill_ids         = list(ids)
        req.extend_input_len = len(ids)
        req.kv_committed_len = 0

        t0 = time.perf_counter()
        runner.prefill_batch([req])
        ttft_ms = (time.perf_counter() - t0) * 1000

        while req.status == ReqStatus.RUNNING:
            runner.decode_step([req])

        ttfts.append(ttft_ms)
        n_out.append(len(req.output_ids))

    total_s = time.perf_counter() - t_wall_start
    total_out = sum(n_out)

    return {
        "mode"     : "target-only",
        "total_s"  : total_s,
        "tps"      : total_out / total_s,
        "n_out"    : n_out,
        "ttft_ms"  : ttfts,
        "acc_rate" : None,
        "tps_step" : None,
    }


# ── Spec decode ───────────────────────────────────────────────────────────────

def run_spec(
    runner:   SpecRunner,
    tok:      Tokenizer,
    prompts:  List[str],
    max_toks: int,
) -> dict:
    """Run all prompts through the speculative decoder."""
    ttfts:  List[float] = []
    n_out:  List[int]   = []

    runner.total_draft_tokens    = 0
    runner.total_accepted_tokens = 0
    runner.total_spec_steps      = 0

    t_wall_start = time.perf_counter()

    for i, prompt in enumerate(prompts):
        ids = _encode(tok, prompt)
        req = _make_req(f"spec_{i}", ids, max_toks)

        t0 = time.perf_counter()
        runner.prefill([req])
        ttft_ms = (time.perf_counter() - t0) * 1000

        while req.status == ReqStatus.RUNNING:
            runner.spec_decode_step([req])

        ttfts.append(ttft_ms)
        n_out.append(len(req.output_ids))

    total_s   = time.perf_counter() - t_wall_start
    total_out = sum(n_out)

    return {
        "mode"     : "spec-decode",
        "total_s"  : total_s,
        "tps"      : total_out / total_s,
        "n_out"    : n_out,
        "ttft_ms"  : ttfts,
        "acc_rate" : runner.acceptance_rate,
        "tps_step" : runner.tokens_per_step,
    }


# ── Pretty print ─────────────────────────────────────────────────────────────

def _p95(vals: List[float]) -> float:
    s = sorted(vals)
    return s[int(0.95 * len(s))]


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals)


def print_results(base: dict, spec: dict, n: int, N: int) -> None:
    speedup = spec["tps"] / base["tps"]

    bar = "═" * 62
    print(f"\n{bar}")
    print(f"  Layer 13 — Speculative Decoding Benchmark")
    print(f"  Prompts: {n}   Max tokens: {args.max_tokens}   N (drafts): {N}")
    print(bar)
    print(f"  {'Metric':<30} {'Target-only':>12} {'Spec-decode':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Total wall time (s)':<30} {base['total_s']:>12.2f} {spec['total_s']:>12.2f}")
    print(f"  {'Output tokens/sec':<30} {base['tps']:>12.1f} {spec['tps']:>12.1f}")
    print(f"  {'TTFT avg (ms)':<30} {_mean(base['ttft_ms']):>12.0f} {_mean(spec['ttft_ms']):>12.0f}")
    print(f"  {'TTFT p95 (ms)':<30} {_p95(base['ttft_ms']):>12.0f} {_p95(spec['ttft_ms']):>12.0f}")
    print(f"  {'Avg output tokens/req':<30} {_mean(base['n_out']):>12.1f} {_mean(spec['n_out']):>12.1f}")

    if spec["acc_rate"] is not None:
        print(f"  {'Acceptance rate':<30} {'—':>12} {spec['acc_rate']:>11.1%}")
    if spec["tps_step"] is not None:
        print(f"  {f'Tokens/step (max={N+1})':<30} {'—':>12} {spec['tps_step']:>12.2f}")

    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Speedup (spec / baseline)':<30} {'1.00×':>12} {speedup:>11.2f}×")
    print(bar)

    if speedup >= 1.5:
        verdict = f"✓ {speedup:.2f}× speedup — speculative decoding helps significantly"
    elif speedup >= 1.0:
        verdict = f"~ {speedup:.2f}× speedup — modest gain (higher N or better draft may help)"
    else:
        verdict = f"✗ {speedup:.2f}× — spec decode slower (overhead > gain; try larger N)"
    print(f"  {verdict}")
    print(bar)


def write_markdown(base: dict, spec: dict, n: int, N: int) -> None:
    speedup = spec["tps"] / base["tps"]
    md_path = Path(args.output)

    section = f"""## Layer 13 — Speculative Decoding Benchmark

**Config**: {n} prompts · max_tokens={args.max_tokens} · N={N} draft tokens · page_size={args.page_size}
**Models**: target=`{args.target}`  draft=`{args.draft}`

| Metric | Target-only | Spec-decode |
|--------|-------------|-------------|
| Total wall time | {base['total_s']:.2f}s | {spec['total_s']:.2f}s |
| Output tok/s | {base['tps']:.1f} | {spec['tps']:.1f} |
| TTFT avg / p95 | {_mean(base['ttft_ms']):.0f}ms / {_p95(base['ttft_ms']):.0f}ms | {_mean(spec['ttft_ms']):.0f}ms / {_p95(spec['ttft_ms']):.0f}ms |
| Avg output tokens | {_mean(base['n_out']):.1f} | {_mean(spec['n_out']):.1f} |
| Acceptance rate | — | {spec['acc_rate']:.1%} |
| Tokens per step | — | {spec['tps_step']:.2f} (max={N+1}) |
| **Speedup** | 1.00× | **{speedup:.2f}×** |
"""
    md_path.write_text(section.strip())
    print(f"\n  Results written → {md_path.resolve()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    N         = args.n_drafts
    max_toks  = args.max_tokens
    prompts   = get_prompts(args.n_prompts)

    print(f"\nTarget : {args.target}")
    print(f"Draft  : {args.draft}")
    print(f"N={N} draft tokens/step · {len(prompts)} prompts · max_tokens={max_toks}\n")

    # ── Phase 1: baseline target-only (load target alone → full KV budget) ────
    # Run baseline first so we don't compete with the draft model for VRAM.
    print(f"[1/2] Baseline — target-only greedy  ({len(prompts)} prompts)")
    print(f"      Loading {args.target} …")
    base_runner = ModelRunner(
        args.target,
        page_size             = args.page_size,
        enable_prefix_caching = False,
    )
    tok = base_runner.tokenizer

    # Warmup
    _w1 = _make_req("warmup_base", _encode(tok, "Hello"), 5)
    _w1.fill_ids = list(_w1.input_ids)
    _w1.extend_input_len = len(_w1.input_ids)
    _w1.kv_committed_len = 0
    base_runner.prefill_batch([_w1])
    while _w1.status == ReqStatus.RUNNING:
        base_runner.decode_step([_w1])
    torch.cuda.synchronize()

    base_results = run_baseline(base_runner, tok, prompts, max_toks)
    print(f"      done in {base_results['total_s']:.1f}s  "
          f"({base_results['tps']:.1f} tok/s)\n")

    # Free target model before loading both models for spec decode
    del base_runner
    torch.cuda.empty_cache()

    # ── Phase 2: speculative decode (target + draft, reduced KV fractions) ────
    print(f"[2/2] Spec-decode  N={N}  ({len(prompts)} prompts)")
    print(f"      Loading {args.target} + {args.draft} …")
    spec_runner = SpecRunner(
        target_path = args.target,
        draft_path  = args.draft,
        N           = N,
        page_size   = args.page_size,
    )

    # Warmup
    _w2 = _make_req("warmup_spec", _encode(tok, "Hello"), 5)
    spec_runner.prefill([_w2])
    while _w2.status == ReqStatus.RUNNING:
        spec_runner.spec_decode_step([_w2])
    torch.cuda.synchronize()

    spec_results = run_spec(spec_runner, tok, prompts, max_toks)
    print(f"      done in {spec_results['total_s']:.1f}s  "
          f"({spec_results['tps']:.1f} tok/s)  "
          f"acc={spec_results['acc_rate']:.1%}")

    # ── Report ────────────────────────────────────────────────────────────────
    print_results(base_results, spec_results, len(prompts), N)
    write_markdown(base_results, spec_results, len(prompts), N)


if __name__ == "__main__":
    main()
