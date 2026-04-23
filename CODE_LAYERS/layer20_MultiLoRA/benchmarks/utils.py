"""
Shared helpers for layer18 KV cache quantization benchmarks.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import List, Optional


# ── Prompts ───────────────────────────────────────────────────────────────────
# A mix of short and long completions to stress both prefill and decode paths.

SHORT_PROMPTS = [
    "The capital of France is",
    "Water boils at",
    "The speed of light is",
    "Python was created by",
    "The largest planet is",
]

LONG_PROMPTS = [
    "Explain in detail how the transformer attention mechanism works, including the mathematical formulation of queries, keys, and values:",
    "Describe the history of machine learning from its origins in the 1950s through the deep learning revolution, covering key milestones and researchers:",
    "Write a detailed explanation of how modern GPU architectures are designed to accelerate matrix multiplication operations for deep learning:",
    "Explain the concept of gradient descent in neural network training, including variants like SGD, Adam, and their mathematical properties:",
    "Describe how distributed training works for large language models, including data parallelism, model parallelism, and pipeline parallelism:",
    "Explain the PagedAttention algorithm used in vLLM and SGLang for efficient KV cache management in LLM serving systems:",
    "What is quantization in the context of large language models? Explain the difference between INT8, FP8, and INT4 quantization:",
]

MIXED_PROMPTS = SHORT_PROMPTS + LONG_PROMPTS


def get_prompts(n: int, kind: str = "mixed") -> List[str]:
    pool = {"short": SHORT_PROMPTS, "long": LONG_PROMPTS, "mixed": MIXED_PROMPTS}[kind]
    return (pool * ((n // len(pool)) + 1))[:n]


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    label:        str
    kv_dtype:     str
    n_requests:   int
    max_tokens:   int
    concurrency:  int

    ttft_ms:      List[float] = field(default_factory=list)
    latency_ms:   List[float] = field(default_factory=list)
    output_tokens: List[int]  = field(default_factory=list)
    errors:        int = 0

    wall_s:       float = 0.0
    vram_before_mb: float = 0.0
    vram_after_mb:  float = 0.0

    @property
    def throughput_tps(self) -> float:
        total = sum(self.output_tokens)
        return total / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def ttft_avg(self) -> float:
        return statistics.mean(self.ttft_ms) if self.ttft_ms else 0.0

    @property
    def ttft_p95(self) -> float:
        if not self.ttft_ms:
            return 0.0
        s = sorted(self.ttft_ms)
        return s[int(0.95 * len(s))]

    @property
    def latency_avg(self) -> float:
        return statistics.mean(self.latency_ms) if self.latency_ms else 0.0

    @property
    def latency_p95(self) -> float:
        if not self.latency_ms:
            return 0.0
        s = sorted(self.latency_ms)
        return s[int(0.95 * len(s))]

    @property
    def avg_output_tokens(self) -> float:
        return statistics.mean(self.output_tokens) if self.output_tokens else 0.0

    @property
    def vram_delta_mb(self) -> float:
        return self.vram_after_mb - self.vram_before_mb

    @property
    def success_rate(self) -> float:
        total = len(self.ttft_ms) + self.errors
        return len(self.ttft_ms) / total if total > 0 else 0.0


# ── Table printing ────────────────────────────────────────────────────────────

def _col(val: str, width: int, align: str = ">") -> str:
    fmt = f"{{:{align}{width}}}"
    return fmt.format(val)


def print_comparison(baseline: BenchResult, compare: BenchResult) -> None:
    bar = "═" * 72
    sep = "─" * 72

    def speedup(a: float, b: float, lower_is_better: bool = False) -> str:
        if a == 0:
            return "  —  "
        ratio = b / a if not lower_is_better else a / b
        sign  = "+" if ratio >= 1.0 else ""
        return f"{sign}{ratio - 1:+.1%}"

    print(f"\n{bar}")
    print(f"  Layer 18 — KV Cache Quantization Benchmark")
    print(f"  Baseline : {baseline.label}  ({baseline.kv_dtype})")
    print(f"  Compare  : {compare.label}  ({compare.kv_dtype})")
    print(f"  Requests : {baseline.n_requests}   "
          f"Max tokens: {baseline.max_tokens}   "
          f"Concurrency: {baseline.concurrency}")
    print(bar)

    w0, w1, w2, w3 = 32, 14, 14, 10
    header = (
        f"  {_col('Metric', w0, '<')}"
        f"{_col('Baseline', w1)}"
        f"{_col('FP8 KV', w2)}"
        f"{_col('Delta', w3)}"
    )
    print(header)
    print(f"  {sep[2:]}")

    def row(label: str, base_val: str, cmp_val: str, delta: str = "") -> None:
        print(
            f"  {_col(label, w0, '<')}"
            f"{_col(base_val, w1)}"
            f"{_col(cmp_val, w2)}"
            f"{_col(delta, w3)}"
        )

    row("Wall time (s)",
        f"{baseline.wall_s:.2f}s",
        f"{compare.wall_s:.2f}s",
        speedup(baseline.wall_s, compare.wall_s, lower_is_better=True))

    row("Throughput (tok/s)",
        f"{baseline.throughput_tps:.1f}",
        f"{compare.throughput_tps:.1f}",
        speedup(baseline.throughput_tps, compare.throughput_tps))

    row("TTFT avg (ms)",
        f"{baseline.ttft_avg:.0f}",
        f"{compare.ttft_avg:.0f}",
        speedup(baseline.ttft_avg, compare.ttft_avg, lower_is_better=True))

    row("TTFT p95 (ms)",
        f"{baseline.ttft_p95:.0f}",
        f"{compare.ttft_p95:.0f}",
        speedup(baseline.ttft_p95, compare.ttft_p95, lower_is_better=True))

    row("Latency avg (ms)",
        f"{baseline.latency_avg:.0f}",
        f"{compare.latency_avg:.0f}",
        speedup(baseline.latency_avg, compare.latency_avg, lower_is_better=True))

    row("Avg output tokens",
        f"{baseline.avg_output_tokens:.1f}",
        f"{compare.avg_output_tokens:.1f}")

    row("VRAM delta (MB)",
        f"{baseline.vram_delta_mb:.0f}",
        f"{compare.vram_delta_mb:.0f}",
        speedup(baseline.vram_delta_mb, compare.vram_delta_mb, lower_is_better=True)
        if baseline.vram_delta_mb > 0 and compare.vram_delta_mb > 0 else "")

    row("Success rate",
        f"{baseline.success_rate:.1%}",
        f"{compare.success_rate:.1%}")

    print(f"  {sep[2:]}")

    # Verdict
    thr_ratio = compare.throughput_tps / baseline.throughput_tps if baseline.throughput_tps > 0 else 1.0
    if thr_ratio >= 1.15:
        verdict = f"✓ FP8 KV is {thr_ratio:.2f}× faster — KV bandwidth is the bottleneck at this batch size"
    elif thr_ratio >= 0.97:
        verdict = f"~ FP8 KV roughly equal ({thr_ratio:.2f}×) — compute-bound at this batch/context size"
    else:
        verdict = f"✗ FP8 KV slower ({thr_ratio:.2f}×) — check for accuracy issues (Qwen2.5 outlier heads?)"
    print(f"\n  {verdict}")
    print(bar)


def write_markdown(baseline: BenchResult, compare: BenchResult, path: str) -> None:
    thr_ratio = compare.throughput_tps / baseline.throughput_tps if baseline.throughput_tps > 0 else 1.0
    md = f"""## Layer 18 — KV Cache Quantization Benchmark

**Baseline**: `{baseline.label}` (`{baseline.kv_dtype}`)
**Compare**:  `{compare.label}` (`{compare.kv_dtype}`)
**Config**: {baseline.n_requests} requests · max_tokens={baseline.max_tokens} · concurrency={baseline.concurrency}

| Metric | {baseline.kv_dtype} | {compare.kv_dtype} |
|--------|---------|---------|
| Wall time | {baseline.wall_s:.2f}s | {compare.wall_s:.2f}s |
| Throughput (tok/s) | {baseline.throughput_tps:.1f} | {compare.throughput_tps:.1f} |
| TTFT avg / p95 (ms) | {baseline.ttft_avg:.0f} / {baseline.ttft_p95:.0f} | {compare.ttft_avg:.0f} / {compare.ttft_p95:.0f} |
| Latency avg (ms) | {baseline.latency_avg:.0f} | {compare.latency_avg:.0f} |
| Avg output tokens | {baseline.avg_output_tokens:.1f} | {compare.avg_output_tokens:.1f} |
| VRAM delta (MB) | {baseline.vram_delta_mb:.0f} | {compare.vram_delta_mb:.0f} |
| Throughput ratio | 1.00× | **{thr_ratio:.2f}×** |
"""
    with open(path, "w") as f:
        f.write(md.strip())
    print(f"\n  Results written → {path}")
