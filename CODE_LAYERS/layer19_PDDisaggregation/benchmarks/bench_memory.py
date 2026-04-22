"""
bench_memory.py — KV cache memory capacity analysis (no model weights needed).

Reads the model's config.json from HuggingFace (or a local path) and computes
exactly how many tokens fit in GPU VRAM at BF16, FP8, and FP4 KV cache dtypes.

Answers:
  1. How many KV cache tokens fit on your GPU per dtype?
  2. At a fixed context length, how many concurrent requests can you serve?
  3. What is the decode memory bandwidth reduction per dtype?
  4. How does FP8 KV + HiCache interact (PCIe transfer cost)?

Usage:
    python bench_memory.py
    python bench_memory.py --model Qwen/Qwen3-1.7B --kv-fraction 0.75
    python bench_memory.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --kv-fraction 0.8
    python bench_memory.py --num-layers 28 --num-kv-heads 8 --head-dim 128  # manual
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KV cache memory capacity analysis — no GPU required"
    )
    p.add_argument("--model",       default="Qwen/Qwen3-1.7B",
                   help="HuggingFace model ID or local path (reads config.json only)")
    p.add_argument("--kv-fraction", type=float, default=0.75,
                   help="Fraction of free VRAM allocated to KV pool (default: 0.75)")
    p.add_argument("--page-size",   type=int,   default=16,
                   help="KV cache page size in tokens (default: 16)")
    p.add_argument("--context-lengths", nargs="+", type=int,
                   default=[512, 1024, 2048, 4096, 8192],
                   help="Context lengths to show concurrency table for")

    # Manual override (skip model download)
    p.add_argument("--num-layers",   type=int, default=None,
                   help="Override: number of transformer layers")
    p.add_argument("--num-kv-heads", type=int, default=None,
                   help="Override: number of KV heads per layer")
    p.add_argument("--head-dim",     type=int, default=None,
                   help="Override: head dimension")
    p.add_argument("--vram-gb",      type=float, default=None,
                   help="Override: total GPU VRAM in GB (skips nvidia-smi)")
    p.add_argument("--weight-gb",    type=float, default=None,
                   help="Override: model weight size in GB (skips loading)")
    return p.parse_args()


# ── Model config loading ──────────────────────────────────────────────────────

@dataclass
class ModelDims:
    num_layers:   int
    num_kv_heads: int
    head_dim:     int
    model_id:     str


def load_model_dims(model_id: str) -> ModelDims:
    """Load just the config.json — no weights downloaded."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        num_layers   = getattr(cfg, "num_hidden_layers", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads",
                       getattr(cfg, "num_attention_heads", None))
        head_dim     = getattr(cfg, "head_dim", None)

        if head_dim is None:
            n_heads  = getattr(cfg, "num_attention_heads", None)
            hidden   = getattr(cfg, "hidden_size", None)
            if n_heads and hidden:
                head_dim = hidden // n_heads

        if not all([num_layers, num_kv_heads, head_dim]):
            raise ValueError(
                f"Could not extract dims from config: "
                f"layers={num_layers} kv_heads={num_kv_heads} head_dim={head_dim}"
            )

        return ModelDims(
            num_layers   = num_layers,
            num_kv_heads = num_kv_heads,
            head_dim     = head_dim,
            model_id     = model_id,
        )

    except ImportError:
        print("WARNING: transformers not installed. Use --num-layers / --num-kv-heads / --head-dim")
        sys.exit(1)
    except Exception as e:
        print(f"WARNING: Could not load config for {model_id}: {e}")
        print("         Use --num-layers / --num-kv-heads / --head-dim to specify manually.")
        sys.exit(1)


# ── GPU memory detection ──────────────────────────────────────────────────────

def get_gpu_memory_gb() -> Tuple[float, float]:
    """Returns (total_gb, free_gb). Falls back to defaults if torch unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU")
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        free_bytes, _ = torch.cuda.mem_get_info()
        return total_bytes / 1024**3, free_bytes / 1024**3
    except Exception:
        print("WARNING: Could not query GPU memory. Using --vram-gb if provided, else 16 GB.")
        return 16.0, 14.0


def estimate_weight_size_gb(dims: ModelDims) -> float:
    """
    Rough BF16 weight estimate: 2 bytes × total_parameters.
    Uses a simple approximation based on hidden_size and layers.
    This is a lower bound — actual models are larger due to embeddings + FFN.
    """
    # hidden_size ≈ num_kv_heads * head_dim * (num_q_heads / num_kv_heads)
    # For GQA models (Qwen3-1.7B: 16Q/8KV heads), hidden_size ≈ 2048
    # We approximate: ~12 × hidden_size² × num_layers parameters
    hidden_size = dims.num_kv_heads * dims.head_dim  # underestimate; real = Q heads * head_dim
    # Rough parameter count: 12 * H^2 * L (attention + FFN)
    # For Qwen3-1.7B: hidden=2048, this gives ~12 × 2048² × 28 ≈ 1.4B → 2.9GB BF16 (too low)
    # Better: look up common sizes
    known = {
        "0.6b": 1.2, "0.6B": 1.2,
        "1.7b": 3.4, "1.7B": 3.4,
        "3b":   6.0, "3B":   6.0,
        "7b":  14.0, "7B":  14.0,
        "8b":  16.0, "8B":  16.0,
        "14b": 28.0, "14B": 28.0,
        "32b": 64.0, "32B": 64.0,
        "72b": 144.0, "72B": 144.0,
    }
    model_lower = dims.model_id.lower()
    for key, gb in known.items():
        if key.lower() in model_lower:
            return gb
    # Fallback: rough formula
    return (12 * hidden_size**2 * dims.num_layers * 2) / 1024**3


# ── KV cache math ─────────────────────────────────────────────────────────────

@dataclass
class DtypeSpec:
    name:       str
    bytes:      float   # bytes per KV element (0.5 for FP4)
    label:      str
    production: bool    # True if in production SGLang/vLLM


KV_DTYPES: List[DtypeSpec] = [
    DtypeSpec("BF16",       2.0,  "bf16 (baseline)",              production=True),
    DtypeSpec("FP8_e4m3",   1.0,  "fp8_e4m3 (SGLang/vLLM)",      production=True),
    DtypeSpec("FP8_e5m2",   1.0,  "fp8_e5m2 (SGLang/vLLM)",      production=True),
    DtypeSpec("INT8_PTH",   1.04, "int8_per_token_head (+3.1%)",  production=True),   # 1 + 3.1% scale overhead
    DtypeSpec("NVFP4",      0.5,  "nvfp4 (Blackwell only)",       production=False),
    DtypeSpec("KIVI_INT2",  0.25, "INT2 KIVI (research)",         production=False),
]


def bytes_per_token(dims: ModelDims, dtype_bytes: float) -> int:
    """Total bytes to store one token's KV across all layers."""
    return int(dims.num_layers * 2 * dims.num_kv_heads * dims.head_dim * dtype_bytes)


def token_capacity(
    dims:          ModelDims,
    dtype_bytes:   float,
    free_vram_gb:  float,
    kv_fraction:   float,
    page_size:     int,
) -> int:
    """Number of tokens that fit in the KV pool (page-aligned)."""
    pool_bytes  = free_vram_gb * 1024**3 * kv_fraction
    bpt         = bytes_per_token(dims, dtype_bytes)
    raw_tokens  = int(pool_bytes / bpt)
    # Round down to page boundary
    return (raw_tokens // page_size) * page_size


def max_concurrency(total_tokens: int, context_len: int) -> int:
    """How many concurrent requests fit given total token capacity."""
    return max(0, total_tokens // context_len)


def decode_bandwidth_gb_per_step(dims: ModelDims, dtype_bytes: float, n_tokens: int) -> float:
    """
    Memory bandwidth consumed loading KV cache in one decode step.
    Each decode step: load all KV for all active tokens across all layers.
    """
    total_bytes = dims.num_layers * 2 * dims.num_kv_heads * dims.head_dim * dtype_bytes * n_tokens
    return total_bytes / 1024**3


# ── Printing ──────────────────────────────────────────────────────────────────

def _col(s: str, w: int, align: str = ">") -> str:
    return f"{{:{align}{w}}}".format(s)


def print_capacity_table(
    dims:         ModelDims,
    free_vram_gb: float,
    total_vram_gb: float,
    weight_gb:    float,
    kv_fraction:  float,
    page_size:    int,
) -> None:
    bar = "═" * 78
    sep = "─" * 78

    pool_gb = free_vram_gb * kv_fraction

    print(f"\n{bar}")
    print(f"  Layer 18 — KV Cache Memory Analysis")
    print(f"  Model : {dims.model_id}")
    print(f"  Dims  : {dims.num_layers} layers × {dims.num_kv_heads} KV heads × {dims.head_dim} head_dim")
    print(f"  GPU   : {total_vram_gb:.1f} GB total  |  {free_vram_gb:.1f} GB free after weights")
    print(f"          KV fraction {kv_fraction:.0%}  →  {pool_gb:.1f} GB allocated to KV pool")
    print(bar)

    # ── Per-dtype capacity ──
    print(f"\n  {'KV dtype':<22} {'Bytes/tok':>10} {'KV pool tokens':>16} {'vs BF16':>8}  {'Status'}")
    print(f"  {sep[2:]}")

    bf16_toks = None
    for dt in KV_DTYPES:
        toks = token_capacity(dims, dt.bytes, free_vram_gb, kv_fraction, page_size)
        if bf16_toks is None:
            bf16_toks = toks

        ratio_str = f"{toks / bf16_toks:.2f}×" if bf16_toks else "—"
        status    = "PRODUCTION" if dt.production else "research  "
        bpt       = bytes_per_token(dims, dt.bytes)

        print(
            f"  {dt.label:<22} "
            f"{bpt:>10,} "
            f"{toks:>16,} "
            f"{ratio_str:>8}  "
            f"{status}"
        )

    print()


def print_concurrency_table(
    dims:          ModelDims,
    free_vram_gb:  float,
    kv_fraction:   float,
    page_size:     int,
    context_lens:  List[int],
) -> None:
    bar = "═" * 78
    print(f"\n  Max concurrent requests at fixed context length")
    print(f"  (how many parallel users can you serve before running out of KV cache?)\n")

    # Header
    dtype_labels = ["BF16", "FP8_e4m3", "INT8_PTH", "NVFP4"]
    selected = [dt for dt in KV_DTYPES if dt.name in dtype_labels]

    w0 = 14
    w1 = 12
    header = f"  {'Context len':<{w0}}" + "".join(_col(dt.name, w1) for dt in selected)
    print(header)
    print(f"  {'─' * (w0 + w1 * len(selected))}")

    for ctx in context_lens:
        row = f"  {ctx:>{w0},} tokens"
        for dt in selected:
            toks = token_capacity(dims, dt.bytes, free_vram_gb, kv_fraction, page_size)
            n    = max_concurrency(toks, ctx)
            row += _col(str(n), w1)
        print(row)

    print()


def print_bandwidth_table(dims: ModelDims, context_lens: List[int]) -> None:
    print(f"\n  Decode HBM bandwidth consumed per step (GB loaded from GPU memory)")
    print(f"  Lower = faster decode. FP8 halves the bandwidth vs BF16.\n")

    selected_names = ["BF16", "FP8_e4m3", "KIVI_INT2"]
    selected = [dt for dt in KV_DTYPES if dt.name in selected_names]

    w0 = 18
    w1 = 14
    header = f"  {'Tokens in flight':<{w0}}" + "".join(_col(dt.name, w1) for dt in selected)
    print(header)
    print(f"  {'─' * (w0 + w1 * len(selected))}")

    for ctx in context_lens:
        row = f"  {ctx:>{w0},}"
        for dt in selected:
            bw = decode_bandwidth_gb_per_step(dims, dt.bytes, ctx)
            row += _col(f"{bw:.3f} GB", w1)
        print(row)

    print()


def print_hicache_note(dims: ModelDims) -> None:
    """
    Layer 17 (HiCache) + Layer 18 (FP8 KV) interaction note.
    FP8 halves PCIe transfer cost when offloading to CPU RAM.
    """
    PCIE_GB_S = 32.0   # PCIe 4.0 x16 bidirectional

    print(f"\n  HiCache interaction (Layer 17 + Layer 18 composability)")
    print(f"  PCIe 4.0 x16 bandwidth ≈ {PCIE_GB_S:.0f} GB/s\n")

    for ctx in [1024, 4096, 16384]:
        bf16_bw  = decode_bandwidth_gb_per_step(dims, 2.0, ctx)
        fp8_bw   = decode_bandwidth_gb_per_step(dims, 1.0, ctx)
        bf16_ms  = bf16_bw  / PCIE_GB_S * 1000
        fp8_ms   = fp8_bw   / PCIE_GB_S * 1000
        print(f"  {ctx:>6,} tokens:  BF16 PCIe load {bf16_ms:5.1f}ms/step  "
              f"→  FP8 {fp8_ms:4.1f}ms/step  ({fp8_ms/bf16_ms:.0%} of BF16 cost)")

    print(
        "\n  With FP8 KV, the κ_crit (PCIe breakeven) point effectively doubles:\n"
        "  you can offload 2× as many tokens before PCIe becomes the bottleneck.\n"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Resolve model dims ──
    if args.num_layers and args.num_kv_heads and args.head_dim:
        dims = ModelDims(
            num_layers   = args.num_layers,
            num_kv_heads = args.num_kv_heads,
            head_dim     = args.head_dim,
            model_id     = args.model,
        )
        print(f"Using manual dims: layers={dims.num_layers} "
              f"kv_heads={dims.num_kv_heads} head_dim={dims.head_dim}")
    else:
        print(f"Loading config for {args.model} …", end=" ", flush=True)
        dims = load_model_dims(args.model)
        print("done")

    # ── Resolve GPU memory ──
    if args.vram_gb:
        total_vram_gb = args.vram_gb
        # Assume weights take ~60% of non-KV VRAM
        weight_gb     = args.weight_gb or estimate_weight_size_gb(dims)
        free_vram_gb  = total_vram_gb - weight_gb
    else:
        total_vram_gb, free_vram_gb = get_gpu_memory_gb()
        weight_gb = args.weight_gb or estimate_weight_size_gb(dims)

    if free_vram_gb <= 0:
        print(f"ERROR: Negative free VRAM ({free_vram_gb:.1f} GB). "
              f"Is the model already loaded? Use --vram-gb and --weight-gb to override.")
        sys.exit(1)

    print(f"GPU: {total_vram_gb:.1f} GB total, {free_vram_gb:.1f} GB free "
          f"(weights estimated at {weight_gb:.1f} GB)\n")

    # ── Run analysis ──
    print_capacity_table(
        dims, free_vram_gb, total_vram_gb, weight_gb,
        args.kv_fraction, args.page_size
    )
    print_concurrency_table(
        dims, free_vram_gb, args.kv_fraction,
        args.page_size, args.context_lengths
    )
    print_bandwidth_table(dims, args.context_lengths)
    print_hicache_note(dims)

    # ── Key takeaway ──
    bf16_toks = token_capacity(dims, 2.0, free_vram_gb, args.kv_fraction, args.page_size)
    fp8_toks  = token_capacity(dims, 1.0, free_vram_gb, args.kv_fraction, args.page_size)
    print(
        f"  Summary for {args.model}:\n"
        f"    BF16 KV: {bf16_toks:,} tokens in pool  →  fits {bf16_toks // 2048:,} reqs at 2048-token context\n"
        f"    FP8 KV:  {fp8_toks:,} tokens in pool  →  fits {fp8_toks // 2048:,} reqs at 2048-token context\n"
        f"    FP8 KV gives {fp8_toks / bf16_toks:.2f}× more KV cache capacity\n"
    )


if __name__ == "__main__":
    main()
