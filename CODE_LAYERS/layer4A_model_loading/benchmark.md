# Layer 4 (Custom Model Init) — Benchmark Results

**Date:** 2026-04-18
**Model:** Qwen/Qwen3-0.6B (weights loaded via safetensors, no AutoModelForCausalLM)
**Hardware:** NVIDIA GeForce RTX 4060 Ti
**Requests:** 20 ShareGPT (seed=42)
**Max new tokens:** 128

---

## Batch Size Sweep

| Batch size | Output tokens | Wall time | Tok/s | Avg TTFT | Avg TPOT | Status |
|-----------|---------------|-----------|-------|----------|----------|--------|
| 8 | 2390 | 17.1s | 139.6 | 544 | 16 | ok |

Best batch size: **8** → **139.6 tok/s**

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
