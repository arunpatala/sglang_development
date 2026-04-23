# Layer 20 — Multi-LoRA Serving

Builds on Layer 19 (PD Disaggregation) by adding **multi-adapter LoRA serving**: multiple
LoRA adapters can be active simultaneously, with different requests in the same batch using
different adapters — all sharing one base model on a single GPU.

Based on two papers:
- **S-LoRA** — "Serving Thousands of Concurrent LoRA Adapters" (OSDI 2024)
- **Punica** — "Multi-Tenant LoRA Serving" (MLSys 2024)

---

## The Problem: One Base Model, Many Fine-Tuned Variants

A production API often needs to serve many fine-tuned variants of the same base model:
a "finance" adapter, a "code" adapter, a "medical" adapter, etc. The naive approach —
one GPU per adapter — is prohibitively expensive.

LoRA makes this tractable: each adapter adds only ~1–10 MB of weights (rank-r matrices)
on top of a shared base model. The challenge is serving them **efficiently in a single
batched forward pass** where different requests use different adapters.

---

## What is LoRA?

LoRA (Low-Rank Adaptation) replaces a weight matrix `W` with `W + B·A·scaling`:

```
output = x @ W.T                          ← base model (frozen)
       + (x @ A.T) @ B.T * scaling        ← LoRA delta (adapter-specific)

where:
  A: [rank, input_dim]   — "shrink" matrix, projects down to low rank
  B: [output_dim, rank]  — "expand" matrix, projects back up
  scaling = lora_alpha / r
```

The base model weights never change. Only A and B are adapter-specific.
At rank=16, a 7B model's attention layer adds ~0.1% extra parameters.

---

## What Changes from Layer 19

| Component | Layer 19 | Layer 20 |
|-----------|----------|----------|
| Model | Single base model | Base model + N LoRA adapters |
| Request | No adapter field | `req.lora_id` — which adapter to use |
| `model_runner.py` | No LoRA | `LoRAManager` initialized at startup |
| `lora/` package | — | **New**: config, adapter, memory pool, manager, Triton kernels |
| `model/attention.py` | Base QKV proj | Base QKV + LoRA delta via SGEMM |
| `model/mlp.py` | Base gate/up/down | Base gate/up/down + LoRA delta |
| Batch metadata | No adapter info | `weight_indices[B]` — slot per token |
| `verify_lora.py` | — | **New**: correctness + multi-adapter tests |
| `config.yml` | No LoRA config | `lora_paths`, `max_loras_per_batch`, `max_lora_rank` |

---

## Architecture

### The 5-Layer Stack

```
LoRAConfig          — reads adapter_config.json (rank, alpha, target_modules)
    │
LoRAAdapter         — loads A/B weights from checkpoint into CPU memory
    │                 normalizes: q+k+v → qkv_proj, gate+up → gate_up_proj
    │
LoRAMemoryPool      — pre-allocated GPU buffers for max_loras_per_batch adapters
    │                 A_buffer["qkv_proj"][layer]: [max_loras, 3*rank, hidden]
    │                 B_buffer["qkv_proj"][layer]: [max_loras, output_dim, rank]
    │                 LRU eviction when pool is full
    │
LoRAManager         — orchestrates load/evict/prepare per batch
    │                 weight_indices[i] = which pool slot token i uses
    │
Triton SGEMM        — batched segmented matmul: all adapters in one kernel launch
```

### GPU Memory Pool (S-LoRA insight)

Instead of loading adapter weights per-request, a fixed-size GPU pool holds
`max_loras_per_batch` adapter slots simultaneously:

```
A_buffer["qkv_proj"]:  [max_loras, 3*rank, hidden_dim]   ← all slots pre-allocated
B_buffer["qkv_proj"]:  [max_loras, output_dim, rank]

slot 0: adapter "finance-lora"   ← loaded
slot 1: adapter "code-lora"      ← loaded
slot 2: EMPTY                    ← available
slot 3: adapter "medical-lora"   ← loaded
```

When a new adapter is needed and the pool is full, the LRU adapter is evicted
(its slot zeroed) and the new adapter is copied in from CPU memory.

### Segmented GEMM (Punica insight)

The key kernel innovation: instead of one matmul per adapter, a single Triton kernel
handles all adapters in the batch simultaneously.

```
Batch tokens: [tok0(finance), tok1(finance), tok2(code), tok3(base), tok4(code)]

weight_indices: [0, 0, 1, -1, 1]   ← pool slot per token

SGEMM kernel:
  segment 0: tok0,tok1 × A_buffer[slot=0]   ← finance adapter
  segment 1: tok2,tok4 × A_buffer[slot=1]   ← code adapter
  segment 2: tok3 × zeros                   ← base model (no LoRA)

All segments run in parallel — one threadblock per segment.
```

For decode (1 token per request), tokens are **sorted by adapter** first, then
the kernel processes all tokens for adapter 0, then adapter 1, etc. — maximizing
memory coalescing and avoiding warp divergence.

### Forward Pass (per layer)

```
x (activations, shape [s, hidden])
  │
  ├─► base Linear(x)                              → base_output [s, out_dim]
  │
  └─► LoRA path:
        sgemm_lora_a(x, A_buffer[layer])          → [s, rank]
        sgemm_lora_b(intermediate, B_buffer[layer]) → delta [s, out_dim]
        output = base_output + delta * scaling
```

For QKV (3 projections fused into one kernel call):
```
A: [num_loras, 3*rank, hidden]   ← q/k/v stacked
B: [num_loras, q_dim+2*kv_dim, rank]

sgemm_lora_a → [s, 3*rank]
qkv_lora_b_fwd → [s, q_dim+2*kv_dim]   ← specialized kernel handles the split
```

---

## Files

| File | Role |
|------|------|
| `lora/lora_config.py` | **New** — reads `adapter_config.json`: rank, alpha, target_modules |
| `lora/lora.py` | **New** — `LoRAAdapter`: loads weights to CPU, normalizes qkv/gate_up stacking |
| `lora/mem_pool.py` | **New** — `LoRAMemoryPool`: pre-allocated GPU A/B buffers, LRU eviction |
| `lora/lora_manager.py` | **New** — `LoRAManager`: load/unload adapters, prepare batch metadata |
| `lora/triton_ops/sgemm_lora_a.py` | **New** — Triton segmented GEMM for LoRA A (shrink) |
| `lora/triton_ops/sgemm_lora_b.py` | **New** — Triton segmented GEMM for LoRA B (expand) |
| `lora/triton_ops/qkv_lora_b.py` | **New** — specialized B kernel for fused QKV |
| `lora/triton_ops/gate_up_lora_b.py` | **New** — specialized B kernel for fused gate/up |
| `model/attention.py` | **Modified** — adds LoRA delta to QKV projection output |
| `model/mlp.py` | **Modified** — adds LoRA delta to gate_up and down projection output |
| `model_runner.py` | **Modified** — initializes `LoRAManager`, passes `lora_id` through batch |
| `request.py` | **Modified** — adds `lora_id: Optional[str]` field |
| `verify_lora.py` | **New** — correctness + multi-adapter tests |
| `config.yml` | **Modified** — `lora_paths`, `max_loras_per_batch`, `max_lora_rank` |

---

## Verify

```bash
python verify_lora.py
```

**Test 1** — Base model (no LoRA): output matches non-LoRA baseline  
**Test 2** — Single adapter: LoRA delta is non-zero, output differs from base  
**Test 3** — Multi-adapter batch: two adapters in same batch produce different outputs  
**Test 4** — Adapter eviction: loading adapter N+1 when pool has N slots evicts LRU correctly  
**Test 5** — Correctness: per-request LoRA output matches running each adapter in isolation  

---

## Benchmark

**Config**: 20 requests · 4 adapters · max_loras_per_batch=4 · rank=16 · page_size=16

| Metric | Base model only | Multi-LoRA (4 adapters) |
|--------|----------------|------------------------|
| Output tok/s | ~200 | ~185 (est.) |
| GPU memory overhead | 0 | ~4 × adapter_size |
| Adapter load time (cold) | — | ~50ms (CPU→GPU copy) |
| Adapter load time (warm) | — | ~0ms (already in pool) |

The throughput overhead is small because the SGEMM kernels are fused — no extra
kernel launches per adapter. The main cost is the additional matmuls for A and B,
which are cheap at low rank (r=16 vs hidden=1024 → 1.5% extra FLOPs).

---

## Key Concepts

### Weight Stacking (normalization)

PEFT checkpoints store separate `q_proj/lora_A`, `k_proj/lora_A`, `v_proj/lora_A`.
SGLang normalizes these into a single stacked tensor at load time:

```python
# On load (CPU, once):
qkv_lora_A = torch.cat([q_lora_A, k_lora_A, v_lora_A], dim=0)
# shape: [3*rank, hidden]  →  stored in A_buffer slot

# At inference (GPU, every step):
# One SGEMM call handles all three projections simultaneously
```

Same for `gate_proj` + `up_proj` → `gate_up_proj`.

### LRU Eviction

When `max_loras_per_batch=4` and a 5th adapter is needed:
1. Find the slot whose adapter was least recently used
2. Zero that slot's A/B buffers (prevents contamination)
3. Copy the new adapter's weights from CPU pinned memory → GPU slot
4. Update `uid_to_buffer_id` mapping

Pinned CPU memory (`pin_memory=True`) makes the CPU→GPU copy async and fast.

### Scaling

```
scaling = lora_alpha / r

# Example: alpha=16, r=16 → scaling=1.0
# Example: alpha=32, r=16 → scaling=2.0
```

Applied as a scalar multiply on the LoRA B output before adding to base output.
Stored per-slot in `scalings[max_loras_per_batch]` tensor, read by the SGEMM kernel.

---

## SGLang Alignment

| Concept | SGLang location | Layer 20 |
|---------|----------------|----------|
| `LoRAAdapter` weight loading | `srt/lora/lora.py` | `lora/lora.py` |
| `LoRAMemoryPool` GPU buffers | `srt/lora/mem_pool.py` | `lora/mem_pool.py` |
| `LoRAManager` orchestration | `srt/lora/lora_manager.py` | `lora/lora_manager.py` |
| SGEMM LoRA A Triton kernel | `srt/lora/triton_ops/sgemm_lora_a.py` | `lora/triton_ops/sgemm_lora_a.py` |
| SGEMM LoRA B Triton kernel | `srt/lora/triton_ops/sgemm_lora_b.py` | `lora/triton_ops/sgemm_lora_b.py` |
| QKV fused LoRA B kernel | `srt/lora/triton_ops/qkv_lora_b.py` | `lora/triton_ops/qkv_lora_b.py` |
| LRU eviction policy | `srt/lora/eviction_policy.py` | `lora/eviction_policy.py` |
| `--lora-paths` CLI flag | `srt/server_args.py` | `config.yml: lora_paths` |
| `--max-loras-per-batch` | `srt/server_args.py` | `config.yml: max_loras_per_batch` |
