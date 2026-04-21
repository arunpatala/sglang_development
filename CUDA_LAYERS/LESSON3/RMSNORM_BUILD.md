# RMSNorm Step-by-Step Build

How the three lessons combine to build a production RMSNorm kernel.

---

## The Formula

```
RMSNorm(x, w) = x / sqrt(mean(x²) + ε) × w

where:
  x:   [batch, hidden_dim]  (fp16 input)
  w:   [hidden_dim]         (fp16 learned weight)
  ε:   small constant (1e-5) for numerical stability
  out: [batch, hidden_dim]  (fp16 output)
```

**Why RMSNorm instead of LayerNorm:**
- LayerNorm computes both mean and variance: 2 passes, more memory
- RMSNorm only needs the RMS (root-mean-square): 1 reduction pass
- Empirically performs as well as LayerNorm in transformer training
- Used in: Llama 2/3, Mistral, Qwen, DeepSeek, Falcon

---

## Building Up from Lessons 1–3

### Lesson 1 contribution: 128-bit vectorized loads

Without vectorized loads, each fp16 element is a separate 16-bit load instruction. With `uint4`, we load 8 fp16 elements in one 128-bit instruction.

```cuda
// 8 fp16 values in one LD.E.128 instruction:
uint4 raw = x[row * n_vec + i];       // n_vec = hidden_dim / 8
__half* vals = (__half*)&raw;
// vals[0] through vals[7] are the 8 fp16 values
```

**Impact:** 8× fewer load instructions per element, saturates memory bus.

### Lesson 2 contribution: warp shuffle sum

Each thread computes the sum-of-squares of its 8 elements. The warp then reduces 32 partial sums into one total using 5 XOR shuffles — no shared memory.

```cuda
float partial_sq = 0.f;
for (int j = 0; j < 8; ++j) {
    float v = __half2float(vals[j]);
    partial_sq += v * v;
}
// Warp reduce: 5 shuffles → every lane holds the warp's sum
partial_sq += __shfl_xor_sync(0xffffffff, partial_sq, 16);
partial_sq += __shfl_xor_sync(0xffffffff, partial_sq, 8);
partial_sq += __shfl_xor_sync(0xffffffff, partial_sq, 4);
partial_sq += __shfl_xor_sync(0xffffffff, partial_sq, 2);
partial_sq += __shfl_xor_sync(0xffffffff, partial_sq, 1);
// Now: partial_sq = sum of squares for 32*8=256 elements (one warp covers 256 fp16)
```

**Limit:** This only reduces within one warp (256 elements). For hidden_dim=4096, we need 512 uint4 loads → 16 warps → cross-warp reduction needed.

### Lesson 3 contribution: block reduce with smem

Each warp writes its partial sum to `smem[warp_id]`. One `__syncthreads()` ensures all writes complete. Warp 0 reads all partial sums from smem and does a second shuffle reduce.

```cuda
// Lane 0 of each warp writes to smem
if (lane == 0) smem[warp_id] = partial_sq;
__syncthreads();   // barrier 1

// Warp 0 reduces across all warp partial sums
float v = (warp_id == 0 && lane < n_warps) ? smem[lane] : 0.f;
if (warp_id == 0) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    // ... all 5 steps ...
    if (lane == 0) smem[0] = v;   // total sum of all squares
}
__syncthreads();   // barrier 2

float total_sq = smem[0];   // every thread has the block total
```

### Combining into RMSNorm

```
hidden_dim = 4096 example:

n_vec = 4096 / 8 = 512          (uint4 loads per row)
n_warps = 16                     (32 × 16 = 512 threads total)
Each thread handles: n_vec / (32 * n_warps) = 512 / 512 = 1 uint4 per pass

┌─────────────────────────────────────────────────────────────────┐
│  PASS 1: sum of squares                                         │
│                                                                 │
│  For i = tid to n_vec-1 step n_threads:                        │
│    uint4 raw = x[row * n_vec + i]         ← LD.E.128 (L1)     │
│    partial_sq += sum(half2float(raw[j])²) ← 8 fma ops          │
│                                                                 │
│  warp_reduce_sum(partial_sq)              ← 5 shfl.sync        │
│  smem[warp_id] = partial_sq              ← st.shared           │
│  __syncthreads()                          ← bar.sync            │
│  warp 0: reduce smem → smem[0]           ← ld.shared + shuffle │
│  __syncthreads()                          ← bar.sync            │
│                                                                 │
│  rms_rcp = rsqrtf(smem[0] / hidden + eps) ← MUFU.RSQ          │
└─────────────────────────────────────────────────────────────────┘
                        ↓ rms_rcp in every thread's register
┌─────────────────────────────────────────────────────────────────┐
│  PASS 2: normalize                                              │
│                                                                 │
│  For i = tid to n_vec-1 step n_threads:                        │
│    uint4 rx = x[row * n_vec + i]          ← LD.E.128 (GDDR6X) │
│    uint4 rw = weight[i]                   ← LD.E.128           │
│    out[j] = half(float(rx[j]) * rms_rcp * float(rw[j]))       │
│    uint4 ro → out[row * n_vec + i]        ← ST.E.128           │
└─────────────────────────────────────────────────────────────────┘
```

---

## HBM Access Accounting

For batch=B, hidden=H, dtype=fp16 (2 bytes/element):

| Operation | Bytes | Notes |
|---|---|---|
| Pass 1: read x | B × H × 2 | One full read of the input |
| Pass 2: read x | B × H × 2 | Second read — this is what fused add+norm eliminates |
| Pass 2: read w | H × 2 | Weight is small (one row, often cached in L2) |
| Pass 2: write out | B × H × 2 | One full write |
| **Total** | **~3 × B × H × 2** | Approximating w as small vs B×H |

**Roofline analysis (RTX 4060 Ti):**
- FLOPs per element: ~4 (square + sum + rsqrt amortized + multiply)
- Bytes per element: ~6 (3 × 2 bytes for read x twice + write)
- Arithmetic intensity: 4/6 ≈ 0.67 FLOP/byte
- Ridge point: 44 TFLOPS / 288 GB/s ≈ 153 FLOP/byte
- **Conclusion: deeply memory-bound** → optimize for bandwidth

**Target: >70% of 288 GB/s = >201 GB/s**

---

## The Fused Variant (Phase 2.7)

Transformers always compute `RMSNorm(hidden + residual)`. The fused kernel:

1. Reads `input` and `residual` from GDDR6X
2. Computes `x = input + residual` in registers
3. Stores `x` in shared memory (x-cache) AND writes `residual = x` to GDDR6X
4. Computes sum of squares (same as plain RMSNorm pass 1)
5. Pass 2: reads `x` from smem (not GDDR6X!) and `weight` from GDDR6X

**HBM access count comparison:**

| Unfused | Bytes | Fused | Bytes |
|---|---|---|---|
| Read input | BH×2 | Read input | BH×2 |
| Read residual | BH×2 | Read residual | BH×2 |
| Write x=inp+res | BH×2 | Write residual (in-place) | BH×2 |
| Read x (RMSNorm pass 1) | BH×2 | (x from smem, not HBM) | 0 |
| Read x (RMSNorm pass 2) | BH×2 | (x from smem, not HBM) | 0 |
| Read weight | H×2 | Read weight | H×2 |
| Write output | BH×2 | Write output | BH×2 |
| **Total ≈** | **5 × BH×2** | **Total ≈** | **3 × BH×2** |

**Savings:** 2 × B × H × 2 bytes per layer = 40% less HBM traffic for this operation.

For Llama 3 70B (B=32, H=8192, 80 layers × 2 norms/layer):
- Unfused: 80 × 2 × 5 × 32 × 8192 × 2 = 419 GB per forward pass
- Fused:   80 × 2 × 3 × 32 × 8192 × 2 = 251 GB per forward pass
- Savings: 168 GB of HBM traffic eliminated

---

## Where RMSNorm Appears in the Model

```python
# Simplified Llama forward pass (one layer):

# ─── Attention sub-layer ──────────────────────────────────────────
hidden = fused_add_rmsnorm(hidden, residual, attn_norm_weight, eps)
# → residual is updated to (old_hidden + old_residual) in-place
# → hidden is now RMSNorm(input + residual)

attn_out = self_attention(hidden, kv_cache)   # Q,K,V projections + attention

# ─── MLP sub-layer ────────────────────────────────────────────────
hidden = fused_add_rmsnorm(attn_out, residual, mlp_norm_weight, eps)
# → same pattern: add residual, normalize

mlp_out = ffn(hidden)   # gate_proj + up_proj → silu_and_mul → down_proj

residual = mlp_out
# → repeat for next layer
```

For a 32-layer model: 64 `fused_add_rmsnorm` calls per forward pass.

---

## Production RMSNorm Parameters (Common Models)

| Model | Hidden Dim | n_warps | Smem (reduce) | Smem (x-cache, fused) |
|---|---|---|---|---|
| Llama 2 7B | 4096 | 16 | 64 B | 8 KB |
| Llama 3 8B | 4096 | 16 | 64 B | 8 KB |
| Llama 3 70B | 8192 | 16* | 64 B | 16 KB |
| Mistral 7B | 4096 | 16 | 64 B | 8 KB |
| Qwen 7B | 4096 | 16 | 64 B | 8 KB |
| DeepSeek V3 | 7168 | 16* | 64 B | 14 KB |
| GPT-NeoX 20B | 6144 | 16* | 64 B | 12 KB |

*: n_warps capped at 16 (512 threads) even when n_vec > 512. Each thread loops over multiple chunks.

All x-cache sizes fit well within the 99 KB limit on sm_89.
