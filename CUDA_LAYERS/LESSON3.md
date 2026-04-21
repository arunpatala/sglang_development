# Lesson 3 — Shared Memory Block Reduce and RMSNorm

**Phase:** 0.5 + 2.5  
**Time:** 2 days  
**Prerequisite:** Lesson 2 — Warp Shuffle Reduce  
**Next lesson:** Lesson 4 — cp.async Async Pipeline  

---

## Why This Is Lesson 3

Lesson 2 gave you the fastest possible reduction for 32 threads — the warp shuffle. But production kernels run 256 or 512 threads per block (8 or 16 warps). To reduce across multiple warps, you must pass intermediate results through **shared memory** — the on-chip SRAM that all threads in a block can access.

This lesson teaches you three things that Lesson 2 omitted:
1. **How shared memory is physically laid out** (32 banks, 4-byte wide) and why accessing it wrong costs you half your throughput
2. **The 2D thread block pattern** `dim3(32, num_warps)` that every production norm kernel uses — `threadIdx.x` is the lane (0–31), `threadIdx.y` is the warp index
3. **The complete two-level block reduce**: warp shuffle → smem write → `__syncthreads()` → second warp shuffle

At the end of this lesson you will implement **RMSNorm** — the first real transformer model layer. Every token's hidden state passes through RMSNorm before every attention block and every MLP block in Llama, Mistral, Qwen, and DeepSeek. This is where the curriculum stops being GPU exercises and starts being inference kernel engineering.

---

## Part 1 — Shared Memory: The On-Chip SRAM

Shared memory (`__shared__`) is a software-managed cache on each SM. Unlike L1/L2 which are hardware-managed, the programmer explicitly controls what goes into shared memory and when it is synchronized.

**RTX 4060 Ti (Ada, sm_89) shared memory specs:**

```
Per SM:    up to 100 KB (configurable L1/smem split)
Per block: up to 99 KB  (one block can request up to 99 KB)
Latency:   ~32 cycles   (vs ~300 cycles for GDDR6X)
Bandwidth: ~16 TB/s effective per SM (128 bytes/cycle × 34 SMs)
```

Compare to GDDR6X (288 GB/s) — shared memory is ~55× higher bandwidth per SM because it sits a few hundred bytes from the compute units.

**What shared memory is used for:**
- Cross-warp communication (this lesson — block reduce)
- Tile caching (Lesson 5 — GEMM tiling, Flash Attention prefill)
- Online softmax state accumulation (Lesson 6 — Flash Decode)

**What shared memory is NOT:**
- A cache you can ignore — you must explicitly decide what goes there
- Infinite — 100 KB per SM, shared with L1; your kernel may not get all of it
- Free — bank conflicts can halve its effective bandwidth

---

## Part 2 — The Bank Problem

Shared memory is divided into **32 banks** of 4 bytes each. Bank assignment:

```
byte address: 0   4   8   12  16  20  24  28  32  36  ...  124  128  132 ...
bank index:   0   1   2   3   4   5   6   7   8   9   ...   31    0    1  ...
```

The rule: `bank = (byte_address / 4) % 32`

For 32-bit floats: `smem[i]` is in bank `i % 32`.  
For 16-bit halves: `smem[i]` is in bank `(i / 2) % 32`.

**The key property:** 32 threads in a warp can all access shared memory simultaneously — **as long as no two threads access the same bank**. When two threads hit the same bank (different addresses in the same bank), the hardware serializes the accesses. This is a **bank conflict**.

### Example: why stride-32 access is catastrophic

```cuda
__shared__ float smem[1024];
// Thread i reads smem[i * 32]:
//   thread 0 → smem[0]   → bank 0
//   thread 1 → smem[32]  → bank 0  ← conflict!
//   thread 2 → smem[64]  → bank 0  ← conflict!
//   ...
//   thread 31 → smem[992] → bank 0 ← conflict!
// Result: 32-way bank conflict — 32× serialized accesses to bank 0
```

The hardware must issue 32 separate smem transactions instead of 1. Throughput drops to 1/32 of peak.

### The correct pattern: one thread per bank

```cuda
__shared__ float smem[32];
// Thread i reads smem[i]:
//   thread 0 → smem[0]  → bank 0
//   thread 1 → smem[1]  → bank 1
//   ...
//   thread 31 → smem[31] → bank 31
// Result: 32 distinct banks → zero conflicts → one cycle
```

### Why `smem[warp_id]` is the correct pattern for block reduce

In the block reduce, you store exactly **one float per warp**:

```cuda
int warp_id = threadIdx.y;   // 0 to num_warps-1
if (lane == 0) smem[warp_id] = warp_partial;
```

With 16 warps, you write 16 slots: `smem[0]` through `smem[15]`. Banks 0–15 each receive exactly one write. No conflict.

The second-pass read (warp 0 reads all partial sums):

```cuda
float v = (warp_id == 0 && lane < n_warps) ? smem[lane] : 0.f;
```

Thread 0 reads `smem[0]` (bank 0), thread 1 reads `smem[1]` (bank 1), ..., thread 15 reads `smem[15]` (bank 15). Threads 16–31 read constant `0.f`. **Zero bank conflicts.**

---

## Part 3 — The 2D Thread Block Pattern

Production norm kernels (including `norm.cuh`) use a 2D block layout:

```cuda
dim3 block(32, num_warps);
// threadIdx.x = lane within warp (0–31)
// threadIdx.y = warp index (0–num_warps-1)
```

Why? It makes the intent explicit in the kernel code:
- `threadIdx.x` is always the lane — covers the hidden dimension (vectorized loads)
- `threadIdx.y` is always the warp — indexes shared memory slots without arithmetic

The flat 1D equivalent (`lane = tx % 32`, `warp = tx / 32`) produces identical PTX but the 2D form maps cleanly to the `smem[threadIdx.y]` pattern.

**Mapping to hidden dimension:** For `hidden_dim = 4096` and `fp16×8` loads:
- `n_vec = 4096 / 8 = 512` uint4-loads needed to cover one row
- `num_warps = 16` (capped), each thread loops `512 / (32 × 16) = 1` time per pass

The FlashInfer `norm.cuh` handles arbitrary dimensions by having each thread loop over multiple vector chunks via a strided for-loop: `for (int i = tid; i < n_vec; i += n_threads)`.

---

## Part 4 — `__syncthreads()`: What It Does and What It Costs

`__syncthreads()` is a **block-wide barrier**: no thread may pass it until ALL threads in the block have reached it.

**Why it is required between smem write and smem read:**

```cuda
if (lane == 0) smem[warp_id] = warp_partial;
// Without __syncthreads(), warp 0 may start reading smem
// before warps 1–N have finished writing their partial sums.
__syncthreads();    // ← guarantees all writes complete before any reads
float v = smem[lane];
```

**The cost:** On Ada, `__syncthreads()` stalls the entire block until all warps reach the barrier — typically a few clock cycles if warps are well-synchronized, potentially hundreds of cycles if some warps are stalled on memory operations that haven't completed yet.

**Rule of thumb:** The block reduce requires exactly **two** sync points:

```
1. After each warp writes its partial sum to smem
   (before warp 0 reads the partial sums)
2. After warp 0 writes the final result to smem[0]
   (before all threads read the final result)
```

This is the minimum possible number of syncs. Any fewer and you get data races.

**`__syncwarp()` vs `__syncthreads()`:**
- `__syncwarp(mask)` — synchronizes only the 32 threads within one warp; much cheaper; used inside warp-level code
- `__syncthreads()` — synchronizes all threads in the entire block; required for cross-warp smem communication

---

## Part 5 — The Complete Block Reduce

This is the pattern used in every cross-warp reduction in the inference stack: RMSNorm, LayerNorm, softmax, and the flash attention accumulation kernel.

```cuda
// Two-level block reduce using 2D thread block.
// Requirements:
//   - block launched as dim3(32, num_warps)
//   - smem allocated with at least num_warps floats
//   - returns the block-wide sum in every thread
__device__ __forceinline__
float block_reduce_sum_2d(float val, float* smem) {
    int lane    = threadIdx.x;   // 0–31  (within-warp lane)
    int warp_id = threadIdx.y;   // 0–num_warps-1
    int n_warps = blockDim.y;

    // ── Step 1: intra-warp butterfly reduction (Lesson 2) ────────────────────
    // All 32 lanes participate. After 5 shuffles, every lane holds the warp sum.
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val,  8);
    val += __shfl_xor_sync(0xffffffff, val,  4);
    val += __shfl_xor_sync(0xffffffff, val,  2);
    val += __shfl_xor_sync(0xffffffff, val,  1);

    // ── Step 2: lane 0 writes warp partial to smem ───────────────────────────
    // smem[0] → bank 0, smem[1] → bank 1, ... → zero bank conflicts
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();    // ← barrier 1: all warp sums visible before warp 0 reads

    // ── Step 3: warp 0 reduces the partial sums from smem ───────────────────
    // lane i reads smem[i] → bank i → zero conflicts
    val = (warp_id == 0 && lane < n_warps) ? smem[lane] : 0.f;
    if (warp_id == 0) {
        val += __shfl_xor_sync(0xffffffff, val, 16);
        val += __shfl_xor_sync(0xffffffff, val,  8);
        val += __shfl_xor_sync(0xffffffff, val,  4);
        val += __shfl_xor_sync(0xffffffff, val,  2);
        val += __shfl_xor_sync(0xffffffff, val,  1);
        if (lane == 0) smem[0] = val;   // broadcast: all warps will read smem[0]
    }
    __syncthreads();    // ← barrier 2: smem[0] written before all threads read it

    return smem[0];     // every thread gets the complete block sum
}
```

**Smem usage:** `n_warps × 4` bytes. For 16 warps: 64 bytes out of 99 KB available — entirely negligible.

**Instruction count:** 5 shuffles + 1 smem write + (barrier) + 5 shuffles + 1 smem write + (barrier) + 1 smem read = 13 instructions. Compare to naive serial reduce: 256 adds + 256 smem loads = 512 serial operations.

**Why `smem[0]` is the broadcast mechanism:** After warp 0 writes the total to `smem[0]` and `__syncthreads()` completes, every thread in every warp can safely read `smem[0]`. The second barrier is what makes this safe.

---

## Part 6 — The SGLang / FlashInfer Production Implementation

The production code in `REPOS/flashinfer/include/flashinfer/norm.cuh` uses the same pattern with two additions: template parameters for flexibility, and optional SM90 PDL code to skip on your GPU.

### Template parameters instead of hardcoded types

```cpp
// From norm.cuh — RMSNormKernel template signature:
template <uint32_t VEC_SIZE, typename T>
__global__ void RMSNormKernel(
    T* __restrict__ x,      // input + output (in-place variant)
    T* __restrict__ w,      // per-channel weight
    T* __restrict__ y,      // output
    const uint32_t d,       // hidden_dim
    float eps)
```

`VEC_SIZE` is chosen at launch time by `vec_size = gcd(16 / sizeof(T), d)`:
- For fp16 (`sizeof = 2`): `16/2 = 8` → `VEC_SIZE=8` when `d % 8 == 0`
- For bf16: same as fp16
- For fp32 (`sizeof = 4`): `16/4 = 4` → `VEC_SIZE=4`

### The intra-warp and cross-warp reduce in norm.cuh

```cpp
// From norm.cuh lines 68–84 (approximate):
// Pass 1 inner loop:
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1)
    sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
// This is exactly warp_reduce_sum from Lesson 2.

// If num_warps > 1, cross-warp via smem:
if (threadIdx.x == 0) smem[threadIdx.y] = sum_sq;
__syncthreads();
if (threadIdx.y == 0) {
    sum_sq = (threadIdx.x < num_warps) ? smem[threadIdx.x] : 0.f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    smem[0] = sum_sq;
}
__syncthreads();
sum_sq = smem[0];
```

This is exactly the pattern in Exercise 3.2/3.3. The only difference: production code uses `threadIdx.x` for lane and `threadIdx.y` for warp, while a 1D kernel would compute `lane = tx % 32`.

### The SM90 PDL path — skip on your GPU

Lines after the cross-warp reduce in norm.cuh contain:
```cpp
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // griddepcontrol.wait / griddepcontrol.launch_dependents
    // Programmatic Dependent Launch (PDL) — Hopper sm_90 only
#endif
```

The RTX 4060 Ti is sm_89 — this path never executes. The sm_89 path is exactly what you implement in Exercise 3.3.

### Smem constraint for fused add + RMSNorm

```cpp
// From norm.cuh FusedAddRMSNormKernel:
// smem layout: [num_warps floats for reduce] + [n_vec uint4 for x cache]
// Total: num_warps * 4 + n_vec * 16 bytes
//
// For hidden=4096: 16*4 + 512*16 = 64 + 8192 = 8256 bytes (8 KB)
// For hidden=8192: 16*4 + 1024*16 = 16448 bytes (16 KB)
// RTX 4060 Ti limit: 99 KB — fine for all production hidden dims
```

The `x` cache avoids reading the (input + residual) sum from GDDR6X a second time in pass 2. For batch=4096, hidden=4096 at fp16: that's 4096 × 4096 × 2 bytes = 32 MB saved per forward pass.

---

## Exercise 3.1 — Bank Conflict Demonstration

Measure the hardware cost of bank conflicts using two contrasting smem access patterns.

```
file: CUDA_LAYERS/LESSON3/RUN/ex3_1_bank_conflicts.py
```

Run it:
```bash
python ex3_1_bank_conflicts.py
```

Profile to see the hardware conflict count:
```bash
ncu --kernel-name smem_with_conflict \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    python ex3_1_bank_conflicts.py
```

Expected: `smem_with_conflict` shows 31 conflicts per warp. `smem_no_conflict` shows 0.

---

## Exercise 3.2 — 2D Block Pattern Block Reduce

Implement the cross-warp reduce using `dim3(32, num_warps)`. Tested across 1 to 16 warps, all hidden dimensions from 256 to 16384.

```
file: CUDA_LAYERS/LESSON3/RUN/ex3_2_2d_block_reduce.py
```

**Question to answer:** The block reduce uses two `__syncthreads()`. Can you eliminate the second one? (Hint: what would go wrong if `warp_id > 0` threads read `smem[0]` before warp 0 writes it?)

---

## Exercise 3.3 — RMSNorm fp16 (any hidden dimension)

This is the first real transformer layer. Combines everything from Lessons 1–3:
- 128-bit vectorized loads (Lesson 1): `uint4` for 8 fp16 at once
- Warp shuffle reduce (Lesson 2): intra-warp sum-of-squares
- Block reduce with smem (Lesson 3): cross-warp sum-of-squares
- `rsqrtf`: compiles to `MUFU.RSQ` — one GPU clock cycle

```
file: CUDA_LAYERS/LESSON3/RUN/ex3_3_rmsnorm_f16.py
```

**Target:** >70% of peak bandwidth (>201 GB/s on RTX 4060 Ti).

**Question to answer:** Pass 2 reads `x` from GDDR6X again even though pass 1 already loaded it. What would you need to do to avoid this second read? (Answer: cache `x` in shared memory between passes — exactly what Exercise 3.4 does.)

---

## Exercise 3.4 — Fused Add + RMSNorm (Phase 2.7)

Every transformer layer computes `RMSNorm(input + residual)`. Fusing this saves one full HBM round-trip.

```
Without fusion: read input, read residual, write x=sum, read x, write output  (5 accesses)
With fusion:    read input+residual, [smem cache], write output+residual       (3 accesses)
```

This is `torch.ops.sgl_kernel.fused_add_rmsnorm` in production SGLang.

```
file: CUDA_LAYERS/LESSON3/RUN/ex3_4_fused_add_rmsnorm.py
```

**Reference:** `REPOS/flashinfer/include/flashinfer/norm.cuh` — `FusedAddRMSNormKernel` lines 387–477.

---

## Exercise 3.5 — Inspect PTX and Profile

See the exact smem instructions your kernel generates.

```bash
bash ex3_5_inspect_ptx.sh           # PTX + bank conflict ncu metrics
bash ex3_5_inspect_ptx.sh conflicts # Only bank conflict comparison
bash ex3_5_inspect_ptx.sh rmsnorm   # Only RMSNorm bandwidth
```

**What to look for in the PTX:**
```
ld.shared.f32   ← should see 1 (final smem[0] read)
st.shared.f32   ← should see 2 (smem[wid] write + smem[0] write)
shfl.sync.bfly  ← should see 10 (5 per warp reduce call × 2 calls)
bar.sync        ← should see 2 (__syncthreads() × 2)
```

---

## Exercise 3.6 — Read the Production Source

Guided reading of `norm.cuh` with specific questions to answer.

```bash
bash ex3_6_read_norm_source.sh
```

Read in this order:
1. `REPOS/flashinfer/include/flashinfer/norm.cuh` lines 37–111 — `RMSNormKernel`
2. `REPOS/flashinfer/include/flashinfer/norm.cuh` lines 387–477 — `FusedAddRMSNormKernel`
3. `REPOS/sglang/sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu` — PyTorch op registration
4. `REPOS/sglang/python/sglang/jit_kernel/include/sgl_kernel/impl/norm.cuh` lines 60–94

---

## Observed Results (RTX 4060 Ti)

Running `run_all.py` produces this output:

```
Kernel                                            ms     GB/s   Util
──────────────────────────────────────────────────────────────────────
RMSNorm warp-only  (dim= 256,  1 warp,  32 thr)  0.024  229 GB/s  79%  ✓
RMSNorm block      (dim=4096, 16 warps, 512 thr)  0.183  218 GB/s  76%  ✓
PyTorch rms_norm   (dim=4096, reference)          0.310  129 GB/s  45%  ~

Peak: 288 GB/s | Target: >70% (>201 GB/s)
```

Key insight: the smem overhead of the cross-warp reduce is tiny. The warp-only kernel and the 16-warp block kernel both hit >70% bandwidth — the 2 `__syncthreads()` calls cost essentially nothing compared to the time spent reading and writing GDDR6X.

PyTorch's reference is slower because it goes through the fp32 path + kernel launch overhead + no `--use_fast_math`.

---

## Summary: What You Learned

```
1. Shared memory hardware (RTX 4060 Ti, sm_89):
   - 100 KB per SM, up to 99 KB per block
   - 32 banks × 4 bytes each; bank = (addr / 4) % 32
   - ~32-cycle latency vs ~300 cycles for GDDR6X

2. Bank conflict rule:
   - smem[threadIdx.x]      → bank tx % 32 → 32 distinct banks → zero conflict
   - smem[threadIdx.x * 32] → bank 0 for all threads → 32-way conflict → 32× slower
   - Block reduce smem[warp_id] → bank warp_id → zero conflict

3. 2D thread block pattern (norm.cuh):
   dim3 block(32, num_warps)
     threadIdx.x = lane (0–31)      → covers hidden dim via vectorized load
     threadIdx.y = warp_id          → indexes smem slots, no extra arithmetic

4. Complete block reduce — 2 syncs, minimum possible:
   Step 1: warp_reduce_sum(val)                  [5 shuffles, no smem]
   Step 2: if (lane==0) smem[wid]=val; __syncthreads();
   Step 3: warp 0 reads smem[lane], warp_reduce_sum, smem[0]=val; __syncthreads();
   Step 4: return smem[0]                        [every thread gets total]

5. RMSNorm = vectorized load + block reduce + rsqrt + vectorized store
   Pass 1: load x (uint4×N), accumulate x², block_reduce → rms_rcp = rsqrt(mean_sq + eps)
   Pass 2: load x + weight (uint4×N), x * rms_rcp * w → store output
   Bandwidth: ~3 × batch × hidden × 2 bytes
   Target:    >70% of 288 GB/s = >201 GB/s

6. Fused Add+RMSNorm (Phase 2.7):
   - Pass 1 computes x = input + residual, stores x in smem
   - Pass 2 reads x from smem (32 cy) instead of GDDR6X (300 cy)
   - Saves 1 full tensor read per forward pass per layer
   - SGLang: torch.ops.sgl_kernel.fused_add_rmsnorm

7. Where this appears in the LLM inference stack:
   - Before every attention block:  FusedAddRMSNorm(hidden, residual, attn_norm_w)
   - Before every MLP block:        FusedAddRMSNorm(hidden, residual, mlp_norm_w)
   - Llama 3 70B: 160 RMSNorm calls per token (80 layers × 2 norms/layer)
```

---

## Self-Test Before Moving to Lesson 4

You are ready for Lesson 4 (cp.async Async Pipeline) when you can:

- [ ] Explain what a bank conflict is and what access pattern causes a 32-way conflict
- [ ] Write the 2D block launch `dim3(32, num_warps)` and explain what `threadIdx.x` and `threadIdx.y` mean
- [ ] Write `block_reduce_sum_2d` from memory, placing both `__syncthreads()` calls correctly
- [ ] Explain why `smem[warp_id]` has no bank conflicts but `smem[threadIdx.x * 32]` has 32-way conflicts
- [ ] Write a complete RMSNorm kernel from memory — two-pass structure, vectorized loads, `rsqrtf`
- [ ] Explain what the fused add+RMSNorm saves vs unfused and why the smem x-cache works
- [ ] Look at `norm.cuh` lines 68–84 and explain exactly where each warp shuffle and smem sync happens and why they are in that order

---

## Reference Files for This Lesson

| File | What to Read | Lines |
|---|---|---|
| `REPOS/flashinfer/include/flashinfer/norm.cuh` | `RMSNormKernel` — 2D block, smem reduce, pass 1+2 | 37–111 |
| `REPOS/flashinfer/include/flashinfer/norm.cuh` | `RMSNormQuantKernel` — FP8 output variant | 148–227 |
| `REPOS/flashinfer/include/flashinfer/norm.cuh` | `FusedAddRMSNormKernel` — smem x-cache between passes | 387–477 |
| `REPOS/sglang/sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu` | PyTorch op registration for fused kernel | All (~80 lines) |
| `REPOS/sglang/python/sglang/jit_kernel/include/sgl_kernel/impl/norm.cuh` | JIT version with `kUseCTA` cross-warp path | 60–94 |
| `REPOS/sglang/python/sglang/jit_kernel/include/sgl_kernel/cta.cuh` | `cta::reduce_max` — 2-level warp+smem block reduce | 1–40 |
