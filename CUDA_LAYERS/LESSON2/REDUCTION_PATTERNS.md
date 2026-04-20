# REDUCTION PATTERNS
## Butterfly reduce, tree reduce, scan, and where each appears in LLM kernels

---

## The One-Sentence Version

A reduction takes N values held by N threads and produces one result visible to all — the pattern used in RMSNorm (sum-of-squares), softmax (max and exp-sum), and attention (online softmax). The butterfly is the fastest warp-level implementation.

---

## Part 1 — The Three Reduction Patterns

### Pattern A: Serial Tree (Bad for GPU)

```
8 threads, values: [a, b, c, d, e, f, g, h]

Step 1: thread 0 = a+b,  thread 2 = c+d,  thread 4 = e+f,  thread 6 = g+h
Step 2: thread 0 = (a+b)+(c+d),            thread 4 = (e+f)+(g+h)
Step 3: thread 0 = (a+b+c+d)+(e+f+g+h)
→ thread 0 has result, threads 1,2,3,4,5,6,7 are idle
```

Problem: half the threads are idle each step. After 3 steps (log2(8)), 7/8 threads are doing nothing. Also requires shared memory for inter-thread communication.

### Pattern B: Butterfly (Good — what `__shfl_xor_sync` implements)

```
8 threads: [a, b, c, d, e, f, g, h]
           L0  L1  L2  L3  L4  L5  L6  L7

XOR 4:  L0↔L4, L1↔L5, L2↔L6, L3↔L7   (all 8 threads active simultaneously)
Result: [a+e, b+f, c+g, d+h,  a+e, b+f, c+g, d+h]

XOR 2:  L0↔L2, L1↔L3, L4↔L6, L5↔L7   (all 8 threads active simultaneously)
Result: [a+c+e+g, b+d+f+h, a+c+e+g, b+d+f+h, ...]  (pattern repeats)

XOR 1:  L0↔L1, L2↔L3, L4↔L5, L6↔L7   (all 8 threads active simultaneously)
Result: [a+b+c+d+e+f+g+h, ...]   ALL LANES HOLD THE TOTAL
```

Every thread participates in every step. No idle threads. No shared memory. Result is broadcast to all lanes automatically.

### Pattern C: Fan-in (Partial Broadcast — for very large reductions)

Used when the reduction spans multiple blocks (e.g., reduce across the entire batch dimension). Block 0 computes partial sum → writes to global memory → a second kernel reads all partial sums and reduces again. Used in PyTorch's `torch.sum` on large tensors.

For intra-block reductions in LLM kernels (hidden_dim ≤ 8192), the two-level warp+smem pattern (Lesson 3) is always used instead.

---

## Part 2 — Why 5 Steps for 32 Threads?

log2(32) = 5. The butterfly needs one step per power of two.

```
Warp size: 32 = 2^5
Steps needed: 5

XOR offsets: 16 (2^4), 8 (2^3), 4 (2^2), 2 (2^1), 1 (2^0)

After step k (XOR with 2^(4-k)):
  Each lane holds partial sum covering 2^(k+1) original values
  Lane i holds the sum of lanes (i & ~(2^(k+1) - 1)) ... (i | (2^(k+1) - 1))
```

Concrete: after XOR=16 (step 1), lane 3 holds the sum of lanes 3 and 19 (3 XOR 16 = 19). After XOR=8 (step 2), lane 3 holds the sum of lanes 3, 11, 19, 27. After all 5 steps: lane 3 holds the sum of all 32 lanes.

---

## Part 3 — Where Reductions Appear in LLM Inference

### RMSNorm (reduce sum-of-squares)

```
Input:  x[batch, hidden_dim]          fp16/bf16

For each row (one block processes one row):
  Step 1: Each thread loads 8 fp16 (128-bit), accumulates partial sum_sq
          sum_sq += x[i]^2 for i in this thread's chunk

  Step 2: warp_reduce_sum(sum_sq)    ← THIS IS LESSON 2
          After: every thread holds total sum_sq for the row

  Step 3: rms_rcp = rsqrt(sum_sq / hidden_dim + eps)

  Step 4: Each thread loads 8 fp16 input + 8 fp16 weight
          output = input * rms_rcp * weight

Reduction type: SUM across hidden_dim
Result needed by: all threads (each must scale its own elements)
```

### Softmax (online max + exp-sum)

```
Input:  scores[seq_len]   one row = one query's attention logits

For each query:
  Pass 1: Each thread holds partial max of its slice
          warp_reduce_max(partial_max)    ← Lesson 2
          After: every thread knows global max of row

  Pass 2: Each thread computes exp(score - max) for its slice
          Accumulates partial exp_sum
          warp_reduce_sum(partial_exp_sum)  ← Lesson 2 again
          After: every thread knows total exp_sum

  Pass 3: Each thread divides its exp values by exp_sum

Two reductions per row: max reduction then sum reduction.
Online softmax (Flash Attention) fuses passes 1+2 into one scan.
```

### Flash Attention Online Softmax (fused)

```
Online softmax processes tokens one tile at a time:

  For each key-value tile:
    1. Compute QK scores for this tile (partial row)
    2. m_new = max(m_old, tile_max)        ← warp_reduce_max per tile
    3. l_new = exp(m_old - m_new) * l_old + sum(exp(score - m_new))
       ^-- this fuses the rescaling of old partials with new exp-sum
    4. O_new = O_old * exp(m_old - m_new) + V_tile × softmax_weights

This requires warp_reduce_max inside the tile loop.
After all tiles: normalize O by l_new.
```

### AllReduce (distributed multi-GPU)

```
Each GPU holds a partial gradient sum.
Goal: every GPU gets the sum across all GPUs.

Two-ring AllReduce:
  Phase 1: ReduceScatter — each GPU sends a chunk, receives a chunk,
           does a local reduce (warp reduce within each GPU's SM)
  Phase 2: AllGather — each GPU broadcasts its reduced chunk

The warp-level reduce you learn in Lesson 2 appears inside each GPU's
local reduction step (inside quick_all_reduce.cu in SGLang).
```

---

## Part 4 — Reduce-Then-Broadcast vs All-Reduce

**Reduce-Then-Broadcast** (what `__shfl_xor_sync` does):

After the 5-step butterfly, every lane holds the complete reduced value. The butterfly is inherently all-reduce — there is no separate broadcast step needed.

This is different from a "reduce to one lane then broadcast":
```cuda
// WORSE: reduce to lane 0, then broadcast
float total = val;
for (int offset = 16; offset > 0; offset >>= 1) {
    total += __shfl_down_sync(0xffffffff, total, offset);
}
// Now only lane 0 has the correct answer
total = __shfl_sync(0xffffffff, total, 0);   // broadcast from lane 0
```

The XOR butterfly avoids this extra step — every lane gets the result in 5 rounds instead of 5+1.

---

## Part 5 — Numerical Stability of Reductions

Floating-point addition is not associative: `(a + b) + c ≠ a + (b + c)` in general due to rounding.

The butterfly order is:
```
round 1: {L0+L16, L1+L17, ..., L15+L31}
round 2: {(L0+L16)+(L8+L24), ...}
```

This is a different summation order than sequential (L0+L1+L2+...+L31), and will give slightly different results. For RMSNorm with fp16 inputs upcast to float32, this is fine — the error is within FP32 epsilon (~1e-7) and the final atol is 1e-2.

For precise numerical guarantees (e.g., gradient accumulation), use Kahan summation or reduce in fp64. For inference kernels, the butterfly order is always acceptable.

---

## Part 6 — Two-Level Block Reduce: The Full Pattern

For blocks with more than one warp (typical: 128–256 threads = 4–8 warps):

```
Block = 256 threads = 8 warps

Phase 1: Intra-warp shuffle (8 warps run simultaneously)
  Warp 0: warp_reduce → partial_0  (5 shuffles, no smem)
  Warp 1: warp_reduce → partial_1
  ...
  Warp 7: warp_reduce → partial_7

Phase 2: Write partial results to shared memory
  Lane 0 of each warp writes to smem[warp_id]
  __syncthreads()   ← only sync needed in the entire kernel

Phase 3: First warp reads smem and reduces again
  Warp 0, lane 0: smem[0] = partial_0
  Warp 0, lane 1: smem[1] = partial_1
  ...
  Warp 0 runs warp_reduce_sum on 8 values → final total in all of warp 0

Total cost:
  Phase 1: 5 shuffles × 8 warps (parallel) = 5 shuffle instructions
  Phase 2: 8 smem writes + 1 barrier = ~30–80 cycles
  Phase 3: 5 shuffles in 1 warp = 5 shuffle instructions
  Total: 15 shuffle instructions + 1 barrier
  vs naive smem: 256 smem writes + 256 smem reads + 1 barrier
```

---

## Part 7 — Reduction vs Scan

A **reduction** produces one output from N inputs. A **scan** produces N outputs where output[i] = reduce(input[0..i]).

| | Reduction | Inclusive Scan | Exclusive Scan |
|---|---|---|---|
| Output size | 1 | N | N |
| Output[3] | sum(all) | sum(0..3) | sum(0..2) |
| Warp intrinsic | `__shfl_xor_sync` | `__shfl_up_sync` | `__shfl_up_sync` |
| Steps | log2(N) | log2(N) | log2(N)+1 |
| In LLM kernels? | Everywhere | Rarely (top-k) | Rarely |

The distinction matters because they use different shuffle variants and have different semantics. For this curriculum (RMSNorm, softmax, attention), you only need reductions.

---

## Quick Reference

```
Butterfly reduce (warp, all lanes hold result):
  for mask in [16, 8, 4, 2, 1]:
    val += __shfl_xor_sync(0xffffffff, val, mask)   # sum
    val  = fmaxf(val, __shfl_xor_sync(..., val, mask))  # max

Sub-warp reduce (kN threads, k < 32):
  for mask in [kN/2, kN/4, ..., 1]:
    val op= __shfl_xor_sync(0xffffffff, val, mask)

Two-level block reduce (B threads, B/32 warps):
  warp_partial = warp_reduce(val)     # phase 1: 5 shuffles per warp
  if lane == 0: smem[warp_id] = warp_partial
  __syncthreads()
  if warp_id == 0:
    v = smem[lane] if lane < n_warps else 0
    block_total = warp_reduce(v)      # phase 3: 5 more shuffles in warp 0

Where it appears in SGLang:
  warp::reduce_sum  → norm.cuh RMSNorm pass-1 (sum_of_squares)
  warp::reduce_max  → cta.cuh, softmax, attention online max
  cta::reduce_max   → two-level pattern above, used in attention decode
```
