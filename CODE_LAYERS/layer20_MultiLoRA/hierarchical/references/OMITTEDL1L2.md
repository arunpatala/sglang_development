# Multi-LoRA Inference — Material Omitted from COMBINEDL1L2.md

**What this file is:** The full text of every section omitted from `COMBINEDL1L2.md`. The "What Is Left Out and Why" appendix of `COMBINEDL1L2.md` names each omission and explains why it was excluded. This file preserves the complete source material so nothing is lost.

**Parent file:** `COMBINEDL1L2.md` (L1 + L2 synthesis)
**Sources:** L2/03 (variant math), L3/02 (Punica CUDA), L3/03 (S-LoRA paging), L4/01 (dLoRA), L4/02 (CaraServe), L4/03 (Loquetier), L4/04 (ServerlessLoRA), L4/05 (Predictive-LoRA), L4/06 (InfiniLoRA)

---

## Omission 1: Punica — SGMV Kernel CUDA Implementation

**Source:** `L3/02_punica_mlsys24.md`
**Venue:** MLSys 2024
**Why omitted from COMBINEDL1L2.md:** COMBINEDL1L2.md explains what SGMV does and why it achieves 12× throughput. The actual CUDA implementation — thread block assignment to segments, gather scatter indices, handling variable-rank adapters in the same launch, the PTX-level memory access pattern — requires CUDA programming knowledge to engage with. The concept belongs at L1/L2; the implementation belongs at L3.

---

### What the SGMV Kernel Actually Does

The fundamental challenge SGMV solves is computing a batched matrix-vector multiply where each row of the batch matrix uses a different weight matrix:

```
For a batch of N tokens, where token i uses adapter (Aᵢ, Bᵢ):

Standard GEMM:          X @ W.T           — all rows use same W
SGMV (down-project):    [x₀ @ A₀.T,       — each row uses different A
                          x₁ @ A₁.T,
                          ...
                          xₙ @ Aₙ.T]
SGMV (up-project):      [h₀ @ B₀.T,       — each row uses different B
                          h₁ @ B₁.T,
                          ...
                          hₙ @ Bₙ.T]
```

### The Naive Approach and Its Cost

Without SGMV, serving a batch with K distinct adapters requires K separate kernel launches:

```python
# Naive: O(K) kernel launches per layer
for adapter_id, token_indices in token_groups_by_adapter.items():
    A = adapter_weights[adapter_id]["A"]
    B = adapter_weights[adapter_id]["B"]
    x_group = x[token_indices]          # gather
    h = x_group @ A.T @ B.T * scale    # two GEMMs
    out[token_indices] += h             # scatter
```

With K=50 adapters per batch, this is 100 kernel launches per layer (A and B separately), per forward pass. Each launch has fixed overhead (~5–20μs on modern GPUs for kernel scheduling). For 32 layers × 2 projections × 50 adapters = **3,200 kernel launches per token step**. Even at 10μs each, that is 32ms of pure scheduling overhead — comparable to the entire decode step.

### The SGMV Algorithm

SGMV fuses all adapter computations into a single kernel launch:

**Step 1: Build segment metadata**

Before the kernel launches, build arrays describing each adapter's tokens:
```python
# seg_starts[i] = index into the batch where adapter i's tokens begin
# seg_lengths[i] = how many tokens use adapter i
# adapter_ptr[i] = GPU memory pointer to adapter i's weight matrix
```

**Step 2: Single kernel launch**

```
sgmv_kernel(
    X,                    # input activations  [N, in_dim]
    out,                  # output buffer      [N, out_dim]
    adapter_weights_A,    # all A matrices concatenated  [K × r, in_dim]
    adapter_weights_B,    # all B matrices concatenated  [K × out_dim, r]
    seg_starts,           # segment start indices        [K]
    seg_lengths,          # segment lengths              [K]
    scaling               # lora_alpha/r
)
```

**Step 3: Thread block assignment**

Each CUDA thread block is assigned to one segment (one adapter's tokens):
- Block processes segment `i` covering tokens `seg_starts[i]:seg_starts[i]+seg_lengths[i]`
- Block loads adapter A matrix from `adapter_weights_A[i*r:(i+1)*r, :]`
- Block computes `x_segment @ A.T` → low-rank intermediate `h_seg ∈ ℝ^{seg_len × r}`
- Block computes `h_seg @ B.T * scale` → adapter delta `delta_seg ∈ ℝ^{seg_len × out_dim}`
- Block writes `delta_seg` to `out[seg_starts[i]:seg_starts[i]+seg_lengths[i], :]`

This gives one kernel launch per layer per projection matrix, regardless of K.

### Why It's Fast: Operational Intensity

For a segment of `s` tokens using an adapter with rank `r` and output dimension `d`:
- FLOPs: `2 × s × r × d` (down-project) + `2 × s × r × d` (up-project) = `4 × s × r × d`
- Bytes read: `s × in_dim × 2` (X) + `r × in_dim × 2` (A) + `out_dim × r × 2` (B)

For large segments (high request volume for one adapter), the operational intensity grows linearly with `s` — the segment effectively becomes a standard GEMM with high intensity. For small segments (one request per adapter), intensity drops. SGMV handles both cases in the same kernel; the overhead is proportional to the number of segments (adapters), not the number of tokens.

### BGMV: The Decode-Phase Variant

For single-token-per-request decode steps, each "batch" is a list of single-row matrices. The full SGMV thread block strategy over-provisions CUDA resources for these tiny GEMMs. BGMV (Batched Gather Matrix-Vector Multiplication) is a SGMV variant optimised for batch size 1 per adapter:

- Uses one warp (32 threads) per adapter segment instead of one full thread block
- Loads A matrix into shared memory once per adapter across all warps in the block
- Multiple adapters' single-token computations are co-scheduled in the same thread block

For decode-dominated workloads (most production serving), BGMV outperforms SGMV. vLLM uses BGMV as its default; SGLang's Chunked SGMV (`csgmv`) hybridises the two approaches by chunking prefill tokens to use BGMV even during prefill.

### The Layer 20 Masked Alternative vs SGMV

| | Layer 20 float mask | SGMV | BGMV |
|---|---|---|---|
| Kernel launches per layer | 1 (always, dense) | 1 (fused) | 1 (fused) |
| Tokens computed | All N (base + LoRA) | Only LoRA tokens | Only LoRA tokens |
| Extra FLOPs | O(N × r × d) always | O(N_lora × r × d) | O(N_lora × r × d) |
| Implementation | 5 lines Python | ~500 lines CUDA | ~300 lines CUDA |
| Multiple adapters | Not supported | Supported | Supported |

For 100 tokens with 10% needing LoRA: mask wastes 90% of extra FLOPs; SGMV wastes 0%.

---

## Omission 2: S-LoRA — Unified Paging Formal Analysis and Tensor Parallelism

**Source:** `L3/03_slora_mlsys24.md`
**Venue:** MLSys 2024
**Why omitted from COMBINEDL1L2.md:** COMBINEDL1L2.md explains how SGLang's `max_loras_per_batch` and `lora_eviction_policy` work — the user-facing controls for S-LoRA's memory management. The formal proof that Unified Paging eliminates fragmentation, the queuing model for adapter loading latency, the specific page allocation algorithm, and the tensor parallel sharding derivation are L3 material for practitioners extending or debugging the memory subsystem.

---

### The Fragmentation Problem: Why Naive Allocation Fails

Without paged allocation, adapter weights are stored in contiguous GPU memory regions:

```
Initial state (after loading 4 adapters of different ranks):
┌────────────────────────────────────────────────────────────┐
│ [Adapter A: 200MB] [Adapter B: 50MB] [Adapter C: 400MB] [Adapter D: 100MB] │
└────────────────────────────────────────────────────────────┘
  Total used: 750MB  |  Free: 250MB (at end)

After evicting B and C (LRU):
┌────────────────────────────────────────────────────────────┐
│ [Adapter A: 200MB] [        FREE: 450MB       ] [Adapter D: 100MB] │
└────────────────────────────────────────────────────────────┘
  Free: 450MB total — but it's non-contiguous!

New request for Adapter E (rank=64, size=600MB):
  Cannot allocate — largest contiguous free block = 450MB < 600MB
  Even though total free (550MB) > 600MB is FALSE (only 450MB) → OOM
```

With non-contiguous fragmented VRAM, the server experiences OOM despite having free memory.

### Unified Paging: The Solution

Unified Paging manages both KV cache entries and adapter weights in the same pool of fixed-size pages:

```python
PAGE_SIZE = 16MB  # configurable, trades fragmentation vs overhead

class UnifiedMemoryPool:
    def __init__(self, total_vram_after_base_model):
        n_pages = total_vram_after_base_model // PAGE_SIZE
        self.free_pages = list(range(n_pages))
        self.allocated = {}  # id → [page_ids]

    def allocate(self, item_id, size_bytes):
        n_pages = math.ceil(size_bytes / PAGE_SIZE)
        if len(self.free_pages) < n_pages:
            raise OOMError("not enough pages")
        pages = self.free_pages[:n_pages]
        self.free_pages = self.free_pages[n_pages:]
        self.allocated[item_id] = pages
        return pages

    def free(self, item_id):
        pages = self.allocated.pop(item_id)
        self.free_pages.extend(pages)  # pages returned to pool
```

Because pages are fixed-size and interchangeable, any freed pages are immediately usable for any new allocation — regardless of shape or rank. External fragmentation is impossible.

### Proof: Unified Paging Is Fragmentation-Free

**Claim:** With fixed-size paging, the maximum wasted memory is `(n_adapters + n_requests) × PAGE_SIZE`.

**Proof sketch:**
- Each allocation wastes at most `PAGE_SIZE - 1` bytes of the last page (internal fragmentation)
- For `n_a` adapters and `n_r` KV cache entries: waste ≤ `(n_a + n_r) × (PAGE_SIZE - 1)`
- This is O(PAGE_SIZE) per item, not O(item_size) — choosing PAGE_SIZE = 16MB gives ≤16MB waste per item
- External fragmentation: zero, because any collection of free pages can serve any allocation

**Practical result:** With PAGE_SIZE = 16MB and 1000 adapters, maximum waste is 16 GB from internal fragmentation — but in practice much less because most adapters are much larger than one page.

### The Unified Pool's Benefit: Dynamic Rebalancing

At different times, different items compete for pages:

```
Heavy prefill phase (many requests, few adapters):
  KV cache pages: 80%  |  Adapter pages: 20%  ← KV needs more

Heavy adapter switching (cold-start, diverse access):
  KV cache pages: 40%  |  Adapter pages: 60%  ← Adapters need more
```

With separate pools (vLLM v1 KV pool + fixed adapter pool), each pool must be sized for its own worst case, over-provisioning both. Unified Paging lets the system dynamically rebalance without human tuning.

### Adapter Loading Latency: The Queuing Model

When an adapter is needed but not in GPU memory, the request must wait. S-LoRA models this as an M/M/1 queue:

- Arrivals: requests for cold adapters at rate λ (requests/sec)
- Service: adapter loading bandwidth B (bytes/sec), adapter size S (bytes)
- Service time: `μ = B / S` (loadings/sec)

The expected waiting time for a cold-start request:

```
E[wait] = (λ / μ) / (μ - λ) × (1/μ)    [M/M/1 mean waiting time]
        = λ × S² / (B × (B - λ×S))
```

For S = 100MB, B = 10 GB/s (PCIe 4.0 H2D), λ = 5 req/sec:
```
μ = 10,000 MB/s / 100 MB = 100 loadings/sec
E[wait] = 5 × 100² / (100 × (100 - 5×100/100)) = 5 × 10000 / (100 × 95) ≈ 5.3ms
```

This is acceptable. But at λ = 90 req/sec (near saturation):
```
E[wait] = 90 × 10000 / (100 × (100 - 90)) = 900,000 / 1000 = 900ms
```

The queue saturates and latency explodes. This is why `--max-loaded-loras` matters: it limits how many adapters are concurrently loaded, preventing queue saturation.

### Tensor Parallelism for LoRA: The Derivation

For tensor-parallel (TP) serving across `t` GPUs, the LoRA weights must be sharded consistently with the base model's TP sharding.

**Column-parallel layers** (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`):

Base weight `W₀ ∈ ℝ^{d×k}` is sharded: GPU `j` holds `W₀[:, j×(k/t):(j+1)×(k/t)]`.

For LoRA: `W₀ + B·A` must be equivalent to the column-parallel split.
- A has shape `[r, k]` — sharded: GPU `j` holds `A[:, j×(k/t):(j+1)×(k/t)]` (same column split as `W₀`)
- B has shape `[d, r]` — **replicated**: all GPUs hold all of B

Derivation: `x @ (W₀ + B·A).T = x @ W₀.T + x @ A.T @ B.T`. For column-parallel `W₀`, `x @ W₀.T` requires each GPU to multiply its x shard against its W₀ shard and all-reduce. For LoRA: `(x_shard @ A_shard.T)` is computed locally, then `@ B.T` is computed locally (B is replicated). No extra communication beyond the base model's all-reduce.

**Row-parallel layers** (`o_proj`, `down_proj`):

Base weight `W₀ ∈ ℝ^{d×k}` is sharded by rows: GPU `j` holds `W₀[j×(d/t):(j+1)×(d/t), :]`.

For LoRA:
- A has shape `[r, k]` — **replicated** (A is small: r × k)
- B has shape `[d, r]` — sharded: GPU `j` holds `B[j×(d/t):(j+1)×(d/t), :]`

The extra cost vs Punica's single-GPU SGMV: one all-reduce for A's contribution across TP ranks. Cost is proportional to `2 × r × k × sizeof(element)` — tiny (rank-8 adapter: 2 × 8 × 4096 × 2 bytes = 131 KB) compared to the base model's all-reduce (`d × k × 2 bytes`).

This TP strategy is implemented in SGLang's production LoRA serving and described in `L2/01_sglang_lora_docs.md`.

---

## Omission 3: dLoRA — Dynamic Merge/Unmerge Credit Algorithm

**Source:** `L4/01_dlora_osdi24.md`
**Venue:** USENIX OSDI 2024
**Why omitted from COMBINEDL1L2.md:** COMBINEDL1L2.md section 12 presents the insight that skewed adapter distributions benefit from merging. dLoRA formalises this as a credit-based algorithm with adaptive thresholds and a request-adapter co-migration mechanism. The full algorithm and the theoretical analysis of when to merge are L4 material for practitioners building a multi-adapter scheduler.

---

### The Central Insight: Merge When Traffic Is Skewed

S-LoRA and Punica always serve adapters in **unmerged mode**:
```
h = W₀·x + B·A·x·scale     # base + adapter, computed separately
```

An alternative is **merged mode** — fold the adapter into the base weights:
```
W = W₀ + B·A·scale          # one-time merge (O(d×k) time, O(d×k) memory)
h = W·x                     # single GEMM, zero overhead at inference time
```

**When is merged better?** When most requests in a time window use the same adapter. The SGMV overhead (extra matmuls, kernel complexity) is a constant tax paid on every token. If 95% of requests use adapter A, serving adapter A in merged mode eliminates that constant tax for 95% of traffic.

**When is unmerged better?** When request traffic is diverse (each request uses a different adapter). Merged mode can only serve one adapter at a time — batches must be homogeneous. SGMV's overhead is justified because it enables heterogeneous batching.

### The Credit System

dLoRA maintains a **credit counter** for each adapter:

```python
credits = defaultdict(float)
MERGE_THRESHOLD = 50.0
UNMERGE_THRESHOLD = 10.0
DECAY_RATE = 0.1  # credits per second decay

def on_request_arrival(adapter_id):
    credits[adapter_id] += 1.0

def on_time_tick(dt):
    for adapter_id in list(credits.keys()):
        credits[adapter_id] -= DECAY_RATE * dt
        if credits[adapter_id] <= 0:
            del credits[adapter_id]

def should_merge(adapter_id):
    return credits.get(adapter_id, 0) > MERGE_THRESHOLD

def should_unmerge(adapter_id):
    return credits.get(adapter_id, 0) < UNMERGE_THRESHOLD
```

Credits accumulate when requests arrive and decay when they don't. A high-traffic adapter quickly crosses `MERGE_THRESHOLD` and gets merged into its own dedicated worker. If traffic drops, credits decay below `UNMERGE_THRESHOLD` and the worker unmerges to serve heterogeneous traffic.

### The Merge/Unmerge Operation

**Merge** (on a worker replica):
```python
def merge_adapter(worker, adapter_id):
    A = adapter_weights[adapter_id]["A"]
    B = adapter_weights[adapter_id]["B"]
    scale = adapter_scaling[adapter_id]
    for layer_idx in range(n_layers):
        for module in target_modules:
            delta = B[layer_idx][module].T @ A[layer_idx][module] * scale
            worker.base_model.layers[layer_idx][module].weight.data += delta.T
    worker.state = "merged"
    worker.merged_adapter_id = adapter_id
```

**Unmerge** (restore base weights):
```python
def unmerge_adapter(worker):
    adapter_id = worker.merged_adapter_id
    # Subtract the previously added delta
    A = adapter_weights[adapter_id]["A"]
    B = adapter_weights[adapter_id]["B"]
    scale = adapter_scaling[adapter_id]
    for layer_idx in range(n_layers):
        for module in target_modules:
            delta = B[layer_idx][module].T @ A[layer_idx][module] * scale
            worker.base_model.layers[layer_idx][module].weight.data -= delta.T
    worker.state = "unmerged"
    worker.merged_adapter_id = None
```

Merge/unmerge cost: proportional to number of target layers × module weight sizes. For LLaMA-7B with 7 target modules at 32 layers: ~7 × 32 × 4096² × 2 bytes = 14.7 GB written. At GPU memory bandwidth of 2 TB/s: ~7.3ms. This is why merge/unmerge should not happen on every request — only when the credit system signals sustained high traffic.

### Request-Adapter Co-Migration

**Problem:** Worker replicas can become load-imbalanced due to varying output lengths.

```
Worker 0: 100 requests, avg output 1000 tokens → busy
Worker 1: 100 requests, avg output 50 tokens  → idle 95% of the time
```

Naive load balancing migrates only future requests. dLoRA migrates both the request and its KV cache state to the underloaded worker.

**Migration decision function:**
```python
def should_migrate(src_worker, dst_worker, request):
    cost = kv_cache_bytes(request) / transfer_bandwidth
    benefit = (src_worker.estimated_completion - dst_worker.estimated_completion)
    return benefit > cost + MIGRATION_OVERHEAD
```

Only migrate when the latency reduction exceeds the migration cost. This prevents thrashing (migrating back and forth when workers have similar loads).

**Results:** 57.9× throughput over vLLM; 1.8× lower latency than S-LoRA (which uses static unmerged serving regardless of traffic shape).

---

## Omission 4: CaraServe — CPU-Assisted Prefill During Cold-Start Loading

**Source:** `L4/02_caraserve_2024.md`
**Why omitted from COMBINEDL1L2.md:** COMBINEDL1L2.md covers the cold-start problem conceptually (Pillar 1 of LoRAX: dynamic loading). CaraServe's specific solution — overlapping CPU prefill with GPU-side adapter loading — requires understanding the CPU-GPU synchronisation model and when CPU prefill is worth the overhead. It is not yet implemented in SGLang or vLLM, making it L4 material for research-track practitioners.

---

### The Cold-Start Timeline Without CaraServe

```
t=0:  Request arrives for adapter A (not in VRAM)
t=0:  Start H2D transfer: adapter A → GPU VRAM  (100-500ms for 50-200MB)
t+Δ: H2D transfer completes
t+Δ: GPU runs prefill (compute-bound, fast on GPU)
t+Δ+ε: First token generated

TTFT = H2D_time + GPU_prefill_time + first_decode_step
     = 100-500ms + 20-200ms + 10-30ms
     = 130-730ms cold start
```

### CaraServe's CPU-Overlap Solution

```
t=0:  Request arrives for adapter A (not in VRAM)
t=0:  Simultaneously:
      Thread 1: Start H2D transfer: adapter A → GPU VRAM
      Thread 2: Start CPU prefill using CPU-resident adapter weights
t+α: CPU prefill completes (α = CPU prefill time)
t+β: H2D transfer completes (β = adapter load time)
t+max(α,β): GPU decode begins using CPU-computed KV state
             (if α > β: GPU was idle waiting for CPU; still faster than serial)
             (if β > α: CPU was idle waiting for GPU; overlap fully exploited)

TTFT = max(H2D_time, CPU_prefill_time) + first_decode_step
     ≈ max(100-500ms, 200-800ms) + 10-30ms
     ← CPU prefill can be slower than GPU while still hiding H2D cost
```

### When CPU Prefill Is Worthwhile

CPU prefill is worthwhile if and only if:
```
max(H2D_time, CPU_prefill_time) < H2D_time + GPU_prefill_time
⟺ CPU_prefill_time < H2D_time + GPU_prefill_time
⟺ CPU_prefill_time - GPU_prefill_time < H2D_time
```

Since CPU prefill is always slower than GPU prefill (`CPU_prefill > GPU_prefill`), the LHS is positive. The condition becomes: "Is the CPU overhead less than the H2D transfer time?"

**For low-rank adapters and long prompts:** H2D time is dominated by adapter size (small for low rank) while CPU prefill time grows with prompt length. The condition is:
```
prompt_length × (CPU_time_per_token - GPU_time_per_token) < adapter_size / H2D_bandwidth
```

CaraServe includes an offline profiler that maps this condition to a (rank, prompt_length) lookup table, consulted at runtime for each cold-start request.

### CPU-GPU Synchronisation Mechanism

The challenge is safely handing off the KV cache state computed on CPU to the GPU for decode:

```python
# Pinned (page-locked) memory for fast DMA
cpu_kv_buffer = torch.empty(kv_shape, pin_memory=True)

# CPU prefill fills cpu_kv_buffer
cpu_prefill_thread(request, adapter_weights_cpu, cpu_kv_buffer)

# DMA transfer CPU buffer → GPU KV cache (fast, via pinned memory)
gpu_kv_buffer = cpu_kv_buffer.cuda(non_blocking=True)
torch.cuda.synchronize()  # wait for DMA

# GPU decode begins — KV cache is ready
decode_step(request, gpu_kv_buffer)
```

The pinned memory allocation eliminates the intermediate copy through pageable host memory, reducing DMA transfer overhead by 30–50%.

### Rank-Aware Scheduling

CaraServe prioritises requests based on their expected completion time, accounting for adapter rank:

```python
def compute_urgency(request, current_time):
    rank = get_adapter_rank(request.adapter_id)
    deadline = request.arrival_time + request.ttft_slo
    remaining = deadline - current_time

    # Higher rank → more compute → needs more lead time
    compute_estimate = rank * request.prompt_length * FLOPS_PER_TOKEN
    return compute_estimate / remaining  # higher urgency = schedule first
```

A rank-64 adapter request arriving at the same time as a rank-8 request with the same SLO deadline gets scheduled first because it requires more GPU time to complete before the deadline.

**Results:** 1.4× average latency speedup; 99% SLO attainment vs 85% for S-LoRA.

---

## Omission 5: LoRA Variant Mathematics — DoRA, rsLoRA, PiSSA, EVA

**Source:** `L2/03_hf_peft_lora.md`
**Why omitted from COMBINEDL1L2.md:** COMBINEDL1L2.md covers standard LoRA initialisation (Kaiming-uniform for A, zeros for B) and the `lora_alpha/r` scaling. The variant algorithms — DoRA's magnitude/direction decomposition, rsLoRA's `alpha/sqrt(r)` scaling stability analysis, PiSSA's SVD initialisation, EVA's data-driven rank allocation — require engaging with the mathematical motivation for each modification. They are PEFT configuration options that practitioners may need for specific use cases.

---

### rsLoRA: Rank-Stabilised Scaling

**Standard LoRA scaling:** `lora_alpha / r`

At rank `r=8` with `alpha=16`: scale = 2.0
At rank `r=64` with `alpha=16`: scale = 0.25

The effective scale decreases with rank. When using higher ranks (for more expressive adaptation), the scale factor drops, effectively reducing the adapter's contribution. This creates a counterintuitive situation: higher-rank adapters can have less impact than lower-rank adapters with the same `alpha`.

**rsLoRA scaling:** `lora_alpha / sqrt(r)`

At rank `r=8` with `alpha=16`: scale = 5.66
At rank `r=64` with `alpha=16`: scale = 2.0

The `1/sqrt(r)` scaling keeps the adapter contribution more stable across ranks. This follows the theory of Neural Tangent Kernels: for width-scaled networks, the correct learning rate scaling is `1/sqrt(width)` rather than `1/width`.

```python
# Standard LoRA
config = LoraConfig(r=64, lora_alpha=16, use_rslora=False)
# scale = 16/64 = 0.25 — very small with high rank

# rsLoRA
config = LoraConfig(r=64, lora_alpha=16, use_rslora=True)
# scale = 16/sqrt(64) = 2.0 — reasonable at high rank
```

**When to use rsLoRA:** When experimenting with ranks > 32, or when standard LoRA fine-tunes are underperforming due to the diminishing scale factor.

### DoRA: Weight-Decomposed Low-Rank Adaptation

Standard LoRA updates a weight matrix with an additive delta `ΔW = B·A`. DoRA decomposes the weight update into two separate components:

```
Standard LoRA:   W' = W₀ + B·A·scale

DoRA:            W' = m ⊙ (W₀ + B·A·scale) / ||W₀ + B·A·scale||
                       ↑                    ↑
                  magnitude              direction
                  (scalar per column)    (normalised matrix)
```

- **Direction** is handled by standard LoRA (`B·A` rank decomposition)
- **Magnitude** is handled by a separate learnable vector `m ∈ ℝ^{out_dim}` (one scalar per output dimension)

This decomposition mirrors how weight updates during full fine-tuning behave: the magnitude and direction of weight changes have different learned patterns. By separating them, DoRA can learn richer adaptations with the same parameter count.

```python
config = LoraConfig(r=8, lora_alpha=16, use_dora=True)
# Adds ~out_dim extra parameters per targeted layer for the magnitude vectors
```

**DoRA overhead:** 1.17–1.39× training time (from decomposed forward pass). For inference: merge the adapter to eliminate overhead:

```python
merged = peft_model.merge_and_unload()
# After merging, DoRA = standard model, zero inference overhead
```

**DoRA runtime inference** (without merging) requires storing the magnitude scaling separately from the B·A delta, which is why Layer 20's `LoRAAdapter.apply()` doesn't directly support DoRA — it would require storing and applying the magnitude vector.

### PiSSA: Principal Singular Values and Singular Vectors

Standard LoRA initialises `B=0, A=random` (identity transform at init). This means the first training steps must "discover" which directions to adapt before making progress.

PiSSA initialises using the **principal singular values** of `W₀`:

```python
# PiSSA initialisation
U, S, Vt = torch.linalg.svd(W0)

# A initialised as top-r rows of Vt (principal input directions)
A_init = Vt[:r, :]                    # [r, k]

# B initialised as top-r columns of U * diag(S[:r])
B_init = U[:, :r] @ torch.diag(S[:r])  # [d, r]

# Residual: modify W₀ to be the "rest" after removing principal components
W0_residual = W0 - B_init @ A_init    # [d, k]
```

Now `B·A` starts at the principal components of `W₀` (the directions with highest variance), and the model is adapted by learning corrections to those components. This is fundamentally different from learning from a zero-delta initialisation.

**Key properties:**
- `B_init @ A_init = U[:,:r] @ diag(S[:r]) @ Vt[:r,:]` — the rank-r approximation of `W₀`
- At initialisation: `W_total = W₀_residual + B·A = W₀` — same behaviour as base model
- But gradients now flow through the principal directions, which converge faster

**Results:** PiSSA converges 2–5× faster than standard LoRA on instruction-following benchmarks. The PiSSA authors attribute this to the adapter starting from the directions of maximum weight variance rather than from zero.

```python
config = LoraConfig(r=8, init_lora_weights="pissa")
# WARNING: takes several minutes for SVD on large models
# Use "pissa_niter_16" for fast approximate SVD
config = LoraConfig(r=8, init_lora_weights="pissa_niter_16")
```

### EVA: Data-Driven Adaptive Rank Allocation

Standard LoRA uses the same rank `r` for every targeted layer. EVA allocates different ranks to different layers based on how much adaptation each layer needs.

**EVA procedure:**

1. Run the model on a calibration dataset, recording input activations at each targeted layer
2. For each layer's activations `X ∈ ℝ^{N×d}`, compute SVD: `U, S, Vt = svd(X.T @ X)`
3. The "explained variance ratio" for rank `r` at this layer: `EVR(r) = sum(S[:r]) / sum(S)`
4. Allocate more ranks to layers with lower EVR (they need more capacity to represent the activation distribution)

```python
from peft import LoraConfig, EvaConfig

config = LoraConfig(
    r=8,            # default rank
    init_lora_weights="eva",
    eva_config=EvaConfig(rho=2.0)  # rho = max allowed rank multiplier
)

# After EVA analysis:
# Layer 0 (q_proj): EVR at r=8 is 0.95 → keep r=8
# Layer 16 (q_proj): EVR at r=8 is 0.40 → needs more → assigned r=16
# Layer 31 (down_proj): EVR at r=8 is 0.65 → assigned r=12
```

`rho=2.0` means the maximum rank any layer can receive is `2r`. This prevents extreme rank allocation while allowing meaningful per-layer adaptation.

**When to use EVA:** For fine-tuning on complex tasks where different model components have very different adaptation needs. Adds calibration time upfront but reduces total parameter count for equivalent quality compared to uniform rank.

---

## Omission 6: ServerlessLoRA — Backbone Sharing and Serverless-Specific Optimisations

**Source:** `L4/04_serverlesslora_2025.md`
**Why omitted from COMBINEDL1L2.md:** COMBINEDL1L2.md does not discuss serverless deployment. ServerlessLoRA's findings (99% weight redundancy in serverless LoRA functions, backbone sharing approach, contention-aware batching) are specific to the serverless execution model (AWS Lambda, Modal, Replicate) and are not applicable to the traditional long-lived server deployment model covered in COMBINEDL1L2.md's SGLang and vLLM sections.

---

### Why Standard Serverless Fails for LoRA

Serverless LLM inference functions run in isolated containers, each loading their own full model:

```
Without backbone sharing (standard serverless):
Function 1: [LLaMA-7B base: 14GB] + [adapter_1: 50MB]  = 14.05 GB
Function 2: [LLaMA-7B base: 14GB] + [adapter_2: 50MB]  = 14.05 GB
...
Function N: [LLaMA-7B base: 14GB] + [adapter_N: 50MB]  = 14.05 GB
```

At N=10 concurrent functions: 140 GB VRAM, with 140 GB spent on 99% identical data (14× the same base model). This is the "massive parameter redundancy" ServerlessLoRA identifies.

### Secure Backbone Sharing

ServerlessLoRA introduces a **shared base model** accessible to all LoRA functions, with OS-level isolation ensuring each function can only access its own adapter weights:

```
Shared Base Model (14 GB, read-only, mmap-backed):
  Function 1 → reads W₀ layers + applies adapter_1
  Function 2 → reads W₀ layers + applies adapter_2
  ...
  Function N → reads W₀ layers + applies adapter_N
```

**Security model:**
- Base model is mapped read-only into each function's address space (kernel-enforced)
- Adapter weights are private to each function (separate VRAM allocations)
- KV caches are private (separate VRAM allocations)
- Functions cannot access each other's adapters or KV caches

VRAM savings:
```
Without sharing: N × 14.05 GB
With sharing:    14 GB + N × 0.05 GB
Savings at N=10: 140.5 GB → 14.5 GB = 89.6% reduction
```

### Pre-Loading: Eliminating Cold Start via Prediction

Serverless functions experience cold starts when the container is first initialised. Standard LoRA serving adds **adapter cold start** on top of model cold start:

```
Cold start without pre-loading:
  t=0:   Function spins up
  t=2s:  Base model loaded from object storage (14 GB at ~7 GB/s)
  t=3s:  Adapter loaded from object storage (50 MB at 7 GB/s)
  t=3s+ε: First request processed

TTFT = 3+ seconds for cold start
```

ServerlessLoRA pre-loads adapters **during the function's warm-up period** (before the first request):

```python
def warm_up():
    # Load base model (required, cannot skip)
    load_base_model()

    # Predict and pre-load hot adapters
    hot_adapters = predict_hot_adapters(recent_traffic_log, top_k=5)
    for adapter_id in hot_adapters:
        prefetch_adapter(adapter_id)  # async, during idle time

# When first request arrives:
def handle_request(request):
    adapter_id = request.adapter_id
    if adapter_already_loaded(adapter_id):
        process_immediately(request)   # warm hit
    else:
        load_adapter(adapter_id)       # still cold, but less common
        process(request)
```

The prediction uses historical access patterns (same idea as L4/05 Predictive-LoRA's LSTM, but simpler — top-K by recent frequency).

**Cold start reduction:** 68% fewer cold starts in experiments with production-like traces.

### Contention-Aware Batching

When multiple LoRA functions attempt to prefill simultaneously, peak VRAM pressure spikes:

```
Normal operation: one function active
  [Base: 14GB] [Adapter A: 50MB] [KV A: 2GB]  → 16.05 GB

Burst: 3 functions active simultaneously
  [Base: 14GB] [Adapter A: 50MB] [KV A: 2GB]
              [Adapter B: 50MB] [KV B: 2GB]
              [Adapter C: 50MB] [KV C: 2GB]  → 20.15 GB
  ↑ May exceed 20 GB GPU VRAM → OOM
```

ServerlessLoRA's contention-aware scheduler:
1. Monitors current VRAM pressure in real time
2. When contention threshold is reached: offload inactive adapters from VRAM to CPU
3. Staggers prefill execution: new prefills are queued until current VRAM pressure drops
4. Priorities based on SLO deadlines — tighter deadlines preempt later ones

**Combined results on Azure trace:**
- TTFT: 820ms → 115ms (86% reduction)
- Cost: $12.5/M tokens → $1.4/M tokens (89% reduction)
- Throughput: 3.2× improvement

---

## Omission 7: Predictive-LoRA — LSTM Traffic Predictor and Page-Based Memory

**Source:** `L4/05_predictive_lora_2025.md`
**Why omitted from COMBINEDL1L2.md:** COMBINEDL1L2.md does not cover serverless deployment. Predictive-LoRA's LSTM-based traffic prediction and page-based memory management are advanced topics specific to serverless LoRA serving where cold-start frequency is high and adapter access patterns are predictable. The prediction mechanism requires ML training infrastructure that is out of scope for COMBINEDL1L2.md.

---

### The Reactive Loading Problem

S-LoRA and LoRAX use reactive loading: the server starts loading an adapter *after* a request arrives for it. This is optimal for minimising VRAM usage (never load adapters you don't need), but suboptimal for latency when cold starts are frequent.

**Timeline comparison:**

```
Reactive (S-LoRA):
  t=0: Request for adapter A arrives
  t=0: Start H2D transfer (adapter not in VRAM)
  t+100ms: Transfer completes
  t+100ms+: Processing begins
  TTFT penalty: 100ms+ cold start on every cache miss

Proactive (P-LoRA):
  t=-100ms: LSTM predicts adapter A will be needed
  t=-100ms: Start H2D transfer (speculative prefetch)
  t=0: Request for adapter A arrives
  t=0: Adapter already in VRAM
  t=0+: Processing begins immediately
  TTFT penalty: 0ms (if prediction correct)
```

Proactive loading shifts the cost from latency-critical (request processing) to latency-tolerant (idle time) — the same principle as CPU instruction prefetching.

### The LSTM Traffic Predictor

**Architecture:**

```
Input at time t (per time window):
  [r₀(t), r₁(t), ..., rₙ(t)]   — request rates per adapter (n adapters)
  [c₀(t), c₁(t), ..., cₙ(t)]   — cumulative counts per adapter
  [hour_of_day, day_of_week]     — temporal features

LSTM hidden state: 64 units × 2 layers (small by design)

Output:
  [r̂₀(t+1), r̂₁(t+1), ..., r̂ₙ(t+1)]   — predicted rates for next window
```

**Prediction → prefetch decision:**

```python
predicted_rates = lstm.predict(current_window)
threshold = percentile(predicted_rates, 80)  # top 20% by predicted rate

for i, rate in enumerate(predicted_rates):
    if rate > threshold and adapter_i not in gpu_vram:
        schedule_prefetch(adapter_ids[i])     # initiate H2D transfer
    elif rate <= threshold and adapter_i in gpu_vram:
        mark_evictable(adapter_ids[i])        # LRU eviction candidate
```

**Training:** Online learning — the LSTM is fine-tuned continuously on new traffic observations, adapting to concept drift (new popular adapters, seasonal patterns). The model checkpoint is tiny (~100KB) and can be updated every few minutes without interrupting serving.

**Why LSTM over simpler predictors:**

| Method | Can model | Limitation |
|---|---|---|
| LRU/LFU | Recent/frequent access | Cannot predict future demand before requests arrive |
| Moving average | Short-term trends | Cannot model multi-day cycles or sudden shifts |
| LSTM | Long-range dependencies, cycles, trend shifts | Needs training, slightly more complexity |

The LSTM captures patterns like "adapter_A is popular 9am-5pm weekdays, adapter_B spikes after product launches" — patterns that LRU and moving averages miss entirely.

### Page-Based Adapter Memory Management

The second innovation in P-LoRA is treating GPU VRAM as a paged memory space:

```python
PAGE_SIZE = 64MB  # or configurable per-deployment

class PagedAdapterPool:
    def __init__(self, total_adapter_vram_bytes):
        n_pages = total_adapter_vram_bytes // PAGE_SIZE
        self.pages = [Page(i) for i in range(n_pages)]
        self.free_list = deque(range(n_pages))
        self.adapter_pages = {}  # adapter_id → frozenset of page_ids

    def allocate(self, adapter_id, adapter_size_bytes):
        n = math.ceil(adapter_size_bytes / PAGE_SIZE)
        if len(self.free_list) < n:
            self._evict_lru_until(n)
        page_ids = [self.free_list.popleft() for _ in range(n)]
        self.adapter_pages[adapter_id] = frozenset(page_ids)
        return page_ids

    def evict(self, adapter_id):
        for pid in self.adapter_pages.pop(adapter_id):
            self.free_list.append(pid)

    def _evict_lru_until(self, needed):
        while len(self.free_list) < needed:
            lru_id = self.lru_tracker.least_recent()
            self.evict(lru_id)
```

**Why pages eliminate fragmentation:**

Adapters of different ranks have different sizes (`r=4 → ~25MB, r=64 → ~400MB`). Without pages, repeated load/evict cycles fragment VRAM:
```
After 100 cycles: 2GB free but in ~50 small non-contiguous gaps
Cannot allocate a 400MB rank-64 adapter despite having 5× its size available
```

With PAGE_SIZE=64MB pages, any 7 free pages can serve a 400MB adapter regardless of which pages they are.

**Combined P-LoRA results** (Azure Functions trace):
- Throughput: 1.52× over S-LoRA
- TTFT: 35% reduction
- VRAM utilisation: 87%+ (vs 60-70% for S-LoRA with fragmentation)
- Cold start rate: 68% reduction (from LSTM prefetching)

---

## Omission 8: Loquetier — Virtualised Module and Unified Fine-Tuning + Serving

**Source:** `L4/03_loquetier_neurips25.md`
**Venue:** NeurIPS 2025
**Why omitted from COMBINEDL1L2.md:** COMBINEDL1L2.md covers inference serving only. Loquetier's contribution is specifically for production systems that must simultaneously train new adapters and serve existing adapters from the same GPU cluster. This mixed workload requires gradient computation during serving and coordinated batching of training and inference steps — concerns beyond the scope of COMBINEDL1L2.md's serving-only coverage.

---

### The Gap Loquetier Fills

Prior work (Punica, S-LoRA, dLoRA) focused exclusively on inference serving:
- Base model: frozen
- Adapters: inference-only (no gradient computation)
- Workload: 100% requests, 0% training

Real production systems combine training and serving:
- Model improvement pipeline: continuously fine-tune adapters on new user data
- Active learning: update adapter weights based on user feedback in real time
- RLHF (Reinforcement Learning from Human Feedback): train while serving

Running training and serving on **separate GPU clusters** doubles infrastructure cost and creates a deployment lag (fine-tune on one cluster, deploy to another). Loquetier enables both on the same cluster.

### The Virtualised Module (VM) Design

Loquetier wraps each base model layer in a Virtualised Module that maintains a pool of adapters, each independently tagged for either training or serving:

```python
class VirtualisedModule(nn.Module):
    def __init__(self, base_layer: nn.Linear):
        super().__init__()
        self.base_layer = base_layer  # frozen, shared
        self.adapters: Dict[str, LoRAAdapter] = {}
        # adapter_id → {A, B, mode ("train" | "serve"), requires_grad}

    def add_adapter(self, adapter_id, r, mode="serve"):
        A = nn.Parameter(torch.zeros(r, self.base_layer.in_features), requires_grad=(mode=="train"))
        B = nn.Parameter(torch.zeros(self.base_layer.out_features, r), requires_grad=(mode=="train"))
        self.adapters[adapter_id] = {"A": A, "B": B, "mode": mode}

    def forward(self, x, batch_metadata):
        base_out = self.base_layer(x)

        # Per-token adapter routing
        for adapter_id, token_indices in batch_metadata.adapter_groups.items():
            if adapter_id not in self.adapters:
                continue
            A = self.adapters[adapter_id]["A"]
            B = self.adapters[adapter_id]["B"]
            mode = self.adapters[adapter_id]["mode"]

            x_group = x[token_indices]
            if mode == "train":
                delta = (x_group @ A.T) @ B.T * self.scaling  # gradient flows
            else:
                with torch.no_grad():
                    delta = (x_group @ A.T) @ B.T * self.scaling  # no gradient

            base_out[token_indices] += delta

        return base_out
```

Key properties:
- Base layer is **never duplicated** regardless of how many adapters are active
- Training and serving adapters coexist in the same VM
- `requires_grad=True` for training adapters — backward pass updates their A and B
- `torch.no_grad()` for serving adapters — inference has no gradient overhead
- The VM is agnostic to training algorithm (SGD, Adam, RL)

### The Fused Kernel: One Launch for Training + Serving

The naive VM implementation (above) still has one issue: separate code paths for training and serving tokens create two different kernel launches per layer, one for `requires_grad=True` tokens and one for `no_grad` tokens.

Loquetier's fused kernel handles both in a single launch:

```
Input batch: [train_tokens: 32, serve_tokens: 128]

Naive:
  Launch kernel 1: process train_tokens with gradient tracking (32 tokens)
  Launch kernel 2: process serve_tokens without gradient (128 tokens)
  → 2 kernel launches, 2 memory round-trips

Fused:
  Launch single kernel:
    Thread blocks handle both token types
    Each thread block checks its token type and sets gradient flag
    Gradient accumulation buffer allocated only for train tokens
    Single memory round-trip for all activations
  → 1 kernel launch, 1 memory round-trip
```

The single-launch design reduces kernel scheduling overhead (particularly important for many small adapters) and improves memory access locality (base model weights loaded once for the combined batch).

### Training + Serving Batch Construction

The scheduler constructs mixed batches that balance training and serving objectives:

```python
def build_mixed_batch(training_requests, serving_requests, max_tokens):
    batch = Batch()
    remaining = max_tokens

    # SLO-driven serving allocation
    serving_tokens = min(remaining, slo_driven_serving_quota(serving_requests))
    batch.add_serving(serving_requests[:serving_tokens])
    remaining -= serving_tokens

    # Fill remainder with training tokens
    training_tokens = min(remaining, training_request_size)
    batch.add_training(training_requests[:training_tokens])

    return batch
```

The SLO-driven quota ensures serving requests meeting their TTFT/ITL SLOs are prioritised — training tokens fill the remaining capacity. When serving load is low, more of the batch is training; at peak serving load, training gets minimal resources.

### Results

On A100 80 GB with LLaMA-7B:

| Task | Baseline | Loquetier | Improvement |
|---|---|---|---|
| Inference only | SOTA co-serving system | Loquetier | **3.0× throughput** |
| Training only | PEFT | Loquetier | ~1.5× throughput |
| Unified (training + serving) | PEFT (sequential) | Loquetier | **46.4× SLO attainment** |

The 46.4× SLO attainment improvement is the headline result: PEFT's single-adapter sequential training + serving (alternating between training steps and inference steps) causes massive serving latency violations under any real request load. Loquetier's fused mixed batches maintain both training progress and serving SLOs simultaneously.

---

## Omission 9: InfiniLoRA — Disaggregated LoRA Execution for MoE Models

**Source:** `L4/06_infinilora_2026.md`
**Why omitted from COMBINEDL1L2.md:** InfiniLoRA represents the frontier of multi-LoRA architecture (April 2026) and requires understanding both PD Disaggregation (Layer 19) and multi-LoRA serving (Layer 20) simultaneously. The disaggregated LoRA Server design, GPU-initiated communication, and MoE LoRA memory analysis require engaging with systems at the intersection of two advanced topics.

---

### Why MoE Models Break Coupled LoRA Serving

For dense models (LLaMA-7B, Qwen3-0.6B), LoRA adds a small fraction of model size:

```
LLaMA-7B, 7 target modules, rank-8:
LoRA params = 32 layers × 7 modules × (4096×8 + 8×4096) = 28M params ≈ 56MB
Base model: 7B params = 14,000 MB
LoRA overhead: 0.4%
```

For MoE models (Mixtral-8×7B, DeepSeek-V2), LoRA applied to expert layers scales with expert count:

```
Mixtral-8×7B, rank-8, with LoRA on expert layers:
Expert FFN per layer: 8 experts × 2 × 14336 × 4096 = 938M params
LoRA for experts: 8 experts × (4096×8 + 8×14336) = 1.2M per layer
32 layers: 38M extra params ≈ 76MB per adapter

BUT: with 64 experts (like in DeepSeek-V2) and rank-16:
64 experts × 32 layers × 2 × (hidden × rank) ≈ 2.4B LoRA params ≈ 4.8GB per adapter
```

For a high-rank adapter on a large MoE model, LoRA weights can approach the same scale as the KV cache. S-LoRA's assumption that "adapters are small" breaks down.

### InfiniLoRA's Architecture: The Dedicated LoRA Server

Instead of running LoRA computation on the same GPU as the base model:

```
Standard (coupled):
  Base Model GPU 0
  ┌─────────────────────────────────────────────────────┐
  │ Base model weights [fixed]  |  LoRA A/B weights [N]  │
  │ Forward pass                |  SGMV kernels           │
  │ KV cache                    |                         │
  └─────────────────────────────────────────────────────┘
         ↑ VRAM competition between KV cache and LoRA weights for MoE

InfiniLoRA (disaggregated):
  Base Model GPUs 0-3 (4 GPUs)
  ┌─────────────────────────────────────────────────────┐
  │ Base model weights [fixed]                          │
  │ Forward pass (base contribution only)               │
  │ Full KV cache budget (no LoRA weights!)             │
  └─────────────────────────────────────────────────────┘
          ↕  GPU-initiated RDMA: activations ↓ + deltas ↑

  LoRA Server GPUs 4-7 (4 dedicated GPUs)
  ┌─────────────────────────────────────────────────────┐
  │ All adapter A/B weights for all N adapters          │
  │ SGMV/TP-aware LoRA computation                      │
  │ No base model weights needed                        │
  └─────────────────────────────────────────────────────┘
```

**Why this helps for MoE:**
- Base model GPUs no longer compete with LoRA weights for VRAM
- Larger KV cache → larger batch size → better base model GPU utilisation
- LoRA Server scales independently: add more LoRA Server GPUs as adapter catalog grows

### Critical-Path Optimisation: GPU-Initiated Communication

In standard disaggregated systems, the CPU orchestrates GPU-to-GPU transfers:

```
CPU-orchestrated (standard):
  Base GPU → [PCIe to CPU RAM] → [CPU: schedule NCCL] → [PCIe to LoRA GPU]
  Overhead: ~100μs CPU interrupt + PCIe round trips = ~200μs total
```

InfiniLoRA uses **GPU-initiated communication** via NCCL's P2P API:

```
GPU-initiated (InfiniLoRA):
  Base GPU → [NVLink/InfiniBand RDMA directly to LoRA GPU]
  Overhead: ~10μs (NVLink P2P latency)
```

The CPU is entirely bypassed. The GPU triggers the send/receive via CUDA stream primitives:

```cuda
// On Base Model GPU: send activations to LoRA Server
ncclSend(activations_ptr, n_elements, ncclFloat16, 
         lora_gpu_rank, nccl_comm, cuda_stream);

// On LoRA Server GPU: receive and compute delta
ncclRecv(activations_buffer, n_elements, ncclFloat16,
         base_gpu_rank, nccl_comm, cuda_stream);
sgmv_kernel(activations_buffer, adapter_weights, delta_buffer, ...);
ncclSend(delta_buffer, n_elements, ncclFloat16,
         base_gpu_rank, nccl_comm, cuda_stream);
```

The entire communication + LoRA computation is a sequence of CUDA stream operations — no CPU involvement after initial setup. This enables the LoRA Server's work to overlap with the next layer's base model computation.

### SLO-Driven Provisioning

InfiniLoRA includes an offline capacity planner that determines LoRA Server sizing:

```python
def provision_lora_servers(
    request_rate_per_adapter: Dict[str, float],  # adapters/sec
    slo_ttft_ms: float,
    slo_itl_ms: float,
    base_model_forward_time_ms: float,
    adapter_compute_time_per_rank: float,         # ms per token per rank
    hw_cost_per_gpu_hour: float,
) -> int:
    # LoRA computation time per token (across all active adapters):
    # Sum over active adapters: compute_time × rank × batch_size
    # Must fit within (slo_itl_ms - base_model_forward_time_ms)
    
    headroom_ms = slo_itl_ms - base_model_forward_time_ms
    lora_work_per_step = sum(request_rate * rank * adapter_compute_time_per_rank
                             for adapter_id, request_rate in request_rate_per_adapter.items())
    
    required_throughput = lora_work_per_step / headroom_ms
    gpus_needed = math.ceil(required_throughput / lora_server_throughput_per_gpu)
    return gpus_needed
```

This offline analysis enables elastic provisioning: scale LoRA Server GPUs up during high-adapter-diversity periods, scale down during homogeneous traffic.

### Evaluation Results

On H100 cluster with DeepSeek-V2 (MoE, 128 experts per layer):

| Metric | S-LoRA (coupled) | InfiniLoRA (disaggregated) |
|---|---|---|
| Serviceable request rate | 1× | **3.05×** |
| Adapters satisfying SLO | 46% | **100%** |
| Base model GPU VRAM for KV cache | 30% (VRAM crowded by LoRA) | 64% (LoRA on separate GPUs) |
| Tail latency (P99 TTFT) | High (VRAM pressure causes stalls) | Stable |

The 3.05× serviceable request rate improvement comes from: (1) base GPUs no longer constrained by LoRA VRAM → larger batches, (2) LoRA computation overlaps with base model decode via async GPU-initiated communication, (3) LoRA Server can use dedicated hardware optimised for SGMV (memory-bandwidth-optimised) vs base model hardware (compute-optimised).

### The Disaggregation Progression

InfiniLoRA extends the disaggregation principle from Layer 19 (PD Disaggregation) to the LoRA serving dimension:

| Disaggregated component | Paper | What it separates |
|---|---|---|
| Prefill vs Decode | DistServe (OSDI 2024) | Compute-bound vs memory-bound phases |
| KV cache vs Compute | Mooncake (FAST 2025) | Cache storage vs compute resources |
| LoRA vs Base model | **InfiniLoRA (2026)** | Adapter compute vs base model compute |

Each disaggregation dimension enables independent scaling and hardware specialisation for its specific resource bottleneck. The trend points toward fully componentised inference infrastructure where each computation type runs on purpose-sized hardware.
