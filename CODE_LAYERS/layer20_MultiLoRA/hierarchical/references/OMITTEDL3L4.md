# Multi-LoRA Inference — Material Omitted from COMBINEDL3L4.md

**What this file is:** The full text of every section omitted from `COMBINEDL3L4.md`. The "What Sits in OMITTEDL3L4.md" appendix of `COMBINEDL3L4.md` names each omission and explains why it was excluded. This file preserves the complete material so nothing is lost.

**Parent file:** `COMBINEDL3L4.md` (L1 + L2 + L3 + L4 synthesis)
**Sources:** L3/02 (Punica CUDA), L3/03 (S-LoRA formal analysis), L4/01 (dLoRA convergence), L4/02 (CaraServe profiler), L4/05 (Predictive-LoRA LSTM), L4/06 (InfiniLoRA provisioning), L3/01 + L2/03 (LoRA variant math)

---

## Omission 1: Punica — SGMV CUDA Thread Block Assignment

**Source:** `L3/02_punica_mlsys24.md`
**Why omitted from COMBINEDL3L4.md:** COMBINEDL3L4 §4 describes the SGMV algorithm at the level of "one kernel launch, segments grouped by adapter, gather scatter indices." The Punica paper's actual CUDA implementation — how thread blocks are assigned to segments, how the kernel handles variable-length segments without padding overhead, and how non-contiguous adapter page addresses are gathered — requires CUDA programming knowledge to engage with. The concept is in COMBINEDL3L4; the implementation is here.

---

### The Core CUDA Design Challenge

Standard cuBLAS and CUTLASS kernels assume that all matrix multiplications in a batch use the same operand dimensions and that all matrices are contiguous in memory. SGMV violates both assumptions simultaneously:

1. **Variable dimensions**: adapters can have different ranks (r=4 for one adapter, r=64 for another). Each segment's weight matrices have different column counts.
2. **Non-contiguous memory**: after S-LoRA's Unified Paging evicts and re-loads adapters, the A and B matrices for a given adapter may be scattered across non-contiguous GPU VRAM pages.
3. **Variable segment lengths**: the number of tokens belonging to each adapter varies widely within a batch — some adapters may have one token, others dozens.

The kernel must handle all three simultaneously in a single launch with minimal padding overhead.

### Thread Block Assignment to Segments

The SGMV kernel assigns one or more CUDA thread blocks to each segment (group of tokens sharing an adapter). The assignment follows this logic:

```
For each segment s with n_s tokens and adapter rank r_s:
  compute_budget = n_s * r_s * d_model  # FLOPs for the down+up projection
  blocks_for_segment = ceil(compute_budget / BLOCK_COMPUTE_BUDGET)
  assign blocks [start, start + blocks_for_segment) to segment s
```

The constant `BLOCK_COMPUTE_BUDGET` is calibrated for the target GPU's shared memory size and register file capacity. On A100, a typical value is 256 tokens × 64 rank elements × a portion of the d_model dimension, sized so that the working set for one block fits in L1 cache.

### Gather: Accessing Non-Contiguous Pages

When S-LoRA's Unified Paging stores adapter weights as non-contiguous pages, the kernel cannot use a simple base pointer plus offset. Instead, the kernel receives a **page table** — an array of GPU virtual addresses, one per page, for each adapter's A and B matrices.

Within the kernel, accessing row `i` of adapter `a`'s A matrix proceeds as:

```
page_idx = i / ROWS_PER_PAGE
row_within_page = i % ROWS_PER_PAGE
page_base_address = page_table[adapter_a][page_idx]
element_address = page_base_address + row_within_page * r_s * sizeof(half)
```

This is a gather operation in the classical sense: the kernel gathers rows from non-contiguous locations in GPU VRAM rather than streaming from a contiguous buffer. The performance cost of the gather depends on how scattered the pages are — pages that happen to be close in physical VRAM produce cache-line hits; fully scattered pages produce L2 cache misses at every row.

The Punica implementation mitigates scatter overhead by sorting pages by physical address before assigning them to blocks, improving cache reuse across consecutive row accesses within a block.

### Handling Variable Segment Lengths

Short segments (few tokens belonging to a rare adapter) create a load imbalance problem: the thread blocks assigned to a short segment finish quickly while thread blocks assigned to a large segment are still running. Standard cuBLAS GEMM batching avoids this by requiring equal-sized operations; SGMV explicitly handles it by:

1. **Minimum segment merge**: segments below a minimum token count threshold are combined with adjacent segments by the CPU before the kernel launches, even if the combined tokens belong to different adapters. The combined segment applies each adapter's weights to only its own tokens (using a per-token adapter index lookup), trading some wasted computation in the combined block for improved occupancy.

2. **Work stealing**: if a thread block completes its assigned segment early and a "steal queue" of unfinished segment work items is non-empty, the block pulls a new work item rather than terminating. This is implemented using atomic operations on a shared work counter in global GPU memory.

### BGMV: The Decode-Phase Variant

SGMV is designed for prefill, where each segment contains multiple tokens. During decode, each active request contributes exactly one token per step. The "segment" for every adapter has length one, making the gather-segment structure of SGMV unnecessary.

BGMV (Batched Gather Matrix-Vector Multiplication) is optimized for this regime. Rather than assigning thread blocks to segments, BGMV assigns one warp (32 CUDA threads) to each token-adapter pair. Each warp independently computes:

```
down = token_activation @ A_adapter.T  # shape [1, rank]
delta = down @ B_adapter.T             # shape [1, d_model]
```

Because all tokens in a decode step are independent — no token's computation depends on another's — BGMV achieves full parallelism with no synchronization overhead between warps. The gather for each warp's adapter weights is similarly gathered from non-contiguous pages using the per-adapter page table.

### Performance Comparison

On A100 80GB, LLaMA-7B, 50 adapters:

| Approach | Kernel launches per decode step | Observed overhead |
|---|---|---|
| Naive per-adapter loop | 100 (50 adapters × 2 projections) | ~32ms scheduling overhead |
| SGMV (prefill) | 1 | +2ms per token |
| BGMV (decode) | 1 | +1.5ms per token |

The near-constant overhead of SGMV and BGMV regardless of adapter count is the key property that enables the "50 adapters at the cost of one" result.

---

## Omission 2: S-LoRA — Unified Paging Formal Analysis and Queuing Model

**Source:** `L3/03_slora_mlsys24.md`
**Why omitted from COMBINEDL3L4.md:** COMBINEDL3L4 §7 describes Unified Paging conceptually: one pool shared by adapter pages and KV pages, LRU eviction, dynamic rebalancing. The S-LoRA paper provides the formal analysis that the conceptual description glosses over — specifically, the proof that Unified Paging eliminates fragmentation, the queuing model that predicts adapter loading latency under load, and the formal derivation of the tensor parallelism sharding strategy.

---

### Proof: Unified Paging Eliminates Fragmentation

**Claim:** Given a pool of P equal-size pages, any sequence of adapter load/evict operations leaves the pool in a state where any allocation requiring at most P pages can always be satisfied from the free list, regardless of the order of prior operations.

**Proof sketch:**

The fragmentation problem in non-paged allocation arises when free space consists of holes too small to satisfy a new allocation. With fixed-size pages:

- Free space is always measured in whole pages.
- A new allocation of size `s` bytes requires `ceil(s / PAGE_SIZE)` pages.
- If `ceil(s / PAGE_SIZE)` pages are free, the allocation always succeeds, because any `k` free pages are combinable — there is no notion of "adjacency" in the allocation.

The key property is that pages are indistinguishable: a page that held adapter A's row 7 is, after eviction, identical to a page that held adapter B's row 3. Any free page can serve any allocation. This is the direct analog of how OS virtual memory works: physical page frames are interchangeable; only the virtual-to-physical mapping is per-allocation.

**Corollary:** In a pool with P pages, where the base model occupies B pages (static, never evicted), the remaining P - B pages can serve any combination of KV cache and adapter pages as long as their total count does not exceed P - B. There is no fragmentation-induced OOM.

**Contrast with variable-size allocation:** In a non-paged allocator (e.g., C `malloc`), after a sequence of alloc/free operations of varying sizes, the largest allocatable contiguous region can be much smaller than the total free bytes. With pages, total free pages = allocatable pages, always.

### Queuing Model for Adapter Loading Latency

S-LoRA models the adapter loading process as a queuing system to predict cold-start latency under given traffic conditions.

**System model:**

- Requests arrive as a Poisson process with rate λ (requests per second).
- Each request specifies one adapter, drawn from a pool of N adapters with access probability distribution p₁, p₂, ..., p_N (typically power-law or Zipf distributed).
- The GPU pool holds at most K adapters simultaneously (K = `max_loras_per_batch`).
- When an adapter is not in the GPU pool, it must be loaded from CPU; loading takes time L seconds (L = adapter_size / PCIe_bandwidth).

**Steady-state GPU pool hit rate:**

Under LRU eviction, the steady-state hit rate for the top-K most popular adapters is:

```
hit_rate = Σᵢ₌₁ᴷ pᵢ
```

where the adapters are sorted by popularity (p₁ ≥ p₂ ≥ ... ≥ p_N). For a Zipf distribution with parameter α and N adapters:

```
pᵢ = (i^(-α)) / (Σⱼ₌₁ᴺ j^(-α))
```

With α = 1.0 (typical for real workloads), K = 8, and N = 1000:

```
hit_rate ≈ Σᵢ₌₁⁸ (1/i) / Σⱼ₌₁¹⁰⁰⁰ (1/j) ≈ 2.72 / 7.49 ≈ 0.36
```

Only 36% of requests hit the GPU pool under this distribution with K=8. Increasing K to 32 raises this to ~0.55. The Zipf tail is long — even holding 32 adapters, nearly half the requests are cold.

**Expected cold-start penalty (M/G/1 queue approximation):**

For the cold-start request subpopulation (fraction 1 - hit_rate):
- Arrival rate to the cold-start queue: λ_cold = λ × (1 - hit_rate)
- Service time: L seconds per adapter load (deterministic for same-size adapters)
- Mean waiting time in the cold-start queue (M/D/1 with utilization ρ = λ_cold × L):

```
E[waiting_time] = (ρ × L) / (2 × (1 - ρ))
```

For ρ < 0.5, waiting time is small relative to L. For ρ → 1 (system near saturation), waiting time diverges. The practical implication: cold-start queuing latency becomes significant when the adapter loading throughput is insufficient to handle the cold-start rate, i.e., when λ × (1 - hit_rate) × L approaches 1.

This queuing model provides a design rule for K: given λ, N, the Zipf parameter α, and the loading time L, choose K large enough that ρ < 0.7 to keep queuing latency within acceptable bounds.

### Tensor Parallelism Sharding Derivation

S-LoRA's TP strategy is derived from the requirement that LoRA delta computation can be sharded across GPUs without introducing additional all-reduce operations beyond those already required by the base model's tensor parallelism.

**For column-parallel modules (q_proj, k_proj, v_proj, gate_proj, up_proj):**

In standard column-parallel TP, the base weight W₀ is sharded column-wise across T GPUs: each GPU holds columns [col_start : col_end]. The output of a column-parallel layer on GPU t is a partial result of dimension d_out/T.

For the LoRA delta `Δ = (x @ A.T) @ B.T * scale`:
- If B is column-sharded: each GPU holds B_t of shape [d_out/T, r], producing the same partial output dimension as the base layer. The A matrix input x @ A.T produces a full rank-r intermediate of shape [1, r] — this must be the same on all GPUs.
- Therefore A must be **replicated** on all GPUs.

**For row-parallel modules (o_proj, down_proj):**

In standard row-parallel TP, each GPU holds rows [row_start : row_end] of W₀ and receives the corresponding shard of the activation. An all-reduce at the end produces the full output.

For the LoRA delta:
- The A matrix down-projects the activation shard held by this GPU: x_shard @ A_shard.T produces a partial rank-r intermediate. An all-reduce across GPUs accumulates the full rank-r intermediate.
- Then B up-projects the full rank-r intermediate: this produces the full delta, which is identical on all GPUs.
- Therefore A must be **row-sharded** and B must be **replicated**.

**All-reduce cost analysis:**

With this sharding strategy:
- Column-parallel: no additional all-reduce vs base model (A is replicated; the B column-sharding produces outputs that merge naturally with the base model's all-reduce)
- Row-parallel: one additional all-reduce for the A matrix intermediate, same cost as the base model's existing all-reduce

Net cost: LoRA TP adds at most one all-reduce per row-parallel layer — identical to the base model's existing communication pattern. The LoRA delta does not increase inter-GPU communication.

---

## Omission 3: dLoRA — Convergence Analysis and Migration Cost Model

**Source:** `L4/01_dlora_osdi24.md`
**Why omitted from COMBINEDL3L4.md:** COMBINEDL3L4 §8 describes the credit-based algorithm and hysteresis thresholds qualitatively. The dLoRA paper's convergence analysis establishes that the credit system is stable — it does not oscillate between merge and unmerge modes — under mild assumptions on the arrival process. The migration cost model provides the formula that determines when co-migration is net beneficial.

---

### Convergence of the Credit Algorithm

**The stability question:** Given a credit system with merge threshold M and unmerge threshold U (U < M), and a Poisson arrival process with rate λ for a given adapter, does the system converge to a stable state or oscillate?

**The credit dynamics:**

Let `c(t)` be the credit score at time t. The dynamics are:

```
c(t + dt) = c(t) + arrivals_in_dt - decay_rate × dt
```

where `arrivals_in_dt` is a Poisson-distributed count with mean λ × dt.

In expectation:

```
E[dc/dt] = λ - decay_rate
```

**Case 1: λ > decay_rate (high traffic)**

Credits grow on average. The system reaches the MERGE_THRESHOLD in expected time:

```
E[time_to_merge] = (M - initial_credit) / (λ - decay_rate)
```

Once merged, the adapter remains merged as long as λ > decay_rate. If traffic drops (λ decreases below decay_rate), credits eventually fall to UNMERGE_THRESHOLD. The hysteresis gap (M - U) acts as a buffer against brief traffic drops.

**Case 2: λ < decay_rate (low traffic)**

Credits decay on average. The system stays unmerged unless there is a burst of arrivals large enough to push credits above M.

**Oscillation condition:** Oscillation occurs when the system alternates between merge and unmerge repeatedly in a short time window. With hysteresis, oscillation requires credits to (a) rise above M, then (b) fall below U, repeatedly. The expected time to complete one oscillation cycle is:

```
E[cycle_time] = E[time_to_merge] + E[time_to_unmerge]
              ≈ (M - U) / |λ - decay_rate|
```

For the typical parameter choice (M - U) ≈ 10 credits and |λ - decay_rate| ≈ 1 credit/second, the expected cycle time is ~10 seconds — much longer than the merge/unmerge operation itself (~100ms for a typical adapter). The system is stable in practice.

**Formal stability theorem (simplified):** For any arrival process with time-averaged rate λ̄, if (M - U) > k × σ where σ is the standard deviation of the arrival process over the measurement window and k is a confidence parameter, the system converges to merged state (λ̄ > decay_rate) or unmerged state (λ̄ < decay_rate) with probability exceeding 1 - 2e^(-k²/2).

### Migration Cost Model

**Decision function:** Co-migration (moving a request and its adapter from worker A to worker B) is beneficial when:

```
E[latency_reduction] > migration_cost
```

**Latency reduction estimate:**

If worker A's queue depth is Q_A and worker B's queue depth is Q_B, with Q_A >> Q_B, the request currently waiting in A would be served in approximately:

```
E[wait_A] = Q_A × avg_decode_step_time
```

After migration to B:

```
E[wait_B] = Q_B × avg_decode_step_time
```

Expected latency reduction:

```
E[reduction] = (Q_A - Q_B) × avg_decode_step_time
```

**Migration cost:**

Migration requires transferring the request's KV cache and the adapter weights from A to B. The cost is:

```
migration_cost = (kv_cache_size + adapter_size) / inter_worker_bandwidth
```

where `kv_cache_size = n_layers × n_kv_heads × head_dim × 2 × seq_len × dtype_bytes` and `adapter_size = 2 × n_layers × n_target_modules × (rank × d_in + d_out × rank) × dtype_bytes`.

**Migration decision rule:**

```
if (Q_A - Q_B) × avg_decode_step_time > migration_cost:
    migrate(request, kv_cache, adapter, from=A, to=B)
```

For LLaMA-7B with rank-8 adapter, 512-token KV cache, on 200 Gbps InfiniBand:
- KV cache size: 28 layers × 8 heads × 128 dim × 2 × 512 tokens × 2 bytes ≈ 57 MB
- Adapter size: 28 layers × 7 modules × 2 × (8 × 4096 + 4096 × 8) × 2 bytes ≈ 25 MB
- Migration cost: 82 MB / 25 GB/s ≈ 3.3 ms
- Break-even queue depth difference: 3.3 ms / 10 ms per step = 0.33 steps

Any queue depth difference greater than 1 step makes migration beneficial for this configuration. In practice, migration is triggered when the difference exceeds 5 steps to avoid thrashing from measurement noise.

**Thrashing prevention:** dLoRA includes a per-request migration cooldown: a request that was just migrated cannot be migrated again for at least `2 × migration_cost` time. This prevents a request from bouncing between workers as queue depths fluctuate.

---

## Omission 4: CaraServe — CPU Profiler Threshold Derivation

**Source:** `L4/02_caraserve_2024.md`
**Why omitted from COMBINEDL3L4.md:** COMBINEDL3L4 §9 states that CPU-assisted prefill is beneficial only when the adapter loading time exceeds the GPU prefill time, and that "CaraServe's profiler determines this threshold per adapter rank and prompt length." The derivation of this threshold is an important operational detail left to this file.

---

### The CPU-Assisted Prefill Condition

Let:
- `T_gpu_prefill(L)` = time for GPU to run prefill on a prompt of length L tokens (for the adapter being loaded)
- `T_cpu_prefill(L)` = time for CPU to run the same prefill on a prompt of length L tokens
- `T_load(r)` = time to transfer an adapter of rank r from CPU to GPU via PCIe

CPU-assisted prefill reduces TTFT when overlap is possible — specifically, when the CPU can complete prefill faster than the adapter loads, so that when the adapter is ready on the GPU, the KV state is also ready.

The overlap benefit exists when:

```
T_load(r) > T_gpu_prefill(L)
```

If T_load < T_gpu_prefill, the GPU will be idle waiting for the KV cache from the CPU — the overlap is wasted and sequential processing is faster.

The TTFT under each approach:
- **Sequential (no CPU assist):** TTFT = T_load(r) + T_gpu_prefill(L) + T_kv_transfer
- **CPU-assisted:** TTFT = max(T_load(r), T_cpu_prefill(L)) + T_kv_transfer

CPU-assisted is strictly beneficial when:

```
max(T_load, T_cpu_prefill) < T_load + T_gpu_prefill
⟺ max(T_load, T_cpu_prefill) < T_load + T_gpu_prefill
```

This simplifies to: CPU-assisted is beneficial **when T_load > T_gpu_prefill**, because in that case:

```
max(T_load, T_cpu_prefill) ≈ T_load  (assuming CPU prefill is fast enough)
sequential = T_load + T_gpu_prefill > T_load
```

CPU-assisted reduces TTFT by exactly `T_gpu_prefill`.

### Deriving the Threshold Prompt Length

CaraServe's profiler derives the **threshold prompt length** `L*` at which T_load(r) = T_gpu_prefill(L) for each adapter rank r.

**Empirical profiling step:**

For each rank r in {4, 8, 16, 32, 64}:
1. Measure T_load(r): transfer adapter from CPU to GPU 100 times, take median.
2. Sweep prompt lengths L ∈ {128, 256, 512, 1024, 2048, 4096}: measure T_gpu_prefill(L) 100 times each, take median.
3. Find L*(r) as the interpolated prompt length where T_gpu_prefill(L*(r)) = T_load(r).

**Typical values on LLaMA-7B, A100, PCIe 4.0 (32 GB/s):**

| Rank r | Adapter size | T_load | L*(r) (approx.) |
|---|---|---|---|
| 4 | ~12 MB | ~0.38 ms | ~150 tokens |
| 8 | ~25 MB | ~0.78 ms | ~300 tokens |
| 16 | ~50 MB | ~1.56 ms | ~600 tokens |
| 32 | ~100 MB | ~3.13 ms | ~1200 tokens |
| 64 | ~200 MB | ~6.25 ms | ~2400 tokens |

**Scheduling decision:**

At runtime, when a cold-start request arrives for adapter with rank r and prompt of length L:

```
if L > L*(r):
    use CPU-assisted prefill  # T_load > T_gpu_prefill
else:
    queue and wait for adapter  # T_gpu_prefill > T_load; CPU assist not worth overhead
```

### The KV State Transfer Overhead

CPU-assisted prefill introduces a hidden cost: the KV cache built by the CPU must be transferred to the GPU after the adapter loads. This KV transfer is on the critical path.

Let `T_kv_transfer(L)` = time to transfer L tokens of KV state from CPU pinned memory to GPU.

KV cache size: `L × n_layers × 2 × n_kv_heads × head_dim × 2 bytes`
For LLaMA-7B, this is: `L × 32 × 2 × 8 × 128 × 2 = L × 131,072 bytes = L × 128 KB`

At PCIe 4.0 (32 GB/s effective for pinned memory DMA): T_kv_transfer(L) = L × 128 KB / 32 GB/s = L × 4 µs.

Refined benefit condition:

```
CPU-assisted is beneficial when:
T_load(r) > T_gpu_prefill(L) + T_kv_transfer(L)
```

For rank-8 adapter (T_load = 0.78 ms) and LLaMA-7B:
- L = 300: T_gpu_prefill ≈ 0.78 ms, T_kv_transfer ≈ 1.2 ms → total ≈ 2 ms > T_load → not beneficial
- L = 600: T_gpu_prefill ≈ 1.56 ms, T_kv_transfer ≈ 2.4 ms → total ≈ 4 ms > T_load → not beneficial

The KV transfer overhead significantly narrows the window where CPU-assisted prefill is beneficial. In practice, CaraServe's profiler accounts for this by measuring the combined cost T_gpu_prefill + T_kv_transfer and finding L* where T_load equals this combined cost. For rank-8 adapters, this typically requires prompts of 1000+ tokens to be beneficial.

**Practical implication:** CPU-assisted prefill is most beneficial for high-rank adapters (large T_load) with long prompts (where T_gpu_prefill + T_kv_transfer grows large but T_load grows faster). For small rank adapters on short prompts, the sequential approach is faster.

---

## Omission 5: Predictive-LoRA — LSTM Training Loop and Online Update Algorithm

**Source:** `L4/05_predictive_lora_2025.md`
**Why omitted from COMBINEDL3L4.md:** COMBINEDL3L4 §15 describes the LSTM predictor at the level of "an LSTM trained on per-adapter request rates predicts future demand." The paper's training loop — how the LSTM is initialized, how it is updated from streaming traffic data, the feature normalization approach, and the sensitivity analysis for the prediction window length — are machine learning engineering details left to this file.

---

### Input Feature Construction

For each time window of duration Δt (typically 60 seconds), the predictor computes features for each adapter i:

**Per-adapter features (for a pool of N adapters):**

| Feature | Computation | Motivation |
|---|---|---|
| `r_i(t)` | requests for adapter i in window t | Direct demand signal |
| `c_i(t)` | cumulative requests for adapter i up to t | Long-range popularity signal |
| `delta_r_i(t)` | r_i(t) - r_i(t-1) | Rate of change |
| `ma_i(t, k)` | moving average of r_i over last k windows | Short-term trend |

**Global features (shared across all adapters):**

| Feature | Computation | Motivation |
|---|---|---|
| `hour_sin` | sin(2π × hour_of_day / 24) | Daily cycle encoding |
| `hour_cos` | cos(2π × hour_of_day / 24) | Daily cycle encoding |
| `day_sin` | sin(2π × day_of_week / 7) | Weekly cycle encoding |
| `day_cos` | cos(2π × day_of_week / 7) | Weekly cycle encoding |
| `total_rate(t)` | Σᵢ r_i(t) | System load |

**Input vector per time step:**

The LSTM input at step t is the concatenation of per-adapter features for all N adapters plus the global features. With N=100 adapters and 4 per-adapter features plus 5 global features, the input dimension is 405.

### LSTM Architecture

The predictor uses a **stateful LSTM** with one hidden layer:

```
Input:   x_t ∈ ℝ^(N×4 + 5)
LSTM:    h_t, c_t = LSTM_cell(x_t, h_{t-1}, c_{t-1})
         hidden_dim = 64  (intentionally small for fast CPU inference)
Output:  ŷ_t = Linear(h_t) ∈ ℝ^N   (predicted request rate per adapter)
         ŷ_t = ReLU(ŷ_t)           (rates must be non-negative)
```

**Why one hidden layer of 64:** The LSTM must run on CPU during serving. At 64 hidden units, a single inference step takes ~0.3 ms on modern CPUs. With 405 input features, the total parameter count is approximately 4 × 64 × (405 + 64) = 120,064 parameters — tiny, easily fits in L2 cache, runs in cache-resident mode.

### Offline Pre-Training

Before deployment, the LSTM is pre-trained on historical traffic data using standard backpropagation through time (BPTT):

**Loss function:**

```
L = (1/T) × Σ_t Σ_i [ŷ_i(t) - r_i(t+1)]²   (MSE on next-window demand)
```

**Normalization:** Each adapter's request rate is normalized by its historical mean and standard deviation before being fed to the LSTM. This prevents high-traffic adapters from dominating the loss function:

```
r̃_i(t) = (r_i(t) - μ_i) / (σ_i + ε)
```

where μ_i and σ_i are estimated from the pre-training data and ε = 1e-8 prevents division by zero.

**Training hyperparameters (from paper):**
- Sequence length (BPTT window): 24 steps (24 hours of hourly data)
- Learning rate: 1e-3 with Adam optimizer
- Batch size: 32 sequences (32 non-overlapping time windows)
- Epochs: 10 passes over training data
- Training data: 30 days of historical access logs

### Online Update Algorithm

After pre-training, the LSTM is continuously updated from streaming traffic data during serving. The update uses a simplified variant of BPTT with a short window, called the **online gradient step**:

**Every `update_interval` seconds (default: 300 seconds / 5 minutes):**

1. **Accumulate** the actual request rates for the last `update_interval` period, producing one new training example: (features at t, actual rates at t+1).

2. **Run forward pass** through the LSTM with the current hidden state to produce prediction ŷ(t).

3. **Compute loss:** L = Σᵢ (ŷᵢ(t) - r_i(t+1))²

4. **Backpropagate** through the last `online_window` = 4 steps (not the full 24-step window, for speed).

5. **Update weights:** θ ← θ - η_online × ∇L, where η_online = 1e-4 (10× smaller than offline LR to prevent forgetting).

6. **Update normalization statistics** incrementally: μ_i ← 0.99 × μ_i + 0.01 × r_i(t), σ_i updated similarly. This is exponential moving average normalization that adapts to long-term distribution shifts.

### Prediction and Prefetch Decision

Every `predict_interval` seconds (default: 60 seconds), the predictor runs:

1. Run LSTM forward pass on current features → ŷ_i(t+1) for each adapter i.

2. Denormalize: r̂_i(t+1) = ŷ_i(t+1) × σ_i + μ_i.

3. **Prefetch decision:** if r̂_i(t+1) > prefetch_threshold and adapter i is not in GPU pool:
   - Schedule background load of adapter i from CPU to GPU.
   - Priority: adapters sorted by (r̂_i - prefetch_threshold) descending.

4. **Eviction decision:** if GPU pool is near full, evict adapters with r̂_i(t+1) < keep_threshold (predicted to be cold in the next window).

### Hyperparameter Sensitivity Analysis

The paper includes a sensitivity study over the key hyperparameters:

| Hyperparameter | Range tested | Optimal | Sensitivity |
|---|---|---|---|
| Prediction window Δt | 30s, 60s, 120s, 300s | 60s | Low (±5% throughput) |
| LSTM hidden dim | 16, 32, 64, 128 | 64 | Medium (±12% throughput) |
| Online update interval | 60s, 300s, 600s, 1800s | 300s | Low (±8% throughput) |
| Prefetch threshold | 0.5×, 1×, 2× mean rate | 1× mean rate | High (±20% throughput) |

The prefetch threshold is the most important hyperparameter: too low causes excessive pre-loading (wastes VRAM on unused adapters), too high causes cold starts on adapters the predictor correctly identified as likely-needed. The paper recommends setting it to the historical mean request rate per adapter as a default, with tuning based on the VRAM budget available for pre-loaded adapters.

---

## Omission 6: InfiniLoRA — SLO Provisioning Solver

**Source:** `L4/06_infinilora_2026.md`
**Why omitted from COMBINEDL3L4.md:** COMBINEDL3L4 §17 describes the capacity planner conceptually as "an offline planner that outputs how many LoRA Server GPUs to provision, how to distribute adapters, and when to scale." The paper provides the optimization problem formulation and the solver. This is relevant for operators sizing an InfiniLoRA deployment but not for understanding its architecture.

---

### The Provisioning Problem

Given a set of A adapters with predicted request rates λ₁, λ₂, ..., λ_A, SLO requirements (P99 TTFT ≤ TTFT_SLO for each adapter), and a set of hardware options with costs c_gpu per GPU-hour, the provisioning problem is:

**Minimize:** Total LoRA Server GPU cost

**Subject to:**
1. For each adapter i: P99 TTFT_i ≤ TTFT_SLO_i
2. Total adapters assigned to each LoRA Server GPU ≤ capacity_per_gpu
3. For each LoRA Server GPU: memory used by assigned adapters ≤ GPU_VRAM

### LoRA Server Capacity Model

Each LoRA Server GPU can serve multiple adapters. Its throughput is bounded by two resources:

**Compute bound:** The LoRA delta computation (A and B matrix multiplies) requires:

```
FLOP_per_token = 2 × n_layers × n_modules × 2 × rank × d_model
```

At a LoRA Server GPU FLOP capacity of F (FLOP/s), the maximum token rate is:

```
max_token_rate_compute = F / FLOP_per_token
```

For LLaMA-7B, rank-8, all 7 modules: FLOP_per_token = 2 × 32 × 7 × 2 × 8 × 4096 ≈ 29M FLOP.
On A100 (312 TFLOP/s FP16): max_token_rate_compute ≈ 10.7 million tokens/second.

**Memory bound:** Loading the A and B matrices for each token requires:

```
bytes_per_token = 2 × n_layers × n_modules × (rank × d_model + d_model × rank) × dtype_bytes
```

For LLaMA-7B, rank-8: bytes_per_token = 2 × 32 × 7 × 2 × 8 × 4096 × 2 bytes ≈ 57.5 MB/token.
At A100 HBM bandwidth (2 TB/s): max_token_rate_memory ≈ 34.8k tokens/second.

The memory bandwidth is the binding constraint: LoRA Server GPUs are memory-bandwidth-bound, not compute-bound. This motivates the hardware-specialized kernel in InfiniLoRA (§17 of COMBINEDL3L4) that optimizes HBM access patterns.

The effective capacity of one LoRA Server GPU: approximately 30,000–40,000 tokens/second.

### Per-Adapter Latency Model

The TTFT for a request using adapter i, in InfiniLoRA's disaggregated architecture, is:

```
TTFT_i = T_prefill_base + T_lora_delta + T_activation_transfer + T_delta_return
```

Where:
- `T_prefill_base` = time for base model GPU to process input tokens
- `T_lora_delta` = time for LoRA Server to compute delta for adapter i (includes queuing delay if LoRA Server is loaded)
- `T_activation_transfer` = time to send activations from base model GPU to LoRA Server (GPU-initiated RDMA)
- `T_delta_return` = time to send delta back from LoRA Server to base model GPU

The dominant term under load is `T_lora_delta`, which includes queuing time at the LoRA Server.

Using an M/D/1 queuing model for the LoRA Server:

```
E[T_lora_delta] = T_service + (ρ / (2 × (1 - ρ))) × T_service
```

where ρ = λ_i / μ_server (adapter request rate / server service rate) and T_service = FLOP_per_token / GPU_FLOP_rate.

The P99 latency (needed to check the SLO constraint) is approximately:

```
P99[T_lora_delta] ≈ E[T_lora_delta] + 3 × σ[T_lora_delta]
                 ≈ E[T_lora_delta] × (1 + 3 × √(ρ / (2 × (1-ρ))))
```

### The Optimization Formulation

**Decision variables:**
- `n_lora_gpus`: number of LoRA Server GPUs to provision
- `x_{i,g}` ∈ {0, 1}: adapter i assigned to LoRA Server GPU g

**Objective:**

```
minimize n_lora_gpus × c_gpu
```

**Constraints:**

1. **SLO constraint** for each adapter i:
   ```
   P99[TTFT_i] ≤ TTFT_SLO_i
   ```
   Substituting the latency model, this becomes a constraint on the load ρ_g on GPU g:
   ```
   ρ_g = (Σᵢ: x_{i,g}=1 λ_i) / μ_server ≤ ρ_max(TTFT_SLO)
   ```
   where ρ_max is derived by inverting the P99 formula for the tightest TTFT_SLO among adapters assigned to GPU g.

2. **Memory constraint** for each GPU g:
   ```
   Σᵢ: x_{i,g}=1 adapter_size_i ≤ GPU_VRAM - system_overhead
   ```

3. **Assignment constraint**: each adapter is assigned to exactly one GPU:
   ```
   Σ_g x_{i,g} = 1  for all i
   ```

**Solver approach:**

This is a bin-packing problem (NP-hard in general). InfiniLoRA uses a two-phase heuristic:

**Phase 1 — Estimate n_lora_gpus:**
Lower bound: n_lora_gpus ≥ ceil(Σᵢ λ_i / (μ_server × ρ_max)), the minimum GPUs to handle total throughput.
Also: n_lora_gpus ≥ ceil(Σᵢ adapter_size_i / GPU_VRAM), the minimum to fit all adapters in memory.
Take the maximum of these two lower bounds.

**Phase 2 — Assign adapters to GPUs (First Fit Decreasing):**
1. Sort adapters by (λ_i / μ_server) descending (most loaded adapters first).
2. For each adapter, assign to the GPU with the lowest current ρ that still satisfies the memory constraint.
3. If no GPU satisfies constraints: increment n_lora_gpus and add a new empty GPU.

This greedy heuristic typically produces solutions within 5–10% of optimal (measured by comparing to LP relaxation) and runs in O(A log A) time — fast enough to re-run on traffic distribution shifts.

### Elastic Scaling Rule

At runtime, InfiniLoRA's planner re-runs the provisioning solver periodically using recent traffic measurements as the λ estimates. A scale-up event occurs when the solver returns a higher n_lora_gpus than the current deployment; a scale-down event occurs when the solver returns a lower value for more than `cooldown_window` consecutive measurement periods (to prevent oscillation).

---

## Omission 7: LoRA Variant Mathematics — DoRA, rsLoRA, PiSSA, EVA

**Source:** `L3/01_lora_original.md`, `L2/03_hf_peft_lora.md`
**Why omitted from COMBINEDL3L4.md:** COMBINEDL3L4 §12 mentions that PEFT supports variants like `use_rslora=True` and `use_dora=True`. The mathematical derivation of each variant — why they were designed, what problem the original LoRA scaling had, and what the formal modification is — requires engaging with the LoRA math at depth. These derivations are separate from the serving systems focus of COMBINEDL3L4.

---

### rsLoRA: Rank-Stabilized Scaling

**Problem with original LoRA scaling (alpha/r):**

In the original LoRA paper, the scaling factor is alpha/r. At initialization, the adapter output is:

```
Δh = (x @ A.T) @ B.T * (alpha / r)
```

Since A is Kaiming-initialized and B is zero, Δh = 0 at init (correct). But during the first gradient step, the magnitude of ∂L/∂A depends on (alpha/r). For high ranks, (alpha/r) is small (e.g., alpha=8, r=64 gives 0.125), causing small gradients and slow initial learning. For low ranks (e.g., r=4), (alpha/r) = 2, causing larger initial gradient steps.

**rsLoRA's modification:**

Replace (alpha/r) with (alpha/√r):

```
Δh = (x @ A.T) @ B.T * (alpha / sqrt(r))
```

**Motivation:** For a Kaiming-initialized A with fan-in = r, the expected L2 norm of each row of A is approximately proportional to 1/√r (this is the Kaiming initialization guarantee). The resulting expected magnitude of (x @ A.T) is proportional to 1/√r as well. Multiplying by (alpha/√r) produces a total magnitude proportional to (alpha/r), which is constant across ranks when alpha = 1.

With the original (alpha/r) scaling, increasing r from 8 to 64 reduces the adapter contribution by 8× at initialization. With (alpha/√r), increasing r from 8 to 64 reduces the contribution by only 2√2 ≈ 2.8×, making high-rank training more stable.

**Practical effect (from PEFT benchmarks):** rsLoRA converges in fewer training steps at high ranks (r=64, r=128) and achieves higher peak quality at the same parameter budget.

**Usage in PEFT:**
```python
LoraConfig(r=64, lora_alpha=16, use_rslora=True)
# Effective scaling: 16 / sqrt(64) = 2.0
```

---

### DoRA: Weight Decomposition Adaptation

**Motivation:** LoRA modifies the weight matrix by adding a low-rank correction: W = W₀ + BA. This constrains the correction to a low-dimensional subspace. But the optimal correction might require changing both the *magnitude* and *direction* of W. A low-rank matrix can change both, but the coupling between magnitude and direction may limit what a low-rank ΔW can express.

**DoRA's insight:** Decompose the weight matrix into magnitude and direction components:

```
W = m × (V / ||V||_c)
```

where:
- `m ∈ ℝ^{1×k}` is the column-wise magnitude (norm of each column of W)
- `V ∈ ℝ^{d×k}` is the directional component (W normalized column-wise)
- `||·||_c` denotes column-wise L2 normalization

**DoRA's training:** Freeze V at initialization but train m (a very small number of parameters) and a LoRA correction ΔV = BA applied to V:

```
W' = m' × ((V + BA) / ||V + BA||_c)
```

where m' = m + Δm is the trained magnitude vector and BA is the low-rank directional correction.

**Why this helps:** By separating magnitude from direction, DoRA allows the training signal to adjust the scale of each output direction independently of its angular correction. This is analogous to how normalization layers work in deep networks — decoupling scale from direction reduces the condition number of the optimization problem.

**PEFT results:** DoRA consistently outperforms standard LoRA at the same rank on instruction-following benchmarks (MMLU, MT-Bench) by 1–2 percentage points.

**Usage in PEFT:**
```python
LoraConfig(r=8, lora_alpha=16, use_dora=True)
```

---

### PiSSA: Principal Singular Value Adaptation

**Motivation:** LoRA initializes A with Kaiming-uniform and B with zeros, ensuring the adapter starts as an identity (no perturbation). But this means the adapter must learn the low-rank correction from a random starting point. If the model's weight matrix is already decomposable into principal singular components (which it is — all matrices have an SVD), the adapter could start aligned with the most informative directions.

**PiSSA's modification:** Initialize A and B from the principal singular value decomposition of W₀:

1. Compute SVD of W₀: W₀ = U Σ Vᵀ
2. Set A = √Σ_r × Vᵀ_r (the top-r right singular vectors scaled by their singular values)
3. Set B = U_r × √Σ_r (the top-r left singular vectors scaled by their singular values)
4. Subtract BA from W₀ to get the "residual" weight: W_residual = W₀ - BA
5. Freeze W_residual; train A and B

After this initialization: BA = W₀_r (the rank-r approximation of W₀). The adapter starts by representing the most important components of the weight matrix, rather than zero.

**Intuition:** The low-rank correction starts in the highest-energy subspace of W₀. If fine-tuning primarily adjusts the principal directions of the weight matrix (as the rank-deficiency hypothesis suggests), PiSSA starts in exactly the right subspace and converges faster.

**PEFT results:** PiSSA typically converges in 30–50% fewer training steps than random LoRA initialization at the same rank, with comparable or better final quality.

**Usage in PEFT:**
```python
LoraConfig(r=8, lora_alpha=8, init_lora_weights="pissa")
```

---

### EVA: Explained Variance Adaptation

**Motivation:** PiSSA uses the SVD of the pre-trained weight matrix to initialize the adapter. But the question of which subspace is important for fine-tuning is determined not by the weight matrix alone, but by how the pre-trained model processes the fine-tuning data. EVA uses the activation distribution on the fine-tuning dataset to initialize the adapter in the subspace that is actually exercised during fine-tuning.

**EVA's approach:**

1. Forward-pass a sample of fine-tuning data through the frozen pre-trained model.
2. For each target layer, collect the activations: X ∈ ℝ^{N×d} where N = num_tokens_seen.
3. Compute the covariance of the activations: C = XᵀX / N ∈ ℝ^{d×d}.
4. Compute the top-r eigenvectors of C: these are the directions of maximum activation variance.
5. Initialize the A matrix of the LoRA adapter using these eigenvectors.
6. Initialize B to zero (standard LoRA).

**Intuition:** The fine-tuning update ΔW is applied to activations X, producing output changes XΔWᵀ. If ΔW is constrained to a rank-r subspace, the most informative choice is the subspace where X has the most variance — because this maximizes the output change for a given ΔW norm. EVA initializes A in exactly this subspace.

**Practical difference from PiSSA:** PiSSA uses the weight matrix's SVD; EVA uses the data's activation covariance. These differ because high-variance activation directions are not necessarily the same as the principal weight directions. For domain-specific fine-tuning (e.g., medical, legal, code), the activation distribution can differ substantially from what the pre-trained weights suggest.

**PEFT results:** EVA consistently outperforms both PiSSA and standard LoRA on domain-specific fine-tuning tasks where the target domain is far from the pre-training data distribution.

**Usage in PEFT:**
```python
# EVA requires a calibration forward pass; usage is more involved:
from peft import get_peft_model, LoraConfig
config = LoraConfig(r=8, lora_alpha=8, init_lora_weights="eva")
model = get_peft_model(base_model, config)
model.initialize_adapter_weights(calibration_dataloader)  # EVA-specific step
```

---

### Summary: LoRA Variant Comparison

| Variant | Key Modification | Best Use Case | PEFT Flag |
|---|---|---|---|
| LoRA (original) | Random init A, zero B | General baseline | default |
| rsLoRA | Scale = alpha/√r instead of alpha/r | High-rank adapters (r≥32) | `use_rslora=True` |
| DoRA | Decompose W into magnitude + direction | Instruction following, preference optimization | `use_dora=True` |
| PiSSA | Init A, B from SVD of W₀ | Fast convergence, limited training budget | `init_lora_weights="pissa"` |
| EVA | Init A from data activation covariance | Domain-specific fine-tuning on distant domains | `init_lora_weights="eva"` |

For Layer 20's minimal implementation using `phh/Qwen3-0.6B-TLDR-Lora`, the adapter was trained with standard LoRA (alpha=32, r=8, default initialization). The variant mathematics above are relevant when training adapters for deployment in production systems, not when consuming pre-trained adapters.
