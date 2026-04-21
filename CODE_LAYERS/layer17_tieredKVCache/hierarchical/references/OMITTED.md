# Tiered KV Cache — Omitted Material

**What this file is:** The full text of every section omitted from `COMBINED.md`. The appendix of `COMBINED.md` names each omission and explains why it was excluded. This file preserves the complete original text so no source material is lost.

**Sources:** L2/01, L2/02, L3/01, L3/02, L3/03, L3/04, L4/01, L4/02, L4/03

---

## Omission 1: CachedAttention — KV Cache Invalidation Avoidance (Positional Encoding Decoupling)

**Source:** `L3/02_cachedattention_atc24.md`
**Why omitted from COMBINED.md:** HiCache does not implement this technique. It is a CachedAttention-specific design that enables cached entries to remain valid after context truncation. Including it in COMBINED.md would imply HiCache has a feature it does not have. The appendix notes it as a known HiCache limitation.

### Original text: Technique 4 — KV Cache Invalidation Avoidance

Standard transformers embed positional information directly into Q/K/V projections. When conversation history grows past the context window, cached KV entries become **invalid** because positions shift.

CachedAttention **decouples positional encoding** from the saved KV caches, allowing cached entries to remain valid even after truncation. The key idea: store KV caches without the rotary positional embedding applied; apply the positional encoding at attention time using the stored token positions, not the cache positions. This means the cached KV tensor is position-agnostic and can be inserted at any offset in a future context without invalidation.

> Note: HiCache does not implement this optimization — it relies on token-exact prefix matching. This is a known limitation for sessions that exceed the context window.

### Key Differences from HiCache — Full Table

| Aspect | CachedAttention | SGLang HiCache |
|---|---|---|
| Scope | Multi-turn conversations only | Any prefix reuse (system prompts, RAG, coding agents) |
| Cache key | Session ID + turn number | RadixTree token prefix |
| Positional encoding | Decoupled (allows truncation) | Coupled to token prefix (token-exact match) |
| Storage backends | Generic (SSD assumed) | Pluggable: file, Mooncake, 3FS, NIXL, AIBrix, dynamic |
| Write policies | Not configurable | `write_back`, `write_through`, `write_through_selective` |
| Integration | Research prototype | Production system in SGLang |
| Tensor parallelism | Not addressed | `all_reduce` synchronization for TP |

---

## Omission 2: InfiniGen — Rehearsal Computation (Speculative QK Approximation)

**Source:** `L4/01_infinigen_osdi24.md`
**Why omitted from COMBINED.md:** HiCache uses access-frequency hit-count tracking (`_inc_hit_count()`) as a proxy for importance, not speculation-based next-layer attention scoring. InfiniGen's rehearsal mechanism is the key algorithmic contribution of the paper but has no direct counterpart in HiCache. COMBINED.md covers the insight (selective loading) and the connection (what HiCache approximates instead) without the mechanism details.

### Original text: Rehearsal Computation

**Key insight**: the set of important tokens for layer `n+1` can be **speculated** using:
1. The input activations of the current layer `n`
2. A partial (cheap) rehearsal using **only part of the Q weight and K cache of layer n+1**

This speculation is:
- **Cheap to compute** — uses a small subset of the attention weight matrix
- **Accurate enough** — identifies the top-k important tokens with high recall
- **Just-in-time** — runs during layer n computation, before layer n+1 needs its KV

### System Architecture Diagram

```
GPU: Layer N computing ──────────────────────┐
CPU: Speculate important tokens for Layer N+1 │
     Prefetch only important KV pages ────────┘
     (discard unimportant pages)
```

### Prefetch granularity

- Standard systems: prefetch **all** KV pages for the upcoming layer
- InfiniGen: prefetch **only the top-k speculated important** KV pages

This reduces the amount of data transferred over PCIe by **50–80%**, directly reducing the memory-bandwidth bottleneck.

### "Rehearsal" computation detail

Using the current layer's input and a slice of the next layer's Q/K projection weights, InfiniGen runs a small matrix multiplication to get approximate attention scores. The top-k indices by score identify which KV pages to prefetch. This is effectively running a low-rank approximation of the next layer's attention before that layer is reached.

### Why PCIe Is the Bottleneck (InfiniGen-measured numbers)

InfiniGen benchmarks reveal that in CPU-offloaded inference:
- With naïve full-KV loading: **~75–90% of decode step time** is spent on PCIe transfers
- With InfiniGen selective loading: **~30–50% of decode step time** on PCIe (remainder is GPU compute)

The optimal working point is to load exactly the pages needed for accurate attention — not more, not less.

This is why the κ_crit analysis (see L4/03) is important: there is a critical ratio where you transition from compute-bound to memory-bound. InfiniGen tries to keep the effective ratio below κ_crit by being selective.

---

## Omission 3: IMPRESS — KV Reordering on Disk and Four Component Details

**Source:** `L4/02_impress_fast25.md`
**Why omitted from COMBINED.md:** IMPRESS's physical disk reordering technique is not implemented in HiCache. HiCache's `page_first` layout is the lossless analog (contiguous within prefix, not sorted by importance across prefixes). The four components together are more granular than needed in the synthesis narrative; COMBINED.md covers the insight (importance-informed I/O, contiguous layout) without the implementation details.

### Original text: Four Key Components

#### 1. Importance-informed KV Identification

- Cross-head attention similarity: importance scores are consistent across heads
- Run a lightweight approximation of attention to get per-token importance scores
- Cost: significantly lower than full attention computation

#### 2. KV Reordering on Disk

- After computing importance scores, **physically reorder** KV pages on disk by importance
- Important pages are stored contiguously → can be loaded with a single large sequential I/O
- Unimportant pages stored separately → can be skipped entirely

#### 3. Importance-informed Cache Management

- Between GPU, CPU, and disk tiers: evict unimportant pages first
- Retain important pages in faster tiers longer (importance-weighted LRU)
- Prefetch only important pages from disk to CPU when a prefix cache hit is detected

#### 4. Caching Accuracy Guarantee

- By selecting only the top-k% of KV pages by importance, IMPRESS maintains **comparable inference accuracy** to loading all KV pages
- The importance threshold is configurable; higher = less I/O but slightly lower accuracy

### Why FAST Matters (Storage Community Signal)

FAST is the premiere storage systems conference. IMPRESS appearing there signals that:
- The storage systems community recognizes LLM KV cache management as a **storage problem**, not just a ML problem
- Future hardware-storage co-design (NVMe-oF, GPU Direct Storage, CXL memory) is expected to address KV cache I/O bottlenecks
- Storage-tier optimization is as important as GPU-tier optimization for long-context inference

This aligns with HiCache's pluggable backend design: as new storage technologies emerge (CXL, GDS, persistent memory), new backends can be added without changing the HiCache architecture.

---

## Omission 4: vLLM Offloading — Version History and Connector API Design

**Source:** `L2/02_vllm_kv_offloading_blog.md`
**Why omitted from COMBINED.md:** Version history (v0.11.0 → v0.12.0 → v0.14.0) is vLLM-specific engineering detail; not applicable to SGLang/HiCache design. The connector API design is the vLLM architectural choice that contrasts with HiCache's embedded design — COMBINED.md notes the embedded vs connector distinction without the API specifics.

### Original text: Connector API Design

The connector API is called **before** each request (load from external source) and **after** computation (store KV data). As of v0.9.0, it supports **asynchronous** load/store, enabling offloading to run in parallel with model computation.

```bash
# vLLM 0.14.0+
vllm serve <model> --kv_offloading_backend native --kv_offloading_size <size_in_GB>

# Older releases
vllm serve <model> --kv-transfer-config \
  '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks": <num_cpu_blocks>}}'
```

### Version Improvements Table

- **v0.12.0**: Up to 4× TTFT reduction, 5× throughput increase vs v0.11.0 (due to physical block layout change)
- **v0.14.0** (upcoming at time of writing): preempted requests can load back from CPU; race condition fix

### Physical Block Sizes for Common Models (16-token blocks)

| Model | Old size (v0.11.0) | New size (v0.12.0+) |
|---|---|---|
| Llama-3.1-8B-Instruct | 32 KB | 2 MB |
| Llama-3.1-70B-Instruct | 8 KB | 1.25 MB |
| DeepSeek-R1-Distill-Qwen-32B (TP=2) | 16 KB | 2 MB |
| Mistral-7B-Instruct-v0.2 | 32 KB | 2 MB |
| Qwen2.5-7B-Instruct | 16 KB | 0.87 MB |

The new layout increased physical block size by `2 × num_layers`, putting all models in the DMA-efficient range. This single layout change — making each logical KV block physically contiguous across all layers — produced the 4–5× improvement without any algorithmic change.

### Single-request TTFT Results (Llama-3.1-8B, H100)

Loading KV from CPU reduces TTFT by **2×–22×** depending on prompt length (shorter prompts see less benefit; longer prompts see more because more computation is avoided).

### End-to-end DMA vs Kernel Numbers

- DMA achieves **5–32% better throughput** than the custom kernel in concurrent workloads
- For cache misses: the custom kernel **reduces throughput by 6%** vs baseline (no offloading) — interference with GPU computation costs more than is saved

---

## Omission 5: LMCache — Enterprise Production Insights and Core Contributions Detail

**Source:** `L3/03_lmcache_arxiv25.md`
**Why omitted from COMBINED.md:** The enterprise insights are specific to LMCache's middleware architecture position and not directly applicable to HiCache deployments. The three core contributions are captured at a high level in COMBINED.md; the detail below is more granular than needed for the synthesis.

### Original text: Enterprise Insights (Production Data)

Large-scale deployment in enterprise settings revealed three production insights:

1. **Fetching KV from remote storage measurably benefits prefill delay** (confirmed at scale) — cross-engine KV sharing reduces effective TTFT even when the KV was computed on a different engine instance.

2. **Context truncation halves prefix cache hit ratio** — a widely-applied industrial technique that discards prefix cache because the token sequence changes when context is truncated. When an application truncates long conversations to fit the context window, the token sequence changes, invalidating cached prefixes. LMCache found this is the single biggest real-world enemy of prefix cache hit rates in enterprise deployments.

3. **Cross-engine sharing is the key differentiator** — sharing KV across multiple inference instances on different GPUs dramatically raises effective cache hit rate. A prefix computed by instance A can be served by instance B without recomputation.

> "Over time, total KV cache stored by users grows far beyond GPU memory capacity." — LMCache paper

### Three Core Contributions — Detail

#### 1. Highly Optimized KV Cache Data Movement

- **Batched data movement operations** — process multiple KV pages together to amortize per-operation overhead
- **Compute and I/O pipelining** — overlap GPU computation with KV transfers using separate CUDA streams
- **Layer-wise streaming** — dedicated CUDA streams per layer, avoiding head-of-line (HOL) blocking: if layer N's KV is slow to arrive from storage, layer N+1's transfer can proceed independently

#### 2. Modular KV Cache Connector

- Decouples LMCache from the rapid evolution of inference engines (vLLM, SGLang change APIs frequently)
- The connector component implements the engine-side API (vLLM Connector API, SGLang plugin API)
- New engine versions require only connector updates, not LMCache core changes
- This is exactly the architectural tradeoff vs HiCache: LMCache can follow engine API changes without modifying its core; HiCache is tightly coupled but gets tighter optimization

#### 3. First-class Control API

- Flexible cache orchestration across GPU, CPU, storage, and network
- APIs for: capacity management, eviction policy control, cross-engine sharing, prefix invalidation
- Unlike HiCache's CLI flags, LMCache exposes programmatic control for applications that need to manage cache state explicitly (e.g., invalidate a user's session cache on logout)

### LMCache Supported Backends (Full List)

CPU DRAM, local disk (NVMe), Redis (remote in-memory store), S3-compatible object storage, remote network storage.

Note: Redis and S3 are native LMCache backends but are not available in HiCache. HiCache's storage backends (Mooncake, 3FS, NIXL) are higher-performance RDMA/distributed-filesystem approaches but not available in LMCache.

---

## Omission 6: KV Cache Survey — Token-Level Methods in Detail

**Source:** `L3/04_kv_cache_survey_tmlr25.md`
**Why omitted from COMBINED.md:** Token-level methods are orthogonal to HiCache (they change what is stored; HiCache changes where). COMBINED.md notes their existence and that they are composable with HiCache. The detail below belongs to Layer 18 (KV Cache Quantization) and related topics, not to the tiered storage narrative.

### Original text: Token-Level Optimizations Detail

#### KV Cache Selection

Drop less important tokens from the cache entirely:

- **H2O (Heavy Hitter Oracle)**: keep the top-k most recently/frequently attended KV entries; evict the rest. The "heavy hitters" are identified by running attention and tracking which keys receive the highest attention weights.
- **StreamingLLM**: keep attention sinks (first few tokens — these almost always receive high attention regardless of content) plus a recent sliding window. Enables infinite-length generation with a fixed VRAM budget.
- **SnapKV**: identifies which KV positions are important based on the observation pattern of attention heads during prefill, then compresses the KV cache by retaining only the observed important positions.

#### Budget Allocation

Adaptive per-layer KV budget:

- **PyramidKV**: allocate more KV budget to lower layers (which have more uniform attention patterns and benefit from more context) and less to higher layers (which have more concentrated attention). Empirically, this outperforms uniform per-layer budgets at the same total memory.

#### KV Merging

Merge similar KV entries to reduce count:

- **KVMerger**: identifies KV entries with similar key vectors (cosine similarity above threshold) and merges them into a single representative entry. The attention computation uses the merged entry as if it were both, weighted by how many original entries it represents.

#### KV Quantization

Lower precision for cached entries — see Layer 18:

- **KIVI**: 2-bit asymmetric quantization with per-channel scale for K and per-token scale for V; ~8× memory reduction
- **KVQuant**: sub-4-bit with pre-RoPE quantization of K; enables 1M-token contexts on A100-80GB

### Quantization in Context (from Survey)

The survey covers KV quantization data points:
- KV quantization typically uses INT8 or INT4 (2–4× compression)
- Small but measurable accuracy degradation on long-context tasks
- Proportionally reduced PCIe transfer time — this directly improves κ_crit for the PCIe bottleneck

HiCache stores **FP16/BF16** (full precision). If combined with KV quantization, HiCache's host pool could store 2–4× more tokens for the same memory budget. KV quantization effectively multiplies the capacity of every tier.

---

## Omission 7: κ_crit Paper — Hardware Interconnect Proposals and Request Batching Detail

**Source:** `L4/03_kvcrit_pcie_bottleneck.md`
**Why omitted from COMBINED.md:** COMBINED.md mentions CXL and NVLink-C2C briefly in the context of monitoring. The full hardware proposal section and the request batching scheduling algorithm are more forward-looking than the HiCache production focus. The Prometheus monitoring PromQL was included in COMBINED.md's §4.

### Original text: Proposed Optimizations

#### 1. Hardware Interconnects

- **CXL memory** (DDR5 over CXL): memory-semantics access to pooled DRAM; avoids PCIe protocol overhead; allows the CPU DRAM pool to be accessed at near-DRAM latency rather than PCIe latency; could raise effective L2→L1 bandwidth by 5–10×
- **NVLink-C2C** (Grace-Hopper Superchip): 900 GB/s CPU↔GPU bandwidth vs 64 GB/s PCIe — a 14× improvement; makes the κ_crit threshold ~14× higher, meaning far more cached tokens can be loaded before the system becomes memory-bound
- **HBM expansion**: stacked memory on the GPU package with no PCIe path at all; effectively unlimited bandwidth at the cost of capacity

#### 2. Model Architecture

- **Grouped Query Attention (GQA)**: reduces KV size by num_heads/num_kv_heads — directly reduces PCIe load for each cached token. Llama-3.1's 8 KV heads (vs 64 query heads) already implement this; a non-GQA model with 64 KV heads has 8× more KV data per token
- **MLA (Multi-head Latent Attention)**: DeepSeek-V2/R1's extreme KV compression via low-rank projection; κ_crit effectively raised by 10–20× — the KV tensors are so much smaller that PCIe stops being the bottleneck at typical workload ratios
- **Hybrid sparse attention**: some layers use full attention KV, others use sliding window (only recent tokens in KV); reduces average KV size per token across the model

#### 3. Scheduling Algorithms

- **Selective prefetch**: only load KV for tokens with high attention importance — cf. InfiniGen, IMPRESS. Reduces PCIe data volume by 50–80% at the cost of some accuracy
- **Request batching**: batch multiple requests that share the same cached prefix → amortize PCIe cost across all requests in the batch. If 20 requests all need the same 10K-token system prompt's KV, loading it once and sharing it costs 1/20th per request
- **Pipelining**: overlap KV loading for the current batch with computation for the previous batch; hides the PCIe latency behind compute time for adjacent batches

### Transfer Efficiency Details

- Theoretical PCIe 5.0 ×16 peak: 64 GB/s
- Sustained measured rate (pinned memory, large blocks): ~48–55 GB/s
- For small blocks (< 512 KB per transfer): drops to 10–20 GB/s due to overhead
- Bi-directional simultaneous transfers: effective rate in each direction ~70% of one-direction peak due to PCIe arbitration

### Implications Table — Full Version

| Scenario | κ ratio | Memory-bound? | HiCache L2 helps? | Notes |
|---|---|---|---|---|
| 1K cached, 1K new | 1 | No | Marginal | κ < κ_crit for all models |
| 10K cached, 100 new | 100 | Yes | Yes | Standard RAG workload |
| 100K cached, 10 new | 10,000 | Severely | Yes, PCIe is new bottleneck | Long-doc analysis |
| GQA model, same scenario | ~12 | Marginal | More helpful | GQA reduces KV per token |
| MLA model (DeepSeek), same | ~0.5 | No | Very helpful | MLA raises κ_crit dramatically |

---

## Omission 8: SGLang HiCache Blog — Benchmark Runner Commands and Full Takeaways

**Source:** `L2/01_sglang_hicache_blog.md`
**Why omitted from COMBINED.md:** The benchmark runner commands are useful for reproducing results but not for understanding the system. The key takeaways section restates concepts already present in the main body. Both are preserved here for completeness.

### Original text: Benchmark Instructions

```bash
# Long-context benchmark
python3 benchmark/hicache/bench_long_context.py \
  --model-path /DeepSeek-R1/ \
  --dataset-path loogle_wiki_qa.json

# Multi-turn benchmark
python3 benchmark/hicache/bench_multiturn.py \
  --model-path $MODEL_PATH --disable-random-sample \
  --output-length 1 --request-length 2048 \
  --num-clients 80 --num-rounds 10 \
  --max-parallel 4 --request-rate 16
```

Full benchmark scripts: https://github.com/sgl-project/sglang/tree/main/benchmark/hicache

### Key Takeaways (from blog post)

- The **capacity bottleneck** of GPU-only RadixCache is the core motivation for HiCache.
- The **HiRadixTree** is the metadata layer — it tracks token-prefix to tier-location mappings.
- **GPU-assisted I/O kernels** and **page-first layout** are the two main data-plane optimizations.
- **Three prefetch policies** offer different TTFT/throughput tradeoffs.
- Real deployments confirm 2–6× throughput gains and 56–84% TTFT reductions.

---

## Omission 9: HiCache Design Doc — PD-Disaggregation Integration and LMCache Flag

**Source:** `L3/01_sglang_hicache_design.md`
**Why omitted from COMBINED.md:** PD-disaggregation (prefill-decode separation) is Layer 19's topic. COMBINED.md covers HiCache's operation within a single inference instance. The LMCache flag is mentioned in §8 (Alternative: LMCache) as a single flag reference; the full context from the design doc is preserved here.

### Original text: Integration with PD-Disaggregation

HiCache can be enabled on both **prefill nodes** and **decode nodes** in PD disaggregation deployments:

- **Prefill nodes**: HiCache accelerates KV lookups for shared prefixes. When a long system prompt was prefilled by a previous request, the prefill node can retrieve it from its HiCache L2 or L3 instead of recomputing — even though the decode was served by a different node.
- **Decode nodes**: decode output is written back to L3 if HiCache is enabled. This allows the accumulated KV from a long decode session to persist and be loaded back if the session resumes.

The Mooncake storage backend and the Mooncake TransferEngine (used for PD KV transfer between prefill and decode nodes) are **separate subsystems** — they coexist independently. A deployment can use Mooncake as HiCache's L3 backend (for prefix reuse) and also use Mooncake TransferEngine (for live KV transfer during disaggregation) simultaneously.

### LMCache Flag Context

```bash
# HiCache and LMCache are mutually exclusive per SGLang instance:
--enable-hierarchical-cache   # HiCache (built-in, recommended)
--enable-lmcache              # LMCache (middleware alternative)
```

Both flags appear in the SGLang design doc's configuration table but are never used together. The design doc notes that `--enable-lmcache` replaces `LMCRadixCache` for `RadixCache` at the engine level — a different replacement strategy from HiCache (which uses `HiRadixCache`).

---

## Omission 10: KV Cache Survey — Full Paper Lineage Table

**Source:** `L3/04_kv_cache_survey_tmlr25.md`
**Why omitted from COMBINED.md:** COMBINED.md presents the lineage in prose form. The survey's tabular summary covers more papers than HiCache's direct lineage and is preserved here in full for reference.

### Original text: Key Papers in the System-level, Multi-tier Thread

| Paper | Venue | Contribution |
|---|---|---|
| PagedAttention (vLLM) | SOSP 2023 | OS-style paged GPU KV memory management |
| RadixAttention (SGLang) | arXiv 2023 | Prefix tree for cross-request KV reuse |
| CachedAttention | ATC 2024 | Three-tier hierarchy with layer-wise loading (→ L3/02) |
| InfiniGen | OSDI 2024 | Selective prefetch of important KV from CPU (→ L4/01) |
| IMPRESS | FAST 2025 | Importance-informed three-tier with storage focus (→ L4/02) |
| LMCache | arXiv 2025 | Middleware KV cache layer, cross-engine sharing (→ L3/03) |
| HiCache (SGLang) | Blog 2025 | Production multi-tier with pluggable backends (→ L2/01) |

### Where HiCache Fits in the Full Taxonomy (Diagram)

```
System-level Optimizations
└── Multi-Tier KV Cache Systems
    ├── GPU + CPU tier (L1+L2)           → HiCache Tier 1+2
    ├── CPU + persistent storage tier    → HiCache Tier 2+3
    ├── Cross-instance sharing via L3    → HiCache shared backends
    └── Prefix reuse via RadixTree       → HiCache's HiRadixTree
```

HiCache is a **lossless, prefix-based, multi-tier system-level KV cache manager**. It does not compress, quantize, or drop KV entries — it focuses entirely on **where** caches live and **how quickly** they can be restored.

### Token-Level Methods That Are Composable With HiCache

While HiCache focuses on **where** to store KV caches (lossless), the survey documents methods for **which** KV entries to keep (lossy). These are orthogonal to HiCache:

- **H2O** (Heavy Hitter Oracle): keep the top-k most recently/frequently attended KV entries
- **StreamingLLM**: keep attention sinks (first few tokens) + recent sliding window
- **PyramidKV**: allocate more KV budget to lower layers
- **KIVI**: quantize KV to 2-bit to fit 4× more in GPU VRAM per tier

These methods can be combined with HiCache: quantize what's in GPU VRAM while offloading the rest to CPU/storage. The combination multiplies the benefit of both approaches.
