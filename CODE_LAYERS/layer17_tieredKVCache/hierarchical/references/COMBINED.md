# Tiered KV Cache — Combined Reference

**What this file is:** A synthesis of all Layer 17 reference material into a single progressive narrative. The reading order moves from "why GPU-only KV cache is not enough" → "what three-tier architecture solves it" → "how data moves efficiently across tiers" → "why PCIe is the fundamental bottleneck" → "how prefetch and write policies manage that bottleneck" → "what storage backends enable" → "where these ideas come from academically" → "what the alternative approach looks like."

**Sources synthesized:**

| Level | File | Source | Key contribution |
|---|---|---|---|
| L2 | `01_sglang_hicache_blog.md` | LMSYS HiCache launch post (2025) | Benchmarks (6×/80%), launch commands, GPU-assisted kernels, real testimonials |
| L2 | `02_vllm_kv_offloading_blog.md` | vLLM offloading blog (2026) | DMA vs custom kernel analysis, physical block size problem, cross-validation |
| L3 | `01_sglang_hicache_design.md` | SGLang official design doc | Authoritative: HiRadixTree, workflow phases, layouts, all config parameters |
| L3 | `02_cachedattention_atc24.md` | CachedAttention (USENIX ATC '24) | Seminal 3-tier paper; layer-wise pre-loading; asynchronous saving |
| L3 | `03_lmcache_arxiv25.md` | LMCache (arXiv 2025) | Alternative middleware approach; cross-engine sharing; embedded vs connector |
| L3 | `04_kv_cache_survey_tmlr25.md` | KV Cache Survey (arXiv 2024) | Research taxonomy; HiCache's place; 200+ papers mapped |
| L4 | `01_infinigen_osdi24.md` | InfiniGen (USENIX OSDI '24) | PCIe quantified (75–90% of decode latency); selective prefetch rationale |
| L4 | `02_impress_fast25.md` | IMPRESS (USENIX FAST '25) | Disk I/O latency problem; importance-informed I/O; KV reordering |
| L4 | `03_kvcrit_pcie_bottleneck.md` | κ_crit framework (arXiv 2025) | Formal theory of compute-bound vs PCIe-bound; 99% latency on transfers |

---

## 1. The Problem: Why GPU-Only KV Cache Is Not Enough

### What the KV cache is

When an LLM processes a prompt, it computes Key and Value tensors for every input token at every attention layer. These tensors — the KV cache — are stored in GPU VRAM so that during autoregressive decode, the model can attend to all previous tokens without recomputing them.

**RadixAttention** (Layer 12) improves this further: it organises the KV cache as a prefix tree, allowing KV tensors from a common prefix to be shared across multiple requests. If two requests start with the same 5,000-token system prompt, only one copy of that prefix's KV tensors lives in GPU VRAM, shared across both.

### The capacity wall

The problem is capacity. GPU VRAM is fixed — typically 24–80 GB per card. As contexts grow longer and more simultaneous sessions run, the KV cache fills VRAM. When it is full, `RadixCache.evict()` discards the least-recently-used nodes to make room. The evicted KV tensors are gone permanently.

For short-lived requests, this is acceptable. For production workloads with temporal locality — the same 25K-token coding agent session returning for turn 9, the same RAG document prefix used by 200 concurrent queries — discarding is expensive:

> "In a coding agent scenario using Qwen3-Coder-480B, dialogues often stretched past 25K tokens around 8 turns per session. Without full KV cache retention, nearly every request required costly re-computation." — Novita AI

The access-pattern reality across diverse LLM workloads: 40–80% of input tokens are a shared prefix (system prompt, conversation history, document context). Evicting these and then re-prefilling them from scratch multiplies TTFT by 5–20× for subsequent turns.

### Where GPU-only caching fails: the request types

The KV Cache Management Survey (arXiv 2412.19442) categorizes LLM serving into three workload classes:

1. **Multi-turn conversations**: each new turn includes the full conversation history. A 10-turn session where each turn adds 200 tokens has a prefix that grows from 200 to 2,000 tokens over the session. With GPU-only LRU eviction, turns 8–10 frequently cause a cold miss.

2. **Shared-prefix workloads** (RAG, enterprise): many users share an identical system prompt + context. As soon as the GPU KV pool fills, the system evicts the shared prefix and must recompute it for every subsequent request.

3. **Long-context single requests** (128K+ tokens): a single request may require more KV than the entire GPU pool. There is no LRU strategy that helps.

All three workloads need the same thing: **KV tensors that survive beyond GPU eviction and can be restored without full re-prefill**.

---

## 2. The Three-Tier Architecture

### The CachedAttention foundation

CachedAttention (USENIX ATC '24) was the first peer-reviewed system to formally establish the three-tier KV cache hierarchy:

```
GPU VRAM     [L1 — fastest, smallest: 24–80 GB]
    ↕  PCIe DMA (pinned memory)
CPU DRAM     [L2 — medium: 128–512 GB]
    ↕  Storage I/O (NVMe / RDMA / network)
SSD / Remote [L3 — slowest, largest: TB scale]
```

The core insight: the eviction path from L1 → L2 is not a deletion — it is a **write**. The cold-start path from L2/L3 → L1 is not a re-prefill — it is a **read**. The model never needs to know the difference.

### HiCache's implementation of the hierarchy

SGLang's HiCache translates this directly into three coordinated components:

**L1 — `MHATokenToKVPool` / `MLATokenToKVPool`** (from Layer 12, unchanged): the GPU VRAM flat tensor buffer. Hot KV pages, directly accessible by the attention kernel. No change to the attention computation itself.

**L2 — `MHATokenToKVPoolHost` / `MLATokenToKVPoolHost`**: a pinned CPU memory buffer allocated with page-locked (`pin_memory=True`) memory. Controlled by `HiCacheController`. Size is configured via `--hicache-ratio` (CPU pool size = ratio × GPU pool size) or `--hicache-size` (exact bytes).

**L3 — `HiCacheStorage` backends**: pluggable interface with built-in implementations for Mooncake (RDMA distributed shared memory), 3FS (DeepSeek's distributed filesystem), NIXL (unified API for GDS/S3/3FS), file (local disk, reference implementation), AIBrix KVCache, and dynamic (user-supplied Python class).

**L1/L2 are private** to each SGLang instance. **L3 is shared** across all instances in the cluster — this is how two separate inference engines on different servers can benefit from the same cached prefix.

### HiRadixTree: the metadata coordinator

`HiRadixCache` extends `RadixCache` with a `HiRadixTree`. Each node in the tree corresponds to a span of consecutive tokens and records **where** that KV cache lives: GPU (`GPU_ONLY`), CPU (`CPU_ONLY`), storage (`STORAGE_ONLY`), or multiple tiers (`MIXED`).

The tree owns **no data** — it is a metadata index. Data lives in the tier-specific pools. When a lookup hits a node marked `CPU_ONLY`, the tree triggers a load operation from `HostKVCache` to the GPU pool. When a lookup hits `STORAGE_ONLY`, the tree triggers a query to `HiCacheController`, which issues an async prefetch from L3.

Three possible outcomes for every request prefix lookup:

```
Request arrives
      │
      ▼ HiRadixCache.match_prefix()
      │
      ├── Full GPU hit  ──────────────────────────► Scheduler (zero stall)
      │
      ├── CPU-resident hit  ─── load_back() ───────► Scheduler (~ms stall)
      │   (layer-wise, compute-transfer overlap)
      │
      ├── L3-resident hit  ─── storage.read() ─────► Scheduler (~100ms stall)
      │   (async prefetch while next batch runs)
      │
      └── Full miss  ──────────────────────────────── Full prefill (unchanged)
```

For **page-granularity matching**: when `--page-size > 1` (e.g., 64 tokens per page), the tree matches at page boundaries rather than token boundaries. Mid-node mismatches cause the node to **split** at the page boundary, improving future match precision. Larger page sizes increase I/O transfer efficiency but reduce match granularity.

For **tensor parallelism**: HiCache uses `all_reduce(op=min)` at two points during prefetch — once to agree on the L3 hit count across TP ranks (preventing divergent prefetch thresholds), and once after prefetch completes to agree on the successfully retrieved prefix length. For MLA models (DeepSeek), all ranks hold the same complete KV per token, so only **one rank** executes write-back to L3 to avoid redundant storage.

---

## 3. The Data Plane: Moving KV Across Tiers Efficiently

### Why raw `cudaMemcpyAsync` is not enough

When copying KV tensors from GPU to CPU (L1→L2) or loading back (L2→L1), two mechanisms exist:
- **DMA** (`cudaMemcpyAsync`): uses the hardware DMA engine; minimal interference with GPU cores; low CPU overhead.
- **Custom CUDA kernel**: uses GPU compute cores to copy data via raw pointers; high parallelism but competes with model computation for GPU resources.

The vLLM KV offloading blog (Jan 2026) benchmarked both at scale:

| Block size | Winner | Notes |
|---|---|---|
| < ~1 MB per transfer | Custom kernel | Faster for small transfers |
| > ~2 MB per transfer | DMA competitive | Equivalent throughput, less interference |
| Concurrent requests, mixed direction | **DMA** | 5–32% better throughput; kernel reduces throughput by 6% vs baseline for cache misses |

The reason DMA wins in concurrent workloads: custom kernels compete with the attention computation for GPU SIMD units. During decode, every GPU cycle lost to a copy kernel is a cycle not computing attention. DMA runs on a separate DMA engine, leaving GPU cores free.

SGLang's GPU-assisted I/O kernels (`sgl-kernel/csrc/kvcacheio/transfer.cu`) are specifically optimized for the KV cache transfer pattern — they achieve **up to 3× higher throughput** than raw `cudaMemcpyAsync` while managing the interference tradeoff. The `--hicache-io-backend` flag switches between `direct` (cudaMemcpyAsync) and `kernel` (GPU-assisted; recommended).

### The physical block size problem

The DMA engine is efficient for **large contiguous transfers**. This is why the memory layout of the KV pool matters:

vLLM's default layout stores each layer's KV cache separately, fragmenting a logical KV block into `num_layers × 2` sub-blocks. For Llama-3.1-8B at 16-token blocks, this produced 32 KB physical blocks — far too small for DMA efficiency. After vLLM's v0.12.0 layout change (storing all layers contiguously per page), blocks grew to 2 MB — in the DMA-efficient range. This single layout change produced 4× TTFT improvement and 5× throughput improvement.

HiCache addresses the same problem with explicit layout control:

| Layout | Description | Best for |
|---|---|---|
| `layer_first` | Layer dimension outermost (GPU's native layout) | GPU computation kernels |
| `page_first` | Page dimension outermost — all layers for a page are contiguous | L3 I/O: one large sequential read per page |
| `page_first_direct` | `page_first` with sub-grouping within each page | L2→GPU transfers at page-layer granularity |

GPU VRAM stays `layer_first` (required by the attention kernel). CPU pool (L2) uses `page_first` or `page_first_direct` for DMA-efficient transfers. L3 backends receive data in `page_first` layout — a single page's entire KV tensor is contiguous on disk or in remote memory, enabling a single large I/O per page.

### Compute-transfer overlap

The most important single optimization for L2→L1 transfer latency: **load KV cache for layer N+1 while the GPU computes attention for layer N**.

This is CachedAttention's "layer-wise pre-loading" technique, implemented in HiCache as `load_to_device_per_layer()`. The transfer and the compute happen on separate CUDA streams, overlapping in time. For a model with 80 layers, only the first layer's transfer is exposed (unavoidable); layers 2–80 are hidden behind the previous layer's computation.

Without overlap: transfer latency = 80 × transfer_time_per_layer (sequential).
With overlap: transfer latency ≈ transfer_time_per_layer (only the first is uncovered).

This is why L2 loads can achieve sub-millisecond effective stall even for large models.

### Zero-copy for L3 backends

Both prefetch (L3→L2) and write-back (L2→L3) pass **memory addresses and sizes** directly to the L3 backend rather than staging through an intermediate buffer. The L3 implementation DMA's directly from/to the pinned L2 buffer. This eliminates one buffer copy per I/O operation — important when transferring gigabytes of KV data.

---

## 4. The PCIe Bottleneck: The κ_crit Framework

### The problem stated precisely

The κ_crit paper (arXiv 2601.19910) provides the theoretical foundation for why HiCache's data-plane optimizations exist and what their limits are.

Let:
- `C` = number of **cached** tokens to load from CPU DRAM
- `P` = number of **new** tokens to prefill (compute KV from scratch)
- `κ = C / P` = the cached-to-prefill ratio

**κ_crit** is the threshold at which the system transitions:
- `κ < κ_crit`: **compute-bound** — GPU is the bottleneck; KV loading fits within compute time; HiCache is fully beneficial
- `κ > κ_crit`: **memory-bound** — PCIe is the bottleneck; GPU waits for data; HiCache helps but is limited by interconnect bandwidth

```
κ_crit ≈ (GPU_FLOPS × bytes_per_KV_entry) / (PCIe_bandwidth × FLOPs_per_token)
```

For representative hardware (H100, PCIe 5.0):

| Model | KV size per token | κ_crit |
|---|---|---|
| Llama-3.1-8B | ~1.0 MB | ~8 |
| Llama-3.1-70B | ~3.5 MB | ~3 |
| DeepSeek-R1-671B (FP8) | ~6.0 MB | ~2 |

### The practical consequence

Real RAG and multi-turn workloads operate at κ ratios of **100–10,000** — two to four orders of magnitude above κ_crit. The empirical measurement:

> **99% of inference latency is PCIe transfers.** GPU utilization is ~28% of rated TDP — mostly idle, waiting for data. — κ_crit paper, measured on H100

InfiniGen (OSDI '24) independently quantified: with naïve full-KV loading from CPU, **75–90% of each decode step** is spent on PCIe transfers.

This is not a failure of HiCache — it is the fundamental physics of the CPU↔GPU interconnect. PCIe 5.0 ×16 delivers ~64 GB/s peak, ~48–55 GB/s sustained for large transfers, and 10–20 GB/s for small (< 512 KB) transfers. Moving one 70B model's KV cache for a 10K-token context requires transferring ~35 GB — more than the GPU can compute in the same time.

### How HiCache addresses this

HiCache's data-plane optimizations directly target the κ_crit constraint:

| Optimization | Effect on PCIe bottleneck |
|---|---|
| `page_first` layout | Increases effective transfer size → from 10–20 GB/s range to 48–55 GB/s range |
| `--hicache-io-backend kernel` | GPU-assisted copies achieve ~3× vs cudaMemcpyAsync for same-size transfers |
| Compute-transfer overlap | Hides transfer latency behind GPU computation; reduces exposed stall time |
| Zero-copy L3 I/O | Eliminates one staging copy per operation; reduces total bytes moved |
| `page_size` tuning | Larger pages = fewer transactions = lower per-transaction overhead |

### Monitoring κ in production

```promql
# Approximate κ ratio (cached tokens loaded vs new tokens generated)
sum(rate(sglang:load_back_tokens_total[5m]))
  / sum(rate(sglang:num_tokens_generated_total[5m]))

# P99 L2→L1 load latency — high value signals PCIe-bound regime
histogram_quantile(0.99, rate(sglang:load_back_duration_seconds_bucket[5m]))
```

When both are high: you are in the PCIe-bound regime. Short-term: increase `--page-size`, use `--hicache-mem-layout page_first_direct`, use `--hicache-io-backend kernel`. Long-term: hardware upgrade (CXL memory provides memory-semantics access without PCIe overhead; NVLink-C2C on Grace-Hopper reaches 900 GB/s vs 64 GB/s PCIe).

### Architecture-level relief

Model architecture changes also improve κ_crit by reducing KV size per token:
- **GQA (Grouped Query Attention)**: reduces KV by num_heads/num_kv_heads — Llama-3.1 uses GQA with 8 KV heads vs 64 query heads (8× KV reduction)
- **MLA (Multi-head Latent Attention)**: DeepSeek's approach with extreme KV compression; κ_crit effectively raised 10–20× vs full-attention models

HiCache supports MLA via `MLATokenToKVPoolHost` (separate host pool implementation with MLA-appropriate layout and single-rank write-back).

---

## 5. Prefetch and Write Policies: Managing the Bottleneck

### The prefetch problem

When a request partially matches L3 storage, HiCache can prefetch the matching pages from L3 to L2 before the prefill phase begins. But the prefetch itself takes time — for a slow backend (NVMe: ~100µs; network storage: 1–10ms), and for a large matched prefix, the prefetch can take longer than just running the full prefill from scratch.

HiCache's `prefetch_threshold` (default: 256 tokens) ensures prefetch is only triggered when the L3 match is large enough to justify the I/O latency. Below threshold, the scheduler skips prefetch and runs full prefill.

**The three termination strategies** balance TTFT vs cache hit rate:

| Policy | Behavior | Best for |
|---|---|---|
| `best_effort` | Prefetch terminates immediately when GPU can start prefill | Latency-sensitive; accepts partial cache hits |
| `wait_complete` | Block until all matched pages are in L2 | Maximum cache hit rate; accepts higher TTFT |
| `timeout` | Wait up to a computed deadline, then proceed with whatever arrived | Production (balanced SLO) |

The `timeout` policy is the most production-practical: it adapts to actual I/O speed via:
```
timeout = prefetch_timeout_base + prefetch_timeout_per_ki_token × num_tokens_to_fetch / 1024
```
This gives slow backends (file on NVMe) more time than fast backends (Mooncake RDMA) proportionally.

### The InfiniGen insight: not everything is worth fetching

InfiniGen (OSDI '24) provides the theoretical justification for *selective* prefetch: for each query, only a small subset of KV entries receive significant attention weight. The rest contribute negligibly to the output. Fetching the full KV for a matched prefix transfers more bytes than necessary over the PCIe bottleneck.

InfiniGen's approach: use the current layer's input activations and a slice of the next layer's Q/K projections to **speculate** which tokens will receive high attention in the next layer. Prefetch only those token's KV pages. This reduces PCIe transfer by 50–80% while maintaining accuracy.

HiCache approximates this differently: **access frequency as a proxy for importance**. The `write_through_selective` policy writes pages to L3 only after their hit count exceeds a threshold — pages that are accessed repeatedly are more likely to be important for future requests. The `_inc_hit_count()` tracker in `HiRadixCache` implements this.

### The IMPRESS insight: disk I/O changes the calculus

IMPRESS (FAST '25) addresses the specific case of L3 backends on disk (NVMe, network storage). Its key finding: **loading all prefix KV from disk does not always reduce TTFT** — the I/O latency can exceed the GPU computation savings.

IMPRESS's solution: compute importance scores per token (lightweight approximation of attention), physically reorder KV pages on disk by importance (important pages contiguous for sequential read), and load only the top-k% of pages. This achieves comparable inference accuracy while dramatically reducing I/O volume.

HiCache's connection:
- `page_first` layout is the lossless version of IMPRESS's contiguous important-page layout — it makes all pages maximally contiguous without selecting which to load
- `write_through_selective` approximates IMPRESS's selective loading using access frequency rather than content-based importance
- HiCache's `prefetch_threshold` is a threshold on total prefetch volume, analogous to IMPRESS's top-k threshold

The combination of InfiniGen + IMPRESS insights → HiCache's `prefetch_threshold` + `write_through_selective` + `best_effort` policy: don't load more than you need, don't load from slow storage unless it is worth it, stop loading when compute is ready.

### Write policies

Three policies control when KV data is written from L1 (GPU) to L2/L3:

| Policy | When data is written | Best for |
|---|---|---|
| `write_through` | Immediately on every eviction | Highest cache hit rate (if bandwidth allows) |
| `write_through_selective` | After access frequency exceeds threshold | Reduces I/O overhead; persists only hot prefixes |
| `write_back` | Only when evicted from the upper tier | Capacity-constrained environments; lowest write I/O |

For L3 (shared backends): when data moves L2→L3, HiCache checks which pages are **not already in L3** before writing — cross-instance deduplication. If two SGLang instances both served the same prefix, only one needs to write it to L3.

---

## 6. Storage Backends and Production Launch

### Backend interface

All L3 backends implement three methods from `HiCacheStorage`:
- `batch_exists_v2(keys)` — check which pages are in L3 (used to avoid duplicate writes)
- `batch_get_v2(keys, dst)` — fetch pages from L3 into L2 pinned buffer
- `batch_set_v2(keys, src)` — write pages from L2 into L3

This minimal interface is what allows the pluggable backend design: any storage system that can implement these three methods can be used as a HiCache L3 tier.

### Built-in backends

| Backend | Transport | Use case |
|---|---|---|
| `mooncake` | RDMA zero-copy distributed memory | Production: lowest latency, cluster-wide shared pool |
| `hf3fs` | DeepSeek's distributed filesystem | Production: best with DeepSeek model deployments |
| `nixl` | Unified API (GDS, 3FS, S3-compatible) | Flexible: GPU Direct Storage, S3, other backends |
| `aibrix` | External KVCache service API | Enterprise: separate KV cache service |
| `file` | Local disk (NVMe, SSD) | Development, single-node NVMe setups |
| `dynamic` | Custom Python class loaded at runtime | Custom or future backends |

### Production launch commands

**Minimal (L2 only, CPU RAM)**:
```bash
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0
```

**DeepSeek-R1 on 8× H20 with 3FS (from LMSYS HiCache blog)**:
```bash
python3 -m sglang.launch_server \
  --model-path /DeepSeek-R1/ --tp 8 --page-size 64 \
  --context-length 65536 --chunked-prefill-size 6144 \
  --mem-fraction-static 0.85 \
  --enable-hierarchical-cache --hicache-ratio 2 \
  --hicache-io-backend kernel --hicache-mem-layout page_first \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete
```

**Qwen3-235B on 8× H800 with Mooncake (RDMA)**:
```bash
MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata" \
MOONCAKE_GLOBAL_SEGMENT_SIZE=816043786240 \
MOONCAKE_PROTOCOL="rdma" \
MOONCAKE_DEVICE="$DEVICE_LIST" \
MOONCAKE_MASTER=127.0.0.1:50051 \
python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 8 --page-size 64 \
  --enable-hierarchical-cache --hicache-ratio 2 \
  --hicache-storage-prefetch-policy timeout \
  --hicache-storage-backend mooncake
```

### Full configuration reference

| Flag | Default | Effect |
|---|---|---|
| `--enable-hierarchical-cache` | off | Required to activate HiCache |
| `--hicache-ratio RATIO` | — | CPU pool = RATIO × GPU pool (must be > 1) |
| `--hicache-size SIZE_GB` | — | CPU pool in GB (overrides ratio) |
| `--page-size N` | 16 | Tokens per page; larger = better I/O, lower match granularity |
| `--hicache-io-backend` | `direct` | `direct` (cudaMemcpyAsync) or `kernel` (GPU-assisted, recommended) |
| `--hicache-mem-layout` | `layer_first` | `layer_first`, `page_first`, or `page_first_direct` |
| `--hicache-write-policy` | `write_back` | `write_back`, `write_through`, `write_through_selective` |
| `--hicache-storage-backend` | — | `mooncake`, `hf3fs`, `nixl`, `file`, `aibrix`, `dynamic` |
| `--hicache-storage-prefetch-policy` | `best_effort` | `best_effort`, `wait_complete`, or `timeout` |
| `--hicache-storage-backend-extra-config` | — | JSON string or `@file.toml` with backend config |
| `--enable-lmcache` | off | Use LMCache instead of HiCache (see §8) |

**Prefetch timeout tuning** (via `--hicache-storage-backend-extra-config`):
```json
{
  "prefetch_threshold": 256,
  "prefetch_timeout_base": 0.5,
  "prefetch_timeout_per_ki_token": 0.25
}
```

---

## 7. The Research Lineage: From CachedAttention to HiCache

The KV cache management survey (arXiv 2412.19442) places HiCache in a clear academic lineage. The **system-level, multi-tier thread** is the direct ancestor:

```
PagedAttention (SOSP '23)          OS-style paged GPU KV memory management
    ↓
RadixAttention / SGLang (arXiv '23) Prefix tree for cross-request KV reuse
    ↓
CachedAttention (ATC '24)          Three-tier GPU/CPU/SSD hierarchy with layer-wise loading
    ↓
InfiniGen (OSDI '24)               Selective CPU prefetch to avoid PCIe saturation
    ↓
IMPRESS (FAST '25)                 Importance-informed I/O for disk-tier efficiency
    ↓
HiCache / SGLang (2025)            Production system with pluggable backends, TP support
```

Each step in the lineage addresses a bottleneck that the previous system revealed:
- PagedAttention → fragment VRAM efficiently
- RadixAttention → reuse across requests via prefix matching
- CachedAttention → survive beyond GPU VRAM via CPU/SSD tiers
- InfiniGen → PCIe is the new bottleneck; load selectively
- IMPRESS → disk I/O has a different latency structure; reorder and filter
- HiCache → production-ready: pluggable backends, TP correctness, write policies, monitoring

### How HiCache implements each CachedAttention technique

CachedAttention proposed four core techniques. HiCache implements three of them exactly:

| CachedAttention technique | HiCache implementation |
|---|---|
| Layer-wise pre-loading (hide L2→L1 latency) | `load_to_device_per_layer()` — compute-transfer overlap on separate CUDA stream |
| Asynchronous saving (L1→L2 non-blocking) | `backup_from_device_all_layer()` + `HiCacheController.write()` — dedicated write thread |
| Scheduler-aware placement (prioritize hot sessions) | `_inc_hit_count()` tracking + `write_through_selective` policy |
| **KV Cache Invalidation Avoidance** (positional encoding decoupling) | **Not implemented in HiCache** — see §Omissions |

### The survey taxonomy

The KV cache survey identifies three levels of optimization:

**Token-level** (lossy — reduces what is stored): H2O (heavy hitter oracle), StreamingLLM (attention sinks + window), KIVI (2-bit quantization), PyramidKV (per-layer budget).

**Model-level** (architectural — changes what is computed): MQA, GQA, MLA.

**System-level** (lossless — changes where and how KV is stored): RadixAttention, PagedAttention, HiCache.

HiCache is strictly system-level and lossless — it does not touch the KV values themselves. Token-level methods (KIVI, H2O) are **orthogonal** and composable: you could quantize KV values (KIVI: 2-bit → 4× more per tier) while simultaneously using HiCache for tiered storage. Model-level changes (MLA, GQA) reduce KV tensor size, making each tier more effective.

---

## 8. Alternative: LMCache

LMCache (arXiv 2510.09665, Columbia/UChicago) solves the same KV capacity problem with a different architectural position:

| Aspect | HiCache | LMCache |
|---|---|---|
| **Position in stack** | Embedded in SGLang memory manager | Middleware layer (external to engine) |
| **Engine coupling** | Tight — direct access to `RadixCache`, `HiCacheController` | Loose — connector API abstracts the engine |
| **KV structure** | Page-first contiguous blocks | Layer-wise streaming (dedicated CUDA stream per layer) |
| **Cross-engine sharing** | Via L3 shared backends (Mooncake, 3FS) | First-class feature (Redis, S3, network) |
| **Backend support** | Mooncake, 3FS, NIXL, AIBrix, file, dynamic | CPU DRAM, disk, Redis, S3-compatible |
| **SGLang flag** | `--enable-hierarchical-cache` | `--enable-lmcache` |
| **Performance** | Tighter integration → more optimized for SGLang | More flexible; slightly lower peak performance |

**When to prefer HiCache**: Mooncake, 3FS, or NIXL backends; maximum single-engine performance; fine-grained prefetch/write-back policy control.

**When to prefer LMCache**: cross-engine sharing across vLLM + SGLang mixed deployments; Redis or S3 storage; middleware approach that can be updated independently of the inference engine.

Both ship in SGLang. They are mutually exclusive per instance.

---

## 9. Consolidated Benchmark Numbers

From production deployments and papers:

| Source | Model | Setup | TTFT improvement | Throughput improvement |
|---|---|---|---|---|
| Novita AI (HiCache) | Qwen3-Coder-480B | 3FS, 25K+ token sessions | **56% TTFT reduction** | **2× throughput** |
| Ant Group (HiCache) | DeepSeek-R1-671B | Mooncake, PD-disaggregated | **84% TTFT reduction** | — |
| LMSYS benchmarks (HiCache) | DeepSeek models | 3FS/Mooncake | up to **80% TTFT reduction** | up to **6× throughput** |
| vLLM offloading blog | Llama-3.1-8B | H100, CPU offloading | **2×–22× TTFT reduction** | up to **9× at 80% hit rate** |
| CachedAttention (ATC '24) | GPT-2-XL | GPU/CPU/SSD | **87% TTFT reduction** | **7.8× prefill throughput** |
| LMCache (arXiv 2025) | Various | CPU DRAM offload | — | up to **15× throughput** |
| IMPRESS (FAST '25) | LLaMA-2-7/13B | GPU/CPU/NVMe | **2.8× TTFT reduction** (disk tier) | — |
| InfiniGen (OSDI '24) | LLaMA, GPT-3 | CPU offload, selective | **3.00× improvement** vs prior methods | — |

---

## Appendix: What Is Left Out and Why

### CachedAttention's positional encoding decoupling technique

CachedAttention's fourth core technique decouples positional encoding (RoPE) from saved KV caches, allowing cached entries to remain valid even after conversation truncation. This is a significant accuracy improvement for very long multi-turn sessions. **HiCache does not implement this** — it relies on token-exact prefix matching, which breaks when a session exceeds the context window and older tokens are dropped. This is a known limitation for sessions that grow past `--context-length`. Implementing pre-computation decoupling would require changes to the model's attention forward pass.

### InfiniGen's rehearsal computation (speculative next-layer attention)

InfiniGen's core technique uses a slice of the next layer's Q/K projection weights and the current layer's input to run a cheap approximate attention for next-layer importance scoring. This speculative computation happens on-GPU during the current layer's compute, identifying which CPU KV pages to prefetch before the next layer needs them. HiCache approximates this via access-frequency hit-count tracking, which is simpler but less precise — it identifies popular prefixes rather than important-per-query tokens. The combination of InfiniGen-style per-query importance scoring with HiCache's multi-tier hierarchy is an open research direction.

### IMPRESS's KV page reordering on disk

IMPRESS physically reorders KV pages on the storage backend so that important pages are stored contiguously, enabling a single sequential I/O to retrieve all important content. HiCache's `page_first` layout ensures pages are contiguous **within a prefix** but does not sort pages by importance across prefixes within the storage backend. Implementing disk-side importance ordering would require the storage backend to track importance scores per page — an extension to the `HiCacheStorage` interface.

### vLLM's version-by-version improvements (v0.11.0 → v0.12.0 → v0.14.0)

The vLLM offloading blog describes incremental improvements: v0.12.0's block layout change (4× TTFT, 5× throughput improvement over v0.11.0), v0.14.0's preemption recovery from CPU, and a race condition fix. These are vLLM-specific engineering details relevant for vLLM deployments but not directly applicable to SGLang/HiCache design.

### LMCache's enterprise production insights

The LMCache paper includes enterprise deployment data: context truncation halves prefix cache hit ratio; cross-engine sharing is the key differentiator at scale. These insights are specific to LMCache's middleware architecture — the cross-engine sharing advantage specifically comes from LMCache's external position (it can be shared across vLLM and SGLang instances simultaneously). HiCache achieves cross-instance sharing differently (via shared L3 backends like Mooncake/3FS).

### KV cache survey's token-level methods in detail

The survey covers H2O, StreamingLLM, PyramidKV, KVMerger, and KIVI in depth. These token-level (lossy) methods are orthogonal to HiCache and are not covered here — they change what is stored, while HiCache changes where it is stored. They are the subject of Layer 18 (KV Cache Quantization).

### Hardware-specific details (CXL memory, NVLink-C2C, GPU Direct Storage)

The κ_crit paper and IMPRESS both discuss next-generation hardware interconnects. CXL memory (memory-semantic DRAM pooling without PCIe overhead), NVLink-C2C (Grace-Hopper's 900 GB/s CPU↔GPU bandwidth vs 64 GB/s PCIe), and GPU Direct Storage (RDMA directly to/from GPU memory bypassing CPU) would each significantly change the tier-access latency and bandwidth assumptions underlying HiCache's design. These are near-future developments; the current HiCache design is PCIe-optimized.
