# References — Tiered KV Cache / HiCache

Organized by **reading level** (L1–L5) and **category**. Use this when writing or extending lesson content, locating production precedents, or designing exercises.

Layer 17 covers SGLang's three-tier KV cache system (HiCache): GPU VRAM → pinned CPU RAM → pluggable storage backend. References therefore cover: **the motivation for KV cache overflow, CPU offload mechanisms, the HiCache design and benchmarks, alternative hierarchical cache systems (LMCache, CachedAttention, InfiniGen, IMPRESS), the underlying PCIe bottleneck analysis, and the broader survey landscape of KV cache management.**

---

## Primary sources: SGLang HiCache documentation and blog

### SGLang HiCache Blog Post — LMSYS Org

- **URL:** https://lmsys.org/blog/2025-09-10-sglang-hicache/
- **Published:** September 10, 2025
- **Authors:** Zhiqiang Xie et al. (SGLang team)
- **Level:** L2–L3
- **What it contributes:**
  - The official SGLang announcement of HiCache with benchmark results: up to 6× throughput improvement and up to 80% reduction in TTFT on long-context workloads.
  - Design motivation from production: coding agent workloads (Qwen3-Coder-480B) with 25 k+ token dialogues exhausting GPU VRAM repeatedly.
  - GPU-assisted I/O kernels delivering up to 3× higher throughput vs `cudaMemcpyAsync` for CPU–GPU transfers.
  - Page-first layout decoupling: host memory buffer uses `page_first` layout while GPU pool stays `layer_first`, enabling larger DMA transactions and up to 2× higher CPU↔storage throughput.
  - Layer-wise overlapping: loads KV cache for layer N+1 while computing layer N during prefill.
  - Backend integrations: Mooncake, 3FS, and NIXL were the three production backends available at launch.
  - Reference command for DeepSeek-R1 on 8×H20 with 3FS backend — the canonical production launch recipe.

### SGLang HiCache Official Documentation

- **URL:** https://mintlify.com/sgl-project/sglang/optimization/hicache
- **Level:** L2
- **What it contributes:**
  - Complete flag reference for all `--hicache-*` parameters with descriptions, types, and defaults.
  - All registered storage backends (`file`, `nixl`, `nixl_v2`, `mooncake`, `hf3fs`, `aibrix`, `dynamic`) with brief descriptions.
  - Integration with PD Disaggregation: how to enable HiCache on prefill nodes and async offload on decode nodes.
  - Cross-cluster KV reuse: how different TP-size deployments can share the same storage layer.
  - Dynamic attach/detach: changing the L3 backend at runtime without server restart.
  - LMCache as an alternative (`--enable-lmcache`) with link to comparison.

### SGLang HiCache System Design Document

- **URL:** https://docs.sglang.io/advanced_features/hicache_design.html
- **GitHub:** https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/hicache_design.md
- **Level:** L3–L4
- **What it contributes:**
  - Architectural overview: analogy to CPU L1/L2/L3 cache hierarchy; L1/L2 private to each instance, L3 shared.
  - Detailed workflow: local match → prefetch → write-back.
  - CPU-to-GPU transfer optimisation details: compute-transfer overlap, GPU-assisted I/O kernels.
  - Write-back optimisation for MLA (Multi-Head Latent Attention): MHA and MLA handle multi-TP write-back differently.
  - `HiCacheStorage(ABC)` interface: `get(key)`, `set(key, value)`, `exists(key)` as the minimum backend contract.
  - Backend descriptions: Mooncake (RDMA + multi-NIC), NIXL (GPU Direct Storage + S3-compatible), AIBrix (production KVCache offloading), HiCacheFile (reference implementation).
  - All `--hicache-*` parameter descriptions in one place.

---

## The core problem: GPU VRAM overflow during inference

### Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)

- **Paper:** Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023
- **URL:** https://arxiv.org/abs/2309.06180
- **Level:** L2–L3
- **What it contributes:**
  - The foundational paper that introduced paged memory management for LLM KV caches, directly analogous to OS virtual memory.
  - Demonstrates that GPU KV cache memory is severely fragmented under the pre-PagedAttention allocation model, wasting 60–80% of GPU memory.
  - Establishes the "page" as the granular unit of KV cache allocation — the same unit HiCache uses for its tier-2 and tier-3 storage (each `page_size` tokens per page).
  - PagedAttention's pool-based allocator is the direct predecessor to `MHATokenToKVPool` and `MHATokenToKVPoolHost` in SGLang.

### RadixAttention: Efficient KV Cache Reuse with Prefix Caching (SGLang)

- **Paper:** Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," NeurIPS 2024
- **URL:** https://arxiv.org/abs/2312.07104
- **Level:** L2–L3
- **What it contributes:**
  - Introduces RadixAttention (Layer 12 in this series): a radix tree of GPU-resident KV pages, enabling prefix reuse across requests.
  - The `RadixCache` class is the direct base that `HiRadixCache` extends in Layer 17.
  - Shows cache hit rates of 30–90% across typical multi-turn and RAG workloads — motivating the need to preserve evicted pages rather than discard them.
  - The "eviction = loss" problem that HiCache exists to solve is implicit in this paper's eviction policy.

---

## Tiered KV cache: seminal systems papers

### CachedAttention / AttentionStore — Hierarchical KV Cache for Multi-Turn Conversations

- **Paper:** Gao et al., "Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention," USENIX ATC 2024
- **URL:** https://www.usenix.org/conference/atc24/presentation/gao-bin-cost
- **arXiv:** https://arxiv.org/abs/2403.19708
- **Published:** July 2024 (USENIX ATC '24, Open Access)
- **Level:** L3–L4
- **What it contributes:**
  - The canonical academic precedent for hierarchical KV cache in multi-turn conversation settings. CachedAttention maintains a GPU → CPU DRAM → SSD three-tier hierarchy, exactly the same topology as HiCache.
  - Results: TTFT reduced by up to 87%, prefill throughput improved by up to 7.8×, end-to-end inference cost reduced by up to 70%.
  - **Layer-wise pre-loading**: overlaps KV cache loading from CPU with GPU computation — the same technique SGLang HiCache uses (`load_to_device_per_layer` + layer-done counter).
  - **Asynchronous saving**: writes KV pages from GPU to CPU on a background stream — mirrors HiCache's `backup_from_device_all_layer` on `write_stream`.
  - **Scheduler-aware fetching**: uses a look-ahead window on the job queue to prefetch KV caches from disk to CPU before the request is dispatched — the same idea as HiCache's `prefetch_thread`.
  - **Positional encoding decoupling**: enables saved KV caches to survive context window truncation — relevant to multi-turn scenarios not covered in Layer 17.
  - Note: earlier arXiv versions of this paper appeared under the name "AttentionStore" — same work, same techniques.

### InfiniGen: Dynamic KV Cache Management with CPU Offloading

- **Paper:** Lee et al., "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management," USENIX OSDI 2024
- **URL:** https://www.usenix.org/conference/osdi24/presentation/lee
- **arXiv:** https://arxiv.org/abs/2406.19707
- **GitHub:** https://github.com/snu-comparch/InfiniGen
- **Published:** July 2024 (USENIX OSDI '24)
- **Level:** L3–L4
- **What it contributes:**
  - Addresses the KV cache offloading problem from a different angle: instead of loading all KV pages from CPU, **speculate which tokens are important** using partial query weights and the previous layer's key cache, then prefetch only those pages.
  - Achieves up to 3× speedup over prior KV cache management methods while maintaining better model accuracy.
  - Key insight: attention patterns are sparse; most layers attend to a small subset of tokens. Prefetching all CPU pages wastes PCIe bandwidth on data that will not be read.
  - Maintains the KV cache pool entirely in CPU memory (not GPU VRAM) — a more aggressive offloading strategy than HiCache's tier-2 model.
  - The speculative prefetch idea is related to HiCache's `best_effort` prefetch policy; InfiniGen formalises when to prefetch and how much.
  - **OSDI 2024** — top-tier systems venue.

### IMPRESS: Importance-Informed Multi-Tier Prefix KV Storage System

- **Paper:** Chen et al., "IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference," USENIX FAST 2025
- **URL:** https://www.usenix.org/conference/fast25/presentation/chen-weijian-impress
- **PDF:** https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf
- **Published:** 2025 (USENIX FAST '25, Open Access, Pages 187–201)
- **Level:** L3–L4
- **What it contributes:**
  - Three-tier prefix KV storage system (GPU, CPU DRAM, SSD) for LLM inference with an **importance-aware selection** layer.
  - Key insight: there is significant similarity in important-token index sets across attention heads; this allows identifying which KV pages to load with much less I/O than loading all pages.
  - **KV reordering**: repacks important KV entries into denser storage chunks, reducing read amplification at the SSD tier.
  - **Score-based cache replacement**: admission and eviction decisions based on `chunk_access_frequency × proportion_of_important_KVs` — a richer policy than pure LRU.
  - Results: TTFT reduced by up to 2.8× vs state-of-the-art systems on OPT-6.7B to OPT-30B with 55–65 GB prefix datasets (128 GB DRAM, 2 TB SSD setup).
  - Directly cited as prior work in AdaptCache (below) and in the SGLang HiCache benchmark comparisons.
  - **FAST 2025** — top-tier storage systems venue.

### AdaptCache: Adaptive Compression for Hierarchical KV Cache Storage

- **Paper:** "AdaptCache: KV Cache Native Storage Hierarchy for Low-Delay and High-Quality Language Model Serving"
- **arXiv:** https://arxiv.org/abs/2509.00105
- **Submitted:** September 2024, revised January 2026
- **Level:** L4
- **What it contributes:**
  - Extends the GPU → DRAM → SSD hierarchy with **lossy KV cache compression**: different compression algorithms and rates per tier and per request, chosen using a utility metric that balances quality loss against loading-delay savings.
  - Core argument: for KV caches that are not frequently reused or are difficult to compress, placing them on disk with a low compression ratio is better than evicting them. For frequently reused caches, aggressive compression lets more fit in DRAM, increasing hit rates.
  - Results: TTFT reduced by 56% vs naive prefill-and-offload on Llama-3.1-8B-Instruct with LongBench datasets.
  - Identifies the same PCIe bottleneck that HiCache addresses: even with a DRAM/SSD hierarchy, disk I/O dominates TTFT when cache hit rates are low.
  - Orthogonal to HiCache: HiCache stores KV pages losslessly; AdaptCache's compression could in principle be applied as a pre-processing step before writing to HiCache's tier-3 backend.

---

## LMCache: enterprise-scale alternative to HiCache

### LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference

- **Paper:** Cheng et al., "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference"
- **arXiv:** https://arxiv.org/abs/2510.09665
- **Tech report:** https://lmcache.ai/tech_report.pdf
- **Submitted:** October 2025 (v2: December 2025)
- **Level:** L3–L4
- **What it contributes:**
  - LMCache is the main alternative to SGLang HiCache; both are mentioned in the SGLang docs as options. Understanding LMCache clarifies what HiCache chose not to do.
  - **Architecture difference**: LMCache sits as a middleware layer between the inference engine and storage; HiCache is built into SGLang's `RadixCache` replacement. LMCache uses a layer-wise streaming model; HiCache uses page-granular DMA.
  - Supports both context caching (KV offload across queries) and PD disaggregation (cross-engine KV transfer) — the two use cases that HiCache and Mooncake respectively address in SGLang.
  - Storage hierarchy: CPU DRAM, local disk, remote disk, and Redis — broader than HiCache's current tier-3 backends.
  - Results: up to 15× throughput improvement vs baseline vLLM on multi-round QA and document analysis.
  - Real-world finding: "prompting shortcuts that randomly split the system prompt can reduce prefix cache hit ratio by half" — motivates cache-aware prompt engineering alongside HiCache.
  - LMCache is integrated in SGLang via `--enable-lmcache` as an opt-in alternative to HiCache.

---

## PCIe bottleneck: analytical foundations

### KV Cache Offloading Bottleneck Analysis: The Critical κ_crit Ratio

- **Paper:** "Characterizing the Performance Bottleneck of KV Cache Offloading"
- **arXiv:** https://arxiv.org/abs/2601.19910
- **Submitted:** December 2025 (MLSys 2025 submission format)
- **Level:** L4
- **What it contributes:**
  - Derives κ_crit: the critical ratio of cached-to-new tokens at which prefill transitions from compute-bound to memory-bound.
  - **κ_crit ranges from 1 to 76** (most under 15 before accounting for effective PCIe bandwidth). Real workloads exceed this by orders of magnitude: median κ_ratio is 100 for conversations and 5,000 for document queries.
  - 99% of latency in KV-offloading setups is spent on PCIe transfers, not GPU computation.
  - PCIe 5.0 provides 64 GB/s — 2% of HBM's bandwidth. Loading a 50 GB KV cache takes 15 ms from HBM but 800 ms from CPU DRAM.
  - This is the analytical paper that explains *why* HiCache's `load_back_duration_seconds` P99 matters so much as a monitoring metric — the PCIe path is almost always the bottleneck, not the GPU compute.
  - Validates HiCache's design choice to minimise the number of PCIe DMA operations (batched all-layer writes, JIT kernels, pinned memory).

### MTDS: Multi-Tier Dynamic Storage of KV Cache Under Resource Constraints

- **Paper:** Wang et al., "Multi-tier dynamic storage of KV cache for LLM inference under resource-constrained conditions," Complex & Intelligent Systems, 2026
- **URL:** https://link.springer.com/article/10.1007/s40747-025-02200-4
- **Published:** January 27, 2026 (Springer Open Access)
- **Level:** L3
- **What it contributes:**
  - Targets edge and resource-constrained deployments (not datacenter) — a different regime from HiCache but with the same three-tier idea.
  - Adaptive hierarchical eviction policy: eviction rates adjusted dynamically based on storage tier utilisation, rather than a fixed LRU policy.
  - Shows that multi-tier KV storage is practical on consumer GPU hardware (edge devices), extending the applicability of the concept beyond H100/A100 datacentres.
  - Compatible with mainstream inference frameworks (vLLM integration demonstrated).

---

## vLLM's CPU offloading implementation (production comparison point)

### vLLM KV Offloading Connector — Blog Post

- **URL:** https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html
- **Published:** January 8, 2026
- **Level:** L2–L3
- **What it contributes:**
  - Describes vLLM's native CPU KV offloading feature introduced in v0.11.0 (late 2025).
  - Architecture: asynchronous load/store API, offloading connector abstraction, CPU backend implementation.
  - Benchmark results: TTFT reduced by 2–22× depending on prompt size; up to 9× throughput increase for concurrent requests on Llama-3.1-8B-Instruct.
  - Key engineering details: GDRCopy-based zero-copy PCIe transfer, progressive block-transfer batching, overlap of KV copy with GPU computation.
  - Directly comparable to HiCache's `backup_from_device_all_layer` + `load_to_device_per_layer` — same problem, different implementation choices.
  - The follow-on RFC (vllm#33526, February 2026) proposes progressive "onloading" to avoid head-of-line blocking when a large and a small request share a prefix — the same problem HiCache's `write_through_threshold` indirectly addresses.

---

## Surveys: situating HiCache in the broader landscape

### A Survey on Large Language Model Acceleration based on KV Cache Management

- **Paper:** Li et al., "A Survey on Large Language Model Acceleration based on KV Cache Management"
- **arXiv:** https://arxiv.org/abs/2412.19442
- **Published in:** Transactions on Machine Learning Research (TMLR), May 2025 (v3: July 2025)
- **GitHub (paper list):** https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management
- **Level:** L2–L3
- **What it contributes:**
  - The most comprehensive survey of KV cache management strategies as of 2025: 200+ papers, organised into token-level, model-level, and system-level optimisations.
  - **System-level taxonomy** (most relevant to Layer 17): memory management, scheduling, and hardware-aware designs — includes prefix caching, CPU/disk offloading, multi-tier hierarchies, and quantisation.
  - Classifies HiCache-type systems as "dynamic selection with permanent eviction + CPU offload" — useful framing for understanding what HiCache is and is not (it is *not* a KV compression system).
  - Covers InfLLM (CPU-GPU orchestration for long sequences), Quest (block representation for top-k retrieval), PQCache (block-based management with LRU) — all related to the CPU offload problem.
  - Published in TMLR (peer-reviewed ML venue) with GitHub curated paper list updated through 2025.

### KV Cache Optimization Strategies for Scalable and Efficient LLM Inference (Survey)

- **Paper:** Xu et al., "KV Cache Optimization Strategies for Scalable and Efficient LLM Inference"
- **arXiv:** https://arxiv.org/abs/2603.20397
- **Submitted:** March 2026
- **Level:** L2
- **What it contributes:**
  - More recent shorter survey (2026) with a clean five-category taxonomy: cache eviction, compression, multi-tier hybrid memory, quantisation, prefix sharing.
  - **Hybrid memory solutions table** (Table 4): places systems including PagedAttention, LayerKV, InfLLM, and offloading-based approaches on a single comparison table with offload destination, mechanism, and key optimisation columns.
  - LayerKV explanation: splits KV cache by transformer *layer* (not by token), keeping some layers on GPU and offloading others to CPU — contrasted with HiCache's page-granular approach.
  - Good starting point for quickly scanning the landscape before reading individual papers.

---

## Source code: the SGLang HiCache implementation

### `hiradix_cache.py` — HiRadixCache

- **File:** `REPOS/sglang/python/sglang/srt/mem_cache/hiradix_cache.py`
- **Level:** L5 (source study)
- **Key anchors:**
  - `HiRadixCache.__init__` (line 65): host pool construction; MHA vs MLA pool selection; `HiCacheController` wiring.
  - `HiRadixCache.evict()` (line 835): write-through vs write-back policy; LRU heap; `_evict_backuped` vs `_evict_regular`.
  - `HiRadixCache.evict_host()` (line 905): CPU pool eviction to storage; node deletion from tree.
  - `HiRadixCache.load_back()` (line 940): CPU-to-GPU restore path; `load_cache_event`; ancestor node protection.
  - `_inc_hit_count()` (line 702): write-through trigger; the `write_through_threshold` parameter.
  - `writing_check()` (line 713): TP `all_reduce` for write-through acknowledgement across TP ranks.

### `memory_pool_host.py` — HostKVCache, MHATokenToKVPoolHost, MLATokenToKVPoolHost

- **File:** `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool_host.py`
- **Level:** L5 (source study)
- **Key anchors:**
  - `alloc_with_pin_memory` (line 132) and `alloc_with_host_register` (line 113): two strategies for pinned memory allocation.
  - `HostKVCache.__init__` (line 154): pool sizing from ratio or bytes; `psutil` safety check; `HICACHE_HOST_MEMORY_RESERVE_BYTES`.
  - `MHATokenToKVPoolHost.__init__` (line 290): layer-first vs page-first buffer layout; `k_data_ptrs` / `v_data_ptrs` as GPU-side pointer arrays.
  - `init_kv_buffer()` (line 350): layout-dependent tensor dimensions (`layer_first`, `page_first`, `page_first_direct`, `page_head`).
  - `load_to_device_per_layer()` (line 396): CPU → GPU per-layer transfer; JIT vs fallback kernels; `io_backend` dispatch.
  - `backup_from_device_all_layer()` (line 513): GPU → CPU all-layer transfer; CUDA write stream; JIT `jit_transfer_hicache_all_layer`.

### `hicache_storage.py` — HiCacheStorage, HiCacheStorageConfig, PoolTransfer

- **File:** `REPOS/sglang/python/sglang/srt/mem_cache/hicache_storage.py`
- **Level:** L5 (source study)
- **Key anchors:**
  - `HiCacheStorageConfig` (line 17): TP/PP rank info, MLA flag, layout flag, `extra_config` dict.
  - `PoolTransfer` (line 62): unified descriptor for CPU↔storage transfers; `host_indices`, `keys`, `hit_policy`.
  - `PoolTransferResult` (line 77): hit page counts per pool; v2 interface return type.
  - `HiCacheStorage` abstract class (line 98): `batch_exists_v2`, `batch_get_v2`, `batch_set_v2` (v2 interface); `get`, `set`, `exists` (legacy v1 contract).
  - `HiCacheFile` (line 277): local disk backend reference implementation.

### `storage/backend_factory.py` — StorageBackendFactory

- **File:** `REPOS/sglang/python/sglang/srt/mem_cache/storage/backend_factory.py`
- **Level:** L5 (source study)
- **Key anchors:**
  - `StorageBackendFactory._registry` (line 19): dict of `{name: {"loader": Callable, ...}}` — lazy loading.
  - `register_backend()` (line 44): registers a backend by name, module path, and class name; no import at registration time.
  - `create_backend()` (line 66): imports the backend class on first call; dispatches to `_create_builtin_backend()` or `_create_dynamic_backend()`.
  - `_create_dynamic_backend()` (line 114): dynamic custom backend path from `extra_config` JSON; required fields `backend_name`, `module_path`, `class_name`.
  - Backend registrations (lines 192–231): `file`, `nixl`, `mooncake`, `hf3fs`, `aibrix`, `eic`, `simm`.

### `managers/cache_controller.py` — HiCacheController

- **File:** `REPOS/sglang/python/sglang/srt/managers/cache_controller.py`
- **Level:** L5 (source study)
- **Key anchors:**
  - `HiCacheController.__init__` (line 247): `write_stream`, `load_stream`, `write_queue`, `load_queue`, `stop_event`; storage backend attach.
  - `write()` (line 663): GPU→CPU enqueue; allocates host indices from pinned pool; appends to `write_queue`.
  - `start_writing()` (line 681): merges pending ops; dispatches `backup_from_device_all_layer` on `write_stream`; records finish event.
  - `load()` (line 709) / `start_loading()` (line 749): CPU→GPU enqueue and dispatch; `LayerDoneCounter` for per-layer completion tracking.
  - `prefetch()` (line 799): storage→CPU prefetch enqueue via `prefetch_queue`.
  - `prefetch_thread_func()` (line 948): background thread consuming `prefetch_queue`; calls `page_get_func` per batch.
  - `write_storage()` (line 998): CPU→storage write pipeline; called by `HiRadixCache.write_backup_storage()` after GPU→CPU completes.

### `observability/metrics_collector.py` — Prometheus metrics

- **File:** `REPOS/sglang/python/sglang/srt/observability/metrics_collector.py`
- **Level:** L4–L5 (source study)
- **Key anchors:**
  - `sglang:cache_hit_rate` (line 272): overall GPU-level prefix cache hit fraction.
  - `sglang:hicache_host_used_tokens` (line 673): CPU pool fill (only created when `enable_hierarchical_cache=True`).
  - `sglang:hicache_host_total_tokens` (line 679): CPU pool capacity.
  - `sglang:evicted_tokens_total` (line 1639): cumulative GPU→CPU token count.
  - `sglang:load_back_tokens_total` (line 1652): cumulative CPU→GPU token count.
  - `sglang:load_back_duration_seconds` (line 1644): histogram of CPU→GPU load latency (buckets 1 ms–1 s).
  - `sglang:eviction_duration_seconds` (line 1631): histogram of GPU→CPU eviction latency.
  - `sglang:prefetched_tokens_total` (line 1476): cumulative storage→CPU token count.
  - `sglang:backuped_tokens_total` (line 1482): cumulative CPU→storage token count.
  - `StorageMetricsCollector` (line 1466): tier-3 I/O bandwidth and page-count histograms.
  - `RadixCacheMetricsCollector` (line 1573): eviction and load-back counters and histograms.

---

## Monitoring: Grafana and Prometheus setup

### SGLang Monitoring Stack — Docker Compose Setup

- **File:** `REPOS/sglang/examples/monitoring/README.md`
- **Level:** L2
- **What it contributes:**
  - Docker Compose setup for Prometheus + Grafana monitoring stack alongside SGLang.
  - Steps to import the pre-built Grafana dashboard and verify scraping from `--enable-metrics`.

### SGLang Pre-built Grafana Dashboard

- **File:** `REPOS/sglang/examples/monitoring/grafana/dashboards/json/sglang-dashboard.json`
- **Level:** L3–L4 (JSON study)
- **What it contributes:**
  - Pre-built dashboard including `Cache Hit Rate` panel (`sglang:cache_hit_rate`).
  - Template for adding custom HiCache panels for host pool utilisation, tier-2 throughput, and load-back latency P99.

---

## What this layer explicitly does not cover (deferred to Layer 18)

| Topic | Why excluded | Where |
|---|---|---|
| Mooncake `TransferEngine` (RDMA PD disaggregation) | Requires understanding of prefill-decode split | Layer 18 |
| `MooncakeStore` as a HiCache L3 backend | Cluster-level RDMA setup context needed | Layer 18 |
| LMCache integration (`LMCRadixCache`) | Alternative cache architecture, not a HiCache tier | Layer 18 |
| `DecodeKVCacheOffloadManager` | Specific to decode servers in PD disaggregation mode | Layer 18 |
| KV cache compression (KIVI, H2O, SnapKV, PyramidKV) | Lossy; orthogonal to lossless tiering | Future layer |
| FlexGen (weight + KV cache offload for single GPU) | Single-GPU constrained regime; different goal | External reading |
