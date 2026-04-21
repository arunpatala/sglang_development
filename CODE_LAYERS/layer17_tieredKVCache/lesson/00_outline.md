# Layer 17 — Lesson Outline

## What This Lesson Covers

Layer 12 introduced `RadixCache`: a prefix-aware GPU memory cache that reuses KV tensors across requests sharing a common prompt prefix. When GPU VRAM fills up, `RadixCache` evicts the least-recently-used nodes — throwing away computed KV tensors that could be reused for the next request with the same prefix. For short-lived workloads this is acceptable. For long-lived document or conversation caches — where the same context is reused for hours or days — silently discarding KV tensors wastes significant prefill compute.

Layer 17 introduces **HiCache** (Hierarchical Cache): a three-tier extension to `RadixCache` that, instead of discarding evicted KV tensors, moves them to cheaper storage — first to pinned CPU RAM, then optionally to persistent remote or local storage — and loads them back to GPU when needed again. The eviction path becomes a write path; the cold-start path becomes a read path.

No new model execution logic is introduced. The `Scheduler`, `Tokenizer`, and attention kernels from previous layers carry forward unchanged. The new work is entirely in the memory management layer: how `HiRadixCache` extends `RadixCache` to coordinate three tiers, how `HostKVCache` manages a pinned CPU buffer with fast GPU↔CPU DMA transfers, how `HiCacheStorage` provides a pluggable interface for tier-3 backends (file, remote, RDMA), and how Prometheus metrics and log output expose the benefit of each tier.

The lesson follows the data path: why GPU eviction is wasteful → the three-tier architecture → the GPU↔CPU host pool → the storage backend interface → observability → how to enable HiCache and tune it.

The central new abstraction is the **write-before-evict / load-before-compute** contract: when `HiRadixCache` would evict a node, it first writes its KV pages to a lower tier; when a request arrives whose prefix is in a lower tier, the cache loads those pages back before the scheduler dispatches the request to the model. This contract is invisible to the model runner — the GPU KV pool always looks like a warm cache.

---

## Sections

### 01 — The Eviction Problem: Why Discarding KV Tensors Is Wasteful (`01_eviction_problem.md`)
- Layer 12's `RadixCache.evict()` frees GPU memory when the pool is full; evicted nodes lose their KV tensors permanently
- The cost of re-prefill: re-running the transformer on a 10 k-token document to rebuild its KV cache takes 2–10 seconds on an A100 at typical batch sizes; this latency is invisible in single-request benchmarks but dominates multi-turn chat and RAG workloads
- The access-pattern insight: long-context workloads have high temporal locality — the same 10 k-token system prompt is reused for thousands of requests over hours; discarding it after one eviction cycle multiplies TTFT by 5–20×
- The gap in Layer 12: `RadixCache` tracks only GPU-resident pages; there is no concept of "evicted but recoverable" — a node is either in the GPU pool or gone
- What HiCache adds: the evicted node's KV pages are written to CPU RAM or disk before the GPU slot is reclaimed; the node becomes "cold" (tracked but not GPU-resident) rather than deleted; a future cache lookup that hits a cold node triggers a host-side load instead of a prefill

### 02 — The Three-Tier Architecture (`02_three_tier_architecture.md`)
- Tier 1 — GPU VRAM: `MHATokenToKVPool` / `MLATokenToKVPool` (from Layer 12); hot KV pages, directly accessible by the attention kernel; smallest capacity (24–80 GB typical)
- Tier 2 — CPU RAM (host pool): `MHATokenToKVPoolHost` / `MLATokenToKVPoolHost`; pinned memory (page-locked), PCIe DMA accessible; medium capacity (128–512 GB typical); managed by `HiCacheController`
- Tier 3 — Storage: `HiCacheStorage` implementations (`HiCacheFile`, `NixlStore`, `MooncakeStore`, etc.); persistent or remote; large capacity (TB scale); slowest access
- `HiRadixCache` as the coordinator: extends `RadixCache`; each node gains a `cache_state` field (`GPU`, `CPU`, `STORAGE`, `MIXED`); eviction from tier 1 triggers a write to tier 2; eviction from tier 2 optionally triggers a write to tier 3; cache lookup of a cold node triggers the reverse path
- The three possible lookup outcomes for a request prefix: (a) full GPU hit — return immediately, (b) partial CPU/storage hit — trigger async load, block until loaded, then return, (c) complete miss — full prefill required
- Data flow diagram:
  ```
  New request arrives
        │
        ▼
  HiRadixCache.match_prefix()
        │ GPU-resident hit?   ──────────────────────────► Scheduler (no stall)
        │ CPU-resident hit?   ─── load_back() ──────────► Scheduler (~ms stall)
        │ Storage hit?        ─── storage.read() ────────► Scheduler (~100ms stall)
        │ Miss                ─────────────────────────── Full prefill
        │
  After decode completes
        │
  HiRadixCache.insert()
        │ GPU pool full?      ─── evict_host() ──────────► CPU pool
        │ CPU pool full?      ─── backup_storage() ───────► HiCacheStorage
  ```

### 03 — Tier 2: The Host Pool (GPU ↔ CPU) (`03_host_pool.md`)
- `HostKVCache` base class (`memory_pool_host.py:154`): manages a pinned CPU tensor buffer allocated with `alloc_with_pin_memory()` or `alloc_with_host_register()`; exposes `alloc()`, `free()`, `load_to_device_per_layer()`, `backup_from_device_all_layer()`
- `MHATokenToKVPoolHost` (`memory_pool_host.py:290`): stores full K and V tensors layer-by-layer for standard multi-head attention; buffer layout mirrors the GPU pool so PCIe DMA strides are contiguous
- `MLATokenToKVPoolHost` (`memory_pool_host.py:787`): compressed variant for Multi-head Latent Attention (MLA) models (e.g. DeepSeek); stores lower-rank KV representations
- GPU→CPU path (`backup_from_device_all_layer`): called during eviction; copies KV pages across all layers from GPU tensor pool to CPU pinned buffer using a dedicated CUDA stream; non-blocking from the request execution stream
- CPU→GPU path (`load_to_device_per_layer`): called when a cold node is needed; copies the CPU pages back to the GPU pool; the scheduler stalls the request until the load completes
- Why pinned memory matters: page-locked (non-swappable) host memory is directly accessible by the DMA engine; avoids a kernel-mode `cudaMemcpy` staging copy; PCIe bandwidth utilisation is significantly higher than with pageable memory
- `hicache_ratio` controls the CPU pool size relative to the GPU pool: `hicache_ratio: 2.0` means the CPU pool holds 2× as many tokens as the GPU pool; set to 0 and use `hicache_size` (bytes) instead for exact control
- `hicache_io_backend`: `"kernel"` uses CUDA async copy kernels (default); alternative backends may use vendor DMA APIs
- `hicache_mem_layout`: `"layer_first"` (default) vs `"page_first_direct"` — affects how KV pages are laid out in the pinned buffer and which GPU kernel strides are used; `page_first_direct` enables a faster but layout-constrained path

### 04 — Tier 3: The Storage Backend Interface (`04_storage_backend.md`)
- `HiCacheStorage` abstract base class (`hicache_storage.py:98`): defines `write(pool_transfer)` and `read(pool_transfer)` — the only interface the controller cares about; each implementation decides how to map page indices to storage addresses
- `HiCacheStorageConfig` (`hicache_storage.py:17`): carries capacity, max page counts, and page shape metadata shared between the host pool and the storage backend
- `PoolTransfer` / `PoolTransferResult` (`hicache_storage.py`): data objects that describe which CPU buffer pages to write or read and their destination page indices; used to avoid copying data — the storage implementation DMAs from/to the pinned buffer directly
- `HiCacheFile` (`hicache_storage.py:277`): reference implementation; writes each KV page as a flat tensor file on the local filesystem; useful for development and single-node setups with NVMe
- `StorageBackendFactory` (`storage/backend_factory.py:16`): pluggable registry; maps backend name strings to backend classes; currently registered backends:
  - `"file"` → `HiCacheFile` (local disk)
  - `"nixl"` → NIXL agent (network RDMA or object storage)
  - `"mooncake"` → `MooncakeStore` (distributed shared memory via `mooncake.store.MooncakeDistributedStore`)
  - `"hf3fs"` → HuggingFace 3FS distributed filesystem
  - `"aibrix"` → external AIBrix KVCache service
  - `"eic"` / `"simm"` → RDMA cluster stores
  - `"dynamic"` → user-registered custom backend
- `HiCacheController` (`managers/cache_controller.py:247`): the async coordinator thread; receives `write()`/`prefetch()` requests from `HiRadixCache`; manages the `write_storage()` pipeline (`cache_controller.py:998`) and the prefetch thread (`cache_controller.py:948`); rate-limits I/O with `prefetch_rate_limited()` (`cache_controller.py:906`)
- Prefetch policy: `"best_effort"` (default) — the controller issues reads for likely-needed storage pages while the GPU is computing; overlap hides most storage latency on NVMe; a full miss on spinning disk still causes a visible TTFT spike
- Write policy: `"write_through"` (default, always syncs to storage as eviction happens) vs `"write_back"` (lazy, risks loss on crash); most production deployments use `write_through` with a fast NVMe or NIXL backend

### 05 — Observability: What HiCache Reports (`05_observability.md`)
- Log line output: every scheduler logging interval prints `#cached-token: N` where N is the count of GPU-resident cached tokens; with HiCache enabled, the scheduler also logs load-back and eviction token counts per interval via `_log_hicache_stats()` (`observability/scheduler_metrics_mixin.py:694`)
- GPU pool metrics (always available):
  - `sglang:cache_hit_rate` — ratio of matched prefix tokens to total input tokens across all requests; the primary health indicator for all caching layers
  - `sglang:kv_used_tokens`, `sglang:kv_available_tokens`, `sglang:kv_evictable_tokens` — GPU pool token counts; defined in `metrics_collector.py` at lines 286–298
- Host pool metrics (HiCache tier 2, visible when `enable_hierarchical_cache=True`):
  - `sglang:hicache_host_used_tokens` — tokens currently occupying the CPU pinned buffer (`metrics_collector.py:673`)
  - `sglang:hicache_host_total_tokens` — total CPU buffer capacity (`metrics_collector.py:679`)
  - `sglang:evicted_tokens_total` — cumulative GPU→CPU token movements (evictions to host); counter (`metrics_collector.py:1639`)
  - `sglang:load_back_tokens_total` — cumulative CPU→GPU token movements (loads back to GPU); counter (`metrics_collector.py:1652`)
  - `sglang:load_back_duration_seconds` — histogram of CPU→GPU load latency; P99 reveals PCIe saturation (`metrics_collector.py:1645`)
- Storage I/O metrics (HiCache tier 3, visible when `hicache_storage_backend` is set):
  - `sglang:prefetched_tokens_total` — tokens moved from storage to CPU by prefetch thread (`metrics_collector.py:1476`)
  - `sglang:backuped_tokens_total` — tokens moved from CPU to storage during write pipeline (`metrics_collector.py:1482`)
  - Storage-side bandwidth and page counts tracked by `StorageMetricsCollector` (`metrics_collector.py:1466`)
- Grafana: the pre-built dashboard at `REPOS/sglang/examples/monitoring/grafana/dashboards/json/sglang-dashboard.json` includes a `Cache Hit Rate` panel (`sglang:cache_hit_rate`); add custom panels for host pool utilisation (`sglang:hicache_host_used_tokens / sglang:hicache_host_total_tokens`) and load-back rate (`rate(sglang:load_back_tokens_total[5m])`) to monitor tier-2 effectiveness
- Useful PromQL:
  ```promql
  # GPU cache hit rate
  sglang:cache_hit_rate

  # CPU pool fill percentage
  sglang:hicache_host_used_tokens / sglang:hicache_host_total_tokens * 100

  # Tokens evicted GPU → CPU per second
  rate(sglang:evicted_tokens_total[1m])

  # Tokens loaded back CPU → GPU per second
  rate(sglang:load_back_tokens_total[1m])

  # Tokens written to storage per second
  rate(sglang:backuped_tokens_total[1m])
  ```
- Key interpretation: if `sglang:load_back_tokens_total` grows while `sglang:cache_hit_rate` stays high, HiCache is working — prefix hits are occurring from CPU RAM rather than requiring full re-prefill; if both are low, requests are not sharing prefixes (no temporal locality)

### 06 — Configuration and the Router Relationship (`06_configuration.md`)
- Minimal enable (tier 2 only, CPU RAM):
  ```bash
  python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    --model meta-llama/Llama-3.1-8B-Instruct
  ```
  `hicache_ratio: 2.0` allocates a CPU pool with 2× the token capacity of the GPU pool; no storage backend — evictions stop at CPU
- Full three-tier enable (with local NVMe):
  ```bash
  python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    --hicache-storage-backend file \
    --hicache-storage-backend-extra-config '{"storage_dir": "/mnt/nvme/sglang_kvcache"}' \
    --model meta-llama/Llama-3.1-8B-Instruct
  ```
- All `server_args.py` HiCache flags (`server_args.py:565–573`):
  | Flag | Default | Effect |
  |---|---|---|
  | `--enable-hierarchical-cache` | `False` | Enables HiCache; switches `RadixCache` to `HiRadixCache` |
  | `--hicache-ratio` | `2.0` | CPU pool size = GPU pool size × ratio |
  | `--hicache-size` | `0` | Alternative: CPU pool size in bytes (overrides ratio) |
  | `--hicache-write-policy` | `"write_through"` | When to write evicted pages to storage: `write_through` or `write_back` |
  | `--hicache-io-backend` | `"kernel"` | DMA implementation for GPU↔CPU transfers |
  | `--hicache-mem-layout` | `"layer_first"` | KV page layout in CPU buffer |
  | `--hicache-storage-backend` | `None` | Tier-3 backend name (`file`, `nixl`, `mooncake`, `hf3fs`, etc.) |
  | `--hicache-storage-prefetch-policy` | `"best_effort"` | When the controller pre-fetches storage pages to CPU |
  | `--hicache-storage-backend-extra-config` | `None` | JSON string passed to the backend constructor |
- Relationship to the SGLang router: the cache-aware router (Layer 15/16) maintains its own approximate radix tree of request prefixes → worker URL mappings; it does not query the engine's HiCache state directly; the two layers compose: the router maximises the chance a request lands on the worker that already has its prefix in GPU VRAM (tier 1); if the router routes correctly but the GPU is full, HiCache ensures the KV tensors survive in CPU or storage and can be loaded back without a re-prefill; they are complementary, not redundant
- What this layer explicitly defers:
  - Mooncake `TransferEngine` (RDMA-based KV transfer for PD disaggregation) → Layer 18
  - `MooncakeStore` as a HiCache tier-3 backend (uses `mooncake.store.MooncakeDistributedStore`) → Layer 18 (cluster-level context required)
  - LMCache integration (`LMCRadixCache`, `lmc_radix_cache.py:63`) — a separate replacement for `RadixCache` using layer-wise streaming; not a HiCache tier → Layer 18
  - Decode-side KV offload (`DecodeKVCacheOffloadManager`) — specific to PD disaggregation decode servers → Layer 18

---

## Supporting Files

- `summary.md` — narrative walkthrough of all six sections with diagrams, code examples, and PromQL queries
- `01_eviction_problem.md` — deep dive on the cost of re-prefill and temporal locality in LLM workloads
- `02_three_tier_architecture.md` — HiRadixCache internals, node state machine, and data flow diagrams
- `03_host_pool.md` — `HostKVCache`, pinned memory, DMA transfer paths, and layout flags
- `04_storage_backend.md` — `HiCacheStorage` interface, `StorageBackendFactory`, `HiCacheController` pipeline
- `05_observability.md` — full metric inventory, PromQL queries, and Grafana setup guide
- `06_configuration.md` — flag reference, example launch commands, and router interaction explained

---

## Key Code Anchors

| Concept | Location |
|---|---|
| `enable_hierarchical_cache` flag | `REPOS/sglang/python/sglang/srt/server_args.py:565` |
| `hicache_ratio` flag | `REPOS/sglang/python/sglang/srt/server_args.py:566` |
| `hicache_storage_backend` flag | `REPOS/sglang/python/sglang/srt/server_args.py:571` |
| `HiRadixCache` class | `REPOS/sglang/python/sglang/srt/mem_cache/hiradix_cache.py:65` |
| `HiRadixCache.evict()` | `REPOS/sglang/python/sglang/srt/mem_cache/hiradix_cache.py:835` |
| `HiRadixCache.evict_host()` | `REPOS/sglang/python/sglang/srt/mem_cache/hiradix_cache.py:905` |
| `HiRadixCache.load_back()` | `REPOS/sglang/python/sglang/srt/mem_cache/hiradix_cache.py:940` |
| `HostKVCache` base class | `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool_host.py:154` |
| `MHATokenToKVPoolHost` | `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool_host.py:290` |
| `MLATokenToKVPoolHost` | `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool_host.py:787` |
| `backup_from_device_all_layer` (GPU→CPU) | `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool_host.py:513` |
| `load_to_device_per_layer` (CPU→GPU) | `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool_host.py:396` |
| `HiCacheStorage` abstract class | `REPOS/sglang/python/sglang/srt/mem_cache/hicache_storage.py:98` |
| `HiCacheStorageConfig` | `REPOS/sglang/python/sglang/srt/mem_cache/hicache_storage.py:17` |
| `HiCacheFile` (local disk backend) | `REPOS/sglang/python/sglang/srt/mem_cache/hicache_storage.py:277` |
| `StorageBackendFactory` | `REPOS/sglang/python/sglang/srt/mem_cache/storage/backend_factory.py:16` |
| `HiCacheController` | `REPOS/sglang/python/sglang/srt/managers/cache_controller.py:247` |
| `HiCacheController.write()` | `REPOS/sglang/python/sglang/srt/managers/cache_controller.py:663` |
| `HiCacheController.prefetch()` | `REPOS/sglang/python/sglang/srt/managers/cache_controller.py:799` |
| `HiCacheController.write_storage()` | `REPOS/sglang/python/sglang/srt/managers/cache_controller.py:998` |
| `_log_hicache_stats()` | `REPOS/sglang/python/sglang/srt/observability/scheduler_metrics_mixin.py:694` |
| `sglang:cache_hit_rate` metric | `REPOS/sglang/python/sglang/srt/observability/metrics_collector.py:272` |
| `sglang:hicache_host_used_tokens` | `REPOS/sglang/python/sglang/srt/observability/metrics_collector.py:673` |
| `sglang:evicted_tokens_total` | `REPOS/sglang/python/sglang/srt/observability/metrics_collector.py:1639` |
| `sglang:load_back_tokens_total` | `REPOS/sglang/python/sglang/srt/observability/metrics_collector.py:1652` |
| `sglang:prefetched_tokens_total` | `REPOS/sglang/python/sglang/srt/observability/metrics_collector.py:1476` |
| `StorageMetricsCollector` | `REPOS/sglang/python/sglang/srt/observability/metrics_collector.py:1466` |
| `RadixCacheMetricsCollector` | `REPOS/sglang/python/sglang/srt/observability/metrics_collector.py:1573` |
| Grafana dashboard JSON | `REPOS/sglang/examples/monitoring/grafana/dashboards/json/sglang-dashboard.json` |
| `LMCRadixCache` (not a HiCache tier) | `REPOS/sglang/python/sglang/srt/mem_cache/storage/lmcache/lmc_radix_cache.py:63` |
