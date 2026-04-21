# 06 — Configuration and the Router Relationship

## What This Section Covers

This section shows how to enable HiCache, explains every flag, gives three reference launch commands for common deployment patterns, and explains how HiCache and the SGLang router (Layers 15–16) relate to each other at runtime.

---

## All HiCache Flags

HiCache flags are defined in `ServerArgs` (`server_args.py:564–573`). All flags are optional; only `--enable-hierarchical-cache` must be set explicitly to turn the feature on.

| Flag | Default | Type | Effect |
|---|---|---|---|
| `--enable-hierarchical-cache` | `False` | bool | Master switch. Replaces `RadixCache` with `HiRadixCache` and allocates the CPU pinned buffer at startup. |
| `--hicache-ratio` | `2.0` | float | CPU pool token capacity = GPU pool token capacity × ratio. Ignored if `--hicache-size` is non-zero. |
| `--hicache-size` | `0` | int | Alternative sizing: exact CPU pool in bytes (as integer). 0 = use ratio. |
| `--hicache-write-policy` | `"write_through"` | str | `"write_through"`: GPU→CPU copy starts immediately when a node is accessed; slot freed after copy completes. `"write_back"`: slot freed immediately; copy happens lazily. |
| `--hicache-io-backend` | `"kernel"` | str | DMA implementation for GPU↔CPU transfers: `"kernel"` (Triton/CUDA, default), `"direct"`, `"kernel_ascend"` (Ascend NPU). |
| `--hicache-mem-layout` | `"layer_first"` | str | Buffer stride layout: `"layer_first"` (best for GPU↔CPU), `"page_first"`, `"page_first_direct"` (best for CPU↔NVMe), `"page_head"`. |
| `--hicache-storage-backend` | `None` | str | Tier-3 backend name. `None` = no tier 3. Options: `file`, `nixl`, `mooncake`, `hf3fs`, `aibrix`, `eic`, `simm`, `dynamic`. |
| `--hicache-storage-prefetch-policy` | `"best_effort"` | str | `"best_effort"`: prefetch thread loads storage pages speculatively. `"none"`: no prefetch, cold storage hits always stall. |
| `--hicache-storage-backend-extra-config` | `None` | str | JSON string passed verbatim to the backend constructor as `extra_config`. Backend-specific options go here. |

---

## Recipe 1: Tier 2 Only (CPU RAM, No Storage)

The simplest configuration. Useful on any node with substantial RAM. Adds a CPU spillover buffer with no external dependencies.

```bash
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0
```

What happens at startup:
1. GPU KV pool is sized as usual (based on `--mem-fraction-static`).
2. `MHATokenToKVPoolHost` allocates `GPU_pool_tokens × 2.0` tokens of pinned CPU RAM — SGLang logs: `Allocating X.XX GB host memory for hierarchical KV cache.`
3. `RadixCache` is replaced with `HiRadixCache`. No storage backend is created.
4. When the GPU pool fills, evicted nodes' KV pages are written to the CPU pool on a background CUDA stream. Subsequent prefix matches for those nodes trigger a CPU→GPU load.

**Memory check**: for a Llama-3.1-8B model with a 10 k-token GPU pool, `hicache_ratio=2.0` allocates ~10 GB of pinned RAM. Verify your machine has sufficient free RAM before starting:

```bash
free -h   # look at "available" column
# 10 GB pinned + OS reserve ~2 GB = need at least 12 GB available
```

---

## Recipe 2: Three Tiers — CPU + Local NVMe

Adds a persistent disk backend. Useful on single nodes with fast NVMe where GPU and CPU are both full.

```bash
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-storage-backend file \
  --hicache-storage-backend-extra-config '{"storage_dir": "/mnt/nvme/sglang_kvcache"}' \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy best_effort
```

The `storage_dir` must be writable. SGLang creates subdirectories for each tensor-parallel rank under this path. With `write_through`, pages are written to disk immediately when evicted from the CPU pool — if the process restarts, the disk pages are still there and can be loaded without re-prefill.

Disk space required: `CPU_pool_bytes` (the disk eventually mirrors the CPU pool at steady state) plus any write-ahead for tokens being evicted. For a 10 GB CPU pool, plan for 15–20 GB of disk headroom.

---

## Recipe 3: Three Tiers — CPU + NIXL Remote Store

For multi-node or disaggregated setups where KV cache should survive node restarts or be shared across instances.

```bash
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-storage-backend nixl \
  --hicache-storage-backend-extra-config '{"nixl_agent_url": "http://nixl-agent:8080"}' \
  --hicache-mem-layout page_first_direct
```

`page_first_direct` is recommended with NIXL: it lays out CPU pages contiguously per KV-page (all layers for one page together), which reduces the number of DMA operations the NIXL agent needs to issue to storage.

---

## Choosing hicache-ratio

A conservative starting point is `2.0` (default). Rules of thumb:

- If `sglang:hicache_host_used_tokens / hicache_host_total_tokens` is sustained near 100%, the CPU pool is undersized — increase the ratio or add RAM.
- If `sglang:evicted_tokens_total` grows fast but `sglang:load_back_tokens_total` stays flat, evictions are not being reused — the workload has low temporal locality; a higher ratio just wastes RAM.
- If `sglang:load_back_duration_seconds` P99 is above 200 ms, PCIe bandwidth is saturated — reducing the pool size will not help; investigate NUMA topology instead.

Maximum practical ratio is limited by available physical RAM. Do not set a ratio that would exceed `psutil.virtual_memory().available - 2 GB` — SGLang will refuse to start and print a clear error.

---

## Choosing hicache-write-policy

**`write_through`** (default): the GPU slot is held until the async CPU copy completes. This means the GPU pool has slightly reduced effective throughput when eviction is heavy — evicted nodes hold their GPU slots for the duration of the DMA. In practice, PCIe DMA is fast (~1 ms for typical page sizes) and the throughput impact is negligible.

**`write_back`**: the GPU slot is freed immediately on eviction. If the node is accessed again before the CPU copy completes, it must be re-prefilled (the CPU pages are not yet ready). This policy reduces eviction latency at the cost of a data loss window if the process crashes mid-copy. Only use it for non-critical, reproducible workloads.

For any deployment with a storage backend, always use `write_through` — disk writes happen after the CPU copy, so a crash during the CPU→disk phase would lose data if the CPU pages are not durable.

---

## Relationship to the SGLang Router

The cache-aware router (Layers 15–16) and HiCache operate independently at different layers. Understanding how they compose is important for setting expectations correctly.

### What the Router Does

The router maintains an **approximate radix tree** of raw text prefixes mapped to worker URLs. When a request arrives, the router finds the worker whose tree has the longest matching prefix and sends the request there. The router's goal is to maximise the probability that the target worker has the relevant prefix in GPU VRAM (tier 1). The router **never queries the engine's HiCache state** — it does not know whether a prefix is GPU-warm, CPU-cold, or in storage.

### What HiCache Does

HiCache is internal to each engine instance. It runs entirely inside the scheduler/cache controller after the router has already chosen which worker to send the request to. When the selected worker receives the request, HiCache's `match_prefix()` is called, and the result determines whether to serve from GPU, load from CPU, or load from storage.

### How They Compose

```
Client
  │
  ▼
Router (cache-aware policy)
  │ Finds worker with longest matching prefix in router's approximate tree
  │ Sends request to that worker
  ▼
Worker N — SGLang engine
  │
  ▼ HiRadixCache.match_prefix()
      GPU hit       → serve immediately (tier 1)
      CPU hit       → load_back() → serve (~10–100 ms stall)
      Storage hit   → prefetch → load → serve (~100–500 ms stall)
      Complete miss → full prefill
```

**Router hit + GPU hit (best case)**: router sends to the right worker, prefix is in GPU VRAM. No stall. This is the common case when the system is not under memory pressure.

**Router hit + CPU/storage hit (HiCache benefit)**: router sends to the right worker, but the prefix was evicted from GPU. Without HiCache, this would require full re-prefill. With HiCache, the prefix is loaded back — stall is 10–500 ms instead of 1–25 seconds.

**Router miss**: router sends to the wrong worker (e.g. with multiple router replicas, or after a worker restart). The wrong worker's HiCache has no record of the prefix — full re-prefill required regardless of HiCache configuration.

### Practical Consequence

Enable both. They solve different parts of the problem:
- **Router cache-awareness** reduces the probability of a router miss — keeps requests on the worker that has seen the prefix before.
- **HiCache** reduces the cost of an eviction on the right worker — turns a 5-second re-prefill into a 50 ms load-back.

With both disabled: every eviction is a full re-prefill; cross-request prefix reuse depends on the GPU pool staying warm.
With only cache-aware routing: prefixes route correctly but still pay full re-prefill when the GPU pool overflows.
With only HiCache: CPU and storage absorb overflow but the router may still route to the wrong worker.
With both: optimal for long-context, multi-turn, and RAG workloads.

---

## What Layer 17 Does Not Cover

These topics are deferred to Layer 18 to keep this layer focused on the core tier-1/2/3 caching mechanism.

| Topic | Reason for deferral |
|---|---|
| Mooncake `TransferEngine` (RDMA PD disaggregation) | Separate feature; requires understanding of prefill-decode split |
| `MooncakeStore` as a HiCache tier-3 backend | Needs Mooncake cluster setup context |
| `LMCRadixCache` (`lmc_radix_cache.py:63`) | An alternative to `RadixCache` using layer-wise streaming; not a HiCache tier |
| `DecodeKVCacheOffloadManager` | Specific to decode servers in PD disaggregation mode |
| NSA (Native Sparse Attention) hybrid stack | Advanced attention variant with a different HiCache integration path |

---

## Key Files Referenced

| File | What it shows |
|---|---|
| `REPOS/sglang/python/sglang/srt/server_args.py:564` | All HiCache `ServerArgs` fields |
| `REPOS/sglang/python/sglang/srt/mem_cache/hiradix_cache.py:65` | `HiRadixCache.__init__` — host pool construction from `server_args` |
| `REPOS/sglang/python/sglang/srt/managers/cache_controller.py:247` | `HiCacheController.__init__` — write policy and io_backend wired in |
| `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool_host.py:154` | `HostKVCache.__init__` — pool sizing and safety check |
| `REPOS/sglang/python/sglang/srt/mem_cache/storage/backend_factory.py:16` | `StorageBackendFactory` — backend selection |
| `REPOS/sglang/sgl-model-gateway/src/policies/cache_aware.rs` | Router's approximate radix tree — confirms router does not query engine HiCache state |
