# Layer 17 — Summary

Layer 17 extends SGLang's GPU KV cache with a hierarchical spillover system called **HiCache**. Where Layer 12's `RadixCache` permanently discards evicted KV tensors to free GPU memory, HiCache writes them to a lower-cost tier first — pinned CPU RAM, then optionally local disk or a remote store — and loads them back when the same prefix is requested again. From the scheduler's perspective, a warm re-request looks like a cache hit regardless of whether the pages were in GPU VRAM, CPU RAM, or on disk.

---

## From Layer 12 to Layer 17

Layer 12's `RadixCache` had one rule for a full GPU pool: evict the least-recently-used node and free its KV pages.

```
GPU pool full → RadixCache.evict() → KV tensors freed → gone
```

For short prompts that change every request, this is fine. For multi-turn chat or RAG workloads where the same 10 k-token system prompt is used by thousands of requests over hours, evicting it means the next request pays a full prefill penalty — 2–10 seconds of GPU compute on an A100 — to rebuild something that was already computed.

HiCache changes the eviction path:

```
GPU pool full → HiRadixCache.evict() → KV tensors copied to CPU RAM → node is "cold"
                                        CPU pool full → write to HiCacheStorage → node is "stored"

Future request, same prefix:
  GPU hit         → no stall
  CPU ("cold")    → load_back() copies pages GPU-ward → ~milliseconds
  Storage         → storage.read() → CPU → GPU-ward → ~tens to hundreds of milliseconds
  Complete miss   → full prefill
```

The eviction threshold and load-back are both invisible to the model runner. The GPU KV pool always looks like a normally cached pool.

---

## The Three-Tier Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Tier 1 — GPU VRAM                                         │
│  MHATokenToKVPool / MLATokenToKVPool                       │
│  Hot KV pages, ~24–80 GB, directly accessed by attention   │
└──────────────────────── evict_host() ──────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────┐
│  Tier 2 — CPU RAM (Host Pool)                              │
│  MHATokenToKVPoolHost / MLATokenToKVPoolHost               │
│  Pinned memory, ~128–512 GB, DMA-accessible                │
└──────────────────────── backup_storage() ──────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────┐
│  Tier 3 — Storage (HiCacheStorage)                         │
│  HiCacheFile / NixlStore / MooncakeStore / hf3fs / …       │
│  TB-scale, local NVMe / RDMA network / object storage      │
└────────────────────────────────────────────────────────────┘
```

`HiRadixCache` (`hiradix_cache.py:65`) extends `RadixCache` and coordinates the three tiers. Each radix-tree node carries a `cache_state` field: `GPU`, `CPU`, `STORAGE`, or `MIXED`. The scheduler checks this state when a prefix match is found and determines whether to proceed immediately or stall for a load.

---

## Tier 2: The Host Pool

Pinned (page-locked) CPU memory is the key property of tier 2. Unlike normal heap memory, pinned memory is not swappable by the OS and can be read by the GPU's DMA engine without an intermediate staging copy. This means GPU↔CPU transfers happen at near full PCIe bandwidth.

**GPU → CPU (eviction):** `backup_from_device_all_layer()` (`memory_pool_host.py:513`) copies all KV layers for a set of token pages from the GPU pool to the CPU buffer. This runs on a dedicated CUDA stream — it does not block in-flight requests.

**CPU → GPU (load-back):** `load_to_device_per_layer()` (`memory_pool_host.py:396`) copies the CPU pages back to the GPU pool per layer. The scheduler stalls the request while this completes. On a PCIe-4 system, loading a 2 k-token page set takes roughly 10–50 ms depending on model size.

Two implementations handle different attention architectures:
- `MHATokenToKVPoolHost` (`memory_pool_host.py:290`): standard multi-head attention; stores full K and V tensors.
- `MLATokenToKVPoolHost` (`memory_pool_host.py:787`): Multi-head Latent Attention (e.g. DeepSeek); stores compressed KV representations, so the host pool holds proportionally more tokens.

The CPU pool size is controlled by `--hicache-ratio` (default `2.0`, meaning the CPU pool holds 2× the token capacity of the GPU pool) or overridden with `--hicache-size` in bytes.

---

## Tier 3: The Storage Backend

`HiCacheStorage` (`hicache_storage.py:98`) is an abstract base class with two methods: `write(pool_transfer)` and `read(pool_transfer)`. A `PoolTransfer` (`hicache_storage.py`) describes which CPU buffer pages to move and where; implementations DMA from/to the pinned buffer directly without an extra copy.

`StorageBackendFactory` (`storage/backend_factory.py:16`) maps backend name strings to classes. Available backends:

| Name | Class | Use case |
|---|---|---|
| `file` | `HiCacheFile` | Local NVMe or HDD; reference implementation |
| `nixl` | `NixlStore` | Network RDMA or object storage via NIXL agent |
| `mooncake` | `MooncakeStore` | Distributed shared memory via Mooncake distributed store |
| `hf3fs` | `HF3FSStore` | HuggingFace 3FS distributed filesystem |
| `aibrix` | `AIBrixStore` | External AIBrix KVCache service |
| `eic` | `EICStore` | RDMA cluster store with NIC affinity |
| `simm` | `SiMMStore` | SiMM RDMA cluster store |
| `dynamic` | user class | Custom user-registered backend |

`HiCacheController` (`managers/cache_controller.py:247`) runs as a background thread. It receives write and prefetch requests from `HiRadixCache`, manages the `write_storage()` pipeline (`cache_controller.py:998`), and runs the prefetch thread (`cache_controller.py:948`). The prefetch logic reads likely-needed storage pages into the CPU pool while the GPU is computing, overlapping storage I/O with computation.

Write policies:
- `"write_through"` (default): writes evicted pages to storage as eviction happens; durable but adds latency to the eviction path.
- `"write_back"`: lazy writes; lower eviction latency, but pages in the CPU pool that have not yet been flushed will be lost if the process crashes.

---

## Observability

### What You See Without HiCache (Layer 12 baseline)

Every scheduler tick prints one line:

```
#reqs: 12 | throughput: 43.1 tok/s | #cached-token: 8192 | #running: 12
```

`#cached-token` counts GPU-resident cached tokens. When the pool evicts, this number drops and TTFT spikes on the next prefix-sharing request.

### What You See With HiCache

Three additional metrics appear in logs: the number of tokens evicted to CPU, loaded back from CPU, and written to / prefetched from storage per interval. These are populated by `_log_hicache_stats()` (`observability/scheduler_metrics_mixin.py:694`).

### Prometheus Metrics

| Metric | Type | Meaning |
|---|---|---|
| `sglang:cache_hit_rate` | Gauge | Fraction of input tokens matched from any GPU-resident prefix |
| `sglang:kv_used_tokens` | Gauge | GPU KV pool tokens in use |
| `sglang:kv_available_tokens` | Gauge | GPU KV pool tokens free |
| `sglang:hicache_host_used_tokens` | Gauge | CPU pool tokens in use (tier 2) |
| `sglang:hicache_host_total_tokens` | Gauge | CPU pool total capacity (tier 2) |
| `sglang:evicted_tokens_total` | Counter | Cumulative tokens moved GPU → CPU |
| `sglang:load_back_tokens_total` | Counter | Cumulative tokens moved CPU → GPU |
| `sglang:load_back_duration_seconds` | Histogram | CPU → GPU load latency |
| `sglang:prefetched_tokens_total` | Counter | Tokens moved storage → CPU by prefetch |
| `sglang:backuped_tokens_total` | Counter | Tokens moved CPU → storage by write pipeline |

Useful PromQL queries:

```promql
# GPU cache hit rate
sglang:cache_hit_rate

# CPU pool fill percentage (watch for sustained >90%)
sglang:hicache_host_used_tokens / sglang:hicache_host_total_tokens * 100

# Eviction rate: GPU → CPU tokens per second
rate(sglang:evicted_tokens_total[1m])

# Load-back rate: CPU → GPU tokens per second
rate(sglang:load_back_tokens_total[1m])

# Storage backup rate
rate(sglang:backuped_tokens_total[1m])

# P99 load-back latency
histogram_quantile(0.99, rate(sglang:load_back_duration_seconds_bucket[5m]))
```

### Interpreting the Numbers

| Observation | Interpretation |
|---|---|
| `load_back_tokens_total` grows, `cache_hit_rate` stays high | HiCache working: prefix hits served from CPU instead of re-prefill |
| `cache_hit_rate` low, `evicted_tokens_total` high | Requests do not share prefixes; HiCache buys little; check workload |
| `hicache_host_used_tokens / total` near 100%, eviction rate high | CPU pool undersized; increase `--hicache-ratio` or add more host RAM |
| `load_back_duration_seconds` P99 > 200 ms | PCIe saturation or NUMA issues; check host topology |
| `prefetched_tokens_total` growing, latency flat | Storage prefetch is hiding tier-3 read latency effectively |

### Grafana

The pre-built dashboard at `REPOS/sglang/examples/monitoring/grafana/dashboards/json/sglang-dashboard.json` includes a `Cache Hit Rate` panel. Add three custom panels to monitor HiCache specifically:

1. **Host pool utilisation** — `sglang:hicache_host_used_tokens / sglang:hicache_host_total_tokens * 100` — gauge, alert threshold at 90%.
2. **Tier-2 throughput** — `rate(sglang:evicted_tokens_total[1m])` and `rate(sglang:load_back_tokens_total[1m])` on the same graph — the gap between eviction rate and load-back rate shows how much cache churn is happening.
3. **Load-back latency P99** — histogram_quantile on `sglang:load_back_duration_seconds`; spikes here directly predict TTFT regression.

Setup instructions for the full monitoring stack (Prometheus + Grafana via Docker Compose) are in `REPOS/sglang/examples/monitoring/README.md`.

---

## Configuration

### Tier 2 only (CPU RAM, no storage)

```bash
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0
```

This is the minimal configuration. The CPU pool will hold 2× the GPU pool's token capacity. No storage backend is configured, so evictions stop at the CPU layer.

### Three tiers (CPU + local NVMe)

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

### Three tiers (CPU + NIXL remote store)

```bash
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-storage-backend nixl \
  --hicache-storage-backend-extra-config '{"nixl_agent_url": "http://nixl-agent:8080"}'
```

### Full flag reference (`server_args.py:565–573`)

| Flag | Default | Effect |
|---|---|---|
| `--enable-hierarchical-cache` | `False` | Enables HiCache; switches from `RadixCache` to `HiRadixCache` |
| `--hicache-ratio` | `2.0` | CPU pool tokens = GPU pool tokens × ratio |
| `--hicache-size` | `0` | Override: CPU pool in bytes (ignores ratio when set) |
| `--hicache-write-policy` | `"write_through"` | `write_through` or `write_back` |
| `--hicache-io-backend` | `"kernel"` | DMA backend for GPU↔CPU transfers |
| `--hicache-mem-layout` | `"layer_first"` | KV page layout in CPU buffer; `page_first_direct` enables a faster kernel path |
| `--hicache-storage-backend` | `None` | Tier-3 backend name; `None` means no tier 3 |
| `--hicache-storage-prefetch-policy` | `"best_effort"` | `best_effort` or `none` |
| `--hicache-storage-backend-extra-config` | `None` | JSON string passed to backend constructor |

---

## Relationship to the SGLang Router

The cache-aware router (Layers 15–16) and HiCache operate at different layers and compose cleanly.

The **router** maintains an approximate radix tree mapping raw text prefixes to worker URLs. When a request arrives, the router finds the worker whose tree contains the longest matching prefix and sends the request there. This maximises the chance the target engine has the prefix in GPU VRAM (tier 1). The router never queries the engine's internal KV cache state.

**HiCache** is internal to each engine. If the router routes correctly but the GPU pool is full and the relevant node has been evicted, HiCache serves it from CPU or storage instead of requiring a full re-prefill. If the router mis-routes (e.g. the session's prefix is on a different worker), HiCache on the wrong worker cannot help — a cold re-prefill is needed regardless.

The practical consequence: router cache-awareness and HiCache are both worth enabling, but they solve different parts of the problem. Router cache-awareness reduces cross-worker prefix mismatches. HiCache reduces the cost of within-worker evictions. With both enabled, prefix locality is maximised at the routing layer and eviction penalties are minimised at the cache layer.

---

## What Layer 17 Explicitly Does Not Cover

| Topic | Where |
|---|---|
| Mooncake `TransferEngine` for RDMA PD disaggregation | Layer 18 |
| `MooncakeStore` as a HiCache tier-3 backend | Layer 18 (cluster-level setup required) |
| `LMCRadixCache` — layer-wise streaming alternative to `RadixCache` | Layer 18 |
| `DecodeKVCacheOffloadManager` — decode-side offload in PD disaggregation | Layer 18 |
| Multi-node KV cache sharing or global KV index | Layer 18 / future |
