# HiCache System Design and Optimization

**Source:** https://docs.sglang.io/advanced_features/hicache_design.html
**Author:** SGLang Team
**Level:** L3 — Technical design reference
**Why here:** The authoritative SGLang documentation for HiCache architecture. Covers HiRadixTree design, all three workflow phases (local match → prefetch → write-back), data transfer optimizations, multi-rank synchronization, and every configuration parameter with precise semantics. Layer 17 lessons `02_three_tier_architecture.md`, `03_host_pool.md`, `04_storage_backend.md`, and `06_configuration.md` all draw their technical detail from this document.

---

## Why and What is HiCache?

In large language model inference, the prefill phase is time-consuming: input sequences are converted into Key-Value cache (KV cache) for decoding. When multiple requests share the same prefix, the KV cache is identical. By caching and reusing shared KV caches, redundant computation is avoided.

**RadixAttention** leverages idle GPU memory for prefix KV cache reuse. **HiCache** extends this to host memory and distributed storage, organized as a classic three-level cache:

- **L1** — GPU VRAM (fastest, smallest)
- **L2** — Host memory / CPU DRAM (medium)
- **L3** — Distributed storage (Mooncake, 3FS, NIXL, file, AIBrix)

L1 and L2 are **private** to each inference instance. L3 is **shared** across all instances in the cluster.

---

## System Architecture

### HiRadixTree: Metadata Organization

HiCache extends RadixAttention's `RadixTree` with a **HiRadixTree**. Each node:
- Corresponds to a span of consecutive tokens
- Records **where** that KV cache is stored: GPU, CPU, L3 storage, or multiple tiers
- For local tiers (L1/L2): maintains precise storage addresses
- For L3: does **not** store or continuously synchronize metadata — queries backend in real time when needed

### Workflow

Three key operations:

1. **Local Match** — search HiRadixTree for tokens in L1/L2; extremely fast (no data copying, tree traversal only). Returns: first part in L1, second part in L2.
2. **Prefetch from L3** — for misses, query L3 for next continuous KV chunk. If hit length > threshold (default 256 tokens), trigger prefetch into L2.
3. **Write-back** — move frequently accessed KV caches from L1 → L2 → L3 for persistence and cross-instance sharing.

---

## Local Match

- Traverses HiRadixTree from root, following child nodes matching the token prefix.
- When `page_size > 1`: matching at **page granularity** (not token granularity) for I/O efficiency.
- If match terminates mid-node: the node is **automatically split** to create an exact boundary, improving future matches.
- Returns a continuous prefix with L1 part first, L2 part second.

---

## Prefetch from L3

**Trigger condition**: after local match, query L3 for remaining tokens. If L3 hit length > threshold → trigger prefetch.

**Three prefetch termination strategies:**

| Policy | Behavior | Best for |
|---|---|---|
| `best_effort` | Terminate immediately when GPU can start prefill | Latency-sensitive workloads |
| `wait_complete` | Block until all prefetch completes | Maximum cache hit rate |
| `timeout` | Wait up to computed timeout, then proceed | Production (balanced SLO) |

**Timeout formula:**
```
timeout = prefetch_timeout_base + prefetch_timeout_per_ki_token × num_token_to_fetch / 1024
```

After prefetch stops, already-fetched data is used with local data for prefill computation.

---

## Data Write-back

**Three write-back policies:**

| Policy | When data is written to next tier | Best for |
|---|---|---|
| `write_through` | Immediately on every access | Highest cache hit rate (if bandwidth permits) |
| `write_through_selective` | After access frequency exceeds threshold | Reduces I/O overhead; backs up only "hot" data |
| `write_back` | Only when evicted from upper tier | Capacity-constrained environments |

**Cross-instance sharing**: when data moves L2 → L3, only data **not already in L3** is transferred. This enables all SGLang instances in the cluster to benefit from cached prefixes.

---

## Multi-Rank Synchronization (Tensor Parallelism)

During TP, HiCache uses `all_reduce` to synchronize state across ranks:
- **During prefetch**: `all_reduce(op=min)` ensures all ranks agree on L3 hit count, preventing inconsistent prefetch threshold decisions.
- **After prefetch completes/terminates**: `all_reduce(op=min)` guarantees consensus on successfully retrieved KV cache prefix length.

**MLA optimization**: For MLA (Multi-Layer Attention) models, all ranks hold the same complete KV data per token. HiCache uses only **one rank** for write-back to avoid redundant storage across ranks.

---

## Data Transfer Optimizations

### Zero-Copy Transfers
Both prefetch and write-back pass **memory addresses and sizes** directly to the L3 backend rather than staging through an intermediate buffer.

### "Batch-Oriented" Data Organization
Three memory layouts:

| Layout | Description | Best for |
|---|---|---|
| `layer_first` | Layer dimension outermost; GPU's native layout | GPU computation kernels |
| `page_first` | Page dimension outermost; all KV data for a page contiguous | I/O efficiency to/from L3 |
| `page_first_direct` | `page_first` with layer sub-grouping within each page | Aggregated L2→GPU transfers at page-layer granularity |

GPU remains `layer_first` (unchanged). CPU (L2) and L3 use `page_first` or `page_first_direct` for larger contiguous transfers.

### CPU-to-GPU Transfer Optimizations
1. **Compute-Transfer Overlap**: during prefill, load KV cache for layer N+1 while computing layer N — hides transfer latency.
2. **GPU-assisted I/O Kernels**: custom kernels specifically optimized for KV cache transfers between CPU and GPU; up to **3× higher transfer speed** vs `cudaMemcpyAsync` alone.

---

## All Configuration Parameters

| Flag | Default | Description |
|---|---|---|
| `--enable-hierarchical-cache` | off | Enable HiCache (required) |
| `--hicache-ratio RATIO` | — | Host KV pool size = RATIO × GPU KV pool size. Must be > 1. |
| `--hicache-size SIZE_GB` | — | Host KV pool size in GB (overrides ratio). Must be > GPU KV size. |
| `--page-size N` | 16 | Tokens per page. Larger = better I/O efficiency, lower hit rate for diverse prefixes. |
| `--hicache-storage-prefetch-policy` | best_effort | `best_effort`, `wait_complete`, or `timeout` |
| `--hicache-write-policy` | write_back | `write_back`, `write_through`, or `write_through_selective` |
| `--hicache-io-backend` | direct | `direct` (cudaMemcpyAsync) or `kernel` (GPU-assisted, recommended) |
| `--hicache-mem-layout` | layer_first | `layer_first`, `page_first`, or `page_first_direct` |
| `--hicache-storage-backend` | — | `file`, `mooncake`, `hf3fs`, `nixl`, `aibrix`, `dynamic` |
| `--enable-lmcache` | off | Use LMCache as alternative to HiCache |
| `--hicache-storage-backend-extra-config` | — | JSON string or `@file.toml` with backend-specific config |

**Prefetch timeout config** (via `--hicache-storage-backend-extra-config`):
```json
{
  "prefetch_threshold": 256,
  "prefetch_timeout_base": 0.5,
  "prefetch_timeout_per_ki_token": 0.25
}
```

---

## Unified Storage Backend Interface

All L3 backends implement `class HiCacheStorage(ABC)`:
- `batch_exists_v2(keys)` — check which pages are in L3
- `batch_get_v2(keys, dst)` — fetch pages from L3 into L2 buffer
- `batch_set_v2(keys, src)` — write pages from L2 into L3

This interface enables:
- **Mooncake**: RDMA zero-copy distributed memory
- **HF3FS (3FS)**: Kubernetes-native distributed storage
- **NIXL**: Unified API for GDS, 3FS, S3-compatible object storage
- **AIBrix KVCache**: production-ready memory tiering / cross-engine reuse
- **File**: local disk (reference implementation)
- **Dynamic**: custom Python class loaded at runtime (`backend_name`, `module_path`, `class_name`)

---

## Integration with PD-Disaggregation

HiCache can be enabled on both **prefill nodes** and **decode nodes** in PD disaggregation deployments:
- Prefill nodes: HiCache accelerates KV lookups for shared prefixes.
- Decode nodes: decode output is written back to L3 if HiCache is enabled.

The Mooncake storage backend and the Mooncake TransferEngine (used for PD KV transfer) are **separate subsystems** — they coexist independently.

---

## Key Takeaways for Layer 17

- `HiRadixTree` is the metadata layer; it does not own data, it tracks where data lives.
- **Local match is instant** (tree traversal, no copies); **prefetch is async** (cache controller in background).
- **Page-first layout** + **zero-copy** are the two main I/O optimizations for L3 backends.
- **Compute-transfer overlap** for L2→GPU is the key latency-hiding technique for CPU tier.
- **Multi-rank sync via `all_reduce`** is the TP correctness mechanism.
- `hicache-ratio` and `page-size` are the two parameters with the most impact to tune first.
