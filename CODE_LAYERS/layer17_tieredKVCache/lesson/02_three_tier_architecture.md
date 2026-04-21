# 02 — The Three-Tier Architecture

## What This Section Covers

HiCache adds two storage tiers below GPU VRAM. This section describes each tier, the `HiRadixCache` class that coordinates them, and the data flow a request follows from arrival to completion — including what happens when its prefix is found in a lower tier.

---

## The Three Tiers at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│  Tier 1 — GPU VRAM                                              │
│  MHATokenToKVPool / MLATokenToKVPool                            │
│  • Hot KV pages, directly accessed by attention kernels         │
│  • Typical capacity: 10–40 GB (model-size and batch-size dep.)  │
│  • Access latency: nanoseconds (on-chip)                        │
└──────────────────────── evict_host() ───────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tier 2 — CPU RAM (Host Pool)                                   │
│  MHATokenToKVPoolHost / MLATokenToKVPoolHost                    │
│  • Pinned (page-locked) memory, DMA-accessible by GPU           │
│  • Typical capacity: 128–512 GB                                 │
│  • Access latency: ~10–100 ms (PCIe bandwidth bound)            │
└──────────────────────── write_storage() ────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tier 3 — Storage (HiCacheStorage)                              │
│  HiCacheFile / HiCacheNixl / MooncakeStore / HiCacheHF3FS / …  │
│  • Local NVMe, remote RDMA memory, or object storage            │
│  • Typical capacity: TB scale                                   │
│  • Access latency: ~50–500 ms (I/O and network bound)           │
└─────────────────────────────────────────────────────────────────┘
```

Tier 1 is the existing GPU KV pool from Layer 12 — no change to the attention kernels or the GPU memory allocator. Tier 2 and Tier 3 are new, and both are optional: you can run HiCache with only tier 2 (no storage backend configured), or with all three tiers.

---

## HiRadixCache: The Coordinator

`HiRadixCache` (`hiradix_cache.py:65`) extends `RadixCache` from Layer 12. The inheritance means all the prefix-tree logic (token hashing, node insertion, LRU eviction priority) is carried over unchanged. What `HiRadixCache` adds is:

1. **A host pool** (`token_to_kv_pool_host`): an instance of `MHATokenToKVPoolHost` or `MLATokenToKVPoolHost`, created in `__init__` based on the attention architecture detected from the GPU pool type.
2. **A cache controller** (`cache_controller`): an instance of `HiCacheController`, which runs background threads for async CPU↔storage I/O.
3. **Extended node state**: each `TreeNode` in the tree gains fields like `evicted` (whether GPU pages have been freed), `backuped` (whether CPU pages are live), `host_value` (CPU buffer indices), and `host_ref_counter`.

```python
# hiradix_cache.py:67 — HiRadixCache.__init__
class HiRadixCache(RadixCache):
    def __init__(self, params, server_args):
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,    # CPU pool = GPU pool × ratio
                server_args.hicache_size,     # or exact bytes
                self.page_size,
                server_args.hicache_mem_layout,
                ...
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(...)

        self.cache_controller = HiCacheController(
            ...,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
        )
```

Both `MHATokenToKVPool` (standard multi-head attention) and `MLATokenToKVPool` (multi-head latent attention, e.g. DeepSeek) are supported. The NSA (Native Sparse Attention) path uses a different stack (`build_nsa_hybrid_stack`) but the coordination principle is the same.

---

## Node State Machine

Every `TreeNode` in the radix tree can be in one of four conceptual states:

```
  ┌───────────────┐
  │  GPU (warm)   │  node.evicted=False, node.value set (GPU indices)
  └───────────────┘
        │
        │ evict() — GPU pool full, write_through policy:
        │   1. backup_from_device_all_layer() copies KV pages → CPU
        │   2. node.backuped = True, node.host_value = CPU indices
        │   3. GPU slot freed (node.value = None, node.evicted = True)
        ▼
  ┌───────────────┐
  │  CPU (cold)   │  node.evicted=True, node.backuped=True
  └───────────────┘
        │
        │ evict_host() — CPU pool full, storage backend configured:
        │   1. write_storage() sends CPU pages → HiCacheStorage
        │   2. CPU slot freed (node.host_value = None)
        │   3. Node stays in tree, tracked by storage key only
        ▼
  ┌───────────────┐
  │  STORAGE      │  node.evicted=True, node.backuped=False (host freed)
  └───────────────┘
        │
        │ evict from storage (TTL or capacity):
        │   _record_remove_event() emits BlockRemoved event
        │   node deleted from tree
        ▼
     (deleted)
```

The transitions are strictly downward during pressure. The reverse — load operations — happen when a request prefix matches a cold or stored node:

```
  STORAGE  →  storage.read()  →  CPU  →  load_to_device_per_layer()  →  GPU
  CPU      →  load_to_device_per_layer()  →  GPU
```

---

## The Request Data Flow

### Scenario A: GPU Hit (Tier 1)

```
Request arrives: "Summarise this document: [10k tokens]..."
  │
  ▼ HiRadixCache.match_prefix()
    Walk tree, find longest matching prefix
    node.evicted == False → GPU pages are live
  │
  ▼ Scheduler dispatches immediately
    Attention kernel reads K/V from GPU pool for matched tokens
    Model runs only on unmatched tail tokens
    → No stall, minimal TTFT
```

### Scenario B: CPU Hit (Tier 2)

```
Request arrives, same prefix — but it was evicted to CPU
  │
  ▼ HiRadixCache.match_prefix()
    Walk tree, find node, node.evicted == True, node.backuped == True
  │
  ▼ HiRadixCache.load_back()   (hiradix_cache.py:940)
    Allocate GPU slots for the evicted pages
    cache_controller: load_to_device_per_layer() per layer
      → PCIe DMA: CPU pinned buffer → GPU KV pool
    load_cache_event.wait() — scheduler stalls until all layers loaded
  │
  ▼ Scheduler dispatches
    Attention kernel reads K/V from GPU pool (now restored)
    → Stall: ~10–100 ms (PCIe bandwidth bound)
    → No re-prefill compute cost
```

### Scenario C: Storage Hit (Tier 3)

```
Request arrives, prefix is in storage (not CPU)
  │
  ▼ HiRadixCache.match_prefix() (with storage lookup)
    Walk tree, find node, or query storage backend batch_exists_v2()
  │
  ▼ HiCacheController.prefetch() — if best_effort policy
    Prefetch thread: storage.read() → CPU pinned buffer
    (runs concurrently while scheduler checks other work)
  │
  ▼ If prefetch completes in time: Scenario B path (CPU → GPU)
    If not: full re-prefill for that segment
    → Stall: ~50–500 ms (storage I/O bound)
```

### Scenario D: Complete Miss

```
Request arrives, prefix not in tree at any tier
  │
  ▼ HiRadixCache.match_prefix() returns 0 matched tokens
  │
  ▼ Scheduler dispatches immediately — but runs model over full input
    KV tensors generated fresh by the model
    HiRadixCache.insert() adds the new node to the tree (GPU tier)
    → No stall, but full prefill compute cost (1–25 seconds)
```

---

## Write-Through vs Write-Back

The eviction path (GPU → CPU) has two modes, controlled by `--hicache-write-policy`:

**`write_through`** (default): when a node's access count reaches the `write_through_threshold` (1 for strict write-through), the cache immediately starts a background copy of its KV pages to the CPU pool. The GPU slot is freed only after the CPU copy completes. This means the node is always CPU-backed before it can be evicted from GPU — no data loss risk, but the eviction path has a copy overhead.

```python
# hiradix_cache.py:708 — write-through trigger on access
def _inc_hit_count(self, node, chunked=False):
    node.hit_count += 1
    if not node.backuped and node.hit_count >= self.write_through_threshold:
        self.write_backup(node)   # start async GPU → CPU copy
```

**`write_back`**: the GPU slot is freed immediately during eviction; the CPU copy happens lazily. Lower eviction latency, but if the process crashes between eviction and the write completing, that node's KV tensors are lost.

For production deployments with a storage backend, `write_through` is strongly recommended.

---

## Tensor Parallelism Synchronisation

When running with tensor parallelism (`tp_size > 1`), each GPU rank manages its own shard of the KV tensors. `HiRadixCache` uses a TP process group (`tp_group`) to synchronise the write-through ack queue:

```python
# hiradix_cache.py:737 — TP sync for write-through acknowledgements
torch.distributed.all_reduce(
    queue_size,
    op=torch.distributed.ReduceOp.MIN,
    group=self.tp_group,
)
```

All ranks must agree that a write has completed before the GPU slot is freed — `MIN` ensures the slowest rank gates progress. This guarantees consistency: no rank sees a node as "evicted" while another rank still holds its KV pages in GPU memory.

---

## What the Scheduler Sees

The scheduler (`Scheduler`) interacts with HiCache through the same `match_prefix` / `insert` / `evict` interface it used with `RadixCache`. The scheduler does not know whether a hit came from tier 1, 2, or 3 — it only knows how many tokens were matched. The stall during a CPU or storage load is handled by the `load_cache_event.wait()` inside `load_back()`, which the scheduler calls before dispatching.

This design keeps the scheduler simple and makes HiCache a drop-in replacement for `RadixCache`.

---

## Key Files Referenced

| File | What it shows |
|---|---|
| `REPOS/sglang/python/sglang/srt/mem_cache/hiradix_cache.py:65` | `HiRadixCache.__init__` — pool construction and controller setup |
| `REPOS/sglang/python/sglang/srt/mem_cache/hiradix_cache.py:835` | `HiRadixCache.evict()` — GPU eviction with write-through and write-back paths |
| `REPOS/sglang/python/sglang/srt/mem_cache/hiradix_cache.py:905` | `HiRadixCache.evict_host()` — CPU eviction to storage |
| `REPOS/sglang/python/sglang/srt/mem_cache/hiradix_cache.py:940` | `HiRadixCache.load_back()` — CPU-to-GPU restore path |
| `REPOS/sglang/python/sglang/srt/mem_cache/hiradix_cache.py:700` | `_inc_hit_count()` — write-through trigger on access |
| `REPOS/sglang/python/sglang/srt/mem_cache/hiradix_cache.py:713` | `writing_check()` — TP-synchronised write-through acknowledgement |
| `REPOS/sglang/python/sglang/srt/managers/cache_controller.py:247` | `HiCacheController` — background I/O coordinator |
