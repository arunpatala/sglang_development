# 04 — Tier 3: The Storage Backend Interface

## What This Section Covers

Tier 2 (CPU RAM) holds evicted KV pages in memory. When the CPU pool itself fills up, those pages need to go somewhere durable: local NVMe, a remote distributed memory store, or an object storage backend. This section describes the `HiCacheStorage` abstract interface, the factory that creates backend instances, the built-in backends, and the `HiCacheController` that orchestrates async I/O between the CPU pool and storage.

---

## HiCacheStorage: The Abstract Interface

`HiCacheStorage` (`hicache_storage.py:98`) is the single interface that every tier-3 backend must implement. `HiRadixCache` and `HiCacheController` only call methods on this interface — they never import a concrete backend class directly.

```python
# hicache_storage.py:98
class HiCacheStorage(ABC):

    # Registration: called once at startup to give the backend a reference
    # to the CPU pool it will DMA from/to
    def register_mem_pool_host(self, mem_pool_host: HostKVCache): ...
    def register_mem_host_pool_v2(self, host_pool, host_pool_name): ...

    # Query: check which of a batch of hash keys exist in storage.
    # Returns the length of the usable consecutive prefix hit.
    def batch_exists_v2(self, keys, pool_transfers, extra_info) -> PoolTransferResult: ...
    def batch_exists(self, keys, extra_info) -> int: ...          # legacy
    def exists(self, key) -> bool: ...

    # Read: load pages from storage into the CPU pool's pinned buffer
    def batch_get_v2(self, transfers, extra_info) -> dict[str, List[bool]]: ...
    def batch_get(self, keys, target_locations, ...) -> List[Tensor | None]: ...  # legacy
    def get(self, key, ...) -> Tensor | None: ...

    # Write: persist pages from the CPU pool's pinned buffer to storage
    def batch_set_v2(self, transfers, extra_info) -> dict[str, List[bool]]: ...
    def batch_set(self, keys, values, ...) -> bool: ...           # legacy
    def set(self, key, value, ...) -> bool: ...

    # Utility
    def clear(self) -> None: ...
    def get_stats(self): ...
```

The v2 interface (`batch_exists_v2`, `batch_get_v2`, `batch_set_v2`) uses `PoolTransfer` descriptors that carry page index arrays and hash keys. The v1/legacy interface passes tensors directly. New backends should implement the v2 interface; the controller calls v2 when available and falls back to v1.

The key design property: the backend is given the pinned CPU buffer reference at startup (`register_mem_pool_host`). Reads and writes operate on buffer indices — `host_indices` is a tensor of integer offsets into the CPU pool. The backend DMAs directly from/to the pinned buffer, avoiding any extra copy into Python-managed memory.

---

## PoolTransfer and PoolTransferResult

`PoolTransfer` (`hicache_storage.py:62`) is the data descriptor for a single pool's transfer:

```python
@dataclass
class PoolTransfer:
    name: PoolName             # "kv", "mamba", "indexer" — which pool
    host_indices: Optional[torch.Tensor]   # offsets into CPU pinned buffer
    device_indices: Optional[torch.Tensor] # offsets into GPU pool (unused for storage)
    keys: Optional[List[str]]             # hash keys for storage lookup
    hit_policy: PoolHitPolicy             # ALL_PAGES or TRAILING_PAGES
```

`PoolTransferResult` (`hicache_storage.py:77`) carries back the number of pages successfully found per pool:

```python
@dataclass
class PoolTransferResult:
    kv_hit_pages: int                      # how many KV pages were found
    extra_pool_hit_pages: dict[str, int]   # pages found for auxiliary pools
```

This structure supports models with multiple parallel KV pools (e.g. DeepSeek DSA, which has a main KV pool and an auxiliary indexer pool). The `hit_policy` determines how strictly all pages must be present — `ALL_PAGES` requires every page in the matched prefix to exist; `TRAILING_PAGES` only requires the last few pages.

---

## HiCacheStorageConfig

`HiCacheStorageConfig` (`hicache_storage.py:17`) is passed to every backend constructor:

```python
@dataclass
class HiCacheStorageConfig:
    tp_rank: int             # tensor parallel rank of this process
    tp_size: int
    pp_rank: int             # pipeline parallel rank
    pp_size: int
    attn_cp_rank: int
    attn_cp_size: int
    is_mla_model: bool       # True for DeepSeek MLA
    enable_storage_metrics: bool
    is_page_first_layout: bool   # True if CPU buffer uses page_first layout
    model_name: Optional[str]    # used for storage namespacing
    extra_config: Optional[dict] # backend-specific JSON config
```

The config provides the parallel topology so backends can correctly namespace their storage keys. A `tp_rank=0, tp_size=4` process writes KV pages that belong to GPU rank 0 — the storage key must encode the rank to avoid collision with data from other ranks.

---

## StorageBackendFactory: Lazy Registration

`StorageBackendFactory` (`storage/backend_factory.py:16`) is a registry that maps backend name strings to backend classes. Backends are loaded lazily — the import only happens when `create_backend()` is called, not at module import time:

```python
# backend_factory.py:16
class StorageBackendFactory:
    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_backend(cls, name, module_path, class_name):
        def loader():
            return cls._load_backend_class(module_path, class_name, name)
        cls._registry[name] = {"loader": loader, ...}

    @classmethod
    def create_backend(cls, backend_name, storage_config, mem_pool_host):
        registry_entry = cls._registry[backend_name]
        backend_class = registry_entry["loader"]()   # import happens here
        return cls._create_builtin_backend(backend_name, backend_class, ...)
```

This means optional backends (NIXL, Mooncake, hf3fs) are not imported unless explicitly requested — no import error for users who do not have those libraries installed.

Registered backends at module load time (`backend_factory.py:192–231`):

| Registration call | Backend name | Module | Class |
|---|---|---|---|
| Line 193 | `"file"` | `sglang.srt.mem_cache.hicache_storage` | `HiCacheFile` |
| Line 197 | `"nixl"` | `sglang.srt.mem_cache.storage.nixl.hicache_nixl` | `HiCacheNixl` |
| Line 203 | `"mooncake"` | `sglang.srt.mem_cache.storage.mooncake_store.mooncake_store` | `MooncakeStore` |
| Line 209 | `"hf3fs"` | `sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs` | `HiCacheHF3FS` |
| Line 215 | `"aibrix"` | `sglang.srt.mem_cache.storage.aibrix_kvcache.aibrix_kvcache_storage` | `AibrixKVCacheStorage` |
| Line 221 | `"eic"` | `sglang.srt.mem_cache.storage.eic.eic_storage` | `EICStorage` |
| Line 227 | `"simm"` | `sglang.srt.mem_cache.storage.simm.hicache_simm` | `HiCacheSiMM` |

The `"dynamic"` backend is a special case: it reads `module_path` and `class_name` from `extra_config` at runtime, allowing users to plug in their own `HiCacheStorage` subclass without modifying SGLang source.

---

## Built-in Backends at a Glance

**`file` — `HiCacheFile`**

The reference implementation. Stores each KV page as a flat binary file on the local filesystem. Each page is identified by a hash key derived from the token sequence prefix. Reads and writes use standard `torch.save` / `torch.load` on a configured `storage_dir`. Suitable for single-node development and NVMe setups. No external dependencies.

**`nixl` — `HiCacheNixl`**

Uses the NIXL agent protocol for network I/O — the agent can back onto remote RDMA memory, NVMe-oF, or object storage depending on configuration. Provides higher throughput than local file I/O for multi-node setups and can serve as a shared KV cache across multiple SGLang instances.

**`mooncake` — `MooncakeStore`**

Uses `mooncake.store.MooncakeDistributedStore` — a distributed shared-memory store from the Mooncake project. Provides RDMA-backed fast reads and writes across nodes. Note: this is Mooncake B (the distributed store) — distinct from Mooncake A (the transfer engine used for PD disaggregation). Layer 18 covers the full Mooncake story.

**`hf3fs` — `HiCacheHF3FS`**

Writes KV pages to a HuggingFace 3FS (Three File System) mount. 3FS is a distributed filesystem designed for ML training workloads; it provides parallel reads from SSDs across many nodes with RDMA-accelerated data paths.

**`aibrix` — `AibrixKVCacheStorage`**

Interfaces with AIBrix's external KVCache service — an external microservice that manages a shared KV cache pool. Suitable for multi-tenant deployments where multiple SGLang instances share a single cache service.

**`eic` / `simm`**

Specialist RDMA cluster stores with NIC-affinity scheduling, designed for low-latency network transfers at scale.

---

## HiCacheController: The Async I/O Coordinator

`HiCacheController` (`managers/cache_controller.py:247`) runs the GPU↔CPU DMA path (Section 03) and also manages the CPU↔storage async I/O. It owns:

- `write_stream` and `load_stream`: dedicated CUDA streams for GPU↔CPU DMA, separate from the inference stream
- `write_queue` and `load_queue`: pending GPU↔CPU operations, batched before dispatch
- `prefetch_queue` (via `threading.Queue`): pending CPU↔storage prefetch reads, processed by a background thread
- `backup_thread`: background thread that moves CPU pages to storage asynchronously

### The Write Path (GPU → CPU → Storage)

```
HiRadixCache.evict()
    │
    ▼ HiCacheController.write(device_indices)
        mem_pool_host.alloc(n) → host_indices
        write_queue.append(CacheOperation(...))
        start_writing()
            │ on CUDA write_stream:
            │   backup_from_device_all_layer(device_pool, host_indices, device_indices)
            │   finish_event.record()
        ack_write_queue.append(...)
    │
    ▼ HiRadixCache.writing_check()  (called each scheduler step)
        finish_event.query() — has the GPU→CPU DMA completed?
        If yes: node.backuped = True
                If enable_storage: write_backup_storage(node)
                    │
                    ▼ HiCacheController.write_storage(host_indices, keys)
                        backup_thread picks up the operation
                        storage_backend.batch_set_v2(transfers)  →  storage
```

The two-phase design (GPU→CPU via CUDA stream, CPU→storage via background thread) decouples the fast DMA path from the slower storage I/O path. The scheduler is not blocked at either step.

### The Prefetch Path (Storage → CPU)

```
HiRadixCache.match_prefix() — storage hit detected
    │
    ▼ HiCacheController.prefetch(request_id, host_indices, new_input_tokens, ...)
        prefetch_queue.put(PrefetchOperation(...))
    │
    ▼ prefetch_thread (background)
        storage_backend.batch_exists_v2(keys) — check what's available
        for each page in order:
            page_get_func(operation, hash_values, host_indices)
                → storage_backend.batch_get_v2(transfers)
                → data lands in CPU pinned buffer at host_indices
            operation.increment(page_size) — signal progress
    │
    ▼ Scheduler polling: operation.completed_tokens
        When enough tokens loaded: issue load() for CPU → GPU
        load_cache_event.wait() for GPU load to complete
        Dispatch request to model with loaded prefix
```

The prefetch path is designed for overlap: the prefetch thread reads from storage while the scheduler checks other work. For requests with long prefixes in storage, this overlap can hide most of the storage I/O latency. The `prefetch_stop_policy` and `prefetch_timeout_base` parameters control how long the scheduler will wait for prefetch before giving up and proceeding with a partial prefix.

---

## Writing a Custom Backend (`"dynamic"`)

To plug in a custom backend without modifying SGLang:

```python
# my_custom_backend.py
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig

class MyStorage(HiCacheStorage):
    def __init__(self, config: HiCacheStorageConfig, extra_config: dict):
        self.config = config
        # ... connect to your storage system

    def exists(self, key): ...
    def get(self, key, target_location=None, target_sizes=None): ...
    def set(self, key, value=None, target_location=None, target_sizes=None): ...
    def batch_get(self, keys, target_locations=None, target_sizes=None): ...
    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None): ...
```

Launch with:

```bash
python -m sglang.launch_server \
  --enable-hierarchical-cache \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{
    "backend_name": "my_storage",
    "module_path": "my_custom_backend",
    "class_name": "MyStorage",
    "your_option": "value"
  }'
```

The `StorageBackendFactory._create_dynamic_backend()` method imports `my_custom_backend.MyStorage` at runtime and passes `storage_config` plus `extra_config` to its constructor.

---

## Key Files Referenced

| File | What it shows |
|---|---|
| `REPOS/sglang/python/sglang/srt/mem_cache/hicache_storage.py:17` | `HiCacheStorageConfig` dataclass |
| `REPOS/sglang/python/sglang/srt/mem_cache/hicache_storage.py:62` | `PoolTransfer` descriptor |
| `REPOS/sglang/python/sglang/srt/mem_cache/hicache_storage.py:98` | `HiCacheStorage` abstract class |
| `REPOS/sglang/python/sglang/srt/mem_cache/hicache_storage.py:277` | `HiCacheFile` — local disk backend |
| `REPOS/sglang/python/sglang/srt/mem_cache/storage/backend_factory.py:16` | `StorageBackendFactory` — lazy registry |
| `REPOS/sglang/python/sglang/srt/mem_cache/storage/backend_factory.py:192` | Backend registration calls |
| `REPOS/sglang/python/sglang/srt/managers/cache_controller.py:247` | `HiCacheController.__init__` |
| `REPOS/sglang/python/sglang/srt/managers/cache_controller.py:663` | `HiCacheController.write()` — GPU→CPU enqueue |
| `REPOS/sglang/python/sglang/srt/managers/cache_controller.py:681` | `start_writing()` — CUDA stream dispatch |
| `REPOS/sglang/python/sglang/srt/managers/cache_controller.py:799` | `HiCacheController.prefetch()` — storage prefetch enqueue |
| `REPOS/sglang/python/sglang/srt/managers/cache_controller.py:948` | `prefetch_thread_func` — background storage reader |
| `REPOS/sglang/python/sglang/srt/managers/cache_controller.py:998` | `write_storage()` — CPU→storage write pipeline |
