# 03 — Tier 2: The Host Pool (GPU ↔ CPU)

## What This Section Covers

Tier 2 is a block of CPU RAM that acts as a fast overflow buffer for the GPU KV pool. When the GPU pool fills up, KV pages are written here instead of being discarded. When a request needs a prefix that was evicted here, the pages are loaded back to the GPU before the request is dispatched. This section covers the memory allocation strategy, the buffer layout options, and the DMA transfer implementation that makes this tier fast enough to be practical.

---

## Why Pinned Memory?

Normal heap memory (`malloc`) is managed by the OS virtual memory system: pages can be swapped out to disk, remapped, or moved. GPU DMA engines cannot access pageable memory directly. When you call `cudaMemcpy` on pageable memory, CUDA silently allocates a temporary pinned staging buffer, copies from pageable → pinned, then DMAs from pinned → GPU. This doubles the effective memory bandwidth consumed and adds an extra copy step.

**Pinned (page-locked) memory** is committed to physical RAM and locked against paging. The GPU DMA engine can access it directly — the data path is:

```
CPU pinned buffer  ──────  PCIe bus  ──────  GPU VRAM
    (physical page, fixed address)         (no staging copy)
```

This makes pinned memory the right choice for a KV cache buffer that will be repeatedly read and written by the GPU. The trade-off is that pinned memory cannot be paged out, so it permanently reduces the OS's available physical memory — which is why you should size the host pool carefully (see Section 06).

---

## HostKVCache: The Base Class

`HostKVCache` (`memory_pool_host.py:154`) is the abstract base class for the tier-2 buffer. Its `__init__` handles the common setup:

```python
# memory_pool_host.py:154
class HostKVCache(abc.ABC):
    def __init__(self, device_pool, host_to_device_ratio, host_size, page_size, layout,
                 pin_memory, device, allocator_type="default"):

        # Pool sizing: use ratio OR exact bytes, whichever is set
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)

        # Align to page boundary
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size

        # Safety check: refuse to allocate if system RAM is insufficient
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(f"Not enough host memory available. ...")

        self.kv_buffer = self.init_kv_buffer()   # allocate pinned tensor
```

The safety check is important: allocating more pinned memory than the system has physical RAM will cause the allocation to fail with a cryptic CUDA error. SGLang checks `psutil.virtual_memory().available` and subtracts a reserve before allocating.

The allocator used to obtain the pinned buffer is controlled by `allocator_type`:
- Default: `alloc_with_host_register` — allocates a plain torch tensor and registers it with `cudaHostRegister`
- `"npu"`: uses `alloc_with_pin_memory` — PyTorch's built-in `pin_memory=True` flag (for Ascend NPU compatibility)

---

## MHATokenToKVPoolHost: Standard Attention

`MHATokenToKVPoolHost` (`memory_pool_host.py:290`) extends `HostKVCache` for standard multi-head attention models (e.g. Llama, Mistral, Qwen).

**Buffer dimensions** depend on the layout flag:

| Layout | Tensor shape | Use case |
|---|---|---|
| `layer_first` (default) | `(2, layers, tokens, heads, dim)` | Efficient per-layer DMA strides |
| `page_first` | `(2, tokens, layers, heads, dim)` | Page-first layout for storage backends |
| `page_first_direct` | `(2, pages, layers, page_size, heads, dim)` | Enables a faster direct-copy kernel path |
| `page_head` | `(2, pages, heads, page_size, layers, dim)` | Alternative page-aligned layout |

The `2` outer dimension is K (key) and V (value). For `layer_first`, `self.k_buffer[layer_id]` gives a contiguous slice of all token positions for a single layer — which is exactly the stride pattern needed for per-layer DMA transfers.

Layer data pointers are cached as GPU tensors for fast kernel access:
```python
# memory_pool_host.py:329
self.k_data_ptrs = torch.tensor(
    [x.data_ptr() for x in self.k_data_refs],
    dtype=torch.uint64,
    device=self.device_pool.device,
)
```

These GPU-side pointer arrays allow CUDA kernels to access the host buffer addresses directly without Python overhead per layer.

---

## MLATokenToKVPoolHost: Latent Attention (MLA)

`MLATokenToKVPoolHost` (`memory_pool_host.py:787`) handles Multi-head Latent Attention as used in DeepSeek models. MLA stores a compressed low-rank KV representation rather than the full K and V tensors per head. This means the per-token storage footprint is smaller, so the CPU pool holds proportionally more tokens for the same number of bytes — tier 2 is effectively more efficient for MLA models.

The transfer logic (`load_to_device_per_layer`, `backup_from_device_all_layer`) mirrors the MHA version but operates on the MLA-specific buffer layout.

---

## GPU → CPU: The Backup Path

When `HiCacheController` receives an eviction request, it calls `backup_from_device_all_layer` on the host pool:

```python
# memory_pool_host.py:513 — MHATokenToKVPoolHost
def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
    if io_backend == "kernel":
        if self.layout == "layer_first":
            if self.can_use_jit:
                jit_transfer_hicache_all_layer(
                    k_ptr_dst=self.k_data_ptrs,    # CPU buffer pointer array
                    v_ptr_dst=self.v_data_ptrs,
                    indices_dst=host_indices,        # where to write in CPU pool
                    k_ptr_src=device_pool.k_data_ptrs,  # GPU buffer pointer array
                    v_ptr_src=device_pool.v_data_ptrs,
                    indices_src=device_indices,      # where to read from GPU pool
                    ...
                )
            else:
                transfer_kv_all_layer(...)   # non-JIT fallback
```

The `io_backend="kernel"` path uses a custom Triton/CUDA kernel (`transfer_kv_all_layer` or the JIT variant `jit_transfer_hicache_all_layer`) that:
1. Takes GPU-side arrays of host and device page indices.
2. Iterates over all layers in a single kernel launch.
3. Writes KV data from the GPU pool's scattered page indices into the CPU buffer's page slots.
4. Runs on a dedicated CUDA `write_stream` — separate from the request execution stream, so in-flight inference is not blocked.

The key performance property: the kernel issues PCIe DMA transfers for all layers in one launch. The GPU's DMA engine handles the actual data movement while the CUDA compute units continue serving requests.

---

## CPU → GPU: The Load-Back Path

When `HiRadixCache.load_back()` is called for a cold node, `HiCacheController` calls `load_to_device_per_layer` once per transformer layer:

```python
# memory_pool_host.py:396 — MHATokenToKVPoolHost
def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend):
    if io_backend == "kernel":
        if self.layout == "layer_first":
            if self.can_use_jit:
                jit_transfer_hicache_one_layer(
                    k_cache_dst=device_pool.k_buffer[layer_id],  # GPU destination
                    v_cache_dst=device_pool.v_buffer[layer_id],
                    k_cache_src=self.k_buffer[layer_id],         # CPU source (pinned)
                    v_cache_src=self.v_buffer[layer_id],
                    indices_dst=device_indices,
                    indices_src=host_indices,
                    element_dim=self.element_dim,
                )
```

This is called per-layer rather than all-at-once. The reason is pipeline overlap: the controller can begin issuing requests for layer 1 while layer 0 is still transferring, hiding some of the PCIe latency through overlap. A `LayerDoneCounter` tracks when each layer's transfer completes; the `load_cache_event` fires when all layers are done.

---

## JIT Kernels vs Fallback Kernels

`can_use_jit` (`memory_pool_host.py:316`) is `True` when:
- Running on CUDA (not NPU)
- The element dimension fits the JIT kernel's size constraints

The JIT kernels (`jit_transfer_hicache_all_layer`, `jit_transfer_hicache_one_layer`) are compiled at startup with Triton for the specific model's head dimension. They are significantly faster than the non-JIT path because:
- No Python loop over layers
- Kernel launches are fused across heads
- Memory access pattern is aligned to cache lines

When JIT is not available (e.g. first time running with a new head dimension, or NPU), the fallback `transfer_kv_all_layer` handles the transfer with a Python-level layer loop.

---

## Buffer Layout and DMA Strides

The layout choice affects DMA efficiency:

**`layer_first` (default)**: the buffer is laid out as `[K/V, layer, token, head, dim]`. For a given layer, all token data is contiguous. The per-layer DMA kernel reads a contiguous slice — ideal for sequential PCIe DMA.

**`page_first_direct`**: layout is `[K/V, page, layer, page_size, head, dim]`. All layers for a given page are contiguous. This enables a single contiguous read per page for storage backends, but the per-layer scatter kernel must stride across non-contiguous memory — slightly slower for pure GPU↔CPU transfers.

For most deployments without a storage backend, `layer_first` is the right choice. `page_first_direct` makes sense when tier-3 I/O is the bottleneck and you want to minimise the number of DMA operations to the storage backend.

---

## How Much CPU Memory Is Allocated?

The formula (from `HostKVCache.__init__`):

```
CPU pool tokens = GPU pool tokens × hicache_ratio   (when hicache_size == 0)
CPU pool bytes  = CPU pool tokens × bytes_per_token
                = CPU pool tokens × (head_dim × head_num × layer_num × dtype_bytes × 2)
```

For a Llama-3-8B model (`num_layers=32, num_heads=32, head_dim=128, dtype=bfloat16`):
- `bytes_per_token = 128 × 32 × 32 × 2 × 2 = 524,288 bytes ≈ 0.5 MB`
- GPU pool of 10,000 tokens → `10,000 × 0.5 MB = 5 GB`
- With `hicache_ratio=2.0` → CPU pool = 20,000 tokens → `10 GB` of pinned RAM

This is why you need to check available RAM before enabling HiCache. SGLang does this check at startup and raises a clear error if there is insufficient memory.

---

## Key Files Referenced

| File | What it shows |
|---|---|
| `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool_host.py:132` | `alloc_with_pin_memory()` — PyTorch pin_memory allocation |
| `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool_host.py:113` | `alloc_with_host_register()` — cudaHostRegister-based allocation |
| `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool_host.py:154` | `HostKVCache.__init__` — pool sizing and safety check |
| `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool_host.py:290` | `MHATokenToKVPoolHost` — standard MHA implementation |
| `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool_host.py:350` | `init_kv_buffer()` — layout-dependent tensor allocation |
| `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool_host.py:396` | `load_to_device_per_layer()` — CPU → GPU per-layer transfer |
| `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool_host.py:513` | `backup_from_device_all_layer()` — GPU → CPU all-layer transfer |
| `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool_host.py:787` | `MLATokenToKVPoolHost` — compressed MLA variant |
