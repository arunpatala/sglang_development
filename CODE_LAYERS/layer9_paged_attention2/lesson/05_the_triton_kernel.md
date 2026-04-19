# 05 — The Triton Kernel

`create_flashinfer_kv_indices_triton` is the kernel that makes `ReqToTokenPool` useful. It takes the GPU-resident page-index table and writes the flat `kv_indices` array that FlashInfer needs, without any Python iteration and without transferring historical index data across the PCIe bus. Understanding the kernel is understanding why the per-step CPU-to-GPU data footprint is O(B) rather than O(total_kv_tokens).

---

## The Problem It Solves

After prefill, the GPU table holds:

```
req_to_token[req_pool_idx, 0] = page_A
req_to_token[req_pool_idx, 1] = page_B
req_to_token[req_pool_idx, 2] = page_C
...
```

FlashInfer needs `kv_indices` — a flat int32 tensor containing these page indices for every active request, laid out as:

```
[page_A_req0, page_B_req0, ...,   page_A_req1, page_B_req1, ..., ...]
```

The naive way to build this is the Python loop from Layer 8, adapted for pages:

```python
for i, req in enumerate(reqs):
    kv_indices_list.extend(req.slot_indices)  # Python list extend, O(num_pages)
kv_indices = torch.tensor(kv_indices_list, device='cuda')  # CPU→GPU copy
```

This has two costs: Python-level iteration over each request's page list, and a host-to-device copy of every page index that has ever been allocated. The page data already lives in `req_to_token` on the GPU. The kernel reads it there, eliminating both costs.

---

## The Kernel Signature

```python
@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,       # [max_batch, max_context_len]  int32 GPU table
    req_pool_indices_ptr,   # [B]  which row per request
    page_kernel_lens_ptr,   # [B]  how many pages each request has
    kv_indptr,              # [B+1]  cumulative page offsets
    kv_start_idx,           # [B] or None — start offset within row (None here)
    kv_indices_ptr,         # [total_pages_in_batch]  output
    req_to_token_ptr_stride: tl.constexpr,  # req_to_token.shape[1]
):
```

The kernel is launched with `grid=(B,)` — one Triton program per active request. `req_pool_indices_ptr` is the `[B]` array of row indices into `req_to_token`, one per request. `page_kernel_lens_ptr` is the `[B]` array of per-request page counts. `kv_indptr` is the cumulative prefix-sum that tells each program where to write in `kv_indices_ptr`. `kv_start_idx` is `None` in standard decoding and exists only for sliding-window attention.

`req_to_token_ptr_stride` is a `tl.constexpr` — a compile-time constant. Triton uses it to compute the row offset `req_pool_index * stride` without a runtime multiply. It is set to `req_to_token.shape[1]`, which is `max_pages_per_req`.

---

## Per-Program Logic

Each program handles one request, identified by `pid = tl.program_id(axis=0)`:

```python
pid = tl.program_id(axis=0)

req_pool_index    = tl.load(req_pool_indices_ptr + pid)
kv_indices_offset = tl.load(kv_indptr + pid)

kv_start = 0
kv_end   = 0
if kv_start_idx:
    kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
    kv_end   = kv_start
kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)
```

`req_pool_index` is the row in `req_to_token` that belongs to this request. `kv_indices_offset` is the starting position in the output `kv_indices_ptr` where this request's page indices should be written. With `kv_start_idx = None`, `kv_start = 0` and `kv_end = num_pages` for this request.

The range `[kv_start, kv_end)` is the slice of `req_to_token[req_pool_index]` that holds valid page indices. For a request with 4 pages, program `pid` reads columns 0, 1, 2, 3 from its row and writes them to `kv_indices[kv_indices_offset : kv_indices_offset + 4]`.

---

## The Copy Loop

```python
num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
for i in range(num_loop):
    offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
    mask   = offset < kv_end - kv_start
    data   = tl.load(
        req_to_token_ptr
        + req_pool_index * req_to_token_ptr_stride
        + kv_start
        + offset,
        mask=mask,
    )
    tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)
```

`BLOCK_SIZE = 512` elements per iteration. For a request with 63 pages, `num_loop = ceil(63/512) = 1` — the entire copy completes in a single iteration with 449 lanes masked off. For a request with 600 pages (possible at very long context), two iterations run.

`tl.arange(0, BLOCK_SIZE)` generates a vector of 512 offsets within the current block. `mask = offset < kv_end - kv_start` disables lanes that extend beyond the request's valid page count. `tl.load(..., mask=mask)` reads a 512-element vector from `req_to_token[req_pool_index, kv_start : kv_start + BLOCK_SIZE]`, masking out-of-range addresses. `tl.store` writes the loaded vector to the corresponding range of `kv_indices`.

The memory access pattern for a single warp is a contiguous read from one row of `req_to_token` and a contiguous write to one slice of `kv_indices`. Both are coalesced accesses — adjacent threads read and write adjacent memory addresses — which maximizes DRAM throughput. All B programs run concurrently on the GPU, so the effective throughput is `B` concurrent coalesced copies.

---

## What Moves, What Does Not

When the kernel completes, `kv_indices` holds the correct flat page-index array for `begin_forward`. No token-level index data crossed the PCIe bus. What the CPU sent was:

- `req_pool_idx_t`: `[B]` int32, `B × 4` bytes
- `num_pages_t`: `[B]` int32, `B × 4` bytes
- `kv_indptr`: `[B+1]` int32, `(B+1) × 4` bytes (built on-GPU, not sent from CPU)

The page indices themselves — the content of `kv_indices` — were read from `req_to_token` on the GPU, where they have lived since prefill wrote them. For a batch with 500 accumulated pages, the kernel reads 500 × 4 = 2000 bytes from on-device memory and writes 2000 bytes to on-device memory. Zero of those bytes crossed the PCIe bus.

In Layer 8, the equivalent step sent `kv_indices_list` — a CPU-assembled list of all token-level slot indices — across the bus every step. For 500 accumulated pages of 16 tokens each, that was 8000 × 4 = 32 KB per step. The Triton kernel eliminates that transfer entirely and replaces it with a GPU-local copy that runs in parallel across all B requests.

---

## Connection to `begin_forward`

The kernel's output is the `kv_indices` argument to `begin_forward`:

```python
self._decode_wrapper.begin_forward(
    kv_indptr,
    kv_indices,          # produced by Triton kernel, never left GPU
    kv_last_page_lens,
    ..., P, ...
)
```

FlashInfer reads `kv_indices` to determine which rows of `k_pool[layer]` and `v_pool[layer]` to attend over for each request. `kv_indptr` tells it where each request's slice starts and ends. `kv_last_page_lens` tells it how many valid tokens are in each request's last page. With `page_size=P`, FlashInfer reads `P × n_kv_heads × head_dim` float values per page index — one rectangle of the pool per page, not one vector per token.
