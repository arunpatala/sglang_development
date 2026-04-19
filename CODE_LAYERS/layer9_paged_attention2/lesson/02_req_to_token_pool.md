# 02 — ReqToTokenPool

`ReqToTokenPool` is the structural bridge between the CPU and the Triton kernel. Every page index that is allocated for a request — whether at prefill or at a conditional decode-step expansion — is written into this table on the GPU. The Triton kernel can then build the `kv_indices` array by reading rows directly from device memory, with no Python iteration and no host-to-device copy of historical data.

---

## Shape and Purpose

```python
class ReqToTokenPool:
    def __init__(self, max_batch: int, max_context_len: int) -> None:
        self.req_to_token = torch.zeros(
            (max_batch, max_context_len), dtype=torch.int32, device=DEVICE
        )
        self.free_slots: List[int] = list(range(max_batch))
```

The table is a 2D `int32` GPU tensor of shape `[max_batch, max_pages_per_req]`. Row `r` holds the physical page indices for the request currently assigned row `r`. Column `c` holds the index of that request's c-th page in the `KVPool`. The invariant is:

```
req_to_token[req_pool_idx, page_pos] = physical_page_idx
```

`max_batch` is the server's concurrency limit (`_MAX_CONCURRENT_REQS = 128`). `max_context_len` is `ceil(_MAX_TOKEN_CONTEXT / page_size)` — with `page_size=16` and a 4096-token context limit, this is 256 columns. The total table size is `128 × 256 × 4 = 131072` bytes — 128 KB on the GPU, allocated once at startup and never resized.

The name mirrors SGLang's `ReqToTokenPool` in `srt/mem_cache/memory_pool.py`. In SGLang the table is conceptually identical: a GPU-resident row-per-request lookup of physical page addresses, designed so that the attention kernel's index builder can run entirely on-device.

---

## Row Ownership

Each active request occupies exactly one row. `alloc()` pops a row index from the Python-side free list:

```python
def alloc(self) -> int:
    if not self.free_slots:
        raise RuntimeError("ReqToTokenPool exhausted — too many concurrent requests")
    return self.free_slots.pop()

def free(self, idx: int) -> None:
    self.free_slots.append(idx)
```

`model_runner.prefill` calls `alloc()` once per request and stores the returned index on `req.req_pool_idx`. `decode_step` uses `req.req_pool_idx` to identify which row to write new page indices into and to tell the Triton kernel which row to read. When a request finishes, `model_runner.decode_step` calls `free(req.req_pool_idx)`, which immediately returns the row to the pool — no GC, no deferred deallocation.

The free-list mechanics are identical to `KVPool.free`: a plain Python list pop and append. This is `O(1)` and runs entirely on the CPU. The GPU table row is not cleared when freed — the next prefill that claims the row will overwrite its valid columns, and the Triton kernel reads only up to `num_pages` columns per request, so leftover stale data in higher columns is never accessed.

---

## Writes: Prefill and Conditional Decode

The table is written in two places. At prefill, all page indices for the prompt are written at once:

```python
# model_runner.prefill
pages_t = torch.tensor(page_indices, dtype=torch.int32, device=DEVICE)
self.req_to_token_pool.req_to_token[req.req_pool_idx, :n_pages] = pages_t
```

This is a contiguous slice assignment of `n_pages` int32 values into a single row — one scatter write, no loop. After this call, the GPU table is the sole source of truth for which pages this request owns.

During decode, a new page is added only when the current last page is completely full — when `seq_len % page_size == 0`. In that case, one new page is allocated and a scalar write adds it to the next column:

```python
# model_runner.decode_step — conditional page allocation
if token_offsets_list[i] == 0:
    new_page = self.kv_pool.alloc(1)[0]
    req.slot_indices.append(new_page)
    self.req_to_token_pool.req_to_token[
        req.req_pool_idx, num_pages_list[i]
    ] = new_page
```

The scalar `= new_page` assignment is a single int32 write into the GPU table. When `token_offset != 0`, the existing last page still has room; no write occurs and the table is unchanged. For a request with `page_size=16`, the table write happens once every 16 decode steps — 15 out of every 16 steps touch neither the table nor the free list.

---

## What the Table Replaces

In Layer 8, the per-request slot list was a Python list on the CPU:

```python
req.slot_indices: List[int]   # grows by 1 each decode step
```

Building `kv_indices` required iterating that list in Python and calling `torch.tensor(kv_indices_list)` — a full CPU-to-GPU transfer of every accumulated token index every step. For a request with 1000 accumulated tokens, that was 4000 bytes transferred and 1000 Python iteration steps, repeated every step.

In Layer 9, `req.slot_indices` still exists as a Python list, but it grows by one entry only every `page_size` steps and holds page indices — `ceil(1000/16) = 63` entries instead of 1000. Its only purpose is to let `kv_pool.free(req.slot_indices)` return all pages at once when the request finishes. The historical page data is mirrored in the GPU table from the moment of each alloc write, and the Triton kernel reads from the GPU table, not from the Python list.
