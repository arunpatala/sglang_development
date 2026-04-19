# 06 — What Comes Next

Layer 7 eliminates padding waste inside the attention kernel. When one request has 1000 tokens of history and another has 3, FlashInfer attends over 1001 and 4 real tokens respectively — not 1001 and 1001. The compute advantage is proportional to the ratio of `average_kv_len / max_kv_len` in the running batch. For heterogeneous request lengths — which is the common case in a real serving system — this is a substantial and sustained improvement over Layer 6.

But two costs that were already present in Layer 6 remain, and Layer 7 makes one of them slightly worse.

---

## The Per-Step Gather Cost

Every decode step, `PackedKVCache.update()` gathers all active requests' historical K/V tensors and concatenates them into a new contiguous ragged buffer:

```python
for i, req in enumerate(self._reqs):
    hist_k_nhd = req.kv_cache._k[layer_idx].squeeze(0).permute(1, 0, 2).contiguous()
    segs_k.append(hist_k_nhd)
    segs_k.append(new_k[i].unsqueeze(0))

k_packed = torch.cat(segs_k, dim=0)   # [total_kv_tokens, n_kv, dim]
```

This `torch.cat` across all 28 layers reads a total of `total_kv_tokens × n_kv_heads × head_dim × 2 × num_layers` bytes of KV data every decode step, just to assemble the ragged input for FlashInfer. As requests accumulate history, `total_kv_tokens` grows, and this gather grows with it. It is an `O(total_kv_tokens)` copy that happens whether or not the attention compute needed it.

Layer 6 had a structurally identical cost inside `BatchedKVCache._init_layer` — it gathered per-request tensors into a padded stack — but that copy also included the padding overhead. Layer 7 removes the padding but keeps the gather. With a paged KV cache, the gather disappears entirely: FlashInfer reads directly from a block table that maps request indices to physical memory pages, with no intermediate concatenation step.

---

## Memory Fragmentation Persists

`PerReqKVCache` still grows by one `torch.cat` per layer per decode step:

```python
# In write_back() — called after every decode step
cache._k[layer_idx] = torch.cat([cache._k[layer_idx], k_tok], dim=2)
```

A request that generates 500 tokens accumulates `500 × 28 = 14000` separate allocation-copy events. Each new tensor is slightly larger than the previous one. The allocator cannot reuse the old tensor's memory until it is garbage-collected — and Python's GC does not guarantee prompt collection. Over hours of serving, the GPU memory pool fills with tensors of irregular sizes, leaving gaps that cannot be filled by new allocations of different sizes. This fragmentation eventually forces an OOM or a server restart.

---

## What Paged Attention Changes

Both problems share the same root cause: KV memory is owned by individual requests as contiguous tensors that grow over time. Paged attention decouples ownership from storage.

Physical KV memory is divided into fixed-size **blocks** — for example, 16 tokens per block, 28 layers, 8 KV heads, 128 head dimensions, in `bfloat16`. This gives each block a fixed size of `16 × 28 × 8 × 128 × 2 = 917504` bytes — roughly 900 KB. A global `BlockManager` tracks which blocks are free. When a request is prefilled, the `BlockManager` allocates enough blocks to cover the prompt tokens. When a decode step adds one more token and the current block is full, the `BlockManager` allocates one more block and appends its index to the request's block table. When a request finishes, all its blocks are returned to the free pool immediately, without waiting for GC.

FlashInfer's paged attention kernel accepts a block table — a tensor mapping `(request_idx, block_idx)` to physical GPU memory addresses — and reads K/V from non-contiguous pages in a single kernel launch. No `torch.cat` is needed. No contiguous buffer is assembled. The kernel's memory access pattern is a function of the block table, not of request length.

---

## What Files Change

`kv_cache.py` becomes a block-table-based allocator: `PerReqKVCache` is replaced by a block table per request, and `PackedKVCache` is replaced by a paged cache variant that passes the block table to FlashInfer. `model_runner.py` gains a `BlockManager` that handles allocation, eviction, and block table construction at the start of each decode step. `model/attention.py` gains a third dispatch branch — `PagedKVCache` → FlashInfer paged kernel — alongside the existing `PerReqKVCache` and `PackedKVCache` branches.

`scheduler.py`, `request.py`, `server.py`, `tokenizer.py`, and `model/` (except `attention.py`) are unchanged. The pattern continues: one mechanism changes, one file changes, the benchmark measures exactly that.
