# 08 — What Comes Next

Layer 6 eliminates head-of-line blocking. A short request exits the decode batch as soon as it finishes and its result is returned immediately, without waiting for longer co-running requests. The batch size is dynamic — it grows as requests arrive and shrinks as requests finish. TTFT is per-request rather than per-batch. These are the gains that make continuous batching the standard approach in production inference systems.

But two costs remain embedded in the design, both visible in `kv_cache.py`.

---

## The Padding Problem

Every decode step, `BatchedKVCache._init_layer` pads shorter requests' K/V histories to `max_kv_len` with zeros:

```python
rk = F.pad(rk, (0, 0, pad, 0))   # left-pad seq dimension to max_kv_len
```

If the running batch contains one request at position 500 and fifteen requests at position 3, each of the fifteen is padded with 497 zeros per layer. `F.scaled_dot_product_attention` then computes attention over all `max_kv_len + 1 = 501` positions for every row in the batch, even though 497 of those 501 positions are masked out. The masked positions are excluded from the output, but they are not excluded from the memory bandwidth and compute cost of the attention kernel.

The waste is proportional to `1 - (average_kv_len / max_kv_len)`. In a batch with heterogeneous lengths, this ratio can be very poor. For the 3-vs-500 example above, average KV length is `(500 + 15 × 3) / 16 ≈ 34`, and `max_kv_len = 500`, so only 7% of the attention compute is doing useful work.

---

## The Memory Fragmentation Problem

`PerReqKVCache` grows by `torch.cat` on every decode step:

```python
self._k[layer_idx] = torch.cat([self._k[layer_idx], new_k], dim=2)
```

`torch.cat` allocates a new tensor every call, copying all existing content. For a 500-token output, this is 500 allocation-and-copy events per layer, across 28 layers, for each active request. The allocator cannot reuse the old memory until the previous tensor is garbage collected, leading to high peak memory pressure and fragmentation: the GPU memory pool fills with tensors of varying sizes, leaving gaps that cannot be filled by new allocations of different sizes.

In practice, a continuous batching system running for hours accumulates enough fragmentation that it either OOMs or must be periodically restarted — both unacceptable for a production serving system.

---

## What Paged Attention Changes

Both problems share the same root cause: KV memory is allocated per-request and grown contiguously. The fix is to decouple physical memory allocation from logical sequence position.

Paged attention, introduced by vLLM and adopted by SGLang, allocates KV memory in fixed-size **blocks** (pages) rather than contiguous tensors. A request's KV cache is a list of page pointers, not a single tensor. Pages are allocated from a global pool as needed and returned to the pool when a request finishes. Any page can hold any request's tokens at any position.

The attention kernel is modified to accept a **block table** — a mapping from (request_idx, block_idx) to a physical GPU memory address — and reads K/V from non-contiguous memory in a single kernel. FlashInfer and FlashAttention implement ragged attention kernels that do this without any padding: each row of the batch reads exactly its true number of KV tokens, with zero waste.

The consequences: **no padding** (each request attends over only its real KV tokens), **no fragmentation** (pages are fixed-size and freely interchangeable), and **no per-step allocation** (K/V is written in-place to pre-allocated pages). KV memory is treated as a shared pool like a page cache, and requests borrow pages rather than owning tensors.

---

## What Files Change

The `model/` package, `tokenizer.py`, `sampling.py`, and `server.py` are unchanged. The `Scheduler` and `Req` classes are extended but not restructured: `Req` gains a block table instead of a `PerReqKVCache`; the scheduler gains a `BlockManager` that tracks page allocation and preemption. `model_runner.prefill` and `model_runner.decode_step` call FlashInfer ragged kernels instead of `F.scaled_dot_product_attention` with a padded `BatchedKVCache`.

The pattern from Layers 4 through 6 holds: one capability is added, one bottleneck is addressed, and the rest of the system is untouched. Layer 7 adds paged KV allocation. Everything above it stays exactly as it is now.
