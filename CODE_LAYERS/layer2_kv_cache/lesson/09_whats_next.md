# 08 — What Comes Next

Layer 2 makes the decode loop fast. The key and value tensors from each forward pass are stored and reused, TPOT drops substantially, and the improvement required only a handful of lines in `model.py`. `server.py`, `benchmark.py`, and the API contract are entirely unchanged.

But the KV cache implementation in `kv_cache.py` still has a visible cost. Each decode step, `LayerCache.update` allocates a new tensor and copies all previously stored data into it alongside the new entry:

```python
self.keys   = torch.cat([self.keys,   new_keys],   dim=-2)
self.values = torch.cat([self.values, new_values], dim=-2)
```

`torch.cat` allocates fresh memory every time. For a request generating 128 tokens, this means 128 allocation-and-copy operations per layer, each copying a slightly larger tensor than the one before. The allocator has to find a contiguous block of memory, copy the old data, write the new data, and free the old block. This overhead is real and measurable in the per-step timings.

Layer 3 eliminates it by changing `LayerCache` alone. Rather than growing the tensor dynamically, it pre-allocates a fixed-size buffer at the start of the request and writes new entries into the next available position in-place:

```python
# Layer 3 sketch — preallocated buffer, no cat
self.keys[:, :, self.cursor, :] = new_keys
self.cursor += 1
return self.keys[:, :, :self.cursor, :]
```

No allocation. No copy. The old data stays where it is. The `update` interface is identical, so `model.py` does not change — again. Only `kv_cache.py` is touched.

The pre-allocated approach requires knowing the maximum sequence length upfront so the buffer can be sized correctly. This is a mild constraint in a sequential server (the max is `prompt_tokens + max_new_tokens`), but it becomes a real design question in a batching system where requests of different lengths share the same GPU memory pool.

That batching question leads directly to the larger challenges that come after Layer 3. Sharing a single GPU efficiently across many concurrent requests requires knowing, for each request, exactly how much cache memory it needs and where that memory lives. With a fixed pre-allocated buffer per request, memory cannot be shared between requests even if one request finishes early and frees its buffer — the next request cannot use that space unless the buffer sizes happen to match. Paged attention solves this by breaking the cache into fixed-size pages that can be allocated and reclaimed independently of request length, allowing the memory manager to pack many requests tightly and reuse pages as requests complete.

For now, the practical lesson from Layer 2 is the pattern: one concept improves, one file changes, everything else stays the same. The benchmark measures exactly what changed. TPOT is faster; TTFT is unchanged. The next layer will show the same structure — a single targeted change with a predictable, measurable effect on the same benchmark.
