# 02 — The KV Pool

`KVPool` is the structural core of Layer 8. Every other change in this layer — `PrefillKVCtx`, `DecodeKVCtx`, the integer index arrays in `decode_step`, the removal of `write_back()` — is a consequence of having a single pre-allocated pool that all requests share. Understanding the pool's layout and its slot arithmetic is the prerequisite for understanding everything else.

---

## Sizing and Allocation

`KVPool` is instantiated in `ModelRunner.__init__`, immediately after `Qwen3ForCausalLM.from_pretrained` returns:

```python
self.model = Qwen3ForCausalLM.from_pretrained(model_path, dtype=DTYPE)

cfg = self.model.model.config
free_bytes, _ = torch.cuda.mem_get_info()
bytes_per_token = (
    cfg.num_hidden_layers * 2          # K + V
    * cfg.num_key_value_heads          # 8
    * cfg.head_dim                     # 128
    * (torch.finfo(DTYPE).bits // 8)   # 2 for bfloat16
)
max_tokens = int(free_bytes * kv_memory_fraction / bytes_per_token)

self.kv_pool = KVPool(
    total_slots = max_tokens,
    n_layers    = cfg.num_hidden_layers,
    n_kv_heads  = cfg.num_key_value_heads,
    head_dim    = cfg.head_dim,
    dtype       = DTYPE,
)
```

The sizing is deferred until after model load for a precise reason: `torch.cuda.mem_get_info()` returns memory that is truly free at that instant, after the model weights have been committed to GPU memory. If the pool were sized from total GPU memory at startup — before model load — it would need to guess how much memory the weights consume, which varies with quantization, tensor parallelism, and activations. Deferring to post-load gives exact accounting with one line.

For Qwen3-0.6B with `kv_memory_fraction=0.85` and no other GPU load, the formula yields roughly `bytes_per_token = 28 × 2 × 8 × 128 × 2 = 114688` bytes per token, and with ~6 GB free after weight load, `max_tokens ≈ 56000` slots. The server can hold approximately 56000 simultaneously active tokens across all requests before the pool runs out.

---

## The Flat Tensor Layout

Inside `KVPool.__init__`:

```python
self.k_pool: List[torch.Tensor] = [
    torch.zeros(total_slots, n_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    for _ in range(n_layers)
]
self.v_pool: List[torch.Tensor] = [
    torch.zeros(total_slots, n_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    for _ in range(n_layers)
]

self.free_slots: List[int] = list(range(1, total_slots))
```

`k_pool` and `v_pool` are Python lists with one entry per transformer layer — 28 entries for Qwen3-0.6B. Each entry is a single flat tensor of shape `[total_slots, n_kv_heads, head_dim]`. Row `i` of `k_pool[layer]` holds the key vector for token slot `i` at layer `layer`. The same row index `i` is used across all 28 layers: slot 5 in `k_pool[0]` and slot 5 in `k_pool[27]` both belong to the same logical token. This uniform row-to-token mapping is what makes `kv_indices` work: a single flat list of slot integers addresses the same token's K and V across all layers simultaneously.

Slot 0 is intentionally left as all-zeros and is never assigned to a real token. FlashInfer's paged attention kernel uses a zero-padding convention: if `kv_indices` contains slot 0, it reads zeros — the additive identity for softmax — which contributes nothing to the output. Having a guaranteed zero row eliminates the need to handle the case where a slot index is out of bounds or uninitialized.

The free-slot list is a plain Python list:

```python
self.free_slots: List[int] = list(range(1, total_slots))
```

At startup, every slot except 0 is free. Slots are consumed from the front and returned by extending the back. There is no ordering guarantee after slots are freed; returned slots may be interleaved with other free slots, and that is fine — `kv_indices` records the exact slot for every token, so FlashInfer does not assume any contiguity.

---

## Allocation and Deallocation

```python
def alloc(self, n: int) -> List[int]:
    if n > len(self.free_slots):
        raise RuntimeError(
            f"KVPool OOM: need {n} slots, only {len(self.free_slots)} free"
        )
    slots = self.free_slots[:n]
    self.free_slots = self.free_slots[n:]
    return slots

def free(self, slots: List[int]) -> None:
    self.free_slots.extend(slots)
```

`alloc(n)` pops the first `n` entries from `free_slots` and returns them as a list. The caller receives a concrete list of slot row indices, which it stores on `req.slot_indices`. `free(slots)` extends `free_slots` with the returned indices in one call.

Both operations are `O(n)` and run entirely in Python on the CPU — no GPU work, no tensor allocation, no CUDA synchronization. In Layer 7, returning `PerReqKVCache` memory required garbage collection: the `kv_cache` attribute had to go out of scope and then wait for Python's GC cycle to reclaim the GPU tensors. In Layer 8, `kv_pool.free(req.slot_indices)` executes at the moment the request finishes — before Python even returns from `decode_step` — and the slots are immediately available to the next `prefill` call. This deterministic, GC-independent reclamation is one of the key memory-management advantages of the pool design.

The `RuntimeError` on exhaustion surfaces at the `alloc` call site in `model_runner.prefill` or `model_runner.decode_step`. In practice, the scheduler can be extended to check `kv_pool.available()` before scheduling a new prefill and defer it if the pool would be exhausted — the same mechanism that production systems use to implement KV cache admission control.

---

## What Lives on the Request Object

In Layer 7, KV history was stored on `req.kv_cache` — a `PerReqKVCache` holding one growing tensor per layer per request. In Layer 8, `req.kv_cache` is gone. History is tracked by `req.slot_indices: List[int]`, which grows by one integer per decode step:

```python
# After prefill:
req.slot_indices = [3, 7, 12, 18, 25, ...]   # one slot per prompt token

# After each decode step (appended in model_runner.decode_step):
req.slot_indices.append(new_slots[i])          # one integer, no tensor allocation
```

The integer list is all that `decode_step` needs to build `kv_indices`. A request with 1000 accumulated tokens has `req.slot_indices` with 1000 entries — a 1000-entry Python list of integers, occupying roughly 8 KB, compared to the Layer 7 equivalent of `28 × 2 × 1 × 8 × 1000 × 128 × 2 = 918 MB` of GPU tensors.
