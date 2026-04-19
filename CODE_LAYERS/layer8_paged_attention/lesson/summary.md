# Layer 8 — Summary

Layer 8 replaces `PerReqKVCache` and `PackedKVCache` with a single pre-allocated `KVPool`, eliminating the per-step float gather that made Layer 7's `decode_step` copy `O(total_kv_tokens)` bytes of KV data every iteration. The scheduler, the request dataclass, `prefill()`'s API, and `tokenizer.py` are unchanged. The model layers gain mechanical `ForwardBatch` threading (`decoder_layer.py`, `qwen3.py`), and `attention.py` is rewritten to use a backend object rather than `hasattr` duck-typing.

---

## From Layer 7 to Layer 8

In Layer 7, every decode step assembled a packed float buffer from all active requests' KV histories:

```python
# Layer 7 — decode_step (key lines)
pack_kv = PackedKVCache(reqs, self._workspace)
pack_kv.plan(num_q_heads=..., num_kv_heads=..., head_dim=..., dtype=DTYPE)

fb = ForwardBatch(mode=ForwardMode.DECODE, kv_cache=pack_kv, attention_mask=None)
logits = self.model(last_toks, forward_batch=fb, position_ids=pos_ids)

pack_kv.write_back()    # scatter new K/V back to each PerReqKVCache
pack_kv.end_forward()   # release FlashInfer state
```

Inside `PackedKVCache.update()`, called once per attention layer per decode step, every request's historical K/V was read from its `PerReqKVCache`, converted to NHD layout, and concatenated into a new contiguous buffer. For a batch with 1000 total accumulated tokens and 28 layers, that is 1000 × 28 × 2 × 8 × 128 × 2 = ~916 MB of float data copied per decode step, growing with every token generated.

In Layer 8, the same decode step reads:

```python
# Layer 8 — decode_step (key lines)
new_slots = [self.kv_pool.alloc(1)[0] for _ in reqs]

kv_lens_plus1  = [len(r.slot_indices) + 1 for r in reqs]
kv_indptr      = torch.tensor([0] + list(accumulate(kv_lens_plus1)), dtype=torch.int32, device=DEVICE)
kv_indices     = torch.tensor(
    [s for req in reqs for s in req.slot_indices + [new_slots[i]]]
    , dtype=torch.int32, device=DEVICE)
kv_last_page_lens = torch.ones(B, dtype=torch.int32, device=DEVICE)

decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self._workspace, "NHD")
decode_wrapper.begin_forward(kv_indptr, kv_indices, kv_last_page_lens,
                             cfg.num_attention_heads, cfg.num_key_value_heads,
                             cfg.head_dim, page_size=1, data_type=DTYPE, q_data_type=DTYPE)

ctx = DecodeKVCtx(wrapper=decode_wrapper,
                  k_pool=self.kv_pool.k_pool, v_pool=self.kv_pool.v_pool,
                  new_slots=new_slots_t)
fb = ForwardBatch(mode=ForwardMode.DECODE, kv_cache=ctx, attention_mask=None)

logits = self.model(last_toks, forward_batch=fb, position_ids=pos_ids)
decode_wrapper.end_forward()

for i, req in enumerate(reqs):
    req.slot_indices.append(new_slots[i])   # integer list append — no tensor allocation
```

`PackedKVCache`, `write_back()`, and the float gather are gone. No K/V data moves after the initial pool write during prefill. The only per-step allocations are a `List[int]` of new slot indices and two `int32` tensors (`kv_indptr` and `kv_indices`) whose total size is proportional to the number of active tokens — but they contain integer slot addresses, not float KV values.

---

## The KV Pool

`KVPool` is allocated once at `ModelRunner.__init__`, immediately after the model weights are loaded, so the sizing formula can measure true free GPU memory:

```python
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

`KVPool` allocates two lists of tensors — `k_pool` and `v_pool` — each holding one `[total_slots, n_kv_heads, head_dim]` tensor per layer. Slot 0 is left as zeros and reserved as a FlashInfer padding dummy; real tokens occupy slots 1 onward. The free-slot list is a plain Python `List[int]` containing every unused slot index.

```python
self.free_slots: List[int] = list(range(1, total_slots))
```

`alloc(n)` pops the first `n` entries from `free_slots` and returns them. `free(slots)` extends `free_slots` with the returned indices. Both operations are `O(n)` and in-place on a Python list — no GPU work, no tensor allocation. When a request finishes, `kv_pool.free(req.slot_indices)` returns all of its slots immediately, making them available to the next request before Python's garbage collector has processed anything.

---

## Prefill: Writing to the Pool

Prefill in Layer 8 allocates pool slots for every prompt token before the forward pass:

```python
prompt_len = len(req.input_ids)
slots = self.kv_pool.alloc(prompt_len)
req.slot_indices = slots

ids  = torch.tensor([req.input_ids], device=DEVICE)
mask = torch.ones(1, prompt_len, dtype=torch.long, device=DEVICE)
pos  = torch.arange(prompt_len, device=DEVICE).unsqueeze(0)

ctx = PrefillKVCtx(slots, self.kv_pool)
fb  = ForwardBatch(mode=ForwardMode.PREFILL, kv_cache=ctx, attention_mask=mask)
logits = self.model(ids, forward_batch=fb, position_ids=pos)
```

`PrefillKVCtx` carries the slot indices and a reference to the pool. During the forward pass, each attention layer calls `ctx.store(layer_idx, k, v)` as a side-effect — writing the freshly computed K/V into the pool while the attention computation itself (`F.sdpa`) runs over the same in-memory tensors:

```python
def store(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
    # k: [1, n_kv_heads, prompt_len, head_dim]
    k_nhd = k.squeeze(0).permute(1, 0, 2).contiguous()   # [L, n_kv, D]
    v_nhd = v.squeeze(0).permute(1, 0, 2).contiguous()
    self._kv_pool.k_pool[layer_idx][self._slot_t] = k_nhd
    self._kv_pool.v_pool[layer_idx][self._slot_t] = v_nhd
```

The `self._slot_t` tensor (a pre-converted `int64` version of the slot list) is used for fancy indexing — a single in-place scatter write per layer. After prefill completes, `req.slot_indices` holds the complete list of pool row indices that contain this request's prompt K/V across all 28 layers. No separate write-back step is needed.

---

## Decode: Indexing the Pool

Each decode step allocates one new slot per request, then constructs two integer arrays that tell FlashInfer how to read the pool:

```python
new_slots = [self.kv_pool.alloc(1)[0] for _ in reqs]

kv_indices_list: List[int] = []
for i, req in enumerate(reqs):
    kv_indices_list.extend(req.slot_indices)   # historical slots
    kv_indices_list.append(new_slots[i])        # new token's slot

kv_indptr      = torch.tensor(kv_indptr_list,   dtype=torch.int32, device=DEVICE)
kv_indices     = torch.tensor(kv_indices_list,  dtype=torch.int32, device=DEVICE)
kv_last_page_lens = torch.ones(B, dtype=torch.int32, device=DEVICE)
```

`kv_indices` is the flat concatenation of every request's slot list — the same ragged shape that `kv_indptr` describes. `kv_last_page_lens` is always all-ones because `page_size=1` means every page holds exactly one token and the last page of every request always contains exactly one real entry.

`begin_forward` is called once with all three arrays plus the head counts. FlashInfer plans the paged decode kernel: it stores the index and will use it in every `wrapper.forward()` call during the 28-layer forward pass. After `end_forward()`, `req.slot_indices.append(new_slots[i])` records the new slot as part of the request's history. The append is a Python list operation — no tensor allocation.

---

## The Attention Dispatch

Layer 8 replaces `attention.py`'s `hasattr` duck-typing with a backend-object pattern borrowed from SGLang. One `PagedBackend` instance is stored on `Qwen3Attention` at `__init__` time, and `forward()` calls exactly one method:

```python
# model/attention.py — the only dispatch line
attn_out = self.backend.forward(q, k, v, self.layer_idx, forward_batch)
```

All kernel logic lives in `model/backend.py`. `PagedBackend.forward()` reads `forward_batch.mode` and routes to one of two private methods.

**PREFILL** (`_prefill_forward`):
```python
if forward_batch.kv_cache is not None:
    forward_batch.kv_cache.store(layer_idx, k, v)   # pool write (side-effect)

k_rep = repeat_kv(k, self.num_kv_groups)             # GQA expand for F.sdpa
v_rep = repeat_kv(v, self.num_kv_groups)

additive_mask = build_additive_mask(
    forward_batch.attention_mask, q_len, kv_len, q.dtype, q.device
)
return F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=additive_mask, scale=self.scale)
```

**DECODE** (`_decode_forward`):
```python
q_fi = q.squeeze(2)    # [B, n_q_heads,  head_dim]
k_fi = k.squeeze(2)    # [B, n_kv_heads, head_dim]
v_fi = v.squeeze(2)

forward_batch.kv_cache.store(layer_idx, k_fi, v_fi)     # write new token first

k_paged = forward_batch.kv_cache.k_pool[layer_idx].unsqueeze(1)
v_paged = forward_batch.kv_cache.v_pool[layer_idx].unsqueeze(1)
# [total_slots, 1, n_kv, head_dim]  ← page_size=1 page dimension

attn_out = forward_batch.kv_cache.wrapper.forward(q_fi, (k_paged, v_paged))
return attn_out.unsqueeze(2)
```

The mask construction (`build_additive_mask`) moved from `Qwen3Model.forward` (where it was computed once and shared across all 28 layers) into `PagedBackend._prefill_forward` (computed once per attention call). The cost is the same — mask construction is cheap — but the responsibility now sits where the kernel logic lives. On the decode path, `forward_batch.attention_mask` is `None`; `build_additive_mask` is never called.

There is no SDPA fallback for decode. Layer 7's ragged packed-KV layout offered a meaningful SDPA alternative because the packed buffer was already contiguous. In Layer 8, the KVPool architecture cannot support padded SDPA for decode without performing the float gather that was just eliminated — so the option does not exist.

GQA: `repeat_kv` is still called on the prefill path (`F.sdpa` requires matching Q/KV head counts). FlashInfer handles the 16Q/8KV ratio natively on the decode path — no `repeat_kv` needed there.

The `unsqueeze(1)` on the pool tensor inserts the page dimension FlashInfer's paged API expects: `[total_slots, page_size, n_kv_heads, head_dim]`. With `page_size=1` this is `[total_slots, 1, n_kv_heads, head_dim]`. FlashInfer slices the relevant rows via `kv_indices` from `begin_forward` — no intermediate buffer is constructed.

---

## The Full Loop

The scheduler and server are identical to Layer 7.

When a new request arrives, the scheduler calls `prefill(req)`. `ModelRunner.prefill` calls `kv_pool.alloc(prompt_len)` to reserve one pool slot per prompt token and stores the returned list on `req.slot_indices`. A `PrefillKVCtx(slots, kv_pool)` is wrapped in a `ForwardBatch(mode=PREFILL, kv_cache=ctx, attention_mask=mask)` and passed to the model. Each of the 28 attention layers routes to `PagedBackend._prefill_forward`: `ctx.store()` writes the prompt's K/V into the assigned pool slots, then `F.sdpa` runs self-attention over the freshly computed rectangular tensors. The first output token is sampled; `req.slot_indices` now permanently holds the pool addresses for every prompt position.

On subsequent scheduler iterations, `decode_step(_running)` runs. For each active request, one new slot is allocated: `self.kv_pool.alloc(1)[0]`. The `kv_indices` list is built by concatenating each request's `slot_indices` with its new slot — an integer list, no float data. `kv_indptr` and `kv_last_page_lens` are derived from the counts. `begin_forward` plans the kernel once. A `DecodeKVCtx` is wrapped in a `ForwardBatch(mode=DECODE, kv_cache=ctx, attention_mask=None)`.

The 28-layer forward pass runs. In each `Qwen3Attention.forward`, `self.backend.forward()` routes to `PagedBackend._decode_forward`: `ctx.store(layer_idx, k_fi, v_fi)` writes the new token's K/V into the pool at `new_slots`, then `wrapper.forward(q_fi, (k_paged, v_paged))` reads the full history for each request via `kv_indices`. No K/V data was gathered, copied, or reallocated. `end_forward()` releases FlashInfer's internal state. `req.slot_indices.append(new_slot)` records the new address. Sampling, result delivery, and finished-request handling are identical to Layer 7.

When a request finishes, `kv_pool.free(req.slot_indices)` returns every slot the request ever occupied to the free list in one call. The slots are available immediately to the next prefill — no GC cycle, no tensor deallocation latency.

---

## What Comes Next

Layer 8 eliminates float KV copies but keeps `page_size=1`: every token occupies its own pool slot, so `kv_indices` has one entry per token across all requests. For a batch with 1000 accumulated tokens, `kv_indices` contains 1000 integers and FlashInfer performs 1000 index lookups per layer per decode step. Layer 9 groups tokens into fixed-size pages (e.g., 16 tokens per page), reducing `kv_indices` to `ceil(total_tokens / 16)` entries. `KVPool` shape becomes `[total_pages, page_size, n_kv_heads, head_dim]`; `alloc` and `free` operate on pages rather than individual tokens; `kv_last_page_lens` carries the actual fill count of the last page per request, which is less than `page_size` for requests whose history is not a multiple of the page size. The same `BatchDecodeWithPagedKVCacheWrapper` API is used — the only difference is the `page_size` argument to `begin_forward` and the shape of the pool tensors.
