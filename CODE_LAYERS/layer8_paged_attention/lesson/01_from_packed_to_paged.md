# 01 — From Packed to Paged

## From Layer 7 to Layer 8

In Layer 7, `decode_step` eliminated the padding waste of Layer 6 by replacing `BatchedKVCache` with `PackedKVCache` and routing the decode path through FlashInfer's ragged prefill kernel. But a gather cost remained: before every layer's attention call, `PackedKVCache.update()` collected each request's historical K/V from its own `PerReqKVCache`, converted to NHD layout, appended the new token, and concatenated all segments into a contiguous ragged buffer:

```python
# Layer 7 — decode_step (key lines)
pack_kv = PackedKVCache(reqs, self._workspace)
pack_kv.plan(num_q_heads=16, num_kv_heads=8, head_dim=128, dtype=DTYPE)

fb = ForwardBatch(mode=ForwardMode.DECODE, kv_cache=pack_kv, attention_mask=None)
logits = self.model(last_toks, forward_batch=fb, position_ids=pos_ids)

pack_kv.write_back()    # scatter new K/V back to each PerReqKVCache
pack_kv.end_forward()   # release FlashInfer state
```

Inside `pack_kv.update()`, called once per layer per decode step, the cost was:

```python
for i, req in enumerate(self._reqs):
    hist_k_nhd = req.kv_cache._k[layer_idx].squeeze(0).permute(1, 0, 2).contiguous()
    segs_k.append(hist_k_nhd)
    segs_k.append(new_k[i].unsqueeze(0))

k_packed = torch.cat(segs_k, dim=0)   # [total_kv_tokens, n_kv, dim]
```

For a batch with `T = total_kv_tokens` accumulated across all requests and 28 layers, the gather reads `T × 28 × 2 × 8 × 128 × 2` bytes of float data every decode step. At `T = 1000`, that is roughly 916 MB per step. `T` grows by `B` tokens every step, so the gather grows without bound.

In Layer 8, the same `decode_step` reads:

```python
# Layer 8 — decode_step (key lines)
new_slots = [self.kv_pool.alloc(1)[0] for _ in reqs]

kv_lens_plus1   = [len(r.slot_indices) + 1 for r in reqs]
kv_indptr_list  = [0] + list(accumulate(kv_lens_plus1))
kv_indices_list = []
for i, req in enumerate(reqs):
    kv_indices_list.extend(req.slot_indices)   # historical slot indices (integers)
    kv_indices_list.append(new_slots[i])        # new token's slot index

kv_indptr         = torch.tensor(kv_indptr_list,  dtype=torch.int32, device=DEVICE)
kv_indices        = torch.tensor(kv_indices_list, dtype=torch.int32, device=DEVICE)
kv_last_page_lens = torch.ones(B, dtype=torch.int32, device=DEVICE)

decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self._workspace, "NHD")
decode_wrapper.begin_forward(
    kv_indptr, kv_indices, kv_last_page_lens,
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
    1, data_type=DTYPE, q_data_type=DTYPE,
)

ctx = DecodeKVCtx(wrapper=decode_wrapper, k_pool=self.kv_pool.k_pool,
                  v_pool=self.kv_pool.v_pool, new_slots=new_slots_t)
fb  = ForwardBatch(mode=ForwardMode.DECODE, kv_cache=ctx, attention_mask=None)

logits = self.model(last_toks, forward_batch=fb, position_ids=pos_ids)
decode_wrapper.end_forward()

for i, req in enumerate(reqs):
    req.slot_indices.append(new_slots[i])   # integer list append, no tensor allocation
```

`PackedKVCache`, `write_back()`, and the float gather are entirely gone. No K/V data moves after the initial pool write during prefill. The per-step cost is now: `B` slot index allocations (Python `list.pop`), one `torch.int32` tensor of length `T + B`, one `torch.int32` tensor of length `T + B` for `kv_indptr`, and one `begin_forward` call. These are integer addresses, not float KV values. The per-step memory bandwidth is proportional to `T` in integers — roughly 4 bytes per token instead of 4096 bytes per token (for bfloat16, 8 KV heads, 128 dim).

---

## The Workspace

The FlashInfer workspace survives from Layer 7:

```python
_WORKSPACE_MB = 256

class ModelRunner:
    def __init__(self, model_path: str, kv_memory_fraction: float = ...) -> None:
        ...
        self._workspace = torch.empty(
            _WORKSPACE_MB * 1024 * 1024, dtype=torch.uint8, device=DEVICE
        )
```

Layer 7 introduced the workspace to avoid mid-step allocation latency during `begin_forward`. Layer 8 continues to pass it to `BatchDecodeWithPagedKVCacheWrapper`. The workspace is pre-allocated once at `ModelRunner.__init__` and reused on every decode step without being freed or resized.

---

## What Is Unchanged

Prefill in Layer 8 has the same outer structure as Layer 7 — B=1 per request, causal mask, `F.sdpa` for attention — but uses `PrefillKVCtx` instead of `PerReqKVCache`:

```python
# model_runner.py — prefill
slots = self.kv_pool.alloc(prompt_len)
req.slot_indices = slots

ctx = PrefillKVCtx(slots, self.kv_pool)
fb  = ForwardBatch(mode=ForwardMode.PREFILL, kv_cache=ctx, attention_mask=mask)
logits = self.model(ids, forward_batch=fb, position_ids=pos)
```

The per-request position IDs in `decode_step` are also structurally identical to Layer 7:

```python
pos_ids = torch.tensor(
    [[len(r.slot_indices)] for r in reqs], dtype=torch.long, device=DEVICE
)   # [B, 1] — each request at its own absolute position
```

Each request's position is its current total token count — prompt tokens plus previously generated tokens. This per-request assignment has been in place since Layer 6 and is unchanged. Sampling, `output_ids` appending, finished-request detection, and the scheduler's `_resolve` call are all identical to Layer 7. The new code is confined to `kv_cache.py`, `model_runner.py` (the decode step setup), `forward_batch.py`, and `model/backend.py`.

The sections below explain each new piece in code order. Section 02 covers `KVPool`: how it is sized, what the flat tensor layout looks like, and how `alloc` and `free` work. Section 03 covers `PrefillKVCtx` and how the pool write happens as a side-effect inside the attention forward. Section 04 covers the decode step index construction and `DecodeKVCtx`. Section 05 covers how `PagedBackend` in `model/backend.py` cleanly replaces the old `hasattr` dispatch in `attention.py`.
