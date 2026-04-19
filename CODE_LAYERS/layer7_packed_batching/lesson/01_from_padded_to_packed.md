# 01 — From Padded to Packed

## From Layer 6 to Layer 7

In Layer 6, `decode_step` solved the ragged-batch problem by padding every request's KV history to the same length and running `F.scaled_dot_product_attention` on the resulting rectangular tensor:

```python
# Layer 6 — decode_step (key lines)
max_kv    = max(kv_lens)

# Left zeros for padding, ones for real KV + new token slot
attn_mask = torch.zeros(B, max_kv + 1, dtype=torch.long, device=DEVICE)
for i, kv_len in enumerate(kv_lens):
    attn_mask[i, max_kv - kv_len:] = 1

# BatchedKVCache pads each request's history to max_kv_len before stacking
batch_kv = BatchedKVCache(reqs, max_kv)
with torch.no_grad():
    logits = self.model(last_toks,
                        attention_mask=attn_mask,
                        kv_cache=batch_kv,
                        position_ids=pos_ids)   # [B, 1, vocab]
batch_kv.write_back()
```

The attention kernel received a `[B, n_kv_heads, max_kv_len + 1, head_dim]` tensor. For every row where `kv_len_i < max_kv_len`, the left columns contained zeros. `_build_additive_mask` converted those zeros to `−inf`, so they received zero weight after softmax — but the kernel still computed attention scores over them. Every multiply-accumulate touching a zero column was wasted.

In Layer 7, the same `decode_step` reads:

```python
# Layer 7 — decode_step (key lines)
pack_kv = PackedKVCache(reqs, self._workspace)

pack_kv.plan(
    num_q_heads  = cfg.num_attention_heads,   # 16
    num_kv_heads = cfg.num_key_value_heads,   # 8
    head_dim     = cfg.head_dim,              # 128
    dtype        = DTYPE,
)

with torch.no_grad():
    logits = self.model(last_toks,
                        attention_mask=None,    # FlashInfer uses indptr, not a mask
                        kv_cache=pack_kv,
                        position_ids=pos_ids)   # [B, 1, vocab]
pack_kv.write_back()
pack_kv.end_forward()
```

`attn_mask` is gone. `attention_mask=None` is passed to the model — FlashInfer uses the `kv_indptr` array built in `PackedKVCache.__init__` to locate each request's real token slice, with no mask tensor required. `plan()` and `end_forward()` are the two new call sites that bookend the forward pass.

The sections below explain each of these changes in detail. Section 02 covers the data layout and `indptr` arithmetic inside `PackedKVCache`. Section 03 covers `update()` and `write_back()`. Section 04 covers the backend dispatch in `attention.py`.

---

## The Workspace

One additional change in Layer 7 appears at `ModelRunner.__init__`, before any requests arrive:

```python
# 256 MB workspace shared across all decode steps.
_WORKSPACE_MB = 256

class ModelRunner:
    def __init__(self, model_path: str) -> None:
        ...
        self._workspace = torch.empty(
            _WORKSPACE_MB * 1024 * 1024, dtype=torch.uint8, device=DEVICE
        )
```

FlashInfer uses internal temporary buffers during kernel planning (`begin_forward`). If no workspace is provided, it allocates these buffers on the fly — inside `decode_step`, while the GPU is otherwise active. This mid-step allocation stalls the CUDA stream and adds latency jitter. By pre-allocating a single 256 MB `uint8` buffer at startup and passing it to every `PackedKVCache`, all temporary memory for the FlashInfer planning phase is satisfied from this buffer. The workspace is reused on every decode step; it is never freed.

---

## What Is Unchanged

The structural change from Layer 6 to Layer 7 is narrow. Prefill is identical:

```python
# model_runner.py — prefill (Layer 7, unchanged from Layer 6)
kv = PerReqKVCache()
logits = self.model(ids, attention_mask=mask, kv_cache=kv, position_ids=pos)
```

`prefill` always uses `PerReqKVCache` and `F.sdpa` — the FlashInfer path is only for the batched decode step, not for B=1 prefill. Per-request position IDs in `decode_step` are also unchanged:

```python
pos_ids = torch.tensor(
    [[kv_len] for kv_len in kv_lens], dtype=torch.long, device=DEVICE
)   # [B, 1] — each request at its own absolute position kv_len_i
```

Each request is still placed at its own absolute RoPE position rather than a shared offset — the same per-request correctness requirement that Layer 6 established. Sampling, `output_ids` updates, finished-request detection, and the scheduler's `_resolve` call are all unchanged. The new code is confined to `PackedKVCache` and the FlashInfer dispatch path in `attention.py`.
