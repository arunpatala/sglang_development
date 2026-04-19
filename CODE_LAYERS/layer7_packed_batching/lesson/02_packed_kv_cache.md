# 02 — The Packed KV Cache

`PackedKVCache` is the replacement for `BatchedKVCache`. Where `BatchedKVCache` padded all per-request KV histories to `max_kv_len` and stacked them into a rectangular tensor that `F.sdpa` could process, `PackedKVCache` concatenates them back-to-back into a single ragged tensor and tells FlashInfer exactly where each request's slice begins and ends. The result is a kernel that attends only over real tokens — no zero columns, no wasted compute.

This section covers how `PackedKVCache` is constructed and how it communicates the ragged layout to FlashInfer. Section 03 covers what happens inside each attention layer when `update()` is called during the forward pass.

---

## The Ragged Layout

For three requests with KV lengths 10, 6, and 4, Layer 6 allocated a padded rectangular block; Layer 7 allocates a packed ragged block:

```
Layer 6 (padded):               Layer 7 (packed, NHD layout):
┌────────────────────────┐      ┌──────────────┐
│ req0  [  pad  ] [ 10 ] │      │ req0 [10+1]  │  tokens 0–10
│ req1  [  pad  ] [  6 ] │      │ req1 [ 6+1]  │  tokens 11–17
│ req2  [  pad  ] [  4 ] │      │ req2 [ 4+1]  │  tokens 18–22
└────────────────────────┘      └──────────────┘
 shape: [3, kv, 10, dim]         shape: [23, kv, dim]
```

The `+1` in each row's count — `10+1`, `6+1`, `4+1` — is the new decode token that `update()` appends inside the attention layer, before FlashInfer sees the tensor. FlashInfer therefore attends over 11, 7, and 5 real tokens respectively. The total tensor length is 23 instead of 30 — and if the longest request had 1000 tokens while the others had 3, the saving would be 3017 vs 3030, with the gap growing further as batches scale.

---

## `__init__`: Computing the Indptrs

```python
class PackedKVCache:
    def __init__(self, reqs: list, workspace: torch.Tensor) -> None:
        self._reqs = reqs
        B = len(reqs)

        kv_lens = [r.kv_cache.get_seq_length() for r in reqs]

        # qo_indptr: each request contributes exactly 1 query token.
        # Shape [B+1]: [0, 1, 2, ..., B]
        self.qo_indptr = torch.arange(B + 1, dtype=torch.int32, device=DEVICE)

        # kv_indptr: cumulative sum of (kv_len_i + 1).
        # +1 because update() appends the new decode token for each request
        # before the FlashInfer kernel runs.
        kv_full_lens = [l + 1 for l in kv_lens]
        kv_cumsum    = [0] + list(accumulate(kv_full_lens))
        self.kv_indptr = torch.tensor(kv_cumsum, dtype=torch.int32, device=DEVICE)

        # FlashInfer wrapper — one workspace reused across all 28 layers.
        self._wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, kv_layout="NHD"
        )

        # Storage for new K/V tokens (saved in update, consumed in write_back).
        self._new_k: Dict[int, torch.Tensor] = {}
        self._new_v: Dict[int, torch.Tensor] = {}
```

**`qo_indptr`** describes the query side. During decode, every request provides exactly one new query token, so the offsets are uniformly spaced: `[0, 1, 2, ..., B]`. This is constructed with `torch.arange(B + 1)` — no computation needed.

**`kv_indptr`** describes the key-value side. It is a cumulative sum of the full KV lengths (`kv_len_i + 1`) for each request. For the example above with lengths 10, 6, 4, `kv_full_lens = [11, 7, 5]` and `kv_cumsum = [0, 11, 18, 23]`. FlashInfer interprets `kv_indptr[i]` as the start and `kv_indptr[i+1]` as the exclusive end of request `i`'s slice in the packed KV tensor. The `+1` is intentional and essential: `update()` physically appends the new token into the packed tensor before `wrapper.forward()` is called, so FlashInfer must be told that each request's slice spans one more token than its historical length.

**The wrapper** is `flashinfer.BatchPrefillWithRaggedKVCacheWrapper`. Despite the name containing "Prefill", FlashInfer uses this wrapper for any operation with a ragged KV layout, including decode steps where `q_len=1` per request. `kv_layout="NHD"` specifies that the packed KV tensor is in tokens × heads × dim order, which is what `update()` produces.

---

## `plan()`: One Planning Call for 28 Layers

```python
def plan(
    self,
    num_q_heads:  int,
    num_kv_heads: int,
    head_dim:     int,
    dtype:        torch.dtype,
) -> None:
    self._wrapper.begin_forward(
        self.qo_indptr,
        self.kv_indptr,
        num_q_heads,
        num_kv_heads,
        head_dim,
        causal=False,
        q_data_type=dtype,
    )
```

`begin_forward` tells FlashInfer everything it needs to select a kernel and allocate any internal temp buffers: the indptr arrays, the head counts, the head dimension, and the data type. `causal=False` is correct here — with `q_len=1` per request, there are no future query positions to mask; each new token can attend over its full history including itself.

Crucially, `plan()` is called **once** before the model's forward pass, not once per attention layer. The 28 attention layers all reuse the same `_wrapper` object with the same plan. The indptr arrays and kernel selection are computed once and reused 28 times, amortising the planning overhead across all layers.

`model_runner.decode_step` calls `plan()` immediately after constructing `PackedKVCache`:

```python
pack_kv = PackedKVCache(reqs, self._workspace)
pack_kv.plan(
    num_q_heads  = cfg.num_attention_heads,   # 16
    num_kv_heads = cfg.num_key_value_heads,   # 8
    head_dim     = cfg.head_dim,              # 128
    dtype        = DTYPE,                     # bfloat16
)
logits = self.model(last_toks, attention_mask=None,
                    kv_cache=pack_kv, position_ids=pos_ids)
```

The model call then proceeds through all 28 decoder layers. Each layer calls `kv_cache.update()` and then `kv_cache.wrapper.forward()` — both using the same plan.

---

## `get_seq_length()` Returns Zero

```python
def get_seq_length(self) -> int:
    return 0
```

In Layer 6, `BatchedKVCache.get_seq_length()` returned `max_kv_len` — the model used this to compute its `past_len` offset for the additive mask and for RoPE base position calculation. In Layer 7, the additive mask is gone: FlashInfer uses `kv_indptr` for masking, not a tensor-based mask. RoPE positions are supplied via explicit `position_ids` per request in `decode_step`. There is no shared offset that the model needs to derive from the cache, so `get_seq_length()` returns 0 as a safe sentinel. The model ignores it on the FlashInfer path.
