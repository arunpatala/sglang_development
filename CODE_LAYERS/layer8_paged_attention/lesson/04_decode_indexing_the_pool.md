# 04 — Decode: Indexing the Pool

The decode step is where Layer 8's paged design pays off. The step's job is to run B concurrent requests, each at a different position in its sequence, over K/V histories of different lengths. In Layer 7, the step assembled a contiguous ragged float buffer from each request's `PerReqKVCache` before FlashInfer could attend over it. In Layer 8, the same histories already live in the pool; the step's only preparation is to build an integer index that tells FlashInfer which pool rows belong to each request.

---

## Building the Index

```python
# model_runner.decode_step — index construction
new_slots = [self.kv_pool.alloc(1)[0] for _ in reqs]

kv_lens_plus1  = [len(r.slot_indices) + 1 for r in reqs]
kv_indptr_list = [0] + list(accumulate(kv_lens_plus1))

kv_indices_list: List[int] = []
for i, req in enumerate(reqs):
    kv_indices_list.extend(req.slot_indices)   # historical slots (already in pool)
    kv_indices_list.append(new_slots[i])        # new token's slot (will be written below)

kv_indptr         = torch.tensor(kv_indptr_list,  dtype=torch.int32, device=DEVICE)
kv_indices        = torch.tensor(kv_indices_list, dtype=torch.int32, device=DEVICE)
kv_last_page_lens = torch.ones(B, dtype=torch.int32, device=DEVICE)
```

`kv_indices` is the flat concatenation of every request's slot list plus its new slot — the full token history for every active request, expressed as pool row indices. The `kv_indptr` cumulative-sum tells FlashInfer where request `i`'s slice starts and ends: `kv_indices[kv_indptr[i]:kv_indptr[i+1]]` is request `i`'s complete KV index. For a batch of two requests with 50 and 11 accumulated tokens respectively, `kv_indptr = [0, 51, 63]` and `kv_indices` has 63 entries.

The new token's slot is included in `kv_indices` before the forward pass. This is mandatory: `DecodeKVCtx.store()` will write the new token's K/V to that slot inside the forward pass, and FlashInfer reads from the pool immediately after. The new token must be in the pool before `wrapper.forward()` sees it, or it will read zeros from the pool row and produce incorrect attention output.

`kv_last_page_lens` is always `ones(B)` because `page_size=1`. With one token per page, every page has exactly one occupant, so the last page of every request is always fully occupied — fill count 1 out of a capacity of 1. This changes in Layer 9 when `page_size` becomes larger than 1 and the last page of a request whose history is not a multiple of `page_size` is only partially filled.

---

## `begin_forward` — Planning the Kernel

```python
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    self._workspace, "NHD", use_tensor_cores=False
)
decode_wrapper.begin_forward(
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    cfg.num_attention_heads,   # 16  (num_qo_heads)
    cfg.num_key_value_heads,   # 8   (num_kv_heads)
    cfg.head_dim,              # 128
    1,                         # page_size
    data_type   = DTYPE,
    q_data_type = DTYPE,
)
```

`begin_forward` runs once per decode step. It takes the three index arrays plus the attention head configuration and selects the appropriate kernel variant: data type (bfloat16), head dimension (128), GQA ratio (16Q/8KV = 2), and page size (1). The plan is stored internally in the wrapper and reused by every `wrapper.forward()` call during the 28-layer forward pass — `begin_forward` is not called 28 times.

`use_tensor_cores=False` is set explicitly because Qwen3-0.6B's GQA ratio of 2 (16 Q heads sharing 8 KV heads) falls below the threshold of 4 where tensor-core acceleration benefits the decode path. Above ratio 4, tensor cores are faster because the kernel can amortize their setup cost over more Q-head groups; below 4, the fused CUDA cores kernel is lower latency.

The "NHD" layout string declares that the query tensor will be presented to `wrapper.forward()` in NHD order — `[batch, n_heads, head_dim]` — and that the pool tensors are also in this layout. This matches the `[total_slots, n_kv_heads, head_dim]` shape of `k_pool[layer]`.

---

## `DecodeKVCtx` — The Decode Context Object

```python
new_slots_t = torch.tensor(new_slots, dtype=torch.int64, device=DEVICE)
ctx = DecodeKVCtx(
    wrapper   = decode_wrapper,
    k_pool    = self.kv_pool.k_pool,
    v_pool    = self.kv_pool.v_pool,
    new_slots = new_slots_t,
)
fb = ForwardBatch(mode=ForwardMode.DECODE, kv_cache=ctx, attention_mask=None)
```

`DecodeKVCtx` carries four items. **`wrapper`** is the already-planned `BatchDecodeWithPagedKVCacheWrapper`; every attention layer calls `ctx.wrapper.forward()` with it. **`k_pool`** and **`v_pool`** are the full lists of pool tensors — one tensor per layer — passed by reference, not copied. **`new_slots`** is a `[B]` int64 tensor of the freshly allocated slot indices, one per request, used by `ctx.store()` to write the new decode token.

`attention_mask` is `None` in `ForwardBatch` on the decode path. FlashInfer uses `kv_indices` from `begin_forward` to locate each request's KV slice; no additive mask tensor is needed or constructed. As established in section 03, the mask is only needed during prefill where `F.sdpa` runs and requires it to be in additive form.

Inside `DecodeKVCtx.store()`:

```python
def store(self, layer_idx: int, k_fi: torch.Tensor, v_fi: torch.Tensor) -> None:
    # k_fi: [B, n_kv_heads, head_dim]  (squeezed q_len=1 dim from attention.py)
    self.k_pool[layer_idx][self.new_slots] = k_fi
    self.v_pool[layer_idx][self.new_slots] = v_fi
```

`k_fi` arrives already in `[B, n_kv_heads, head_dim]` shape — the q_len dimension has been squeezed in `PagedBackend._decode_forward` before `store` is called. `self.new_slots` is a `[B]` int64 tensor. The advanced-index assignment writes each request's new K/V row into its designated pool slot simultaneously. This is `B` rows written in one operation — the GPU scatter equivalent of the per-request write, fully vectorized.

---

## After the Forward Pass

```python
decode_wrapper.end_forward()

for i, req in enumerate(reqs):
    req.slot_indices.append(new_slots[i])   # integer list append
```

`end_forward()` releases FlashInfer's internal planning state for this step. The wrapper can be reconstructed cheaply on the next call to `decode_step` — it is not persistent across steps.

The slot append is a plain Python integer list operation. No tensor is allocated; no GPU memory is touched. This is the only bookkeeping the request object needs: its `slot_indices` grows by exactly one integer, recording the pool address where this step's K/V was written. The next call to `decode_step` will extend `kv_indices_list` with this new entry, and FlashInfer will find the token where `store` placed it.
