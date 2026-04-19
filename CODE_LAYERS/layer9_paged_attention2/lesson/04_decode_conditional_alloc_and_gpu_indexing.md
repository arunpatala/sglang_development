# 04 — Decode: Conditional Allocation and GPU Indexing

The decode step is where Layer 9's three optimizations — `ReqToTokenPool`, the Triton kernel, and variable `page_size` — interact. Section 03 explained how prefill populates the pool and the table. This section traces the decode step's preparation phase: the per-request arithmetic that determines whether a new page is needed, how the GPU index arrays are built without CPU loops, and how `DecodeKVCtx` conveys the (page, offset) write address to the attention backend.

---

## The Sequence-Length Formula

```python
seq_lens_list      = [len(r.input_ids) + len(r.output_ids) - 1 for r in reqs]
token_offsets_list = [sl % P                                for sl in seq_lens_list]
num_pages_list     = [len(r.slot_indices)                   for r in reqs]
```

`seq_len` is the number of tokens already in the KV cache — equivalently, the RoPE position of the current input token. It is `len(r.input_ids) + len(r.output_ids) - 1`. The subtraction of one is crucial: `r.output_ids[-1]` is the token being processed in this decode step. It was appended to `output_ids` by the previous call but has not yet been stored in the pool. The formula counts only the tokens already in the pool.

For a request with a 10-token prompt that has produced 3 tokens so far (so `output_ids = [tok1, tok2, tok3]`, processing `tok3` this step): `seq_len = 10 + 3 - 1 = 12`. Position 12 is where the current token lands. `token_offset = 12 % 16 = 12` — it will be written into offset 12 of the current last page.

`token_offset = seq_len % page_size` is the within-page position of the current token. It tells both whether a new page is needed (`token_offset == 0` means the previous last page just filled up) and where inside the last page to write.

---

## Conditional Page Allocation

```python
last_page_idx_list = []
for i, req in enumerate(reqs):
    if token_offsets_list[i] == 0:
        new_page = self.kv_pool.alloc(1)[0]
        req.slot_indices.append(new_page)
        self.req_to_token_pool.req_to_token[
            req.req_pool_idx, num_pages_list[i]
        ] = new_page
        last_page_idx_list.append(new_page)
        num_pages_list[i] += 1
    else:
        last_page_idx_list.append(req.slot_indices[-1])
```

A new page is allocated only when `token_offset == 0`. This fires once every `page_size` decode steps — every 16 steps for `page_size=16`. When it fires, `kv_pool.alloc(1)` pops one page from the free list (a Python list pop), `req.slot_indices.append` records it on the request object, and a scalar int32 assignment writes it into the GPU table at column `num_pages_list[i]`. Three operations; none of them involve GPU kernel launches or tensor allocations.

When `token_offset != 0`, the existing last page still has room. The new token will be written into `req.slot_indices[-1]` at position `token_offset`. No pool alloc, no table update, no list append. For the 15 out of every 16 decode steps where a page does not fill, this branch is a single Python list read.

`last_page_idx_list[i]` is the page where the current token will land — either the newly allocated page or the existing last page. It is the first index of the 2D pool write in `DecodeKVCtx.store`.

---

## Building GPU Index Tensors

After the O(B) Python loop, five small `[B]` tensors are transferred to the GPU:

```python
seq_lens_t        = torch.tensor(seq_lens_list,       dtype=torch.int32, device=DEVICE)
token_offsets_t   = torch.tensor(token_offsets_list,  dtype=torch.int32, device=DEVICE)
num_pages_t       = torch.tensor(num_pages_list,      dtype=torch.int32, device=DEVICE)
req_pool_idx_t    = torch.tensor(req_pool_idx_list,   dtype=torch.int32, device=DEVICE)
last_page_idx_t   = torch.tensor(last_page_idx_list,  dtype=torch.int64, device=DEVICE)
token_offsets_i64 = token_offsets_t.to(torch.int64)
```

Each tensor is `B × 4` bytes — for a batch of 16 requests, 64 bytes each. The total CPU-to-GPU transfer is under 400 bytes, independent of how many tokens have accumulated in the cache.

In Layer 8, the analogous transfer was `T + B` int32 values for `kv_indices_list` — `O(total_kv_tokens)` bytes. For B=16 requests with 500-token average history, that was `8016 × 4 = 32 KB` per step. Layer 9's transfer is the same 400 bytes regardless of history depth.

---

## `kv_indptr` on the GPU via `cumsum`

```python
self._kv_indptr_buf[0] = 0
torch.cumsum(num_pages_t, dim=0, out=self._kv_indptr_buf[1 : B + 1])
kv_indptr = self._kv_indptr_buf[: B + 1]
```

`kv_indptr` is the `[B+1]` prefix-sum that tells FlashInfer where request `i`'s slice starts in `kv_indices`: `kv_indices[kv_indptr[i] : kv_indptr[i+1]]` is request `i`'s complete page index list. In Layer 8, this was computed with `itertools.accumulate` in Python and then transferred to the GPU as a tensor. In Layer 9, `num_pages_t` is already on the GPU, so `torch.cumsum` computes the prefix sum on-device and writes into a pre-allocated buffer. No CPU computation, no new tensor allocation, no host-to-device copy of the `indptr` data.

`_kv_indptr_buf` is a `[max_batch + 1]` int32 tensor allocated once in `ModelRunner.__init__`. Slicing it with `[: B + 1]` gives a view — no allocation — that is passed directly to `begin_forward`.

---

## `kv_last_page_lens`

```python
kv_last_page_lens = token_offsets_t + 1   # [B], range 1..page_size
```

`kv_last_page_lens[i]` is the number of valid tokens in the last page of request `i` after this step's token is written. `token_offsets_t + 1` gives values in the range `[1, page_size]`. FlashInfer uses this to know how many rows in the last page contain real data. For a request at `token_offset = 0`, this is 1 — one token in a freshly allocated page. For `token_offset = 15` (the 16th token in a page), this is 16 — the page is full.

In Layer 8, `kv_last_page_lens` was always `torch.ones(B)` because every page held exactly one token. That constant is replaced here by an arithmetic operation on `token_offsets_t`, which already lives on the GPU.

---

## The Triton Kernel Call

```python
total_pages_in_batch = sum(num_pages_list)
kv_indices = torch.empty(total_pages_in_batch, dtype=torch.int32, device=DEVICE)

create_flashinfer_kv_indices_triton[(B,)](
    self.req_to_token_pool.req_to_token,
    req_pool_idx_t,
    num_pages_t,
    kv_indptr,
    None,
    kv_indices,
    self.req_to_token_pool.req_to_token.shape[1],
)
```

`kv_indices` is a freshly allocated int32 tensor with `total_pages_in_batch` entries. The Triton kernel fills it on the GPU by reading from `req_to_token` — all B requests in parallel, one threadblock each. Section 05 explains the kernel's logic in detail.

After the kernel returns, `kv_indices` is the flat concatenation of every active request's page index list, in the same layout as Layer 8's `kv_indices_list` — just with `ceil(kv_tokens / page_size)` entries per request instead of one per token.

---

## `DecodeKVCtx` and `begin_forward`

```python
self._decode_wrapper.begin_forward(
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    cfg.num_attention_heads,
    cfg.num_key_value_heads,
    cfg.head_dim,
    P,               # page_size — 16, not 1
    data_type   = DTYPE,
    q_data_type = DTYPE,
)

ctx = DecodeKVCtx(
    wrapper           = self._decode_wrapper,
    k_pool            = self.kv_pool.k_pool,
    v_pool            = self.kv_pool.v_pool,
    last_page_indices = last_page_idx_t,
    token_offsets     = token_offsets_i64,
)
fb = ForwardBatch(mode=ForwardMode.DECODE, kv_cache=ctx, attention_mask=None)
```

`begin_forward` is called once per decode step. The `_decode_wrapper` instance was created once in `ModelRunner.__init__`; it is not reconstructed each step. Reuse avoids the CUDA object initialization overhead that Layer 8 incurred by constructing a new `BatchDecodeWithPagedKVCacheWrapper` every step.

`DecodeKVCtx` replaces Layer 8's `new_slots: [B]` index with the pair `last_page_indices: [B]` and `token_offsets: [B]`. This enables the 2D pool write in `store`:

```python
def store(self, layer_idx: int, k_fi: torch.Tensor, v_fi: torch.Tensor) -> None:
    self.k_pool[layer_idx][self.last_page_indices, self.token_offsets] = k_fi
    self.v_pool[layer_idx][self.last_page_indices, self.token_offsets] = v_fi
```

`k_pool[layer_idx][last_page_indices, token_offsets]` is 2D advanced indexing: for each `i`, write `k_fi[i]` into `k_pool[layer_idx][last_page_indices[i], token_offsets[i]]`. This selects the correct within-page cell for each request's new token simultaneously. The `wrapper.forward` call that follows reads the new token from this cell via the `kv_indices` plan — the correctness requirement is the same as Layer 8: the pool write must complete before `wrapper.forward` executes.
