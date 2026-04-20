# 05 — prefill_batch: The Extend Kernel

`prefill_batch` is the unified entry point for all first-time token processing in Layer 11. It handles fresh requests (full prompts), continuation chunks (mid-prefill requests), and batches of both at once. The function is structured as eight sequential steps that transform scheduler-prepared request objects into a single FlashInfer `BatchPrefillWithPagedKVCacheWrapper` call.

---

## Step 1: Allocate req_to_token Rows for New Requests

```python
# model_runner.py — prefill_batch step 1
for req in reqs:
    if req.req_pool_idx is None:
        req.req_pool_idx = self.req_to_token_pool.alloc()
```

A `req_pool_idx` is needed before `compute_write_info` can write page indices into `req_to_token`. For continuation chunks, `req_pool_idx` was already assigned on the first chunk's call — the condition guards against re-allocation.

---

## Step 2: Page Packing via compute_write_info

```python
# model_runner.py — prefill_batch step 2
write_infos: List[WriteInfo] = []
for req in reqs:
    wi = compute_write_info(
        kv_pool          = self.kv_pool,
        rtp              = self.req_to_token_pool,
        slot_indices     = req.slot_indices,
        req_pool_idx     = req.req_pool_idx,
        kv_committed_len = req.kv_committed_len,
        n_fill           = req.extend_input_len,
    )
    write_infos.append(wi)
```

As section 04 explains, this step handles the partial-page continuation case and updates `slot_indices` and `req_to_token` in-place. After this loop, every request's `slot_indices` reflects all pages — prefix and new — and the GPU table is consistent.

---

## Steps 3–4: Pack Token IDs and Position IDs

```python
# model_runner.py — steps 3-4
all_ids: List[int] = []
for req in reqs:
    all_ids.extend(req.fill_ids)

ids_t = torch.tensor(all_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
# shape: [1, total_tokens]

pos_list: List[int] = []
for req in reqs:
    for j in range(req.extend_input_len):
        pos_list.append(req.kv_committed_len + j)
pos_t = torch.tensor(pos_list, dtype=torch.long, device=DEVICE).unsqueeze(0)
# shape: [1, total_tokens]
```

Position IDs start at `kv_committed_len + j`, not at zero. For a fresh request, `kv_committed_len = 0` and the positions are `0, 1, 2, ...` as usual. For a continuation chunk with `kv_committed_len = 512`, positions are `512, 513, 514, ...`. Rotary embeddings in the model use these IDs to compute the correct positional frequencies; an off-by-one here would cause every layer's attention to compute wrong RoPE angles for the continuation tokens.

`ids_t` is `[1, total_tokens]` — a single sequence containing all requests' tokens concatenated. The `[1, ...]` batch dimension is a historical convention from the F.sdpa-based prefill; `qo_indptr` tells the extend kernel where each request's tokens start and end within this flat sequence.

---

## Step 5: Build qo_indptr, kv_indptr, kv_last_page_lens

```python
# model_runner.py — step 5
B = len(reqs)
qo_indptr_list  = [0]
num_pages_list  = []
kv_last_pg_list = []

for req in reqs:
    qo_indptr_list.append(qo_indptr_list[-1] + req.extend_input_len)
    total_committed = req.kv_committed_len + req.extend_input_len
    n_pages = len(req.slot_indices)
    num_pages_list.append(n_pages)
    last_fill = total_committed % P
    kv_last_pg_list.append(last_fill if last_fill != 0 else P)

qo_indptr_t       = torch.tensor(qo_indptr_list,  dtype=torch.int32, device=DEVICE)
kv_last_page_lens = torch.tensor(kv_last_pg_list, dtype=torch.int32, device=DEVICE)

kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=DEVICE)
torch.cumsum(num_pages_t, dim=0, out=kv_indptr[1:])
```

`qo_indptr [B+1]` holds the token offsets into `ids_t` per request: request 0 occupies `ids_t[0 : qo_indptr[1]]`, request 1 occupies `ids_t[qo_indptr[1] : qo_indptr[2]]`, and so on. This is what FlashInfer uses to split the packed query tensor into per-request slices.

`kv_indptr [B+1]` holds the page offsets into `kv_indices`. It is built from `num_pages_list` (the number of pages per request after the new allocation) via `cumsum`. This is identical to the decode-path `kv_indptr` construction from Layer 9.

`kv_last_page_lens [B]` communicates the fill level of each request's last page after this extend round. The formula is `total_committed % P`, defaulting to `P` when the page is exactly full. This differs from the decode path where `kv_last_page_lens = token_offsets + 1` (range 1..P based on the current token's within-page position).

---

## Step 6: Build kv_indices via Triton Kernel

```python
# model_runner.py — step 6
total_pages_batch = int(num_pages_t.sum().item())
kv_indices = torch.empty(total_pages_batch, dtype=torch.int32, device=DEVICE)

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

The same Triton kernel used in the decode path reads `req_to_token` on-GPU and fills `kv_indices`. After the `compute_write_info` calls in step 2, `req_to_token` holds all pages — both prefix pages carried over from prior chunks and newly allocated pages. The Triton kernel copies each request's row slice into the corresponding segment of `kv_indices`, making the full page history available to FlashInfer.

---

## Steps 7–8: begin_forward, Forward Pass, end_forward

```python
# model_runner.py — step 7
self._extend_wrapper.begin_forward(
    qo_indptr_t, kv_indptr, kv_indices, kv_last_page_lens,
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
    P, causal=True, q_data_type=DTYPE,
)

ctx = ExtendKVCtx(
    wrapper     = self._extend_wrapper,
    k_pool      = self.kv_pool.k_pool,
    v_pool      = self.kv_pool.v_pool,
    qo_indptr   = qo_indptr_list,
    write_infos = write_infos,
    page_size   = P,
)

with torch.no_grad():
    logits = self.model(ids_t, attention_mask=None, kv_cache=ctx, position_ids=pos_t)
# logits shape: [1, total_tokens, vocab_size]

self._extend_wrapper.end_forward()
```

`begin_forward` with `causal=True` applies a causal mask across all positions within each request's token block. Requests do not attend to each other's tokens — `qo_indptr` partitions the query sequence and FlashInfer handles each request's causal block independently. The `attention_mask=None` passed to the model reflects this: causality is enforced by FlashInfer, not by a manually constructed additive mask.

After `end_forward`, step 8 updates `kv_committed_len`, checks `is_last_chunk`, and either samples the first output token (last chunk) or sets status to `PREFILLING` (more chunks remain):

```python
# model_runner.py — step 8
for i, req in enumerate(reqs):
    req.kv_committed_len += req.extend_input_len

    if not req.is_last_chunk:
        req.status = ReqStatus.PREFILLING
        continue

    last_tok_pos = qo_indptr_list[i + 1] - 1
    next_tok = self._sample(logits[0, last_tok_pos], req.temperature)
    req.output_ids.append(next_tok)
    ...
```

`qo_indptr_list[i + 1] - 1` is the index of the last token in request `i`'s block within `logits[0, :, :]`. For a batch with two requests of lengths 512 and 400, request 0's last logit is at position 511 and request 1's is at position 911. Sampling from the wrong position would produce a garbage first output token.

Section 06 explains what happens inside the model during the forward pass — specifically, how `PagedExtendBackend._extend_forward` handles the K/V writes and the FlashInfer paged-prefill attention call.
