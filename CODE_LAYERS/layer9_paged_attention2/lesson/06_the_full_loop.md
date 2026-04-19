# 06 — The Full Loop

The previous sections explained each Layer 9 component in isolation. This section traces two concurrently running requests — one with a long history, one recently prefilled — through a complete decode step, connecting every component in the order the code executes.

---

## Setup

The scheduler is running. Request A was prefilled with a 30-token prompt and has generated 18 tokens; its `output_ids` has 19 entries (18 generated plus the first token returned by prefill). The current decode step is processing `output_ids[18]` — the token generated last step. `len(r.input_ids) + len(r.output_ids) - 1 = 30 + 19 - 1 = 48` tokens are already in the KV cache; the current token lands at position 48. With `page_size=16`: `num_pages_A = ceil(48/16) = 3` pages occupied at the start of this step. `token_offset_A = 48 % 16 = 0` — the last page just filled; a new page is needed.

Request B was prefilled with a 10-token prompt and has generated 5 tokens; `output_ids` has 6 entries, processing `output_ids[5]`. `seq_len_B = 10 + 6 - 1 = 15`. `num_pages_B = ceil(15/16) = 1` page. `token_offset_B = 15 % 16 = 15` — the last page has 15 tokens; the new token is the 16th, filling the page.

Both are in `_running`. The scheduler fires `model_runner.decode_step([req_A, req_B])`.

---

## Step 1 — Compute Per-Request Metadata

```python
B = 2
seq_lens_list      = [48, 15]
token_offsets_list = [0,  15]
num_pages_list     = [3,  1 ]
req_pool_idx_list  = [req_A.req_pool_idx, req_B.req_pool_idx]  # e.g. [2, 7]
```

This is the O(B) Python loop — two list reads per request, no iteration over slot lists. `token_offset_A = 0` signals that request A needs a new page. `token_offset_B = 15` signals that request B's last page has 15 existing tokens and this step will fill it.

---

## Step 2 — Conditional Page Allocation

```python
# i=0, req_A: token_offset == 0 → alloc new page
new_page_A = self.kv_pool.alloc(1)[0]   # e.g. page 412
req_A.slot_indices.append(412)
self.req_to_token_pool.req_to_token[2, 3] = 412   # scalar GPU write
last_page_idx_list.append(412)
num_pages_list[0] = 4

# i=1, req_B: token_offset == 15 → existing page suffices
last_page_idx_list.append(req_B.slot_indices[-1])   # e.g. page 77
# no alloc, no table write
```

For request A: one Python list pop (`kv_pool.free_slots.pop()`), one Python list append, one scalar int32 GPU write. No kernel launch, no tensor allocation. For request B: one Python list read. The GPU table for request B is unchanged — `req_to_token[7, 0] = 77` still holds the correct last page from prefill.

After this step: `num_pages_list = [4, 1]`. `last_page_idx_list = [412, 77]`.

---

## Step 3 — Build GPU Tensors and `kv_indptr`

```python
seq_lens_t      = tensor([48, 15], int32)
token_offsets_t = tensor([0,  15], int32)
num_pages_t     = tensor([4,  1 ], int32)
req_pool_idx_t  = tensor([2,  7 ], int32)
last_page_idx_t = tensor([412, 77], int64)
token_offsets_i64 = tensor([0, 15], int64)

kv_last_page_lens = token_offsets_t + 1   # tensor([1, 16], int32)
```

`kv_last_page_lens[0] = 1`: request A's new page has 1 valid token (the one being written this step). `kv_last_page_lens[1] = 16`: request B's last page is completely full after this write.

```python
self._kv_indptr_buf[0] = 0
torch.cumsum(num_pages_t, dim=0, out=self._kv_indptr_buf[1:3])
# _kv_indptr_buf[:3] = [0, 4, 5]
kv_indptr = self._kv_indptr_buf[:3]
```

`cumsum([4, 1]) = [4, 5]`. `kv_indptr = [0, 4, 5]`: request A's KV slice is `kv_indices[0:4]` (its 4 pages); request B's is `kv_indices[4:5]` (its 1 page). Total pages in batch: 5.

---

## Step 4 — Triton Kernel: Fill `kv_indices`

```python
kv_indices = torch.empty(5, dtype=torch.int32, device=DEVICE)

create_flashinfer_kv_indices_triton[(2,)](
    req_to_token_pool.req_to_token,   # [128, 256] on GPU
    req_pool_idx_t,                   # [2]  = [2, 7]
    num_pages_t,                      # [2]  = [4, 1]
    kv_indptr,                        # [3]  = [0, 4, 5]
    None, kv_indices, 256,
)
```

Two Triton programs run in parallel.

Program 0 (request A, `pid=0`): reads `req_pool_index=2`, `kv_indices_offset=0`, `kv_end=4`. Copies `req_to_token[2, 0:4]` → `kv_indices[0:4]`. For example, `req_to_token[2, :] = [101, 205, 307, 412, 0, 0, ...]`; after the copy, `kv_indices[0:4] = [101, 205, 307, 412]`.

Program 1 (request B, `pid=1`): reads `req_pool_index=7`, `kv_indices_offset=4`, `kv_end=1`. Copies `req_to_token[7, 0:1]` → `kv_indices[4:5]`. `req_to_token[7, 0] = 77`; after the copy, `kv_indices[4] = 77`.

Result: `kv_indices = [101, 205, 307, 412, 77]`. Five page indices, produced on the GPU with no CPU loops and no PCIe transfer of index data.

---

## Step 5 — `begin_forward` and Context Construction

```python
self._decode_wrapper.begin_forward(
    kv_indptr,          # [0, 4, 5]
    kv_indices,         # [101, 205, 307, 412, 77]
    kv_last_page_lens,  # [1, 16]
    16, 8, 128, 16,     # num_qo_heads, num_kv_heads, head_dim, page_size=16
    data_type=bfloat16, q_data_type=bfloat16,
)

ctx = DecodeKVCtx(
    wrapper=self._decode_wrapper,
    k_pool=kv_pool.k_pool, v_pool=kv_pool.v_pool,
    last_page_indices=tensor([412, 77]),
    token_offsets=tensor([0, 15]),
)
fb = ForwardBatch(mode=ForwardMode.DECODE, kv_cache=ctx, attention_mask=None)
```

`begin_forward` ingests the five-element `kv_indices` and two-element `kv_indptr`. It selects the kernel variant (bfloat16, head_dim=128, GQA-2, page_size=16) and stores the plan. This plan is shared across all 28 attention layer calls.

`DecodeKVCtx` carries `last_page_indices=[412, 77]` and `token_offsets=[0, 15]`. These are the 2D write coordinates: request A's new K/V goes into `k_pool[layer][412, 0]` and request B's into `k_pool[layer][77, 15]`.

---

## Step 6 — The 28-Layer Forward Pass

The model call begins. Embeddings and RoPE run as in all previous layers. In each `Qwen3Attention.forward`:

```python
attn_out = self.backend.forward(q, k, v, self.layer_idx, forward_batch)
```

`PagedBackend.forward` reads `mode == ForwardMode.DECODE` and calls `_decode_forward`. For layer 0:

```python
q_fi = q.squeeze(2)    # [2, 16, 128]
k_fi = k.squeeze(2)    # [2,  8, 128]
v_fi = v.squeeze(2)

ctx.store(0, k_fi, v_fi)
# k_pool[0][412, 0] = k_fi[0]   → request A's new token at offset 0 of page 412
# k_pool[0][77,  15] = k_fi[1]  → request B's new token at offset 15 of page 77
# v_pool[0] identically.

k_paged = kv_pool.k_pool[0]   # [total_pages, 16, 8, 128]
v_paged = kv_pool.v_pool[0]

attn_out = ctx.wrapper.forward(q_fi, (k_paged, v_paged))
# FlashInfer attends q_fi[0] over pages kv_indices[0:4] = [101,205,307,412]
#   page 101: tokens 0–15, all valid (full page)
#   page 205: tokens 16–31, all valid
#   page 307: tokens 32–47, all valid
#   page 412: token 0 only (kv_last_page_lens[0]=1)
# FlashInfer attends q_fi[1] over pages kv_indices[4:5] = [77]
#   page 77: tokens 0–14 existing + token 15 just written (kv_last_page_lens[1]=16)

attn_out = attn_out.unsqueeze(2)   # [2, 16, 1, 128]
```

Request A's query attends over 48 real token positions: 3 full pages of 16 tokens plus 1 new token in a fresh page. Request B's query attends over 16 real token positions: 15 pre-existing tokens plus the 1 just written. No padding, no wasted attention over zeros, no float gather.

This repeats for layers 1 through 27. The same `begin_forward` plan is reused. `ctx.store(layer_idx, ...)` writes each layer's K/V to the same pages (412 and 77) but different pool tensors (`k_pool[1]`, `k_pool[2]`, etc.).

---

## Step 7 — `end_forward`, Slot Update, and Sampling

```python
self._decode_wrapper.end_forward()

# No slot appends this step for request B (token_offset_B was 15, not filling next step)
# req_A.slot_indices was already extended in Step 2 — page 412 appended there

for i, req in enumerate([req_A, req_B]):
    next_tok = sample_token(logits[i, -1], req.temperature)
    req.output_ids.append(next_tok)

    if next_tok == eos or req.output_len >= req.max_new_tokens:
        req.status   = ReqStatus.FINISHED
        self.kv_pool.free(req.slot_indices)
        self.req_to_token_pool.free(req.req_pool_idx)
```

Suppose request B emits EOS. `kv_pool.free([77])` returns page 77 to the free list immediately — one Python list extend, no GPU work, no GC cycle. `req_to_token_pool.free(7)` returns row 7 to the pool row free list. Both are available to the next `prefill` call before `decode_step` returns.

Request A continues with `slot_indices = [101, 205, 307, 412]` (4 pages, 48 tokens stored). Its next decode step will see `seq_len = 30 + 20 - 1 = 49`, `token_offset = 49 % 16 = 1`, `num_pages = 4`. Page 412 still has room — no new alloc, no table write. The new token lands at `k_pool[layer][412, 1]`.

---

## What the Trace Shows

A decode step over requests with 48 and 15 accumulated tokens sends 5 page indices to the GPU-side FlashInfer call (via the Triton kernel), instead of the 65 token-level slot indices that Layer 8 would have assembled in Python. The CPU-to-GPU data footprint is five `[B]=2` integer tensors totalling under 100 bytes, versus Layer 8's `torch.tensor(kv_indices_list)` call that would have sent 65 × 4 = 260 bytes of token indices for this small example — and scales to tens of kilobytes for larger batches and histories.

The pool write touches `k_pool[layer][412, 0]` and `k_pool[layer][77, 15]` — two within-page cells per layer, not two new pool rows. FlashInfer reads `4 + 1 = 5` page rectangles of `[16, 8, 128]` float16 values per layer, directly from the pool, with no gather step.
