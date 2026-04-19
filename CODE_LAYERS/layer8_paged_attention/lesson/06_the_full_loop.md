# 06 — The Full Loop

The previous sections each explained one piece of the Layer 8 decode path in isolation. This section traces two concurrently running requests — one with a long history, one recently prefilled — through a complete decode step in the order the code actually executes, connecting every component by name.

---

## Setup

The scheduler is running. Request A was prefilled earlier with a 30-token prompt and has generated 20 tokens; `req_A.slot_indices` holds 50 entries — the pool rows for its 30 prompt tokens plus 20 decode tokens across all 28 layers. Request B was prefilled just one iteration ago with a 10-token prompt and has generated 1 token; `req_B.slot_indices` holds 11 entries. Both requests are in `_running`. The scheduler's decode step fires: `self.model_runner.decode_step([req_A, req_B])`.

---

## Step 1 — Allocate Slots and Build Inputs

```python
B         = 2
new_slots = [self.kv_pool.alloc(1)[0], self.kv_pool.alloc(1)[0]]
# e.g. new_slots = [1203, 847]  — two fresh rows from the free list

pos_ids = torch.tensor([[50], [11]], device=DEVICE)   # [2, 1]
last_toks = torch.tensor(
    [[req_A.output_ids[-1]], [req_B.output_ids[-1]]], device=DEVICE
)   # [2, 1]
```

As section 02 established, `alloc(1)` is a Python list pop — no GPU work. `pos_ids` is each request's current total token count: 50 for A and 11 for B. This per-request absolute RoPE position ensures that request A's new token is encoded at position 50 and request B's at position 11. A shared position like `max_kv + step` would encode both at position 51, assigning the wrong rotary angle to request B's query. The per-request position assignment has been in place since Layer 6.

---

## Step 2 — Build the Index Arrays

```python
kv_lens_plus1   = [51, 12]   # len(req_A.slot_indices)+1, len(req_B.slot_indices)+1
kv_indptr_list  = [0, 51, 63]
kv_indices_list = [*req_A.slot_indices, 1203,   # A's history + A's new slot
                   *req_B.slot_indices, 847]    # B's history + B's new slot

kv_indptr         = torch.tensor([0, 51, 63],   dtype=torch.int32, device=DEVICE)
kv_indices        = torch.tensor(kv_indices_list, dtype=torch.int32, device=DEVICE)
# kv_indices has 63 entries: A's 50 + 1 new, B's 11 + 1 new
kv_last_page_lens = torch.ones(2, dtype=torch.int32, device=DEVICE)
```

The new slots (1203 and 847) are included in `kv_indices` before any GPU work happens. As section 04 established, this is mandatory: `DecodeKVCtx.store()` will write K/V to these rows inside the forward pass, and FlashInfer reads from the pool immediately via `kv_indices`. The new rows must be present in the index before `wrapper.forward()` reads them.

`kv_indptr = [0, 51, 63]` declares that request A's KV slice is `kv_indices[0:51]` — its 50 historical slots plus its new slot — and request B's is `kv_indices[51:63]`. FlashInfer will attend A's query over pool rows at positions `kv_indices[0:51]` and B's query over pool rows at positions `kv_indices[51:63]`.

---

## Step 3 — Plan and Construct Contexts

```python
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self._workspace, "NHD")
decode_wrapper.begin_forward(
    kv_indptr, kv_indices, kv_last_page_lens,
    16, 8, 128, 1, data_type=DTYPE, q_data_type=DTYPE
)

ctx = DecodeKVCtx(wrapper=decode_wrapper, k_pool=kv_pool.k_pool,
                  v_pool=kv_pool.v_pool, new_slots=torch.tensor([1203, 847]))
fb  = ForwardBatch(mode=ForwardMode.DECODE, kv_cache=ctx, attention_mask=None)
```

`begin_forward` runs once. It ingests the three index arrays and the head configuration, selects the kernel variant (bfloat16, head_dim=128, GQA-2, page_size=1), and stores the plan internally. The same plan is used by all 28 calls to `wrapper.forward()` during the forward pass.

`DecodeKVCtx` carries references — not copies — to the pool lists and the planned wrapper. Constructing it allocates no GPU memory.

---

## Step 4 — The 28-Layer Forward Pass

The model call begins. Embeddings and RoPE computations run as in all previous layers. In each `Qwen3Attention.forward`, after projecting Q/K/V and applying RoPE:

```python
attn_out = self.backend.forward(q, k, v, self.layer_idx, forward_batch)
```

`PagedBackend.forward` reads `forward_batch.mode == ForwardMode.DECODE` and calls `_decode_forward`. For layer 0:

```python
q_fi = q.squeeze(2)   # [2, 16, 128]
k_fi = k.squeeze(2)   # [2,  8, 128]
v_fi = v.squeeze(2)

ctx.store(0, k_fi, v_fi)
# Writes: k_pool[0][1203] = k_fi[0], k_pool[0][847] = k_fi[1]
# v_pool[0] similarly.

k_paged = kv_pool.k_pool[0].unsqueeze(1)   # [total_slots, 1, 8, 128]
v_paged = kv_pool.v_pool[0].unsqueeze(1)

attn_out = ctx.wrapper.forward(q_fi, (k_paged, v_paged))
# FlashInfer attends q_fi[0] over pool rows kv_indices[0:51] — rows for A's 51 tokens
# FlashInfer attends q_fi[1] over pool rows kv_indices[51:63] — rows for B's 12 tokens
# Output: [2, 16, 128]

attn_out = attn_out.unsqueeze(2)   # [2, 16, 1, 128]
```

No float K/V data is gathered or copied. FlashInfer reads directly from `k_pool[0]` and `v_pool[0]` at the rows listed in `kv_indices`. Request A's attention covers 51 real token positions; request B's covers 12. No padding, no masked columns, no wasted multiply-accumulates over zeros.

This repeats for layers 1 through 27. The same `kv_indptr` plan from `begin_forward` is used every time. `ctx.store(layer_idx, ...)` writes K/V for the new token at the corresponding layer row of slots 1203 and 847 on each iteration.

---

## Step 5 — `end_forward`, Slot Append, and Sampling

```python
decode_wrapper.end_forward()

for i, req in enumerate([req_A, req_B]):
    req.slot_indices.append(new_slots[i])
# req_A.slot_indices now has 51 entries; req_B has 12

for i, req in enumerate([req_A, req_B]):
    next_tok = self._sample(logits[i, -1], req.temperature)
    req.output_ids.append(next_tok)

    if next_tok == self.eos_id or req.output_len >= req.max_new_tokens:
        req.status   = ReqStatus.FINISHED
        req.t_finish = time.perf_counter()
        newly_finished.append(req)
        self.kv_pool.free(req.slot_indices)   # return all 51 (or 12) slots at once
```

`end_forward()` releases FlashInfer's planning state. The wrapper goes out of scope; the workspace memory is still reserved (it is permanent on `ModelRunner`).

The slot append is a Python integer list operation. Slot 1203 joins `req_A.slot_indices`; slot 847 joins `req_B.slot_indices`. No tensor is allocated; no GPU synchronization occurs.

If request B emits EOS here, `kv_pool.free(req_B.slot_indices)` immediately returns all 12 of its slots to the free list. The slots are available to the next `prefill` call before `decode_step` returns — no GC cycle, no deferred deallocation. Request B is removed from `_running`. The scheduler calls `_resolve(req_B)`, decodes the output tokens to text, and posts the result to the asyncio event loop. The HTTP handler for request B's connection unblocks and returns. Request A continues decoding alone on the next scheduler iteration with `slot_indices` of length 51.

---

## What the Trace Shows

A decode step over requests with 50 and 11 accumulated tokens touches 51 + 12 = 63 pool rows per layer — 63 token-attention pairs per layer, no more. In Layer 7, the same step would have assembled a ragged buffer of 63 float rows from the per-request caches, costing `63 × 28 × 2 × 8 × 128 × 2 = 57.7 MB` of float reads. In Layer 8, the same 63 pool rows are read by FlashInfer directly — no assembly step, no `torch.cat`, no `write_back`. The rows were written once during prefill and each decode step, and they are read in place for the lifetime of the request.
