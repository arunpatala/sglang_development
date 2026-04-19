# 05 — The Full Loop

The previous sections each explained one piece of the decode path in isolation. This section traces a complete decode step with two concurrently running requests — one long, one short — in the order the code actually executes, connecting every component by name.

---

## Setup

The scheduler is running. Request A was prefilled earlier with a 30-token prompt and has generated 20 tokens; its `PerReqKVCache` holds K/V for 50 tokens across all 28 layers. Request B was prefilled just one iteration ago with a 10-token prompt and has generated 1 token; its `PerReqKVCache` holds K/V for 11 tokens. Both are in `_running`. The scheduler's Step 2 fires: `self.model_runner.decode_step([req_A, req_B])`.

---

## Step 1 — Build Inputs

```python
B       = 2
kv_lens = [50, 11]   # req_A has 50 tokens of history, req_B has 11
max_kv  = 50         # (not used in Layer 7, but still computed for pos_ids)

last_toks = torch.tensor([[req_A.output_ids[-1]],
                           [req_B.output_ids[-1]]], device=DEVICE)  # [2, 1]

pos_ids = torch.tensor([[50], [11]], device=DEVICE)   # [2, 1]
```

`last_toks` is the most recently generated token for each request — the next token to be fed into the model. `pos_ids` is `kv_len_i` for each request: 50 for A and 11 for B. These per-request positions are passed to `apply_rotary_pos_emb` in each attention layer, giving A and B the correct RoPE encoding for their respective positions in the sequence. A shared offset like `max_kv + step` would assign the same angle to both, breaking per-request correctness. As section 04 established, this per-request position assignment has been in place since Layer 6 and is unchanged here.

---

## Step 2 — Construct `PackedKVCache` and Plan

```python
pack_kv = PackedKVCache([req_A, req_B], self._workspace)
```

Inside `__init__`:
- `kv_full_lens = [51, 12]` — history + the new token each request will contribute
- `kv_cumsum    = [0, 51, 63]`
- `qo_indptr    = [0, 1, 2]`   — one query token per request
- `kv_indptr    = [0, 51, 63]` — A's slice is tokens 0–50, B's is tokens 51–62

The FlashInfer wrapper is constructed with the workspace tensor. Then:

```python
pack_kv.plan(num_q_heads=16, num_kv_heads=8, head_dim=128, dtype=torch.bfloat16)
```

`begin_forward` runs once. FlashInfer selects the kernel for this batch shape and stores the plan. The same plan will be used 28 times — once per attention layer.

---

## Step 3 — Forward Pass (28 Attention Layers)

The model call runs with `attention_mask=None` and `kv_cache=pack_kv`. The embedding and RoPE computations happen identically to previous layers. In each `Qwen3Attention.forward`, after projecting Q/K/V and applying RoPE:

```python
# hasattr(pack_kv, "wrapper") → True → FlashInfer path
q_fi = q.squeeze(2)   # [2, 16, 128]   (HND → NHD, q_len removed)
k_fi = k.squeeze(2)   # [2,  8, 128]
v_fi = v.squeeze(2)

k_packed, v_packed = pack_kv.update(self.layer_idx, k_fi, v_fi)
```

Inside `update()` for, say, layer 0:

- Request A: `hist_k = req_A.kv_cache._k[0]` — shape `[1, 8, 50, 128]`. Reshaped to `[50, 8, 128]`. The new token `k_fi[0]` (`[8, 128]`, unsqueezed to `[1, 8, 128]`) is appended → `[51, 8, 128]`.
- Request B: `hist_k = req_B.kv_cache._k[0]` — shape `[1, 8, 11, 128]`. Reshaped to `[11, 8, 128]`. New token appended → `[12, 8, 128]`.
- `torch.cat([51-tensor, 12-tensor], dim=0)` → `k_packed` of shape `[63, 8, 128]`.

`kv_indptr = [0, 51, 63]` tells FlashInfer: A's keys are at positions 0–50, B's at 51–62.

```python
attn_out = pack_kv.wrapper.forward(q_fi, k_packed, v_packed)
# FlashInfer attends q_fi[0] over k_packed[0:51] — 51 real tokens for A
# FlashInfer attends q_fi[1] over k_packed[51:63] — 12 real tokens for B
# Output: [2, 16, 128]
attn_out = attn_out.unsqueeze(2)   # [2, 16, 1, 128]
```

No padding. No masked columns. No wasted multiply-accumulates. This repeats for all 28 layers, each calling `pack_kv.update()` with the same `kv_indptr` plan.

---

## Step 4 — `write_back`, `end_forward`, Sampling

After the 28-layer forward pass returns `logits [2, 1, vocab_size]`:

```python
pack_kv.write_back()
```

For each of the 28 layers, `write_back()` takes `self._new_k[layer_idx]` (shape `[2, 8, 128]`) and appends `new_k[0]` to `req_A.kv_cache._k[layer_idx]` and `new_k[1]` to `req_B.kv_cache._k[layer_idx]`, both reshaped to `[1, 8, 1, 128]`. After `write_back()`, req_A's cache holds 51 tokens and req_B's holds 12 — each grown by exactly one.

```python
pack_kv.end_forward()
```

FlashInfer releases its internal state. `pack_kv` goes out of scope and is garbage-collected.

```python
for i, req in enumerate([req_A, req_B]):
    next_tok = self._sample(logits[i, -1], req.temperature)
    req.output_ids.append(next_tok)
    if next_tok == self.eos_id or req.output_len >= req.max_new_tokens:
        req.status   = ReqStatus.FINISHED
        req.t_finish = time.perf_counter()
        newly_finished.append(req)
```

If req_B emits EOS (it has only `max_new_tokens=1` remaining), it is added to `newly_finished`. `decode_step` returns `[req_B]` to the scheduler. The scheduler calls `_resolve(req_B)`: `decode_output` decodes the token list to text, a result dict is built, and `loop.call_soon_threadsafe(req_B.future.set_result, result)` posts the resolution to the asyncio event loop. req_B's HTTP handler unblocks and returns. req_B is removed from `_running`. req_A continues decoding alone on the next scheduler iteration.

---

## What the Trace Shows

The full decode step for two requests with KV lengths 50 and 11 processes `51 + 12 = 63` token-attention pairs. Layer 6 would have processed `(50 + 1) × 2 = 102` — one padded column for every real one in the shorter request. The attention compute is `63/102 ≈ 62%` of what Layer 6 would have done for the same two requests. The gap grows as request age disparity increases: with lengths 1000 and 1, Layer 7 processes 1003 pairs; Layer 6 would process 2002.
