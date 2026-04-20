# 07 — The Full Loop

The previous sections explained each component of Layer 11 in isolation. This section traces a complete two-round scheduler sequence: a long request being chunked while two shorter requests decode in parallel.

---

## Setup

The server is running. Two requests — R1 and R2 — are in `_running`, each having generated 10 tokens. A third request R3 arrives with a 1024-token prompt. `chunked_prefill_size = 512`, `max_prefill_tokens = 2048`, `page_size = 16`, `max_running_reqs = 16`.

`_chunked_req = None`. `_waiting` contains R3.

---

## Round 1 — First Chunk of R3 + Decode of R1 and R2

**Step 1 — PrefillAdder.build():**

`_chunked_req is None`, so case 2 fires. The loop peeks at R3 (`prompt_len=1024 > chunked_prefill_size=512`). The chunked path is taken: `req.fill_ids = input_ids[0:512]`, `req.extend_input_len = 512`, `new_chunked_req = R3`, `batch = [R3]`, loop breaks.

`prefill_batch = [R3]` is returned. `R3.kv_committed_len = 0`.

**Step 2 — prefill_batch([R3]):**

`compute_write_info(kv_pool, rtp, R3.slot_indices=[], R3.req_pool_idx=<new>, kv_committed_len=0, n_fill=512)`:
- `n_leftover = 0 % 16 = 0`, no existing partial page.
- `remaining = 512`, allocate `ceil(512/16) = 32` new pages.
- `R3.slot_indices = [p0, p1, ..., p31]`, `req_to_token[row, 0:32] = [p0..p31]`.
- Returns `WriteInfo(existing_page=None, n_leftover=0, existing_slots=0, new_pages=[p0..p31])`.

All 512 token IDs are packed into `ids_t [1, 512]`. Position IDs are `[0, 1, ..., 511]`.

`qo_indptr = [0, 512]`. `kv_indptr = [0, 32]`. `kv_last_page_lens = [512 % 16 = 0 → use 16] = [16]`.

Wait — 512 is a multiple of 16, so the last page is exactly full: `last_fill = 512 % 16 = 0`, so `kv_last_pg = P = 16`.

The Triton kernel reads `req_to_token[row, 0:32]` and fills `kv_indices = [p0..p31]`.

`begin_forward(qo_indptr=[0,512], kv_indptr=[0,32], kv_indices=[p0..p31], kv_last_page_lens=[16], ...)`.

Inside the model, each of the 28 attention layers calls `ctx.store(layer_idx, k, v)` — writing 32 pages of K/V into the pool — then `extend_wrapper.forward(q_fi, (k_pool[layer], v_pool[layer]))` — computing causal self-attention over 512 tokens.

`end_forward()`. `R3.kv_committed_len = 0 + 512 = 512`. `R3.is_last_chunk = (512 >= 1024)? No`. `R3.status = PREFILLING`.

**Step 3 — Update _chunked_req:**

`adder.new_chunked_req = R3`, so `self._chunked_req = R3`.

R3 stays in `PREFILLING`; it does not join `_running`.

**Step 4 — Decode of R1 and R2:**

`decode_step([R1, R2])` runs identically to Layer 9: conditional page alloc, Triton kv_indices build, `begin_forward`, 28-layer forward, `end_forward`, sampling. R1 and R2 each advance by one token. If either finishes, it calls `kv_pool.free` and `req_to_token_pool.free`.

---

## Round 2 — Second (Last) Chunk of R3 + Decode of R1 and R2

**Step 1 — PrefillAdder.build():**

`_chunked_req is not None` (it's R3), so case 1 fires.

`start = R3.kv_committed_len = 512`. `size = chunked_prefill_size = 512`. `end = min(512 + 512, 1024) = 1024`.

`R3.fill_ids = input_ids[512:1024]`. `R3.extend_input_len = 512`.

Returns `[R3]`.

**Step 2 — prefill_batch([R3]):**

`compute_write_info(kv_pool, rtp, R3.slot_indices=[p0..p31], R3.req_pool_idx=row, kv_committed_len=512, n_fill=512)`:
- `n_leftover = 512 % 16 = 0`. Last page from round 1 is exactly full.
- `remaining = 512`, allocate `32` new pages: `[q0..q31]`.
- `R3.slot_indices = [p0..p31, q0..q31]`. `req_to_token[row, 32:64] = [q0..q31]`.
- Returns `WriteInfo(existing_page=None, n_leftover=0, existing_slots=0, new_pages=[q0..q31])`.

Token IDs `input_ids[512:1024]` packed into `ids_t [1, 512]`. Position IDs `[512, 513, ..., 1023]`.

`qo_indptr = [0, 512]`. `kv_indptr = [0, 64]` (64 total pages — 32 old + 32 new). `kv_last_page_lens = [16]` (1024 % 16 = 0 → P = 16).

Triton kernel reads `req_to_token[row, 0:64]` = `[p0..p31, q0..q31]` — all 64 pages.

`begin_forward(qo_indptr=[0,512], kv_indptr=[0,64], kv_indices=[p0..p31,q0..q31], kv_last_page_lens=[16], causal=True, ...)`.

Inside the model, each attention layer:
1. `ctx.store(layer_idx, k, v)` — writes the second chunk's K/V into pages `q0..q31`.
2. `extend_wrapper.forward(q_fi, (k_pool[layer], v_pool[layer]))` — FlashInfer attends over all 64 pages (1024 positions): the 32 cached pages from round 1 plus the 32 newly written pages. Causality is enforced — token 512 can attend to tokens 0–512; token 1023 can attend to tokens 0–1023.

`end_forward()`. `R3.kv_committed_len = 512 + 512 = 1024`. `R3.is_last_chunk = (1024 >= 1024)? Yes`.

The last logit (at `qo_indptr[1] - 1 = 511` within the packed sequence) is sampled. `R3.output_ids = [first_token]`. `R3.status = RUNNING`.

**Step 3 — Update _chunked_req:**

R3 is no longer `PREFILLING`; `self._chunked_req = None`. R3 joins `_running`.

**Step 4 — Decode of R1 and R2:**

Same as round 1. R1 and R2 advance by one token while R3 was being chunked.

---

## What the Trace Shows

R3 was chunked across two rounds. Between round 1 and round 2, two decode steps ran for R1 and R2 — they were not starved. The total K/V written for R3 (64 pages, 1024 positions across 28 layers) is identical to what a single-pass prefill would have written, but spread across two scheduler iterations.

The second chunk's extend call used all 64 pages in `kv_indices`. FlashInfer read the cached pages from round 1 directly from the pool — no recomputation, no re-transfer of the round-1 K/V data.

Section 08 discusses what the remaining inefficiency in Layer 11 is and how Layer 12's prefix caching addresses it.
