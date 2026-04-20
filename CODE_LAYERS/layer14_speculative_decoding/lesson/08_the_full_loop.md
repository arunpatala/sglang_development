# 08 — The Full Loop

This section traces one complete `spec_decode_step` for two requests — R1 (accept all 5) and R2 (accept 2 of 5) — to show every phase in code order.

---

## Setup

`N = 5`. Both R1 and R2 are in `_running` with 20 accumulated tokens each. `req.output_ids` has 10 entries (10 generated tokens); `kv_committed_len = 19` (the last generated token is the current input token, not yet in the pool).

For R1: draft will generate `[A, B, C, D, E]`, target will agree with all 5 → `accept_len = 5`.
For R2: draft will generate `[P, Q, R, S, T]`, target will agree with P, Q but reject R → `accept_len = 2`.

---

## Phase 1 — Draft Phase (5 steps)

**Draft step 1:** `decode_step_for_dreqs([R1, R2], [d_R1, d_R2])`.
- `seq_len_R1 = len(d_R1.output_ids) - 1 = 10 - 1 = 9`. Wait — `output_ids` has 10 entries; `seq_len = 10 - 1 = 9` tokens in pool... actually the request has 20 accumulated tokens (10 prompt + 10 output = 20 total). Let me reclarify: `len(req.input_ids) = 10`, `len(req.output_ids) = 10 (but the last one is the current input token)`. So `seq_len = len(input_ids) + len(output_ids) - 1 = 10 + 10 - 1 = 19`.
- Draft model writes K/V at position 19 (the current input token), returns next token.
- `d_R1.draft_tokens = [A]`, `d_R1.output_ids` appends A (now 11 entries including the confirmed 10, but let's simplify: think of `d_R1.output_ids` as mirroring `req.output_ids` + draft tokens).

For the trace, let's think of it simply:
- Before draft phase: R1 and R2 each have 20 confirmed tokens in pool.
- Draft step 1 processes position 20 → tokens A and P.
- Draft step 2 processes position 21 → tokens B and Q.
- Draft step 3 processes position 22 → tokens C and R.
- Draft step 4 processes position 23 → tokens D and S.
- Draft step 5 processes position 24 → tokens E and T.

After draft phase:
- `d_R1.draft_tokens = [A, B, C, D, E]`
- `d_R2.draft_tokens = [P, Q, R, S, T]`
- Draft pool for R1 has K/V for positions 0–24 (20 confirmed + 5 draft).
- Draft pool for R2 has K/V for positions 0–24.

---

## Phase 2 — Verify Extend (1 target call)

`_verify_extend([R1, R2], [d_R1, d_R2])`:

For R1: `all_ids = [t_20, A, B, C, D, E]` (6 tokens). `kv_committed_len = 19`. `extend_input_len = 6`.
For R2: `all_ids = [t_20, P, Q, R, S, T]` (6 tokens). Same.

`extend_for_verify` packs all 12 tokens into `ids_t [1, 12]`. `qo_indptr = [0, 6, 12]`.

`compute_write_info` for R1: `kv_committed_len = 19`, `n_fill = 6`. `n_leftover = 19 % 16 = 3`, so last page has 3 tokens and 13 free slots. `existing_slots = min(13, 6) = 6` — all 6 tokens fit within the existing last page! `remaining = 0`, no new pages allocated for R1. `WriteInfo(existing_page=last_R1_page, n_leftover=3, existing_slots=6, new_pages=[])`.

Similarly for R2 (same arithmetic).

Position IDs for R1: `[19, 20, 21, 22, 23, 24]`. For R2: `[19, 20, 21, 22, 23, 24]`.

`begin_forward(qo_indptr=[0,6,12], kv_indptr=[0,n_pages_R1, n_pages_R1+n_pages_R2], ...)`.

Target model forward pass:
- `o_proj`, `gate_proj`, etc. — 7 `GPTQLinear.forward` calls per of 28 layers.
- `ExtendKVCtx.store(layer, k, v)` writes the 6 tokens' K/V for R1 into `last_R1_page[3:9]` and for R2 into `last_R2_page[3:9]`.
- FlashInfer extend: causal attention over 6 query positions, full KV history.

Returns `logits [1, 12, vocab_size]`. Split: `logits[0, :6]` for R1, `logits[0, 6:12]` for R2.

---

## Phase 3 — Accept/Reject

**R1:** `preds_R1 = argmax(logits[0, :6]) = [A, B, C, D, E, bonus_R1]`.
- `preds_R1[0] = A == d_R1.draft_tokens[0] = A` → accept.
- `preds_R1[1] = B == B` → accept.
- ... all 5 match → `accept_len_R1 = 5`, `bonus_R1 = preds_R1[5]`.

**R2:** `preds_R2 = argmax(logits[0, 6:12]) = [P, Q, X, ..., bonus_R2]` where X ≠ R.
- `preds_R2[0] = P == P` → accept.
- `preds_R2[1] = Q == Q` → accept.
- `preds_R2[2] = X ≠ R` → reject → break.
- `accept_len_R2 = 2`, `bonus_R2 = preds_R2[2] = X`.

---

## Phase 4 — Commit and Rewind

**R1 commit:**
- `req_R1.output_ids += [A, B, C, D, E, bonus_R1]` — 6 new tokens.
- Target pool: positions 19–24 already written (all valid). No rewind needed (`accept_len = N`).
- `d_R1.output_ids += [A, B, C, D, E, bonus_R1]`.

**R1 draft rewind:**
- Valid draft pool: positions 0–24 (20 confirmed + 5 accepted). All valid.
- `_rewind_draft_kv(d_R1, accept_len=5)`: `valid_pages = ceil(26/P)` (26 = 20 + 5 + bonus). No pages freed.
- One additional draft decode step writes bonus_R1's K/V at position 25.

**R2 commit:**
- `req_R2.output_ids += [P, Q, X]` — 3 new tokens (2 accepted + bonus).
- Target pool: positions 19–24 were written. Valid range: 19, 20, 21 (confirmed + P + Q) and position 22 = X (bonus). Total valid positions: 0–22. `valid_pages = ceil(23/16) = 2`.
- `_rewind_target_kv(d_req_R2, accept_len=2)`: free pages beyond page index 1. Pages for positions 22–24 (R, S, T) that were written into page 1 — but page 1 covers positions 16–31; the K/V for positions 23–24 (S, T) within page 1 are now stale but remain in the buffer. The next step's `kv_last_page_lens` will set the valid count correctly.

**R2 draft rewind:**
- Draft pool has positions 0–24 (all 25 written). Valid range: 0–22 (20 + P + Q + bonus). `valid_pages = ceil(23/16) = 2`.
- `_rewind_draft_kv(d_R2, accept_len=2)`: free pages beyond index 1. `d_R2.slot_indices = d_R2.slot_indices[:2]`.
- Trim `d_R2.output_ids` to remove rejected tokens R, S, T.
- One additional draft decode step writes bonus X's K/V at position 22.

---

## Statistics Update

```
_total_accepted += 5 + 2 = 7
_total_proposed += 5 + 5 = 10
_total_tokens   += 6 + 3 = 9
_total_steps    += 1   (one batch step)
```

`acceptance_rate = 7/10 = 0.70`. `tokens_per_step = 9/1 = 9` (batch step over 2 requests — 4.5 tokens per request). Averaged over many steps: `tokens_per_step / B ≈ 4.5` tokens per request per target call, consistent with the `1 + N × p = 1 + 5 × 0.7 = 4.5` formula.

Both R1 and R2 are now ready for the next `spec_decode_step`. The mirroring invariant holds: both target and draft pools cover exactly the same confirmed sequence for each request.
