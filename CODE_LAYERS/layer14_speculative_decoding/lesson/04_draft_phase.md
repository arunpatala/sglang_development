# 04 — The Draft Phase

The draft phase generates N candidate tokens autoregressively from the draft model, starting from the current confirmed context. Each candidate token is produced by one full forward pass through the draft model — cheaper than the target because the draft model has fewer layers and smaller hidden dimension.

---

## _draft_phase

```python
# spec_runner.py — SpecRunner._draft_phase
def _draft_phase(
    self,
    reqs: List[Req],
    d_reqs: List[DraftReq],
) -> None:
    """Run N draft decode steps to generate N candidate tokens per request."""
    for step in range(self._n_speculative_tokens):
        draft_tokens_this_step = self.draft_mr.decode_step_for_dreqs(reqs, d_reqs)
        for req, d_req, tok in zip(reqs, d_reqs, draft_tokens_this_step):
            d_req.draft_tokens.append(tok)
            d_req.output_ids.append(tok)   # advances draft's sequence position
```

`N = self._n_speculative_tokens` is typically 5. The loop runs 5 times; on each iteration, `decode_step_for_dreqs` runs one forward pass through the draft model for all requests simultaneously and returns one token per request.

`d_req.draft_tokens` accumulates the N candidates: `[d1, d2, ..., dN]`. `d_req.output_ids` is extended identically so that the next step's `decode_step_for_dreqs` uses `d_req.output_ids[-1]` as the current input token (the token generated in the previous step, now at the front of the next decode call).

---

## decode_step_for_dreqs

`decode_step_for_dreqs` is a `ModelRunner` method that operates on `DraftReq` objects instead of `Req` objects. The internal mechanics are identical to `decode_step`: compute `seq_len = len(d_req.output_ids) - 1`, check `token_offset`, conditionally allocate a new page, build `kv_indptr`/`kv_indices` via the Triton kernel, call `begin_forward`, run the draft model's forward pass, call `end_forward`, sample tokens.

The only difference from a standard `decode_step` call is that it reads from `d_req.output_ids` (not `req.output_ids`) for the sequence length computation and takes `d_req.slot_indices` (not `req.slot_indices`) for the pool page lookup. The sampling result is returned (not appended to `d_req.output_ids` — that happens in `_draft_phase`'s outer loop).

---

## Cost of the Draft Phase

The draft model (Qwen3-0.6B hypothetically: 16 layers, hidden_size=896, 8 KV heads, head_dim=64) has approximately:

```
layers:          16 (vs target's 28)
hidden_size:     896 (vs 2048)
FFN intermediate: 4864 (vs 11008)
weight params:   ~0.62B (vs 1.7B)
weight memory:   ~1.2 GB bfloat16
```

Each draft decode step involves 16 × 7 = 112 linear operations on smaller matrices vs the target's 28 × 7 = 196. The draft decode step is approximately 4–7× faster than the target decode step (the exact ratio depends on the batch size and memory bandwidth utilization).

For `N = 5` draft steps: total draft cost ≈ `5 × T_draft ≈ 5 × 0.15 × T_target = 0.75 × T_target`. One target verify (next section) costs approximately `1 × T_target`. Total spec step cost ≈ `1.75 × T_target` for an expected yield of approximately 3.7 tokens. Layer 13 (no speculation) would take `3.7 × T_target` for the same yield. The speedup is approximately `3.7 / 1.75 ≈ 2.1×`.

---

## The Input Token Tracking

At the start of `_draft_phase`, `d_req.output_ids[-1]` is the last confirmed token — the same as `req.output_ids[-1]`. This is the current input token for the first draft decode step. The seq_len formula:

```python
seq_len = len(d_req.output_ids) - 1
```

For a fresh call with 10 previously committed tokens: `seq_len = 10`. The draft model's KV pool holds K/V for positions 0–9. The first draft decode writes K/V at position 10 and returns the draft token for position 10.

After step 1: `d_req.output_ids` has 11 entries. `seq_len = 10` for step 2 — wait, `len(output_ids) - 1 = 11 - 1 = 10`? No: `d_req.output_ids.append(tok)` in the outer loop adds the new token, so after step 1, `len(d_req.output_ids) = 11`. Step 2's `decode_step_for_dreqs` sees `len = 11`, so `seq_len = 10` — but this should be 11 (10 prior tokens + the first draft token at position 10).

The subtlety: `seq_len = len(output_ids) - 1` accounts for the fact that `output_ids[-1]` is the current input token (not yet in the pool). After appending `d1` to `d_req.output_ids`, `output_ids[-1] = d1` is the new current input token for step 2. `seq_len = 11 - 1 = 10` means positions 0–9 are in the pool (from prior committed context) and position 10 (d1) has just been written in step 1's forward pass. Step 2 writes d2 at position 11. The formula is consistent.

---

## Cleared at Each spec_decode_step

At the start of `_draft_phase`, `d_req.draft_tokens` is cleared:

```python
for d_req in d_reqs:
    d_req.draft_tokens = []
```

This prevents stale candidates from a previous speculation step from contaminating the current one. The `output_ids` list is not cleared — it persists across `spec_decode_step` calls, growing with each accepted token, just as `req.output_ids` does.

Section 05 explains how the verify extend uses all N draft tokens plus the current confirmed token in a single batched target forward pass.
