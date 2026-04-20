# 06 — Accept/Reject and KV Rewind

After the verify extend returns N+1 logit vectors per request, `_accept_reject` determines how many draft tokens are accepted, selects the bonus token, and updates the request state. For accepted positions, no pool modification is needed — the K/V is already in the right place. For rejected positions, both the target and draft KV pools must be rolled back.

---

## _accept_reject

```python
# spec_runner.py — AcceptResult
class AcceptResult:
    accepted_toks: List[int]   # the accepted draft tokens
    bonus_token:   int         # target's prediction at the rejection/end position
    accept_len:    int         # number of accepted draft tokens (0..N)
```

```python
# spec_runner.py — SpecRunner._accept_reject
def _accept_reject(
    self,
    req: Req,
    d_req: DraftReq,
    logits: torch.Tensor,   # [N+1, vocab_size]
) -> AcceptResult:
    N     = self._n_speculative_tokens
    preds = logits.argmax(dim=-1)   # [N+1] — target's greedy prediction at each position

    accept_len = 0
    for j in range(N):
        target_pred = preds[j].item()     # target's prediction at position j → for d_{j+1}
        draft_tok   = d_req.draft_tokens[j]   # the j-th candidate token
        if target_pred == draft_tok:
            accept_len += 1
        else:
            break

    bonus_token = preds[accept_len].item()   # always: target's prediction at the first rejection

    accepted_toks = d_req.draft_tokens[:accept_len]
    return AcceptResult(accepted_toks, bonus_token, accept_len)
```

The loop compares `preds[j]` — the target's argmax prediction for what comes after position `j` — against `d_req.draft_tokens[j]` — the draft model's token at position j. Note the offset: `preds[0]` is the prediction after `t_k` (the confirmed token), which should match `d1 = draft_tokens[0]`.

The loop breaks at the first mismatch. `accept_len` is the longest consistent prefix. `bonus_token = preds[accept_len]` is the target's correction at the rejection site — or, if all N tokens were accepted, the target's next prediction after dN.

---

## Committing Accepted Tokens

```python
# spec_runner.py — after _accept_reject
for tok in result.accepted_toks:
    req.output_ids.append(tok)
req.output_ids.append(result.bonus_token)
```

Both the accepted draft tokens and the bonus token are appended to `req.output_ids`. The target pool already holds their K/V (from the verify extend). No additional target model call is needed.

`d_req.output_ids` is updated identically:

```python
for tok in result.accepted_toks:
    d_req.output_ids.append(tok)
d_req.output_ids.append(result.bonus_token)
```

This keeps the draft's `output_ids` in sync with the target's for the next step's draft phase.

---

## _rewind_target_kv

The target pool has K/V written for all N+1 positions from the verify extend. Only the first `accept_len + 1` positions are valid (accepted tokens + confirmed token t_k). Positions `accept_len + 1` through N are rejected draft tokens whose K/V should not persist.

```python
# spec_runner.py — _rewind_target_kv
def _rewind_target_kv(self, req: Req, accept_len: int) -> None:
    # Number of target pool tokens after the verify extend:
    # original seq_len (before this spec step) + N + 1
    total_in_pool = len(req.output_ids) - 1 - self._n_speculative_tokens + accept_len + 1
    # = original_kv_committed_len + (N+1 tokens written) - (N - accept_len rejected)
    # = original + accept_len + 1

    P           = self.target_mr.kv_pool.page_size
    valid_pages = math.ceil(total_in_pool / P)

    # Pages beyond valid_pages were allocated for rejected positions
    if valid_pages < len(req.slot_indices):
        pages_to_free = req.slot_indices[valid_pages:]
        self.target_mr.kv_pool.free(pages_to_free)
        req.slot_indices = req.slot_indices[:valid_pages]
        # Also update req_to_token_pool — only valid_pages columns are valid
        # (the table doesn't need explicit clearing; the valid region is tracked
        # by slot_indices length and num_pages computations in the next prefill_batch)
```

For `accept_len = 2`, `N = 5`, original pool at 20 tokens: the verify extend wrote N+1=6 tokens' K/V, consuming pages for positions 20–25. The valid range after acceptance is positions 20–22 (3 tokens: t_k at 20, d1 at 21, d2 at 22). Bonus token at 23. Total valid: 24 positions. Pages 0–1 (positions 0–15 and 16–23 in a 16-token page) — `ceil(24/16) = 2` pages. If the verify extend allocated pages for positions 20–25 (spanning pages at offsets 16–31), any pages beyond `ceil(24/16) = 2` are freed.

The KV values at positions 22, 23, 24 (within page 1) for the rejected draft tokens d3, d4, d5 remain in the page buffer but are never read — `kv_last_page_lens` in the next step will report the correct fill level (24 % 16 = 8, so the last page has 8 valid tokens), and FlashInfer will not attend over positions 8–15 of that page.

---

## _rewind_draft_kv

The draft pool has K/V for all N draft tokens (written during the draft phase). Only `accept_len` of them are valid. The rejected positions and their pool entries must be freed before the draft phase of the next step.

```python
# spec_runner.py — _rewind_draft_kv
def _rewind_draft_kv(self, d_req: DraftReq, accept_len: int) -> None:
    P = self.draft_mr.kv_pool.page_size
    valid_tokens = len(d_req.output_ids) - self._n_speculative_tokens + accept_len
    # output_ids currently includes the N draft tokens; subtract (N - accept_len) rejected ones
    valid_pages  = math.ceil(valid_tokens / P)

    if valid_pages < len(d_req.slot_indices):
        pages_to_free = d_req.slot_indices[valid_pages:]
        self.draft_mr.kv_pool.free(pages_to_free)
        d_req.slot_indices = d_req.slot_indices[:valid_pages]
```

After `_rewind_draft_kv`, `d_req.output_ids` is trimmed to the accepted prefix:

```python
# Remove rejected draft tokens from d_req.output_ids
n_reject = self._n_speculative_tokens - accept_len
if n_reject > 0:
    d_req.output_ids = d_req.output_ids[:-n_reject]
```

Then one additional draft decode step is called to write the bonus token's K/V into the draft pool, restoring the mirroring invariant (see section 03).

---

## Statistics Update

```python
# spec_runner.py — after commit and rewind
self._total_accepted  += result.accept_len
self._total_proposed  += self._n_speculative_tokens
self._total_tokens    += result.accept_len + 1   # accepted + bonus
self._total_steps     += 1

# acceptance_rate property
@property
def acceptance_rate(self) -> float:
    if self._total_proposed == 0:
        return 0.0
    return self._total_accepted / self._total_proposed

# tokens_per_step property
@property
def tokens_per_step(self) -> float:
    if self._total_steps == 0:
        return 1.0
    return self._total_tokens / self._total_steps
```

These running statistics are used to tune `N` dynamically in production systems — if the acceptance rate drops below a threshold (e.g., 0.5), reducing `N` from 5 to 3 reduces draft overhead without sacrificing much yield. Layer 14 exposes these metrics but does not implement dynamic `N` adjustment; section 07 explains this in detail.
