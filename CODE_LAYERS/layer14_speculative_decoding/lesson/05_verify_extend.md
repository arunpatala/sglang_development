# 05 — The Verify Extend

After the draft phase generates N candidate tokens per request, the target model must evaluate all N+1 positions — the current confirmed token plus the N draft tokens — in a single forward pass. This is the verify extend: one call to `prefill_batch`-style machinery that reads the target's cached KV history and attends over N+1 new query positions.

---

## The N+1 Token Window

Suppose the confirmed sequence for a request is `[t0, t1, ..., t_k]` where `t_k = req.output_ids[-1]` is the confirmed last token (whose K/V is not yet in the target pool — it was the current input token for the last decode step). The draft phase generated `[d1, d2, ..., dN]`.

The verify extend processes the sequence `[t_k, d1, d2, ..., dN]` — N+1 tokens. The first token `t_k` is the confirmed current token; its K/V is being written now. Draft tokens `d1..dN` are the candidates.

```python
# spec_runner.py — SpecRunner._verify_extend
def _verify_extend(
    self,
    reqs: List[Req],
    d_reqs: List[DraftReq],
) -> torch.Tensor:
    """
    Run one batched extend over [last_confirmed_tok, d1, ..., dN] for all requests.
    Returns logits: [B, N+1, vocab_size]
    """
    N       = self._n_speculative_tokens
    B       = len(reqs)

    all_ids: List[int] = []
    for req, d_req in zip(reqs, d_reqs):
        current_tok  = req.output_ids[-1]               # the confirmed current input token
        draft_tokens = d_req.draft_tokens               # [d1..dN]
        all_ids.extend([current_tok] + draft_tokens)    # N+1 tokens per request

    return self.target_mr.extend_for_verify(reqs, all_ids, n_extend=N + 1)
```

`extend_for_verify` is a targeted use of `prefill_batch` machinery:

- Each request's `fill_ids` = `[current_tok, d1, ..., dN]` — N+1 tokens.
- `kv_committed_len` = number of tokens already in the target pool (the prompt + all prior accepted output tokens, minus 1 for the current input token which is not yet in the pool). Actually `kv_committed_len = len(req.output_ids) - 1` — the positions before the current step.
- `extend_input_len = N + 1`.

`compute_write_info` allocates pages for the N+1 tokens. The Triton kernel builds `kv_indices` including the prior pool pages. `begin_forward` plans the extend kernel. The model forward pass writes K/V for all N+1 tokens into the target pool and returns `logits [B, N+1, vocab_size]`.

---

## The Logit Alignment

The verify extend returns logits at each of the N+1 positions. The semantic meaning:

```
logits[b, 0, :] — distribution at position kv_committed_len + 0 (after seeing t_k)
                   → the target's prediction for what comes after t_k
logits[b, 1, :] — distribution after seeing t_k and d1
                   → the target's prediction for what comes after d1 (from its perspective)
...
logits[b, N, :] — distribution after seeing all N+1 tokens
                   → the target's bonus token
```

`logits[b, j, :]` is used to verify draft token `d_{j+1}`. Specifically, `argmax(logits[b, j-1, :]) == d_j` is the accept condition: the target model agrees that `d_j` was the best prediction after seeing the prefix up to `d_{j-1}`.

`logits[b, N, :]` — the prediction after all N+1 tokens — is the bonus token. It is sampled regardless of whether any draft tokens were rejected: even if all N draft tokens are rejected, the bonus token is committed.

---

## Why This Is Safe to Batch

The extend kernel with `causal=True` ensures that each position in the N+1 sequence attends only over earlier positions. Position 0 (`t_k`) attends over the full prior KV history. Position 1 (`d1`) attends over position 0 and prior history. Position N (`dN`) attends over positions 0 through N-1 and prior history. The causal mask exactly replicates the attention that would be computed if the target model were to evaluate each position autoregressively, one at a time.

This is the fundamental correctness guarantee of speculative decoding: the verify extend's logits at position `j` are identical to the logits the target would have produced if it had decoded positions 0 through j autoregressively. The batching is purely a performance optimization — not an approximation.

For multiple requests in the batch (`B > 1`), each request's N+1 tokens form an independent causal block within the packed sequence. FlashInfer's `qo_indptr` partitions the queries per request; no cross-request attention occurs.

---

## K/V Written into the Target Pool

A crucial side effect of `extend_for_verify`: the K/V for all N+1 positions is written into the target pool. If 3 of 5 draft tokens are accepted, positions 0–3 are now committed — their K/V is in the target pool. Positions 4–5 (the two rejected positions) also had their K/V written, but they correspond to the wrong token sequence (rejected draft tokens). These must be unwound. Section 06 explains the KV rewind.
