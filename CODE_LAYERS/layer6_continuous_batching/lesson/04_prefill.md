# 04 — Prefill

```python
def prefill(self, req: Req) -> None:
    ids  = torch.tensor([req.input_ids], device=DEVICE)        # [1, L]
    mask = torch.ones(1, len(req.input_ids), dtype=torch.long, device=DEVICE)
    pos  = torch.arange(len(req.input_ids), device=DEVICE).unsqueeze(0)  # [1, L]

    kv = PerReqKVCache()
    with torch.no_grad():
        logits = self.model(ids, attention_mask=mask, kv_cache=kv, position_ids=pos)

    req.kv_cache      = kv
    req.t_first_token = time.perf_counter()

    next_tok = self._sample(logits[0, -1], req.temperature)
    req.output_ids.append(next_tok)

    if next_tok == self.eos_id or req.output_len >= req.max_new_tokens:
        req.status   = ReqStatus.FINISHED
        req.t_finish = time.perf_counter()
    else:
        req.status = ReqStatus.RUNNING
```

Prefill runs once per request, always as a B=1 forward pass. The scheduler calls it as soon as a request is dequeued from `_waiting`, before any decode step for that request occurs. This is the most important structural difference from Layer 5: in Layer 5, all B requests were prefilled together in a single `[B, max_prompt_len]` forward call. In Layer 6, each request gets its own `[1, prompt_len]` pass, run individually and in sequence as requests arrive.

---

## Building the Prefill Inputs

`input_ids` is already a list of integers when `Req` is constructed — the server tokenised and formatted the prompt before calling `add_request`. `prefill` wraps it in a `[1, L]` tensor where `L = len(req.input_ids)`. There is no left-padding: a B=1 input with no padding does not need an `attention_mask` to hide padding positions. The mask is still passed for API consistency — `torch.ones(1, L)` marks all positions as real — but its content does not affect the computation.

`position_ids` is `torch.arange(L).unsqueeze(0)`, giving positions `0, 1, 2, ..., L-1` for the `L` prompt tokens. In Layer 5, explicit position IDs were needed to correct for left-padding: a padded batch row of length 50 with 10 real tokens needed RoPE positions `0..9`, not `40..49`. Here there is no padding, so the sequential range is exactly right. RoPE will assign cosine/sine angles based on these positions, and the same positions will be used when the request later participates in decode steps — section 05 shows how per-request position IDs are reconstructed during decode.

---

## The B=1 Forward Pass

```python
kv = PerReqKVCache()
logits = self.model(ids, attention_mask=mask, kv_cache=kv, position_ids=pos)
```

A fresh `PerReqKVCache` is constructed for this request. During the forward pass, each of the 28 attention layers calls `kv.update(layer_idx, k, v)` with the key and value tensors for the full prompt. By the time the call returns, `kv._k` and `kv._v` each contain 28 entries, one per layer, each of shape `[1, n_kv_heads, L, head_dim]`. This is the request's complete KV history: it encodes the full prompt's representation and will be used as context for every subsequent decode step.

`logits` has shape `[1, L, vocab_size]`. Only the last-position logit `logits[0, -1]` is used: it represents the model's distribution over the next token given the full prompt. This is the same slice used in Layer 5's `out.logits[:, -1, :]`, but here for B=1.

---

## First Token, TTFT, and Status Transition

`req.t_first_token = time.perf_counter()` is recorded immediately after the forward pass returns, before sampling. TTFT measures the wall time from request construction (`t_arrive`, set at `Req` init) to this timestamp. It includes the time the request spent waiting in `_waiting` if the scheduler was busy, plus the GPU time for the forward pass itself.

Sampling produces the first output token. The server chose `temperature=0.0` for deterministic greedy decoding or `temperature>0` for stochastic sampling, and this is honoured per-request via `req.temperature`.

The status transition logic:

```python
if next_tok == self.eos_id or req.output_len >= req.max_new_tokens:
    req.status   = ReqStatus.FINISHED
    req.t_finish = time.perf_counter()
else:
    req.status = ReqStatus.RUNNING
```

If the very first sampled token is EOS — the model decided the prompt itself is the complete answer — the request is done. The scheduler checks this immediately after `prefill` returns:

```python
if req.status == ReqStatus.FINISHED:
    self._resolve(req)
else:
    self._running.append(req)
```

A one-token response never enters the decode batch. This edge case would cause a silent hang in Layer 5, where a request that finished in zero decode steps would have contributed an empty `generated[i]` list without any special handling of the EOS-on-first-token case. Layer 6 handles it explicitly at the prefill boundary.

---

## No Prompt-Length Alignment

Layer 5's `prepare_batch` left-padded all prompts to the same length so they could be stacked into a `[B, max_prompt_len]` tensor. This imposed a synchronisation cost: a short 10-token prompt shared a batch row with a long 500-token prompt, forcing the GPU to process 490 extra (masked) positions for the short prompt during prefill.

Layer 6's B=1 prefill eliminates this completely. Each prompt is processed at exactly its own length. A 10-token prompt runs a `[1, 10]` forward pass; a 500-token prompt runs a `[1, 500]` forward pass. Prefill compute is proportional to the actual prompt length, not the maximum in the batch. The cost of this is sequential execution: if three requests arrive simultaneously, their prefills run one after the other rather than fused into a single kernel call. Production systems use chunked prefill to amortise this — section 08 discusses the trade-offs.
