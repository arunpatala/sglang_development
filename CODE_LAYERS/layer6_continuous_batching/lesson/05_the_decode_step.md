# 05 — The Decode Step

Step 2 of the scheduler loop calls `self.model_runner.decode_step(self._running)` on every iteration where `_running` is non-empty. This is where continuous batching happens: all currently active requests — which arrived at different times, have different prompt lengths, and have accumulated different KV histories — are processed together in a single `[B, 1]` forward pass. The four inputs built below (`last_toks`, `attn_mask`, `pos_ids`, `BatchedKVCache`) are each a direct answer to the question: how do you batch requests with ragged KV lengths?

```python
def decode_step(self, reqs: List[Req]) -> List[Req]:
    B       = len(reqs)
    kv_lens = [r.kv_cache.get_seq_length() for r in reqs]
    max_kv  = max(kv_lens)

    last_toks = torch.tensor([[r.output_ids[-1]] for r in reqs], ...)  # [B, 1]
    attn_mask = ...   # [B, max_kv + 1]
    pos_ids   = ...   # [B, 1]

    batch_kv = BatchedKVCache(reqs, max_kv)
    logits   = self.model(last_toks, attention_mask=attn_mask,
                          kv_cache=batch_kv, position_ids=pos_ids)  # [B, 1, vocab]
    batch_kv.write_back()
    ...
```

`decode_step` runs once per scheduler loop iteration when `_running` is non-empty. Unlike prefill — which is one B=1 forward pass per request, run sequentially — `decode_step` batches all currently active requests into a single `[B, 1]` forward pass. The challenge is that each request has a different KV length, accumulated since its own prefill. This section traces how the four inputs — `last_toks`, `attn_mask`, `pos_ids`, and `BatchedKVCache` — are constructed to handle those ragged lengths correctly.

---

## Input Tokens: `last_toks`

```python
last_toks = torch.tensor(
    [[r.output_ids[-1]] for r in reqs], dtype=torch.long, device=DEVICE
)   # [B, 1]
```

Each request contributes exactly one token: its most recently generated output. During prefill, the first token was appended to `req.output_ids`. On each decode step, the token generated in the previous step is the input. This is the same pattern as Layer 5's `current_tokens = next_tokens.unsqueeze(1)`, but without the `torch.where` masking for finished requests — requests leave `_running` immediately when they finish, so no padding rows are needed.

---

## Attention Mask: `attn_mask`

```python
attn_mask = torch.zeros(B, max_kv + 1, dtype=torch.long, device=DEVICE)
for i, kv_len in enumerate(kv_lens):
    attn_mask[i, max_kv - kv_len:] = 1   # real KV positions + new token slot
```

The mask has shape `[B, max_kv + 1]`. The `max_kv` columns correspond to the historical KV entries (which will be left-padded in `BatchedKVCache`), and the final `+1` column corresponds to the new token being generated. Each row is filled with zeros and then the right portion — from `max_kv - kv_len_i` onward — is set to 1.

Consider three requests with KV lengths 10, 7, and 3, and `max_kv = 10`:

```
Request 0 (kv_len=10):  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1 | 1]   # no padding
Request 1 (kv_len=7):   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1 | 1]   # 3 left-padding zeros
Request 2 (kv_len=3):   [0, 0, 0, 0, 0, 0, 0, 1, 1, 1 | 1]   # 7 left-padding zeros
```

The model converts `attn_mask` to an additive mask inside `_build_additive_mask`: zero positions become −∞ in the attention logits, so those positions receive zero attention weight after softmax. The left-padding zeros in the `BatchedKVCache` (pure-zero tensors) are masked out and never contribute to the output.

---

## Position IDs: `pos_ids`

```python
pos_ids = torch.tensor(
    [[kv_len] for kv_len in kv_lens], dtype=torch.long, device=DEVICE
)   # [B, 1]
```

Each request is assigned its own next position: `kv_len_i`. This is the length of its accumulated KV history, which equals the number of real tokens processed so far (prompt + output tokens generated up to this step). For a request that was prefilled with a 10-token prompt and has generated 5 tokens, `kv_len = 15` and `pos_ids[i] = 15`. RoPE will compute the cosine and sine values for position 15, matching what a single-request run would produce.

This is the continuous-batching counterpart of Layer 5's per-request decode position:

```python
# Layer 5
decode_position_ids = (prompt_lens + decode_step).unsqueeze(1)   # [B, 1]
```

In Layer 5, all requests had the same `decode_step` counter because they were decoded in lockstep. In Layer 6, each request's position is `kv_len_i`, which independently counts how many tokens that request has processed. Two requests that arrived at different times will have different positions at any given decode step, and `pos_ids` captures that correctly.

---

## The Batched Forward Pass

```python
batch_kv = BatchedKVCache(reqs, max_kv)
logits = self.model(last_toks, attention_mask=attn_mask,
                    kv_cache=batch_kv, position_ids=pos_ids)   # [B, 1, vocab_size]
batch_kv.write_back()
```

`BatchedKVCache` is constructed with the current `reqs` and `max_kv`. As the forward pass progresses through layers, each attention layer calls `batch_kv.update(layer_idx, new_k, new_v)`. On first call for layer `i`, `_init_layer` pads each request's `PerReqKVCache` to `max_kv` and stacks into `[B, n_kv_heads, max_kv, head_dim]`. The new token's K/V — shape `[B, n_kv_heads, 1, head_dim]` — is then appended, yielding `[B, n_kv_heads, max_kv + 1, head_dim]`, and this is passed to `F.scaled_dot_product_attention`. The attention mask hides the left-padded zeros, so the computation is equivalent to running each request independently at its true KV length.

After the forward pass, `batch_kv.write_back()` appends the last-position K/V from the batched tensor to each request's `PerReqKVCache`, growing each by exactly one token. Section 03 covers this in detail.

---

## Sampling and State Update

```python
newly_finished: List[Req] = []
for i, req in enumerate(reqs):
    next_tok = self._sample(logits[i, -1], req.temperature)
    req.output_ids.append(next_tok)

    if next_tok == self.eos_id or req.output_len >= req.max_new_tokens:
        req.status   = ReqStatus.FINISHED
        req.t_finish = time.perf_counter()
        newly_finished.append(req)

return newly_finished
```

`logits[i, -1]` is the last-position (and only) logit vector for request `i`, of shape `[vocab_size]`. Sampling is per-request, using each request's own `temperature`. Newly finished requests are collected and returned to the scheduler. The scheduler resolves their futures and filters them out of `_running` before the next iteration. Requests that did not finish stay in `_running` with their `PerReqKVCache` now one token longer, ready to participate in the next decode step.
