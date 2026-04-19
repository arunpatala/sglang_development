# 05 — The Finished Mask

## The Problem: Requests Finish at Different Steps

In Layer 2, a single request ran until it emitted EOS or exhausted `max_new_tokens`. The loop had one exit condition and one active sequence. In Layer 3, the loop runs B sequences simultaneously, and they do not finish at the same step. A short answer to a simple question might complete in 5 tokens. A detailed technical response might require 128 tokens. Both are in the same batch, both share the same forward call on every step.

The decode loop cannot stop when the first request finishes — that would cut off all other requests. It also cannot keep sampling tokens for a finished request in the normal way — those tokens would be appended to `generated[i]` and returned as part of the response, corrupting the output. A mechanism is needed to keep the loop running until every request has finished, while cleanly excluding finished requests from the output.

The `finished` boolean tensor is that mechanism.

---

## Initialisation

```python
finished = next_tokens == self.eos_id   # [B] bool
```

`next_tokens` is the `[B]` tensor produced by `sample_batch` immediately after prefill — the first generated token for each request. `finished[i]` is `True` if request `i`'s very first token was EOS, meaning the model believed the prompt already contained a complete response and produced no content at all. In practice this is rare but possible, and handling it here prevents a downstream crash.

The initial token collection is done immediately after this:

```python
for i, tok in enumerate(next_tokens.tolist()):
    if not finished[i]:
        generated[i].append(tok)
```

Only tokens from requests that are not already finished are added to `generated`. A request that produced EOS as its first token has an empty `generated[i]` list and will have `text = ""` in its result dict.

---

## Pad Injection

```python
current_tokens = next_tokens.unsqueeze(1)   # [B, 1]
current_tokens = torch.where(
    finished.unsqueeze(1),
    torch.full_like(current_tokens, self.pad_id),
    current_tokens,
)
```

`torch.where(condition, x, y)` selects from `x` where `condition` is `True` and from `y` where it is `False`. Here, for every row where `finished[i]` is `True`, the token is replaced with `pad_token_id`. For every active row, the token is kept as-is.

`finished.unsqueeze(1)` adds a dimension to make the shape `[B, 1]`, matching `current_tokens`. The result is a `[B, 1]` tensor where finished rows contain the pad token and active rows contain the next token to generate.

The forward pass receives this `[B, 1]` input. Finished requests are not excluded from the computation — removing rows mid-loop would change the batch shape and break the KV cache, which stores tensors of shape `[B, heads, seq_len, head_dim]` and cannot easily drop a row. Instead, finished requests contribute a meaningless pad token. The attention layers process it, append its key and value to the cache, and produce logits for it. Those logits are then discarded by the token collection logic at the end of the step.

This is a pragmatic choice: a small amount of wasted compute per step (one forward position per finished request) in exchange for a uniform `[B, 1]` shape throughout the loop.

---

## Attention Mask Extension

```python
attention_mask = torch.cat(
    [attention_mask, torch.ones(B, 1, dtype=torch.long, device="cuda")],
    dim=1,
)
```

On every decode step, the attention mask grows by one column. The new column is all ones — even for finished requests. This is necessary to keep the cache shape consistent. The KV cache now holds one more position per layer per batch element. If a finished request's new column were marked as 0, the model's masking utilities would produce an inconsistent mask relative to the cache size, causing errors or incorrect attention outputs. The pad token in the finished row will have near-zero influence on the attention anyway; extending the mask with 1 is the safe choice.

---

## The `finished` Update

```python
newly_finished = next_tokens == self.eos_id
for i, tok in enumerate(next_tokens.tolist()):
    if not finished[i] and not newly_finished[i]:
        generated[i].append(tok)

finished = finished | newly_finished
```

After each decode step, the newly sampled `next_tokens` are checked against `eos_id`. `newly_finished` is a `[B]` bool tensor marking which requests just emitted EOS on this step. The token collection loop adds the sampled token to `generated[i]` only if request `i` was not already finished and did not just finish this step — an EOS token is never appended to the output. `finished` is then updated with bitwise OR: once a request is marked finished, it stays finished.

The loop termination condition is:

```python
if finished.all():
    break
```

`finished.all()` returns `True` when every element of the `[B]` bool tensor is `True`. The loop exits immediately when the last active request produces EOS, without completing the remaining `max_new_tokens - 1 - current_step` iterations. For a batch where all requests finish at similar step counts, this saves meaningful wall time. For a heterogeneous batch, it exits as soon as the slowest request finishes.

---

## What the Result Looks Like

After the loop, `generated[i]` contains exactly the tokens produced by request `i` between the first non-EOS token and the last token before EOS — or up to `max_new_tokens - 1` tokens if EOS was never emitted. `len(generated[i])` is the completion token count for request `i`, which is reported individually in the result dict as `completion_tokens`. TPOT is shared across all requests in the batch because `step_times` measures the wall time of the forward call, which processes all B requests simultaneously.
