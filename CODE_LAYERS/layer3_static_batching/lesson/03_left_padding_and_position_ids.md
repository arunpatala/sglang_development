# 03 — Left Padding and Position IDs

## The Line That Fixes Batched Attention

In `generate_batch()`, immediately before the prefill forward pass, a single line computes explicit position IDs:

```python
prefill_position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
```

And inside the decode loop, another line computes per-request decode positions:

```python
decode_position_ids = (prompt_lens + decode_step).unsqueeze(1)   # [B, 1]
```

Without these two lines, the batch logits would not match what B separate B=1 runs would produce. `verify_batch.py` validates this: it compares logits from a B=4 batch against 4 independent B=1 runs and checks that the maximum difference stays within bfloat16 tolerance. Both `position_ids` computations are required to pass that test. This section explains why.

---

## Why Left Padding Alone Is Not Enough

As section 02 established, the tokenizer left-pads all prompts to the same length. For a batch of two prompts with 10 and 50 tokens respectively, the `input_ids` tensor looks like this after padding to length 50:

```
Row 0: [PAD, PAD, PAD, ..., PAD, t0, t1, ..., t9]   (40 padding + 10 real tokens)
Row 1: [t0, t1, t2, ..., t49]                        (50 real tokens, no padding)
```

The `attention_mask` correctly marks the real tokens with 1 and the padding with 0. The model's attention computation uses this mask to prevent attending to padding positions. So far this is correct.

The problem is how the model assigns positions. Transformer models use positional encodings to tell the model where each token sits in the sequence. Qwen3 uses Rotary Position Embeddings (RoPE), which encode position information into the query and key vectors of every attention head. The encoding is a rotation applied to the vector, and the angle of that rotation depends on the position index.

If no explicit `position_ids` are passed to the model, HuggingFace assigns sequential positions `0, 1, 2, ..., max_len-1` to every row uniformly. For row 0 in the example above, the 40 padding positions receive indices 0–39 and the 10 real tokens receive indices 40–49. For row 1, the 50 real tokens receive indices 0–49. The 10 real tokens in row 0 are now encoded with positions 40–49 instead of 0–9, which is what an independent B=1 run would assign them. Their query and key vectors are computed with different rotations. Their attention patterns differ. Their logits are wrong.

---

## The Prefill Fix

The correct position for a real token at column `j` in row `i` is its rank among the real tokens of that row — 0 for the first real token, 1 for the second, and so on regardless of how much padding precedes it.

The `cumsum` formula derives exactly this from the `attention_mask`:

```python
prefill_position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
```

For a mask row `[0, 0, 0, 1, 1, 1, 1, 1]` (three padding, five real tokens), the computation is:

```
attention_mask:       [0, 0, 0, 1, 1, 1, 1, 1]
cumsum(-1):           [0, 0, 0, 1, 2, 3, 4, 5]
subtract 1:           [-1,-1,-1, 0, 1, 2, 3, 4]
clamp(min=0):         [0, 0, 0,  0, 1, 2, 3, 4]
```

The five real tokens receive positions 0, 1, 2, 3, 4 — exactly what an independent B=1 run would assign to a 5-token prompt. The three padding positions receive 0, which is harmless because the attention mask zeroes out their contribution to the attention computation anyway.

This tensor has the same shape as `input_ids` — `[B, max_len]` — and is passed directly to the model:

```python
out = self.model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=prefill_position_ids,
    past_key_values=past_kv,
    use_cache=True,
)
```

With explicit `position_ids`, HuggingFace bypasses its default sequential assignment and uses these values for the RoPE rotations. Every real token in every row is encoded with the same position it would have in an independent single-request run.

---

## The Decode Fix

During the decode loop, each step processes a `[B, 1]` tensor — one new token per request. Without explicit `position_ids`, the model would assign position 0 to every row, since each decode input has only one column. The correct position for request `i` on decode step `k` is `prompt_lens[i] + k` — the length of its actual prompt plus the number of decode steps already completed.

```python
decode_position_ids = (prompt_lens + decode_step).unsqueeze(1).to("cuda")   # [B, 1]
decode_step += 1
```

`prompt_lens` is the `[B]` tensor of real prompt lengths computed during tokenization. For a batch where prompt lengths are `[10, 50]`, on decode step 0 the position IDs are `[[10], [50]]`. On step 1 they are `[[11], [51]]`. Each request independently tracks its own position in the sequence, correctly continuing from where its prefill ended. A shared `max_prompt_len + step` offset would give position 50 to both requests on step 0, which is correct for the 50-token prompt but wrong for the 10-token one.

The consequence of getting decode `position_ids` wrong is the same as getting prefill `position_ids` wrong: the RoPE rotations are incorrect, attention patterns differ from a B=1 run, and logits diverge. The divergence compounds over decode steps because each step's wrong logit leads to a different sampled token, which the next step then encodes with another wrong position.

---

## How the KV Cache Interacts with Position IDs

The KV cache stores the key and value tensors produced during prefill. Because we passed correct `position_ids` during prefill, those tensors have correct RoPE encodings. During each decode step, the attention layer computes fresh K and V for the single new token using its decode `position_ids`, appends them to the cache, and attends the new query against the full accumulated K and V. The positions form a consistent sequence: 0, 1, 2, ..., prompt_len-1 from prefill, then prompt_len, prompt_len+1, ... from decode. Every attention score is computed between correctly positioned queries and keys.

This is what `verify_batch.py` checks: that the final logits from the batched run match the final logits from individual B=1 runs, confirming that padding, caching, and position encoding have all been handled correctly.
