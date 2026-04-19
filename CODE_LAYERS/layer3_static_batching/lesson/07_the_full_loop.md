# 07 — The Full Loop

The previous sections each explained one piece of `generate_batch` in isolation. This section traces a single call from start to finish, connecting every piece in the order the code actually runs.

---

## Step 1 — Tokenize

```python
input_ids, attention_mask, prompt_lens_list = self.tokenizer.prepare_batch(batch_messages)
prompt_lens = torch.tensor(prompt_lens_list, dtype=torch.long, device="cuda")
```

`prepare_batch` formats each of the B conversations with `apply_chat_template` and then tokenizes all B strings together with `padding=True`. Because `padding_side="left"` was set on the tokenizer, shorter prompts are padded at the front. The result is `input_ids` of shape `[B, max_prompt_len]` where every row ends with its last real token at column `-1`. `attention_mask` is 1 for real tokens and 0 for padding. `prompt_lens_list` records how many real tokens each row contains — this survives the rest of the method as the per-request position offset needed in the decode loop.

---

## Step 2 — Prefill

```python
prefill_position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

past_kv = KVCache()
with torch.no_grad():
    out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                     position_ids=prefill_position_ids, past_key_values=past_kv, use_cache=True)

past_kv = out.past_key_values
next_tokens = sample_batch(out.logits[:, -1, :], temperature)  # [B]
ttft_ms = ...
```

Before the forward pass, `prefill_position_ids` is computed from `attention_mask`. Without it, HuggingFace would assign positions `0..max_len-1` to every row globally, giving a left-padded 10-token prompt RoPE positions 40–49 instead of 0–9. The `cumsum` formula assigns each real token its correct within-sequence position, matching what an independent B=1 run would produce.

One forward pass processes all B prompts simultaneously. As each of the 28 attention layers runs, it calls `past_kv.update` for every token it processes, populating the KV cache with shape `[B, heads, max_prompt_len, head_dim]`. By the time the call returns, the cache holds key and value tensors for the entire prompt across all layers. `out.logits[:, -1, :]` extracts the last-position logits for every row — shape `[B, vocab_size]` — and `sample_batch` draws one first token per request, returning `next_tokens [B]`. TTFT is recorded immediately after.

---

## Step 3 — The Decode Loop

```python
finished = next_tokens == self.eos_id   # [B] bool
...
for _ in range(max_new_tokens - 1):
    if finished.all():
        break

    current_tokens = torch.where(finished.unsqueeze(1), pad_tensor, next_tokens.unsqueeze(1))

    attention_mask = torch.cat([attention_mask, torch.ones(B, 1, ...)], dim=1)
    decode_position_ids = (prompt_lens + decode_step).unsqueeze(1)   # [B, 1]
    decode_step += 1

    with torch.no_grad():
        out = self.model(input_ids=current_tokens, attention_mask=attention_mask,
                         position_ids=decode_position_ids, past_key_values=past_kv, use_cache=True)

    past_kv = out.past_key_values
    next_tokens = sample_batch(out.logits[:, -1, :], temperature)   # [B]
    finished = finished | (next_tokens == self.eos_id)
```

Every piece of earlier machinery comes together here. `finished` tracks which of the B requests have emitted EOS. `torch.where` replaces finished requests' tokens with `pad_id` so that their rows stay in the batch at the right shape without contributing meaningful content. `attention_mask` grows by one column on every step so the model's masking utilities remain consistent with the growing KV cache. `decode_position_ids` gives request `i` position `prompt_lens[i] + decode_step` rather than the shared `max_prompt_len + decode_step`, maintaining correct RoPE encoding for each request independently. The forward pass sees `[B, 1]` — one new token per request — and extends the KV cache by one position per layer per row. `sample_batch` draws the next `[B]` tokens. `finished` accumulates monotonically. The loop exits as soon as `finished.all()` is `True`.

---

## Step 4 — Decode and Return

```python
texts = self.tokenizer.decode_batch(generated)

results = []
for i in range(B):
    results.append({
        "text": texts[i],
        "prompt_tokens": prompt_lens_list[i],
        "completion_tokens": len(generated[i]),
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
    })
```

`generated[i]` contains only the tokens produced by request `i` — those appended before its EOS, never including the EOS token itself. `decode_batch` converts each token list to a string. The result dict for each request carries its own `prompt_tokens` and `completion_tokens` counts, because these differ per request. `ttft_ms` and `tpot_ms` are shared across the batch: TTFT was one batched prefill that served all B requests simultaneously, and TPOT is the average wall time of the decode steps, which each processed all B requests in one forward call.

The list of B result dicts is returned to `server.py`, which wraps each into a `GenerateResponse` and assembles the `BatchGenerateResponse` with aggregate throughput stats. The single-request `/generate` endpoint calls `generate_batch([messages])` and returns `results[0]`, so it continues to work without any changes.
