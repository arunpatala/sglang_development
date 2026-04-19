# 07 — The Full Loop

The previous sections each explained one piece of `KVCacheModel.generate` in isolation. This section traces a single call from start to finish, connecting every piece in the order the code actually runs.

---

## Step 1 — Tokenize

```python
formatted = self.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
input_ids = self.tokenizer(formatted, return_tensors="pt").input_ids.to(self.device)
prompt_tokens = input_ids.shape[1]
```

The messages list is formatted into the string the model was trained on, then tokenized into `input_ids` of shape `[1, prompt_len]` and moved to the GPU. `prompt_tokens` is saved for the response payload. This step is identical to Layers 0 and 1.

---

## Step 2 — Prefill

```python
past_kv = KVCache()
t_prefill = time.perf_counter()
with torch.no_grad():
    out = self.model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)

past_kv = out.past_key_values
next_token_id = sample_next_token(out.logits[0, -1, :], temperature)
ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)
```

An empty `KVCache()` is created — no layers initialised yet. The model runs with the full prompt. As each of the 28 attention layers processes the prompt tokens, it calls `past_kv.update(new_k, new_v, layer_idx)`. The `update` call on each `LayerCache` stores the key and value tensors for the first time — since `self.keys is None`, no `torch.cat` is needed. By the time the model returns, the cache holds tensors of shape `[1, 8, prompt_len, 128]` for every layer, covering every prompt token.

`out.logits[0, -1, :]` extracts the last-position logit vector — the prediction for what token should follow the end of the prompt. `sample_next_token` converts it to a single token ID. TTFT is recorded immediately: it captures exactly the cost of this forward pass and the first sample, nothing more. The first generated token is available here, before the decode loop starts.

---

## Step 3 — The Decode Loop

```python
for _ in range(max_new_tokens - 1):
    t_step = time.perf_counter()

    current_token = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)

    with torch.no_grad():
        out = self.model(
            input_ids=current_token,
            past_key_values=past_kv,
            use_cache=True,
        )

    past_kv = out.past_key_values
    next_token_logits = out.logits[0, -1, :]
    next_token_id = sample_next_token(next_token_logits, temperature)

    step_times.append(time.perf_counter() - t_step)

    if next_token_id == self.eos_id:
        break

    generated_ids.append(next_token_id)
```

Every step here brings in what the earlier sections explained. `current_token` is always `[1, 1]` — the single most recently sampled token. Inside the forward pass, each attention layer computes fresh K and V for this one new token and calls `past_kv.update(new_k, new_v, layer_idx)`. Because `self.keys` is no longer `None`, `LayerCache.update` appends via `torch.cat`, growing the sequence dimension by 1. The layer then receives back the full accumulated K and V — prompt keys and values plus all previously generated token keys and values — and attends the new token's single query against all of them. The attention cost is `O(L + k)` where `L` is the prompt length and `k` is the number of tokens generated so far, compared to `O((L + k)²)` in Layer 1.

`out.logits[0, -1, :]` has shape `[vocab_size]` — one position was processed, so there is only one logit vector. `sample_next_token` draws the next token. `step_times` records the wall time of every decode step. The EOS check comes after timing: if EOS is sampled the step is still timed (the model did real work) but the token is not appended to `generated_ids` and the loop breaks.

---

## Step 4 — Decode and Return

```python
tpot_ms = round(
    (sum(step_times) / len(step_times)) * 1000, 1
) if step_times else ttft_ms

text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

return {
    "text": text,
    "prompt_tokens": prompt_tokens,
    "completion_tokens": completion_tokens,
    "latency_ms": latency_ms,
    "ttft_ms": ttft_ms,
    "tpot_ms": tpot_ms,
}
```

`step_times` contains only decode step durations — prefill was timed separately as TTFT and never entered the list. TPOT is therefore an unambiguous average of pure decode steps. In Layer 1, `step_times[0]` was the prefill step and had to be treated as TTFT; computing TPOT required slicing `step_times[1:]`. In Layer 2 that special-casing is gone because prefill and decode are structurally separate.

`tokenizer.decode(generated_ids, skip_special_tokens=True)` converts the accumulated integer list to a string. `generated_ids` contains only the tokens appended inside the loop — never the EOS token, never the prompt. The result dict is returned to `server.py`, which wraps it into a `GenerateResponse` Pydantic model. `benchmark.py` reads `ttft_ms` and `tpot_ms` from it directly. Neither file required any changes.
