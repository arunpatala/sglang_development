# 01 — The Decode Loop

## From Layer 1 to Layer 2

In Layer 1, every decode step fed the full growing sequence back to the model:

```python
# Layer 1 — ids grows by one token each step
ids = input_ids                                      # shape [1, prompt_len]
generated_ids = []

for step in range(max_new_tokens):
    with torch.no_grad():
        out = self.model(input_ids=ids, use_cache=False)

    next_token_id = sample_next_token(out.logits[0, -1, :], temperature)

    if next_token_id == self.eos_id:
        break

    generated_ids.append(next_token_id)
    ids = torch.cat([ids, torch.tensor([[next_token_id]], device=self.device)], dim=1)
```

`ids` starts at `[1, prompt_len]` and grows by one column each step. By step 50 the model is processing a sequence of `prompt_len + 50` tokens — even though only one new token has been added. The attention computation over all those tokens, at every layer, is the redundant work Layer 2 eliminates.

In Layer 2 the loop is restructured into two separate phases. The first forward pass — prefill — processes the full prompt and builds a cache of the key and value tensors produced by every attention layer. Every subsequent decode step sends only the single newest token and reads the rest from the cache:

```python
# Layer 2 — prefill once, then one token per decode step
past_kv = KVCache()
out = self.model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
past_kv = out.past_key_values               # cache populated with all prompt K/V
next_token_id = sample_next_token(out.logits[0, -1, :], temperature)

for _ in range(max_new_tokens - 1):
    current_token = torch.tensor([[next_token_id]], device=self.device)   # always [1, 1]
    out = self.model(input_ids=current_token, past_key_values=past_kv, use_cache=True)
    past_kv = out.past_key_values           # cache extended by one position
    next_token_id = sample_next_token(out.logits[0, -1, :], temperature)
```

`input_ids` inside the loop is always `[1, 1]` — one token — regardless of how long the context has grown. The model reads everything it needs from `past_kv`. The change to `model.py` from Layer 1 is two new arguments (`past_key_values`, `use_cache=True`) and one assignment per call (`past_kv = out.past_key_values`). `server.py`, `benchmark.py`, and the API are untouched. Each section that follows explains one piece of what this loop is doing.

---

## The Complete `KVCacheModel.generate`

Here is the full generate method from `model.py`, unabridged:

```python
def generate(
    self,
    messages: list[dict],
    max_new_tokens: int = 64,
    temperature: float = 1.0,
) -> dict:
    t0 = time.perf_counter()

    # --- Step 1: apply chat template and tokenize ---
    formatted = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = self.tokenizer(formatted, return_tensors="pt").input_ids.to(self.device)
    prompt_tokens = input_ids.shape[1]
    logger.info(f"prompt_tokens={prompt_tokens}")

    # --- Step 2: prefill ---
    past_kv = KVCache()
    t_prefill = time.perf_counter()
    with torch.no_grad():
        out = self.model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)

    past_kv = out.past_key_values
    logger.info(f"after prefill: {past_kv}")
    next_token_logits = out.logits[0, -1, :]
    next_token_id = sample_next_token(next_token_logits, temperature)
    ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)

    generated_ids: list[int] = []
    step_times: list[float] = []

    if next_token_id == self.eos_id:
        text = ""
        tpot_ms = 0.0
    else:
        generated_ids.append(next_token_id)

        # --- Step 3: decode loop ---
        for _ in range(max_new_tokens - 1):
            t_step = time.perf_counter()

            current_token = torch.tensor(
                [[next_token_id]], dtype=torch.long, device=self.device
            )

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

        tpot_ms = round(
            (sum(step_times) / len(step_times)) * 1000, 1
        ) if step_times else ttft_ms

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    completion_tokens = len(generated_ids)

    logger.info(
        f"DONE: completion_tokens={completion_tokens} "
        f"latency={latency_ms}ms "
        f"ttft={ttft_ms}ms "
        f"tpot={tpot_ms}ms "
        f"tok/s={round(completion_tokens / (latency_ms / 1000), 1)}"
    )

    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
    }
```

---

## Line-by-Line Walkthrough

### Step 1 — Format and Tokenize

```python
formatted = self.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
input_ids = self.tokenizer(formatted, return_tensors="pt").input_ids.to(self.device)
prompt_tokens = input_ids.shape[1]
```

This is identical to Layer 1. The messages list is formatted into the string the model was trained on, then tokenized into a `LongTensor` of shape `[1, prompt_len]`. `prompt_tokens` is saved for the response payload. Nothing about this step changes with the KV cache.

### Step 2 — Prefill

```python
past_kv = KVCache()
t_prefill = time.perf_counter()
with torch.no_grad():
    out = self.model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)

past_kv = out.past_key_values
logger.info(f"after prefill: {past_kv}")
next_token_logits = out.logits[0, -1, :]
next_token_id = sample_next_token(next_token_logits, temperature)
ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)
```

A fresh `KVCache()` is created — empty, no layers initialised yet. The prefill timer starts. The model runs with the full prompt, and as each of its 28 attention layers processes the tokens, it calls `past_kv.update(new_k, new_v, layer_idx)` to store the key and value tensors it computed. By the time the model returns, `past_kv` contains the key and value tensors for every prompt token at every layer.

`past_kv = out.past_key_values` reassigns the variable to the cache object that `out` holds a reference to — which is the same `KVCache` instance that was mutated during the forward pass. The logger line prints a summary like `KVCache(layers=28, seq_len=47, memory=6.5 MB)`, giving a concrete view of what just happened.

The first generated token is available here — before the decode loop begins. In Layer 1, the first token was produced inside the loop on step 0. In Layer 2 it is produced by the prefill call, which also builds the cache. This is why the loop is bounded by `max_new_tokens - 1` rather than `max_new_tokens`: one token has already been produced. TTFT is recorded immediately after the first sample, isolating the cost of the first forward pass.

### The EOS Check After Prefill

```python
if next_token_id == self.eos_id:
    text = ""
    tpot_ms = 0.0
else:
    generated_ids.append(next_token_id)
    # ... decode loop ...
```

If the model's very first token is EOS — which can happen when the model believes the prompt already contains a complete response — the decode loop is skipped entirely. `text` is empty, TPOT is zero, and the method returns immediately. If the first token is not EOS, it is appended to `generated_ids` and the decode loop begins.

### Step 3 — The Decode Loop

```python
for _ in range(max_new_tokens - 1):
    t_step = time.perf_counter()

    current_token = torch.tensor(
        [[next_token_id]], dtype=torch.long, device=self.device
    )

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

Each iteration is one decode step. `current_token` is always a `[1, 1]` tensor containing only the token most recently sampled — the model never receives the full growing sequence again. Inside the forward pass, each attention layer computes key and value tensors for this single new token, calls `past_kv.update(new_k, new_v, layer_idx)` to append them to the cache, and receives back the full accumulated set to attend over. `out.logits` has shape `[1, 1, vocab_size]` — one position — and `out.logits[0, -1, :]` extracts the `[vocab_size]` vector for that position.

The step timer starts before `current_token` is constructed and stops after `sample_next_token` returns. It captures the full cost of one decode step: tensor creation, GPU forward pass, and CPU sampling. `step_times` collects these timings and is used to compute TPOT at the end.

The EOS check comes after timing. If EOS is sampled the loop breaks, and that final token is not appended to `generated_ids`. Its timing is still recorded in `step_times` and included in the TPOT average — the model did real work to produce it.

### Computing TPOT and Decoding the Output

```python
tpot_ms = round(
    (sum(step_times) / len(step_times)) * 1000, 1
) if step_times else ttft_ms

text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
```

In Layer 2, `step_times` contains only decode step timings — prefill is measured separately as TTFT and never enters this list. In Layer 1, `step_times[0]` was the prefill step and had to be treated as the special TTFT case; TPOT was `step_times[1:]`. In Layer 2 this split is structural rather than by index: TPOT is an unambiguous average of pure decode steps.

`tokenizer.decode(generated_ids, skip_special_tokens=True)` converts the list of integer IDs to a string. `generated_ids` contains only the tokens produced during generation, not the prompt. The returned `text` is a clean response string ready to be sent back to the caller.

### The Returned Dict

```python
return {
    "text": text,
    "prompt_tokens": prompt_tokens,
    "completion_tokens": completion_tokens,
    "latency_ms": latency_ms,
    "ttft_ms": ttft_ms,
    "tpot_ms": tpot_ms,
}
```

The response schema is identical to Layer 1. `server.py` wraps this dict into a `GenerateResponse` Pydantic model without modification. `benchmark.py` reads `ttft_ms` and `tpot_ms` from it without modification. Neither file needed to change.
