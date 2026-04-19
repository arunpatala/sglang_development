# 01 — The Decode Loop

## From Layer 2 to Layer 3

In Layer 2, every call to `generate()` processed one request. The prefill sent a `[1, prompt_len]` tensor and each decode step sent a `[1, 1]` tensor:

```python
# Layer 2 — one request at a time
past_kv = KVCache()
out = self.model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
next_token_id = sample_next_token(out.logits[0, -1, :], temperature)

for _ in range(max_new_tokens - 1):
    current_token = torch.tensor([[next_token_id]], device=self.device)  # [1, 1]
    out = self.model(input_ids=current_token, past_key_values=past_kv, use_cache=True)
    next_token_id = sample_next_token(out.logits[0, -1, :], temperature)
```

The GPU processed one token per decode step. For Qwen3-0.6B, which has 1.2 GB of weights, the arithmetic for a single `[1, 1]` input is trivial — but those weights still have to be loaded from GPU memory on every step. The hardware is almost entirely idle waiting on memory reads rather than doing arithmetic: roughly 5% GPU utilisation.

In Layer 3, `generate_batch()` accepts B conversations at once. The prefill sends `[B, max_prompt_len]` — all prompts in a single forward pass — and each decode step sends `[B, 1]`:

```python
# Layer 3 — B requests simultaneously
input_ids, attention_mask, prompt_lens_list = self.tokenizer.prepare_batch(batch_messages)
prefill_position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

past_kv = KVCache()
out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                 position_ids=prefill_position_ids, past_key_values=past_kv, use_cache=True)
next_tokens = sample_batch(out.logits[:, -1, :], temperature)  # [B]

for _ in range(max_new_tokens - 1):
    if finished.all():
        break
    current_tokens = ...   # [B, 1], pad-injected for finished requests
    out = self.model(input_ids=current_tokens, attention_mask=attention_mask,
                     position_ids=decode_position_ids, past_key_values=past_kv, use_cache=True)
    next_tokens = sample_batch(out.logits[:, -1, :], temperature)  # [B]
```

The decode tensor is now `[B, 1]` — B tokens per step instead of one. The weight load is the same, but the arithmetic is B× larger. GPU utilisation climbs toward 80% at B=16. Each section below explains one piece of what this loop is doing.

---

## The Complete `BatchedKVCacheModel.generate_batch`

Here is the full `generate_batch` method from `model.py`, unabridged:

```python
def generate_batch(
    self,
    batch_messages: list[list[dict]],   # list of B conversations
    max_new_tokens: int = 128,
    temperature: float = 1.0,
) -> list[dict]:
    B = len(batch_messages)
    t0 = time.perf_counter()

    # --- Step 1: format + tokenize all prompts ---
    input_ids, attention_mask, prompt_lens_list = self.tokenizer.prepare_batch(
        batch_messages
    )
    max_prompt_len = input_ids.shape[1]
    prompt_lens = torch.tensor(prompt_lens_list, dtype=torch.long, device="cuda")

    # --- Step 2: batched prefill ---
    prefill_position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

    past_kv = KVCache()
    t_prefill = time.perf_counter()
    with torch.no_grad():
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=prefill_position_ids,
            past_key_values=past_kv,
            use_cache=True,
        )
    past_kv = out.past_key_values
    ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)

    # --- Step 3: sample first tokens ---
    next_tokens = sample_batch(out.logits[:, -1, :], temperature)  # [B]

    # --- Step 4: decode loop ---
    finished   = next_tokens == self.eos_id
    generated  = [[] for _ in range(B)]
    step_times = []
    decode_step = 0

    for i, tok in enumerate(next_tokens.tolist()):
        if not finished[i]:
            generated[i].append(tok)

    for _ in range(max_new_tokens - 1):
        if finished.all():
            break

        t_step = time.perf_counter()

        current_tokens = next_tokens.unsqueeze(1)   # [B, 1]
        current_tokens = torch.where(
            finished.unsqueeze(1),
            torch.full_like(current_tokens, self.pad_id),
            current_tokens,
        )

        attention_mask = torch.cat(
            [attention_mask, torch.ones(B, 1, dtype=torch.long, device="cuda")],
            dim=1,
        )

        decode_position_ids = (prompt_lens + decode_step).unsqueeze(1).to("cuda")  # [B, 1]
        decode_step += 1

        with torch.no_grad():
            out = self.model(
                input_ids=current_tokens,
                attention_mask=attention_mask,
                position_ids=decode_position_ids,
                past_key_values=past_kv,
                use_cache=True,
            )

        past_kv = out.past_key_values
        next_tokens = sample_batch(out.logits[:, -1, :], temperature)  # [B]
        step_times.append(time.perf_counter() - t_step)

        newly_finished = next_tokens == self.eos_id
        for i, tok in enumerate(next_tokens.tolist()):
            if not finished[i] and not newly_finished[i]:
                generated[i].append(tok)

        finished = finished | newly_finished

    # --- Step 5: build result dicts ---
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    tpot_ms = round(
        (sum(step_times) / len(step_times)) * 1000, 1
    ) if step_times else ttft_ms

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

    return results
```

---

## Line-by-Line Walkthrough

### Step 1 — Format and Tokenize All Prompts

```python
input_ids, attention_mask, prompt_lens_list = self.tokenizer.prepare_batch(batch_messages)
max_prompt_len = input_ids.shape[1]
prompt_lens = torch.tensor(prompt_lens_list, dtype=torch.long, device="cuda")
```

`prepare_batch` applies the chat template to every conversation and tokenizes the resulting strings as a padded batch. The return values are `input_ids` of shape `[B, max_prompt_len]`, `attention_mask` of shape `[B, max_prompt_len]`, and `prompt_lens_list` — the real (unpadded) token count for each request. `prompt_lens` is a GPU tensor copy of that list, used later to compute per-request position IDs in the decode loop. Section 02 covers the tokenizer in detail.

### Step 2 — Batched Prefill

```python
prefill_position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

past_kv = KVCache()
with torch.no_grad():
    out = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=prefill_position_ids,
        past_key_values=past_kv,
        use_cache=True,
    )
past_kv = out.past_key_values
ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)
```

One forward pass processes all B prompts. The model receives a `[B, max_prompt_len]` input and produces `out.logits` of shape `[B, max_prompt_len, vocab_size]`. As each of the 28 attention layers runs, it calls `past_kv.update` for every token in every row, populating the KV cache with shape `[B, heads, max_prompt_len, head_dim]`. By the time the call returns, the cache holds key and value tensors for all B prompts across all layers.

`prefill_position_ids` is passed explicitly to correct a RoPE encoding problem introduced by left-padding — section 03 explains why it is necessary and how the `cumsum` formula works. TTFT is measured here, covering exactly the cost of this first forward pass.

### Step 3 — Sample First Tokens

```python
next_tokens = sample_batch(out.logits[:, -1, :], temperature)  # [B]
```

`out.logits[:, -1, :]` slices the last-position logit vector for every row simultaneously, giving a `[B, vocab_size]` matrix. `sample_batch` converts this to a `[B]` tensor of token IDs — one first-generated token per request. Section 04 covers the sampling implementation.

### Step 4 — The Decode Loop

The first sampled tokens are appended to `generated` for any request that did not immediately produce EOS, then the decode loop begins. Each iteration runs one decode step across all B requests in parallel.

`current_tokens` is built from `next_tokens` reshaped to `[B, 1]`, with finished requests having their token replaced by `pad_id`. This keeps the batch shape uniform — the forward call always receives exactly `[B, 1]` — while ensuring finished requests do not write meaningful content into the cache. `attention_mask` is extended by one column of ones per step, keeping the model's mask consistent with the growing sequence. `decode_position_ids` gives each request its own absolute position rather than a shared offset — section 03 covers this in detail.

After each forward pass, `out.logits[:, -1, :]` again gives `[B, vocab_size]`, from which `sample_batch` draws the next `[B]` tokens. Newly finished requests are merged into `finished` and the loop exits early if `finished.all()`. Section 05 covers the finished mask in full.

### Step 5 — Build Result Dicts

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

`generated` is a list of B lists of token IDs. `decode_batch` converts each to a string. TPOT is the average of `step_times`, which contains only decode step durations — prefill is measured separately as TTFT and never enters the average. The response dict schema is identical to Layer 2's, so `server.py` wraps each result into a `GenerateResponse` Pydantic model without modification.

The single-request `generate()` method is a one-line wrapper that calls `generate_batch([messages])` and returns `results[0]`. The `/generate` endpoint continues to work exactly as it did in Layer 2.
