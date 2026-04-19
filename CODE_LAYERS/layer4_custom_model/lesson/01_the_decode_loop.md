# 01 — The Decode Loop

## From Layer 3 to Layer 4

In Layer 3, two lines in `model_runner.py` handed the entire architecture to HuggingFace:

```python
# Layer 3 — model init
from transformers import AutoModelForCausalLM
self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
```

Every forward call returned a `ModelOutput` namedtuple. The generate loop had to index into it to extract logits and re-assign the KV cache from the output on every decode step:

```python
# Layer 3 — forward call in generate_batch
out = self.model(
    input_ids=ids,
    attention_mask=mask,
    past_key_values=past_kv,
    use_cache=True,
)
past_kv = out.past_key_values   # re-assign every step
logits  = out.logits[:, -1, :]
```

In Layer 4, the init becomes two different lines:

```python
# Layer 4 — model init
from model import Qwen3ForCausalLM
self.model = Qwen3ForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
```

The forward call returns a logits tensor directly. The `KVCache` object is passed in and mutated in-place by each of the 28 attention layers, so there is nothing to extract or re-assign after the call:

```python
# Layer 4 — forward call in generate_batch
logits = self.model(
    ids,
    attention_mask=mask,
    kv_cache=kv,
    position_ids=pos,
)
logits = logits[:, -1, :]       # kv already updated in-place, no re-assign
```

Everything else in `generate_batch` — the tokenizer call, the prefill position IDs, the pad injection, the attention mask extension, the decode position IDs, the finished mask, `sample_batch` — is byte-for-byte identical to Layer 3. The rest of this layer is entirely about what is inside the `model/` package.

---

## The Complete `BatchedModel.generate_batch`

The full `generate_batch` method from `model_runner.py`, unabridged, shows the Layer 3 logic running against our new model:

```python
def generate_batch(
    self,
    batch_messages: list[list[dict]],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
) -> list[dict]:
    B = len(batch_messages)
    t0 = time.perf_counter()

    # ── Tokenise ──────────────────────────────────────────────────
    input_ids, attention_mask, prompt_lens_list = self.tokenizer.prepare_batch(
        batch_messages
    )
    max_prompt_len = input_ids.shape[1]
    prompt_lens = torch.tensor(prompt_lens_list, dtype=torch.long, device="cuda")

    # ── Prefill ───────────────────────────────────────────────────
    prefill_pos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)  # [B, max_len]

    kv = KVCache()
    t_prefill = time.perf_counter()
    with torch.no_grad():
        logits = self.model(
            input_ids,
            attention_mask=attention_mask,
            kv_cache=kv,
            position_ids=prefill_pos,
        )
    ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)

    next_tokens = sample_batch(logits[:, -1, :], temperature)  # [B]

    # ── Decode loop ───────────────────────────────────────────────
    finished   = next_tokens == self.eos_id
    generated  = [[] for _ in range(B)]
    step_times: list[float] = []
    decode_step = 0

    for i, tok in enumerate(next_tokens.tolist()):
        if not finished[i]:
            generated[i].append(tok)

    for _ in range(max_new_tokens - 1):
        if finished.all():
            break

        t_step = time.perf_counter()

        current = next_tokens.unsqueeze(1)
        current = torch.where(
            finished.unsqueeze(1),
            torch.full_like(current, self.pad_id),
            current,
        )

        attention_mask = torch.cat(
            [attention_mask, torch.ones(B, 1, dtype=torch.long, device="cuda")],
            dim=1,
        )

        decode_pos = (prompt_lens + decode_step).unsqueeze(1)  # [B, 1]
        decode_step += 1

        with torch.no_grad():
            logits = self.model(
                current,
                attention_mask=attention_mask,
                kv_cache=kv,
                position_ids=decode_pos,
            )

        next_tokens = sample_batch(logits[:, -1, :], temperature)
        step_times.append(time.perf_counter() - t_step)

        newly_finished = next_tokens == self.eos_id
        for i, tok in enumerate(next_tokens.tolist()):
            if not finished[i] and not newly_finished[i]:
                generated[i].append(tok)
        finished = finished | newly_finished

    # ── Build results ─────────────────────────────────────────────
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    tpot_ms = (
        round((sum(step_times) / len(step_times)) * 1000, 1)
        if step_times else ttft_ms
    )
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

### Step 1 — Tokenise

```python
input_ids, attention_mask, prompt_lens_list = self.tokenizer.prepare_batch(batch_messages)
prompt_lens = torch.tensor(prompt_lens_list, dtype=torch.long, device="cuda")
```

`prepare_batch` formats each of the B conversations with `apply_chat_template`, tokenises the resulting strings as a left-padded batch, and returns `input_ids [B, max_prompt_len]`, `attention_mask [B, max_prompt_len]`, and `prompt_lens_list` — the real unpadded token count for each request. `prompt_lens` is the GPU tensor version of that list, used later to compute per-request decode position IDs. The tokenizer is inherited from Layer 3 unchanged.

### Step 2 — Prefill

```python
prefill_pos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)  # [B, max_prompt_len]

kv = KVCache()
logits = self.model(input_ids, attention_mask=attention_mask,
                    kv_cache=kv, position_ids=prefill_pos)
ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)
```

One forward pass processes all B prompts simultaneously. `prefill_pos` corrects the RoPE encoding that would otherwise be wrong for left-padded sequences — the same `cumsum` fix carried from Layer 3. The critical difference from Layer 3 is the call signature: `kv_cache=kv` instead of `past_key_values=kv, use_cache=True`, and the return value is a plain `logits` tensor instead of a `ModelOutput`. Each of the 28 attention layers inside `Qwen3ForCausalLM` calls `kv.update(layer_idx, k, v)` during this pass, populating the cache in-place. There is no `kv = out.past_key_values` line — the same `kv` object now holds all 28 layers' cached keys and values. TTFT is recorded immediately after this call.

### Step 3 — Sample First Tokens

```python
next_tokens = sample_batch(logits[:, -1, :], temperature)  # [B]
```

`logits[:, -1, :]` extracts the last-position logit vector for every row, giving `[B, vocab_size]`. `sample_batch` converts this to a `[B]` tensor of token IDs. This is identical to Layer 3.

### Step 4 — The Decode Loop

On every step, finished requests have their token replaced with `pad_id` via `torch.where`, keeping the input a uniform `[B, 1]`. `attention_mask` grows by one column of ones per step. `decode_pos` gives each request its own absolute position — `prompt_lens[i] + decode_step` — rather than a shared offset from `max_prompt_len`. The forward call receives `current [B, 1]` and returns `logits [B, 1, vocab_size]` directly. `sample_batch` draws the next `[B]` tokens. Newly finished requests are merged into `finished` with `|=` and the loop exits early if `finished.all()` is true.

### Step 5 — Build Results

```python
texts = self.tokenizer.decode_batch(generated)
```

`generated` is a list of B lists of token IDs accumulated during the decode loop, excluding any tokens produced after EOS. `decode_batch` converts each list to a string. TPOT is the average of `step_times`, which measures only decode forward passes — prefill cost is captured separately in TTFT. The result dict schema is identical to Layer 3, so `server.py` handles it without modification.

---

## What the `model/` Package Provides

The generate loop calls `self.model(...)` twice — once for prefill, once per decode step — and reads `kv.get_seq_length()` implicitly through the model's internal mask computation. That is the entire interface between `model_runner.py` and the `model/` package. Every other detail — how weights are loaded, how RoPE is computed, how GQA expands KV heads, how the causal mask is constructed — lives in `model/` and is explained in sections 02 through 06.
