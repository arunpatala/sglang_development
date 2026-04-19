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
# Layer 3 — forward call
out = self.model(
    input_ids=ids,
    attention_mask=mask,
    past_key_values=past_kv,
    use_cache=True,
)
past_kv = out.past_key_values   # re-assign every step
logits  = out.logits[:, -1, :]
```

In Layer 4, the init becomes:

```python
# Layer 4 — model init
from model import Qwen3ForCausalLM
self.model = Qwen3ForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
```

The forward call returns `(logits, past_kv)` directly. `KVCache` — Layer 3's HF-compatible cache — is passed as `past_key_values`. HF's attention layers call `kv.update(key, value, layer_idx)` on it during the forward pass, updating it in-place. The returned second value is therefore redundant and can be discarded:

```python
# Layer 4 — forward call
kv = KVCache()
logits, _ = self.model(
    ids,
    attention_mask=mask,
    past_key_values=kv,
    position_ids=pos,
)
logits = logits[:, -1, :]   # kv already updated in-place, _ discarded
```

The only visible change from the caller's perspective is:
1. `self.model` is loaded via our own `from_pretrained` instead of HF's
2. `past_key_values=kv` replaces `past_key_values=past_kv, use_cache=True`
3. `logits, _ = ...` replaces `out = ...; logits = out.logits; past_kv = out.past_key_values`

Everything else — tokenizer, left-padding, cumsum position IDs, pad injection, mask extension, finished mask, `sample_batch` — is byte-for-byte identical to Layer 3. The rest of this layer is entirely about what is inside `model/config.py` and `model/qwen3.py`.

---

## The Complete `BatchedModel.generate_batch`

The full `generate_batch` from `model_runner.py`, showing the Layer 3 logic running against our new model:

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
    prompt_lens = torch.tensor(prompt_lens_list, dtype=torch.long, device="cuda")

    # ── Prefill ───────────────────────────────────────────────────
    prefill_pos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

    kv = KVCache()
    t_prefill = time.perf_counter()
    with torch.no_grad():
        logits, _ = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=kv,
            position_ids=prefill_pos,
        )
    ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)

    next_tokens = sample_batch(logits[:, -1, :], temperature)   # [B]

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

        decode_pos  = (prompt_lens + decode_step).unsqueeze(1)   # [B, 1]
        decode_step += 1

        with torch.no_grad():
            logits, _ = self.model(
                current,
                attention_mask=attention_mask,
                past_key_values=kv,
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

`prepare_batch` formats each conversation with `apply_chat_template`, tokenises with left-padding, and returns `input_ids [B, max_prompt_len]`, `attention_mask [B, max_prompt_len]`, and `prompt_lens_list`. The tokenizer is inherited from Layer 3 unchanged.

### Step 2 — Prefill

```python
prefill_pos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)   # [B, max_prompt_len]

kv = KVCache()
logits, _ = self.model(input_ids, attention_mask=attention_mask,
                       past_key_values=kv, position_ids=prefill_pos)
ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)
```

`KVCache` implements HF's `DynamicCache` interface — specifically `update(key, value, layer_idx)`, `get_seq_length`, `get_mask_sizes`, and `is_sliding`. HF's attention layers call `kv.update(...)` for each of the 28 layers during the forward pass, appending K and V in-place. By the time `self.model(...)` returns, `kv` already holds all 28 layers' cached keys and values — the `_` is genuinely discardable.

`prefill_pos` corrects RoPE positions for left-padded sequences: the `cumsum` formula assigns each real token its rank within the request, giving position 0 to the first real token regardless of how many padding tokens precede it. This is the same fix introduced in Layer 3, carried over unchanged.

### Step 3 — Sample First Tokens

```python
next_tokens = sample_batch(logits[:, -1, :], temperature)   # [B]
```

`logits[:, -1, :]` extracts the last-position logit vector for every row, giving `[B, vocab_size]`. `sample_batch` converts this to a `[B]` tensor of token IDs. Identical to Layer 3.

### Step 4 — The Decode Loop

On every step, finished requests have their token replaced with `pad_id` via `torch.where`, keeping the input a uniform `[B, 1]`. `attention_mask` grows by one column of ones per step, keeping it consistent with the growing KV cache. `decode_pos` gives each request its own absolute position — `prompt_lens[i] + decode_step` — preserving per-request RoPE correctness for sequences of different lengths. The same `kv` object is passed on every call; HF's attention layers append to it each time. `sample_batch` draws the next `[B]` tokens; `finished` accumulates with `|=` and the loop exits when `finished.all()` is true.

### Step 5 — Build Results

```python
texts = self.tokenizer.decode_batch(generated)
```

`generated` is a list of B token-ID lists, excluding tokens after EOS. TPOT is the mean of `step_times` covering only decode steps; TTFT covers only the prefill. The result dict schema is identical to Layer 3, so `server.py` handles it without modification.

---

## What the `model/` Package Provides

The generate loop calls `self.model(...)` and passes `kv` as `past_key_values`. That is the entire interface between `model_runner.py` and the `model/` package. How the config is parsed and the model skeleton is built is covered in section 02; how weights are streamed into it is covered in section 03.
