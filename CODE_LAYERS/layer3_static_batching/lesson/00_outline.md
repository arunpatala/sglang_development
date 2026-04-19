# Layer 3 — Lesson Outline

## What This Lesson Covers

Layer 2 processed one request at a time. Every decode step sent a `[1, 1]` tensor to the GPU, leaving the hardware roughly 5% utilised. Layer 3 processes B requests simultaneously: prefill becomes a single `[B, max_prompt_len]` forward pass, and every decode step sends `[B, 1]` — B tokens at once. GPU utilisation climbs toward 80%+ as B increases, and total output throughput more than doubles on the benchmark.

The change is confined to `model.py` and a new `tokenizer.py`. `kv_cache.py` is carried over from Layer 2 unchanged — the cache already stored a batch dimension. `server.py` adds one new endpoint (`/generate_batch`) but leaves `/generate` intact. The benchmark sweeps over batch sizes instead of sending requests one by one.

The sections follow the code in `generate_batch()` top to bottom, then address the cost model and the numbers the benchmark produces.

---

## Sections

### 01 — The Decode Loop (`01_the_decode_loop.md`)
- Layer 2's single-request loop vs Layer 3's `generate_batch()` structure: prefill once, decode loop with `[B, 1]`
- Step 1: `tokenizer.prepare_batch()` produces `input_ids [B, max_prompt_len]`, `attention_mask`, `prompt_lens`
- Step 2: batched prefill — one forward pass over all B prompts simultaneously; KV cache populated with shape `[B, heads, max_prompt_len, head_dim]`; TTFT measured here
- Step 3: `sample_batch(out.logits[:, -1, :], temperature)` — samples the first token for each of the B requests
- Step 4: decode loop — feeds `[B, 1]`, extends `attention_mask`, updates `past_kv`, updates `finished`; terminates when `finished.all()`
- Step 5: build result dicts, one per request

### 02 — The Tokenizer (`02_the_tokenizer.md`)
- Why tokenization was extracted into `tokenizer.py`: keeps `model.py` focused on tensors and forward passes; mirrors SGLang's `TokenizerManager`
- `Tokenizer.__init__`: `padding_side="left"` — why left padding is required for decoder-only models; fallback `pad_token_id = eos_token_id`
- `apply_chat_template`: formats a single conversation into the model's expected string
- `encode_batch`: calls HuggingFace tokenizer with `padding=True`; returns `(input_ids, attention_mask, prompt_lens)` where `prompt_lens` is `attention_mask.sum(dim=1)`
- `prepare_batch`: convenience wrapper that formats then encodes a list of conversations; what `generate_batch()` calls at Step 1

### 03 — Left Padding and Position IDs (`03_left_padding_and_position_ids.md`)
- The left-padding requirement: all prompts must align so the last real token sits at position `-1`; right-padding breaks this for batch sampling
- The RoPE position bug: without explicit `position_ids`, HuggingFace assigns positions `0..max_len-1` globally — a 10-token prompt padded to 50 gets RoPE positions 40–49 instead of 0–9, producing wrong attention and breaking logit parity with a B=1 run
- Prefill fix: `prefill_position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)` — real tokens get per-sequence positions `0, 1, 2, ...`; padding tokens get `0` (masked anyway)
- Decode fix: `decode_position_ids = (prompt_lens + decode_step).unsqueeze(1)` — each example `i` is at its own absolute position, not the shared `max_prompt_len + step`
- `verify_batch.py` validates that B=4 logits match 4×B=1 logits within bfloat16 tolerance after applying both fixes

### 04 — Batched Sampling (`04_batched_sampling.md`)
- Layer 2's `sample_next_token(logits: [vocab_size]) -> int` vs Layer 3's `sample_batch(logits: [B, vocab_size]) -> Tensor[B]`
- `argmax(dim=-1)` at temperature 0: returns a `[B]` tensor of greedy choices, one per row
- `torch.multinomial(probs, num_samples=1).squeeze(-1)`: multinomial returns `[B, 1]`; the squeeze collapses it to `[B]`
- First token sampling: `out.logits[:, -1, :]` — shape `[B, vocab_size]`, the last-position logits for each of the B prompts

### 05 — The Finished Mask (`05_the_finished_mask.md`)
- The problem: B requests finish at different steps; the loop cannot stop after any single request emits EOS
- `finished = next_tokens == self.eos_id` — a `[B]` bool tensor initialised after the first sample
- Pad injection: `torch.where(finished.unsqueeze(1), pad_tensor, current_tokens)` — finished requests receive `pad_token_id` as input so they do not pollute the cache with meaningful tokens; their output is discarded
- `attention_mask` extension: one column of ones appended each step for all B requests, including finished ones — keeps the cache shape consistent
- `finished = finished | newly_finished` — monotonically accumulates; a request never un-finishes
- Loop termination: `if finished.all(): break` — the loop runs until the last active request emits EOS or `max_new_tokens - 1` steps complete

### 06 — Padding Waste and GPU Utilisation (`06_padding_waste.md`)
- GPU utilisation numbers: Layer 2's `[1, 1]` → ~5% utilised; Layer 3's `[B, 1]` → B× more work per step, utilisation climbs with B
- Throughput vs latency tradeoff: total tok/s rises with B; TTFT grows with B (more prompts to pad and prefill); TPOT stays near-constant (still memory-bandwidth bound)
- The padding waste: a batch containing a 10-token and a 1000-token prompt pads to `[2, 1000]` — 99% of the prefill row for the short prompt is padding, wasting compute
- Why static batching cannot fix this: the batch is assembled before prefill; the padded length is fixed to the longest prompt in the batch
- The head-of-line blocking problem: short requests must wait for the longest request in the batch to finish before results are returned

### 07 — The Full Loop (`07_the_full_loop.md`)
- End-to-end trace of a single `generate_batch` call, connecting all prior sections in order
- Tokenizer produces `input_ids`, `attention_mask`, `prompt_lens_list`; left-padding aligns last real token to column `-1`
- `prefill_position_ids` corrects RoPE encoding; one batched prefill populates the KV cache and yields `next_tokens [B]`; TTFT recorded
- Decode loop: `torch.where` pad injection, mask extension, per-request `decode_position_ids`, forward pass, `sample_batch`, `finished |=`, `finished.all()` exit
- `tokenizer.decode_batch` converts token lists to strings; individual counts and shared timing assembled into result dicts

### 08 — What Comes Next (`08_whats_next.md`)
- The head-of-line blocking problem stated concisely: a 5-token request in a batch of 20 must wait for the longest request to finish before its result is returned
- Continuous batching: evict finished requests from the batch mid-loop and insert new requests in their place — no request waits for others
- What file changes: the server scheduling loop; `model.py`, `kv_cache.py`, and `tokenizer.py` are untouched
- The KV cache complication: inserting a new mid-flight request requires its prefill to be mixed with ongoing decode steps — this is the key engineering challenge continuous batching solves

---

## Supporting Files

- `summary.md` — blog-post-style summary covering all sections
- `sglang_reference.md` — maps layer3 concepts (batching, tokenizer separation, position_ids, finished mask) to their implementations in the SGLang source tree

---

## Key Code Anchors

| Concept | Location |
|---|---|
| Batch tokenize | `model.py` line 100: `self.tokenizer.prepare_batch(batch_messages)` |
| `input_ids` shape | `model.py` line 103: `max_prompt_len = input_ids.shape[1]` |
| Prefill `position_ids` | `model.py` line 121: `(attention_mask.long().cumsum(-1) - 1).clamp(min=0)` |
| Batched prefill | `model.py` line 126: `out = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=prefill_position_ids, ...)` |
| TTFT | `model.py` line 134: `ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)` |
| First sample | `model.py` line 141: `next_tokens = sample_batch(out.logits[:, -1, :], temperature)  # [B]` |
| `finished` init | `model.py` line 145: `finished = next_tokens == self.eos_id` |
| Pad injection | `model.py` line 163: `torch.where(finished.unsqueeze(1), torch.full_like(..., self.pad_id), current_tokens)` |
| `attention_mask` extension | `model.py` line 172: `torch.cat([attention_mask, torch.ones(B, 1, ...)], dim=1)` |
| Decode `position_ids` | `model.py` line 179: `(prompt_lens + decode_step).unsqueeze(1)` |
| Decode forward | `model.py` line 183: `out = self.model(input_ids=current_tokens, attention_mask=attention_mask, position_ids=decode_position_ids, ...)` |
| Decode sample | `model.py` line 192: `next_tokens = sample_batch(out.logits[:, -1, :], temperature)  # [B]` |
| `finished` update | `model.py` line 200: `finished = finished \| newly_finished` |
| Sampling | `sampling.py`: `sample_batch` |
| Left padding | `tokenizer.py` line 36: `self._tok.padding_side = "left"` |
| Batch encode | `tokenizer.py` line 106: `self._tok(texts, return_tensors="pt", padding=True, truncation=False)` |
