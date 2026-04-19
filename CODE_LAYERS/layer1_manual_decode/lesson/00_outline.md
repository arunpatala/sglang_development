# Layer 1 — Lesson Outline

## What This Lesson Covers

Layer 0 called `model.generate()` and got text back. Layer 1 opens that black box: we call `model()` directly, receive a tensor of logits, and implement the token selection loop ourselves. The computation is identical — same cost, same throughput — but every step is now in our code and fully visible. This is the foundation that makes all future optimisations possible.

---

## Sections

### 01 — What Changed and Why (`01_what_changed.md`)
- Layer 0 recap: `model.generate()` hid everything behind one call
- Layer 1: `model()` called directly; the loop is ours
- Why this matters: `model.py` owns inference, `server.py` is a thin HTTP wrapper
- The architectural principle: future layers change only `model.py`; `server.py` stays untouched

### 02 — The Forward Pass (`02_the_forward_pass.md`)
- What `self.model(input_ids=ids, use_cache=False)` actually returns
- `out.logits` shape: `[batch=1, seq_len, vocab_size]`
- Why the model produces one score vector per position
- Why we only look at the last position: `out.logits[0, -1, :]`
- `use_cache=False` — prevents HuggingFace from building its own internal cache silently

### 03 — Causal Masking (`03_causal_masking.md`)
- Why full attention would let the model cheat: seeing future tokens makes next-token prediction trivial
- The mask: a triangular matrix that sets all future-position attention scores to `-∞` before softmax
- What `-∞` scores become after softmax: exactly zero weight, so future positions contribute nothing
- A worked 4-token example showing the score matrix before and after masking
- How the mask enables parallel training: all positions trained simultaneously without leaking information
- The key consequence: a token's key and value vectors depend only on what came before it and are immutable once computed
- Why this is the reason the KV cache is a valid optimisation

### 04 — Prefill and Decode (`04_prefill_and_decode.md`)
- Even without a KV cache, these are two conceptually distinct phases
- Prefill: step 0, processes all prompt tokens in parallel, cost dominated by prompt length
- Decode: steps 1..N, each appends one token and reprocesses the full growing sequence
- In layer1 the code looks identical for both — but naming them now prepares for layer2 where they genuinely diverge in implementation
- Why the first step is more expensive than subsequent ones

### 05 — Logits, Softmax, and Sampling (`05_logits_and_sampling.md`)
- What a logit is: raw unnormalised score from the model's final linear layer
- Why you cannot interpret logits directly as probabilities
- Softmax: converts logits to a probability distribution that sums to 1
- Greedy decoding: `argmax` — always pick the highest-probability token, deterministic
- Temperature: scaling logits before softmax — low temp sharpens the distribution, high temp flattens it
- `torch.multinomial`: drawing a sample from the probability distribution
- The `sample_next_token` implementation in `sampling.py` covering all three cases

### 06 — The Decode Loop Line by Line (`06_the_decode_loop.md`)
- Full walkthrough of `NaiveModel.generate` in `model.py`
- Initialise `ids = input_ids` (the prompt)
- The `for step in range(max_new_tokens)` loop
- Forward pass, slice `[0, -1, :]`, call `sample_next_token`
- EOS check: stop if the model signals it is done
- `torch.cat` to grow the sequence by one token
- After the loop: `tokenizer.decode(generated_ids, skip_special_tokens=True)`

### 07 — TTFT and TPOT (`07_ttft_and_tpot.md`)
- Two new metrics added to the response schema in layer1
- TTFT (Time To First Token): `step_times[0]` — cost of the first forward pass, which includes the full prompt
- TPOT (Time Per Output Token): average of `step_times[1:]` — pure decode steps only
- Why they are measured separately: they have different causes and will respond differently to optimisations
- In layer1 with no cache, TPOT grows with sequence length because each step reprocesses more tokens
- What these numbers will tell us once the KV cache is added in layer2

### 08 — What Comes Next (`08_whats_next.md`)
- The surgical change layer2 makes: `past_key_values`
- Because `model.py` owns the loop, the addition is a few lines in one file
- `server.py` and `benchmark.py` do not change between layer1 and layer2
- What TTFT and TPOT numbers are expected to do

---

## Supporting Files

- `summary.md` — blog-post-style summary covering all sections
- `sglang_reference.md` — maps layer1 concepts to their implementations in the SGLang source tree

---

## Key Code Anchors

All lesson code references point to `model.py` in this layer:

| Concept | Location |
|---|---|
| Model forward call | `model.py` line 140: `out = self.model(input_ids=ids, use_cache=False)` |
| Logits slice | `model.py` line 144: `out.logits[0, -1, :]` |
| Sampling | `sampling.py`: `sample_next_token` |
| EOS check | `model.py` line 149: `if next_token_id == self.eos_id: break` |
| Sequence growth | `model.py` line 158: `ids = torch.cat([ids, next_token_tensor], dim=1)` |
| TTFT | `model.py` line 169: `ttft_ms = round(step_times[0] * 1000, 1)` |
| TPOT | `model.py` lines 173–176: average of `step_times[1:]` |
