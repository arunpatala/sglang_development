# Layer 3 — Summary

Layer 3 makes inference faster by processing multiple requests simultaneously. The model is the same, `kv_cache.py` is unchanged, and the `/generate` endpoint still works — but a new `generate_batch()` method processes B requests in a single forward pass instead of one at a time, more than doubling total output throughput on the benchmark.

---

## From Layer 2 to Layer 3

In Layer 2, every call to `generate()` processed one request. The prefill was `[1, max_prompt_len]` and each decode step was `[1, 1]`:

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

In Layer 3, `generate_batch()` accepts B conversations at once. The prefill is `[B, max_prompt_len]` — all prompts processed in one forward pass — and each decode step is `[B, 1]` — one new token per request per step:

```python
# Layer 3 — B requests at once
input_ids, attention_mask, prompt_lens_list = self.tokenizer.prepare_batch(batch_messages)
prefill_position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

past_kv = KVCache()
out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                 position_ids=prefill_position_ids, past_key_values=past_kv, use_cache=True)
next_tokens = sample_batch(out.logits[:, -1, :], temperature)  # [B]

for _ in range(max_new_tokens - 1):
    if finished.all():
        break
    current_tokens = torch.where(finished.unsqueeze(1), pad_tensor, next_tokens.unsqueeze(1))  # [B, 1]
    out = self.model(input_ids=current_tokens, attention_mask=attention_mask,
                     position_ids=decode_position_ids, past_key_values=past_kv, use_cache=True)
    next_tokens = sample_batch(out.logits[:, -1, :], temperature)  # [B]
```

The GPU now does B× more useful work per step. Layer 2's `[1, 1]` decode tensor kept the GPU at roughly 5% utilisation. Layer 3's `[B, 1]` tensor pushes it toward 80% at B=16 on an RTX 4060 Ti. Each section below explains one part of what this loop is doing.

---

## The Tokenizer

Tokenization was extracted from `model.py` into its own `Tokenizer` class in `tokenizer.py`. The single call that replaces several lines of inline tokenization from prior layers is:

```python
input_ids, attention_mask, prompt_lens_list = self.tokenizer.prepare_batch(batch_messages)
```

`prepare_batch` applies the chat template to each of the B conversations, then calls HuggingFace's tokenizer with `padding=True` to produce a padded batch. The return values are `input_ids` of shape `[B, max_prompt_len]`, `attention_mask` of shape `[B, max_prompt_len]` (1 for real tokens, 0 for padding), and `prompt_lens_list` — a Python list of the real token count for each request. `prompt_lens_list` is used later to compute per-example position IDs for the decode loop.

The tokenizer is initialised with `padding_side="left"`. Left padding ensures the last real token of every sequence sits at index `-1` in every row, which is where the model must attend to produce the next token. Right padding would shift this position by a different amount for every request, making it impossible to extract the correct logit with a single `[:, -1, :]` slice.

---

## Left Padding and Position IDs

Left padding creates a subtle problem. Without explicit `position_ids`, HuggingFace assigns sequential positions `0, 1, 2, ..., max_len-1` to every row globally. A 10-token prompt padded to 50 gets its real tokens at RoPE positions 40–49 instead of 0–9. The model's attention pattern depends on these positions through rotary embeddings, so the logits differ from what a B=1 run would produce — the batch result is wrong.

The fix for prefill is:

```python
prefill_position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
```

For a row with mask `[0, 0, 1, 1, 1]`, `cumsum` gives `[0, 0, 1, 2, 3]`, subtracting 1 gives `[-1, -1, 0, 1, 2]`, and clamping gives `[0, 0, 0, 1, 2]`. The real tokens receive positions 0, 1, 2 — exactly what a B=1 run would assign. The padding positions receive 0, but they are masked in attention anyway.

The decode fix is separate. Each decode step, request `i` generates its next token at position `prompt_lens[i] + decode_step`, not the shared `max_prompt_len + decode_step`:

```python
decode_position_ids = (prompt_lens + decode_step).unsqueeze(1)   # [B, 1]
```

Without this, two requests with different prompt lengths would both be assigned the same decode position, again breaking their rotary embeddings relative to a B=1 run. `verify_batch.py` validates that the batch logits match the corresponding B=1 logits within bfloat16 tolerance after both fixes are applied.

---

## Batched Sampling

In Layer 2, sampling produced a single integer:

```python
def sample_next_token(logits, temperature):   # [vocab_size] → int
    ...
    return int(torch.multinomial(probs, num_samples=1).item())
```

In Layer 3, sampling operates on the full `[B, vocab_size]` matrix and returns a `[B]` tensor:

```python
def sample_batch(logits, temperature):        # [B, vocab_size] → Tensor[B]
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

`argmax(dim=-1)` operates row-wise, returning one greedy token per row. `torch.multinomial` with `num_samples=1` returns shape `[B, 1]`; the `.squeeze(-1)` collapses it to `[B]`. The `[:, -1, :]` slice on `out.logits` extracts the last-position logit vector for each of the B prompts simultaneously, giving the `[B, vocab_size]` input `sample_batch` expects.

For now, every request in the batch shares a single `temperature` value. In continuous batching, each request arrives independently and carries its own sampling parameters, so the sampling call will need to accept a per-request parameter vector rather than a scalar.

---

## The Finished Mask

B requests do not all finish at the same step. The loop cannot stop when any single request emits EOS — it must continue until the last active request finishes. A `[B]` boolean tensor tracks this:

```python
finished = next_tokens == self.eos_id   # [B] bool, initialised after first sample

# Inside the loop:
current_tokens = torch.where(
    finished.unsqueeze(1),
    torch.full_like(current_tokens, self.pad_id),
    current_tokens,
)
finished = finished | newly_finished
```

Once a request sets its `finished` flag, every subsequent step injects `pad_token_id` as its input token. The forward pass still runs for that row — the batch shape must remain `[B, 1]` — but its output logits are ignored and its generated token list is closed. The loop terminates with `if finished.all(): break` rather than waiting for `max_new_tokens - 1` steps if all requests finish early. `finished` is monotonically accumulated with bitwise OR, so a request never transitions back to active once it has emitted EOS.

---

## Padding Waste and GPU Utilisation

Static batching buys throughput at the cost of TTFT and wasted prefill compute. The GPU utilisation numbers from the model docstring make the gain concrete: Layer 2's `[1, 1]` decode tensor kept the GPU at roughly 5% utilisation; Layer 3's `[B, 1]` tensor does B× more arithmetic per step, pushing utilisation toward 80% as B grows.

The cost is padding waste. Every prompt in a batch is padded to the length of the longest prompt. A batch containing a 10-token and a 1000-token prompt produces an `input_ids` tensor of shape `[2, 1000]`. The prefill processes 2000 token positions, but 990 of them — the padding row for the short prompt — contribute nothing to the output. For batches with highly variable prompt lengths, the majority of prefill compute is spent on padding.

The second cost is head-of-line blocking. Once a batch is assembled and prefill begins, no request can leave the batch until the decode loop terminates. A 5-token request and a 1000-token response request in the same batch of 16 share the same decode loop. The 5-token request's result is ready after a handful of decode steps, but it is not returned to the caller until the longest request in the batch finishes. This latency inflation is the fundamental flaw of static batching that continuous batching addresses.

---

## The Full Loop

Now that all the parts have been explained, it is worth tracing a single `generate_batch` call from end to end to see how they fit together.

The call arrives with a list of B conversations. The tokenizer formats each one with `apply_chat_template`, tokenizes the resulting strings as a left-padded batch, and returns `input_ids [B, max_prompt_len]`, `attention_mask [B, max_prompt_len]`, and `prompt_lens_list`. Left-padding aligns every prompt so its last real token sits at column `-1`. The `attention_mask` marks which positions are real and which are padding.

Before the prefill forward pass, `prefill_position_ids` is computed from `attention_mask` using the `cumsum` formula. This corrects the RoPE encoding that HuggingFace would otherwise assign incorrectly to padded rows. One batched forward pass then processes all B prompts simultaneously, populating the KV cache with shape `[B, heads, max_prompt_len, head_dim]` and producing `out.logits [B, max_prompt_len, vocab_size]`. The `[:, -1, :]` slice extracts the last-position logits for each row, and `sample_batch` draws one token per request from the resulting `[B, vocab_size]` matrix, returning `next_tokens [B]`. TTFT is recorded here.

The decode loop begins. On every step, finished requests have their token replaced with `pad_id` via `torch.where`, keeping the input shape a uniform `[B, 1]`. The `attention_mask` is extended by one column of ones, and `decode_position_ids` gives each request its own absolute position — `prompt_lens[i] + decode_step` — rather than a shared offset. Another forward pass runs over `[B, 1]`, the KV cache grows by one position per layer per request, and `sample_batch` draws the next `[B]` tokens. Newly finished requests are merged into `finished` with `|=`. The loop exits when `finished.all()` is `True`.

After the loop, `tokenizer.decode_batch` converts each request's token list to a string. One result dict is returned per request, with individual `prompt_tokens` and `completion_tokens` counts and a shared `ttft_ms` and `tpot_ms` measured from the single batched computation.

---

## What Comes Next

The throughput gain of static batching comes with an unavoidable latency cost: every request in a batch waits for the slowest one. A serving system that prioritises tail latency — guaranteeing that short requests are not blocked behind long ones — cannot use static batching as its scheduling primitive.

Continuous batching replaces the static batch with a dynamic one. Rather than assembling B requests, running prefill, and then running a fixed decode loop until all B finish, a continuous batching scheduler allows finished requests to leave the batch mid-loop and new requests to enter with their own prefill injected between decode steps. The batch size fluctuates dynamically, and no request waits for another. The KV cache, `tokenizer.py`, and `model.py`'s forward call are unchanged. The work moves into the scheduler — the loop that decides which requests enter the next forward pass and what shape the input tensor takes at each step.
