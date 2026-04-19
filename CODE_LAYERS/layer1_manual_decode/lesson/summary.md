# Layer 1 — Summary

Layer 1 does not make inference faster. It makes it visible. The same computation as Layer 0 — the same model, the same number of forward passes, the same throughput — but now every step of the autoregressive loop is written in our code and open to inspection. This is the foundation that makes all future optimisations possible.

---

## Token-Level Generation

In Layer 0, `model.generate()` produced text as if by magic. In reality every language model generates one token at a time: it receives the current sequence, produces a score for every possible next token, picks one, appends it, and repeats. Written explicitly:

```python
ids = prompt_tokens
while True:
    next_token = model(ids)
    if next_token == EOS:
        break
    ids = ids + [next_token]
```

This loop is the entirety of autoregressive generation. The "predict next token" objective, applied repeatedly, is what gives a language model its capabilities — to do it well the model must implicitly learn vast amounts about how the world works, because all of that structure is reflected in how text is written. Every sentence on the internet is a training signal.

---

## The Forward Pass

In Layer 0 the entire generation process was triggered by one call to `model.generate()`. In Layer 1 we replace that with a direct call to the model itself:

```python
with torch.no_grad():
    out = self.model(input_ids=ids, use_cache=False)

next_token_logits = out.logits[0, -1, :]   # shape: [151936]
```

This is a single forward pass. The model takes the current sequence of token IDs, runs it through all of its layers, and returns a result object. The only field we use is `out.logits`, a three-dimensional tensor of shape `[batch_size, sequence_length, vocab_size]`. The first dimension is always 1 because we process one request at a time. The second is the number of tokens in the sequence. The third is the vocabulary size — 151,936 for Qwen3. For a 20-token prompt this is a tensor of shape `[1, 20, 151936]`.

The model produces a score vector at every sequence position, not just the last one. This is a consequence of how transformers are trained: given the sequence "The cat sat", the model predicts the token after "The", after "The cat", and after "The cat sat" all in one pass, which makes training efficient. At inference time only the last position matters — the logits at `[-1]` represent what the model believes should follow the entire sequence so far. Every other position's scores are discarded.

The slice `out.logits[0, -1, :]` extracts exactly that vector. Index `0` selects the only batch element, `-1` selects the final sequence position regardless of length, and `:` keeps all 151,936 vocabulary scores. The result is a one-dimensional tensor: one number per token in the vocabulary, with higher values indicating the model considers that token a more likely continuation.

`use_cache=False` prevents HuggingFace from quietly saving intermediate attention computations between calls. We disable it deliberately so that every step pays the full cost of reprocessing the sequence from scratch — expensive, but transparent. Layer 2 will turn this back on in a controlled way.

---

## Prefill and Decode

Every generation request has two phases, even when the code looks the same for both.

**Prefill** is the first forward pass. The model processes all prompt tokens in parallel — the GPU sees the entire prompt as one chunk of work and runs attention across all positions simultaneously. Model weights are loaded into GPU registers once and applied across all positions in one pass. The cost scales with prompt length but it all happens in a single, highly parallelisable forward pass.

**Decode** is every subsequent forward pass. The GPU processes one new token per step. This is fundamentally different: the model's weight matrices still need to be loaded from memory — 1.2 GB for Qwen3-0.6B — but only to produce a single token's worth of output. Most GPU compute cores sit idle waiting on memory bandwidth. Decode is memory-bandwidth bound rather than compute bound.

In Layer 1, decode is additionally expensive because the full growing sequence is passed on every step. A prompt of length `L` generating `T` tokens means forward passes over sequences `L`, `L+1`, ..., `L+T`. Layer 2 will eliminate this redundancy.

---

## Logits, Softmax, and Sampling

The 151,936 numbers from the last position are called **logits** — raw, unbounded scores from the model's final linear layer. To turn them into a probability distribution, softmax is applied: each value is exponentiated and divided by the sum of all exponentiated values, producing a vector that is always positive and sums to 1.

```python
probs = torch.softmax(logits, dim=-1)
```

With a probability distribution in hand, there are two main ways to pick a token.

**Greedy decoding** always picks the highest-probability token — the argmax. It is deterministic: the same prompt always produces the same output. It is also the cheapest option since no sampling is involved.

**Temperature sampling** draws a random sample from the distribution. Temperature controls how peaked or flat the distribution is before sampling: values below 1 make the model more confident and focused; values above 1 make it more diffuse and creative; temperature=1 leaves the distribution unchanged.

```python
def sample_next_token(logits, temperature):           # sampling.py
    if temperature == 0.0:
        return int(logits.argmax(dim=-1).item())       # greedy
    if temperature != 1.0:
        logits = logits / temperature                  # scale
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())  # sample
```

Beyond temperature, real systems also use top-k (restrict sampling to the k most likely tokens), top-p / nucleus sampling (restrict to the smallest set of tokens whose cumulative probability exceeds p), and repetition penalty (reduce logits for tokens already in the sequence). All of these modify the logits or probabilities before the final draw.

---

## The Decode Loop

The full loop in `model.py` brings everything together:

```python
ids = input_ids                          # start with the prompt
generated_ids = []
step_times = []

for step in range(max_new_tokens):
    t_step = time.perf_counter()

    with torch.no_grad():
        out = self.model(input_ids=ids, use_cache=False)

    next_token_logits = out.logits[0, -1, :]
    next_token_id = sample_next_token(next_token_logits, temperature)

    step_times.append(time.perf_counter() - t_step)

    if next_token_id == self.eos_id:
        break

    generated_ids.append(next_token_id)
    ids = torch.cat([ids, torch.tensor([[next_token_id]], device=self.device)], dim=1)

text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
```

Two accumulators are initialised before the loop. `ids` is the growing tensor passed to the model on every step — it starts as the prompt and gains one column per iteration. `generated_ids` is a plain Python list of the token IDs the model has produced so far, kept strictly separate from `ids` because at the end we only want to decode the new tokens, not the prompt that was already there.

Each iteration follows a fixed sequence of four actions. First, a forward pass over the full current sequence measures one complete step. On step 0 this is the prefill; on every subsequent step it is a decode step, reprocessing the prompt plus everything generated so far. Second, the last-position logits are sliced and passed to the sampler, which returns a single integer token ID. Third, the step time is recorded — this captures both the GPU forward pass and the CPU sampling, and is the raw material for TTFT and TPOT. Fourth, two branching things happen based on the sampled token: if it is the EOS token the loop breaks immediately; otherwise the token ID is appended to `generated_ids` and `ids` is extended by one column via `torch.cat`.

The `torch.cat` step is worth noting explicitly. `ids` has shape `[1, N]` after `N` tokens. The new token is wrapped in a `[1, 1]` tensor and concatenated along dimension 1, producing `[1, N+1]`. The model then receives this longer sequence on the next iteration — the prompt never disappears from the input, it just accumulates a growing tail of generated tokens.

The EOS check is placed after timing but before appending. This means the EOS token is counted in the step time (TTFT if step 0, TPOT if later) but is never added to `generated_ids`. The output text is clean and the token counts are accurate.

After the loop, `tokenizer.decode(generated_ids, skip_special_tokens=True)` converts the integer list to a string. Decoding `ids` instead would include the full formatted prompt, which is not what the caller wants. Total latency covers everything from the start of `generate` — tokenization, all forward passes, sampling, and final decoding — and is computed from a timer started before step 1.

---

## TTFT and TPOT

Layer 1 adds two latency metrics to the response that Layer 0 did not have.

**TTFT — Time To First Token.** The duration of step 0: the prefill forward pass plus the first sample. This is how long a user waits before anything appears. TTFT is sensitive to prompt length — a longer prompt takes more compute in that first pass.

**TPOT — Time Per Output Token.** The average duration of steps 1 onwards: pure decode steps, excluding the prefill cost. TPOT is what determines the perceived speed of a streaming response. A TPOT of 20 ms corresponds to 50 tokens per second, or roughly 10–12 words per second.

```python
ttft_ms  = step_times[0] * 1000
tpot_ms  = mean(step_times[1:]) * 1000
```

They matter independently because they respond to different optimisations. A KV cache (Layer 2) dramatically reduces TPOT while leaving TTFT roughly unchanged. Batching can increase TPOT for individual requests while improving overall throughput. Tracking both numbers is what makes it possible to understand which part of the system a change is actually improving.

---

## What Comes Next

Layer 2 makes one change to `model.py`: it saves the key and value tensors from each forward pass and passes only the single new token on each subsequent decode step, rather than the full growing sequence. `server.py` and `benchmark.py` are untouched. TPOT drops significantly; TTFT stays the same. The benchmark will show the difference against the same 20 ShareGPT conversations, same seed, same hardware.
