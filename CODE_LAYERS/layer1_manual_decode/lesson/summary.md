# Layer 1 — Summary

Layer 1 does not make inference faster. It makes it visible. The same computation as Layer 0 — the same model, the same number of forward passes, the same throughput — but now every step of the autoregressive loop is written in our code and open to inspection. This is the foundation that makes all future optimisations possible.

---

## From `model.generate()` to the Decode Loop

In Layer 0, text generation was a single call:

```python
output = model.generate(input_ids, max_new_tokens=64)
```

That one line hides everything: the forward passes, the logit extraction, the sampling, the EOS check, the sequence growth. In Layer 1 we replace it with the loop that `model.generate()` was running internally:

```python
ids = input_ids                 # prompt tokens, shape [1, prompt_len]
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

This is the entirety of autoregressive generation. Every language model, no matter how large, runs some version of this loop: call the model, extract logits from the last position, pick the next token, append it, repeat until EOS or the token budget is exhausted. Each section below explains one part of what this loop is doing.

---

## The Forward Pass

Each iteration starts with a direct call to the model:

```python
out = self.model(input_ids=ids, use_cache=False)
```

This is a single forward pass. `ids` is the full current sequence — the original prompt plus every token generated so far. The model runs it through all of its layers and returns a result object. The field we use is `out.logits`, a tensor of shape `[batch_size, sequence_length, vocab_size]`. For Qwen3, `vocab_size` is 151,936. For a 20-token sequence this is a tensor of shape `[1, 20, 151936]` — one score per vocabulary token, at every position in the sequence.

The model produces scores at every position because that is how it was trained: given "The cat sat", it predicts the token after "The", after "The cat", and after "The cat sat" all in one pass. At inference time only the last position's scores matter — that is the prediction for what should come next given everything so far. The line `out.logits[0, -1, :]` extracts exactly that: batch element 0, last position, all vocabulary scores.

`use_cache=False` prevents HuggingFace from silently building an internal cache. Every step recomputes the full sequence from scratch, making the cost honest and visible. Layer 2 turns this back on.

---

## Causal Masking

The model produces scores at every position simultaneously, which raises a question: when computing position 5's representation, can the model look at position 10? If it could, training would be trivial — predict token 6 while already knowing the answer. The causal mask prevents this.

Before the softmax step inside each attention layer, every score that corresponds to a future position is set to negative infinity:

```
Sequence: [A, B, C, D]

Attention scores — after causal mask applied (future → -∞):
         A     B     C     D
A  [  3.1,  -∞,   -∞,   -∞  ]   A sees only itself
B  [  0.8,  4.3,  -∞,   -∞  ]   B sees A and B
C  [ -0.2,  2.1,  3.7,  -∞  ]   C sees A, B, C
D  [  1.5, -0.8,  2.2,  4.1 ]   D sees everything
```

`exp(-∞) = 0`, so masked positions contribute zero weight in the subsequent weighted sum. Token A's output is based only on itself. Token D can draw on all four positions. Each token attends only to itself and to positions that came before it.

This mask is applied automatically inside the model on every forward pass. Our loop does not manage it. The consequence that matters for later layers is that a token's internal representation — and therefore its key and value vectors — depends only on positions before it. Appending a new token to the sequence cannot change what any prior token produced. This immutability is what makes the KV cache a valid optimisation in Layer 2.

---

## Logits and Sampling

The `[vocab_size]` vector from the last position contains raw, unbounded scores called **logits**. They cannot be interpreted directly as probabilities — a logit of 4.2 does not mean a 42% chance. To turn them into a proper probability distribution, softmax is applied: each value is exponentiated and divided by the sum of all exponentiated values, producing a vector that is always positive and sums to 1.

From that distribution, `sample_next_token` picks a single token:

```python
def sample_next_token(logits, temperature):           # sampling.py
    if temperature == 0.0:
        return int(logits.argmax(dim=-1).item())       # greedy: always highest
    if temperature != 1.0:
        logits = logits / temperature                  # scale before softmax
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())  # random draw
```

At `temperature=0`, this is greedy decoding — the highest-scoring token is always chosen, deterministically. At `temperature=1`, a token is sampled randomly in proportion to its probability — common tokens are more likely but not guaranteed. Values between 0 and 1 make the distribution sharper; values above 1 make it flatter and more exploratory.

Real systems also expose top-k (only sample from the k most likely tokens), top-p / nucleus sampling (only sample from the smallest set of tokens whose cumulative probability exceeds p), and repetition penalty (reduce logits for tokens already in the output). All of these are modifications to the logits or probabilities before the final sample is drawn, slotting in exactly where the temperature scaling step is now.

---

## Prefill and Decode

Every pass through the loop is one of two things, even though the code looks identical for both.

**Step 0 is prefill.** `ids` contains only the prompt tokens. The model processes all of them in parallel — the GPU runs attention across all positions simultaneously, model weights are loaded once and applied across all positions in a single pass. For a 400-token prompt, all 400 tokens are processed in one forward call. This is the transformer's core parallelism: unlike a recurrent network, it does not process tokens one at a time.

**Steps 1 onward are decode.** `ids` now contains the prompt plus all tokens generated so far. Each decode step produces exactly one new token, but the model must still load all its weights from memory — 1.2 GB for Qwen3-0.6B — to produce that single output. Most GPU compute sits idle waiting on memory bandwidth rather than doing arithmetic. Decode is memory-bandwidth bound.

In Layer 1 decode is additionally expensive because the full growing sequence is re-processed from scratch on every step. A prompt of length `L` generating `T` tokens means forward passes over sequences of length `L`, `L+1`, `L+2`, ..., `L+T`. The redundant recomputation of all prior token representations on every step is what Layer 2 eliminates with the KV cache.

---

## TTFT and TPOT

`step_times` collects the wall time of each iteration. Because prefill and decode have different costs and respond to different optimisations, the loop splits them into two metrics:

```python
ttft_ms = step_times[0] * 1000          # step 0: prefill + first sample
tpot_ms = mean(step_times[1:]) * 1000   # steps 1+: pure decode average
```

**TTFT — Time To First Token** is `step_times[0]`. It captures how long a user waits before anything appears. It is dominated by prompt length — a 500-token prompt takes significantly longer to prefill than a 20-token prompt, because every token attends to every other token and the attention computation is quadratic in sequence length.

**TPOT — Time Per Output Token** is the average of `step_times[1:]`. It captures the steady-state speed of generation. A TPOT of 20 ms corresponds to 50 tokens per second, or roughly 10–12 words per second. In Layer 1, TPOT is not constant — it creeps upward as the sequence grows, because each decode step processes one more token than the previous one.

Tracking them separately is what makes optimisations interpretable. Adding a KV cache (Layer 2) dramatically reduces TPOT while leaving TTFT roughly unchanged. Batching improves overall throughput but may increase TPOT for individual requests. Without this split, a single latency number would obscure which phase a change is actually affecting.

---

## What Comes Next

Layer 2 makes one targeted change to `model.py`: it saves the key and value tensors from each forward pass and on every subsequent decode step passes only the single new token rather than the full growing sequence. The model reads the cached tensors instead of recomputing them. `server.py` and `benchmark.py` are untouched. The API is unchanged. The benchmark will run the same 20 ShareGPT conversations, same seed, same hardware — and TPOT will drop substantially while TTFT stays the same.
