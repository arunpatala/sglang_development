# 04 — Prefill and Decode

## Two Phases, One Loop

Every generation request passes through two distinct phases. In Layer 1 they look identical in code — both are a call to `self.model(input_ids=ids, use_cache=False)` — but they are doing fundamentally different work, they have different costs, and they will respond differently to optimisations in later layers. Naming them now makes the rest of the curriculum much easier to follow.

## The Prefill Phase

The first forward pass is special. It processes the entire prompt — all the tokens from the formatted conversation — in one go:

```python
ids = input_ids          # shape: [1, prompt_len]

with torch.no_grad():
    out = self.model(input_ids=ids, use_cache=False)

next_token_logits = out.logits[0, -1, :]   # scores for what comes after the prompt
```

This is the **prefill**. The model reads the full context — the system instruction, the user's message, all the chat template markers — and produces logits at the last position. That last position's logits tell us what the model thinks should come first in its reply.

The cost of prefill scales with the number of prompt tokens, but crucially, all those tokens are processed **in parallel** in a single forward pass. The GPU sees the entire prompt as a batch of work and computes attention across all positions simultaneously. This is the transformer's core strength: unlike a recurrent network that processes tokens one at a time, the transformer handles a 500-token prompt in roughly the same wall-clock time as a 50-token prompt when the GPU is large enough to hold the whole thing at once. The model's weight matrices are loaded into GPU registers once and applied across all positions in parallel.

For short prompts this is very fast. For long prompts — retrieved documents, long conversation histories, code files passed as context — prefill can still be a significant fraction of total request latency, because the attention computation grows quadratically with sequence length (every token attends to every other token). But the key property is that it happens all at once: prefill is a single, highly parallelisable forward pass.

## The Decode Phase

After prefill, we have the first generated token. We append it to the sequence and call the model again:

```python
ids = torch.cat([ids, next_token_tensor], dim=1)   # prompt + 1 new token

with torch.no_grad():
    out = self.model(input_ids=ids, use_cache=False)

next_token_logits = out.logits[0, -1, :]   # scores for the second generated token
```

This is the **decode**. We repeat it — add one token, run one forward pass, get the next token — until we hit the end-of-sequence token or reach `max_new_tokens`. Each decode step is conditioned on the full sequence so far: the prompt plus every token the model has generated up to this point.

Decode is fundamentally different from prefill in its parallelism characteristics. Each decode step generates exactly one token, which means the GPU is doing a tiny amount of useful work — one token's worth of computation — but it still has to load the entire set of model weights from memory to do it. For Qwen3-0.6B that means moving 1.2 GB of weight data through the GPU's memory system to produce a single integer. The GPU's thousands of compute cores are largely idle, waiting on memory bandwidth rather than doing arithmetic. This is why decode is said to be **memory-bandwidth bound** rather than **compute bound** like prefill.

In Layer 1, things are worse still because we pass the entire growing sequence on every decode step, so the model also has to recompute attention over all prior tokens from scratch. A prompt of length `L` generating `T` tokens means forward passes over sequences of length `L`, `L+1`, `L+2`, ..., `L+T`. The total attention computation grows quadratically with output length. This redundant recomputation is what Layer 2 eliminates.

## Why They Feel Different Even Now

Even in Layer 1, where both phases use the same code path, step timing reveals the difference. The first step takes noticeably longer than subsequent ones when the prompt is long:

```python
step_times: list[float] = []

for step in range(max_new_tokens):
    t_step = time.perf_counter()

    with torch.no_grad():
        out = self.model(input_ids=ids, use_cache=False)

    next_token_logits = out.logits[0, -1, :]
    next_token_id = sample_next_token(next_token_logits, temperature)

    step_times.append(time.perf_counter() - t_step)
    # step_times[0]  ← prefill cost (processes prompt_len tokens)
    # step_times[1]  ← decode step 1 (processes prompt_len + 1 tokens)
    # step_times[2]  ← decode step 2 (processes prompt_len + 2 tokens)
    # ...
```

`step_times[0]` is the prefill cost — the model processes the full prompt for the first time. `step_times[1]` processes one extra token. Each subsequent step processes one more. Without a cache, TPOT (Time Per Output Token) slowly increases as the sequence grows, because each decode step has slightly more work to do than the one before.

This per-step timing data is what the layer uses to compute TTFT and TPOT, which are covered in detail in section 06.

## The Conceptual Boundary

The reason to care about the prefill/decode distinction now — even though Layer 1 does not treat them differently — is that Layer 2 will. When we add a KV cache, the two phases genuinely diverge:

- Prefill processes all prompt tokens and **saves** the computed key and value tensors.
- Each decode step processes only the **one new token** and reads back the saved tensors rather than recomputing them.

Once that change is in place, prefill cost stays the same but decode cost drops dramatically — each decode step becomes O(1) in sequence length instead of O(n). Understanding that these phases exist as distinct concepts is what makes that change make sense when you see it.
