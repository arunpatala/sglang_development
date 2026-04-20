# Speculative Decoding: A Complete Guide to Making LLMs Faster Without Sacrificing Quality

*For engineers and practitioners who want to understand not just what speculative decoding does, but exactly how it works, why it's safe, and how to configure it. Assumes you know what LLMs do — no prior reading required.*

---

You've probably noticed the pattern: you upgrade to a better, larger AI model and the answers get noticeably smarter — but the responses also get noticeably slower. You add more powerful GPUs and the situation barely improves. This is confusing because we're told that modern GPUs are incredibly fast.

Here is the thing: they are incredibly fast at arithmetic. But LLM inference isn't bottlenecked by arithmetic. It's bottlenecked by *memory*. To understand why speculative decoding works — and why it works so well — you first need to understand exactly what the GPU is spending most of its time doing. Spoiler: it's waiting, not computing.

---

## Part 1: The Problem — Why LLMs Are Slow Even on Fast Hardware

### What happens every time an LLM generates a single word

When an LLM generates a token (a word, or part of a word), here is what physically happens inside the machine:

1. The GPU reads the model's weights — all of them — from GPU memory (HBM) into the compute cores.
2. The compute cores multiply the input token's representation against those weights through each transformer layer.
3. The output is a probability distribution over the vocabulary.
4. A token is sampled from that distribution.
5. That token is appended to the input sequence.
6. Steps 1–5 repeat for the next token.

Step 1 is the problem. For a 70B parameter model stored in FP16 (2 bytes per parameter), the model weighs:

```
70,000,000,000 parameters × 2 bytes = 140 GB
```

Every single token generation requires reading 140 GB from memory. That's not 140 GB once at startup — that's 140 GB **per token**.

### Why this is so slow: the arithmetic intensity problem

A GPU like an NVIDIA A100 has:
- **Peak arithmetic throughput:** 312 TFLOPS (312 trillion operations per second)
- **Peak memory bandwidth:** 2 TB/s (2 trillion bytes per second)

Let's compute what actually happens during one token generation:

```python
model_params     = 70e9          # 70 billion parameters
bytes_per_param  = 2             # FP16
model_size_bytes = 140e9         # 140 GB

memory_bandwidth = 2e12          # 2 TB/s
compute_tflops   = 312e12        # 312 TFLOPS

# Time to load the model weights from memory
load_time = model_size_bytes / memory_bandwidth
# = 140e9 / 2e12 = 0.070 seconds = 70 milliseconds

# Time to actually compute with those weights
flops_per_token = 2 * model_params   # rough estimate for one forward pass
compute_time = flops_per_token / compute_tflops
# = 140e12 / 312e12 = 0.00045 seconds = 0.45 milliseconds
```

The result:

```
Memory load time per token:    70.0 ms
Compute time per token:         0.449 ms
Ratio:                         156× slower to load than to compute

The GPU spends 70 ms loading weights and 0.45 ms computing.
It is idle — literally waiting — for 96% of each token generation.
```

Adding more compute cores does nothing when the bottleneck is loading data from memory. This is called being **memory-bandwidth-bound**. The arithmetic units are starved for data, not for work.

### Why you can't generate multiple tokens at once

Here is the next problem: tokens must be generated **one at a time, in strict sequence**.

This is because of how transformers work. To predict token 5, the model needs to have already seen tokens 1 through 4. Token 5 becomes part of the input context for predicting token 6. There is no way to know token 6 before you know token 5. The dependency is hard.

So for a response that is 200 tokens long, the large model must run 200 sequential forward passes. At 70 ms each on our A100, that's 14 seconds — for a single response. And the GPU is idle 96% of that time.

### The asymmetry that makes the solution possible

Here is the key observation that unlocks everything:

> **Generating N tokens sequentially requires N forward passes through the model. But verifying whether N tokens are correct requires only ONE forward pass.**

Why? Because a transformer can process an entire sequence in parallel. When you feed the model a sequence of tokens and ask "what comes after each position?", the causal attention mask handles the ordering — each position only attends to what came before it — but all positions are processed simultaneously in one pass.

This means: if we could *guess* the next several tokens, we could hand the entire guess to the large model at once and get back a verdict on all of them in a single pass. The cost of verifying 8 guesses is barely more than the cost of verifying 1.

That is the foundation of speculative decoding.

---

## Part 2: The Core Idea — Two Models Working Together

### Why a small model can help

Before explaining the mechanism, there's one more important observation: **not all tokens are equally difficult to predict**.

Consider this sentence:

> "The capital of France is Paris."

Predicting "Paris" after "The capital of France is" is easy. Almost any model will get this right. But predicting the next token after "The transformer architecture uses a mechanism called" requires real understanding — the model needs to know that "attention" is the right continuation, not "recurrence" or "convolution."

Large models — 70B, 100B parameters — are primarily better at the **hard cases**. For easy, predictable tokens, a much smaller model (say, 7B parameters) produces exactly the same answer.

This means: if we can have a small model handle the easy guesses, and only call on the large model to verify (and correct the hard ones), we save enormous amounts of the large model's expensive time.

### The junior and senior developer analogy

The clearest mental model: two developers working together.

A **junior developer** types quickly. They draft a block of code in a few minutes — fast, but not always perfect. Sometimes they produce exactly the right code; sometimes they get something slightly wrong.

A **senior developer** is slower but authoritative. They always produce exactly the right answer. In a standard workflow, the senior writes every single line themselves, one at a time.

Speculative decoding changes the workflow: the junior drafts a block of code quickly. The senior reviews the block, line by line:
- "Would I have written exactly this?" — if yes, accept it and move on.
- "No, this line is wrong" — correct it and stop reviewing the rest of the block (everything after the first mistake is thrown out).

The result: the senior's final output is identical to what they would have written themselves. But they reviewed a block at once instead of writing every line from scratch, so they finish much faster.

In our case:
- The **junior developer** = a small, fast **draft model** (e.g., Llama-3.2-1B)
- The **senior developer** = a large, accurate **target model** (e.g., Llama-3.1-8B or Llama-3.3-70B)
- The "block of code" = a sequence of K candidate tokens the draft model generates
- The "line-by-line review" = the target model's single forward pass that verifies all K tokens at once

The user always receives the target model's authoritative output. The draft model's suggestions are never shown directly — they're internal candidates that may or may not be accepted.

---

## Part 3: How It Actually Works — The Four-Phase Loop

Every speculative decoding generation cycle repeats four phases. Let's walk through each carefully.

### A quick baseline comparison

In standard (baseline) decoding, generating 200 tokens looks like this:

```
For each of 200 tokens:
    1. Load 140 GB of target model weights from memory (70 ms)
    2. Compute one forward pass (0.45 ms)
    3. Get probability distribution, sample one token
    4. Append token to sequence

Total: 200 passes × ~70 ms = ~14 seconds
```

In speculative decoding, the same 200 tokens might require only 40–60 target model passes — because each pass commits multiple tokens. Let's see how.

---

### Phase 1 — Draft: the small model makes its guesses

The small (draft) model takes the current sequence and runs forward K times in quick succession, generating K candidate tokens.

Why is K typically 4–8? This is a tunable parameter. More candidates means more potential savings if they're accepted, but also more wasted work if they're rejected. We'll return to the tradeoffs in the metrics section.

Because the draft model is 10–100× smaller than the target, generating K tokens costs only a fraction of one target forward pass. For example:

- Target model (70B): ~70 ms per token
- Draft model (7B): ~7 ms per token
- Generating 5 draft tokens: 5 × 7 ms = 35 ms

That 35 ms of draft generation will potentially get us 5 tokens verified in a single target pass — instead of needing 5 × 70 ms = 350 ms of target generation.

The draft model does this *without knowing* whether its guesses will be accepted. It just makes its best predictions as fast as it can.

---

### Phase 2 — Verify: one pass checks everything

Now the target model runs — but instead of running on just the current input, it runs on the **current input plus all K draft tokens appended to it**.

For example, if the current sequence is:

```
"The capital of France is"
```

And the draft model proposed:

```
"Paris and it is"
```

The target model receives:

```
"The capital of France is Paris and it is"
```

...and runs one single forward pass over the entire thing.

Here is what happens inside that forward pass: the transformer's **causal attention mask** ensures that each position only attends to the tokens that came before it. This means the model computes, at each position, the probability distribution for what comes next — *as if* it had generated up to that point sequentially.

In other words, in a single forward pass, the target model produces:
- "What would I predict after 'The capital of France is'?" → probability distribution over vocabulary
- "What would I predict after 'The capital of France is Paris'?" → another distribution
- "What would I predict after 'The capital of France is Paris and'?" → another distribution
- ...and so on for each draft token position

All of these predictions are computed **simultaneously**, in one pass. The code for this looks like:

```python
def parallel_verification(target_model, input_ids, draft_tokens):
    # Concatenate the original input with the draft tokens
    full_sequence = torch.cat([input_ids, draft_tokens], dim=-1)

    # One forward pass over the full sequence
    # The causal mask handles ordering internally
    logits = target_model(full_sequence).logits

    num_input = input_ids.shape[-1]
    num_draft  = draft_tokens.shape[-1]

    # Extract the logits at each position just BEFORE each draft token
    # (position i gives us "what would come after the first i tokens")
    verification_logits = logits[:, num_input - 1 : num_input + num_draft - 1, :]
    probs = F.softmax(verification_logits, dim=-1)

    # For each draft position, get the target's probability for the draft's choice
    draft_probs = [
        probs[0, i, draft_tokens[0, i].item()].item()
        for i in range(num_draft)
    ]
    return draft_probs, probs
```

The critical efficiency: **one target forward pass costs about the same as generating one token, but it gives us information about all K draft token positions**. The causal mask is what makes this possible — the same architectural feature that lets transformers train efficiently also lets them verify multiple speculative tokens in parallel.

---

### Phase 3 — Accept or reject: comparing predictions token by token

After the target model's forward pass, we have two sets of predictions for each draft token position:

1. **What the draft model predicted** (and what it chose as its draft token)
2. **What the target model predicts** at that same position

Now we compare them, from left to right, stopping at the first disagreement.

The rule for each position:

- If the target model **agrees** with or is **more confident** about the draft token than the draft model was: **accept** the token, move to the next position.
- If the target model **disagrees** (assigns lower probability to the draft's choice than the draft did): **reject** this token and all subsequent draft tokens.

Formally:
```
P(accept token at position i) = min(1, p_target(token_i) / p_draft(token_i))
```

Let's make this concrete. Suppose the draft proposed three tokens and the target compares:

| Position | Draft token | p_draft | p_target | Decision |
|----------|-------------|---------|----------|----------|
| 1 | "Paris" | 0.60 | 0.80 | ACCEPT — target is even more confident |
| 2 | "and" | 0.70 | 0.75 | ACCEPT — target agrees |
| 3 | "it" | 0.50 | 0.10 | REJECT — target is much less confident; draft overestimated |

At position 3, the draft was confident ("it" with 50% probability) but the target disagrees strongly (only 10% probability). We reject it — and discard everything after it.

---

### Phase 4 — Correction and restart: always moving forward

When a rejection happens, we don't just throw everything away and start over. Two things happen:

**First:** The target model commits its *own* prediction at the rejection point. Whatever the target would have generated at position 3 — not "it", but perhaps "a" — gets committed as the accepted token at that position.

**Second:** The draft model restarts from this new accepted position, generating a new batch of K candidates.

A bonus token is always produced. If all K draft tokens were accepted, the target model also generates a K+1th token (its own prediction for what comes after the last accepted token). So the best-case outcome per cycle is K+1 tokens from one target model pass.

This means **the cycle always makes forward progress**:

**Worst case:** Every single draft token is wrong. The target rejects all K of them. We get 1 token (the target's correction at position 1). This is exactly the same as baseline — no regression, just the added cost of the cheap draft generation.

**Best case:** All K draft tokens are accepted. We get K+1 tokens from a single target forward pass. If K=5, that's 6 tokens for the cost of 1 target pass instead of 6 target passes.

**Typical case:** Some prefix of the draft tokens are accepted. With a well-matched draft model, you typically see 2–3× overall speedup.

> *"Even if the small model is wrong 60% of the time, you can still see significant speedups because of the parallel verification."* — Pierce Freeman

Now, one thing is still unclear: what exactly does "reject probabilistically" mean, and why is this safe? That leads to the most important property of speculative decoding.

---

## Part 4: The Guarantee — Why the Output Is Exactly Right

### The concern

It might seem like speculative decoding introduces some approximation. You're using a smaller, less accurate model to generate tokens — surely some of those accepted tokens are "wrong" in some sense?

The answer is no — and the mathematical argument for why is worth understanding. It's the reason speculative decoding can be deployed in production without any quality testing or regression analysis. The output is not *approximately* correct — it is *exactly* correct by construction.

### The acceptance rule, explained

Let's look at the acceptance rule again:

```
P(accept) = min(1, p_target / p_draft)
```

This breaks down into two cases:

**Case 1: The target is at least as confident as the draft** (`p_target ≥ p_draft`)

In this case, `p_target / p_draft ≥ 1`, so `min(1, p_target/p_draft) = 1`. The token is **always accepted**.

This makes intuitive sense: if the target model is equally or more confident about this token choice than the draft model was, then the draft model didn't overstate its case. Accepting the token is safe.

**Case 2: The draft was overconfident** (`p_draft > p_target`)

In this case, `p_target / p_draft < 1`. We accept the token with probability `p_target / p_draft`.

For example: if the draft assigned 20% probability to a token but the target only assigns 5%, we accept it only 5/20 = 25% of the time. Most of the time, we reject it.

Here is the concrete interpretation of different acceptance scenarios:

| Scenario | p_target | p_draft | P(accept) | Plain English |
|----------|----------|---------|-----------|---------------|
| Target strongly agrees | 0.25 | 0.10 | 1.00 | Always accept — target is even more sure than the draft |
| Slight draft overconfidence | 0.15 | 0.18 | 0.83 | Accept 83% of the time |
| Significant overconfidence | 0.05 | 0.20 | 0.25 | Reject 75% of the time |
| Extreme overconfidence | 0.01 | 0.20 | 0.05 | Almost always reject |

### What happens on rejection: the residual distribution

When a token is rejected, we need to pick a replacement. You might think: "just use whatever the target model would have generated." But this would be wrong — and the reason is subtle.

If we always resample from the full target distribution `p_target` on rejection, we'd be double-counting. Think about it this way: the draft already had a chance to propose each token. If the draft proposed token X, we either accepted it (if the target agreed) or rejected it and sampled from `p_target`. But `p_target` includes token X again! So token X could get "two chances" to be selected, which would distort the final distribution.

The correct fix is to sample from the **residual distribution**:

```
p_residual(x) = max(0, p_target(x) − p_draft(x)),  normalized to sum to 1
```

This is `p_target` with the part that the draft already "used up" subtracted out, then renormalized. It represents only the probability mass that the draft *didn't adequately represent*.

### Why this guarantees exact output

Consider any token X. There are two ways it can end up as the output:

**Path A:** The draft proposes X, and we accept it.
- This happens with probability: `p_draft(X) × min(1, p_target(X)/p_draft(X))`
- Which simplifies to: `min(p_draft(X), p_target(X))`

**Path B:** The draft proposes some other token Y, we reject Y, and then we sample X from the residual.
- The probability of this path producing X works out to: `max(0, p_target(X) − p_draft(X))`

**Total probability of outputting X** = Path A + Path B:
```
min(p_draft(X), p_target(X)) + max(0, p_target(X) − p_draft(X))
= p_target(X)
```

In both cases, X is emitted with exactly the target's probability. The acceptance rule and correction distribution together perfectly recover the target distribution — no approximation, no bias, no tradeoff.

> *"The algorithm speeds up generation from autoregressive models by computing several tokens in parallel, without affecting output quality; in fact, the method guarantees an identical output distribution."* — Leviathan, Kalman, Matias (Google Research, original inventors)

This guarantee is why speculative decoding can be deployed without any evaluation: you don't need to A/B test it or run regression benchmarks on output quality. The math guarantees it. You only need to measure **speed**.

Three numbers govern how much speed you actually get.

---

## Part 5: The Three Metrics That Govern Speedup

Once you understand the mechanism, the performance of speculative decoding reduces to three numbers. These are the quantities you'd monitor in a production dashboard.

### Metric 1: Acceptance rate (α)

The acceptance rate is the fraction of draft tokens that the target model accepts:

```
α = (number of accepted draft tokens) / (total draft tokens proposed)
```

This is the single most important number. Everything else follows from it.

A concrete example: if you propose 5 draft tokens per round and on average 3 are accepted, your acceptance rate is α = 3/5 = 0.6.

How to interpret α:

| Acceptance rate α | Interpretation | What to do |
|-------------------|---------------|------------|
| α ≥ 0.7 | Excellent — draft is well-matched to target | Increase draft length, enjoy the speedup |
| 0.5 ≤ α < 0.7 | Good — worthwhile to deploy | Monitor, tune draft length, consider fine-tuning |
| α < 0.5 | Poor — gains are small | Try a different draft model or fine-tune it |

If your acceptance rate is below 0.5, the overhead of running the draft model is starting to outweigh the savings from parallel verification.

### Metric 2: Speculative token count (γ)

The speculative token count is how many draft tokens you generate per round. This is a **configurable parameter** you set at inference time — it doesn't require retraining anything.

The relationship with α:
- If your acceptance rate is high (α ≥ 0.8), you can afford to propose more tokens per round (γ = 7–10). Most of them will be accepted, so longer drafts help.
- If your acceptance rate is lower (α ≈ 0.5–0.6), keep γ smaller (γ = 3–5). Proposing 10 tokens when only 2–3 will be accepted wastes draft model computation on rejected tokens.

There's no universal optimal γ — it depends on your acceptance rate, your draft model speed, and your target model speed. We'll revisit this when we discuss dynamic scheduling.

### Metric 3: Acceptance length (τ) — and the speedup formula

The acceptance length is the average number of tokens committed per speculative decoding round. This is what directly determines speedup.

The theoretical formula, derived from the geometric series:

```
τ = (1 − α^(γ+1)) / (1 − α)
```

This formula captures the compounding effect of acceptance probability. The intuition: at position 1, the probability of acceptance is α. At position 2, it's α × α (both positions must be accepted). At position 3, it's α³, and so on. The formula sums this geometric series.

Let's make this concrete. Suppose you have:
- Target model: 100 ms per forward pass
- Draft model: 15 ms per token (so 5 draft tokens = 75 ms)
- Draft length γ = 5

The speedup at different acceptance rates:

| Acceptance Rate | Tokens per Round | Round Cost | Effective ms/token | Speedup vs Baseline |
|----------------|-----------------|------------|-------------------|---------------------|
| 0.5 | 1.97 | 175 ms | 88.8 ms | 1.12× |
| 0.6 | 2.38 | 175 ms | 73.5 ms | 1.36× |
| 0.7 | 2.94 | 175 ms | 59.5 ms | 1.68× |
| 0.8 | 3.69 | 175 ms | 47.4 ms | 2.11× |
| 0.9 | 4.69 | 175 ms | 37.3 ms | 2.68× |

*(Round cost = 5 × 15ms draft + 100ms target = 175ms. Baseline = 100ms/token.)*

Two things stand out:
1. The speedup is real even at modest acceptance rates (0.6–0.7 gives 1.4–1.7×).
2. The speedup grows rapidly as acceptance rate improves. Going from 0.5 to 0.9 acceptance rate takes you from 1.12× to 2.68× — a massive difference.

**The single biggest lever for improving speculative decoding performance is improving α.** That means choosing the right draft model.

---

## Part 6: Choosing the Right Draft Model

The draft model is the most important configuration decision. Here are the four requirements, in order of strictness:

### Requirement 1: Same tokenizer (non-negotiable)

The draft model and target model must use **exactly the same tokenizer** — the same vocabulary, the same mapping from text to token IDs.

Why? Because in the accept/reject step, we compare probabilities for specific token IDs. If the draft model uses a different tokenizer, "Paris" might be token ID 4521 in the draft's vocabulary and token ID 7832 in the target's vocabulary. The comparison would be meaningless — we'd be asking "does the target also assign high probability to token 7832?" when the draft actually predicted "Paris," not token 7832.

If tokenizers don't match, speculative decoding cannot work at all. This is not something you can tune around — it's a hard constraint.

### Requirement 2: 10–20× fewer parameters

The draft model should be substantially smaller than the target:
- 70B target → 7B draft is a natural pairing (10× smaller)
- 8B target → 1B draft works well (8× smaller)
- 70B target → 1B draft might be too small (acceptance rate drops)

Why does the size ratio matter? Two reasons:

1. **Speed:** The draft model must generate K tokens faster than the target would generate one. If the draft is only 2× smaller, you're barely saving anything.

2. **Acceptance rate:** A model that's 100× smaller will have lower acceptance rates than one that's 10× smaller. The sweet spot is 10–20× — small enough to be fast, large enough to be accurate.

Common production pairings:
- Llama-3.2-1B as draft for Llama-3.1-8B
- Llama-3.1-8B as draft for Llama-3.3-70B
- Both from the same model family — this matters for the next requirement.

### Requirement 3: Similar training distribution

The draft and target models should have been trained on similar data and with similar objectives. If the target was instruction-tuned for helpful AI assistance and the draft was trained on raw code, the draft's predictions will frequently not match what the target would generate in a chat context.

Same model family (e.g., all Llama models, all Qwen models) generally means good alignment. Cross-family pairings need explicit benchmarking — run your workload and measure the acceptance rate before committing.

### Requirement 4: Fine-tune the draft for your domain (when possible)

For production deployments, fine-tuning the draft model on your specific workload can significantly boost acceptance rates.

The approach:
1. Collect outputs from the target model on a representative sample of your production inputs.
2. Fine-tune the draft model to predict those same outputs.
3. The draft model learns to approximate the target's behavior on your specific distribution.

This is the difference between a generic draft and a production-grade draft. Generic: α ≈ 0.55–0.65. Domain-fine-tuned: α ≈ 0.70–0.85.

**The bottom line:** A good draft model is necessary but not sufficient. The workload also has to cooperate.

---

## Part 7: When Speculative Decoding Helps and When It Doesn't

### Workloads where it works well

**Code generation** is the best case. Code is highly repetitive — variable names, keywords, standard patterns. A small model trained on code predicts these correctly at very high rates. You'll often see α > 0.80 for code generation tasks, translating to 2× or more speedup.

**Translation, summarization, and transcription** are also excellent. The output is heavily constrained by the input — there are only so many sensible ways to translate a sentence, and a small model with access to the source text will often predict the same tokens as a large model.

**Greedy or low-temperature decoding** (temperature ≤ 0.3) is where speculative decoding shines. When the model is making confident, focused predictions, there's high agreement between draft and target.

**Large size ratios** — 70B target with a 7B draft gains more than an 8B target with a 1B draft. The large model's forward pass is more expensive, so each accepted draft token saves more time.

### Workloads where it struggles

**High-temperature creative writing** (temperature > 0.8) is the hard case. At high temperature, the model is sampling from a flat, diverse distribution. A small draft model can't reliably predict what the large model's random sample will be. Acceptance rates drop significantly.

**Mismatched draft models** — different training distributions, different tokenizers (which rules out speculative decoding entirely), or very different model families.

**Very small target models** that already run fast and fit comfortably in GPU memory — the overhead of running a draft model may not be worth it.

### The surprising lower bound: even a terrible draft helps

Here is the most counterintuitive result in the speculative decoding literature: even a **bigram model** — a simple statistical model that predicts the next token based only on the current token, no GPU required, negligible cost — as the draft achieves a **1.25× speedup**.

With only ~20% acceptance rate and γ=3, a model that knows almost nothing about language still provides measurable speedup. Why? Because the parallel verification step amortizes the target model's memory load across however many tokens it accepts. Even 1 accepted token out of 3 is better than the target generating that token alone.

The implication: the floor of speculative decoding is never zero. Any draft model that produces at least some correct guesses will beat baseline. The question is just how much.

---

## Part 8: Static vs Dynamic — Tuning γ Per Round

### The problem with a fixed γ

So far we've treated γ (the number of draft tokens per round) as a constant you set once. In practice, the optimal γ varies **dramatically** from one round to the next within the same generation.

Here's why: some rounds the context is highly predictable (you're in the middle of a code block with a clear pattern) and many draft tokens will be accepted. Other rounds the context is ambiguous (the model is about to make a complex reasoning step) and most draft tokens will be rejected.

If you set γ=5 for everything, you're proposing too few tokens on the easy rounds and too many on the hard rounds.

An **oracle** — a hypothetical agent that knows in advance which draft tokens will be accepted — can compute the perfect γ for each round. Measured on real workloads, oracle scheduling uses:
- **29% fewer target model forward passes**
- **33% fewer draft model forward passes**

...compared to static γ=5. That's a substantial gap left on the table by the fixed-γ approach.

### Three scheduling approaches

**Constant scheduling:** γ is fixed for the entire generation. Simple, predictable, suboptimal.

**Heuristic scheduling:** After each round, adjust γ based on what happened:
- If all K draft tokens were accepted → increase γ next round (the context is predictable, go deeper)
- If any draft tokens were rejected → decrease γ next round (be more conservative)

This is reactive — it adjusts based on the previous round, which is a reasonable proxy for the next round's predictability.

**Dynamic scheduling:** Instead of waiting until the end of a round to adjust, check the draft model's confidence *during* the round. After generating each draft token, examine the softmax probability that the draft model assigned to its own choice. If the confidence drops below a threshold — the draft model itself is unsure — stop drafting early and hand off to the target now, even if you haven't reached γ tokens yet.

This is more responsive because you're stopping as soon as you detect a likely rejection, rather than generating and discarding tokens you already know will probably be rejected.

### Using dynamic scheduling

Dynamic scheduling is now the default in HuggingFace Transformers ≥ 4.45.0. You don't need to change any code to get it:

```python
# Dynamic scheduling is automatically used — no code changes required
outputs = model.generate(**inputs, assistant_model=assistant_model)
```

To tune it:

```python
# The confidence threshold below which drafting stops early
# Default of 0.4 works well for most models and tasks
assistant_model.generation_config.assistant_confidence_threshold = 0.4

# Maximum tokens drafted per round (even if confidence stays high)
assistant_model.generation_config.num_assistant_tokens = 20
```

**The key practical result:** Dynamic scheduling is always at least as good as heuristic scheduling, and often dramatically better. In cases where heuristic scheduling actually *caused slowdowns* (certain model pairings where the constant γ was too high), dynamic scheduling recovered positive speedup by halting early when the draft model signaled low confidence.

---

## Part 9: Choosing Your Variant — The Five Techniques

The classic setup — one separate draft model, one target model — is just one of five techniques available in production. Each one answers the same question differently: "how do we generate cheap draft tokens?"

### The five variants side by side

**1. Classic draft model:** A separate smaller model generates the draft tokens. Works with any model pair that shares a tokenizer. Requires extra VRAM for the second model. This is the foundational approach — everything else is a variation on this theme.

**2. N-gram matching:** Instead of a second model, look at the text that's already been generated and find repeating patterns. If "np.mean" has appeared twice already, and we're generating "np.", we can guess "mean" as the next token. Zero VRAM overhead — no second model at all. Works best for code and highly repetitive text. Performs poorly when the text is diverse.

**3. Suffix decoding:** A more sophisticated version of N-gram. Maintains two suffix trees: one for the current request and one shared globally across all requests served by the same instance. Uses these trees to find longer matching patterns and makes more confident predictions. CPU-based, no GPU overhead. Excellent for agentic workflows and code generation where long repeating sequences appear.

**4. MLP Speculator (Medusa-style):** A lightweight multi-layer perceptron with multiple heads, attached directly to the target model. Each head predicts a token at a different future offset (t+1, t+2, t+3, ...). Uses the target model's own context embedding as input — so it has access to the same information the target model has. Requires training this MLP on the target model's outputs. Roughly 1/10th the parameters of a full draft model.

**5. EAGLE / EAGLE-2 / EAGLE-3:** A more powerful version of the MLP Speculator idea. Instead of using just the final embedding, EAGLE attaches a lightweight transformer head that taps into the target model's *internal* feature maps from multiple layers (low, middle, and high). EAGLE-3, the current state of the art, also trains on its own predictions rather than ground truth, fixing a distribution mismatch problem. Requires training but achieves higher acceptance rates than other approaches.

### When to use which

Here is the decision guide based on actual benchmarks on Llama models:

| Your situation | Best technique | Reason |
|---------------|---------------|--------|
| General chat, model ≤ 8B | EAGLE / EAGLE-3 | Learned heads predict conversational text better than pattern-matching |
| Code generation, model ≤ 8B | Suffix Decoding | Code is repetitive; suffix matching exploits this at zero overhead |
| Code or agents, model ≥ 70B | EAGLE-3 | At scale, learned predictions beat pattern-matching even for code |
| Strict VRAM budget | Suffix Decoding | No second model, no training, no extra memory — just turn it on |
| Starting out / prototyping | Suffix Decoding | Zero setup cost; switch to EAGLE-3 later if you need more gains |

> **Practical recommendation:** Start with Suffix Decoding. It requires no training, no extra VRAM, and no changes to your model. You get a meaningful speedup immediately. If your workload is 70B+ models or you want to maximize gains, train an EAGLE-3 head.

### A different approach: one model that drafts itself (LayerSkip)

All five variants above share one property: they use the full target model for verification. LayerSkip takes a different approach — it uses the *same single model* for both drafting and verification, but at different computational depths.

The idea: transformer models have many layers, and intermediate layers already produce useful (if imperfect) representations of the output. LayerSkip modifies the model so it can "exit early" — use only the first E layers to produce a draft token, then continue with the remaining layers to verify.

| Comparison | Classic (two-model) | LayerSkip (self-speculative) |
|------------|---------------------|------------------------------|
| Draft comes from | A separate smaller model | The target model's own early layers |
| Extra VRAM needed | Yes — full second model | No — same model, just stops early |
| Training required | No — use any same-tokenizer pair | Yes — requires LayerSkip fine-tuning |
| Works with off-the-shelf models | Yes | No — only LayerSkip checkpoints |
| Typical speedup | 1.5–2.5× | 1.8–2.0× |

The advantage of LayerSkip: no second model, no extra VRAM, and the draft tokens are produced by the same model that verifies them, so they tend to be better aligned. The disadvantage: it requires retraining the target model with a specific recipe (progressive layer dropout and an early exit loss function at every layer).

Pre-trained LayerSkip checkpoints are available for Llama2 (7B, 13B, 70B) and Llama3 (8B, 1B) from Meta on HuggingFace.

**Regardless of which variant you use, the correctness guarantee is the same.** Every token that makes it into the output was validated by the full target model. The only thing that varies is how the draft candidates are generated.

---

## Part 10: In Production

### It's already running everywhere

Speculative decoding is not an experimental technique you'd adopt cautiously. It is the standard approach to LLM inference at production scale:

- **Google** — deployed in AI Overviews (Google Search) and multiple other Google products. Confirmed by the original inventors.
- **Meta** — production inference systems across Meta's AI products.
- **Anthropic** — deployed in the infrastructure serving Claude.
- **vLLM, SGLang, TGI, TensorRT-LLM, llama.cpp** — all support it natively. It's a first-class feature, not a plugin.

### Why it's safe to deploy

Three properties make speculative decoding unusual among inference optimizations:

**1. Exact quality guarantee — no testing required for quality.**

Most inference optimizations (quantization, pruning, distillation) introduce some accuracy tradeoff. You need to run evaluations to make sure the tradeoff is acceptable. With speculative decoding, the mathematical proof guarantees that output quality is unchanged. You don't need to run A/B tests or LLM evaluation suites to check quality. You only need to measure speed.

**2. No target model changes.**

The target model's weights are completely unchanged. You don't modify it, quantize it, or retrain it. If you already have a serving setup with your target model, adding speculative decoding only adds the draft model (or a draft head, depending on variant). Your target model is not touched.

**3. Graceful degradation.**

The worst case — every single draft token rejected — is mathematically identical to baseline decoding. There is no configuration of speculative decoding that makes outputs *both* slower and worse. If the draft model is terrible, you're no worse than baseline. As the draft model improves, you improve. There's no downside risk.

> *"Same parameters, same hardware, faster models."* — Pierce Freeman

---

## Further Reading

Sections deliberately excluded from this article to keep the flow clean. All are in `L2_blog_omitted.md`.

**§A — The full mathematical proof**
The complete SpecJudge + SpecNormalize decomposition proof (Case 1: x ∈ D; Case 2: x ∉ D), showing step by step why both cases emit exactly `p_target(x)`. Includes the key property: guaranteed single-round resolution, no re-sampling loop needed.

**§B — vLLM benchmark tables in full**
Complete benchmark data: Llama-3.1-8B on L40S and Llama-3.3-70B on H200, across ShareGPT (chat) and SWE-bench (code). All five techniques side by side with tokens/sec and speedup ratios. Includes vLLM CLI configuration snippets for each.

**§C — LayerSkip full treatment**
The training recipe (progressive layer dropout + early exit loss function), the three cache types (shared weights, shared KV cache, KVQ cache), the tradeoff curve for exit layer depth, and the full summarization benchmark comparing LayerSkip against two-model speculative decoding.

**§D — Dynamic scheduling benchmarks**
Full results across six model pairs and three task types, comparing constant vs heuristic vs dynamic scheduling. Includes the DISCO variant (learned classifier instead of confidence threshold, arXiv:2405.04304).

**§E — Extensions: DistillSpec, Online SD, tree attention, Medusa**
DistillSpec (distill target distribution into draft — and the subtle trap of why higher accuracy ≠ higher acceptance rate). Online speculative decoding (continuously fine-tune the draft from live mismatches). Tree attention (verify branching candidate trees in one pass, the mechanism behind EAGLE-2/3). Medusa (multiple prediction heads inside one model, verified via tree attention).

**§F — Worked example: "The quick brown fox"**
Full token-level trace: draft proposes "jumps over the lazy dog", target accepts "jumps over the", rejects "lazy" (correct: "log"), commits "log", restarts. Shows exactly what the correction step looks like in practice.

**§G — Correction distribution: full code and all scenarios**
Python implementation of `accept_probability()` and `correction_distribution()` with the complete scenarios table, plus a note on how the stochastic acceptance rule simplifies to the greedy special case at temperature=0.

---

*Sources: [ML Mastery](https://machinelearningmastery.com/the-machine-learning-practitioners-guide-to-speculative-decoding/) · [Jarvis Labs / vLLM](https://jarvislabs.ai/blog/speculative-decoding-vllm-faster-llm-inference) · [HuggingFace TGI](https://huggingface.co/docs/text-generation-inference/main/en/conceptual/speculation) · [HuggingFace Dynamic Speculation](https://huggingface.co/blog/dynamic_speculation_lookahead) · [LayerSkip](https://huggingface.co/blog/layerskip) · [Adaptive ML](https://www.adaptive-ml.com/post/speculative-decoding-visualized) · [Pierce Freeman](https://pierce.dev/notes/how-speculative-decoding-works/) · [Brenndoerfer](https://mbrenndoerfer.com/writing/speculative-decoding-accelerating-llm-inference) · [Data Processing Club](https://data-processing.club/speculative/)*
