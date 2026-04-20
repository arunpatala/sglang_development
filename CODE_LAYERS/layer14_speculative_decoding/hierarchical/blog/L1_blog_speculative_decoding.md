# How AI Models Get Faster Without Getting Worse: Speculative Decoding Explained

*For engineers, product builders, and anyone who has wondered why "smarter AI" often means "slower AI".*

---

Upgrading to a larger model improves quality and increases latency. On the fastest hardware available, this tradeoff holds. The reason isn't compute — modern GPUs have more arithmetic throughput than LLM inference can use. The reason is memory. Every token a large model generates requires reading its entire weight matrix from GPU memory. More parameters means more data to read, more time per token, slower responses. Better hardware helps at the margin but doesn't change the fundamental constraint.

Speculative decoding changes the constraint rather than the hardware.

---

## Why your GPU is mostly idle during LLM inference

Every time a large language model generates a word, it reads **all of its parameters from memory**. Not the relevant section — all of them. For a large model, that's hundreds of gigabytes read for every single token produced.

It compounds further: each token depends on every token before it, so generation is strictly sequential. Token two cannot start until token one is done.

Modern GPUs can perform hundreds of trillions of arithmetic operations per second. Memory bandwidth is around trillions of bytes per second — two orders of magnitude lower. During inference the transformer performs only a handful of operations per byte it reads, which means the arithmetic units are sitting mostly idle, waiting for memory to deliver the next round of weights.

This is the **memory-bound** problem. You can add more compute, but if memory reads are the ceiling, most of that compute is wasted.

Two observations break this open.

**First:** not every token is equally hard to predict. Consider:

> *"What is the square root of 7? The square root of 7 is 2.646."*

The second "7" is trivial — it copies from the question. "2.646" requires actual computation. A small, fast model handles the easy cases fine. It only falls short on the hard ones, and large models are mainly better precisely at the hard ones.

**Second:** because memory is the bottleneck, idle compute is available for free. Running additional arithmetic doesn't slow down memory reads. Extra work can happen in parallel, at no cost to the bottleneck.

These two observations together point to a solution: use a small, fast model to do the easy work in parallel, and let the large model verify and correct. The large model's quality is preserved. The small model's speed is harvested.

---

## The idea: a junior and a senior working together

The clearest way to hold this in your head: two developers working together.

A **junior developer** types quickly. They draft code suggestions rapidly — not always perfect, but fast and usually reasonable for common patterns. A **senior developer** knows exactly what should be written. They're slower but authoritative.

Baseline LLM inference is like the senior writing every line themselves — one token at a time, full weight read, one by one.

Speculative decoding changes the workflow. The junior drafts a block of several lines quickly. The senior reviews: for each line, "would I have written exactly this?" If yes, accepted, move on. If no, fix it from the first mistake and continue from there.

In practice, two models play the two roles:
- A **smaller, faster model** proposes several candidate tokens ahead.
- A **larger, more capable model** verifies them in a single forward pass and accepts what matches.

The junior's output is **never shown to the user directly**. The user always receives the senior's final authoritative version. But the senior's time — the expensive resource — is used far more efficiently.

---

## How it actually works

To understand what's happening, it helps to first look at the problem with standard generation.

In baseline autoregressive decoding, the large model generates one token, appends it to the sequence, then generates the next — one pass, one token, repeat. If the response is 200 tokens long, the large model runs 200 times. Each run reads all its weights from memory. There is no way to parallelise this loop because each token depends on the one before it.

Speculative decoding restructures this loop into three repeating steps:

---

### Step 1 — Draft

The small model takes the current sequence and runs forward several times in quick succession, generating a block of candidate tokens — typically 3 to 12. Because the small model is much cheaper per pass (10–100× smaller), this entire block costs roughly the same as a fraction of one large-model pass.

The small model does not know whether its candidates will be accepted. It is making its best prediction, fast.

> Think of this as the junior developer writing a block of code before the senior has reviewed it. The junior doesn't wait for approval on line 1 before writing line 2.

---

### Step 2 — Verify

The large model now takes the original sequence *and all the draft tokens appended to it* and runs a single forward pass over the whole thing simultaneously.

This is the key efficiency: the large model processes the draft block in **one pass**, not one pass per token. Because the transformer architecture evaluates all positions in parallel, the cost of verifying 8 draft tokens is barely more than verifying 1. The KV cache holds the already-computed context from before the draft, so only the draft positions incur fresh computation.

One expensive large-model operation checks everything the small model proposed.

> The senior developer reads the entire drafted block at once, rather than waiting to see each line one at a time.

---

### Step 3 — Accept or reject

The large model now has its own probability distribution at each draft position — what it would have predicted at each step if it had been generating normally. It compares this against what the small model actually proposed.

The rule at each position:
- If the large model's prediction **agrees** with the draft token (or assigns it equal or higher probability than the small model did): **accept**. Move to the next position.
- If the large model's prediction **disagrees**: **reject** this token and discard everything after it. Use the large model's own prediction at this position instead, and restart the cycle.

When a rejection occurs, the large model's own token is committed at that point — so the cycle always makes forward progress. Even a complete rejection (every draft token wrong) still produces one new token and is no worse than baseline.

**Worst case:** all draft tokens rejected → one large-model token committed. Identical to baseline.

**Best case:** all draft tokens accepted → 3–12 tokens committed in a single large-model pass.

---

### The guarantee that makes this work

The acceptance rule is not editorial. The large model is not judging whether the draft is "good" — it is checking whether the draft token is exactly what it would have produced in normal generation.

This makes the output **mathematically identical** to what you would get from running the large model alone, at every token position:

> *"The algorithm speeds up generation from autoregressive models by computing several tokens in parallel, without affecting output quality; in fact, the method guarantees an identical output distribution."* — Leviathan, Kalman, Matias (Google Research, original authors)

Nothing is approximated. Nothing is skipped. The speedup comes entirely from doing the large model's verification work in batches rather than one token at a time.

---

## How fast is fast?

**Original paper (Google Research, 2022):** accelerating an 11B parameter T5-XXL model for translation, using a 60M parameter T5-small as the draft — **~3× speed improvement**.

**Real-world range:** for code completion and chat, **2–4× faster output** with identical quality. The specific gain depends on how well the draft model aligns with the target — more on that below.

**What users notice:** instead of tokens appearing one by one, responses arrive in short bursts — multi-token chunks committing together. In interactive chat, this registers as a qualitatively snappier experience even before throughput numbers are measured.

---

## What determines the speedup

Three factors interact:

**Draft model size.** The draft must be significantly smaller than the target — 10–100× smaller in practice — so its per-token cost is negligible. Smaller drafts are cheaper but less accurate.

**Draft length.** More tokens proposed per cycle means more parallelism, but also more chances for an early rejection to discard a long tail. Typical values: 3–12 tokens per cycle.

**Acceptance rate.** This is the governing variable. If the draft and target come from the same model family, use the same tokenizer, and were trained on similar data, their predictions align well and acceptance rates are high (70–85% in favorable conditions). Misaligned pairs accept far less and gains shrink.

These interact: a shorter draft with high acceptance rate often beats a longer draft with frequent early rejections. Tuning these three is the practical work of deploying speculative decoding.

---

## What this changes for real products

In a standard deployment, every output token costs the same: one full pass through the large model. Response time scales directly with output length. Long responses are slow; there is no way to make them otherwise without changing the model.

Speculative decoding breaks that coupling. Multiple tokens can be committed in a single model pass. A response that previously required 200 large-model operations might require 60. The response arrives faster — not because the model runs faster, but because it runs fewer times.

This has a specific effect that throughput numbers don't capture: it compresses the long tail. The requests that used to straggle — long outputs, complex completions — now complete within a tighter range. For products where response time is visible to users, that tightening is the actual win.

Quality is unchanged. The large model remains the final check on every token committed.

---

## It's already in production

This isn't a research technique waiting to ship. It's deployed.

- **Google AI Overviews (Google Search):** confirmed by the original authors. Used in production at Google scale, serving real traffic, with quality maintained.
  > *"We have applied speculative decoding in a number of Google products, where we see remarkable speed-ups in inference, while maintaining the same quality of responses."*

- **LM Studio:** added speculative decoding support in a recent release. Local users can enable it today.

- **Local coding tools:** `llama.cpp` supports running a draft and target model together — a 3B model drafting, a 14B model verifying, on mid-range hardware — delivering 2–4× faster code completions in your editor without cloud costs or privacy trade-offs.

What used to be a forced tradeoff — *fast model or accurate model* — now isn't.

---

## Where the field has gone

The baseline described here — one draft model, one target model, linear verification — is the starting point. The field has moved considerably further, all sharing the same core principle: **draft cheaply, verify in parallel, commit only what the target would have produced.**

- **EAGLE / EAGLE-3** — instead of a separate model, a lightweight head attached to the target model drafts using the target's own internal layer states. No second model to maintain; no tokenizer mismatch risk.
- **Multi-Token Prediction (MTP)** — prediction heads baked directly into the model architecture at training time, each head responsible for a token at an increasing future offset. Used in DeepSeek-R1.
- **Tree attention** — rather than a single linear draft sequence, the draft generates a tree of candidate continuations. The target verifies all branches in one pass and picks the longest accepted path.
- **Self-speculative decoding** — a single model acts as both draft and target, using early exit layers or skipped layers to generate cheap candidate tokens from its own weights.

These directions are explored in the further reading section below.

---

## Further reading

The sections deliberately omitted from this article, and where to find them:

**Advanced variants (EAGLE-3, MTP, tree attention, self-speculative decoding)**
For readers ready to go deeper on the architecture — how EAGLE operates on hidden states instead of a separate model, how DeepSeek-R1-style multi-token prediction heads work, and how tree-shaped candidate sets improve on linear drafts. See `blog_article_omitted.md` §A, or the L3 blog (`blog_l3_technical.md`).

**P99 tail latency and production systems in depth**
The full treatment of why MoE architectures specifically stress latency budgets, how cascaded safety systems compound the problem, why replica scaling doesn't fix tails, and draft model training as production infrastructure. See `blog_article_omitted.md` §B.

**Practical setup and code**
The `llama.cpp` command to run a draft+target pair locally, with flags explained; the NVIDIA TensorRT EAGLE-3 configuration code snippet. See `blog_article_omitted.md` §C.

**The batch size tradeoff**
Speculative decoding speedup shrinks with larger batch sizes — a known tradeoff not covered here because it requires understanding batch scheduling mechanics. Covered at L3/L4 level.

**The hardware roofline model**
The full arithmetic intensity analysis of decode vs. prefill, H100 utilization numbers, and why the FLOP/byte ratio during decode is below 1. L3 territory.

---

*Sources: [Chris Thomas (Feb 2025)](https://christhomas.co.uk/blog/2025/02/16/speculative-decoding-using-llms-efficiently/) · [NVIDIA Developer Blog (Sep 2025)](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/) · [Google Research (Dec 2024)](https://research.google/blog/looking-back-at-speculative-decoding/) · [Nebius Blog](https://nebius.com/blog/posts/moe-spec-decoding)*
