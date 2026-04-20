# Speculative Decoding, Visualized

**Source:** https://www.adaptive-ml.com/post/speculative-decoding-visualized
**Author:** Dylan Ebert (Adaptive ML)
**Published:** January 22, 2026
**Level:** L2 — Best visual/conceptual explainer; probability geometry made concrete
**Why here:** The clearest visual treatment of why speculative decoding preserves the exact target distribution. The "green region = guaranteed acceptance, red excess = overestimation" framing is the most memorable explanation of rejection sampling available anywhere. Replaces the original Reddit r/LocalLLaMA interactive explainer link.

---

## Summary

One paragraph summary of the entire mechanism:

> *"Large language models generate text one token at a time. Each token requires a full forward pass through billions of parameters. Speculative decoding makes this faster by using a small draft model to propose tokens, then verifying them all at once with the large target model."*

The article's unique contribution: a **visual probability geometry** showing exactly how the acceptance criterion works and why it preserves the target distribution.

---

## Why generation is slow

You can't predict token 5 without first knowing token 4. The process is inherently sequential.

A 70B parameter model taking 50ms per token needs **5 full seconds** to generate 100 tokens, waiting for each step to complete before starting the next.

**The asymmetry that makes speculative decoding possible:**

> Verifying N tokens takes **one forward pass**, while generating N tokens takes **N forward passes**.

A language model can score an entire sequence in parallel — computing the probability distribution at every position simultaneously. If we could guess the next several tokens correctly, we could verify all of them in a single pass.

---

## The draft-then-verify loop

1. Generate K draft tokens using the **draft model**.
2. Verify all K tokens in **one target model forward pass**.
3. Accept tokens until a mismatch, then resample and discard the rest.

**Best case:** All K tokens are accepted, plus a bonus token sampled by the target model = **K+1 tokens from a single pass**.

**Guarantee:** Even when some are rejected, you're guaranteed at least one token of progress.

---

## How distribution preservation works (the key insight)

This is the article's most important contribution. The acceptance criterion is not "accept if the tokens match exactly" — it is a probabilistic rule that guarantees the output is drawn from exactly the target's distribution.

### The probability geometry

At each token position, compare `p_target` (target model's probability for that token) vs `p_draft` (draft model's probability):

```
P(accept) = min(1, p_target / p_draft)
```

Visually:
- **Green region** = `min(p_target, p_draft)` — guaranteed acceptance mass.
- **Red excess** = where the draft overestimated (`p_draft > p_target`) — this is rejected proportionally.

When `p_target ≥ p_draft`: no red region — **always accept**.
When `p_draft > p_target`: red excess appears — accept with probability `p_target / p_draft`.

### The residual distribution

When a token is rejected, you **cannot simply resample from `p_target`** — that would double-count tokens the draft already had a chance to propose. Instead, resample from the **residual distribution**:

```
p_residual(x) = max(0, p_target(x) - p_draft(x))   (normalized)
```

The residual is `p_target` minus the guaranteed acceptance mass — normalized to form a valid probability distribution.

### Why this gives exactly the target distribution

Consider outputting token x. Two paths:

**Path 1:** Draft proposes x, we accept it.
```
P(Path 1 outputs x) = p_draft(x) × min(1, p_target(x)/p_draft(x))
                    = min(p_draft(x), p_target(x))
```

**Path 2:** Draft proposes some other token y, we reject, resample x from residual.
```
P(Path 2 outputs x) = max(0, p_target(x) - p_draft(x))
```

Total = `min(p_draft(x), p_target(x)) + max(0, p_target(x) - p_draft(x))` = **`p_target(x)`** ✓

In all cases, the probability of outputting x is exactly the target's probability. This is the mathematical guarantee.

---

## When does it help?

Speedup depends on acceptance rate and the cost ratio between models.

**Works best when:**
- Draft aligns well with target (distilled versions, same-family smaller models).
- Size ratio is large (70B/7B gains more than 7B/1B).
- Generation is predictable (code, structured output, common phrases).

### Worked speedup calculation

Suppose:
- Target model: 50ms per forward pass
- Draft model: 5ms per forward pass
- Draft length K = 5
- Average acceptance: 3 tokens

```
Cost per iteration = 50ms (target) + 5 × 5ms (drafts) = 75ms
Tokens generated   = 3 accepted + 1 resampled = 4 tokens
Effective rate     = 75ms / 4 = 18.75ms per token
Baseline           = 50ms per token
Speedup            = 50 / 18.75 = 2.67×
```

**The optimal K tradeoff:** Too few drafts wastes verification parallelism. Too many wastes effort on tokens that will be rejected.

**Worst case:** First speculative token is wrong → run both models → get 1 token → slower than baseline.
**Expected case:** 2–3× speedup with a well-matched draft model.

---

## Connection to Layer 14

| Article concept | Layer 14 code |
|----------------|---------------|
| Green/red probability geometry | `_accept_reject()` in `spec_runner.py` |
| `P(accept) = min(1, p_target/p_draft)` | The **greedy special case**: `argmax(target) == draft_token` (temperature=0) |
| Residual distribution | When a token is rejected, target generates its own token from `p_target` |
| K+1 tokens per round | `num_spec_tokens + 1` in `verify_extend` |
| Acceptance rate tracking | `_total_accepted / _total_proposed` in `lesson/07_statistics.md` |

> **Note:** Layer 14 uses greedy (temperature=0) decoding — the stochastic acceptance criterion simplifies: accept iff `argmax(target_logits) == draft_token`. The residual simplifies to always outputting the target's argmax on rejection. The probability geometry in this article is the general (temperature > 0) case that the Layer 14 code is a special case of.

---

## Limits of this article (for book context)

- Does not show code; purely conceptual.
- Covers only the classic two-model approach — no EAGLE, Medusa, N-gram.
- Does not address KV cache management when two models share a prefix (Layer 14's core engineering).
- The interactive visualizations are live on the web page; this markdown captures only the text.
