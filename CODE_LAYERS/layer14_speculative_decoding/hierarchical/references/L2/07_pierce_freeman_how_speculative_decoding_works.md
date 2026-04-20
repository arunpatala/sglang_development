# How Speculative Decoding Works

**Source:** https://pierce.dev/notes/how-speculative-decoding-works/
**Author:** Pierce Freeman
**Level:** L2 — Best concise explainer; best/worst/average case analysis
**Why here:** Most compact and readable technical explanation available. Exceptionally clear treatment of the *best case / worst case / average case* triad and the "you only need the small model to be right some of the time" insight. Good as a fast L2 primer before the longer Brenndoerfer or Jarvis Labs articles.

---

## Summary

Speculative decoding speeds up inference by 2–4× in the best cases, "without any approximation or quantization tricks." A small draft model guesses multiple tokens; the large target model verifies them all in one forward pass. When the draft is right, you get multiple tokens for the price of one large-model pass. When it's wrong, you haven't lost much — you still get at least one token.

---

## The bottleneck: multi-GPU coordination overhead

As models scale, they're served across multiple GPUs linked with NVLink or InfiniBand.

- An A100 80GB fits roughly 40B params in FP16.
- Anything larger shards across multiple GPUs.
- Every token generation requires: coordinate across all GPUs → run forward pass → repeat.

The constraint is **autoregressive inference**: you can only generate one token at a time, since each new token must be conditioned on all previous ones.

This is compounded by scale: a 70B model takes roughly 10× longer per token than a 7B model — more transformer blocks, more floating-point arithmetic, more coordination.

---

## The "obvious tokens" insight

> *"Not all tokens are created equal. Some next-token choices are obvious, even trivial. If I write 'The capital of France is...', you know the next token should be 'Paris' without needing a 70B parameter model to figure it out."*

A much smaller model could predict these tokens just as accurately. Speculative decoding exploits this: use a fast draft model for easy tokens, reserve the big model for verification.

This is a clever hack that works **because many token sequences in natural language are predictable enough that a small model gets them right.** And when the small model is wrong, you haven't lost much — you were going to need the big model anyway.

---

## How it works: two phases

### Phase 1: Speculation

Run a small, fast model (1–7B params) to generate a sequence of K candidate tokens. Since this model is much smaller, it can generate K tokens faster than the large model could generate even one.

### Phase 2: Verification

Take all K candidate tokens and feed them into the large model in **one single forward pass**. The large model processes the entire sequence at once and outputs probability distributions at every position — verifying all draft tokens simultaneously.

Then: **accept tokens from left to right** as long as they match (or are close to) what the large model predicted. At the first mismatch, stop. Reject that token and all subsequent speculative tokens. Use the large model's prediction instead.

### The actual sampling criterion

More precisely: accept each token with probability `min(1, p_large / p_draft)`.

- If `p_large ≥ p_draft` → always accept (probability = 1).
- If `p_large < p_draft` → accept probabilistically.

This rejection-sampling scheme ensures the **final distribution is mathematically identical** to what you'd get from just running the large model alone. You can accept many tokens even when probabilities differ significantly.

---

## Performance characteristics: best / worst / average

### Best case
All K speculative tokens are accepted. You get K tokens for the price of one large-model forward pass (plus cheap draft generation).

```
If K=4 and all accepted → 4x speedup over baseline
```

### Worst case
The very first speculative token is wrong. You reject it and all subsequent tokens, keeping only the large model's prediction.

```
→ generated 1 token but ran both small + large model
→ slightly slower than standard autoregressive generation
```

### Average case
Some prefix of the speculative tokens are accepted.

```
If you accept 2 out of 4 tokens on average → ~2x speedup overall
```

**The key insight:** You only need the small model to be right *some* of the time for this to be worthwhile.

> *"Even if the small model is wrong 60% of the time, you can still see significant speedups because of the parallel verification."*

---

## Choosing the right draft model

**Size rule:** Draft model should be 10–20× smaller than the target model. A 7B draft paired with a 70B target is common.

- If the draft is too large, drafting itself becomes slow — defeating the purpose.

**Alignment rule:** The draft model should make similar predictions in similar contexts. If the draft has a completely different training distribution or instruction-following behavior, it will consistently be rejected.

**Options:**
1. Use an off-the-shelf checkpoint that has been benchmarked for alignment with the target.
2. Train draft models specifically for this purpose, using the same data distribution as the target with fewer parameters. Boosts acceptance rates significantly in production.

---

## Real-world performance

| Factor | Effect on speedup |
|--------|------------------|
| Text predictability (code, structured) | Higher — code is repetitive, easy to predict |
| Creative writing | Lower — more diverse token choices |
| Better draft-target alignment | Higher acceptance rate → more speedup |
| Larger size ratio (70B/7B vs 7B/1B) | 70B/7B gains more |
| GPU memory bandwidth | Speedup tied to memory-bound nature of inference |

Typical: **2–3× speedup** with a well-matched draft model. Up to **4×** in best cases.

---

## Production deployment

The technique is deployed at most frontier labs. Combined with quantization and better chips, it's working behind the scenes to dramatically reduce latency over the past few years.

> *"Same parameters, same hardware, faster models."*

---

## Connection to Layer 14

| Article concept | Layer 14 code |
|----------------|---------------|
| Draft model (1–7B) | `spec_runner.draft_model_runner` |
| K candidate tokens | `num_spec_tokens` |
| One target forward pass (verify) | `_verify_extend()` |
| Accept left-to-right until mismatch | `_accept_reject()` |
| At mismatch: target generates its own token | The +1 in K+1 token output |
| Best/worst/average case analysis | `tokens_per_step` metric in `lesson/07_statistics.md` |

---

## Limits of this article (for book context)

- Concise by design — does not cover KV cache management, the N+1 window trick, or causal masking in verify.
- No code examples (read `01_mlmastery_practitioner_guide.md` for working HuggingFace code).
- No variant coverage (EAGLE, Medusa, N-gram, self-speculative).
- Does not discuss batching — Layer 14's `verify_extend` must handle multiple concurrent requests in a batch, not just one sequence.
