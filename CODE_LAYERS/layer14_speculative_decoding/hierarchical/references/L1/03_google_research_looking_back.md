# Looking Back at Speculative Decoding

**Source:** https://research.google/blog/looking-back-at-speculative-decoding/
**Authors:** Yaniv Leviathan (Distinguished Engineer), Matan Kalman (Software Engineer), Yossi Matias (VP, Google Research)
**Date:** December 6, 2024
**Level:** L1 — Orientation (with L3 depth on the sampling theory)
**Why here:** Written by the original authors; covers the "why we built this" story; confirms real production use (AI Overviews, Google Search); includes the speculative sampling theory cleanly.

---

## Summary

Speculative decoding, introduced by Google Research in 2022, has proven to be an effective technique for faster and cheaper LLM inference without compromising quality. It is now used in Google products including AI Overviews in Google Search.

---

## Background: why LLMs are slow

An LLM generates output one token at a time. The sentence *"One small step for man, one giant leap for mankind"* is 12 tokens — the LLM must run 12 times.

Larger models are slower because each decoding step must **read the entirety of the model's weights** — on the order of a terabyte of data for each word produced.

Since each token depends on the ones before it, they must be generated **one by one**, reading all weights again and again.

---

## Two observations that motivate speculative decoding

### Observation 1: Some tokens are easier to generate than others

> *"What is the square root of 7? The square root of 7 is 2.646."*

Generating the second "7" is easy — it copies the one from the question. Generating "2.646" is hard — the model must compute or recall the answer.

**Implication:** large models are better mainly in difficult cases. In many easy cases, a small model can approximate the large model well.

### Observation 2: The bottleneck for LLM inference is usually memory

Modern ML hardware can perform hundreds of **trillions of operations per second**, but memory bandwidth is around **trillions of bytes per second** — two orders of magnitude lower.

The Transformer architecture performs only a **few operations per byte read** during inference, meaning there are **spare computational resources** available — compute is idle while memory is the bottleneck.

**Implication:** we can run extra computation in parallel without slowing down the bottleneck.

---

## Speculative execution (the inspiration)

Speculative execution is a CPU optimization: perform a task before you know if it is needed, increasing concurrency.

Abstract form:

```
Y = f(X)        # slow
Z = g(Y)        # slow, depends on Y
```

With a fast approximation `f*(X)`:
- Run `f(X)` and `g(f*(X))` in parallel.
- When `f(X)` finishes: if `f*(X) == f(X)`, accept the result of `g(f*(X))`.
- If not: discard and run `g(Y)` serially.

Output is guaranteed identical either way. The more accurate `f*`, the more concurrency gained.

---

## Speculative sampling (the key extension)

LLMs output **probability distributions**, not single tokens. Direct speculative execution would be inefficient: if both `f` and `f*` output uniform distributions over 100 tokens, a sample from `f*` would match `f`'s sample only 1% of the time.

**Speculative sampling** solves this: accept or reject `f*`'s guesses **probabilistically** based on the ratio of `f(X)` to `f*(X)`, guaranteeing:
- **Optimality:** maximum acceptance rate for a given `f*`.
- **Identical output distribution** to the target model.

This is the key correctness guarantee: speculative decoding output is **mathematically identical** to standard decoding.

---

## Speculative decoding in practice

LLMs are autoregressive: `f(X)` = "take a sequence, output a distribution for the next token." `g(Y)` = same function, applied one step later.

Speculative decoding parallelizes: compute the token **and** the tokens following it simultaneously, then use speculative sampling to decide which to keep.

**Results from the original paper:**
- Accelerating an **11B parameter T5-XXL** model for translation.
- Using a **60M parameter T5-small** as the draft ("guessing mechanism").
- Result: **~3× improvement in speed.**

---

## Production deployment

> *"We have applied speculative decoding in a number of Google products, where we see remarkable speed-ups in inference, while maintaining the same quality of responses."*

**AI Overviews in Google Search** uses speculative decoding to produce results faster while maintaining response quality. It remains a significant part of their inference optimizations even as other techniques are added.

---

## The broader landscape (as of Dec 2024)

The speculative decoding paradigm has spawned many extensions:
- **Distributed setups** using multiple draft guesses.
- **Knowledge distillation** from target to draft.
- **Self-speculative** decoding: one model acts as both draft and target.
- **Tree attention:** verifying all draft tokens together.
- Applied to **image and speech** generation as well.

---

## Key quotes

> *"The algorithm speeds up generation from autoregressive models by computing several tokens in parallel, without affecting output quality; in fact, the method guarantees an identical output distribution."*

> *"Producing results faster with the same hardware also means that fewer machines are needed for serving the same amount of traffic, which translates yet again to a reduction in the energy costs of serving the same model."*

---

## Limits of this article (for book context)

- Covers the **sampling-based** acceptance rule, not just the greedy argmax rule used in Layer 14.
- Layer 14's `_accept_reject` is the **greedy special case** (temperature=0); this post's sampling theory is the more general `lesson/09_whats_next.md` extension.
- Production numbers (exact speedup) are not disclosed — "remarkable" is the word used.
