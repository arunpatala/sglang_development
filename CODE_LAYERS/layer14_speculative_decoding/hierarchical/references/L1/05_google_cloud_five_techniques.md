# Five Techniques to Reach the Efficient Frontier of LLM Inference

**Source:** https://cloud.google.com/blog/topics/developers-practitioners/five-techniques-to-reach-the-efficient-frontier-of-llm-inference
**Author:** Karl Weinmeister (Director, Developer Relations, Google Cloud)
**Date:** March 27, 2026
**Level:** L1 — Orientation (landscape view)
**Why here:** Places speculative decoding inside the full inference optimization landscape. Best article for readers who want to know *where* speculative decoding fits relative to batching, paging, routing, and quantization — without going deep on any one.

---

## Summary

Every dollar spent on model inference buys a position on a graph of latency vs throughput. There is an **efficient frontier** — a curve of optimal configurations. Most production systems operate below it. Five techniques close the gap.

---

## The efficient frontier concept

Borrowed from portfolio theory: the efficient frontier is the boundary where you have squeezed maximum performance from your hardware.

**Two dynamics:**
1. **Getting to the frontier** — applying existing techniques. Within your control.
2. **The frontier moving outward** — new research, hardware, open source. Outside your control.

Your job: stay close to today's frontier while building infrastructure flexible enough to absorb tomorrow's.

---

## Why an efficient frontier exists for LLM inference

Every LLM request has two phases with **different hardware bottlenecks**:

| Phase | Bottleneck | Characteristic |
|-------|-----------|----------------|
| **Prefill** | Compute (tensor cores) | Processes entire input prompt in parallel; GPU is highly utilized. |
| **Decode** | Memory bandwidth (HBM) | Generates one token at a time; must fetch all model weights + KV cache for each token. |

This mismatch — compute-bound prefill vs memory-bandwidth-bound decode — is why the frontier exists. You can trade latency for throughput, but can't improve both without the frontier shifting.

### The two axes

| Axis | Key metrics | Hardware constraint |
|------|------------|---------------------|
| **Latency (X)** | Time to First Token (TTFT) + Time Between Tokens (TBT) | Compute (prefill) + memory bandwidth (decode) |
| **Throughput (Y)** | Total tokens/second across all users | Batch size × memory capacity |

---

## The five techniques

### 1. Semantic routing across model tiers

Not every query needs a 400B parameter model. A lightweight classifier at the gateway edge routes:
- Complex reasoning → frontier-class models.
- Simple tasks → small, quantized models.

Result: dramatic throughput improvement without sacrificing aggregate quality.

### 2. Prefill and decode disaggregation

Run dedicated **prefill clusters** (compute-dense) and **decode clusters** (high-bandwidth). Connect with high-speed networks that transfer only the compressed KV cache state.

Pushes both phases toward their theoretical hardware limits independently.

### 3. Quantization

Reduce model weights from FP16 to INT8 or INT4:
- Halves or quarters memory footprint.
- Because decode is memory-bandwidth-bound, 4-bit weights can be read up to **4× faster** than 16-bit.
- Direct TBT (Time Between Tokens) improvement.

Modern techniques (AWQ, GPTQ) preserve near-FP16 quality at INT4 speeds.

### 4. Context routing (the biggest lever most teams miss)

With prefix caching, the KV cache for shared prefixes (e.g., the same 100-page RAG document) is computed once and reused.

**The catch:** standard L4 load balancers scatter requests randomly. If two requests for the same document land on different GPUs, the cache is useless.

**Context-aware L7 routing** inspects the prompt prefix and routes to the pod that already holds that context in cache — slashing TTFT by up to 85%.

*Vertex AI case study:*
- 35% faster TTFT for Qwen3-Coder.
- 2× better P95 tail latency for DeepSeek V3.1.
- Prefix cache hit rate: 35% → 70%.

### 5. Speculative decoding

During the decode phase, tensor cores are mostly idle — bottleneck is memory bandwidth.

Speculative decoding **exploits this wasted compute**:
1. A small fast draft model generates several candidate tokens cheaply.
2. The large target model verifies all candidates in **a single forward pass** — a parallel compute-bound operation, rather than sequential memory-bound.
3. If correct: 4–5 tokens generated for the memory cost of one.

> **This directly breaks the TBT floor set by memory bandwidth.**

The draft model is tiny relative to the main model. The latency tradeoff is worthwhile.

*Self-speculative decoding variant:* some newer models use **specialized internal prediction heads** instead of a separate draft model, eliminating the overhead of managing two models.

---

## Bottom line

**Getting to the frontier is within your control.** The techniques exist today. If you're not applying them, you're operating below the curve and overpaying for every token.

**The frontier keeps moving.** Build infrastructure flexible enough to absorb the next advance.

> *"The organizations that will win on inference economics aren't the ones with the most GPUs. They're the ones that systematically close the gap to today's frontier while staying ready for tomorrow's."*

---

## Where speculative decoding fits in this picture

Speculative decoding is **technique #5 of five** — one tool in the stack, specifically targeting the decode TBT floor that memory bandwidth creates. The other four (routing, disaggregation, quantization, prefix caching) are complementary and should be applied together.

Layer 14 implements the core mechanism of technique #5 from scratch, giving the reader the mental model needed to reason about the other four.

---

## Limits of this article (for book context)

- Strategic/landscape framing — does not go deep on implementation of any one technique.
- Speculative decoding section is intentionally brief; the other articles in this L1 folder provide the depth.
- Self-speculative decoding (prediction heads) is the Medusa / MTP direction; Layer 14 uses a separate `ModelRunner` (the classic draft-target approach).
