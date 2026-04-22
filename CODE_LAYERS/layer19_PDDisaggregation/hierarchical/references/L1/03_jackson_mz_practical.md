# Practical Guide to LLM: Prefill & Decode Disaggregation

**Source:** https://betterprogramming.pub/practical-guide-to-llm-prefill-decode-disaggregation-bd7f9ee4eaf5
**Author:** Jackson MZ (Research Engineer, ex-DeepMind, ex-Google)
**Published:** April 5, 2025 — 2-minute read
**Series:** "Practical Guide to LLM" (short intuition + implementation posts)
**Level:** L1 — Shortest useful introduction; mental model in 2 minutes
**Why here:** The most concise L1 entry point. Sets up the mental model (compute-bound prefill, bandwidth-bound decode, separate them) and the flow diagram (`Prefill cluster → KV cache → Queue → Decode cluster → output response`) in a single short read. Links to an implementation in under 99 lines of code. Designed to be read before all other references.

---

## Core Mental Model

### Prefill Phase
- Calculates full attention for the user's input prompt
- All input tokens processed in parallel → large matrix multiplications
- **Compute-bound**: the GPU's tensor cores are the bottleneck

### Decode Phase
- Autoregressive output token generation
- One token at a time — each token depends on the previous
- Less compute-intensive but requires large KV cache access
- Efficient at larger batch sizes (amortises memory reads)
- **Bandwidth-bound**: GPU waits for HBM reads, not for compute

### Why Disaggregate?

> "To make sure inference is not compute- and bandwidth-bound at the same time, we want to separate them."

Running both phases on the same GPU means the GPU is simultaneously under two different types of resource pressure — compute saturation during prefill and memory bandwidth saturation during decode. They interfere with each other.

### The Flow

```
Prefill cluster → KV cache → Queue → Decode cluster → output response
```

1. Incoming request arrives at the **prefill cluster**.
2. Prefill processes the prompt, builds the KV cache.
3. KV cache enters the **queue** (transfer buffer).
4. **Decode cluster** receives the KV cache and begins generating tokens.
5. Output response streams to the client.

---

## Implementation

The article links to the vLLM disaggregated prefill documentation as the canonical implementation reference:
- **vLLM Disaggregated Prefilling:** https://docs.vllm.ai/en/stable/features/disagg_prefill.html

A companion post in the same series covers the implementation in under 99 lines of code:
- **GitHub gist:** available via the article's companion repository

---

## Series Context

This is part of "Practical Guide to LLM" — a series of short posts providing intuitions and implementation hints for LLM basics:
- Post on prefill and decode phase details (prerequisite)
- This post on disaggregation (the current article)
- Follow-up posts on related LLM optimisation techniques

---

## Key Takeaways for Layer 19

- **Best used as the first read** — before any other reference in this directory.
- The one-line summary of why to disaggregate: "To make sure inference is not compute- and bandwidth-bound at the same time."
- The 4-step flow (`Prefill cluster → KV cache → Queue → Decode cluster`) is the correct mental model for understanding SGLang's router + prefill server + decode server architecture.
- Decode is more efficient with **larger batch sizes** — this is why the decode pool typically needs more GPUs than the prefill pool in production deployments.
- For the implementation details, move to the vLLM docs (L2) or SGLang docs (L2/01).
