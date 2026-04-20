# Speculative Decoding: Using LLMs Efficiently

**Source:** https://christhomas.co.uk/blog/2025/02/16/speculative-decoding-using-llms-efficiently/
**Author:** Chris Thomas (AI consultant)
**Date:** February 16, 2025
**Level:** L1 — Orientation
**Why here:** Best "junior/senior developer" analogy. Practical VS Code demo. Good entry point for L1 readers who use coding tools.

---

## Summary

Speculative decoding makes large language models work more efficiently. Even with powerful hardware, code completion can feel sluggish — the bottleneck isn't computational power, it's how efficiently we use it.

---

## The core analogy

Think of speculative decoding like having two developers working together: a **junior dev** who quickly drafts code, and a **senior dev** who reviews and corrects it. Instead of the senior dev writing everything from scratch, they can review multiple lines at once, saving significant time.

Two models work in tandem:
- A **smaller, faster model** (like Qwen 2.5 Coder 1.5B or 3B) quickly generates potential completions.
- A **larger, more capable model** (like Qwen 2.5 Coder 14B) verifies those completions.

This makes better use of modern hardware in two ways:
1. While the large model is loading its weights from memory, the smaller model is already generating suggestions.
2. The large model can verify multiple tokens in parallel rather than generating them one at a time.

**Result:** code completion 2–4× faster, with the same quality as the larger model alone.

---

## Why LLM inference is slow (the technical reason, briefly)

LLMs generate text one token at a time. Each token's generation requires reading all the model's parameters from memory. This process is **memory-bound**: speed is limited by how quickly the model can read its parameters from DRAM, not by compute.

Speculative decoding mitigates this by:

- **Reducing memory access:** Instead of generating each token sequentially with the target model, the target model processes multiple draft tokens in a single pass — its parameters are read fewer times.
- **Parallel processing:** The target model evaluates multiple draft tokens in one pass (as a batch), leveraging the parallelizability of the transformer network.
- **Compute/bandwidth balance:** LLMs are bandwidth-limited. Speculative decoding trades available compute for lower bandwidth requirements.

---

## Three factors that determine efficiency

1. **Draft model size:** The draft must be significantly smaller than the target so it generates tokens quickly.
2. **Draft length:** More draft tokens = more potential speedup, but also more chances of rejection.
3. **Acceptance rate:** The draft model must be well-aligned with the target so enough draft tokens are accepted.

---

## Practical setup (VS Code example)

Using the `llama-vscode` extension and `llama.cpp` HTTP server:

```
llama-server.exe -md qwen2.5-coder-3b-q8_0.gguf -m qwen2.5-coder-14b-q8_0.gguf --port 8012 -c 2048 --n-gpu-layers 99 -fa -ub 1024 -b 1024 -dt 0.1 --ctx-size 0 --cache-reuse 256
```

- `-md`: draft model (3B)
- `-m`: main model (14B)

The extension handles accept (Tab), accept first line (Shift+Tab), accept next word (Ctrl/Cmd+Right).

---

## Real-world benefits

- **2–4× faster** suggestions appear as you type.
- **Local processing** improves privacy and reduces cloud costs.
- Developers no longer choose between speed (small model) or quality (large model) — they get both.
- Mid-range machines can run sophisticated code completion locally.

---

## Where it's going

- Already deployed in production: **AI Overviews in Google Search** uses speculative decoding.
- **LM Studio** added speculative decoding support (latest beta).
- Growing relevance as on-device AI processing becomes important for privacy and personalisation.

---

## Key quote

> "The key insight isn't just about making things faster — it's about making better use of the computing resources we already have."

---

## Limits of the analogy

The "junior/senior developer" metaphor breaks when:
- The junior's suggestions are **never shown to the user** if rejected (unlike real code review).
- Acceptance isn't about quality judgment — it's about whether the senior would have produced the **identical** token.
