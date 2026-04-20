# An Introduction to Speculative Decoding for Reducing Latency in AI Inference

**Source:** https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/
**Author:** Jamie Li (NVIDIA)
**Date:** September 17, 2025
**Level:** L1 — Orientation (with L3 depth on EAGLE-3 and MTP)
**Why here:** Authoritative GPU-framing; "chief scientist / lab assistant" analogy; covers classic draft-target, EAGLE-3, and MTP in one post; includes NVIDIA TensorRT code snippet.

---

## Summary

GPUs offer massive compute, yet much of that power sits idle during autoregressive generation — each token requires a full forward pass, reloading weights, and synchronizing memory. Speculative decoding breaks through this wall by predicting and verifying multiple tokens simultaneously.

---

## The core analogy

A **chief scientist** relies on an **efficient assistant** to handle routine experiments. The assistant rapidly works through the checklist; the scientist validates and steps in to correct when necessary.

With speculative decoding:
- The **draft mechanism** (assistant) proposes multiple possible continuations.
- The **target model** (scientist) verifies them in batches.

The benefit: fewer sequential steps, reduced memory bandwidth bottleneck, preserved output quality.

---

## The draft-target approach (classic implementation)

Two models:
- **Target model:** large, high-quality, final authority.
- **Draft model:** smaller, faster, often distilled from the target.

### Step 1 — Draft generation

A smaller mechanism generates 3–12 candidate tokens. Typically a separate smaller model trained on the same data distribution, with the target's output as ground truth.

### Step 2 — Parallel verification

The target model processes the input sequence **and all draft tokens simultaneously** in a single forward pass, computing probability distributions for each position.

The key efficiency: the KV cache holds previously computed values, so only the new speculated tokens incur compute cost during verification.

### Step 3 — Rejection sampling

Compares `P(Draft)` against `P(Target)` for each position:
- If `P(Target) ≥ P(Draft)`: token accepted.
- If `P(Target) < P(Draft)`: token rejected; all subsequent draft tokens discarded; reverts to standard generation from the last accepted token.

Only when a draft token matches what the target would have generated is it accepted — guaranteeing **identical output** to standard decoding.

**Acceptance rate** = accepted tokens / total draft tokens generated. Higher acceptance = bigger speedup. Worst case: all rejected, one target token generated (same as baseline).

---

## The EAGLE / EAGLE-3 approach

**EAGLE (Extrapolation Algorithm for Greater Language-Model Efficiency):** operates at the feature level, using a lightweight autoregressive prediction head that ingests hidden states from the target model's internal layers — eliminating the overhead of training a separate second model.

**EAGLE-3** improvements:
- **Multi-layer fused features:** takes low, middle, and high-level embeddings from the target into the drafting head.
- **Dynamic draft tree:** context-aware, beam-search candidate selection based on cumulative log probs.
- **Instance-adaptive drafting:** the head evaluates its own confidence as it builds the tree; stops drafting below a threshold.
- **Tree attention:** the target verifies the entire candidate tree in one pass, pruning invalid branches.

The EAGLE head = a lightweight Transformer decoder layer + final linear layer. It generates a **tree** of candidates rather than a single linear sequence.

---

## Multi-Token Prediction (DeepSeek-R1 style)

MTP uses **multi-head prediction** baked into the model itself:
- Each head drafts one future token at increasing distances.
- The main model verifies in order, keeping the longest accepted prefix.
- No separate draft model needed.

**Difference from EAGLE:** MTP uses specialized multi-token prediction heads; EAGLE uses a single head extrapolating internal feature states.

---

## Latency impact (concrete numbers)

Standard decode: 3 tokens at 200 ms/pass = **600 ms**

With speculative decoding (2 draft tokens, 1 verification pass):
- Verification pass ≈ 250 ms
- 3 tokens committed = **250 ms** instead of 600 ms

Users see responses materialize in faster, **multi-token chunks** rather than word by word. Especially noticeable in interactive chatbots.

---

## Quick implementation sketch (NVIDIA TensorRT-Model Optimizer, EAGLE-3)

```python
import transformers
import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import EAGLE3_DEFAULT_CFG

mto.enable_huggingface_checkpointing()

base_model = "meta-llama/Llama-3.2-1B"
model = transformers.AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype="auto", device_map="cuda"
)

config = EAGLE3_DEFAULT_CFG["config"]
config["eagle_architecture_config"].update({
    "hidden_size": model.config.hidden_size,
    "vocab_size": model.config.vocab_size,
    "draft_vocab_size": model.config.vocab_size,
    "max_position_embeddings": model.config.max_position_embeddings,
})

mtsp.convert(model, [("eagle", config)])
```

Full tutorial: [TensorRT-Model-Optimizer/examples/speculative_decoding](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/speculative_decoding/example.ipynb)

---

## Key quote

> "Compared with standard autoregressive decoding, which produces one token per pass, this technique lets the system generate multiple tokens at once, cutting latency and boosting throughput without any impact on accuracy."

---

## Limits of this article (for book context)

- EAGLE-3 and MTP are production-grade extensions; Layer 14 implements **linear greedy draft-target** — the simpler baseline this article introduces.
- The code snippet uses NVIDIA's `modelopt` library, not SGLang or raw PyTorch — useful for L4/L5 but opaque at L1.
