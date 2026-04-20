# The Machine Learning Practitioner's Guide to Speculative Decoding

**Source:** https://machinelearningmastery.com/the-machine-learning-practitioners-guide-to-speculative-decoding/
**Level:** L2 — Definitions + motivation
**Why here:** Best single article for anchoring the three key metrics (acceptance rate α, speculative token count γ, acceptance length τ) with formulas and worked numbers. Includes a complete HuggingFace Transformers implementation. Colab notebook linked.

---

## Summary

Speculative decoding reduces LLM inference latency without sacrificing output quality. This article covers the "why" (memory-bound generation), the "how" (draft → verify → reject-sample), the key metrics you need to track, and a working implementation using HuggingFace.

---

## Why LLM inference is slow

### Sequential generation

LLMs generate autoregressively — one token at a time. Each new token depends on all previous ones:

1. Model receives input tokens.
2. Runs a forward pass through all layers.
3. Predicts probability distribution for next token.
4. Samples or selects most likely token.
5. Appends that token to input.
6. Repeat.

To generate the sentence "The scientist discovered a new species" (six tokens): six complete forward passes sequentially.

### The memory bandwidth bottleneck

**Each forward pass requires loading the entire model's weights from memory into the compute cores.** For large models, this means loading terabytes of data per generated token. The GPU's compute cores sit idle while waiting for data — this is being **memory-bound**.

### Not all tokens are equally hard

> *After "The discovery was made in the", predicting "Amazon" is easier because it appeared earlier in the context. But predicting "species" after "The scientist discovered a new" requires understanding semantic context.*

If some tokens are easy, a smaller, faster model could handle them.

---

## How speculative decoding works

Three steps: **draft → verify → reject-sample**

### Step 1: Token speculation (draft generation)

A smaller, faster **draft model** generates 3–10 candidate tokens ahead. Not as accurate as the target, but much faster. Also called "assisted generation."

### Step 2: Parallel verification

The **target model** takes one single forward pass with the original input plus all draft tokens. Because of how transformer models work, this produces probability distributions for every position simultaneously — verifying all draft tokens at once.

> *Computational cost ≈ one standard forward pass, but potentially validating multiple tokens.*

### Step 3: Rejection sampling

For each draft token position, compare:
- **P(draft):** probability the draft model assigned to its chosen token.
- **P(target):** probability the target model assigns to that same token.

```
For each draft token in sequence:
    if P(target) >= P(draft):
        Accept (target agrees or is more confident)
    else:
        Accept with probability P(target) / P(draft)
        if rejected:
            Discard this token and all following draft tokens
            Generate one new token from target model
            Break; start next speculation round
```

### Worked example

Draft model proposed: **"discovered a breakthrough"**

| Token | P(draft) | P(target) | Decision |
|-------|----------|-----------|----------|
| "discovered" | 0.6 | 0.8 | ACCEPT (0.8 ≥ 0.6) |
| "a" | 0.7 | 0.75 | ACCEPT (0.75 ≥ 0.7) |
| "breakthrough" | 0.5 | 0.2 | REJECT; target generates "new" |

Result: 3 tokens from one target forward pass (plus draft generation).

### Best case

When all K draft tokens are accepted: target verifies K and simultaneously generates one additional token → **K+1 tokens in a single target forward pass**.

---

## Key performance metrics

### Acceptance rate (α)

```
α = (number of accepted tokens) / (total draft tokens proposed)
```

| Range | Interpretation |
|-------|---------------|
| α ≥ 0.7 | Excellent speedup, draft is well-matched |
| α = 0.5–0.7 | Good speedup, worthwhile |
| α < 0.5 | Poor speedup, consider a different draft model |

*Example: draft 5 tokens per round, average 3 accepted → α = 0.6*

### Speculative token count (γ)

How many tokens the draft model proposes per round. **Configurable.**

- High α → use larger γ (7–10 tokens)
- Low α → use smaller γ (3–5 tokens)

### Acceptance length (τ)

Average tokens accepted per round. Theoretical formula:

```
τ = (1 - α^(γ+1)) / (1 - α)
```

Real-world: 2–3× speedup with α ≥ 0.6, γ ≥ 5.

---

## HuggingFace implementation (complete)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

target_model_name = "google/gemma-7b-it"
draft_model_name  = "google/gemma-2b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Both models must share the same tokenizer
tokenizer = AutoTokenizer.from_pretrained(target_model_name)

target_model = AutoModelForCausalLM.from_pretrained(
    target_model_name, torch_dtype=torch.float16, device_map="auto"
)
draft_model = AutoModelForCausalLM.from_pretrained(
    draft_model_name, torch_dtype=torch.float16, device_map="auto"
)
```

**Baseline (no speculation):**
```python
baseline_output = target_model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)
```

**With speculative decoding:**
```python
speculative_output = target_model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    assistant_model=draft_model,    # enables speculative decoding
    num_assistant_tokens=10
)
```

Colab notebook: https://github.com/balapriyac/data-science-tutorials/blob/main/llm-inference-optimization/speculative-decoding/speculative_decoding_v1.ipynb

---

## When to use (and when not to)

### Good use cases

- Input-grounded tasks: translation, summarization, transcription.
- Greedy decoding (always selecting most likely token).
- Low-temperature sampling (focused, predictable outputs).
- Production deployments where adding GPUs is not an option.

### When not to use

- High-temperature sampling (creative writing) — benefits drop.
- Draft model poorly matched to target.
- Very small target models that already fit easily in memory.

---

## Choosing a good draft model

Four requirements:

1. **Same tokenizer as the target** — non-negotiable.
2. **At least 10× fewer parameters** — otherwise draft generation is slow, defeating the purpose.
3. **Similar training data** — maximizes acceptance rate.
4. **Same architecture family when possible**.

For domain-specific applications: fine-tune a small model on target model outputs. Collect outputs from the target on representative inputs → fine-tune a small model to predict those same outputs. Boosts acceptance rates significantly in production.

---

## Limits of this article (for book context)

- Uses the HuggingFace `assistant_model` API — which handles KV management internally; Layer 14 builds the dual `ModelRunner` + `KVPool` from scratch.
- Rejection sampling shown here is the **stochastic (temperature > 0)** version. Layer 14's `_accept_reject` uses the **greedy (temperature = 0)** special case: accept if `argmax(target) == draft_token`.
- The formula τ = (1 − α^(γ+1))/(1 − α) is the exact geometric series form; Layer 14's `tokens_per_step` counter measures this empirically.
