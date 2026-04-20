# Faster Assisted Generation with Dynamic Speculation

**Source:** https://huggingface.co/blog/dynamic_speculation_lookahead
**Authors:** Jonathan Mamou, Oren Pereg, Joao Gante, Lewis Tunstall, Daniel Korat, Nadav Timor, Moshe Wasserblat (Intel Labs + HuggingFace)
**Published:** October 8, 2024
**Level:** L2 — "Why static N is wrong; how dynamic N helps"
**Why here:** Directly tied to `lesson/07_statistics.md`. The `acceptance_rate` and `tokens_per_step` counters in Layer 14 are the empirical signals this paper formalizes. Shows why a fixed `num_spec_tokens` is suboptimal and proves an oracle-guided dynamic approach is measurably better.

---

## Summary

The **speculation lookahead** (SL) — how many draft tokens are generated per round — is typically a static constant. But the optimal SL varies token by token. This blog introduces **dynamic speculative decoding**: adjusting SL at each iteration based on the draft model's confidence. Now default behavior in HuggingFace Transformers ≥ 4.45.0.

---

## The problem with static speculation

Standard speculative decoding generates a **fixed N draft tokens** per round. An **oracle** (which knows which draft tokens will be accepted before running the target) can compute the optimal N at each step.

### Empirical evidence

From a code generation example (MBPP dataset):

| Approach | Target forward passes | Draft forward passes |
|----------|----------------------|---------------------|
| Static SL (N=5) | 38 | 192 |
| Oracle SL | 27 | 129 |

The oracle performs 29% fewer target passes and 33% fewer draft passes. Oracle SL shows **high variance** across iterations — proving that a fixed N is always leaving performance on the table.

---

## Three SL scheduling approaches

| Approach | How it works |
|----------|-------------|
| **Constant** (Leviathan et al.) | Fixed `num_assistant_tokens` throughout generation |
| **Heuristic** | Increase N if all draft tokens accepted; decrease if any rejected |
| **Dynamic** (this paper) | After each draft token, check draft model confidence. If confidence < `assistant_confidence_threshold`, halt early and send to target |

### Dynamic approach: how it decides

After generating each draft token, compute the **softmax of the draft model's logits** for that token. If the softmax probability (confidence) falls below the configured threshold, **stop drafting immediately** — even if the maximum `num_assistant_tokens` hasn't been reached — and send what we have to the target for verification.

```python
# HuggingFace API for dynamic speculation (Transformers ≥ 4.45.0)
# No code changes needed — dynamic is now the default

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.4b-deduped").to("cuda")
assistant_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m-deduped").to("cuda")

outputs = model.generate(**inputs, assistant_model=assistant_model)
```

**Tuning parameters:**
```python
# Confidence threshold (default: 0.4 is optimal for most cases)
assistant_model.generation_config.assistant_confidence_threshold = 0.4

# Maximum tokens drafted per round (even if confident)
assistant_model.generation_config.num_assistant_tokens = 20

# Revert to heuristic or constant:
assistant_model.generation_config.num_assistant_tokens_schedule = 'heuristic'
assistant_model.generation_config.assistant_confidence_threshold = 0
assistant_model.generation_config.num_assistant_tokens = 5
```

---

## Benchmark results

All experiments: greedy decoding (temperature = 0), RTX 4090.

| Target | Draft | Task | Heuristic speedup | Dynamic speedup |
|--------|-------|------|-------------------|-----------------|
| `facebook/opt-6.7b` | `facebook/opt-125m` | summarization | 1.82× | **2.71×** |
| `facebook/opt-6.7b` | `facebook/opt-125m` | open-ended generation | 1.23× | **1.59×** |
| `Salesforce/codegen-6B-mono` | `Salesforce/codegen-350M-mono` | code (python) | 0.89× ❌ slowdown | **1.09×** |
| `google/flan-t5-xl` | `google/flan-t5-small` | summarization | 1.18× | **1.31×** |
| `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.2-1B` | summarization | 1.00× (no gain) | **1.52×** |
| `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.2-1B` | open-ended | 1.00× | **1.18×** |
| `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.2-1B` | code (python) | 1.09× | **1.15×** |

> Dynamic speculation is **always at least as good** as heuristic, and often dramatically better. Even cases where heuristic caused slowdown (codegen-6B), dynamic recovers a speedup.

---

## Why this matters for Layer 14

Layer 14 implements **fixed** `num_spec_tokens` — equivalent to the "constant" approach in this paper. The `acceptance_rate` and `tokens_per_step` counters in `lesson/07_statistics.md` are exactly what you'd feed into a dynamic scheduler:

- If `acceptance_rate` drops, reduce `num_spec_tokens`.
- If `acceptance_rate` is high, increase `num_spec_tokens`.

The heuristic scheduler does this reactively (one round at a time). The dynamic scheduler does it *within* a round (token-by-token confidence check). Both use the same empirical signal that Layer 14 measures.

A natural Layer 14 extension: **implement dynamic speculation** by adding a confidence threshold check inside the draft loop in `spec_runner.py`.

---

## Related paper

> Mamou et al. (2024), *Accelerating Speculative Decoding using Dynamic Speculation Length*, arXiv:2405.04304

DISCO: uses a learned **classifier** (instead of a threshold on softmax probability) to decide whether the draft model should continue or hand off to the target. More accurate but requires training.

---

## Limits of this article (for book context)

- Benchmarks are on RTX 4090; production serving on A100/H100 may show different patterns due to memory bandwidth differences.
- The confidence threshold is a global hyperparameter; better approaches (DISCO) train per-model classifiers.
- Layer 14 does not implement confidence-based early stopping — this is explicitly an "extension" direction mentioned in `lesson/09_next_steps.md`.
