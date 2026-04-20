# Faster Text Generation with Self-Speculative Decoding (LayerSkip)

**Source:** https://huggingface.co/blog/layerskip
**Authors:** Aritra Roy Gosthipaty, Mostafa Elhoushi, Pedro Cuenca, Vaibhav Srivastav (HuggingFace + Meta)
**Published:** November 20, 2024
**Level:** L2 — Contrast architecture; one-model variant
**Why here:** Before introducing Layer 14's two-`ModelRunner` design, it is worth understanding the alternative: a single model that drafts from its own early layers and verifies with its later layers. LayerSkip is the canonical implementation of this idea. Useful contrast that motivates *why* Layer 14 chose the two-model path.

---

## Summary

**LayerSkip** is a self-speculative decoding technique where a large language model uses its own **early layers** to draft tokens and its **deeper layers** to verify them. No second model required — just one model doing both drafting and verification.

Result: faster generation **and** memory savings (no separate draft weights to load).

---

## How it differs from classic speculative decoding

| Aspect | Classic (two-model) | LayerSkip (self-speculative) |
|--------|---------------------|------------------------------|
| Draft generator | Separate smaller model | Early layers of the target model |
| Verify model | Full target model | Remaining deeper layers |
| VRAM | Two sets of weights | One set of weights |
| Same-tokenizer constraint | Yes (mandatory) | Yes (same model) |
| Training requirement | None (any paired models) | Requires LayerSkip fine-tuning recipe |
| Best use | Any paired model family | Models pre-trained with LayerSkip recipe |

---

## The core mechanism

### Early exit and unembedding

In a standard transformer, the LM head (output projection) is applied only after the **final layer**. LayerSkip modifies the model so the LM head can also be applied at any **intermediate layer** — the process of projecting from an intermediate hidden state to vocabulary probabilities is called **unembedding**.

This requires special training; an intermediate layer's hidden state is not naturally interpretable by a vocabulary head trained on final-layer representations.

### Training recipe (LayerSkip fine-tuning)

Two modifications during training:

1. **Layer dropout:** Progressively higher dropout rates for deeper layers. Trains the model to be less reliant on its later layers. Speeds up training and improves generalization.

2. **Early exit loss:** The total loss is the sum of normalized losses from every intermediate exit layer:
   ```
   L_total = Σ L_exit_i  (normalized per layer)
   ```
   Forces the LM head to learn to interpret outputs from any layer.

Pre-trained checkpoints available for: Llama2 7B/13B/70B, Llama3 8B, Llama3.2 1B, Code Llama.

```
facebook/layerskip-llama2-7B
facebook/layerskip-llama2-13B
facebook/layerskip-llama2-70B
facebook/layerskip-llama3-8B
facebook/layerskip-llama3.2-1B
```

### Inference: self-drafting and self-verification

1. **Self-drafting:** Run only layers 0…E (early exit at layer E). Apply LM head at layer E → draft tokens.
2. **Self-verification:** Take the draft tokens and run the remaining layers E+1…N. The key-value pairs from the early layers are **cached and reused** — verification only computes the later layers.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

early_exit_layer = 4
checkpoint = "facebook/layerskip-llama2-7B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to("cuda")

inputs = tokenizer("Alice and Bob", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, assistant_early_exit=early_exit_layer)
```

---

## Three caches shared between draft and verify

This is the key efficiency advantage:

| Cache type | What it stores | Benefit |
|------------|---------------|---------|
| Shared weights | Same layers 0…E used for both phases | No weight duplication |
| Shared KV cache | Key-value pairs from layers 0…E cached in draft phase | Verification skips recomputing early layers |
| Exit query cache (KVQ) | Query vector from layer E saved | Verification resumes seamlessly from the draft handoff point |

The combination (KVQ cache) reduces both memory overhead and inference latency.

---

## How early to exit? The tradeoff

- **Exit earlier** → draft tokens are faster but less accurate → lower acceptance rate.
- **Exit later** → draft tokens are more accurate but slower → closer to full model speed.

The optimal early exit layer is a hyperparameter that depends on model size and task domain.

### Benchmark (summarization, A100 80GB)

| Model | Method | Time/output (s) | Efficiency vs. baseline |
|-------|--------|----------------|------------------------|
| Llama3 8B | Standard spec (Llama-3.2-1B draft) | 29.08 | 1.53× |
| Llama3 8B | Standard spec (Llama-3.2-3B draft) | 28.73 | 1.00× (break-even) |
| Llama3 8B | **LayerSkip early exit @ layer 4** | **28.87** | **1.83×** |
| Llama2 70B | Standard spec (Llama-2-13B draft) | 112.97 | 2.44× |
| Llama2 70B | **LayerSkip early exit @ layer 10** | **113.2** | **2.06×** |

LayerSkip is faster than two-model speculative decoding for most model sizes. The 70B exception may reflect insufficient LayerSkip training tokens (328M vs 52B for 7B model).

---

## Why Layer 14 chose two-model instead

LayerSkip requires **re-training the target model** with the LayerSkip recipe. Layer 14 prioritizes using off-the-shelf models (any quantized small model + any large target) without retraining. The two-model approach:

- Works with any model family combination (just same tokenizer required).
- Allows using GPTQ-quantized draft models from Layer 13.
- Has no training dependency.

LayerSkip's advantage: **zero additional VRAM for the draft model** and **shared KV cache**. Layer 14's `DraftReq` mirrors this principle — but across two physically separate model pools rather than within one model.

> **Connection:** LayerSkip's "shared KV cache" idea is implemented in Layer 14 as KV mirroring (`lesson/03_draft_kv_mirroring.md`). The invariant that draft and target have seen the same prefix is the same; it's just realized differently.

---

## Related self-speculative approaches

- **Draft & Verify**: skips pre-determined attention and feed-forward layers.
- **MagicDec**: uses a subset of the KV cache as the "early exit" signal, useful for long-context.
- **Jacobi Decoding** and **Lookahead Decoding**: use random or n-gram "guess tokens" as draft.

---

## Limits of this article (for book context)

- Requires LayerSkip checkpoints; not general-purpose for arbitrary model pairs.
- HuggingFace transformers has only implemented "Shared Weights" (not KVQ cache yet as of Nov 2024).
- The early exit layer needs manual tuning per model and domain.
- Production systems (SGLang, vLLM) have limited LayerSkip support relative to EAGLE/two-model approaches.
