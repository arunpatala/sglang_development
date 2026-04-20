# HuggingFace TGI: Speculation Conceptual Guide

**Source:** https://huggingface.co/docs/text-generation-inference/main/en/conceptual/speculation
**Level:** L2 — Definitions + variants side-by-side
**Why here:** Authoritative TGI documentation covering three speculative methods (Medusa, N-gram, draft model) in plain language. Best single place to anchor the vocabulary of "speculation" before Layer 14 introduces its specific two-ModelRunner design.

---

## Summary

Speculative decoding, assisted generation, Medusa, and others are different names for the same core idea: generate tokens before the large model runs, then check if they are valid. If guesses are correct enough, this achieves 2–3× faster inference (sometimes more, especially for code) without a quality loss.

The reason it works: **LLMs are usually memory-bound, not compute-bound.** Verifying multiple draft tokens in one forward pass costs roughly the same as generating one token — so you effectively get several tokens for the price of one.

---

## Three methods in TGI

### 1. Medusa

[Paper: arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774)

Medusa adds **multiple fine-tuned LM heads** to the existing model. Each head predicts a different token offset (1 step ahead, 2 steps, 3 steps, …). A single forward pass through the model plus all Medusa heads produces several candidate tokens simultaneously.

The original model's weights stay frozen; only the extra heads are fine-tuned.

**Pre-built Medusa models (HuggingFace):**

| Model | HF Hub path |
|-------|-------------|
| Gemma 7B-it | `text-generation-inference/gemma-7b-it-medusa` |
| Mixtral 8×7B | `text-generation-inference/Mixtral-8x7B-Instruct-v0.1-medusa` |
| Mistral 7B v0.2 | `text-generation-inference/Mistral-7B-Instruct-v0.2-medusa` |

TGI usage: point to a Medusa-enabled model checkpoint — everything loads automatically.

To create your own Medusa heads: see the [Train Medusa guide](https://huggingface.co/docs/text-generation-inference/main/en/basic_tutorials/train_medusa).

### 2. N-gram

When you don't have a Medusa model or resources to fine-tune, use `--speculate 2` in TGI.

N-gram works by **finding matching tokens in the previous sequence** and using those as speculation for new tokens.

Example:
> If the tokens "np.mean" appear multiple times in the sequence, the model can speculate that the next continuation of "np." is probably also "mean".

Best for: **code or highly repetitive text**.
Not beneficial when speculation misses too much (creative, open-ended).

TGI flag: `--speculate 2` (the number sets the lookahead window size).

### 3. Draft model (classic two-model)

A separate smaller model generates candidate tokens; the target model verifies them. The TGI conceptual docs link to a [detailed explanation of assisted generation](https://huggingface.co/blog/assisted-generation) for the full mechanism.

---

## Why this terminology matters

| Term | Meaning |
|------|---------|
| Speculative decoding | General approach: guess multiple tokens, verify in one pass |
| Assisted generation | HuggingFace Transformers name for the same thing |
| Draft model | The small, fast generator |
| Target model | The large, accurate verifier |
| Speculation lookahead | Number of tokens the draft generates per round |
| Acceptance rate | Fraction of draft tokens the target accepts |

---

## Mapping to Layer 14

| TGI concept | Layer 14 equivalent |
|-------------|---------------------|
| Draft model | `spec_runner.draft_model_runner` (a `ModelRunner` with quantized weights) |
| Target model | `spec_runner.target_model_runner` |
| Medusa heads | Not in Layer 14 (separate design; Medusa integrates heads into one model) |
| N-gram | Not in Layer 14 (requires no second model; Layer 14 always uses a real draft model) |
| Speculate N | `num_spec_tokens` in `SpecRunner` — same concept |
| Acceptance rate | `spec_runner.acceptance_rate` |

---

## Limits of this article (for book context)

- TGI conceptual docs are intentionally brief — they link to the Medusa paper and the assisted-generation blog for depth.
- They do **not** explain the KV cache implications of running two models simultaneously (which is the main engineering challenge Layer 14 addresses).
- They do **not** explain the rejection-sampling acceptance criterion in detail (see `01_mlmastery_practitioner_guide.md` for that).
- For production variant comparison (EAGLE vs. N-gram vs. Suffix in real benchmarks) see `02_jarvislabs_speculative_decoding_vllm.md`.
