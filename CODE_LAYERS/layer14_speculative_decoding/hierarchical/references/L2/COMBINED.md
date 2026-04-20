# Speculative Decoding — L2 Combined Reference

**What this file is:** A synthesis of all 9 L2 articles for speculative decoding, combined into a single progressive reference. Every concept from every source is included. No topic is silently dropped — anything that breaks the main narrative flow is collected in the appendices.

**Reading arc:** Why LLMs are slow → the key insight → the core loop → worked examples → mathematical guarantee → key metrics → speedup benchmarks → choosing a draft model → when to use → terminology → the five variants → dynamic scheduling → self-speculative decoding → extensions panorama → production context.

**Sources synthesized:**
- `01` — ML Mastery practitioner guide: α, γ, τ metrics; HuggingFace implementation; worked probability table
- `02` — Jarvis Labs / vLLM: five-technique comparison; real benchmark tables; "quick brown fox" worked example
- `03` — HuggingFace TGI docs: vocabulary anchor; Medusa / N-gram / draft model terminology
- `04` — HuggingFace dynamic speculation (Intel + HF): static N problem; dynamic scheduling; benchmarks
- `05` — HuggingFace LayerSkip (Meta + HF): self-speculative decoding; three caches; why two-model was chosen
- `06` — Adaptive ML visualized: green/red probability geometry; distribution preservation proof; speedup calculation
- `07` — Pierce Freeman: multi-GPU bottleneck; best/worst/average case; "you only need the draft right some of the time"
- `08` — Brenndoerfer interactive guide: fully quantified memory analysis; causal masking; speedup table; correction distribution code
- `09` — Data Processing Club: SpecJudge/SpecNormalize; full proof; bigram result; extensions panorama

---

## 1. Why LLMs Are Slow

### 1.1 The sequential dependency chain

Large language models generate text autoregressively: compute logits → sample → append → repeat. Each token depends on every token before it. Token t+1 cannot begin until token t is complete.

You can't predict token 5 without first knowing token 4. The sentence "The scientist discovered a new species" is six tokens — the model runs six sequential forward passes. A 70B parameter model taking 50ms per token needs **5 full seconds** to generate 100 tokens, waiting for each step to complete before starting the next.

This strict dependence chain prevents full GPU parallelism even on the fastest hardware available.

*(Sources: 01, 06, 07, 09)*

### 1.2 The memory bandwidth bottleneck

Each forward pass requires loading the **entire model's weights from GPU memory** into compute cores. Not the relevant section — all of them. For large models, this means loading terabytes of data per generated token. The GPU's compute cores sit idle while waiting for data. This is being **memory-bound**.

The numbers, fully quantified (08):

```python
# A 70B model on A100 80GB
model_params     = 70e9
bytes_per_param  = 2          # FP16
model_size_bytes = 140e9      # 140 GB

memory_bandwidth = 2e12       # 2 TB/s (A100)
compute_tflops   = 312e12     # 312 TFLOPS FP16

load_time        = model_size_bytes / memory_bandwidth  # ~70 ms per token
compute_time     = (2 * model_params) / compute_tflops  # ~0.45 ms per token

# Arithmetic intensity = 1 FLOP/byte
# GPU's compute sweet spot: ~150 FLOPs/byte
```

```
Model size:                    140 GB
Memory load time per token:    70.0 ms
Compute time per token:         0.449 ms
Arithmetic intensity:           1.0 FLOPs/byte
Memory loading is 156× slower than compute
GPU sits 96% idle, waiting for data — not computing
```

Adding more CUDA cores changes nothing. The bottleneck is memory movement, not arithmetic.

*(Source: 08)*

### 1.3 Multi-GPU coordination overhead

As models scale, they're served across multiple GPUs linked with NVLink or InfiniBand:

- An A100 80GB fits roughly 40B parameters in FP16.
- Anything larger shards across multiple GPUs.
- Every token generation requires: coordinate across all GPUs → run forward pass → repeat.

A 70B model takes roughly 10× longer per token than a 7B model — more transformer blocks, more floating-point arithmetic, more inter-GPU synchronization overhead.

*(Source: 07)*

### 1.4 The asymmetry that unlocks the solution

Here is the critical observation:

> **Verifying N tokens takes one forward pass. Generating N tokens takes N forward passes.**

A language model can score an entire sequence in parallel — computing the probability distribution at every position simultaneously. The transformer processes all positions at once. If we could guess the next several tokens correctly, we could verify all of them in a single pass.

This asymmetry is what speculative decoding exploits. The constraint is autoregressive generation, not autoregressive verification.

*(Sources: 02, 06, 07)*

---

## 2. The Key Insight: Not All Tokens Are Equally Hard

Some next-token choices are obvious — even trivial:

> *"Not all tokens are created equal. Some next-token choices are obvious, even trivial. If I write 'The capital of France is...', you know the next token should be 'Paris' without needing a 70B parameter model to figure it out."* — Pierce Freeman (07)

From ML Mastery (01):

> *"After 'The discovery was made in the', predicting 'Amazon' is easier because it appeared earlier in the context. But predicting 'species' after 'The scientist discovered a new' requires understanding semantic context."*

**The implication:** Large models are better mainly in difficult cases. For easy tokens — function words, repeated phrases, predictable continuations — a much smaller model can approximate the large model's predictions accurately.

This is a clever observation that works because many token sequences in natural language are predictable enough that a small model gets them right. And when the small model is wrong, you haven't lost much — you were going to need the big model anyway.

*(Sources: 01, 07, 09)*

---

## 3. The Core Loop: How Speculative Decoding Works

Two models:
- The **draft model** — small, fast, approximate. (e.g., Llama-3.2-1B as draft for Llama-3.1-8B)
- The **target model** — large, accurate, the final authority. (e.g., Llama-3.1-8B or Llama-3.3-70B)

Each generation cycle runs four phases:

### 3.1 Phase 1 — Draft generation

The draft model autoregressively generates K candidate tokens (K typically 4–8, configurable). Since the draft model is much smaller, it can generate K tokens faster than the large model could generate even one.

The draft model does not know whether its candidates will be accepted. It makes its best prediction, fast.

*(Sources: 01, 02, 06, 07, 08, 09)*

### 3.2 Phase 2 — Parallel verification

The target model takes the original input sequence **plus all K draft tokens** and runs a single forward pass over the whole thing simultaneously.

This is the key efficiency: one expensive target-model operation checks everything the draft proposed. The cost of verifying 8 draft tokens is barely more than verifying 1 — because the transformer evaluates all positions in parallel.

The KV cache holds previously computed context, so only the new draft positions incur fresh computation.

*(Sources: 02, 06, 07, 08)*

#### Why one pass works: causal masking

This is the architectural mechanism that makes parallel verification possible.

When the target receives `[prompt, draft₁, draft₂, draft₃, draft₄]`, the **causal attention mask** ensures each position only attends to preceding tokens. This creates independent probability computations at every position — all done simultaneously in one forward pass:

- Position `draft₁` computes `p(draft₁ | prompt)`
- Position `draft₂` computes `p(draft₂ | prompt, draft₁)`
- Position `draft₃` computes `p(draft₃ | prompt, draft₁, draft₂)`
- ...

```python
def parallel_verification(target_model, input_ids, draft_tokens):
    full_sequence = torch.cat([input_ids, draft_tokens], dim=-1)

    with torch.no_grad():
        logits = target_model(full_sequence).logits

    num_input = input_ids.shape[-1]
    num_draft  = draft_tokens.shape[-1]

    # Logits at positions just before each draft token
    verification_logits = logits[:, num_input - 1 : num_input + num_draft - 1, :]
    probs = F.softmax(verification_logits, dim=-1)

    # Extract probability of each draft token at each position
    draft_probs = [probs[0, i, draft_tokens[0, i].item()].item()
                   for i in range(num_draft)]
    return draft_probs, probs
```

The same causal mask that enables efficient transformer training enables efficient speculative verification. Same mechanism, dual purpose.

*(Source: 08)*

### 3.3 Phase 3 — Accept or reject

For each draft token position, compare:
- `P(target)` — what the target assigns to the drafted token at that position
- `P(draft)` — what the draft model assigned to the same token

The rule:
- If `P(target) ≥ P(draft)`: **accept** — the target agrees, or is even more confident than the draft.
- If `P(target) < P(draft)`: **accept with probability `P(target)/P(draft)`** — the draft was overconfident; accept probabilistically. At full rejection, discard this token and everything after it.

Tokens are accepted from left to right. At the first rejection, the chain stops.

*(Sources: 01, 02, 06, 07, 08, 09)*

### 3.4 Phase 4 — Correction and restart

When a token is rejected, you **cannot simply resample from `p_target`** — that would double-count tokens the draft already had a chance to propose. Instead, resample from the **residual distribution** (see §5 for the math), then restart the draft cycle from the new position.

A bonus token is always generated at the end of each cycle — either the target's K+1'th prediction (if all draft tokens were accepted) or the correction token at the rejection point. The system always makes forward progress.

*(Sources: 06, 08, 09)*

### 3.5 Best / worst / average case

**Best case:** All K draft tokens accepted. Target generates K tokens (draft-accepted) plus 1 bonus token = **K+1 tokens in one target forward pass**.

```
If K=4 and all accepted → 5 tokens from one target pass
```

**Worst case:** The very first draft token is wrong. Reject it and all subsequent tokens; keep only the target's correction token.

```
→ generated 1 token but ran both small + large model
→ slightly slower than standard autoregressive generation
```

**Average case:** Some prefix of the draft tokens are accepted.

```
If 2 out of 4 tokens accepted on average → ~2× speedup overall
```

**The key insight:** You only need the small model to be right *some* of the time for this to be worthwhile.

> *"Even if the small model is wrong 60% of the time, you can still see significant speedups because of the parallel verification."* — Pierce Freeman (07)

*(Sources: 06, 07)*

---

## 4. Worked Examples

### 4.1 "The quick brown fox" — token-level trace

*(Source: 02, Jarvis Labs)*

**Setup:** Draft model sees "The quick brown fox" and proposes 5 tokens: *"jumps over the lazy dog"* (the correct phrase is "jumps over the log")

**Verification:** Target model runs one single pass over original + 5 draft tokens.

**Result:**

| Token | Decision |
|-------|----------|
| "jumps" | ACCEPTED |
| "over" | ACCEPTED |
| "the" | ACCEPTED |
| "lazy" | REJECTED — target calculated "log" was correct |

**Outcome:** 3 tokens generated in the time it normally takes for 1 → **3× speedup** for this step.

**Correction:** Target replaces "lazy" with "log". Draft starts again from "…jumps over the log".

---

### 4.2 "Discovered a breakthrough" — probability table

*(Source: 01, ML Mastery)*

Draft model proposed: **"discovered a breakthrough"**

| Token | P(draft) | P(target) | Decision |
|-------|----------|-----------|----------|
| "discovered" | 0.6 | 0.8 | ACCEPT (0.8 ≥ 0.6) |
| "a" | 0.7 | 0.75 | ACCEPT (0.75 ≥ 0.7) |
| "breakthrough" | 0.5 | 0.2 | REJECT; target generates "new" |

**Result:** 3 tokens from one target forward pass (2 accepted + 1 corrected).

---

### 4.3 Speedup calculation

*(Source: 06, Adaptive ML)*

Suppose:
- Target model: 50ms per forward pass
- Draft model: 5ms per forward pass
- Draft length K = 5
- Average acceptance: 3 tokens per round

```
Cost per iteration = 50ms (target) + 5 × 5ms (drafts) = 75ms
Tokens generated   = 3 accepted + 1 resampled = 4 tokens
Effective rate     = 75ms / 4 = 18.75ms per token
Baseline           = 50ms per token
Speedup            = 50 / 18.75 = 2.67×
```

**The optimal K tradeoff:** Too few drafts wastes verification parallelism. Too many wastes effort on tokens that will be rejected. The right K is governed by the acceptance rate (see §6).

---

## 5. The Mathematical Guarantee: Output Is Exactly p_target

This is the central claim that makes speculative decoding production-worthy. The output is not an approximation — it is statistically identical to pure autoregressive sampling from the target model.

### 5.1 The probability geometry

*(Source: 06, Adaptive ML — the clearest visual treatment)*

At each token position, compare `p_target` vs `p_draft`:

```
P(accept) = min(1, p_target / p_draft)
```

Visually:
- **Green region** = `min(p_target, p_draft)` — guaranteed acceptance mass.
- **Red excess** = where the draft overestimated (`p_draft > p_target`) — this is rejected proportionally.

When `p_target ≥ p_draft`: no red region — **always accept**.
When `p_draft > p_target`: red excess appears — accept with probability `p_target / p_draft`.

---

### 5.2 SpecJudge and SpecNormalize

*(Source: 09, Data Processing Club — the most memorable framing)*

The naive approach (accept only exact matches) fails even when draft = target perfectly — with a 100-token vocabulary, the probability of exact match is only 1%. A probabilistic rule is needed.

**SpecJudge** (the acceptance decision):

For each position i, given draft probability `q_d(xᵢ)` and target probability `p_t(xᵢ)`:

```
(1) If q_d(xᵢ) ≤ p_t(xᵢ): Accept always.

(2) Otherwise: Accept with probability p_t(xᵢ) / q_d(xᵢ).
    (Sample r ~ Uniform(0,1); accept if r < p_t / q_d)
```

**SpecNormalize** (the correction on rejection):

When a token is rejected, sample a replacement from:

```
p_resid(x) = normalize( max(0, p_t(x) - q_d(x)) )
```

---

### 5.3 The full proof

*(Source: 09)*

Define D = {x | q_d(x) > p_t(x)} — tokens where the draft overestimates.

**Case 1: x ∈ D** (draft overestimates this token)
- Generated by draft with probability q_d(x)
- Accepted via SpecJudge with probability p_t(x) / q_d(x)
- Net probability = q_d(x) × (p_t(x) / q_d(x)) = **p_t(x)** ✓

**Case 2: x ∉ D** (target assigns equal or higher probability)
- Accepted whenever generated by draft: probability q_d(x)
- When a token in D is rejected (rejection mass C = Σ_D (1 - p_t/q_d) dq_d), sample x from p_resid
- Net additional probability: C × (p_t(x) - q_d(x)) / C = p_t(x) - q_d(x)
- Total: q_d(x) + (p_t(x) - q_d(x)) = **p_t(x)** ✓

In both cases, the token is emitted with exactly the target's probability. The acceptance criterion is self-correcting by design.

**Key property:** Standard rejection sampling may reject and re-sample repeatedly. SpecJudge guarantees resolution in **at most one round** — accept via (1), accept via (2), or reject and sample once from SpecNormalize. No re-sampling loop.

*(Sources: 06, 09)*

---

### 5.4 The correction distribution: full code

*(Source: 08, Brenndoerfer)*

```python
def accept_probability(p_target, q_draft):
    return min(1.0, p_target / q_draft)

def correction_distribution(p_target, q_draft):
    residual = np.maximum(0, p_target - q_draft)   # element-wise
    total = np.sum(residual)
    if total > 0:
        return residual / total      # normalized residual
    else:
        return p_target              # fallback: full target distribution
```

Acceptance scenarios:

| Scenario | p_target | q_draft | P(accept) |
|----------|----------|---------|-----------|
| Close agreement | 0.15 | 0.18 | 0.83 |
| Target prefers more | 0.25 | 0.10 | 1.00 (always accept) |
| Draft overestimates | 0.05 | 0.20 | 0.25 |

Accepted samples + residual resamples = **exactly `p_target`**. No approximation; no accuracy tradeoff to justify to stakeholders.

---

## 6. Key Metrics: α, γ, τ

*(Source: 01, ML Mastery — the canonical treatment)*

### 6.1 Acceptance rate (α)

```
α = (number of accepted tokens) / (total draft tokens proposed)
```

| Range | Interpretation |
|-------|---------------|
| α ≥ 0.7 | Excellent speedup; draft is well-matched |
| α = 0.5–0.7 | Good speedup; worthwhile to deploy |
| α < 0.5 | Poor speedup; consider a different draft model |

*Example: draft 5 tokens per round, average 3 accepted → α = 0.6*

### 6.2 Speculative token count (γ)

How many tokens the draft model proposes per round. **Configurable at inference time.**

- High α → use larger γ (7–10 tokens)
- Low α → use smaller γ (3–5 tokens)

The relationship between γ and speedup is not linear — see τ below.

### 6.3 Acceptance length (τ)

Average number of tokens accepted per round. Theoretical formula:

```
τ = (1 - α^(γ+1)) / (1 - α)
```

This is a geometric series formula — it captures how acceptance probability compounds across positions.

Real-world: **2–3× speedup** with α ≥ 0.6 and γ ≥ 5.

### 6.4 Speedup table across acceptance rates

*(Source: 08 — target=100ms, draft=15ms, K=5)*

| Acceptance Rate | Expected Tokens per Round | Speedup |
|----------------|--------------------------|---------|
| 0.5 | 1.97 | 1.12× |
| 0.6 | 2.38 | 1.36× |
| 0.7 | 2.94 | 1.68× |
| 0.8 | 3.69 | 2.11× |
| 0.9 | 4.69 | 2.68× |

> Acceptance rate is the primary lever. Even modest rates (0.6–0.7) yield meaningful speedups. Below ~0.5 the overhead starts to dominate.

The efficiency argument:

| Action | Cost |
|--------|------|
| Generate 1 draft token (small model) | ~5% of one target forward pass |
| Verify 5 draft tokens (target model) | ~same as verifying 1 (one forward pass) |

So: generate 5 drafts cheaply (~0.25× cost) + verify all 5 once (1.0× cost) = 1.25× total cost.
If 4/5 are accepted = 4 tokens for 1.25× → **3.2× speedup**.

*(Source: 08)*

---

## 7. Speedup Results and Benchmarks

### 7.1 Original paper results: T5-XXL

*(Source: 09, Data Processing Club)*

Leviathan et al. (2023) — T5-XXL (11B parameters) + T5-Small (77M parameters):

| Task | γ | Acceptance rate | Speedup |
|------|---|-----------------|---------|
| Translation | 5 | 53% | 2.3× |
| Summarization | 7 | 75% | 3.4× |

### 7.2 The bigram model result

*(Source: 09 — the most surprising finding in the field)*

Using a **bigram model** (no GPU, negligible cost) as the draft:
- Acceptance rate: ~20%
- γ = 3
- Speedup: **1.25×**

> *"Even a 'dumb' draft model provides a speed boost because verification is parallel."*

This is the most important existence proof: even when the draft is nearly useless, the parallel verification step provides a non-trivial speedup. The floor is not zero.

### 7.3 vLLM benchmarks

*(Source: 02, Jarvis Labs — five techniques, two models, two datasets)*

**Setup:**

| Model | GPU | Datasets |
|-------|-----|---------|
| Llama-3.1-8B | 1× L40S (48GB) | ShareGPT (1,000 prompts) + SWE-bench Lite (300 samples) |
| Llama-3.3-70B | 2× H200 (141GB each) | Same |

**Llama-3.1-8B — General chat (ShareGPT):**

| Technique | tok/s | Speedup |
|-----------|-------|---------|
| Baseline | 421 | 1.00× |
| N-gram | 491 | 1.17× |
| Suffix | 585 | 1.39× |
| EAGLE-3 | 589 | 1.40× |
| **EAGLE** | **601** | **1.43×** |

**Llama-3.1-8B — Coding (SWE-bench):**

| Technique | tok/s | Speedup |
|-----------|-------|---------|
| Baseline | 370 | 1.00× |
| EAGLE-3 | 379 | 1.03× |
| EAGLE | 398 | 1.08× |
| N-gram | 406 | 1.10× |
| **Suffix** | **534** | **1.45×** |

**Llama-3.3-70B — General chat (ShareGPT):**

| Technique | tok/s | Speedup |
|-----------|-------|---------|
| Baseline | 343 | 1.00× |
| EAGLE (standard) | 351 | 1.02× |
| N-gram | 385 | 1.12× |
| Suffix | 455 | 1.33× |
| **EAGLE-3** | **537** | **1.57×** |

**Llama-3.3-70B — Coding (SWE-bench):**

| Technique | tok/s | Speedup |
|-----------|-------|---------|
| Baseline | 375 | 1.00× |
| EAGLE (standard) | 377 | 1.01× |
| N-gram | 416 | 1.11× |
| Suffix | 522 | 1.39× |
| **EAGLE-3** | **601** | **1.60×** |

**Key finding:** For small models (8B), EAGLE wins on chat; Suffix wins on code. For large models (70B+), EAGLE-3 wins everywhere.

### 7.4 Real-world range

From multiple sources: **2–4× speedup** with a well-matched draft model. Typical expectation: 2–3×. Factors:

| Factor | Effect on speedup |
|--------|------------------|
| Text predictability (code, structured output) | Higher |
| Creative writing / diverse topics | Lower |
| Better draft-target alignment | Higher acceptance rate → more speedup |
| Larger size ratio (70B/7B vs 7B/1B) | 70B/7B gains more |
| GPU memory bandwidth | Speedup tied to memory-bound nature |

*(Sources: 01, 07, 02)*

---

## 8. Choosing the Right Draft Model

### 8.1 The four requirements

*(Source: 01, ML Mastery)*

1. **Same tokenizer as the target** — non-negotiable. Draft and target must share vocabulary. Otherwise tokens cannot be directly compared in the accept/reject step.

2. **At least 10× fewer parameters** — otherwise draft generation is slow, defeating the purpose. If the draft is too large, drafting itself dominates the round cost.

3. **Similar training data** — maximizes acceptance rate. The more the draft's training distribution resembles the target's, the more often it predicts what the target would have predicted.

4. **Same architecture family when possible** — Llama-3.2-1B as draft for Llama-3.1-8B; Llama-3.1-8B as draft for Llama-3.3-70B. Cross-family pairings need benchmarking.

### 8.2 The size ratio rule

*(Sources: 07, 08)*

Draft model should be **10–20× smaller** than the target:
- A 7B draft paired with a 70B target is a common production pairing.
- A 1B draft paired with an 8B target also works well (same family).
- If the draft is too small, acceptance rate drops; if too large, draft cost dominates.

### 8.3 Domain-specific fine-tuning

*(Sources: 01, 07, 08)*

A generic off-the-shelf draft model delivers real gains. A draft model **fine-tuned on target model outputs** in your deployment domain delivers significantly more:

- Collect outputs from the target on representative inputs.
- Fine-tune a small model to predict those same outputs.
- Boosts acceptance rates significantly in production.

> Same-family model pairs (e.g., Llama-3.1-8B draft + Llama-3.3-70B target) work well out of the box. Cross-family pairings need benchmarking to verify acceptance rates.

---

## 9. When to Use Speculative Decoding (and When Not To)

*(Source: 01, 06, 07)*

### Good use cases

- **Input-grounded tasks:** translation, summarization, transcription. The output is constrained by the input — predictable, high acceptance rate.
- **Code generation** — highly repetitive patterns; small models often predict them correctly.
- **Greedy decoding** (always selecting most likely token).
- **Low-temperature sampling** (focused, predictable outputs).
- **Production deployments** where adding GPUs is not an option.
- When the **size ratio** between target and draft is large (70B/7B gains more than 7B/1B).

### When benefits drop

- **High-temperature sampling** — creative writing with diverse token choices; acceptance rate falls because the draft can't predict random samples from a flat distribution.
- **Draft model poorly matched to target** — different training distribution, different instruction-following behavior, consistently rejected.
- **Very small target models** that already fit easily in memory — the memory-bound bottleneck is less severe; the relative cost of running a draft may not be worth it.

> *"Even if the small model is wrong 60% of the time, you can still see significant speedups because of the parallel verification."* — Pierce Freeman (07)

The minimum useful acceptance rate is roughly 50%. Below that, the overhead of draft generation dominates.

---

## 10. Vocabulary: All the Names for the Same Idea

*(Source: 03, HuggingFace TGI docs)*

Different frameworks and papers use different names for the same concept. This table anchors the vocabulary:

| Term | Meaning |
|------|---------|
| Speculative decoding | General approach: guess multiple tokens ahead, verify in one pass |
| Assisted generation | HuggingFace Transformers name for the same mechanism |
| Speculation | The drafting phase |
| Draft model | The small, fast candidate generator |
| Target model | The large, accurate verifier — the final authority |
| Speculation lookahead / γ | Number of tokens the draft generates per round |
| Acceptance rate / α | Fraction of draft tokens the target accepts |
| Acceptance length / τ | Average tokens committed per round |

HuggingFace, vLLM, TGI, SGLang, and TensorRT all implement this mechanism; the parameter names differ slightly across APIs.

---

## 11. The Five Speculative Decoding Variants

*(Source: 02, Jarvis Labs; 03, HuggingFace TGI)*

### 11.1 Classic draft model (two-model)

A separate, smaller model generates candidate tokens; the target verifies them. Best for open-ended text generation. Requires extra VRAM for the draft model weights. Same tokenizer is mandatory.

**Pre-requisite:** both models must share the same vocabulary.

*This is what Layer 14 implements.*

### 11.2 N-gram matching

Finds patterns in previously generated text and predicts repetitions. If the token sequence "np.mean" appeared earlier, seeing "np." suggests predicting "mean".

- No VRAM overhead — no second model.
- Best for code or highly repetitive text.
- Performs poorly on diverse, creative tasks.
- Available in TGI with `--speculate 2` (sets lookahead window size).

### 11.3 Suffix decoding

Like N-gram but uses **dual suffix trees** (per-request + global) for much faster pattern lookup. CPU-based; no GPU overhead.

- **Local tree:** patterns from the current request.
- **Global tree:** patterns from all previous requests across the server.
- Uses greedy expansion based on path confidence: D(N) = D(Parent) × C(N).
- Excellent for agentic loops, code generation, long contexts with repetition.
- Strict VRAM budget? Start here — no training, no extra weights, immediate speedup.

### 11.4 MLP Speculator (IBM Medusa variant)

A multi-headed lightweight MLP that **attaches directly to the target model**, using its context embedding vector. Three heads predict tokens at t+1, t+2, t+3 separately:

- Head 1 predicts T2. Head 2 uses T2 to predict T3. Etc.
- Each head has its own training loss — separate loss curves (t+1 converges faster than t+3).
- Trained in two stages: input alignment (standard text), then output alignment (distillation from target).
- ~1/10th parameters of a full draft model → minimal VRAM overhead.
- 2–3× speedup; high acceptance rate from leveraging target's own context.

### 11.5 EAGLE / EAGLE-2 / EAGLE-3

EAGLE attaches a **lightweight draft head** (1–2 transformer layers) to the target model, reusing the target's internal feature maps.

**Evolution:**
- **EAGLE-1:** single-layer head reusing top-layer features. Limitation: "feature uncertainty" — trained on perfect data but infers from its own noisy predictions.
- **EAGLE-2:** dynamic draft trees that adapt branch structure by confidence. Smarter guessing.
- **EAGLE-3 (current state-of-art):**
  - **Multi-layer fusion:** takes features from Low + Middle + High layers.
  - **Training-Time Test (TTT):** trains on its own predictions (not ground truth), fixing the distribution mismatch problem.
  - Acceptance rate stays flat at 70–80% even deep into the sequence.
  - **Tree attention:** generates a candidate tree, verifies with one target forward pass, prunes invalid branches.

### 11.6 Pre-built Medusa model checkpoints (HuggingFace)

*(Source: 03)*

Medusa adds multiple fine-tuned LM heads to an existing model. Each head predicts a different token offset. Single forward pass through the model + all Medusa heads produces several candidate tokens simultaneously. Original weights stay frozen; only the extra heads are fine-tuned.

Ready-to-use Medusa models:

| Model | HuggingFace Hub path |
|-------|---------------------|
| Gemma 7B-it | `text-generation-inference/gemma-7b-it-medusa` |
| Mixtral 8×7B | `text-generation-inference/Mixtral-8x7B-Instruct-v0.1-medusa` |
| Mistral 7B v0.2 | `text-generation-inference/Mistral-7B-Instruct-v0.2-medusa` |

### 11.7 Decision guide

*(Source: 02)*

| Use case | Best technique | Why |
|----------|---------------|-----|
| General chatbots (small model, 8B) | EAGLE / EAGLE-3 | Learned draft heads predict fluid human conversation better than heuristics |
| Coding (small model, 8B) | Suffix Decoding | Low overhead; code's repetitive patterns favor suffix matching |
| Coding + agents (large model, 70B+) | EAGLE-3 | At scale, EAGLE-3's accurate predictions beat even pattern-matching |
| Strict VRAM budget | Suffix Decoding | No extra weights, no training; "free" speedup |

> **Start with Suffix Decoding.** It's the low-hanging fruit — no training, no extra VRAM, works immediately. For 70B+ models needing maximum throughput, invest in EAGLE-3.

### 11.8 vLLM configuration examples

*(Source: 02)*

**N-gram:**
```bash
--speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'
```

**Suffix:**
```bash
--speculative-config '{"method": "suffix"}'
```

**EAGLE-3 (8B):**
```bash
--speculative-config '{"method":"eagle3", "model":"yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "draft_tensor_parallel_size":1, "num_speculative_tokens":2}'
```

**HuggingFace Transformers (classic draft model):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

target_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b-it", torch_dtype=torch.float16, device_map="auto"
)
draft_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it", torch_dtype=torch.float16, device_map="auto"
)

speculative_output = target_model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    assistant_model=draft_model,    # enables speculative decoding
    num_assistant_tokens=10
)
```

---

## 12. Static vs Dynamic Speculation: The γ Scheduling Problem

*(Source: 04, HuggingFace dynamic speculation)*

### 12.1 The problem with static N

Standard speculative decoding generates a **fixed N draft tokens** per round. This is suboptimal: the optimal number varies token by token depending on context.

An **oracle** (which knows which draft tokens will be accepted before running the target) can compute the optimal N at each step. Empirical evidence from code generation (MBPP dataset):

| Approach | Target forward passes | Draft forward passes |
|----------|----------------------|---------------------|
| Static SL (N=5) | 38 | 192 |
| Oracle SL | 27 | 129 |

The oracle performs **29% fewer target passes** and **33% fewer draft passes**. Oracle SL shows high variance across iterations — proving that a fixed N is always leaving performance on the table.

### 12.2 Three scheduling approaches

| Approach | How it works |
|----------|-------------|
| **Constant** (Leviathan et al.) | Fixed `num_assistant_tokens` throughout generation |
| **Heuristic** | Increase N if all draft tokens accepted last round; decrease if any rejected |
| **Dynamic** (Intel + HF, 2024) | After each draft token, check draft model confidence. If below threshold, halt early and send to target |

### 12.3 Dynamic approach: how it decides

After generating each draft token, compute the **softmax of the draft model's logits** for that token. If the softmax probability (confidence) falls below the configured threshold, **stop drafting immediately** — even if the maximum `num_assistant_tokens` hasn't been reached — and send to the target for verification.

```python
# HuggingFace API for dynamic speculation (Transformers ≥ 4.45.0)
# Dynamic is now the default — no code changes needed

outputs = model.generate(**inputs, assistant_model=assistant_model)

# Tuning parameters:
assistant_model.generation_config.assistant_confidence_threshold = 0.4  # default optimal
assistant_model.generation_config.num_assistant_tokens = 20              # max per round

# Revert to heuristic:
assistant_model.generation_config.num_assistant_tokens_schedule = 'heuristic'
assistant_model.generation_config.assistant_confidence_threshold = 0
assistant_model.generation_config.num_assistant_tokens = 5
```

### 12.4 Dynamic vs heuristic benchmark results

All experiments: greedy decoding (temperature = 0), RTX 4090.

| Target | Draft | Task | Heuristic | Dynamic |
|--------|-------|------|-----------|---------|
| `facebook/opt-6.7b` | `facebook/opt-125m` | summarization | 1.82× | **2.71×** |
| `facebook/opt-6.7b` | `facebook/opt-125m` | open-ended | 1.23× | **1.59×** |
| `Salesforce/codegen-6B-mono` | `codegen-350M-mono` | code (python) | 0.89× ❌ | **1.09×** |
| `google/flan-t5-xl` | `flan-t5-small` | summarization | 1.18× | **1.31×** |
| `meta-llama/Llama-3.1-8B` | `Llama-3.2-1B` | summarization | 1.00× (no gain) | **1.52×** |
| `meta-llama/Llama-3.1-8B` | `Llama-3.2-1B` | open-ended | 1.00× | **1.18×** |
| `meta-llama/Llama-3.1-8B` | `Llama-3.2-1B` | code (python) | 1.09× | **1.15×** |

> Dynamic speculation is **always at least as good** as heuristic, and often dramatically better. Even cases where heuristic caused slowdown (codegen-6B), dynamic recovers a speedup.

### 12.5 DISCO: the learned classifier variant

*(Source: 04)*

> Mamou et al. (2024), *Accelerating Speculative Decoding using Dynamic Speculation Length*, arXiv:2405.04304

DISCO uses a learned **classifier** (instead of a threshold on softmax probability) to decide whether the draft model should continue or hand off to the target. More accurate than the confidence threshold approach but requires training a per-model classifier.

---

## 13. Self-Speculative Decoding: LayerSkip

*(Source: 05, HuggingFace blog, Meta)*

### 13.1 Classic vs self-speculative: the comparison

| Aspect | Classic (two-model) | LayerSkip (self-speculative) |
|--------|---------------------|------------------------------|
| Draft generator | Separate smaller model | Early layers of the target model |
| Verify model | Full target model | Remaining deeper layers |
| VRAM | Two sets of weights | One set of weights |
| Same-tokenizer constraint | Yes (mandatory) | Yes (same model) |
| Training requirement | None (any paired models) | Requires LayerSkip fine-tuning |
| Best use | Any paired model family | Models pre-trained with LayerSkip recipe |

### 13.2 How it works: early exit and unembedding

In a standard transformer, the LM head (output projection) is applied only after the **final layer**. LayerSkip modifies the model so the LM head can be applied at any **intermediate layer** — the process of projecting from an intermediate hidden state to vocabulary probabilities is called **unembedding**.

This requires special training: an intermediate layer's hidden state is not naturally interpretable by a vocabulary head trained on final-layer representations.

**LayerSkip training recipe:**
1. **Layer dropout:** Progressively higher dropout rates for deeper layers. Trains the model to be less reliant on its later layers.
2. **Early exit loss:** Total loss = sum of normalized losses from every intermediate exit layer. Forces the LM head to learn to interpret outputs from any layer.

**Inference:**
1. **Self-drafting:** Run only layers 0…E (early exit at layer E). Apply LM head at layer E → draft tokens.
2. **Self-verification:** Take draft tokens and run the remaining layers E+1…N. Key-value pairs from early layers are **cached and reused** — verification only computes the later layers.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

early_exit_layer = 4
checkpoint = "facebook/layerskip-llama2-7B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to("cuda")

inputs = tokenizer("Alice and Bob", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, assistant_early_exit=early_exit_layer)
```

Available checkpoints: `facebook/layerskip-llama2-7B`, `-13B`, `-70B`, `layerskip-llama3-8B`, `layerskip-llama3.2-1B`.

### 13.3 Three caches shared between draft and verify

| Cache type | What it stores | Benefit |
|------------|---------------|---------|
| Shared weights | Same layers 0…E used for both phases | No weight duplication |
| Shared KV cache | Key-value pairs from layers 0…E cached in draft phase | Verification skips recomputing early layers |
| Exit query cache (KVQ) | Query vector from layer E saved | Verification resumes seamlessly from draft handoff |

The KVQ cache reduces both memory overhead and inference latency — verification picks up exactly where drafting left off.

### 13.4 The early exit tradeoff

- **Exit earlier** → draft tokens faster but less accurate → lower acceptance rate.
- **Exit later** → draft tokens more accurate but slower → closer to full model speed.

The optimal early exit layer is a hyperparameter that depends on model size and task domain.

### 13.5 Benchmark (summarization, A100 80GB)

| Model | Method | Time/output (s) | Efficiency vs. baseline |
|-------|--------|----------------|------------------------|
| Llama3 8B | Standard spec (Llama-3.2-1B draft) | 29.08 | 1.53× |
| Llama3 8B | Standard spec (Llama-3.2-3B draft) | 28.73 | 1.00× (break-even) |
| Llama3 8B | **LayerSkip early exit @ layer 4** | **28.87** | **1.83×** |
| Llama2 70B | Standard spec (Llama-2-13B draft) | 112.97 | 2.44× |
| Llama2 70B | **LayerSkip early exit @ layer 10** | **113.2** | **2.06×** |

LayerSkip is faster than two-model speculative decoding for the 8B case. The 70B exception may reflect insufficient LayerSkip training tokens for that model (328M vs 52B for 7B model).

### 13.6 Why Layer 14 chose two-model

LayerSkip requires **re-training the target model** with the LayerSkip recipe. Layer 14 prioritizes using off-the-shelf models — any quantized small model + any large target — without retraining. The two-model approach:

- Works with any model family combination (just same tokenizer required).
- Allows using GPTQ-quantized draft models from Layer 13.
- Has no training dependency.

LayerSkip's advantage: zero additional VRAM for the draft model and shared KV cache. Layer 14's `DraftReq` mirrors this principle — but across two physically separate model pools rather than within one model.

### 13.7 Related self-speculative approaches

- **Draft & Verify:** skips pre-determined attention and feed-forward layers (instead of a contiguous prefix). Uses Bayesian optimization to find optimal layer removal. Speedup: up to 1.99×.
- **MagicDec:** uses a subset of the KV cache as the "early exit" signal; useful for long-context workloads.
- **Jacobi Decoding** and **Lookahead Decoding:** use random or n-gram "guess tokens" as draft; no separate model or training.

---

## 14. Extensions Panorama

*(Source: 09, Data Processing Club — the most complete single-article coverage)*

### 14.1 DistillSpec — align draft to target via knowledge distillation

Standard speculative decoding: draft and target are independently trained; acceptance depends on how well q_d matches p_t.

**Subtle trap:** higher token-level accuracy of the draft model doesn't always help. If the draft is *more confident* than the target about a token the target is uncertain about, that token will be rejected — even if it's "correct."

**DistillSpec fix:** distill from target → draft so q_d more closely approximates p_t. Minimize KL(p_t || q_d) rather than cross-entropy against ground truth labels.

Result: **10–45% additional speedup** on top of baseline speculative decoding.

### 14.2 Online Speculative Decoding — fine-tune the draft on the fly

After verification, store mismatches in a training buffer and continuously fine-tune the draft on the target's outputs. Adapts to shifts in query distribution without pre-collecting training data. Especially effective for domain-specific deployment where production queries differ from training data.

### 14.3 Draft & Verify — layer truncation as draft

Remove some transformer layers from the target to create a faster draft. Leverages skip connections (x^(l+1) = x^l + f(x^l)) — modern LLMs have small representation changes across layers, so removing some layers doesn't degrade accuracy dramatically. Uses Bayesian optimization to find optimal layer removal. Speedup: up to **1.99×**. Benefit: single model deployment, no separate draft weights.

### 14.4 Tree attention — verify multiple branches at once

Instead of one draft sequence, build a **tree** of candidates. Tree attention lets the target verify all branches in one forward pass using a sparse, hierarchical mask where each token attends only to its ancestor nodes.

Example: four candidate completions of "machine ..."
```
machine → learning → algorithm → is
       → learning → system → design
       → translation → models → are
       → translation → system → design
```
Tree has 11 nodes, verified in one forward pass. Accept/reject by tree traversal using SpecJudge at each node.

This is the mechanism behind EAGLE-2 and EAGLE-3's "dynamic draft trees."

### 14.5 Medusa — multiple draft heads inside one model

Add K extra LM heads to the final hidden layer of the target:
- Head 1 predicts token t+2
- Head 2 predicts token t+3
- ...
- Head K predicts token t+K+1

All heads produce candidates in parallel → combine into s₁×s₂×...×sK candidate sequences → verify via tree attention. Single model deployment; no separate draft model weights needed.

---

## 15. Production Context

*(Source: 08, Brenndoerfer; 07, Pierce Freeman)*

Speculative decoding is not a research technique waiting to ship. It is standard practice in high-performance inference stacks.

### Deployed today

- **Google** — confirmed in AI Overviews and multiple products.
- **Meta** — production inference systems.
- **Anthropic** — deployed in Claude serving infrastructure.
- Most major LLM serving platforms: vLLM, SGLang, TGI, TensorRT-LLM, llama.cpp.

> *"Same parameters, same hardware, faster models."* — Pierce Freeman (07)

### What makes it production-worthy

- **Exact quality guarantee** — no accuracy tradeoff to justify to stakeholders. The mathematical proof (§5) means engineers can deploy with confidence.
- **No target model changes** — target model weights are unchanged; deploy as-is. Quantize the draft model, not the target.
- **Works with existing checkpoints** — no new training required for the target model.
- **Graceful degradation** — the worst case is identical to baseline; there is no regime where speculative decoding makes output slower *and* worse.

---

## Appendix A: Layer 14 Code Mappings

Collected from all nine "Connection to Layer 14" and "Mapping to Layer 14" sections:

| Concept | Layer 14 code |
|---------|---------------|
| Draft model | `spec_runner.draft_model_runner` (a `ModelRunner` with quantized weights) |
| Target model | `spec_runner.target_model_runner` |
| K candidate tokens / γ | `num_spec_tokens` in `SpecRunner` |
| One target forward pass for verification | `_verify_extend()` |
| Accept left-to-right until mismatch | `_accept_reject()` in `spec_runner.py` |
| `P(accept) = min(1, p_target/p_draft)` | Greedy special case: `argmax(target) == draft_token` (temperature=0) |
| Residual / SpecNormalize | When rejected, target generates its own argmax token |
| K+1 tokens per round | `num_spec_tokens + 1` in `verify_extend` |
| Acceptance rate α | `spec_runner.acceptance_rate` = `_total_accepted / _total_proposed` |
| Acceptance length τ | `tokens_per_step` counter in `lesson/07_statistics.md` |
| Medusa heads | Not in Layer 14 (separate design; Medusa integrates heads into one model) |
| N-gram | Not in Layer 14 (requires no second model; Layer 14 always uses a real draft model) |
| LayerSkip shared KV cache | Conceptual mirror: Layer 14's KV mirroring (`lesson/03_draft_kv_mirroring.md`) |
| EAGLE-3 / MLP Speculator | Beyond Layer 14 scope (Layer 14 implements classic draft-target) |
| Tree attention | Not in Layer 14 (linear draft only); EAGLE/EAGLE-2 extension direction |
| DistillSpec | Motivation for fine-tuning draft on target outputs (L5 exercise) |
| Dynamic speculation | Layer 14 uses fixed `num_spec_tokens` — dynamic is an explicit extension in `lesson/09_next_steps.md` |
| `draft_tensor_parallel_size` | Draft model's tensor parallel rank in multi-GPU setups |
| `kv_memory_fraction` | VRAM split between draft and target KV pools |

---

## Appendix B: Limits of Each Source (Collected)

Each article explicitly documents what it does not cover. Collected here for completeness:

**01 (ML Mastery):** Uses HuggingFace `assistant_model` API which handles KV management internally. Shows stochastic rejection sampling (temperature > 0). Layer 14 uses greedy (temperature = 0): accept if `argmax(target) == draft_token`. Formula τ = (1−α^(γ+1))/(1−α) is measured empirically in Layer 14 as `tokens_per_step`.

**02 (Jarvis Labs):** Focuses on vLLM API. Benchmark numbers specific to Llama 3.1/3.3 on L40S/H200 — different hardware will show different patterns. EAGLE-3 and MLP Speculator are beyond Layer 14's scope.

**03 (HuggingFace TGI):** Intentionally brief conceptual docs. Does not explain KV cache implications of running two models simultaneously (the main engineering challenge in Layer 14). Does not explain the rejection-sampling criterion in detail.

**04 (Dynamic Speculation):** Benchmarks on RTX 4090; production H100/A100 may show different patterns. Confidence threshold is a global hyperparameter; DISCO trains per-model classifiers for better precision. Layer 14 does not implement confidence-based early stopping — explicit extension direction.

**05 (LayerSkip):** Requires LayerSkip checkpoints; not general-purpose for arbitrary model pairs. HuggingFace transformers has only implemented "Shared Weights" (not KVQ cache as of Nov 2024). Early exit layer needs manual tuning per model and domain. Production systems (SGLang, vLLM) have limited LayerSkip support.

**06 (Adaptive ML):** Does not show code — purely conceptual. Covers only classic two-model approach. Does not address KV cache management. Interactive visualizations are live on the web page; markdown captures only the text.

**07 (Pierce Freeman):** Concise by design — does not cover KV cache management, N+1 window trick, or causal masking. No code examples. No variant coverage. Does not discuss batching.

**08 (Brenndoerfer):** Treats single-request inference. Layer 14's `SpecRunner` handles batched serving (multiple requests simultaneously, shared KV pool, per-request `DraftReq`). The `parallel_verification` code is userland HuggingFace — Layer 14's `verify_extend` is lower-level. Correction distribution shown is stochastic version; Layer 14 uses greedy.

**09 (Data Processing Club):** Math presented without implementation code. Treats single-request inference. Tree attention and Medusa sections are L3 previews of EAGLE-family methods. DistillSpec and Online SD are natural L5 extensions.

**Common limit across all 9 sources:** None of these articles explains:
- How two KV pools are allocated and managed simultaneously (`DraftReq` + `Req`).
- The N+1 window trick in `verify_extend` (why N draft tokens + 1 target token fit in one forward pass with correct causal masking).
- Production gotchas: P99 latency, prefill dominance, VRAM fragmentation.
- How to choose `kv_memory_fraction` to split VRAM between draft and target.

Those are L3 territory (lesson files) and L4 (SGLang docs, production references).
