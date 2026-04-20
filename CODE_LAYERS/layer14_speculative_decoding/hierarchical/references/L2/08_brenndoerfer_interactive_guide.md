# Speculative Decoding: Fast LLM Inference Without Quality Loss

**Source:** https://mbrenndoerfer.com/writing/speculative-decoding-accelerating-llm-inference
**Author:** Michael Brenndoerfer
**Published:** January 16, 2026
**Level:** L2–L3 boundary — most thorough L2 article; adjustable reading level; full code
**Why here:** The most complete single article for an L2 reader who wants worked code, the mathematical guarantee, speedup calculations across acceptance rates, and the causal masking explanation — without yet reading the full lesson files. Adaptive reading levels make it accessible to both L2 and early L3 readers. A 55-minute read.

---

## Summary

Speculative decoding delivers **2–3× speedup** with **exact, mathematically guaranteed** output quality — not an approximation. Unlike quantization or pruning, it leaves the target model completely unchanged. Covers: memory bandwidth bottleneck → two-model architecture → parallel verification → accept/reject criterion → speedup math → production deployment.

---

## The memory bandwidth bottleneck (quantified)

```python
# A 70B model on A100 80GB
model_params    = 70e9
bytes_per_param = 2          # FP16
model_size_bytes = 140e9     # 140 GB

memory_bandwidth = 2e12      # 2 TB/s (A100)
compute_tflops   = 312e12    # 312 TFLOPS FP16

load_time    = model_size_bytes / memory_bandwidth  # ~70 ms
flops_per_token = 2 * model_params
compute_time    = flops_per_token / compute_tflops  # ~0.45 ms

# Arithmetic intensity = 1 FLOP/byte (far below GPU's ~150 FLOPs/byte sweet spot)
```

**Output:**
```
Model size: 140 GB
Memory load time per token: 70.0 ms
Compute time per token: 0.449 ms
Arithmetic intensity: 1.0 FLOPs/byte
Memory loading is 156× slower than compute
```

The GPU sits **96% idle**, waiting for data — not computing. Adding more CUDA cores changes nothing. The bottleneck is memory movement, not arithmetic.

**The key insight:** If you process multiple tokens simultaneously, you amortize the 140GB weight-loading cost across all of them. Speculative decoding does exactly this.

---

## The efficiency argument

| Action | Cost |
|--------|------|
| Generate 1 draft token (small model) | ~5% of one target forward pass |
| Verify 5 draft tokens (target model) | ~same as verifying 1 (one forward pass, causal mask handles parallelism) |

So: generate 5 drafts cheaply (~0.25× cost) + verify all 5 once (1.0× cost) = **1.25× total cost**.
If 4/5 are accepted = 4 tokens for 1.25× → **3.2× speedup**.

---

## The four-phase loop

Each speculative decoding **round**:

1. **Draft phase:** Draft model autoregressively generates K candidate tokens (K typically = 4–8).
2. **Verify phase:** Target model processes all K candidates in **one forward pass**.
3. **Accept/reject phase:** Compare draft and target probabilities token by token.
4. **Correction phase:** If a token is rejected, sample from an adjusted distribution.

Repeat from the new position until the desired output length is reached.

---

## Acceptance criterion: the mathematical guarantee

For each draft token at position i:

```python
def accept_probability(p_target, q_draft):
    return min(1.0, p_target / q_draft)
```

| Scenario | p_target | q_draft | P(accept) |
|----------|----------|---------|-----------|
| Close agreement | 0.15 | 0.18 | 0.83 |
| Target prefers more | 0.25 | 0.10 | 1.00 (always accept) |
| Draft overestimates | 0.05 | 0.20 | 0.25 |

### The correction distribution

When rejected, **cannot resample from `p_target`** — would double-count tokens the draft proposed:

```python
def correction_distribution(p_target, q_draft):
    residual = np.maximum(0, p_target - q_draft)   # element-wise
    total = np.sum(residual)
    if total > 0:
        return residual / total      # normalized
    else:
        return p_target              # fallback: full target distribution
```

Accepted samples + residual resamples = **exactly `p_target`**. The output is not an approximation — it is statistically identical to pure autoregressive sampling from the target.

---

## Parallel verification with causal masking

This is the architectural trick that makes verification efficient.

When the target receives `[prompt, draft₁, draft₂, draft₃, draft₄]`, the **causal attention mask** ensures each position only attends to preceding tokens. This creates independent probability computations at every position — all done **simultaneously in one forward pass**:

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

    # Extract probability of each draft token
    draft_probs = [probs[0, i, draft_tokens[0, i].item()].item()
                   for i in range(num_draft)]
    return draft_probs, probs
```

The same causal mask that enables efficient transformer training enables efficient speculation verification. Same mechanism, dual purpose.

---

## Speedup analysis across acceptance rates

```python
def calculate_speedup(target_ms, draft_ms, K, acceptance_rate):
    round_time = K * draft_ms + target_ms
    # Expected accepted tokens (geometric series)
    expected_tokens = sum(acceptance_rate**i for i in range(1, K + 1)) + 1
    standard_time = expected_tokens * target_ms
    return standard_time / round_time, expected_tokens
```

**Results (target=100ms, draft=15ms, K=5):**

| Acceptance Rate | Expected Tokens | Speedup |
|----------------|----------------|---------|
| 0.5 | 1.97 | 1.12× |
| 0.6 | 2.38 | 1.36× |
| 0.7 | 2.94 | 1.68× |
| 0.8 | 3.69 | 2.11× |
| 0.9 | 4.69 | 2.68× |

> Acceptance rate is the primary lever. Even modest rates (0.6–0.7) yield meaningful speedups. Below ~0.5 the overhead starts to dominate.

---

## Choosing a draft model

Three requirements:
1. **Shared vocabulary** — the tokenizers must produce the same token IDs.
2. **10–20× smaller** — if draft is too large, drafting dominates the round cost.
3. **Aligned distribution** — similar training data and instruction-following style.

Same-family model pairs (e.g., Llama-3.1-8B draft + Llama-3.3-70B target) work well. Cross-family pairings need benchmarking.

Domain-specific fine-tuning of the draft on target model outputs can significantly boost acceptance rates in production.

---

## Production deployment context

Speculative decoding is deployed today at Google, Meta, Anthropic, and most major LLM serving platforms. It is not a research curiosity — it is standard practice in high-performance inference stacks.

What makes it production-worthy:
- **Exact quality guarantee** — no accuracy tradeoff to justify to stakeholders.
- **No target model changes** — target model weights unchanged; deploy as-is.
- **Works with existing checkpoints** — no new training required for the target.

---

## Limits of this article (for book context)

- This article treats single-request inference. Layer 14's `SpecRunner` handles **batched** serving (multiple requests simultaneously), requiring the draft and target KV pools to be managed per-request.
- The `parallel_verification` code above is userland HuggingFace — Layer 14's `verify_extend` is lower-level: it slots the draft tokens into an existing `RadixAttention` KV pool and uses the output logits at each offset.
- The `correction_distribution` is the full stochastic version. Layer 14 uses greedy (temperature=0): `accept iff argmax(target_logits) == draft_token`.
- Does not cover EAGLE, Medusa, N-gram, or self-speculative variants.
- Related math-only deep dive (more L3): https://mbrenndoerfer.com/writing/speculative-decoding-math-acceptance-criterion
