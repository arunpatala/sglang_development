# Speculative Decoding: Mechanism, Metrics, and Deployment

*A complete practitioner's guide — from the memory bottleneck to benchmarks. Stands alone; no prior reading required.*

---

Upgrading to a larger model improves quality and increases latency. On the fastest hardware available, this tradeoff holds — and the reason is not what most people assume. Modern GPUs have far more arithmetic throughput than LLM inference can use. The bottleneck is memory. A 70B model in FP16 weighs 140 GB. Every token it generates requires loading those 140 GB from GPU memory into compute cores — at roughly 70 ms per token on an A100. Arithmetic takes 0.45 ms. The GPU sits **96% idle**, waiting for data.

Adding faster GPUs or more CUDA cores changes nothing when the ceiling is memory bandwidth. Speculative decoding changes the constraint rather than the hardware.

---

## Why your GPU is mostly idle during LLM inference

The quantified picture:

```
Model size (70B FP16):         140 GB
Memory load time per token:     70.0 ms
Compute time per token:          0.449 ms
Arithmetic intensity:            1.0 FLOPs/byte
GPU's compute sweet spot:      ~150 FLOPs/byte

Memory loading is 156× slower than compute.
GPU sits 96% idle per token generated.
```

This compounds with scale: LLMs generate text **one token at a time**, and each token depends on every previous one. Token t+1 cannot start until token t is complete. A 70B model taking 50 ms per token needs 5 full seconds to generate 100 tokens — sequentially, with the GPU mostly idle through all of it.

There is one asymmetry that makes a solution possible:

> **Generating N tokens requires N forward passes. Verifying N tokens requires one.**

A transformer can score an entire sequence in parallel — computing probability distributions at every position simultaneously via causal attention. If we could guess the next several tokens correctly, we could verify all of them in a single target-model pass. That is the mechanism speculative decoding uses.

Two observations make it feasible. **First:** not every token is equally hard to predict. "The capital of France is..." resolves to "Paris" without a 70B model — a much smaller model gets it right. Large models are mainly better at the hard cases. **Second:** because memory is the bottleneck, idle compute is available for free. Running extra arithmetic in parallel doesn't slow down the memory reads that are already happening.

Together: use a small, fast model to generate candidate tokens using the idle compute, and let the large model verify them in a single pass using its already-scheduled memory reads.

---

## The idea: a junior and a senior working together

The clearest way to hold this: two developers working together.

A **junior developer** types quickly — drafts a block of code fast, not always perfect. A **senior developer** is slower but authoritative. Instead of the senior writing every line from scratch, the junior drafts a block; the senior reviews each line and accepts what's right, fixes the first mistake, and continues from there. The senior's output is unchanged — it's the same code they'd have written — but their time is used far more efficiently.

In practice, two models take these roles: a **smaller, faster model** (draft) proposes candidate tokens; a **larger, more capable model** (target) verifies them in one pass and accepts what matches. The user always receives the target model's authoritative output. The draft is never directly visible.

---

## How it works: the four-phase loop

In baseline decoding, the large model generates one token, appends it, generates the next — 200 tokens means 200 target-model passes, each reading 140 GB. Speculative decoding restructures this into a repeating four-phase cycle.

---

### Phase 1 — Draft

The small model autoregressively generates K candidate tokens (K typically 4–8). Because it's 10–100× smaller, generating K tokens costs a fraction of one target-model pass. The draft model makes its best prediction; it doesn't know whether its candidates will be accepted.

---

### Phase 2 — Verify

The target model takes the original input **plus all K draft tokens** and runs a single forward pass over the entire sequence simultaneously.

This works because of the transformer's causal attention mask — each position can only attend to preceding tokens, so probability distributions at every draft position are computed independently and in parallel:

```python
def parallel_verification(target_model, input_ids, draft_tokens):
    full_sequence = torch.cat([input_ids, draft_tokens], dim=-1)
    logits = target_model(full_sequence).logits

    num_input = input_ids.shape[-1]
    num_draft  = draft_tokens.shape[-1]

    # Logits at the position just before each draft token
    verification_logits = logits[:, num_input - 1 : num_input + num_draft - 1, :]
    probs = F.softmax(verification_logits, dim=-1)

    draft_probs = [probs[0, i, draft_tokens[0, i].item()].item()
                   for i in range(num_draft)]
    return draft_probs, probs
```

The same causal mask that enables efficient transformer training enables efficient speculative verification. One expensive target-model operation checks everything the draft proposed.

---

### Phase 3 — Accept or reject

At each draft position, the target compares its own prediction against the draft. Tokens are evaluated left to right:

- If the target **agrees** (assigns equal or higher probability to the draft token than the draft did): **accept**, move to the next position.
- If the target **disagrees**: **reject** this token and discard everything after it.

---

### Phase 4 — Correction and restart

When a token is rejected, the target's own prediction at that position is committed and the cycle restarts from there. A bonus token is always generated: either the target's K+1th prediction (if all drafts were accepted) or the correction token at the rejection point. The cycle always makes forward progress.

**Worst case:** every draft token rejected → one target-model token committed, identical to baseline.

**Best case:** every draft token accepted → K+1 tokens committed in a single target-model pass.

**Average case:** some prefix accepted → 2–3× speedup with a well-matched draft model.

> *"Even if the small model is wrong 60% of the time, you can still see significant speedups because of the parallel verification."* — Pierce Freeman

The accept/reject step carries a precise mathematical rule — and that rule is why the output is guaranteed identical to standard decoding.

---

## The acceptance criterion and the guarantee

The rule at each draft position:

```
P(accept) = min(1, p_target / p_draft)
```

| Scenario | p_target | p_draft | P(accept) |
|----------|----------|---------|-----------|
| Target agrees strongly | 0.25 | 0.10 | 1.00 — always accept |
| Close agreement | 0.15 | 0.18 | 0.83 |
| Draft overestimates | 0.05 | 0.20 | 0.25 |

When `p_target ≥ p_draft`, the token is always accepted. When `p_draft > p_target`, the draft was overconfident — accept probabilistically, proportional to how much the target agrees.

When a token is rejected, the system cannot simply resample from `p_target` — that would double-count tokens the draft already had a chance to propose. Instead, it samples from the **residual distribution**: `max(0, p_target(x) − p_draft(x))`, normalized. This is the correction that closes the proof.

The mathematical result: in every case — accepted via the rule, or rejected and corrected — the token is emitted with exactly `p_target(x)`. The output distribution is identical to what you would get from running the target model alone.

> *"The algorithm speeds up generation from autoregressive models by computing several tokens in parallel, without affecting output quality; in fact, the method guarantees an identical output distribution."* — Leviathan, Kalman, Matias (Google Research, original authors)

Nothing is approximated. Nothing is skipped. The speedup comes entirely from amortizing the 140 GB weight load across multiple committed tokens. Three metrics govern how much of that amortization is realized in practice.

---

## The three governing metrics: α, γ, τ

**Acceptance rate (α):** fraction of draft tokens the target accepts.

```
α = accepted tokens / total draft tokens proposed
```

| α | Interpretation |
|---|---------------|
| ≥ 0.7 | Excellent — draft is well-matched |
| 0.5–0.7 | Good — worthwhile to deploy |
| < 0.5 | Poor — consider a different draft model |

**Speculative token count (γ):** how many tokens the draft proposes per round — configurable at inference time.

- High α → increase γ (7–10 tokens)
- Low α → decrease γ (3–5 tokens)

**Acceptance length (τ):** average tokens committed per round, from the geometric series:

```
τ = (1 − α^(γ+1)) / (1 − α)
```

Concretely, speedup as a function of α (target=100ms, draft=15ms, K=5):

| Acceptance Rate | Expected Tokens/Round | Speedup |
|----------------|-----------------------|---------|
| 0.5 | 1.97 | 1.12× |
| 0.6 | 2.38 | 1.36× |
| 0.7 | 2.94 | 1.68× |
| 0.8 | 3.69 | 2.11× |
| 0.9 | 4.69 | 2.68× |

Acceptance rate is the primary lever. Even modest rates (0.6–0.7) yield meaningful speedups. Below ~0.5 the overhead of draft generation dominates and gains shrink. These metrics are only as good as the draft model that produces them.

---

## Choosing the right draft model

Four requirements — in order of strictness:

1. **Same tokenizer as the target** — non-negotiable. Draft and target must share vocabulary; otherwise tokens cannot be compared in the accept/reject step.

2. **10–20× fewer parameters** — if the draft is too large, draft generation dominates the round cost and negates the speedup. A 7B draft for a 70B target is a common production pairing; a 1B draft for an 8B target works well within the same model family.

3. **Similar training data and instruction-following style** — the more the draft's distribution resembles the target's, the more often it predicts what the target would have predicted. Cross-family pairings need benchmarking.

4. **Same architecture family when possible** — Llama-3.2-1B → Llama-3.1-8B → Llama-3.3-70B is a natural cascade; each model is a good draft for the next.

For domain-specific deployments: fine-tuning the draft on target model outputs from representative production inputs significantly boosts acceptance rates. A good draft model is necessary — but the workload also has to cooperate.

---

## When it helps and when it doesn't

**Works best with:**
- Code generation — highly repetitive, easy to predict.
- Translation, summarization, transcription — output is constrained by the input.
- Greedy or low-temperature decoding — focused, predictable outputs.
- Large size ratios — 70B/7B gains more than 7B/1B.

**Diminishing returns with:**
- High-temperature creative writing — diverse token choices; the draft can't reliably predict from a flat distribution.
- Misaligned draft models — different training distribution, consistently rejected.
- Very small target models that already fit comfortably in memory.

**The floor is not zero:** using a bigram model (no GPU, negligible cost) as the draft — acceptance rate ~20%, γ=3 — still yields a **1.25× speedup**. Verification is parallel regardless of how cheap or "dumb" the draft is. Even a nearly useless draft beats baseline because the parallel verification step amortizes the target's weight load.

The number of draft tokens per round — γ — is a parameter worth tuning dynamically, not just setting once.

---

## Static vs dynamic scheduling

Standard speculative decoding uses a fixed γ per session. An oracle that knows which draft tokens will be accepted can compute the optimal γ at each step. Empirically, oracle scheduling uses **29% fewer target passes** and **33% fewer draft passes** than static γ=5 on the same task.

Three approaches, in increasing sophistication:

| Approach | How it works |
|----------|-------------|
| **Constant** | Fixed γ throughout generation |
| **Heuristic** | Increase γ if all drafts accepted last round; decrease if any rejected |
| **Dynamic** | After each draft token, check the draft model's softmax confidence. If below threshold, halt drafting early and send to target now. |

Dynamic scheduling is now the default in HuggingFace Transformers ≥ 4.45.0:

```python
# Dynamic is the default — no code change needed
outputs = model.generate(**inputs, assistant_model=assistant_model)

# Tune the confidence threshold (default 0.4 works well for most cases)
assistant_model.generation_config.assistant_confidence_threshold = 0.4
```

Dynamic is **always at least as good** as heuristic — and in cases where heuristic caused slowdown (e.g., codegen-6B), dynamic recovers a speedup. Beyond scheduling, the variant you deploy sets the ceiling on these gains.

---

## The variant landscape

The classic two-model setup (draft model + target model) is one of five techniques available in production serving frameworks. Each makes a different tradeoff:

| Variant | How it drafts | VRAM overhead | Best for |
|---------|--------------|---------------|----------|
| Classic draft model | Separate smaller model | Full second model | Open-ended text, any task |
| N-gram matching | Repeating patterns in generated text | None | Code, repetitive text |
| Suffix decoding | Dual suffix trees (per-request + global) | None (CPU) | Code, agentic loops, long context |
| MLP Speculator (Medusa) | Lightweight multi-head MLP on target's embeddings | ~1/10th of a full model | Balanced: speed + low overhead |
| EAGLE / EAGLE-3 | Lightweight head reusing target's internal features | ~1/10th of a full model | General chat, 70B+ at scale |

**Decision guide:**

| Use case | Best technique | Why |
|----------|---------------|-----|
| General chatbots (≤8B) | EAGLE / EAGLE-3 | Learned heads predict fluid conversation better than heuristics |
| Coding (≤8B) | Suffix Decoding | No overhead; code's repetitive patterns favor suffix matching |
| Coding + agents (70B+) | EAGLE-3 | At scale, EAGLE-3's learned predictions beat even pattern-matching |
| Strict VRAM budget | Suffix Decoding | No extra weights, no training, immediate speedup |

> **Practical starting point:** deploy Suffix Decoding first — no training, no extra VRAM, works immediately. For 70B+ models where maximum throughput matters, invest in EAGLE-3.

**Self-speculative decoding (LayerSkip):** a single model drafts from its own early layers and verifies with its deeper layers — no second model, no extra VRAM, shared KV cache between draft and verify phases. Faster than two-model speculative decoding for 8B models in benchmarks. The tradeoff: requires retraining the target model with the LayerSkip recipe (progressive layer dropout + early exit loss). Off-the-shelf checkpoints available for Llama2 7B/13B/70B and Llama3 8B.

| | Classic (two-model) | LayerSkip (self-speculative) |
|---|---|---|
| Draft source | Separate smaller model | Early layers of the target |
| Extra VRAM | Full second model | None |
| Training required | No | Yes (LayerSkip recipe) |
| Works with any model pair | Yes (same tokenizer) | No (requires LayerSkip checkpoint) |

Regardless of which variant you use, the output guarantee is the same: every committed token was validated by the full target model.

---

## In production

Speculative decoding is not a research technique waiting to ship. It is standard practice across the major LLM inference stacks:

- **Google** — AI Overviews in Google Search, and multiple other Google products.
- **Meta** — production inference systems.
- **Anthropic** — deployed in Claude serving infrastructure.
- **Frameworks:** vLLM, SGLang, TGI, TensorRT-LLM, llama.cpp — all support it natively.

Three properties make it production-safe:

- **Exact quality guarantee** — the mathematical proof means no accuracy tradeoff to justify to stakeholders or test for in evaluation.
- **No target model changes** — target weights are unchanged; quantize the draft, not the target.
- **Graceful degradation** — the worst case is identical to baseline. There is no regime where speculative decoding makes output slower *and* worse.

> *"Same parameters, same hardware, faster models."* — Pierce Freeman

---

## Further reading

Sections deliberately omitted from this article — too detailed for blog flow, valuable for practitioners going deeper. All are in `L2_blog_omitted.md`.

**§A — The full mathematical proof**
The SpecJudge + SpecNormalize decomposition proof (Case 1: x ∈ D; Case 2: x ∉ D) showing exactly why both cases emit the token with probability `p_target(x)`. Includes the key property: guaranteed resolution in one round, no re-sampling loop.

**§B — vLLM benchmark tables in full**
All four benchmark tables: Llama-3.1-8B on L40S and Llama-3.3-70B on H200, across ShareGPT (chat) and SWE-bench (coding). N-gram vs Suffix vs EAGLE vs EAGLE-3 side by side with tokens/sec and speedup ratios.

**§C — LayerSkip full treatment**
The training recipe (layer dropout + early exit loss), the three caches (shared weights, shared KV, KVQ), the early exit tradeoff, full benchmark results, and why the two-model approach was preferred for Layer 14.

**§D — Dynamic scheduling benchmark tables**
Full results comparing constant vs heuristic vs dynamic across six model pairs and three task types. Includes the DISCO learned-classifier variant (arXiv:2405.04304).

**§E — Extensions: DistillSpec, Online SD, tree attention, Medusa**
DistillSpec and its subtle trap (higher draft accuracy ≠ more accepted tokens). Online speculative decoding (fine-tune draft on the fly from mismatches). Tree attention (verify branching candidate trees in one pass). Medusa heads (multiple LM heads inside one model, verified via tree attention).

**§F — The "quick brown fox" worked example**
The full token-level trace from the Jarvis Labs article: draft proposes "jumps over the lazy dog", target accepts "jumps over the", rejects "lazy" (correct: "log"), and restarts.

**§G — Correction distribution code and acceptance scenarios**
Full Python implementation of `accept_probability()` and `correction_distribution()` with the complete scenarios table.

---

*Sources: [ML Mastery](https://machinelearningmastery.com/the-machine-learning-practitioners-guide-to-speculative-decoding/) · [Jarvis Labs / vLLM](https://jarvislabs.ai/blog/speculative-decoding-vllm-faster-llm-inference) · [HuggingFace TGI](https://huggingface.co/docs/text-generation-inference/main/en/conceptual/speculation) · [HuggingFace Dynamic Speculation](https://huggingface.co/blog/dynamic_speculation_lookahead) · [LayerSkip](https://huggingface.co/blog/layerskip) · [Adaptive ML](https://www.adaptive-ml.com/post/speculative-decoding-visualized) · [Pierce Freeman](https://pierce.dev/notes/how-speculative-decoding-works/) · [Brenndoerfer](https://mbrenndoerfer.com/writing/speculative-decoding-accelerating-llm-inference) · [Data Processing Club](https://data-processing.club/speculative/)*
