# Accelerating Decode-Heavy LLM Inference with Speculative Decoding on AWS Trainium and vLLM

**Source:** https://aws.amazon.com/blogs/machine-learning/accelerating-decode-heavy-llm-inference-with-speculative-decoding-on-aws-trainium-and-vllm/
**Authors:** AWS Machine Learning Blog
**Level:** L3 — Production deployment guide; decode-heavy workload analysis, draft model tuning, concrete benchmarks
**Why here:** The most practical treatment of the two levers practitioners actually control (`draft model selection` and `num_speculative_tokens`). Unique for showing both best-case and worst-case prompt behavior in the same benchmark, and for discussing the Qwen3 model family as a concrete worked example. Addresses the "when does speculative decoding NOT help?" question that L2 articles mostly ignore.

---

## One-line summary

Speculative decoding on AWS Trainium2 reduces inter-token latency by up to 3× for decode-heavy, structured workloads; for open-ended generation it provides no consistent benefit.

---

## The core problem being solved

**Decode-heavy workloads** (writing assistants, coding agents, report generation) produce far more output tokens than they consume. This makes the *decode stage* the dominant cost. During autoregressive decoding:
- Tokens are generated sequentially
- Hardware accelerators are **memory-bandwidth-bound** and underutilized
- Each decode step reads the full KV cache from HBM (high-bandwidth memory) for just one token

Speculative decoding attacks this by:
1. Draft model proposes `n` candidate tokens quickly
2. Target model verifies all `n` tokens in one forward pass
3. Fewer serial decode steps → lower latency + higher hardware utilization

---

## Two critical tuning parameters

This article is unique in naming exactly two knobs and showing their tradeoff curves:

### 1. Draft model selection

Requirements for the draft model:
- **Must share tokenizer and vocabulary** — speculative decoding operates on token IDs verified directly by the target model
- **Recommend same architectural family** — their next-token predictions agree more often

Results comparing Qwen3-0.6B vs Qwen3-1.7B as drafts for Qwen3-32B:

| Draft model | Speed advantage | Acceptance rate | Net result |
|-------------|----------------|-----------------|-----------|
| Qwen3-0.6B | Faster to run | ~60% lower than 1.7B | Slower overall — compute savings cancelled by rejections |
| Qwen3-1.7B | Moderate | Higher | **Better balance — recommended** |

> "The smaller 0.6B model was faster to run, but its acceptance rate was roughly 60% lower, enough to cancel out the compute savings."

**Key insight:** A faster-but-worse draft can be *worse* than a slower-but-better draft. The acceptance rate penalty can dominate the speed advantage.

### 2. `num_speculative_tokens` (draft window size)

Evaluated values from 5 to 15:

| Setting | Effect |
|---------|--------|
| 5 | Limited speedup — too few tokens proposed to skip meaningful decode steps |
| 7 | Best balance for Qwen3-1.7B + Qwen3-32B |
| 15 | Increased rejections — wasted draft compute + higher target verification cost |

**Context-dependence:** The "best" `num_speculative_tokens` varies by prompt type. The article uses both structured prompts and open-ended prompts in the benchmark. The sweet spot differs between them.

---

## Why speculative decoding helps decode (not prefill)

The article explicitly isolates where the speedup comes from:

### TTFT (Time to First Token): **unchanged**
TTFT is dominated by the prefill phase — encoding the input context. Speculative decoding does not touch prefill. TTFT with and without speculative decoding are nearly identical.

### ITL (Inter-Token Latency): **reduced for structured prompts**
ITL is the decode stage. Speculative decoding reduces the *number of target-model decode steps*, not the cost of individual steps. For structured prompts where the draft model is accurate:
- Inter-token latency dropped from ~45 ms to ~15 ms (3× improvement)

### End-to-end latency: **proportional to how much time is spent decoding**

---

## Best-case vs worst-case: prompt structure matters

The benchmark uses two representative prompt types and reports the results side by side — this is the key teaching moment of the article.

### Structured prompt: "Repeat the following line exactly 50 times"
- Draft model reliably predicts what target would generate
- Large fraction of target decode steps are skipped
- ITL: **~15 ms per token** (vs 45 ms baseline)
- Speculative curve consistently below baseline throughout the run

### Open-ended prompt: "I believe the meaning of life is"
- Draft model frequently diverges from target — high rejection rate
- Token rejections negate potential gains
- ITL: **~45 ms per token** (identical to baseline)
- Speculative and baseline curves largely overlap

**Implication for deployment decisions:**
- Code generation, structured data extraction, templated reports, configuration synthesis → **good candidates**
- Creative writing, open-ended reasoning, diverse conversational AI → **speculative decoding adds overhead without benefit**

---

## NeuronX Distributed Inference (NxDI): four speculation modes

AWS Neuron SDK provides four modes for speculative decoding on Trainium/Inferentia:

| Mode | Description | This post |
|------|-------------|-----------|
| **Vanilla** | Separate draft and target models compiled independently | Simplest baseline |
| **Fused** | Draft and target compiled together for better performance | **Used in this post** |
| **EAGLE** | Draft model leverages hidden-state context from target to improve acceptance rates | More advanced |
| **Medusa** | Multiple parallel prediction heads reduce draft-model overhead | More advanced |

The post uses **fused speculation** (`enable_fused_speculation=true`) — draft and target are compiled as a unit. This reduces data movement overhead between the two model forward passes.

---

## Experimental setup

Two vLLM inference services on the same EKS cluster, identical except for decoding method:

```
Baseline (qwen-vllm):
  - Qwen3-32B target
  - Standard autoregressive decoding

Speculative (qwen-sd-vllm):
  - Qwen3-32B target
  - Qwen3-1.7B draft
  - num_speculative_tokens=7
  - enable_fused_speculation=true
```

Both run on Trn2 (trn2.48xlarge), same tensor parallelism, same sequence length, same batching limits, same Neuron DLC image.

Load generation: `llmperf` tool, same traffic patterns against both endpoints simultaneously.

---

## Lessons learned (direct quotes, condensed)

1. **Draft model must share tokenizer** — mismatched vocabularies make verification impossible
2. **Same family → higher acceptance rates** — shared training objectives → similar probability distributions
3. **Smaller is not always better for the draft** — 0.6B was too inaccurate despite being faster
4. **`num_speculative_tokens=7` was optimal** for this model pair on structured prompts
5. **5 is too conservative, 15 increases rejections** — tuning required per workload
6. **Speculative decoding selectively helps** — structured outputs benefit, open-ended do not
7. **TTFT is never affected** — only the decode stage benefits

---

## How this maps to Layer 14

| AWS production concept | Layer 14 equivalent |
|----------------------|---------------------|
| `num_speculative_tokens` | `num_spec_tokens` in `SpecRunner` |
| Draft model selection (same family) | `DraftModelRunner` uses separate smaller model; tokenizer must match |
| Fused vs vanilla speculation | `SpecRunner` couples both `ModelRunner`s; similar to fused mode |
| Acceptance rate monitoring | `_total_accepted / _total_proposed` stats in `lesson/07_statistics.md` |
| Structured vs open-ended workload | Acceptance rate variance by prompt type; dynamic speculation addresses this |
| TTFT unchanged | Prefill path in SGLang is separate from speculative path |

The AWS article's "vanilla" mode maps directly to Layer 14's two-`ModelRunner` approach. The "fused" and "EAGLE" modes are extensions described in `lesson/08_eagle.md` (if present).

---

## Limits of this article (for book context)

- AWS-specific infrastructure (Trainium, NxDI, EKS) — not directly applicable to NVIDIA + SGLang deployments
- Only benchmarks Qwen3 family — acceptance rates vary significantly by model family
- Does not show the acceptance/rejection algorithm code — treats speculative decoding as a black box at the vLLM config level
- No discussion of batched speculative decoding (how does it interact with continuous batching?) — a critical production concern glossed over
- Fused mode reduces latency further but is less transparent about what's happening internally
