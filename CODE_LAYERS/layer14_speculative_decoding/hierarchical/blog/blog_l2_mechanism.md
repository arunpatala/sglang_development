# Speculative Decoding Explained: Mechanism, Trade-offs, and Production Reality

*For engineers who configure or evaluate LLM deployments. Assumes familiarity with inference infrastructure, but not with the internals of the algorithm.*

---

You've heard that speculative decoding makes LLMs faster without sacrificing quality. The claim is real. But "faster without sacrificing quality" raises an immediate question for anyone who's worked with inference systems: **how?** There's no free lunch in compute. Something is being traded. Understanding what is the difference between deploying it correctly and being surprised by it later.

This post covers the three-step mechanism, the metric that governs all of its performance, the tunable factors you control, and what it actually does to your latency distribution in production.

---

## Why the bottleneck isn't where you think

LLM inference during the decode phase is **memory-bandwidth-bound**, not compute-bound. Each token generation requires one full forward pass that reads the model's entire weight matrix from GPU memory. On an H100, peak arithmetic throughput is roughly 1,000 TFLOPS. Peak memory bandwidth is roughly 3.35 TB/s. The transformer during single-token decode performs only a few arithmetic operations per byte of weight data read — far below what would be needed to saturate compute.

The result: GPU arithmetic units are underutilized during decode. Memory is the bottleneck; compute sits idle.

This matters because it means **running additional computation in parallel doesn't slow you down** — you're already waiting on memory reads. Speculative decoding's entire premise is to use that idle compute productively: run a small, cheap model in parallel to propose tokens, then use the large model's already-scheduled memory reads to verify them.

---

## The mechanism: three steps

### Step 1 — Draft generation

A smaller model — the draft model — generates a sequence of `γ` candidate tokens (typically 3–12). The draft model is much smaller than the target: 3B–8B parameters against a 70B+ target is common. Because the draft model is small, its per-token cost is a small fraction of the target model's.

The draft model must share the same vocabulary and tokenizer as the target. It's usually from the same model family, often trained specifically to minimize the probability gap with the target.

### Step 2 — Parallel verification in a single forward pass

The target model processes the original input sequence **plus all `γ` draft tokens** in one forward pass. This is the key efficiency: the KV cache covers all previously generated tokens. The target's forward pass is nearly the same cost as generating a single new token — but it now produces probability distributions at `γ+1` positions simultaneously.

This is not a trick. The transformer's architecture handles this naturally: during verification, all draft tokens are visible to each other and to the input, and the target computes each position's distribution in parallel. One forward pass → `γ+1` probability distributions.

### Step 3 — Accept or reject, position by position

Starting from position 1, the target compares:
- `P_target(token)` — what the target assigns to the draft's chosen token
- `P_draft(token)` — what the draft assigned to the same token

**Acceptance rule:**
- If `P_target ≥ P_draft` → **accepted**. The draft was conservative enough; the target would have been at least as confident.
- If `P_target < P_draft` → **rejected**. The draft was overconfident; discard this token and everything after it.

At the rejection point, the target samples from a corrected distribution and takes that as the next committed token. This is not a fallback — it's a principled correction that preserves the target's true output distribution.

**Crucially:** regardless of how many tokens are rejected, the output is **mathematically identical** to what the target would have produced generating each token alone. This isn't approximation. It's an exact equivalence.

> *"The algorithm guarantees an identical output distribution."* — Leviathan, Kalman, Matias (Google Research, 2022)

---

## Acceptance rate: the metric everything depends on

The performance of speculative decoding is governed almost entirely by the **acceptance rate** `α` — the fraction of draft tokens accepted per step.

- At `α = 1.0` (every draft token accepted): you commit `γ` tokens per target forward pass. Speedup approaches `γ`.
- At `α = 0.0` (every draft token rejected): you commit 1 token per target forward pass, same as baseline. No improvement; no regression either.
- In practice, good draft-target pairs achieve `α` between 0.70 and 0.90 for typical conversational workloads.

With `γ = 5` and `α = 0.80`, you commit roughly 4 tokens per target pass — a 4× improvement in effective throughput. With a draft model that costs 5% of the target's per-token compute, the net speedup is around 3.5–3.8×.

**The acceptance rate is also what makes quality lossless.** Rejection happens precisely when the draft diverges from the target — and the target's correction ensures the committed token is always drawn from the target's distribution. Nothing is approximated.

---

## Three factors you control

### 1. Draft model size

Smaller draft models are faster (lower per-token cost) but diverge more from the target (lower `α`). Larger draft models align better but cost more per token.

Typical operating ranges:
- Draft: 1.5B–8B parameters
- Target: 14B–70B+ parameters
- Size ratio: 10:1 to 50:1 is common

A draft model trained specifically on the same data distribution and with the same tokenizer as the target consistently outperforms a generic off-the-shelf small model.

### 2. Draft length (γ)

More draft tokens per step = more parallelism, but also more positions where rejection can occur. Long drafts that get rejected early waste draft compute and yield a single committed token — same cost as baseline.

Typical values: 3–8 tokens for general workloads. Some systems use 12 for highly predictable tasks (translation, code completion with strong context).

Dynamic draft length — where the draft model stops early when its confidence drops — is used by EAGLE-3 and similar production systems.

### 3. Token alignment between draft and target

Draft tokens from a misaligned model produce lower `α` even if the model is otherwise capable. Alignment deteriorates when:
- The draft model was fine-tuned on a different instruction format
- The tokenizer vocabularies differ even slightly
- The draft model hasn't seen the same RLHF/preference data as the target

In practice: use draft and target from the same model family, same tokenizer, as close in training data as possible.

---

## What speculative decoding does to your latency distribution

This is the part most deployment guides skip, and it's the most important for production systems.

Speculative decoding is usually described as a throughput optimization. At batch size 1, that's accurate. But its real value in production is different: **it reshapes the latency distribution**.

### The problem with tail latency

What breaks user-facing products isn't average latency — it's P90 and P99. Under sustained load, a large model's decode phase has long, variable tails: expert routing overhead (for MoE models), memory access variability, queueing at moderate concurrency.

Adding replicas helps average latency and queue depth. It rarely fixes the tail. Once a system crosses a concurrency threshold, P99 latency stops improving linearly and begins to cliff. More capacity yields diminishing returns for tail behavior.

### How speculative decoding improves P99

Speculative decoding reduces the **number of expensive target model forward passes** required to produce a response. Fewer target-model steps means fewer opportunities for each step's variance to compound. For a response that previously required N target-model decode steps, speculative decoding with acceptance rate `α` and draft length `γ` reduces that to roughly `N / (1 + α·γ)` steps.

Fewer steps → tighter latency bounds → better P99.

### Streaming vs non-streaming matters

Chat interfaces that stream tokens hide this improvement partially: the first tokens appear quickly regardless. Non-streaming products — where users see nothing until the full response is ready — feel the improvement directly, because speculative decoding reduces total completion time, not just time-to-first-token.

If your product is non-streaming, speculative decoding has a disproportionate impact. The full decode time is the user-facing latency.

### Cascaded systems compound the benefit

Real products rarely run one model. Safety classifiers, rerankers, post-processors each consume part of the latency budget. A model that barely fits its SLA in isolation fails inside a pipeline. Speculative decoding's P99 improvement gives you headroom for the rest of the cascade.

---

## What's available to configure today

Most major LLM serving frameworks ship speculative decoding support:

| Framework | Support | Config approach |
|-----------|---------|-----------------|
| **llama.cpp** | Draft-target via `-md` / `-m` flags | Run both models in one server process |
| **vLLM** | `--speculative-model`, `--num-speculative-tokens` | Standard serve args |
| **SGLang** | `--draft-model-path`, `--num-draft-tokens` | Matches vLLM conventions |
| **NVIDIA TensorRT** | EAGLE-3 via `modelopt` library | Requires converting the checkpoint |
| **Hugging Face TGI** | `--speculative-decoding-draft-model-id` | Single flag |

**llama.cpp local example:**

```bash
llama-server \
  -md qwen2.5-coder-3b-q8_0.gguf \
  -m qwen2.5-coder-14b-q8_0.gguf \
  --port 8012 --n-gpu-layers 99 \
  -dt 0.1 --cache-reuse 256
```

- `-md`: draft model path
- `-m`: target model path
- `-dt 0.1`: draft token acceptance threshold (filter very low-confidence drafts)
- `--cache-reuse 256`: KV cache reuse window for prefix sharing with draft

---

## Advanced variants: the same idea, more elaborate mechanisms

The draft-target setup above is the baseline. Production systems often use more sophisticated drafting mechanisms:

**EAGLE / EAGLE-3:** Instead of a separate draft model, EAGLE attaches a lightweight prediction head that uses the target model's **own hidden states** as input. No separate model to maintain. No tokenizer mismatch risk. The head is a single transformer layer plus a linear projection. EAGLE-3 extends this with a dynamic draft tree, multi-layer feature fusion, and instance-adaptive stopping.

**Multi-Token Prediction (MTP):** Used in DeepSeek-R1 style architectures. Multiple prediction heads are baked into the model itself during training — each head drafts one future token at increasing distances. No separate model needed. Verification happens in order; the longest accepted prefix is committed.

**Tree attention:** Instead of a linear draft sequence, the draft mechanism generates a tree of candidate continuations. The target verifies the entire tree in one pass (using masked attention over the tree structure), pruning invalid branches. More candidates per pass → higher probability of finding a long accepted run.

All variants share the correctness guarantee: the target's distribution is never approximated. What varies is how efficiently the draft candidates are generated and how many are verified per target pass.

---

## The one thing to design in early

Retrofitting speculative decoding after a serving stack is built is painful. It touches batching logic, KV cache memory management, request scheduling, and observability (your metrics need to count draft tokens separately from committed tokens).

Designed in from the beginning, the system is coherent: capacity planning accounts for draft model memory, the serving topology accommodates two model weights on the same GPU or across GPUs, and monitoring surfaces acceptance rate as a first-class signal.

**Acceptance rate in monitoring** is the key diagnostic. A drop in acceptance rate is an early signal that the draft and target have drifted — possibly due to a target model update that wasn't propagated to the draft.

---

## Summary

Speculative decoding works because:
1. Memory is the bottleneck during decode — idle compute is available
2. A small model can cheaply draft plausible tokens at positions where it will likely match the target
3. The target verifies all draft tokens in one pass — nearly free given the KV cache
4. The accept/reject rule is mathematically exact — no quality is traded

The one metric to watch: **acceptance rate**. It governs speedup, tail latency improvement, and alignment health. Everything else — draft model size, draft length, model family alignment — is in service of that number.

For the algorithm's mathematical foundation, hardware roofline analysis, EAGLE-3 architecture internals, and production code: see the L3 blog.

---

*Sources: [Google Research (Dec 2024)](https://research.google/blog/looking-back-at-speculative-decoding/), [NVIDIA Developer Blog (Sep 2025)](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/), [Nebius Blog](https://nebius.com/blog/posts/moe-spec-decoding), [Chris Thomas (2025)](https://christhomas.co.uk/blog/2025/02/16/speculative-decoding-using-llms-efficiently/)*
