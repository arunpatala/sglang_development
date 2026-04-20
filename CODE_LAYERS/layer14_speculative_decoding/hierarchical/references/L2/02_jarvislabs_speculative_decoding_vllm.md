# Speculative Decoding in vLLM: Complete Guide to Faster LLM Inference

**Source:** https://jarvislabs.ai/blog/speculative-decoding-vllm-faster-llm-inference
**Level:** L2–L3 — Definitions + motivation + benchmark data
**Why here:** Best article for comparing **five** speculative decoding variants side by side (draft model, N-gram, suffix decoding, MLP speculator, EAGLE) with real vLLM benchmark tables. "The quick brown fox" worked example is the clearest token-level trace in any L2 article.

---

## Summary

vLLM supports five speculative decoding techniques. This article explains each theoretically, shows benchmarks on Llama-3.1-8B (1× L40S) and Llama-3.3-70B (2× H200) across general chat (ShareGPT) and coding (SWE-bench), and gives decision guidelines for which to use.

---

## The core workflow: Guess, Check, and Keep

The key insight: while LLMs must generate tokens one by one (each depends on the previous), they can **verify multiple tokens in parallel**. Transformer architectures process entire sequences in a single forward pass.

So instead of running the expensive target model N times to generate N tokens, run it **once to verify N guessed tokens simultaneously**.

### Step-by-step

1. **Drafting:** The small draft model quickly reads the context and proposes K tokens (typically 5).
2. **Verification:** The big target model takes original input + K guessed tokens, runs **one single forward pass** to check them all.
3. **Rejection sampling:** Compares `P_target` vs `P_draft` token-by-token:
   - `P_target ≥ P_draft`: accept immediately.
   - `P_target < P_draft`: reject probabilistically; the lower the target confidence relative to draft, the higher the rejection probability.
4. **Correction:** At the first bad token, discard it and everything after; target inserts the correct token; draft starts guessing again.

> Crucial requirement: draft and target must share the **same vocabulary size / tokenizer**. Otherwise tokens cannot be directly compared.

### Worked example: "The quick brown fox"

**Step 1 — The Guess:**
Draft model sees "The quick brown fox" and predicts 5 tokens: *"jumps over the lazy dog"* (note: the correct phrase is "jumps over the log")

**Step 2 — The Check:**
Target model runs one single pass over the original + 5 guesses.

**Step 3 — The Result:**
| Token | Decision |
|-------|----------|
| "jumps" | ACCEPTED |
| "over" | ACCEPTED |
| "the" | ACCEPTED |
| "lazy" | REJECTED — target calculated "log" was correct |

**Step 4 — The Speed-Up:**
3 tokens generated in the time it normally takes for 1 → **3× speed-up** for this step.

**Step 5 — The Fix:**
Target replaces "lazy" with "log". Draft starts again from "…jumps over the log".

---

## Five techniques in vLLM

### 1. Draft model (classic two-model)

A separate, smaller model (e.g., Llama-3.1-8B as draft for Llama-3.3-70B as target). Best for open-ended text. Requires extra VRAM. Same tokenizer mandatory.

### 2. N-gram matching

Finds patterns in previously generated text and predicts repetitions. Example: if "np.mean" appeared before, seeing "np." suggests predicting "mean".

- No VRAM overhead (no second model).
- Best for code or highly repetitive text.
- Performs poorly on diverse, creative tasks.

### 3. Suffix decoding

Like N-gram but uses **dual suffix trees** (per-request + global) for much faster pattern lookup. CPU-based; no GPU overhead.

- Local tree: patterns from the current request.
- Global tree: patterns from all previous requests across the server.
- Uses greedy expansion based on Path Confidence D(N) = D(Parent) × C(N).
- Excellent for agentic loops, code generation (high text repetition).

### 4. MLP Speculator (IBM, Medusa variant)

A multi-headed lightweight MLP that **attaches directly to the target model** using its context embedding vector. Three heads predict tokens at t+1, t+2, t+3 separately:

- Head 1 predicts T2. Head 2 uses T2 to predict T3. Etc.
- Each head has its own training loss → separate loss curves (t+1 converges faster than t+3).
- Trained in two stages: input alignment (standard text), then output alignment (distillation from target).
- ~1/10th parameters of a full draft model → minimal VRAM overhead.
- 2–3× speedup; high acceptance rate by leveraging target's own context.

### 5. EAGLE / EAGLE-2 / EAGLE-3

EAGLE attaches a **lightweight draft head** (1–2 transformer layers) to the target model, reusing the target's internal feature maps.

**Evolution:**
- **EAGLE-1:** single-layer head reusing top-layer features. Limitation: "feature uncertainty" — trained on perfect data but infers from its own noisy predictions.
- **EAGLE-2:** dynamic draft trees that adapt branch structure by confidence. Smarter guessing.
- **EAGLE-3 (current state-of-art):**
  - **Multi-layer fusion:** takes features from Low + Middle + High layers.
  - **Training-Time Test (TTT):** trains on its own predictions (not ground truth), fixing the distribution mismatch problem.
  - Result: acceptance rate stays flat at 70–80% even deep into the sequence.
  - **Tree attention:** generates a candidate tree, verifies with one target forward pass, prunes invalid branches.

---

## Benchmark results

### Setup

| Model | GPU | Datasets |
|-------|-----|---------|
| Llama-3.1-8B | 1× L40S (48GB) | ShareGPT (1,000 prompts) + SWE-bench Lite (300 samples) |
| Llama-3.3-70B | 2× H200 (141GB each) | Same |

### Llama-3.1-8B — General chat (ShareGPT)

| Technique | tok/s | Speedup |
|-----------|-------|---------|
| Baseline | 421 | 1.00× |
| N-gram | 491 | 1.17× |
| Suffix | 585 | 1.39× |
| EAGLE-3 | 589 | 1.40× |
| **EAGLE** | **601** | **1.43×** |

### Llama-3.1-8B — Coding (SWE-bench)

| Technique | tok/s | Speedup |
|-----------|-------|---------|
| Baseline | 370 | 1.00× |
| EAGLE-3 | 379 | 1.03× |
| EAGLE | 398 | 1.08× |
| N-gram | 406 | 1.10× |
| **Suffix** | **534** | **1.45×** |

### Llama-3.3-70B — General chat (ShareGPT)

| Technique | tok/s | Speedup |
|-----------|-------|---------|
| Baseline | 343 | 1.00× |
| EAGLE (standard) | 351 | 1.02× |
| N-gram | 385 | 1.12× |
| Suffix | 455 | 1.33× |
| **EAGLE-3** | **537** | **1.57×** |

### Llama-3.3-70B — Coding (SWE-bench)

| Technique | tok/s | Speedup |
|-----------|-------|---------|
| Baseline | 375 | 1.00× |
| EAGLE (standard) | 377 | 1.01× |
| N-gram | 416 | 1.11× |
| Suffix | 522 | 1.39× |
| **EAGLE-3** | **601** | **1.60×** |

---

## Decision guide

| Use case | Technique | Why |
|----------|-----------|-----|
| General chatbots (small model) | EAGLE / EAGLE-3 | Learned draft heads predict fluid human conversation better than heuristics. |
| Coding (small model) | Suffix Decoding | Low overhead; code's repetitive patterns favor suffix matching. |
| Coding + agents (large model 70B+) | EAGLE-3 | At scale, EAGLE-3's accurate predictions beat even pattern-matching. |
| Strict VRAM budget | Suffix Decoding | No extra weights, no training; "free" speedup. |

> **Start with Suffix Decoding.** It's the low-hanging fruit — no training, no extra VRAM, works immediately. For 70B+ models needing maximum throughput, invest in EAGLE-3.

---

## vLLM configuration examples

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

---

## Limits of this article (for book context)

- Focuses on **vLLM** API; Layer 14 builds the dual `ModelRunner` from scratch in SGLang style.
- Benchmark numbers are specific to Llama 3.1/3.3 on L40S/H200 — your mileage will vary.
- EAGLE-3 and MLP Speculator are beyond Layer 14's scope (Layer 14 implements classic draft-target, the "Draft Model" category here).
- Layer 14's `acceptance_rate` = this article's α; `tokens_per_step` = this article's τ.
