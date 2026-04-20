# EAGLE-3 Speculative Decoding: 2–6× Faster LLM Inference Guide

**Source:** https://www.e2enetworks.com/blog/Accelerating_LLM_Inference_with_EAGLE
**Author:** AI Engineer @ E2E Networks
**Published:** November 13, 2025 — 46 min read
**Level:** L4 — Most complete treatment of EAGLE → EAGLE-2 → EAGLE-3 progression; architecture details, tree attention, training-time testing, SGLang integration
**Why here:** The deepest single public article on EAGLE-3. Explains exactly what problem each generation of EAGLE solves, the multi-layer feature fusion mechanism, the training-time testing innovation (why EAGLE-3 beats EAGLE-2), the tree attention mask, and how to tune `--speculative-eagle-topk` in SGLang. The Layer 14 `lesson/08_eagle.md` extension lesson should use this as its primary technical reference.

---

## The problem with traditional speculative decoding (Layer 14's STANDALONE mode)

Traditional two-model speculative decoding has three hard constraints:

### 1. Tokenizer compatibility constraint
Draft and target must share the **same tokenizer and vocabulary** — speculative decoding compares token IDs directly. Using a draft model from the same family (e.g., Llama-8B + Llama-70B) is the easiest way to ensure this and also gets the highest acceptance rates (similar training data → similar distributions).

**Problem:** Many production models ship as a single size with no smaller sibling. Custom fine-tuned models have no off-the-shelf draft.

### 2. Training and data requirements
If you've fine-tuned the target on domain-specific data, a vanilla base model from the same family has lower acceptance rates — it learned different patterns. Training a custom draft model requires access to the original training data, which proprietary models don't expose.

### 3. The independence problem
The draft model operates **completely independently** from the target — it doesn't reuse the target's intermediate representations. Each forward pass in the draft model duplicates the same prefix computation already done by the target model.

---

## Why EAGLE takes a different approach

EAGLE trains a **lightweight draft head** that attaches to the target model and reuses its internal features, rather than running a completely separate model. The draft head is:
- 1–2 transformer decoder layers (vs 32+ layers in the target)
- Conditioned on the target model's hidden states
- <5% additional parameters for even 70B-scale models

This eliminates problems 1 and 3: tokenizer compatibility is guaranteed (the draft head runs inside the target model), and the independence problem is solved (the draft head reuses the target's representations).

---

## EAGLE-3 vs EAGLE-2 vs EAGLE: What changed

### EAGLE (original)
- Single-layer draft head conditioned on the **top-layer hidden state** of the target
- Problem: top-layer features are optimized for predicting the immediate next token, not tokens 2–5 steps ahead

### EAGLE-2
- Added **dynamic draft tree** with confidence-based pruning (instead of fixed branching)
- Still uses only the top-layer features
- Problem: distribution mismatch — the draft head was trained on ground-truth target features but at inference must use its own previous outputs as inputs (compound errors)

### EAGLE-3 (key innovations)
1. **Multi-layer feature fusion** — uses features from low, mid, and high layers of the target
2. **Training-time testing** — simulates actual inference conditions during training (some positions use target features, others use the draft's own previous predictions)

---

## EAGLE-3 Architecture: Multi-Layer Feature Fusion

### The problem with top-layer-only features
Top-layer features encode the model's "final decision" for the immediate next token. For predicting 3–5 tokens ahead, these features miss the richer intermediate representations that contain long-range structural information.

### EAGLE-3's solution: Three-level fusion

For a model like Llama-3.1-8B with hidden dimension 4096:

```
Low-layer features (dim=4096) ─┐
Mid-layer features (dim=4096) ─┼─ Concatenate (12,288) → FC layer → 4096
High-layer features (dim=4096) ┘

The FC layer learns which features from each level matter most
for predicting multiple tokens ahead.
```

For Llama-3.1-8B (32 layers), the three levels might be layers 4, 16, and 30. The exact choice is a hyperparameter (configurable as `eagle_aux_hidden_state_layer_ids`).

---

## EAGLE-3 Architecture: Training-Time Testing

This is the key innovation over EAGLE-2. It directly addresses the distribution mismatch.

### The gap EAGLE-2 suffers from
During **training:** The draft head sees clean ground-truth features from the target model at every position.
During **inference:** After predicting token t+1, the draft head must use its own (potentially wrong) prediction to generate the features for t+2. The target model is not re-run between draft steps.

This gap causes EAGLE-2's acceptance rate to **degrade across positions** — the further into the draft sequence, the worse it gets.

### EAGLE-3's training-time testing
During training, EAGLE-3 uses a **mixed regime**:
- Some positions: ground-truth target features (standard training)
- Other positions: the draft model's own outputs fed back in (simulating inference)

This trains the draft head to handle its own errors — it learns to be robust to the position-dependent degradation that EAGLE-2 couldn't escape.

**Result:** EAGLE-3 maintains ~70–80% acceptance rate across all positions in draft sequences, vs EAGLE-2's declining acceptance rate at later positions.

---

## Tree-Based Generation and Verification

### Building the candidate tree

Starting from the current context, the draft head generates K candidates per position:

```
Context: "The weather is"

Draft step 1 (top-3 candidates):
  "sunny" (p=0.5), "getting" (p=0.3), "very" (p=0.15)

Draft step 2 (2 candidates per branch):
  sunny → "today" (p=0.7), "and" (p=0.2)
  getting → "better" (p=0.6), "worse" (p=0.3)
  very → "hot" (p=0.8)

Tree: 3 root candidates × 2 children = 8 total draft tokens
```

### Dynamic tree pruning (inherited from EAGLE-2)

Without constraints, the tree explodes exponentially. Two controls keep it manageable:

1. **Confidence-based pruning:** When the draft head's confidence (softmax probability) drops below a threshold, that branch stops growing immediately.
2. **Fixed token budget:** Only the top-N candidates by ranking are sent to the target for verification. Configured via `--speculative-num-draft-tokens` in SGLang.

**Result:** The tree is adaptive — deep and narrow for predictable contexts ("The capital of France is"), shallow and branching for uncertain contexts (creative writing).

### Tree Attention: verifying all candidates in one forward pass

Each candidate token can only attend to tokens on its **own path from the root**:

```
Attention mask for the weather tree:

          The  weather  is   sunny  getting  very  today  and  better  worse  hot
The        1     0       0    0      0        0     0      0    0       0      0
weather    1     1       0    0      0        0     0      0    0       0      0
is         1     1       1    0      0        0     0      0    0       0      0
sunny      1     1       1    1      0        0     0      0    0       0      0
getting    1     1       1    0      1        0     0      0    0       0      0
very       1     1       1    0      0        1     0      0    0       0      0
today      1     1       1    1      0        0     1      0    0       0      0
and        1     1       1    1      0        0     0      1    0       0      0
better     1     1       1    0      1        0     0      0    1       0      0
worse      1     1       1    0      1        0     0      0    0       1      0
hot        1     1       1    0      0        1     0      0    0       0      1
```

Each row shows which keys that token attends to. "today" sees "sunny" (its parent) but not "getting" or "very". This keeps all branches causally independent while allowing one forward pass to verify all candidates.

The target model processes all candidate tokens in **one forward pass**, producing probability distributions for each position. This is where the speedup comes from.

### Acceptance using rejection sampling

Walk the tree from the root. For each candidate:
- Compute acceptance probability: `min(1, p_target / p_draft)`
- If accepted: continue to this branch's children
- If rejected: discard this branch and all descendants; sample replacement from residual distribution

This is exactly the same rejection sampling mechanism as Layer 14's `_accept_reject()`, extended to tree-structured candidates.

---

## SGLang configuration for EAGLE-3

Key parameters that map to EAGLE-3's tree:

| Parameter | Description | Tuning guidance |
|-----------|-------------|-----------------|
| `--speculative-algorithm EAGLE3` | Enable EAGLE-3 mode | — |
| `--speculative-draft-model-path` | Path to EAGLE-3 draft head | Use models from `jamesliu1/sglang-EAGLE3-*` |
| `--speculative-eagle-topk` | Branching factor (K) per step | 4 for Llama/Grok; 1 for others |
| `--speculative-num-steps` | Depth of autoregressive drafting | 3 for most models |
| `--speculative-num-draft-tokens` | Total token budget for verification | 16 for Llama + EAGLE-3 |

The auto-tuning behavior (when all three are left unset) uses conservative defaults. For maximum throughput, set all three explicitly and tune using `bench_speculative.py`.

---

## Performance figures (from E2E Networks benchmarks)

| Model | Hardware | Method | Speedup |
|-------|----------|--------|---------|
| Llama-3.1-8B | 1× H100 | SGLang baseline | 158 tok/s |
| Llama-3.1-8B | 1× H100 | EAGLE-2 | 244 tok/s (+54%) |
| Llama-3.1-8B | 1× H100 | EAGLE-3 | 373 tok/s (+136%) |

These numbers match the SGLang docs — they're both quoting the same SGLang EAGLE-3 paper benchmark.

---

## How EAGLE relates to Layer 14

| Layer 14 concept | EAGLE-3 extension |
|-----------------|-------------------|
| `SpecRunner` with two `ModelRunner` instances | EAGLE-3 uses a draft **head** inside the target model, not a separate ModelRunner |
| `num_spec_tokens = k` (chain) | `--speculative-num-draft-tokens = 16` (tree, budget) |
| One forward pass verification | Tree attention — still one forward pass, but verifies a tree of candidates |
| `DraftReq.draft_kv_pool` | EAGLE-3 doesn't need a separate KV pool — it reuses target's KV through multi-layer feature extraction |
| `_kv_rewind()` on rejection | Still needed — rejected tree branches must have their KV pages reclaimed |
| Sequential chain (no branching) | Tree allows branching — effectively explores multiple futures simultaneously |

The Layer 14 STANDALONE mode is the foundation. EAGLE-3 is the production evolution that fixes its main limitation (top-layer-only, independent draft) while keeping the same acceptance/rejection framework.
