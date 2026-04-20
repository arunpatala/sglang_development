# SpecDecode-Bench: "Speculative Decoding: Performance or Illusion?"

**Source:** https://specdecode-bench.github.io/ | arXiv: 2601.11580
**Authors:** Xiaoxuan Liu, Jiaxiang Yu, Jongseok Park, Ion Stoica, Alvin Cheung (UC Berkeley / Sky Computing Lab)
**Published:** 2026
**Level:** L4 — First systematic production-grade benchmark of speculative decoding; 5 variants × 4 models × 6 datasets × batch 1–128 on H100
**Why here:** The most important empirical answer to "does speculative decoding actually work in production?" Debunks the 3–4× speedup numbers cited in most papers (which test at batch=1 only). Provides the clearest analysis of when each SD method wins, why verification dominates execution time, and why adaptive combining could reach 4.9× on code workloads.

---

## TL;DR (from the paper)

1. SD works in production, but **speedups shrink as batch size grows** — the system becomes compute-bound.
2. **Verification is the bottleneck** — dominates 42–95% of runtime.
3. **No single method wins everywhere:** EAGLE-3 is best all-around; n-gram wins code editing; draft-model wins on large targets.
4. **Chain > tree** for most practical batch sizes.
5. **Acceptance behavior varies wildly** across token positions, requests, and datasets.
6. **Oracle analysis:** 4.9× theoretical ceiling on code-editing workloads via adaptive combining — current methods are far below this.

---

## Why this study exists

Prior evaluations had three critical shortcomings:
- **Research prototypes** — not production-grade systems (no CUDA graphs, no continuous batching)
- **Batch size = 1** — unrealistic, inflates speedup numbers
- **No systematic comparison** — practitioners had no guidance on which method to use for which workload

This study fixes all three: vLLM v0.10.1/v0.11.1, full production optimizations, realistic batches.

---

## Experimental setup

| Dimension | Coverage |
|-----------|---------|
| SD variants | n-gram, EAGLE, EAGLE-3, Draft-Model, MTP |
| Models | Llama-3.1-8B, Llama-3-70B, Qwen3-8B, GLM-4.5-Air-106B |
| Datasets | CNN/DailyMail, ShareGPT, InstructCoder, GSM8K, AIME22–24, GPQA-Main |
| Batch sizes | 1 to 128 (up to 512 for profiling) |
| Hardware | NVIDIA H100 80GB (1 GPU for 8B, 4 GPUs TP for 70B/106B) |
| Engine | vLLM with all default optimizations: KV cache management, continuous batching, chunked prefill, CUDA Graph |

Primary metric: **token throughput** (generated tokens/sec). Speedup = throughput with SD / throughput without SD.

Draft tokens per step: k=3 for all methods (does not include the bonus token).

---

## Key finding 1: Speedups shrink with batch size

At batch=1, EAGLE achieves up to **1.96× speedup** on Llama-3-70B.
By batch=128, this drops to **1.21×** for Llama-3.1-8B on GSM8K.

**Why:** At small batches, the GPU has idle compute to spare for speculation. At large batches, the system is already compute-saturated — speculative verification competes with the regular forward pass for compute resources.

**Amplified for larger models:** Going from batch=1 to batch=32, speedup reduction is:
- 8B model: 4.3% reduction
- 70B model (4 GPUs): 14.0% reduction

The 70B model on 4 GPUs is already more compute-bound at small batches, so the batch size effect hits harder.

**Implication for Layer 14:** The `lesson/07_statistics.md` tracking of `acceptance_rate` matters for a second reason beyond token generation efficiency — at high batch sizes, even perfect acceptance doesn't help much because verification cost saturates compute.

---

## Key finding 2: Method selection guide

| Scenario | Best method | Why |
|----------|-------------|-----|
| Large target model (70B+) | Draft-Model | Draft model's forward pass is only ~12.5% of target's — speculation is cheap |
| 8B target model | EAGLE-3 or EAGLE | Draft forward pass is ~37.5% of target's — draft-model becomes expensive |
| All-around | EAGLE-3 | Consistent across all settings |
| Code editing (BLEU-4 > 0.6) | n-gram | Prompt-output overlap drives free speculation |
| Math reasoning | EAGLE-3 | n-gram barely helps; structured reasoning has low n-gram overlap |

**The draft-model scaling insight:**
- 70B target: draft-model forward pass = ~12.5% of target time → cheap speculation
- 8B target: draft-model forward pass = ~37.5% of target time → expensive speculation
- This flips which method wins, depending on the target model size.

---

## Key finding 3: Verification dominates (42–95% of runtime)

Time breakdown across all configurations:

| Method | Verification % of runtime |
|--------|--------------------------|
| n-gram (any model at large batch) | ~95% (drafting is essentially free) |
| Draft-model (Qwen3-8B, batch=1) | ~42% (drafting takes ~47%) |
| EAGLE / EAGLE-3 | 60–80% across most settings |

**Critical implication:** Rejection sampling overhead is **negligible** (<1.7% of runtime). The bottleneck is the target model verification pass, not the sampling math.

> "Every rejected token wastes verification compute. When acceptance rates are low and batch sizes are large, this wasted work becomes the primary bottleneck."

This is the strongest argument for adaptive draft length — see the oracle analysis below.

---

## Key finding 4: Tree vs chain verification

Tested in SGLang v0.5.9 (not vLLM — vLLM's tree path was not sufficiently optimized for fair comparison):

| Configuration | Speedup (Qwen3-8B/GSM8K, batch=1) | Accepted length | Acceptance rate |
|--------------|-----------------------------------|-----------------|-----------------|
| Chain k=3 | 1.65× | 2.25 | 0.415 |
| Tree k=6 (branching=2) | 1.75× | 2.50 | ~0.25 |
| Tree k=21 (branching=4) | 1.85× | 2.92 | 0.095 |

**At batch=64:** The k=21 tree **falls below 1.0×** on all workloads. Wider trees fail at scale.

**Why:** Trees increase accepted length per step, but acceptance rate drops sharply (0.415 → 0.095). The extra verification work on rejected branches is wasted at large batch sizes where verification already dominates.

**Conclusion: Chain-style verification is the more robust choice for production** at realistic batch sizes. This is why Layer 14 uses a simple chain (K draft tokens, one verification pass) rather than a tree.

---

## Key finding 5: Acceptance behavior varies wildly

Acceptance rate is not a single number — it varies within a request, across requests, and across datasets.

### Within a request
- **Reasoning workloads (AIME):** Both n-gram and EAGLE-3 accept more tokens as generation progresses (the model accumulates repetitive step-by-step patterns)
- **n-gram drops near the end:** When the model shifts from reasoning steps to writing a final answer, n-gram loses its advantage

### Across requests (same dataset, InstructCoder, Llama-3-70B)
| Method | Per-step accepted span range |
|--------|------------------------------|
| EAGLE | 2.7–7.4 tokens (consistent) |
| n-gram | 1.1–15.0 tokens (bursty — low usually, high on repetitive code) |
| Draft-model | 5.6–18.3 tokens |

### Across datasets
Each method has a niche. EAGLE is the most stable. n-gram has heavy-tailed distributions.

---

## Key finding 6: When n-gram beats EAGLE

n-gram wins when **BLEU-4 score > ~0.6** between prompt and output (i.e., when 60%+ of output 4-grams appear in the prompt):
- Code editing tasks (InstructCoder): outputs repeat large spans from the input
- Passage rewriting: output rewrites the prompt with minimal changes
- Template generation: output fills in a template from the prompt

At BLEU-4 > 0.6, n-gram achieves **up to 100% higher speedup** than EAGLE/EAGLE-3. With k=5, the advantage grows further.

**Practical test:** Compute BLEU-4 between your prompt and expected output. If > 0.6, use n-gram. Otherwise, use EAGLE-3.

---

## Oracle analysis: How far from optimal?

An oracle simulator that knows the exact acceptance span at every step and proposes exactly that many tokens gives the theoretical upper bound.

**Results for InstructCoder, Llama-3.1-8B, batch=1:**
| Method | Speedup |
|--------|---------|
| Best fixed k (n-gram, k=5) | ~2.1× |
| Oracle (n-gram) | ~2.75× |
| Oracle Combine (best method per step) | **4.9×** |

The oracle reveals:
- Fixed-k configurations waste verification on rejected tokens — **30% below oracle** even at batch=1
- The gap widens at larger batch sizes
- The Oracle Combine strategy (pick n-gram or EAGLE at each token position, propose exactly the right k) achieves 4.9× — 2.3× better than the best single method

---

## Memory overhead

| Method | GPU memory overhead |
|--------|---------------------|
| n-gram | 0% (no additional model) |
| EAGLE/EAGLE-3 | <10% (weights + per-token KV cache) |
| Draft-model (Qwen3-0.6B + Qwen3-8B) | +77% per-token memory (1.77× baseline) |

Draft-model methods incur the most overhead — both the draft model weights and a separate KV cache pool. This is the Layer 14 constraint: `kv_memory_fraction` splits available VRAM between the two pools, and setting it wrong causes OOM.

---

## Research directions implied by the results

1. **Adaptive proposal length** — choose k based on recent acceptance history (Layer 14's `lesson/07_statistics.md` enables this)
2. **Method selection at runtime** — pick n-gram vs EAGLE per token position based on prompt-output overlap predictor
3. **Oracle Combine implementation** — build a lightweight predictor that achieves the 4.9× theoretical ceiling
4. **Verification cost reduction** — reject fewer tokens or verify them more cheaply (the 42–95% bottleneck)

---

## How this maps to Layer 14

| SpecDecode-Bench finding | Layer 14 implication |
|--------------------------|---------------------|
| Speedups shrink with batch size | `lesson/07_statistics.md`: track batch size alongside acceptance_rate |
| Verification = 42–95% of runtime | The target model forward pass is the true bottleneck; KV rewind is cheap by comparison |
| EAGLE-3 > EAGLE-2 > chain at batch=1 | `lesson/08_eagle.md`: why EAGLE was built on top of Layer 14's architecture |
| Chain > tree at batch > 32 | Layer 14 correctly uses chain speculation (k draft tokens, not a tree) |
| n-gram wins at BLEU-4 > 0.6 | NGRAM mode in production; no `ModelRunner` needed |
| Oracle gap = 4.9× potential | Dynamic speculation (`lesson/07`) is step 1 toward closing this gap |
| Rejection sampling < 1.7% overhead | The `_accept_reject()` code is not the bottleneck; verification cost is |
