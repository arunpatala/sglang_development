# Blog Plan: Three-Level Treatment of Speculative Decoding

## The three blogs

| Blog | File | Audience | Tone | Length |
|------|------|----------|------|--------|
| L1 | `blog_l1_intuition.md` | Product people, adjacent engineers, curious developers who use LLM tools but don't build stacks | Conversational, analogy-driven, no code, no math | ~1,000 words |
| L2 | `blog_l2_mechanism.md` | Practitioners — engineers who configure or evaluate LLM deployments | Step-by-step mechanism, some numbers, light config, production framing | ~1,800 words |
| L3 | `blog_l3_technical.md` | Builders — ML engineers implementing or auditing speculative decoding systems | Algorithm + math, hardware analysis, EAGLE/MTP, code, production architecture | ~2,500 words |

**Shared principle:** No duplicated explanations across levels. Re-emphasis is allowed (a key concept may be stated at all three levels but with increasing precision). Content unique to one level stays there.

---

## Section-by-section mapping

Each row maps one section of `COMBINED.md` to what appears in each blog level.

| COMBINED.md Section | L1: Intuition Blog | L2: Mechanism Blog | L3: Technical Blog |
|---------------------|-------------------|-------------------|-------------------|
| **1. Why LLMs are slow** | "Like reading an encyclopedia cover-to-cover to write one word" — one sentence on memory-bound; no numbers | Memory-bound decode explained with bandwidth vs FLOPS gap (qualitative); the arithmetic intensity gap named | Roofline model framing; HBM bandwidth specific numbers; decode-phase arithmetic intensity < 1 FLOP/byte |
| **2. Two observations** | Easy tokens exist; idle compute exists — both as plain-language insight | Quantify what "easy" means (small model matches large model most of the time); idle CUDA cores as the target resource | How the memory bottleneck creates systematic underutilization; arithmetic intensity during decode vs prefill |
| **3. Analogies** | Junior/senior developer in full; one-line mention that other analogies exist | Scientist/assistant analogy to connect to pipeline concept; CPU speculative execution briefly | Analogies dropped — replaced by the algorithm |
| **4. How it works** | "Draft, verify, commit" in one paragraph — no mechanics exposed | Three steps in full: draft generation, parallel single-pass verification, accept/reject per position; KV cache efficiency mentioned | Rejection sampling probability rule; the bonus token; greedy vs sampling variants; correctness proof sketch |
| **5. Speedup numbers** | "2–4× faster, same quality" with one Google example | Why the range varies; acceptance rate as the governing variable; worst-case is identical to baseline | Theoretical speedup formula as function of acceptance rate α and draft length γ; empirical calibration |
| **6. Three efficiency factors** | Named only: draft size, draft length, acceptance rate | Tradeoff explained for each; typical values given (3–8B draft, 3–12 tokens, 70–85% acceptance) | Optimal draft length derivation; acceptance rate estimation before deployment |
| **7. Hardware angle** | Omitted | Why verification is cheap: KV cache covers most; draft model runs in parallel | FLOPS/byte for decode phase; why H100 is still underutilized; what concurrency gains are actually available |
| **8. Advanced variants** | One sentence: "More sophisticated variants exist — EAGLE, MTP — the mechanism improves but the idea stays the same" | EAGLE: operates on hidden states, no separate model; MTP: multi-head baked into model; tree attention concept | EAGLE-3 architecture (multi-layer fused features, dynamic tree, instance-adaptive); MTP joint training; tree attention implementation |
| **9. Real products / P99** | "Streaming demos hide problems; non-streaming products feel every millisecond" | P99 vs average; cascaded systems compound budgets; replicas don't fix the tail; speculative decoding reshapes the distribution | Queueing theory angle; prefill dominance numbers for long-context; draft model training as post-training infrastructure |
| **10. Where deployed today** | Google AI Overviews, LM Studio, local coding tools — emphasis on "it's already in your hands" | Deployment landscape; framework support (vLLM, SGLang, llama.cpp, TensorRT) | Production deployment considerations: KV cache budget split, draft/target alignment, serving topology |
| **11. Practical setup / code** | Mention VS Code + llama.cpp demo as concrete proof it works locally | `-md` / `-m` flags; what acceptance threshold does; accept/reject UI shortcuts | TensorRT EAGLE-3 config code; SGLang equivalent; what `convert()` does at the layer level |
| **12. Analogy limits** | Full section — most important mental correction for L1 readers; rejected tokens are invisible | Brief: connect the limit to the mechanism ("the acceptance is probabilistic, not editorial") | Omitted — the algorithm has replaced the analogy |
| **13. Key quotes** | 2–3 quotes (Chris Thomas, Google Research) | 2 quotes (Google Research correctness guarantee, Nebius P99 framing) | 1 quote (Google Research correctness guarantee for the math section) |

---

## Content uniqueness rules across blogs

- **L1 only:** Full junior/senior analogy; VS Code demo mention; "already in your hands" deployment framing; full analogy-limits section; three-word topic summaries
- **L2 only:** Three-step walkthrough; acceptance rate as the governing metric; P99 / cascaded systems full treatment; deployment framework survey; config flags
- **L3 only:** Rejection sampling math; hardware roofline framing; EAGLE-3 architecture details; MTP joint training; optimal draft length derivation; production code
- **Shared (re-emphasis):** The correctness guarantee ("identical output distribution") appears at all three levels — L1 as a claim, L2 as a mechanism, L3 as a proof sketch

---

## What is deliberately omitted from all three blogs

- KV rewind and page management on rejection — L4/L5 book content only
- Draft model knowledge distillation training details — L5 only
- Batch size interaction with speculative decoding speedup — brief note in L3 only
- Missing `05_google_cloud_five_techniques.md` content — cannot synthesize from absent source
- Specific Google production speedup numbers — not publicly disclosed
