# References — speculative decoding

Organized by **reading level** (L1–L5 from `WRITING_GUIDE/PERSONAS.md`) and **category**. Use this when writing or extending lesson content, locating production precedents, or designing exercises.

---

## Foundational papers

### Leviathan et al. — the original greedy/token-match version (ICML 2023)

- **Title:** Fast Inference from Transformers via Speculative Decoding
- **Authors:** Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research)
- **Venue:** ICML 2023, PMLR 202:19274–19286
- **HTML:** https://proceedings.mlr.press/v202/leviathan23a.html
- **PDF:** https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf
- **Level:** L3 spine / L4 correctness detail
- **What it contributes:**
  - Establishes the draft-then-verify idea for greedy decoding.
  - Proves output distribution is unchanged under token-match acceptance.
  - Reports 2–3× speedup on T5-XXL without model changes.
  - The acceptance rule in `lesson/06_accept_reject_rewind.md` is the greedy variant of this paper.

### Chen et al. — sampling-based variant and rejection-sampling correctness proof (2023)

- **Title:** Accelerating Large Language Model Decoding with Speculative Sampling
- **Authors:** Charlie Chen, Sebastian Borgeaud, Geoffrey Irving et al. (DeepMind)
- **Link:** https://arxiv.org/abs/2302.01318
- **Level:** L4 (correctness extension of the greedy rule)
- **What it contributes:**
  - The correct rejection-sampling rule for temperature > 0: compare full token distributions of draft and target at each position; resample from the corrected distribution at rejection site.
  - This is the extension described in `lesson/09_whats_next.md` as the "open direction" beyond Layer 14's greedy `_accept_reject`.
  - Proves output distribution of the target is exactly preserved under temperature sampling.

---

## Survey papers

### Xia et al. — comprehensive survey (ACL Findings 2024)

- **Title:** Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding
- **Authors:** Heming Xia et al. (HK Polytechnic, Peking University, Microsoft Research Asia, Alibaba)
- **Venue:** Findings of ACL 2024
- **arXiv:** https://arxiv.org/abs/2401.07851
- **ACL:** https://aclanthology.org/2024.findings-acl.456/
- **HuggingFace paper page:** https://huggingface.co/papers/2401.07851
- **Level:** L3–L4
- **What it contributes:**
  - Formal taxonomy of drafter strategies (draft models, n-gram, heads, self-speculation) and verification strategies.
  - Comparative benchmarks of leading methods in third-party environments.
  - Best starting point when designing which variant to add to Layer 14 next.
  - Maintained GitHub repository of related papers (linked from the arXiv page).

### Hu et al. — in-depth survey of generation-refinement frameworks (2025)

- **Title:** Speculative Decoding and Beyond: An In-Depth Survey of Techniques
- **Authors:** Yunhai Hu et al. (NYU, University of Pennsylvania)
- **arXiv:** https://arxiv.org/html/2502.19732v2
- **Level:** L4
- **What it contributes:**
  - Taxonomy organized by **generation strategy** (n-gram → draft model → tree) and **refinement mechanism** (single-pass vs iterative).
  - Covers system-level implementations across compute environments.
  - Extends beyond text to images and speech — useful for understanding where the field is going after Layer 14.

### Yan et al. — empirical study: draft model latency is the bottleneck (NAACL 2025)

- **Title:** Decoding Speculative Decoding
- **Authors:** Minghao Yan, Saurabh Agarwal, Shivaram Venkataraman (U. Wisconsin-Madison)
- **Venue:** NAACL 2025
- **arXiv:** https://arxiv.org/abs/2402.01528
- **ACL:** https://aclanthology.org/2025.naacl-long.328/
- **Level:** L4
- **What it contributes:**
  - 350+ experiments across LLaMA-65B and OPT-66B with draft models ranging from 5× to 528× smaller than target.
  - Key finding: **draft model latency** is the primary bottleneck, not language modeling quality — a direct counterpoint to the naive assumption that "better draft → more speedup."
  - Connects to Layer 14's `tokens_per_step` and `acceptance_rate` statistics: acceptance rate alone does not predict wall-clock speedup; draft cost does.
  - Proposed hardware-efficient draft model achieves 111% higher throughput than naive draft choice — good context for the `draft_kv_fraction` choices in `spec_runner.py`.

### Multimodal speculative decoding survey (2026)

- **Title:** Speculative Decoding for Multimodal Models: A Survey
- **Link:** https://www.preprints.org/manuscript/202603.2344
- **Level:** L4 / horizon reading
- **What it contributes:**
  - Covers VLMs, text-to-image, speech, and diffusion models.
  - Organizes by draft generation stage vs verification/acceptance stage — same two-phase structure as Layer 14.
  - Useful when thinking about where speculative decoding goes beyond LLM text inference.

---

## Production variants (beyond greedy single-draft)

### EAGLE / EAGLE-2 — tree-structured draft with dynamic budget

- **EAGLE-2 paper:** https://arxiv.org/html/2406.16858v1
- **EAGLE-3 overview:** https://www.e2enetworks.com/blog/Accelerating_LLM_Inference_with_EAGLE
- **Level:** L4
- **What it contributes:**
  - Lightweight draft head: target model's embedding + LM head + one trainable layer (much cheaper than a separate full model).
  - Dynamic draft trees: instead of always drafting N tokens linearly, builds a tree of candidates; acceptance rates adapt to context.
  - 3–4.26× speedup on LLaMA-class models; 20–40% faster than EAGLE-1.
  - Maps to `lesson/09_whats_next.md` — the "what comes next" beyond Layer 14's linear draft.
  - SGLang's `--speculative-algorithm EAGLE` / `EAGLE3` exposes this directly.

### Medusa — multiple draft heads on the same model

- **Hugging Face overview:** https://huggingface.co/docs/text-generation-inference/main/en/conceptual/speculation
- **Level:** L4
- **What it contributes:**
  - No separate draft model; multiple fine-tuned prediction heads on the target itself.
  - Medusa-1: frozen base model, up to 2.2× speedup. Medusa-2: joint fine-tuning, 2.3–2.8×.
  - Contrast to Layer 14: Layer 14 uses a separate `ModelRunner` for draft; Medusa folds draft heads into the target architecture.

### Self-speculative decoding (LayerSkip) — early layers as draft

- **Blog:** https://huggingface.co/blog/layerskip
- **Level:** L4
- **What it contributes:**
  - Uses the same model's early layers as the draft; no extra model memory or separate KV pool.
  - Contrast to Layer 14: Layer 14 needs two `KVPool` instances (target + draft); LayerSkip needs only one.

### Dynamic speculation — adaptive N at runtime

- **Blog:** https://hf.co/blog/dynamic_speculation_lookahead
- **Level:** L4
- **What it contributes:**
  - Adjusts N (draft steps per spec step) live based on observed acceptance rate.
  - If acceptance_rate < threshold → reduce N. If acceptance_rate > threshold → increase N up to configured max.
  - This is exactly the "dynamic N adjustment" described in `lesson/07_statistics.md` and `lesson/09_whats_next.md` as a Layer 14 extension exercise.
  - Default mode in HuggingFace Transformers ≥ 4.45.0.

---

## Explainer blogs and tutorials (L1–L2 reading)

| Level | Link | Why useful |
|-------|------|------------|
| L1 | [Chris Thomas (Feb 2025)](https://christhomas.co.uk/blog/2025/02/16/speculative-decoding-using-llms-efficiently/) | "Junior/senior developer" analogy; good first metaphor for L1 orientation. |
| L1 | [NVIDIA Technical Blog](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/) | Clean intro with diagrams; memory-bound framing. |
| L2 | [Google Research retrospective](https://research.google/blog/looking-back-at-speculative-decoding/) | Production deployment angle (Google AI Overviews); good for L2 "why it matters in practice." |
| L2 | [ML Mastery practitioner guide](https://machinelearningmastery.com/the-machine-learning-practitioners-guide-to-speculative-decoding/) | Covers acceptance rate, throughput math, practical configuration guidance. |
| L2–L3 | [Medium tutorial](https://medium.com/@buildingblocks/speculative-decoding-tutorial-007936be2bbb) | Step-by-step PyTorch + HuggingFace implementation walkthrough; good bridge from L2 to L3. |

---

## Production engine implementations (L4–L5 reading)

### SGLang

- **Docs:** https://docs.sglang.ai/backend/speculative_decoding.html
- **Source:** https://github.com/sgl-project/sglang (`srt/speculative/`)
- **Key files mapping to Layer 14:**
  - `srt/speculative/eagle_worker.py` ↔ `spec_runner.py` `SpecRunner`
  - `srt/speculative/eagle_utils.py` → `verify_input` ↔ `_accept_reject`
  - KV cache rollback in spec worker ↔ `_rewind_target_kv` / `_rewind_draft_kv`
  - `spec_verify_stats` ↔ `acceptance_rate` / `tokens_per_step` properties
- **Relevant CLI params:** `--speculative-algorithm`, `--speculative-num-draft-tokens`, `--speculative-num-steps`, `--speculative-eagle-topk`
- **Benchmark context (1× H100, LLaMA 3.1 8B):** baseline 158 tok/s → EAGLE-2 244 tok/s → EAGLE-3 373 tok/s

### vLLM

- **Speculators library:** https://github.com/vllm-project/speculators
- **What it covers:** Training + deployment framework for draft models; DFlash algorithm; integration with vLLM production stack.

---

## Blogs and articles — full list by level

### L1 — Orientation (no code, metaphor + tradeoff)

> **Downloaded:** All five L1 articles are available in `references/L1/`. See `references/L1/README.md` for reading order.

| Title | Link | Why useful |
|-------|------|------------|
| Chris Thomas (Feb 2025) | https://christhomas.co.uk/blog/2025/02/16/speculative-decoding-using-llms-efficiently/ | "Junior/senior developer" analogy; memory-bound framing; clean motivation. |
| NVIDIA Technical Blog | https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/ | Authoritative intro with diagrams; GPU memory-bandwidth framing. |
| Google Research retrospective | https://research.google/blog/looking-back-at-speculative-decoding/ | Production deployment (AI Overviews); "what it felt like shipping this." |
| Nebius: MoE + speculative decoding in production | https://nebius.com/blog/posts/moe-spec-decoding | Why large MoE models break latency budgets; spec decoding as a fix; P99 tail latency reality. Good human-stakes framing for L1. |
| Google Cloud: five inference techniques | https://cloud.google.com/blog/topics/developers-practitioners/five-techniques-to-reach-the-efficient-frontier-of-llm-inference | Positions spec decoding in the wider inference optimization landscape; good orientation. |

### L2 — Definitions + motivation (terms re-anchored, small examples, minimal code)

> **Downloaded:** All nine L2 articles are available in `references/L2/`. See `references/L2/README.md` for reading order.

| Title | Link | Why useful |
|-------|------|------------|
| ML Mastery practitioner guide | https://machinelearningmastery.com/the-machine-learning-practitioners-guide-to-speculative-decoding/ | Covers acceptance rate, throughput math, configuration guidance in plain language. |
| Jarvis Labs: speculative decoding in vLLM | https://jarvislabs.ai/blog/speculative-decoding-vllm-faster-llm-inference | Walks through the guess-check-keep process with concrete token examples; good bridge to L3. |
| HuggingFace TGI speculation conceptual docs | https://huggingface.co/docs/text-generation-inference/main/en/conceptual/speculation | Covers Medusa, n-gram, and draft model variants side by side; good for comparing vocabulary. |
| Dynamic speculation blog (Intel + HF) | https://hf.co/blog/dynamic_speculation_lookahead | Adaptive N explained with example; ties directly to `lesson/07_statistics.md`. |
| HuggingFace: self-speculative decoding (LayerSkip) | https://huggingface.co/blog/layerskip | One-model variant clearly explained; useful contrast before introducing the two-`ModelRunner` architecture. |
| Reddit r/LocalLLaMA interactive explainer | https://www.reddit.com/r/LocalLLaMA/comments/1r9vsye/nice_interactive_explanation_of_speculative/ | Community-shared interactive visualization; supplementary. (See entries below for better standalone versions.) |
| Adaptive ML: Speculative Decoding, Visualized | https://www.adaptive-ml.com/post/speculative-decoding-visualized | Best visual treatment of probability geometry (green/red acceptance regions); distribution-preservation proof. Jan 2026. |
| Pierce Freeman: How speculative decoding works | https://pierce.dev/notes/how-speculative-decoding-works/ | Most concise explainer; best/worst/average case triad; "you only need the draft right some of the time." |
| Brenndoerfer: Interactive guide (55 min) | https://mbrenndoerfer.com/writing/speculative-decoding-accelerating-llm-inference | Most complete L2-L3 article: quantified memory bottleneck, full acceptance criterion code, speedup calculator, causal masking. Jan 2026. |
| Data Processing Club: Speculative Decoding Explained (Ryoma Sato) | https://data-processing.club/speculative/ | Cleanest proof that speculative decoding is exact (SpecJudge / SpecNormalize terminology). Also covers DistillSpec, Online SD, Draft & Verify, LayerSkip, tree attention, Medusa — best single survey of extensions at L2 depth. The "bigram gives 1.25×" fact is the most surprising L2 insight. |

### L3 — Book spine level (pseudocode, mechanism, invariants)

> **Downloaded:** All four L3 articles are available in `references/L3/`. See `references/L3/README.md` for reading order and Layer 14 lesson mapping.

| # | Title | Link | Local file | Why useful |
|---|-------|------|-----------|------------|
| 01 | PyTorch: A Hitchhiker's Guide to Speculative Decoding | https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/ | `references/L3/01_pytorch_hitchhikers_guide.md` | Production deployment + MLP Speculator architecture + two-phase training. Real benchmark: 2× Llama2 13B, 3× Granite 20B Code. |
| 02 | Medium tutorial (Building Blocks) | https://medium.com/@buildingblocks/speculative-decoding-tutorial-007936be2bbb | `references/L3/02_medium_building_blocks_tutorial.md` | Step-by-step greedy implementation in PyTorch + HuggingFace; "what lesson/04–06 looks like in userland code." (Member-only; preview + reconstruction.) |
| 03 | AWS: speculative decoding on Trainium + vLLM | https://aws.amazon.com/blogs/machine-learning/accelerating-decode-heavy-llm-inference-with-speculative-decoding-on-aws-trainium-and-vllm/ | `references/L3/03_aws_trainium_vllm_speculative_decoding.md` | Best treatment of the two tuning levers: draft model selection + `num_speculative_tokens`. Best-case vs worst-case prompt analysis. |
| 04 | Brenndoerfer: Speculative Decoding Math — Algorithms & Speedup Limits | https://mbrenndoerfer.com/writing/speculative-decoding-math-acceptance-criterion | `references/L3/04_brenndoerfer_speculative_decoding_math.md` | Full mathematical derivation of acceptance criterion α(x)=min(1,p/q); residual distribution proof; speedup formula S=E[N]/(1+k/c); optimal K table; runnable NumPy code. 53 min. |

### L4 — Production + engines (real stacks, gotchas, tradeoffs)

> **Downloaded:** All eight L4 articles are available in `references/L4/`. See `references/L4/README.md` for reading order and Layer 14 lesson mapping.

| # | Title | Link | Local file | Why useful |
|---|-------|------|-----------|------------|
| 01 | SGLang speculative decoding docs | https://docs.sglang.ai/advanced_features/speculative_decoding.html | `references/L4/01_sglang_speculative_decoding_docs.md` | Primary production reference. All CLI flags, all algorithms (EAGLE-2/3/MTP/STANDALONE/NGRAM). Benchmark: 158 → 373 tok/s on H100. |
| 02 | vLLM speculative decoding docs (v0.8.5) | https://docs.vllm.ai/en/v0.8.5/features/spec_decode.html | `references/L4/02_vllm_speculative_decoding_docs.md` | Parallel engine; losslessness guarantees and floating-point caveats; draft TP constraint; n-gram and EAGLE configs. |
| 03 | SpecDecode-Bench: "Speculative Decoding: Performance or Illusion?" | https://specdecode-bench.github.io/ | `references/L4/03_specdecode_bench_performance_or_illusion.md` | 5 SD variants × 4 models × 6 datasets × batch 1–128 on H100. Speedups shrink with batch size; verification dominates (42–95%); chain > tree at batch > 32; n-gram wins code (BLEU-4 > 0.6); oracle 4.9× ceiling. |
| 04 | Red Hat: Speculators — production-ready spec decoding | https://developers.redhat.com/articles/2025/11/19/speculators-standardized-production-ready-speculative-decoding | `references/L4/04_redhat_speculators_production.md` | Standardized HuggingFace speculator format; vLLM one-liner deploy; EAGLE-3 models for Llama/Qwen3/Llama-4-Maverick. |
| 05 | Baseten: speculative decoding engine builder | https://www.baseten.co/blog/speculative-decoding-engine-builder-integration/ | `references/L4/05_baseten_speculative_decoding_engine_builder.md` | Complete YAML config; GPU memory allocation (`kv_cache_free_gpu_mem_fraction`); when NOT to use SD. |
| 06 | EAGLE-3 overview (E2E Networks) | https://www.e2enetworks.com/blog/Accelerating_LLM_Inference_with_EAGLE | `references/L4/06_eagle3_overview_e2e_networks.md` | Deepest treatment of EAGLE → EAGLE-2 → EAGLE-3; multi-layer feature fusion; training-time testing; tree attention mask; SGLang config tuning. |
| 07 | MagicDec: long context + speculative decoding | https://arxiv.org/html/2408.11049v5 | `references/L4/07_magicdec_long_context_speculative_decoding.md` | Sparse KV self-speculation; critical sequence length S_inflection; 2.51× speedup at batch=32–256 for 32K+ context. |
| 08 | QuantSpec: quantized KV cache self-speculative | https://arxiv.org/html/2502.10424v1 | `references/L4/08_quantspec_quantized_kv_self_speculative.md` | Hierarchical INT4/INT8 KV sharing eliminates draft pool; double FP buffer solves KV rollback; >90% acceptance rate; 2.49× at 128K context. |

### L5 — Build track (runnable code, training, contribution)

| Title | Link | Why useful |
|-------|------|------------|
| suryavanshi/speculative_decoding (GitHub) | https://github.com/suryavanshi/speculative_decoding | HuggingFace-compatible `SpeculativeDecoder` class + benchmark script; swap models easily. |
| romsto/Speculative-Decoding (GitHub) | https://github.com/romsto/Speculative-Decoding | Clean standalone PyTorch reference; comparison point for `spec_runner.py` design choices. |
| shreyansh26/Speculative-Sampling (GitHub) | https://github.com/shreyansh26/Speculative-Sampling | Reference implementation of Chen et al. (DeepMind) speculative sampling, including the full stochastic acceptance criterion and residual distribution. Canonical companion to the paper. Contrast with `spec_runner.py`'s greedy variant. |
| HuggingFace transformers: `generation/utils.py` | https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py | Production source of `assisted_decoding()` (HF's speculative decoding). The L5 companion to `02_jarvislabs_speculative_decoding_vllm.md`. Search for `_assisted_decoding` to see how acceptance, KV cache extension, and dynamic N scheduling are implemented in the real HF library. |
| PyTorch TorchSpec: training at scale | https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/ | Disaggregated training architecture (inference engines → hidden state store → training workers); trained Kimi K2.5 EAGLE-3 draft on 1,500 H200 GPU-hours; +60% throughput at batch 1. Good for L5 extension: "how would you train the draft model used in Layer 14?" |
| torchspec-project/TorchSpec (GitHub) | https://github.com/torchspec-project/TorchSpec | Source for the above; examples for Kimi-K2.5, MiniMax-M2.5, Qwen3-Coder-Next. |
| vLLM Speculators library | https://github.com/vllm-project/speculators | Standardized training + deployment framework for speculator models; DFlash algorithm. |

---

## Reference implementations from scratch

- **romsto/Speculative-Decoding:** https://github.com/romsto/Speculative-Decoding — Level L5
- **suryavanshi/speculative_decoding:** https://github.com/suryavanshi/speculative_decoding — Level L5

---

## How this maps to the hierarchical model

| Cluster | Key references |
|---------|---------------|
| **14a why** (lesson/01) | Leviathan §1–2; Chris Thomas (L1); NVIDIA blog (L1); Nebius MoE blog (L1); PyTorch Hitchhiker's Guide (L3); Adaptive ML Visualized (L2) |
| **14b architecture** (lessons/02–03) | Leviathan §3; LayerSkip contrast (L2); Jarvis Labs (L2); EAGLE-2 paper (L4); QuantSpec / MagicDec KV tradeoffs (L4) |
| **14c loop** (lessons/04–06) | Leviathan §3–4; Chen et al. (L4); shreyansh26 impl (L5); Brenndoerfer math (L3); PyTorch Hitchhiker's Guide (L3); Medium tutorial (L3); HF utils.py (L5) |
| **14d measurement + next** (lessons/07–09) | Yan et al. latency bottleneck (L4); SpecDecode-Bench (L4); Dynamic speculation blog (L2); SGLang docs (L4); vLLM docs (L4); EAGLE-3 overview (L4); TorchSpec training (L5); Xia et al. benchmarks (L3–L4) |

---

## See also

- `WRITING_GUIDE/PERSONAS.md` — which reference depth fits which reader level.
- `WRITING_GUIDE/HIERARCHICAL.md` — how these references attach to topic nodes as L1–L5 artifacts.
- `lesson/00_outline.md` — full section list with code anchors into `spec_runner.py`.
