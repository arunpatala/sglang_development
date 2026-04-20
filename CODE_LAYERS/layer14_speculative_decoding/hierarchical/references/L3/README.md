# L3 References: Speculative Decoding

**Level:** L3 — Book spine level (pseudocode, mechanism, invariants)

**Reader profile:** Comfortable with attention mechanisms, softmax, KV cache basics. Ready to work through pseudocode, step through algorithm variants, and see real benchmark numbers. Satisfied when they can predict *why* a configuration would be faster or slower, not just *that* it is.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_pytorch_hitchhikers_guide.md` | PyTorch / IBM Research | Production deployment + MLP Speculator architecture + two-phase training methodology. Real benchmark: 2× Llama2 13B, 3× Granite 20B Code. |
| 02 | `02_medium_building_blocks_tutorial.md` | Building Blocks / Medium | Step-by-step greedy implementation in PyTorch + HuggingFace. Best "implement it yourself before reading Layer 14" article. (Member-only; preview + reconstruction.) |
| 03 | `03_aws_trainium_vllm_speculative_decoding.md` | AWS Machine Learning Blog | Two-lever analysis: draft model selection + `num_speculative_tokens` tuning. Unique: shows best-case (structured) vs worst-case (open-ended) prompts in the same benchmark. |
| 04 | `04_brenndoerfer_speculative_decoding_math.md` | Michael Brenndoerfer (Jan 2026) | Full mathematical derivation: acceptance criterion `α(x)=min(1,p/q)`, residual distribution proof, `E[N]` formula, speedup formula `S=E[N]/(1+k/c)`, optimal k derivation, runnable NumPy code. 53 min. |

---

## Recommended reading order

**Before Layer 14 (warm-up path, ~60 min):**
02 → 01

- **02 first:** Implement the greedy case yourself using GPT-2 and HuggingFace. Locks in the draft/verify/accept loop as lived experience.
- **01 next:** See how IBM extended this to production — multi-head speculators, modified KV kernels, and two-phase training.

**After Layer 14 lesson/04–06 (deepen path, ~90 min):**
04 → 03

- **04 after accept/reject:** The full math of `α(x) = min(1, p/q)` and the residual distribution proves why the Layer 14 code is correct. The speedup formula gives you the analytical tools to predict if your config is optimal.
- **03 for production tuning:** After understanding the mechanism, the AWS article's two-lever analysis (draft model size vs `num_speculative_tokens`) gives you the practical playbook for deployment decisions.

**Full L3 path (sequential, ~2.5 hours):** 02 → 01 → 04 → 03

---

## How these map to Layer 14 lessons

| Layer 14 lesson | Most relevant L3 reference |
|-----------------|---------------------------|
| `01_from_one_to_n_plus_one.md` | 01 (IBM's framing of memory-bandwidth bottleneck) |
| `02_spec_runner_architecture.md` | 01 (MLP Speculator = alternative architecture to two-ModelRunner) |
| `03_draft_kv_mirroring.md` | 01 (modified paged attention kernel; KV sharing across heads) |
| `04_draft_phase.md` | 02 (greedy draft loop implementation) |
| `05_verify_extend.md` | 02 (batch verification in one forward pass), 04 (E[N] formula) |
| `06_accept_reject_rewind.md` | 04 (acceptance criterion + residual distribution proof; KV rollback gotcha) |
| `07_statistics.md` | 04 (optimal k table; α = empirical E[N] parameter), 03 (adaptive k strategies) |

---

## What L3 adds that L2 didn't cover

| Topic | L2 left it as... | L3 explains it as... |
|-------|-----------------|----------------------|
| Acceptance criterion `min(1, p/q)` | "A formula that ensures correctness" | Derived from rejection sampling; residual distribution proved necessary |
| Two models sharing the target's KV cache | Mentioned in passing | Modified paged attention kernel; no per-head KV duplication |
| Speculator training | Not mentioned | Two-phase: causal LM pre-training → base model output distillation |
| Why k=5 is the common default | "Works well in practice" | First-order condition from the speedup formula; typical optimal k ∈ [4, 8] |
| Structured vs open-ended prompts | Not distinguished | Acceptance rate drops on creative text; speculative decoding degrades to baseline |
| Draft model size sweet spot | "Use a smaller model from the same family" | Quantified tradeoff: medium draft (7B) outperforms large draft (30B) despite lower α |

---

## Common L3 limits to name for readers

These articles **do not fully explain:**
- How SGLang's `SpecRunner` schedules two separate `ModelRunner` instances with different request queues
- KV pool fragmentation and `kv_memory_fraction` tuning in practice
- P99 tail latency characteristics under continuous batching
- EAGLE and tree-based speculation (EAGLE achieves higher α by conditioning the draft on target hidden states — see L4 and the EAGLE papers in `REFERENCES.md`)
- How to train your own speculator using TorchSpec or the IBM fms-fsdp recipe

Those live in L4 (production engine docs, EAGLE papers) and L5 (reference implementations, source code).
