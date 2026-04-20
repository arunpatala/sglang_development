# Speculative Decoding and Beyond: An In-Depth Survey of Techniques

**Authors:** Yunhai Hu\*, Zining Liu\*, Zhenyuan Dong\*, Tianfan Peng\*, Bradley McDanel, Sai Qian Zhang  
**Institutions:** New York University, University of Pennsylvania, Shenzhen Institute of Information Technology, Franklin and Marshall College  
**arXiv:** 2502.19732 | **Submitted:** February 2025 | **Last revised:** October 2025 (v4)  
**PDF:** `03_hu2025_speculative_decoding_and_beyond.pdf`  
**URL:** https://arxiv.org/abs/2502.19732

---

## Why this survey is here

This is the **broadest and most current** of the four surveys. It extends speculative decoding into a general **generation-refinement framework** applicable to text, images, speech, and recommendation systems. It is also the only survey to give systematic treatment to **system-level implementations** (kernel design, batch processing, distributed execution). For anyone reading Layer 14 who wants to understand where SD fits in the larger ML efficiency landscape, this is the survey to consult.

---

## Summary

The paper reframes speculative decoding not as a single technique but as one member of a **generation-refinement framework family**. All methods in this family share a two-phase structure:

1. **Generation phase** — produce draft tokens via some strategy (n-gram, retrieval, draft model, etc.)
2. **Refinement phase** — verify/correct the draft (single-pass, iterative, or accept/reject)

This framing naturally unifies methods that existing surveys treated separately (Jacobi decoding, LOOKAHEAD, retrieval-augmented generation, SD, diffusion refinement).

---

## Taxonomy overview

The survey's full taxonomy (Figure 4) maps 100+ papers across:

### Generation strategies (Section IV)

| Strategy | Methods | Training required? |
|---|---|---|
| **Predefined fill tokens** | Jacobi, LOOKAHEAD, CLLMs | No (▲) / Fine-tune (■) |
| **Retrieval-based** | LLMA, REST, Speculative RAG | No (▲) / Fine-tune (■) |
| **N-gram-based** | ANPD, The N-Grammys, ADED | No (▲) |
| **Autoregressive (independent drafter)** | SpecDec, BiLD, OSD, DistillSpec, FastDraft | Distillation (◆) / No (▲) |
| **Autoregressive (dependent drafter, layer-skip)** | LayerSkip, Kangaroo, EESD, SPEED, Draft&Verify | Various |
| **Autoregressive (dependent drafter, FFN heads)** | EAGLE, Falcon, HASS, Hydra | Fine-tune (■) / Distill (◆) |
| **Multi-token prediction** | Blockwise, Medusa, Meta MTP, Amphista | Full training (∙) / PEFT (🟊) |

**Training notation:** ▲ = no training, ∙ = train from scratch, ■ = fine-tune, 🟊 = PEFT, ◆ = knowledge distillation

### Refinement strategies (Section V)

| Strategy | Examples |
|---|---|
| **Single-pass linear** | SpecDec, Leviathan/Chen speculative sampling, MTAD |
| **Single-pass tree-based** | SpecInfer, Sequoia, EAGLE, EAGLE-2, Medusa, ReDrafter |
| **Iterative refinement** | Jacobi decoding, CLLMs, consistency models |

---

## System-level implementations (Section VI)

This section is unique in the survey literature.

### Parallel speculative decoding

- **SPEED** — combines speculative execution with parameter sharing; multiple tokens processed through shared decoder
- **PPD / PASS** — pipeline-parallel approaches where draft and target run on separate GPU partitions
- **Ouroboros / ParallelSpec** — overlap draft generation with verification to hide latency

### Distributed speculative decoding

- **SpecExec** — designed for edge devices with limited bandwidth; minimizes data transfer between draft and target
- **EdgeLLM** — runs draft on CPU/edge, target on cloud GPU; asynchronous verification
- **Dovetail** — knowledge distillation for heterogeneous hardware deployment

### Compiler/hardware optimizations

- **SpecPIM** — Processing-In-Memory for draft model; eliminates HBM bandwidth bottleneck for draft
- **MagicDec** — sparse KV cache for draft model in long-context batched scenarios
- **BASS / SEED** — batch-aware scheduling that dynamically adjusts draft parameters based on real-time GPU utilization
- **PipeInfer** — pipeline parallelism for SD on multi-GPU systems

---

## Multi-modal extensions (Section VII)

This section has no equivalent in the other surveys.

### Vision (Section VII-A)

SD applied to image/video generation:
- Drafting low-resolution tokens, verifying at full resolution
- **LANTERN** — relaxed acceptance for vision where approximate correctness is acceptable
- **SJD** — Jacobi decoding adapted for diffusion models

### Multimodal LLMs (Section VII-B)

- **VADUSA** — Medusa variant for multimodal models (vision + language tokens)
- **IbED** — interleaved batch execution for vision-language speculative decoding

### Recommendation systems (Section VII-C)

- **AtSpeed** — SD for collaborative filtering (sequence recommendation)
- **DARE** — retrieval-augmented speculative decoding for recommendation

---

## Key insights not in other surveys

### 1. Iterative refinement as unified concept

Methods like Jacobi decoding and consistency models iterate over all positions simultaneously (not sequentially) and converge to the correct distribution. The survey shows this is another way to break sequential dependencies — distinct from draft-and-verify but complementary to it.

### 2. Training cost taxonomy

The `▲ ∙ ■ 🟊 ◆` notation throughout the paper lets readers immediately assess deployment feasibility:
- `▲` methods (no training) are immediately usable in SGLang NGRAM mode
- `◆` methods require distillation data from the target — significant setup cost
- `∙` methods need full retraining — typically only feasible for model providers

### 3. Distribution-preserving vs. approximate

The survey cleanly separates:
- **Lossless SD** — provably maintains target distribution (speculative sampling, token tree verification)
- **Approximate SD** — accepts some quality degradation for higher throughput (BiLD, early-exit without distillation)
- **Iterative refinement** — may not exactly match target distribution but converges to high-quality outputs

---

## Connections to Layer 14

| Survey topic | Layer 14 code |
|---|---|
| Training taxonomy (▲/∙/■/🟊/◆) | `lesson/10_eagle.md` EAGLE training section |
| System parallelism (SPEED, PPD) | `lesson/12_system_integration.md` |
| Iterative refinement (Jacobi, CLLMs) | `lesson/09_variants.md` non-AR alternatives |
| Multi-modal SD | future extension beyond Layer 14 scope |
| SpecPIM / edge SD | `lesson/08_production_gotchas.md` hardware constraints |

---

## What this survey adds over Surveys 01 and 02

| Dimension | Survey 01 (Xia) | Survey 02 (Ryu/Kim) | **This survey (Hu et al.)** |
|---|---|---|---|
| Cutoff | Jun 2024 | Nov 2024 | Oct 2025 (v4) |
| Scope | Text SD only | Text SD only | Text + image + speech + reco |
| System-level | Minimal | Deployment challenges | Dedicated section (kernels, distributed) |
| Iterative methods | Not covered | Not covered | Full section |
| Method count | ~50 | ~40 | 100+ |
| Taxonomy axis | Drafter / verification | Model-centric / draft-centric | Generation strategy / refinement strategy |

---

## Recommended reading order

Read **before** this survey:
- Survey 01 (Xia 2024) — the foundational definitions this survey builds on
- Survey 02 (Ryu/Kim 2024) — a gentler introduction to the dual taxonomy

Read **after** this survey:
- L4/07 (MagicDec) — implements the "Compiler/Hardware" MagicDec idea described in Section VI-C
- L5 original papers — for deep-dive into any specific method in the taxonomy
