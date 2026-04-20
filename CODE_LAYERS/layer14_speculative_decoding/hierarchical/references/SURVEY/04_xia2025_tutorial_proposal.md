# Tutorial Proposal: Speculative Decoding for Efficient LLM Inference

**Authors:** Heming Xia, Cunxiao Du, Yongqi Li, Qian Liu, Wenjie Li  
**Institutions:** Hong Kong PolyU, Sea AI Lab  
**arXiv:** 2503.00491 | **Submitted:** March 2025  
**PDF:** `04_xia2025_tutorial_proposal.pdf`  
**URL:** https://arxiv.org/abs/2503.00491

---

## Why this survey is here

This is a **tutorial proposal** submitted to an NLP conference (likely ACL/EMNLP/NAACL 2025), not a traditional survey paper. It is the **most accessible entry point** into the SD literature — designed for an audience with only basic ML knowledge. As a structured 2.5-hour tutorial by the author of the first comprehensive SD survey (Survey 01 in this directory), it presents a curated, pedagogically ordered path through the field that maps almost exactly to the progression in Layer 14's lessons.

This document is small (188 KB PDF) but high-value: it shows **what the experts consider the essential learning path** for someone entering the SD field.

---

## Summary

The tutorial proposes a 2.5-hour session covering SD from foundations to cutting-edge methods. Its structure is explicitly designed for newcomers who are "interested in LLM efficiency, sparsity, and computational parallelism."

---

## Tutorial outline (Section 3)

| Segment | Duration | Content |
|---|---|---|
| **1. Introduction** | 15 min | Overview; benefits of SD |
| **2. Preliminaries** | 15 min | Autoregressive decoding; memory bottleneck in LLM inference |
| **3. Taxonomy** | 40 min | Formal definition; draft architectures; verification strategies |
| **4. Cutting-edge algorithms** | 40 min | EAGLE, EAGLE-2; adaptive tree verification; draft alignment |
| **5. Metrics and evaluations** | 20 min | EDL; fair comparison across scenarios |
| **6. Downstream adaptations** | 20 min | RAG-SD; long-context SD; multimodal SD |
| **7. Demonstration** | 10 min | Hands-on exercise showing how SD works |
| **8. Conclusions** | 10 min | Strengths; challenges; future directions |

---

## Key pedagogy points

### Three factors for speedup (Section 1)

The tutorial crystallizes SD efficiency into three levers:
1. **Inference efficiency of the draft model** — fast drafting with low FLOPs
2. **Acceptance rate** — fraction of draft tokens the target accepts
3. **Average token acceptance length** — EDL; the combined effect of both above

This matches exactly the framing used in Layer 14's `lesson/07_statistics.md`.

### Draft model taxonomy (from Section 3)

Two dimensions:
- **Independent drafting** — separate smaller LM from same model family (Leviathan, Chen, MLP Speculators)
- **Self-drafting** — lightweight module integrated into the target (Medusa, EAGLE, LayerSkip)

Three verification modes:
- **Greedy verification** — accept if argmax matches
- **Speculative sampling** — accept with `min(1, p/q)`; resample residual
- **Token tree verification** — parallel verification of a tree of candidate paths

### Cutting-edge focus: EAGLE family (Section 4)

The tutorial dedicates its longest technical segment (40 min) to EAGLE and EAGLE-2:
- **EAGLE** — predicts last-layer feature vectors autoregressively; avoids token-space drafting
- **EAGLE-2** — dynamic draft tree construction based on acceptance confidence; 3.5–4× speedup

The tutorial's choice to center the cutting-edge discussion on EAGLE confirms its status as the de facto state-of-the-art method for Layer 14 to reference.

### Downstream adaptations (Section 6)

Three extensions the tutorial highlights as important research directions:
- **Retrieval-augmented SD** — draft tokens retrieved from a datastore (REST, Speculative RAG)
- **Long-context SD** — sparse KV cache strategies; MagicDec approach
- **Multimodal SD** — draft in low-resolution / compressed token space

---

## Target audience statement

> "This tutorial will be accessible to anyone with a basic knowledge of machine learning and natural language processing... particularly those interested in LLM efficiency, sparsity, and computational parallelism."

This maps directly to Layer 14's L3 readers (engineers who understand transformers but are new to SD).

---

## Future research directions (Section 8 summary)

The tutorial's conclusion flags three open problems:
1. **Better speedup through accuracy-efficiency tradeoff** — smarter draft architecture search
2. **Batched inference at scale** — SD for batch_size > 1 without speedup collapse
3. **Integration with other techniques** — SD + quantization, SD + MoE, SD + KV compression

These map to the research frontiers discussed in Layer 14's `lesson/14_future_directions.md`.

---

## Connections to Layer 14

| Tutorial segment | Layer 14 lesson |
|---|---|
| Preliminaries (memory bottleneck) | `lesson/01_autoregressive_baseline.md` |
| Formal definition + speculative sampling | `lesson/05_speculative_decoding_intro.md`, `lesson/06_accept_reject_rewind.md` |
| Draft model taxonomy | `lesson/09_variants.md` |
| EAGLE / EAGLE-2 deep-dive | `lesson/10_eagle.md` |
| Token tree verification | `lesson/11_tree_attention.md` |
| Metrics (EDL, acceptance rate) | `lesson/07_statistics.md` |
| Downstream adaptations | `lesson/12_system_integration.md` |
| Future directions | `lesson/14_future_directions.md` |

---

## Relationship to other surveys in this directory

| Paper | Relationship |
|---|---|
| **Survey 01 (Xia comprehensive)** | Same first author; this tutorial is a pedagogically curated distillation of that survey |
| **Survey 02 (Ryu/Kim)** | Complements with the dual taxonomy framing |
| **Survey 03 (Hu et al.)** | Goes much broader (multimodal, systems); this tutorial stays focused on core SD |

---

## Recommended reading order

**Use this tutorial as your first read** before any other survey in this directory. Its 2.5-hour outline provides an optimal learning sequence that the other surveys assume you already know.

After completing the tutorial outline:
- Read Survey 01 (Xia) for formal definitions and Spec-Bench results
- Read Survey 02 (Ryu/Kim) for the model-centric/draft-centric dual taxonomy
- Read Survey 03 (Hu et al.) for multimodal extensions and system-level analysis
