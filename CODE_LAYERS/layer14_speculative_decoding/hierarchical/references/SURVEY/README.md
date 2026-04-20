# Survey Papers — Speculative Decoding

This directory contains PDF downloads and detailed markdown summaries for the major survey papers on speculative decoding. Surveys are distinct from individual research papers: they synthesize the field, provide taxonomy, and are the highest-leverage reading for understanding how any individual paper (including Layer 14) fits into the larger ecosystem.

---

## Files in this directory

| # | Markdown summary | PDF file | Authors | Date | arXiv |
|---|---|---|---|---|---|
| 01 | `01_xia2024_comprehensive_survey.md` | `01_xia2024_comprehensive_survey_speculative_decoding.pdf` | Xia et al. | Jun 2024 | 2401.07851 |
| 02 | `02_ryu2024_closer_look_efficient_inference.md` | `02_ryu2024_closer_look_efficient_inference.pdf` | Ryu & Kim | Nov 2024 | 2411.13157 |
| 03 | `03_hu2025_speculative_decoding_and_beyond.md` | `03_hu2025_speculative_decoding_and_beyond.pdf` | Hu et al. | Feb 2025 (v4: Oct 2025) | 2502.19732 |
| 04 | `04_xia2025_tutorial_proposal.md` | `04_xia2025_tutorial_proposal.pdf` | Xia et al. | Mar 2025 | 2503.00491 |

---

## Quick overview

### 01 — Xia et al. (2024): The foundational survey
*"Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding"*

- **The first survey** of the field; canonical reference for formal definitions
- Introduces Spec-Bench, the first standardized SD benchmark
- Taxonomy: drafter selection × verification strategy
- Appears at ACL Findings 2024
- Maintained living paper list: https://github.com/hemingkx/SpeculativeDecodingPapers
- **Read level:** L3–L4

### 02 — Ryu & Kim (2024): The dual-taxonomy survey
*"Closer Look at Efficient Inference Methods: A Survey of Speculative Decoding"*

- Introduces the **model-centric vs. draft-centric** taxonomy (orthogonal to Survey 01)
- Covers methods that appeared after Survey 01 cutoff: EAGLE-2, Hydra, S2D, HASS
- Most thorough treatment of **real-world deployment challenges**: throughput, long-context, model parallelism, hardware limits
- Introduces **EDL (Effective Decoding Length)** as a unified efficiency metric
- **Read level:** L3–L4

### 03 — Hu et al. (2025): The broad-scope survey
*"Speculative Decoding and Beyond: An In-Depth Survey of Techniques"*

- Broadest scope: text + **images + speech + recommendation systems**
- 100+ papers organized by generation strategy × refinement mechanism
- Dedicated section on **system-level implementations**: parallel SD, distributed SD, hardware/compiler optimizations
- Covers iterative refinement (Jacobi, CLLMs) as a distinct paradigm alongside SD
- Most recent survey, continuously updated (v4: October 2025)
- **Read level:** L4

### 04 — Xia et al. (2025): The tutorial proposal
*"Tutorial Proposal: Speculative Decoding for Efficient LLM Inference"*

- A **2.5-hour tutorial** designed for ML newcomers; the most accessible entry point
- By the same first author as Survey 01 — pedagogically curated distillation
- Maps almost exactly to Layer 14's lesson sequence
- Identifies the three factors for speedup: draft efficiency × acceptance rate × EDL
- Centers the "cutting-edge algorithms" discussion on EAGLE/EAGLE-2 (40 of 170 minutes)
- **Read level:** L2–L3 (best first read)

---

## Recommended reading order

```
FIRST:   04 (tutorial) — establishes vocabulary and learning path
SECOND:  01 (Xia comprehensive) — formal definitions, Spec-Bench results
THIRD:   02 (Ryu/Kim) — deployment challenges, dual taxonomy, newer methods
FOURTH:  03 (Hu et al.) — multimodal extensions, system-level, full taxonomy
```

---

## How surveys differ from L1–L5 references

| Reference type | Purpose | Where |
|---|---|---|
| **L1–L2 blogs** | Build intuition, motivate the topic | `references/L1/`, `references/L2/` |
| **L3 technical blogs** | Show mechanisms, pseuodcode, worked examples | `references/L3/` |
| **L4 production articles** | Real stacks, benchmarks, deployment gotchas | `references/L4/` |
| **SURVEY (this dir)** | Map the entire field; place any paper in context | `references/SURVEY/` |
| **L5 original papers** | Primary sources; formal proofs; citable results | `references/L5/` (to be created) |

Surveys occupy a unique position: they are not leveled by difficulty but by scope. They should be consulted **before writing any lesson** that touches on variants, comparisons, or "what comes next" framing.

---

## Coverage comparison

| Survey | Methods covered | Multimodal | System-level | Benchmark |
|---|---|---|---|---|
| 01 (Xia 2024) | ~50 | No | No | Spec-Bench (7B–70B) |
| 02 (Ryu 2024) | ~40 | No | Deployment challenges | EDL analysis |
| 03 (Hu 2025) | 100+ | Yes | Full section | Analysis only |
| 04 (Xia tutorial) | ~20 (curated) | Mentioned | No | Fair comparison table |

---

## Connection to Layer 14 lessons

| Lesson | Most relevant survey |
|---|---|
| `lesson/01` — autoregressive baseline | Survey 04 §2 (Preliminaries) |
| `lesson/05` — speculative decoding intro | Survey 01 §4 (formal definition) |
| `lesson/06` — accept/reject/rewind | Survey 01 §6 (verification strategies) |
| `lesson/07` — statistics | Survey 02 (EDL metric), Survey 04 §5 (metrics) |
| `lesson/08` — production gotchas | Survey 02 §4 (deployment challenges) |
| `lesson/09` — variants | Survey 01 §5 (drafter taxonomy), Survey 03 §IV |
| `lesson/10` — EAGLE | Survey 04 §4 (cutting-edge), Survey 02 §2.1.2 |
| `lesson/11` — tree attention | Survey 01 §6.3 (tree verification), Survey 03 §V-A2 |
| `lesson/12` — system integration | Survey 03 §VI (system-level) |
| `lesson/14` — future directions | Survey 02 §4 (challenges), Survey 04 §8 (future) |
