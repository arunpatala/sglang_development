# Closer Look at Efficient Inference Methods: A Survey of Speculative Decoding

**Authors:** Hyun Ryu (KAIST), Eric Kim (Georgia Institute of Technology)  
**arXiv:** 2411.13157 | **Submitted:** November 2024 | **Revised:** November 2024 (v2)  
**PDF:** `02_ryu2024_closer_look_efficient_inference.pdf`  
**URL:** https://arxiv.org/abs/2411.13157

---

## Why this survey is here

This survey arrives **ten months after** the first Xia et al. survey (01), capturing a new wave of methods: EAGLE-2, Hydra, S2D, HASS, and others that appeared in mid-2024. Its distinct **dual taxonomy** (draft-centric vs. model-centric) is a useful mental model for categorizing methods that Layer 14's architecture must accommodate. It also has the most thorough treatment of **real-world deployment challenges** of any survey.

---

## Summary

The paper surveys speculative decoding by separating concerns into two orthogonal dimensions:

- **What are you optimizing?** → Model-centric (improve the drafter architecture) or Draft-centric (improve the candidate selection/verification logic)
- **How does the drafter relate to the target?** → Independent or Dependent

This 2×2 framing lets readers quickly locate any method in the literature and understand its design philosophy.

---

## Taxonomy

### Model-Centric Implementations (Section 2.1)

Methods that focus on **improving draft quality** through architectural changes.

#### Independent Draft Models (Section 2.1.1)

| Method | Key idea |
|---|---|
| **Speculative Sampling** (Leviathan, Chen) | Same-series smaller model; distribution alignment via pretraining |
| **Chimera** | Trigram encoder (short-range) + full context encoder (long-range) |
| **S2D** | Sorted fine-tuning (SoFT) to create versatile sub-draft-models for multiple targets |
| **Online Speculative Decoding (OSD)** | Dynamically adjusts draft model based on query distribution during inference |

#### Dependent Draft Models (Section 2.1.2)

| Method | Mechanism |
|---|---|
| **Medusa** | Extra decoding heads on last hidden layer; tree verification |
| **Hydra** | Medusa + each head also conditions on previously speculated tokens; higher acceptance length |
| **EAGLE** | Predicts last-layer feature vectors autoregressively, then decodes tokens |
| **SpecDec++** | Prediction head estimates acceptance probability; dynamic stop criterion |
| **LayerSkip / EESD** | Early-exit drafting with KD (EESD) or layer-skip loss (LayerSkip) |

---

### Draft-Centric Implementations (Section 3.1)

Methods that focus on **improving candidate selection** rather than the drafter itself.

#### Probability-Based Approaches (Section 3.1.1)

- **Nucleus Sampling** — filters low-probability tail; standard SD baseline
- **HASS (Harmonized Speculative Sampling)** — aligns training and inference distributions; higher acceptance without changing the draft model

#### Search Optimization (Section 3.1.2)

- **Beam search** for draft candidates; risk of repetitive outputs
- **Diverse beam search** — multiple beams with diversity penalty

#### Tree and Graph-Based (Section 3.1.3)

- **SpecInfer** — tree-based attention masking for parallel multi-path verification
- **EAGLE-2** — dynamic draft trees; adaptive tree structure based on context acceptance likelihood
- **Sequoia** — tree-of-trees using optimal tree construction algorithms

#### Hybrid and Adaptive (Section 3.1.4)

- **Dynamic draft length** — adjusts `num_speculative_tokens` per step based on running acceptance rate
- **Confidence-based triggering** — only invokes target model when draft confidence drops

---

## Real-World Deployment Challenges (Section 4)

This is the survey's most distinctive contribution. Five challenges are identified:

### 1. Throughput at scale

Speculative decoding is designed for **latency reduction at batch_size=1**. At batch_size > 8, the draft overhead often outweighs verification gains because:
- The target model can process large batches more efficiently than the draft model generates tokens
- The acceptance rate degrades with batching (distribution shift)

**Implication for Layer 14:** `kv_memory_fraction` and `num_spec_tokens` must be tuned per batch size regime.

### 2. Long-context generation

KV cache for both draft and target models grows with context length, creating memory pressure. Methods like **MagicDec** (sparse draft KV) and **QuantSpec** (quantized KV) address this.

### 3. Model parallelism

When the target model is sharded across GPUs:
- Draft generation creates load imbalance (draft is much smaller, runs on fewer GPUs)
- Communication overhead during verification can negate speedup

### 4. Hardware limitations

Draft model must fit in leftover GPU memory after loading the target model. For 70B+ targets on consumer hardware, this often means the draft model is too small to have useful acceptance rates.

### 5. Generalizability

Speedup is highly task-dependent:
- **Best case:** code completion, summarization, translation (repetitive structure → high acceptance)
- **Worst case:** creative writing, diverse sampling, math reasoning (low acceptance rate)

---

## Effective Decoding Length (EDL)

The survey introduces a clean metric — **EDL (Effective Decoding Length)** — the average number of tokens accepted per target model call. This unifies the efficiency discussion across methods:

```
EDL = (total accepted tokens) / (total target model calls)
```

EDL directly determines the speedup ratio over autoregressive decoding (which has EDL = 1.0 by definition).

| Method | Typical EDL |
|---|---|
| Draft model (small LM) | 2.5–3.5 |
| EAGLE-2 | 3.5–4.5 |
| N-gram (code editing) | 4.0–6.0 |
| Oracle (theoretical max) | varies by task |

---

## Connections to Layer 14

| Survey topic | Layer 14 code |
|---|---|
| Independent vs. dependent drafter taxonomy | `lesson/09_variants.md` |
| Tree-based verification (SpecInfer, EAGLE-2) | `lesson/11_tree_attention.md` |
| EDL metric | `lesson/07_statistics.md` |
| Throughput challenge | `lesson/08_production_gotchas.md` |
| Dynamic draft length | SGLang `speculative_num_steps` param |

---

## What this survey adds over Survey 01 (Xia 2024)

- **Later methods:** Hydra, HASS, S2D, EAGLE-2, EESD — all appeared after Survey 01's cutoff
- **Deployment realism:** The deployment challenges section is unique to this survey
- **Dual taxonomy:** The model-centric/draft-centric split complements Survey 01's drafter-selection/verification-strategy split
- **EDL metric:** A cleaner efficiency measure than raw speedup ratios

---

## Recommended reading order

Read **before** this survey:
- Survey 01 (Xia comprehensive) — foundational definitions and Spec-Bench results

Read **alongside** this survey:
- L4/03 (SpecDecode-Bench) — empirical validation of the deployment challenges described here
- L4/06 (EAGLE-3 overview) — extends EAGLE-2 covered in this survey

Read **after** this survey:
- Survey 03 (Hu et al.) — broadens to images, speech, and system-level analysis
