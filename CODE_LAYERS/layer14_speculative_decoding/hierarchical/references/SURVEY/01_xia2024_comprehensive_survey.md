# Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding

**Authors:** Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, Zhifang Sui  
**Institutions:** Hong Kong PolyU, Peking University, Microsoft Research Asia, Alibaba Group  
**arXiv:** 2401.07851 | **Submitted:** January 2024 | **Revised:** June 2024 (v3)  
**PDF:** `01_xia2024_comprehensive_survey_speculative_decoding.pdf`  
**URL:** https://arxiv.org/abs/2401.07851  
**Living paper list:** https://github.com/hemingkx/SpeculativeDecodingPapers

---

## Why this survey is here

This is the **first comprehensive survey on speculative decoding** and the field's canonical reference. It introduced the formal definition that subsequent research (including SGLang's layer14 implementation) builds on. Heming Xia also authored the Tutorial Proposal (survey 04 in this directory), making this the intellectual root of the SD literature.

---

## Summary

The survey provides a systematic overview of speculative decoding from its historical origins through the state of the art as of mid-2024. It structures the field around two central questions: **how to design a good drafter** and **how to verify efficiently**.

### Key contributions

1. **First survey** of speculative decoding as a unified field.
2. **Formal definition and formulation** — gives the math every subsequent paper references:
   - Draft model `M_q`, target model `M_p`
   - Acceptance probability: `min(1, p(x)/q(x))`
   - Residual distribution for rejected tokens
3. **New taxonomy** organizing methods by drafter type and verification strategy.
4. **Spec-Bench** — benchmark enabling apples-to-apples comparison of leading methods in a unified testing environment.

---

## Historical timeline (Section 3)

The survey traces SD back to:
- **Blockwise Decoding (Stern et al., 2018)** — the first draft-then-verify approach, using extra FFN heads atop the transformer decoder.
- **SpecDec (Xia et al., 2023)** — independent non-autoregressive drafter, ~5× speedup.
- **Leviathan et al. (2023) / Chen et al. (2023)** — formal losslessness proof via speculative sampling.

---

## Taxonomy (Sections 5–7)

### Drafter selection (Section 5)

| Category | Examples | Tradeoff |
|---|---|---|
| **Independent small LM** | Llama 68M → Llama 70B | Easy alignment; requires model family match |
| **Non-autoregressive** | SpecDec | Fast drafting; lower acceptance rate |
| **Draft heads (self-drafting)** | Medusa, EAGLE, Hydra | No separate model; fine-tuning required |
| **Layer-skipping** | LayerSkip, EESD | No auxiliary model; early-exit accuracy loss |
| **N-gram / retrieval** | REST, ANPD | Training-free; domain-dependent gains |

### Verification strategies (Section 6)

| Strategy | Description |
|---|---|
| **Greedy verification** | Accept if `argmax p == draft token` — lossless for greedy decoding only |
| **Speculative sampling** | Accept with probability `min(1, p/q)`; resample from residual if rejected — lossless for any sampling |
| **Token tree verification** | Verify a tree of candidates in parallel using causal tree attention (SpecInfer, EAGLE-2, Medusa) |

### Drafter–target alignment (Section 7)

Methods to close the distribution gap:
- **Knowledge distillation** (DistillSpec, OSD)
- **Online adaptation** — draft model updates during inference based on target outputs
- **Harmonized training** (HASS) — aligns training and inference probabilities

---

## Spec-Bench results (Section 8)

Tested on Vicuna-7B, 13B, 33B, LLaMA-2-Chat-70B across translation, summarization, RAG, math, and code tasks:

| Method | Avg speedup (7B) | Notes |
|---|---|---|
| EAGLE | 2.8–3.1× | Best overall on most tasks |
| Medusa | 2.1–2.4× | No separate draft model |
| Lookahead | 1.5–1.8× | Training-free |
| Draft model (LM) | 1.9–2.3× | Baseline SD |

Key finding: **tree-based verification methods (EAGLE, Medusa) consistently outperform linear verification** by 20–40%.

---

## Connections to Layer 14

| Survey topic | Layer 14 code |
|---|---|
| Formal speculative sampling definition | `lesson/06_accept_reject_rewind.md`, `_accept_reject()` |
| Residual distribution | `lesson/07_statistics.md` |
| Drafter taxonomy (independent vs. self-drafting) | `lesson/09_variants.md`, `lesson/10_eagle.md` |
| Token tree verification | `lesson/11_tree_attention.md` |
| Spec-Bench evaluation methodology | `lesson/13_benchmarking.md` |

---

## What this survey doesn't cover

- **Production deployment** at scale (batch effects, P99 latency) — see L4 references (SpecDecode-Bench, SGLang docs)
- **Long-context behavior** — see MagicDec (L4/07)
- **EAGLE-3 and MTP** — post-submission, see EAGLE-3 overview (L4/06) and SGLang docs (L4/01)
- **Self-speculative decoding with quantization** — see QuantSpec (L4/08)

---

## Recommended reading order

Read **before** this survey:
- L3/04 (Brenndoerfer math) — to understand the acceptance criterion being formalized

Read **alongside** this survey:
- L5 original papers (Leviathan et al., Chen et al.) — the foundational proofs being surveyed
- Survey 04 (Xia tutorial proposal) — a more accessible entry point by the same first author

Read **after** this survey:
- Survey 02 (Ryu & Kim) — November 2024 perspective adding new methods
- Survey 03 (Hu et al.) — broadens to multimodal and system-level analysis
