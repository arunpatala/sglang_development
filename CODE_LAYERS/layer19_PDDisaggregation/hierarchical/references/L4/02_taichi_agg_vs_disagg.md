# TaiChi: Prefill-Decode Aggregation or Disaggregation? Unifying Both

**Source:** https://arxiv.org/abs/2508.01989
**Authors:** Chao Wang et al. (Sun Yat-sen University, Huawei Cloud, The Chinese University of Hong Kong)
**Submitted:** August 4, 2025
**Level:** L4 — Research analysis; SLO-regime theory; the "when to disaggregate" answer
**Why here:** TaiChi provides the most rigorous and complete answer to the question "is PD disaggregation always better?" The answer is **no** — it depends on the SLO regime. This paper defines exactly when to use each approach and proposes a hybrid system (TaiChi) that switches between them dynamically, achieving 77% goodput improvement over state-of-the-art systems. Every production disaggregation deployment decision should be informed by this paper's SLO-regime analysis.

**BibTeX:**
```bibtex
@article{wang2025taichi,
  title   = {Prefill-Decode Aggregation or Disaggregation?
             Unifying Both for Goodput-Optimized LLM Serving},
  author  = {Chao Wang and others},
  journal = {arXiv preprint arXiv:2508.01989},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.01989}
}
```

---

## The Central Question

Since 2024, two camps have produced competing optimisations:

- **PD Aggregation** (Orca, SARATHI/Sarathi-Serve): co-locate prefill and decode, manage interference through chunked prefill and careful batching.
- **PD Disaggregation** (DistServe, Splitwise, Mooncake, NVIDIA Dynamo): physically separate onto dedicated GPU pools, eliminate interference entirely.

Each camp claims advantages. TaiChi settles the debate empirically and theoretically.

---

## The SLO-Regime Framework

TaiChi shows that **neither approach dominates** — the optimal choice depends on the SLO constraints:

### When PD Aggregation is Optimal

**Tight TTFT + relaxed TPOT** (e.g., chatbots that must respond quickly but users tolerate slower typing speed):

- All GPUs can contribute to prefill simultaneously → low TTFT.
- Decode TPOT violations are tolerable → interference-induced TPOT spikes don't matter.
- Aggregation achieves maximum GPU utilisation by batching prefill and decode together.
- Disaggregation would hurt TTFT because fewer GPUs handle prefill (only the prefill pool).

### When PD Disaggregation is Optimal

**Tight TPOT + relaxed TTFT** (e.g., coding assistants streaming long outputs where smooth token pace matters but first-token latency is acceptable):

- Dedicated decode pool is never interrupted by prefill → stable TPOT.
- Higher TTFT is acceptable → prefill pool can take longer.
- Aggregation would cause prefill interference to spike TPOT, violating the strict TPOT SLO.

### The Balanced SLO Problem

**Tight TTFT + tight TPOT**: neither approach is optimal.

- PD aggregation: TPOT violations due to prefill interference.
- PD disaggregation: TTFT violations because fewer instances handle prefill.
- Neither can satisfy both SLOs at the same request rate → goodput is bounded below optimal.

---

## TaiChi's Solution: Hybrid-Mode Inference

TaiChi proposes a system that **dynamically switches between PD aggregation and disaggregation** based on real-time SLO signals.

### Key Mechanism: P-Heavy and D-Heavy Workers

TaiChi divides the GPU pool into two dynamic categories:

- **P-heavy workers**: primarily process prefill batches (optimised for compute throughput). These are the disaggregation-mode prefill workers.
- **D-heavy workers**: primarily process decode batches (optimised for memory bandwidth). These are the disaggregation-mode decode workers.

In **aggregation mode**: all workers process mixed P+D batches (SARATHI-style chunked prefill).

In **disaggregation mode**: P-heavy workers run prefill only; D-heavy workers run decode only; KV is transferred between them.

In **hybrid mode** (TaiChi's contribution): the split between P-heavy and D-heavy is **dynamic** — adjustable in response to real-time SLO violation signals. When TTFT SLO is tight, more workers shift to P-heavy; when TPOT is tight, more shift to D-heavy.

### SLO Monitoring Loop

```
Monitor TTFT violations → too many → shift some D-heavy workers to P-heavy
Monitor TPOT violations → too many → shift some P-heavy workers to D-heavy
At equilibrium: minimum workers of each type for both SLOs to be satisfied
```

---

## Results

Implemented on vLLM; evaluated on DeepSeek-R1, Llama-70B, and other models:

| Metric | TaiChi vs State-of-the-Art |
|---|---|
| Goodput improvement | **Up to 77% over SOTA** |
| TTFT reduction (vs PD disaggregation) | **Up to 13.2×** (disaggregation has too-high TTFT when TTFT SLO is tight) |
| TPOT reduction (vs PD aggregation) | **Up to 1.69×** (aggregation has too-high TPOT when TPOT SLO is tight) |

---

## The SLO Decision Matrix for Layer 19 Deployments

| Workload type | TTFT constraint | TPOT constraint | Recommended approach |
|---|---|---|---|
| Chatbot (responsiveness matters) | Tight | Relaxed | PD Aggregation (SARATHI-style) |
| Code streamer (smooth output) | Relaxed | Tight | PD Disaggregation |
| Agentic AI (long chains, low latency) | Tight | Tight | TaiChi hybrid or large disaggregation cluster |
| Batch inference (offline) | Relaxed | Relaxed | Aggregation (maximise throughput) |
| RAG with long prompts | Moderate | Tight | PD Disaggregation |
| Multi-turn chat | Moderate | Moderate | TaiChi or cache-aware aggregation |

---

## Connection to SGLang's Current Design

SGLang's PD disaggregation is a **pure disaggregation mode** — once enabled, the cluster is split into fixed prefill and decode pools. This is optimal for TPOT-tight workloads but not for TTFT-tight workloads.

TaiChi's contribution is the observation that **the split should be dynamic**. Future SGLang evolution may incorporate:
- Dynamic worker reassignment between prefill/decode pools based on SLO signals.
- A "hybrid batch" mode where some requests are processed with full disaggregation and others with local prefill on the decode worker.

---

## What TaiChi Does Not Address

TaiChi focuses on the aggregation/disaggregation tradeoff. It does not address:
- MoE-specific constraints (DeepEP dispatch mode conflicts that make disaggregation mandatory for DeepSeek-V3).
- KV transfer cost as a function of network topology.
- Multi-tier KV cache (HiCache) interaction with dynamic worker reassignment.

These remain open research directions.

---

## Key Takeaways for Layer 19

- **Disaggregation is not always better**: under tight TTFT + relaxed TPOT SLOs, aggregation achieves higher goodput.
- **The SLO regime determines the architecture**: TPOT-tight workloads need disaggregation; TTFT-tight workloads may be better served with aggregation.
- **77% goodput improvement** from dynamic hybrid switching shows the gap is not marginal — the right architecture choice is one of the most impactful decisions in LLM serving.
- SGLang's current pure disaggregation mode is optimal for decode-quality-sensitive workloads (code generation, long outputs). For latency-sensitive chatbots, the tradeoff deserves evaluation.
- TaiChi's SLO monitoring loop is the correct abstraction for production deployments that serve mixed workload types.
