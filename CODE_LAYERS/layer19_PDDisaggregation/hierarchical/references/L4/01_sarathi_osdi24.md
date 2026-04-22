# SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills

**Source:** https://arxiv.org/abs/2308.16369
**Extended version:** Sarathi-Serve, USENIX OSDI 2024
**Venue:** arXiv August 2023; OSDI 2024
**Authors:** Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Ramachandran Ramjee (Microsoft Research India)
**Level:** L4 — Aggregation-side optimisation; chunked prefill; the incumbent technique before disaggregation
**Why here:** SARATHI defines the **aggregation-side baseline** that PD disaggregation must outperform. Its chunked prefill technique is not only the standard comparison point in disaggregation papers — it is also **used inside SGLang's prefill servers** via `--chunked-prefill-size` to limit prefill-induced stalls. Understanding SARATHI clarifies what disaggregation is actually solving beyond what chunked prefill can achieve.

**BibTeX:**
```bibtex
@inproceedings{agrawal2024sarathi,
  title     = {{SARATHI}: Efficient LLM Inference by Piggybacking Decodes
               with Chunked Prefills},
  author    = {Amey Agrawal and Ashish Panwar and Jayashree Mohan
               and Nipun Kwatra and Bhargav S. Gulavani and Ramachandran Ramjee},
  booktitle = {18th USENIX Symposium on Operating Systems Design and
               Implementation (OSDI 24)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2308.16369}
}
```

---

## The Problem SARATHI Addresses

Standard LLM inference suffers from two inefficiencies:

1. **Head-of-line blocking**: a single long prefill request (e.g., 10K-token prompt) monopolises the GPU for hundreds of milliseconds. All decode requests are blocked until the prefill completes.

2. **Pipeline parallelism bubbles**: when using pipeline parallelism, large prefill microbatches have variable duration, creating imbalanced stage times and wasteful pipeline bubbles.

---

## SARATHI's Solution: Chunked Prefills

Instead of processing a large prefill request in a single step, SARATHI splits it into **fixed-size chunks** and schedules each chunk as a separate mini-batch step.

### Chunked Prefill Mechanics

```
Standard: [PPPPPPPPPPPP | D D D D D D D D D]
           ←— 12 tokens prefill —→ ←— decode —→
           One prefill blocks all decodes for the entire duration.

SARATHI:  [PPPP | D D D D] → [PPPP | D D D D] → [PPPP | D D D D]
          Chunk 1 + decodes   Chunk 2 + decodes   Chunk 3 + decodes
```

Each chunk is small enough to complete quickly, then decode requests run. This interleaving prevents head-of-line blocking.

### Decode-Maximal Batching

SARATHI constructs each batch using:
- **One prefill chunk** (saturates GPU compute during the chunk duration)
- **As many decode requests as fit in remaining KV memory** (they "piggyback" on the compute wave)

During the batch:
- The prefill chunk runs matrix multiplications that saturate GPU FLOP throughput.
- The decode requests run in parallel but are memory-bound — their incremental compute cost is an order of magnitude lower than a full decode batch.
- Result: decode requests complete while the GPU is primarily computing the prefill chunk.

### Pipeline Bubble Reduction

Chunked prefill ensures every stage in a pipeline parallel setup receives a chunk of uniform compute size, eliminating microbatch duration imbalance. SARATHI reduces pipeline bubbles by **6.29×** on GPT-3 with pipeline parallelism.

---

## Results

Evaluated on LLaMA-13B (A6000) and LLaMA-33B (A100):

| Model | Metric | Improvement |
|---|---|---|
| LLaMA-13B (A6000) | Decode throughput | **Up to 10× improvement** |
| LLaMA-13B (A6000) | End-to-end throughput | **Up to 1.33×** |
| LLaMA-33B (A100) | End-to-end throughput | **1.25×** |
| LLaMA-33B (A100) | Decode throughput | **4.25×** |
| GPT-3 (pipeline parallel) | Pipeline bubbles | **6.29× reduction** |
| GPT-3 (pipeline parallel) | End-to-end throughput | **1.91×** |

---

## SARATHI in SGLang

SGLang implements chunked prefill via `--chunked-prefill-size`. In PD disaggregation mode, the **prefill server** typically uses chunked prefill to limit individual step duration:

```bash
python -m sglang.launch_server \
  --disaggregation-mode prefill \
  --chunked-prefill-size 32768 \
  ...
```

This ensures the prefill server doesn't take arbitrarily long on very large prompts, which would increase the time before KV transfer begins and the decode server can start.

---

## What Chunked Prefill Cannot Fix (Motivation for Disaggregation)

SARATHI reduces interference but cannot eliminate it:

| Problem | SARATHI (chunked prefill) | PD Disaggregation |
|---|---|---|
| Head-of-line blocking | Reduced (chunk duration instead of full prefill) | Eliminated (prefill on separate GPU) |
| Decode TPOT increase during prefill | Reduced but non-zero (decode still waits for chunk) | Zero (decode GPU never runs prefill) |
| Resource coupling | Still coupled (same GPU, same TP config) | Decoupled (independent TP per phase) |
| Hardware optimisation | Cannot use different hardware per phase | Can use compute-optimised vs memory-optimised GPUs |
| TPOT variance | Still variable (chunk scheduling adds jitter) | Stable (decode GPU is dedicated) |

**The key limit of chunked prefill**: even a 1-token prefill chunk requires exclusive GPU access for one forward step, during which decode is blocked. At very high prefill rates (long prompts, many concurrent requests), this residual interference is significant.

DistServe measures that with chunked prefill (SARATHI), collocated systems still show 3–5× higher TPOT variance vs disaggregated systems under production SLO constraints.

---

## Why This Matters for Layer 19 Architecture Decisions

TaiChi (L4/02) uses Sarathi-Serve as the aggregation baseline in its comparison. Understanding what SARATHI achieves helps calibrate the value of full disaggregation:

- **If your workload has short prompts** (< 512 tokens), chunked prefill may be sufficient — the chunk duration is so short that TPOT impact is negligible.
- **If your workload has long prompts** (> 2K tokens) or strict TPOT SLOs, chunked prefill leaves significant residual interference — disaggregation is needed.
- **For MoE models** (DeepSeek), chunked prefill does not address the expert dispatch mode conflict — disaggregation is required regardless of prompt length.

---

## Key Takeaways for Layer 19

- SARATHI's chunked prefill is the **standard collocation-side mitigation** — it is a prerequisite for fair comparison against disaggregated systems.
- `--chunked-prefill-size` in SGLang's prefill server is the direct implementation of SARATHI's chunked prefill technique.
- Chunked prefill reduces but cannot eliminate the resource coupling problem — it remains impossible to optimise TP configuration independently for each phase.
- The 10× decode throughput improvement from SARATHI shows that proper batching matters even before disaggregation — disaggregation builds on top of chunked prefill, not instead of it.
- **OSDI 2024** — same venue as DistServe, confirming that both aggregation and disaggregation are active research areas at the top systems tier.
