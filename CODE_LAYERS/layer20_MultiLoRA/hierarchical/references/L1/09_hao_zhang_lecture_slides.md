# Lecture Slides: LLM Systems — Prefill-Decode Disaggregation

**Source:** https://www.cs.cmu.edu/~hzhang2/courses/15849/lectures/lec10.pdf
**Author:** Prof. Hao Zhang (CMU, School of Computer Science) — also lead PI of DistServe (Hao AI Lab, UCSD)
**Series:** CMU 15-849 LLM Systems, Spring 2025 — Lecture 10
**Level:** L1 (visual lecture slides) — Visual goodput definition; interference diagrams; continuous batching vs disaggregation
**Why here:** The PI of DistServe's lecture slides for the CMU graduate LLM Systems course. Provides the most visual and academically structured introduction to disaggregation: goodput definition with SLO visualisation, interference measurement from the DistServe paper, continuous batching limitations, and disaggregated architecture walkthrough. The most authoritative L1 source.

---

## Lecture Overview

**Course:** CMU 15-849: LLM Systems (graduate seminar)
**Lecture:** 10 — Disaggregated LLM Serving
**Prerequisites in the course:** LLM inference basics, continuous batching (Orca), paged attention (vLLM), SARATHI chunked prefill

---

## Goodput: The Key Metric

### Definition from the Slides

> "Goodput = number of requests that meet **both** the TTFT SLO **and** the TPOT SLO per unit time"

Unlike raw throughput (requests/second), goodput measures **useful work**: requests that satisfy the latency SLOs experienced by real users. A system that serves more requests per second but violates TTFT/TPOT SLOs for many of them has lower goodput.

### SLO Space Visualisation

The slides show the SLO space as a 2D grid:
```
TPOT SLO axis (ms/token)
    │   ┌──────────┐
    │   │ GOOD     │
    │   │ REQUESTS │
    ├───┤          │
    │   └──────────┘
    └──────────────────── TTFT SLO axis (ms)
```
- Only requests that land in the upper-left quadrant (TTFT < SLO_TTFT AND TPOT < SLO_TPOT) count as "good".
- Disaggregation expands the number of requests in this quadrant by removing the interference that forces trade-offs between the two axes.

---

## Prefill-Decode Interference in Monolithic Serving

### What the Slides Show

1. **Batch timeline in monolithic serving**: prefill requests and decode steps interleaved in the same batch, GPU alternating between compute saturation and memory saturation.
2. **Interference effect**: a large prefill request enters the batch → all active decode requests experience a "jitter event" (ITL spike).
3. **Measurement**: decode ITL as a function of prefill batch size → clear positive correlation.
4. **TTFT impact**: as decode requests fill the batch, new prefill requests queue longer → TTFT increases even when the server is not at capacity.

---

## Continuous Batching and Its Limits

### SARATHI / Chunked Prefill as the Prior Art

Before disaggregation:
- **Continuous batching (Orca)**: decode requests added to the batch continuously as space permits. Eliminates the "waiting for full batch" problem but doesn't eliminate prefill-decode interference.
- **Chunked prefill (SARATHI)**: breaks long prefills into smaller chunks interleaved with decode steps. Smooths ITL but does not eliminate the fundamental compute-bound/memory-bound conflict.

**The limit**: even with chunked prefill, the same GPU is doing both compute-heavy and memory-heavy work within the same scheduling unit. The interference is reduced but not eliminated.

---

## Disaggregated Architecture (Slides)

### System Diagram

```
[Incoming requests]
         ↓
    [Router / KV-aware scheduler]
         ├──────────────────────────┐
         ↓                          ↓
[Prefill workers]          [Decode workers]
  - Compute-optimised        - Memory-optimised
  - Handle prompt input      - Handle token generation
  - Produce KV caches        - Consume KV caches
         │                          │
         └──── [KV transfer] ────────┘
                 (RDMA / NVLink)
```

### Key Properties in the Slides

1. **KV cache flows from prefill to decode** — the core transfer overhead.
2. **Prefill workers can batch multiple prompts together** — amortises compute overhead.
3. **Decode workers maintain separate, persistent KV caches** — enables long-context continuity.
4. **The router tracks KV cache locations** — enables prefix cache reuse across multiple requests sharing system prompts.

---

## Goodput Improvement Measurements (from DistServe)

The slides reference DistServe's OSDI 2024 results:

| Model + Hardware | Goodput Improvement over vLLM |
|---|---|
| OPT-13B on A100 | 7.4× |
| OPT-66B on A100 | 4.0× |
| LLaMA-2-70B on A100 | 6.1× |

**Under strict SLO constraints** (TTFT < 2s, TPOT < 100ms): disaggregation allows serving 4–7× more requests per second that actually satisfy both SLOs simultaneously — not just raw throughput improvement.

---

## Comparison: Monolithic vs Disaggregated

| Aspect | Monolithic (vLLM, SARATHI) | Disaggregated (DistServe, SGLang, NVIDIA Dynamo) |
|---|---|---|
| Phase colocation | Yes (same GPUs) | No (separate pools) |
| TTFT optimisation | Limited by decode interference | Direct: prefill workers dedicated |
| TPOT optimisation | Limited by prefill bursts | Direct: decode workers dedicated |
| Hardware flexibility | Same GPU type for both | Different GPU types per phase |
| KV transfer overhead | None | 27–107ms (RDMA) |
| Goodput (under SLO) | Baseline | 4–7× higher (DistServe results) |
| Operational complexity | Low | Higher |

---

## SGLang Implementation (Referenced in Lecture)

The lecture slides reference the SGLang disaggregated serving design as the practical implementation:

```bash
# Prefill server
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake

# Decode server  
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake

# Router
python -m sglang.srt.disaggregation.mini_lb \
  --prefill http://prefill_host:port \
  --decode http://decode_host:port
```

---

## Key Takeaways for Layer 19

- **Goodput** is not a rough approximation of throughput — it is a precise metric: requests/second satisfying TTFT SLO AND TPOT SLO simultaneously. This distinction matters because optimising for raw throughput may improve neither metric from the user's perspective.
- The lecture slides are the most authoritative L1 source because the author is also the PI of DistServe — the concepts here are the original formulations, not translations.
- **Continuous batching does not solve the prefill-decode interference problem** — it just reduces the batch delay. SARATHI reduces ITL jitter. Only disaggregation eliminates interference by physical separation.
- The 4–7× goodput improvement reported in DistServe is **under strict SLO constraints** — without SLOs, the improvement is smaller. This is why measuring with SLOs (as goodput does) tells a different story than raw throughput benchmarks.
- The KV-aware router that tracks cache locations is the enabling component that makes prefix caching across requests possible — not just a load balancer.
