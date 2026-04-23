# L1 References: Prefill-Decode (PD) Disaggregation

**Level:** L1 — Introductions, mental models, and visual explainers

**Reader profile:** New to LLM inference or to the concept of disaggregation. Wants a clear mental model of why prefill and decode are different workloads, what disaggregation solves, and whether it's worth adopting — without reading a systems paper. Satisfied when they can explain in one paragraph why separating phases is beneficial and what the key cost is (KV transfer latency).

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_tds_compute_vs_memory.md` | Towards Data Science (Apr 2026) | Longest and most complete L1 article: production failure story, arithmetic intensity numbers, KV cache Python formula, 5-check deployment decision framework, cost arithmetic. |
| 02 | `02_jarvislabs_meta_pd.md` | Jarvis Labs Blog (Jan 2026) | Best concrete math: step-by-step KV cache size calculation, network bandwidth comparison table, Mermaid architecture diagrams. Written from a practitioner's research notes perspective. |
| 03 | `03_jackson_mz_practical.md` | Better Programming / Medium (Apr 2025) | Shortest useful introduction (2-min read). One-sentence framing per phase, minimal-code mental model. Best first-read before anything else. |
| 04 | `04_naddod_pd_overview.md` | NADDOD / Medium (Aug 2025) | 5-minute structured overview. Covers phase definitions, disaggregation rationale, 3 advantages, 4 challenges with solutions. |
| 05 | `05_bentoml_pd_handbook.md` | BentoML LLM Inference Handbook | Handbook-style reference. Covers when disaggregation is NOT worth it (20–30% performance drop in tested cases). |
| 06 | `06_learncodecamp_ttft_itl.md` | LearnCodeCamp | Pure prerequisite: TTFT and ITL definitions, compute-bound vs memory-bound intuition, chunked prefill primer. Read this before any other reference. |
| 07 | `07_llm_perf_estimator.md` | joursbleu.github.io (interactive) | Interactive roofline calculator: input your model + GPU, see prefill latency, decode throughput, and roofline plot in browser. |
| 08 | `08_perplexity_nvidia_spotlight.md` | NVIDIA Blog (Dec 2024) | Production case study: Perplexity AI at 435M queries/month, actively deploying disaggregated serving. Answers "is this production-ready?" |
| 09 | `09_hao_zhang_lecture_slides.md` | CMU LLM Systems 2025 (PDF) | Lecture slides by DistServe's PI. Visual goodput definition, interference diagrams, continuous batching vs disaggregation clarification. |

---

## Recommended reading order

**2-minute first contact:** 03 → done
- Sets the mental model in one read.

**Fast path (20 min):** 06 → 02 → 08
- 06: understand TTFT/ITL before reading anything else.
- 02: get the concrete math and architecture diagrams.
- 08: confirm disaggregation is production-proven.

**Thorough L1 path (90 min):** 06 → 03 → 04 → 05 → 02 → 01 → 07 → 08 → 09
- Builds from definitions → mental model → advantages/challenges → limits → math → full analysis → interactive tool → production proof → academic lecture.

---

## How these map to Layer 19

| Layer 19 lesson | Most relevant L1 reference |
|---|---|
| `01_the_interference_problem.md` — why collocated scheduling fails | 01 (monolithic serving cost analysis, GPU utilisation measurements), 02 (interference flowchart) |
| `02_pd_architecture.md` — what disaggregation looks like | 02 (architecture Mermaid diagrams, independent scaling scenarios) |
| `03_kv_transfer.md` — KV cache size and transfer cost | 01 (Python KV formula + Llama 8B/70B examples), 02 (network bandwidth table, transfer time math) |
| `06_tradeoffs_and_limits.md` — when NOT to disaggregate | 01 (5-check decision framework), 05 (20–30% degradation on unsuitable workloads) |
| Prerequisites for the lesson | 06 (TTFT + ITL definitions) |

---

## Common L1 limits to name for readers

These articles **do not explain:**
- Goodput as a formal metric (TTFT × TPOT SLO space) — that's DistServe (L3/01).
- How Mooncake's Transfer Engine does multi-NIC RDMA and topology-aware path selection — that's L3/03.
- Why MoE models (DeepSeek-V3) require disaggregation regardless of SLO constraints — that's the LMSYS blog (L2/02).
- When aggregation (chunked prefill, SARATHI) is better than disaggregation — that's TaiChi (L4/02).
- Any source code or implementation detail — those are L4–L5.
