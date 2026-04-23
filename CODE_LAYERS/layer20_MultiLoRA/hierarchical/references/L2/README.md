# L2 References: Prefill-Decode (PD) Disaggregation

**Level:** L2 — Practitioner / deployment perspective

**Reader profile:** Knows LLM inference basics (prefill, decode, KV cache). Has run SGLang or vLLM. Wants to understand why collocating prefill and decode causes problems, and how to enable PD disaggregation in SGLang. Satisfied when they can write working launch commands and understand what each flag does.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_sglang_pd_docs.md` | SGLang Official Docs | Complete launch-level reference: `--disaggregation-mode`, Mooncake and NIXL backends, router setup, all environment variables for prefill and decode server configuration, heterogeneous TP staging buffer. |
| 02 | `02_lmsys_deepseek_96h100.md` | LMSYS Blog (May 2025) | Production deployment recipe: DeepSeek-V3 on 96 H100 GPUs with PD disaggregation + large-scale expert parallelism. Throughput numbers (52.3k/22.3k tokens/sec), handshake protocol diagram, expert parallelism details. |

---

## Recommended reading order

**Fast path (20 min):** 01
- The official SGLang docs are the most direct answer to "how do I enable PD disaggregation." Read this first and you can launch a disaggregated serving setup.

**Thorough path (45 min):** 01 → 02
- 02 shows how PD disaggregation operates at production scale with a large MoE model (DeepSeek-V3). Seeing the multi-node commands, expert parallelism integration, and throughput results deepens understanding of what disaggregation enables at scale.

---

## How these map to Layer 19

| Layer 19 lesson | Most relevant L2 reference |
|---|---|
| `01_the_interference_problem.md` — why collocated scheduling fails | 01 (Issues with Unified Scheduling: prefill interruption + DP attention imbalance) |
| `02_pd_architecture.md` — phase-specific workers | 01 (launch commands for prefill and decode modes, router setup) |
| `03_kv_transfer.md` — how KV is moved between workers | 01 (Mooncake RDMA vs NIXL UCX, NVLink transport, staging buffer) |
| `05_production_deployment.md` — multi-node scale | 02 (DeepSeek 96 H100 recipe, expert parallelism, throughput benchmarks) |

---

## Common L2 limits to name for readers

These articles **do not explain:**
- The goodput framing (TTFT × TPOT SLO space) and why naive throughput is the wrong metric.
- How DistServe or Splitwise derived the motivation for disaggregation through interference measurement.
- The Mooncake Transfer Engine internals: GPUDirect RDMA, multi-NIC pooling, topology-aware path selection.
- How NIXL abstracts UCX/GDS/NVMe-oF through its Transfer Agent.
- When disaggregation hurts rather than helps (SLO regime analysis from TaiChi).

Those live in L3 (design papers) and L4 (research papers) references.
