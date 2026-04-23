# L3 References: Prefill-Decode (PD) Disaggregation

**Level:** L3 — Technical / design-focused

**Reader profile:** Wants to understand the design decisions behind PD disaggregation: why goodput is the right metric, how KV transfer is implemented, what Mooncake's RDMA engine does, and how NVIDIA Dynamo orchestrates disaggregated serving at enterprise scale. Has read the L2 docs and wants the theoretical and engineering depth.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_distserve_osdi24.md` | USENIX OSDI 2024 | DistServe — the foundational paper that introduced goodput as the metric, quantified prefill-decode interference, and proved that disaggregation on separate GPUs can serve 7.4× more requests under SLO constraints. |
| 02 | `02_splitwise_isca24.md` | ISCA 2024 | Splitwise — independent co-discovery from Microsoft Azure; uses real production traces; characterises prompt-computation vs token-generation phases; proposes hardware heterogeneity (different GPU SKUs per phase); 1.4× throughput at 20% lower cost. |
| 03 | `03_mooncake_fast25.md` | USENIX FAST 2025 | Mooncake — the production KV transfer infrastructure for Kimi (Moonshot AI); GPUDirect RDMA, multi-NIC pooling, topology-aware path selection; disaggregated KV cache pool across CPU/DRAM/SSD; 525% throughput increase in simulated long-context scenarios. |
| 04 | `04_nvidia_dynamo.md` | NVIDIA GTC 2025 + NIXL | NVIDIA Dynamo — enterprise-scale disaggregated serving framework; KV-aware routing; 4-plane architecture (request, control, discovery, event); NIXL as the KV transfer library; 30× throughput improvement on DeepSeek-R1 on GB200 NVL72. |

---

## Recommended reading order

**Fast path (60 min):** 01 → 03
- 01 for the theoretical motivation (goodput framing, interference quantification, GPU-level performance).
- 03 for how it works in production: the Mooncake RDMA transfer engine that SGLang and vLLM both use.

**Thorough path (2–3 hours):** 01 → 02 → 03 → 04
- 02 for the hardware heterogeneity insight and production trace grounding.
- 04 for how NVIDIA abstracts disaggregated serving into a multi-framework orchestration layer.

---

## How these map to Layer 19

| Layer 19 lesson | Most relevant L3 reference |
|---|---|
| `01_the_interference_problem.md` — interference quantification | 01 (DistServe measures colocation cost), 02 (Splitwise Azure production traces) |
| `02_pd_architecture.md` — goodput framing, resource allocation | 01 (goodput definition, per-phase parallelism co-optimization) |
| `03_kv_transfer.md` — transfer engine mechanics | 03 (Mooncake RDMA, multi-NIC, topology-aware path selection) |
| `04_routing_and_scale.md` — KV-aware routing, cluster management | 04 (Dynamo 4-plane architecture, KV-aware routing, NIXL) |
| `05_production_deployment.md` — launch commands and tuning | 01 (per-phase resource allocation), 02 (heterogeneous hardware clusters) |
