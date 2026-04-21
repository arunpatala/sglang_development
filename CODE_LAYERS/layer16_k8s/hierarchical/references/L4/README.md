# L4 References: LLM Inference on Kubernetes

**Level:** L4 — Advanced production engineering (internal design, metrics deep-dive, production HA)

**Reader profile:** Reads source code. Writes production runbooks. Understands CRDT semantics, gRPC streaming, and circuit breakers. Wants to know the exact metric name, the RBAC rule, the CRDT type used for worker registry sync, and where the radix tree HA is today (early 2026). Satisfied when they can design the complete observability stack, explain the router's HA limitations with precision, and understand where Layer 16's architecture ends and llm-d begins.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_sglang_model_gateway_docs.md` | SGLang GitHub (raw) | Full SGLang Model Gateway documentation: all 40+ Prometheus metrics, service discovery config (§13), pod labelling guide, RBAC YAML, PromQL queries, HA design recommendations. The definitive reference. |
| 02 | `02_nvidia_nim_helm_deployment.md` | NVIDIA NIM Docs | Industry reference for Helm-based LLM serving. Same pattern as Layer 16 (GPU pod + PVC + secret) but abstracted. Comparison table: NIM Helm vs. Layer 16 YAML. |
| 03 | `03_llmd_cncf_kubernetes_distributed_llm.md` | CNCF / llm-d | "What comes after Layer 16": disaggregated P/D serving, hierarchical KV cache offloading, inference-aware routing via GIE, scale-to-zero. CNCF Sandbox March 2026. Benchmarks. |
| 04 | `04_sglang_router_ha_grpc_mesh_issue.md` | SGLang GitHub Issue #10839 | Design document: gRPC bidirectional mesh + CRDTs for router HA without external state store. Explains *why* lesson/08 describes multi-replica cache degradation (radix tree sync not yet production-stable as of Feb 2026). |

---

## Recommended reading order

**Fast path (45 min):** 01 → 04
- 01 for the complete metric list and production recommendations.
- 04 to understand the HA design constraint described in lesson/08.

**Thorough path (90 min):** 01 → 04 → 03 → 02
- 03 after 04 to see how llm-d solves the same HA problem at scale.
- 02 as a calibration point: what NVIDIA automates vs. what Layer 16 teaches explicitly.

---

## How these map to Layer 16

| Layer 16 lesson | Most relevant L4 reference |
|---|---|
| `05_router_deployment.md` — all gateway flags | 01 (complete docs, load balancing policy parameters) |
| `06_service_discovery_internals.md` — `service_discovery.rs` | 01 (RBAC YAML, pod labelling, service discovery section) |
| `07_observability.md` — 40+ SMG metrics | 01 (all metric names, PromQL queries, alerting rules) |
| `08_high_availability.md` — multi-replica cache degradation | 04 (CRDT design, issue #10839, radix tree sync status) |
| Beyond Layer 16 — production scale | 03 (llm-d architecture and roadmap) |
| Beyond Layer 16 — industry Helm patterns | 02 (NIM Helm chart structure and values) |

---

## Common L4 limits to name for readers

These references **do not explain:**
- The actual Rust source code of `service_discovery.rs` (see `lesson/06_service_discovery_internals.md` and the local file `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs`).
- How `kube-rs`'s `watcher()` works internally (see [kube-rs docs](https://docs.rs/kube) and `examples/pod_watcher.rs`).
- The full vLLM production stack observability README (Helm charts, Prometheus Adapter, Grafana dashboard screenshots).

Those are L5 (source study) materials.
