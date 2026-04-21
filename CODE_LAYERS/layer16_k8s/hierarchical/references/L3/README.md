# L3 References: LLM Inference on Kubernetes

**Level:** L3 — Technical (mechanisms, production configuration, ecosystem context)

**Reader profile:** Knows K8s RBAC, Helm, and has deployed services to production. Reads Go/Rust/Python when necessary to understand behavior. Wants to understand *how* the SGLang router's service discovery works, what KEDA actually does differently from prometheus-adapter, and where LWS and GIE fit. Satisfied when they can configure KEDA, explain the ServiceMonitor alignment problem, and describe the difference between L7 gateway routing and application-layer routing.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_sglang_router_cli_k8s_flags.md` | SGLang Docs | Complete `--service-discovery` flag table, PD mode discovery, cache-aware policy parameters, data-parallel-aware routing, multi-model support. The Layer 16 router command is documented here. |
| 02 | `02_gpu_autoscaling_prometheus_hpa_vllm.md` | Medium (April 2026) | End-to-end HPA pipeline: kube-prometheus-stack → ServiceMonitor → prometheus-adapter → Custom Metrics API → HPA. Three-part alignment problem. "CPU autoscaling doesn't work for GPU." |
| 03 | `03_keda_prometheus_scaler.md` | KEDA Docs | KEDA `ScaledObject` reference with Prometheus trigger. Scale-to-zero, scale-from-zero. `cooldownPeriod`, `stabilizationWindowSeconds` tuning for GPU pods. Cloud managed Prometheus support. |
| 04 | `04_ingress_nginx_sticky_sessions.md` | Ingress-NGINX Docs | Cookie affinity annotations, `balanced` vs `persistent` mode, INGRESSCOOKIE mechanism, `upstream-hash-by` caveats for multi-replica ingress. |
| 05 | `05_lws_vllm_multi_node_deployment.md` | LWS + vLLM Docs | LeaderWorkerSet for 2-node × 8-GPU vLLM. Marks the boundary between Layer 16 (single-GPU-per-pod) and multi-node tensor-parallel deployment. LWS env vars, leader-only Service. |
| 06 | `06_gateway_api_inference_extension.md` | Kubernetes Blog (June 2025) | GIE: InferenceModel and InferencePool CRDs, Endpoint Picker (EPP), benchmarks vs round-robin. Where the K8s ecosystem is standardizing LLM-aware routing. |

---

## Recommended reading order

**Fast path (45 min):** 01 → 03 → 04
- 01 to lock in the router's K8s flags.
- 03 for KEDA as the simpler autoscaling path.
- 04 for session affinity configuration.

**Thorough path (90 min):** 01 → 02 → 03 → 04 → 05 → 06
- 02 before 03 to understand the problem KEDA solves (prometheus-adapter complexity).
- 05 after understanding Layer 16 to see what comes next (multi-node).
- 06 to understand the direction the K8s ecosystem is heading.

---

## How these map to Layer 16

| Layer 16 lesson | Most relevant L3 reference |
|---|---|
| `05_router_deployment.md` — `--service-discovery` | 01 (complete flag table), 04 (RBAC confirmation) |
| `06_service_discovery_internals.md` — watcher loop | 01 (CLI → behavior mapping) |
| `07_observability.md` — ServiceMonitor, alerts | 02 (three-part alignment problem), 03 (KEDA for scaling signals) |
| `08_high_availability.md` — session affinity, KEDA | 03 (ScaledObject config), 04 (cookie annotation), 05 (LWS context for "what's beyond") |
| Layer 16 boundary → beyond | 05 (LWS), 06 (GIE) |

---

## Common L3 limits to name for readers

These articles **do not explain:**
- The internal Rust implementation of `service_discovery.rs` (watcher loop, kube-rs API).
- How the SGLang router's CRDT-based gRPC mesh works for HA state sync (issue #10839).
- The production observability recommendation table in the SGLang model gateway docs (40+ metrics).
- The full llm-d architecture (disaggregated prefill/decode, hierarchical KV cache).

Those live in L4 references and lesson files (`06_service_discovery_internals.md`, `07_observability.md`).
