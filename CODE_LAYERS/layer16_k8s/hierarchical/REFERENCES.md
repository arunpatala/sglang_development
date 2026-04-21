# References — LLM Router on Kubernetes

Organized by **reading level** (L1–L5) and **category**. Use this when writing or extending lesson content, locating production precedents, or designing exercises.

Layer 16 builds directly on Layer 15's routing gateway. The routing policies (round-robin, least-load, cache-aware) are unchanged. The single new concept is replacing the static `--worker-urls` list with Kubernetes service discovery. References therefore cover: **K8s deployment mechanics, RBAC, GPU scheduling, service discovery, observability, and high availability** — not routing theory (see `layer15_router/hierarchical/REFERENCES.md` for that).

---

## Primary source: SGLang's own K8s documentation

### SGLang Kubernetes Deployment Guide

- **Docs:** https://www.mintlify.com/sgl-project/sglang/deployment/kubernetes
- **Level:** L2–L3
- **What it contributes:**
  - Single-node and multi-node (StatefulSet) deployment YAMLs for SGLang on Kubernetes.
  - PVC configuration for model weight cache.
  - Prometheus metrics integration (`--enable-metrics`, `--metrics-port 8080`).
  - HPA configuration for auto-scaling SGLang deployments.
  - The reference point for how the SGLang project itself recommends K8s deployment — Layer 16 follows the same patterns.

### SGLang Router CLI Reference — Kubernetes Integration Section

- **Docs:** https://sgl-project.github.io/advanced_features/router.html
- **Level:** L3
- **What it contributes:**
  - Complete table of `--service-discovery` flags: `--service-discovery`, `--selector`, `--service-discovery-namespace`, `--service-discovery-port`, `--prefill-selector`, `--decode-selector`.
  - Standard mode vs PD mode service discovery commands.
  - Confirms that `--worker-urls` and `--service-discovery` are mutually exclusive — enabling service discovery replaces the static URL list entirely.
  - Origin of GitHub issue #3073 (Jan 2025) which first proposed K8s service discovery; closed Mar 2025 when the feature was merged.

### SGLang Model Gateway Docs — Service Discovery Section

- **Docs:** https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/sgl_model_gateway.md
- **Level:** L4
- **What it contributes:**
  - Section 13 of the gateway docs covers Kubernetes service discovery configuration in full.
  - Pod labelling guide for service discovery: labels workers must have to be discovered.
  - Best practice recommendation: "Use Kubernetes Service Discovery: Let the gateway automatically discover and manage workers."
  - Parameter table for `--service-discovery-namespace`, `--selector`, and PD mode selectors.
  - Alert rule YAML example (`HighErrorRate`) directly from the SGLang team.

### SGLang Distributed StatefulSet Example

- **GitHub:** https://github.com/sgl-project/sglang/blob/main/docker/k8s-sglang-distributed-sts.yaml
- **Level:** L4–L5
- **What it contributes:**
  - Production YAML for running SGLang in tensor-parallel mode across multiple nodes using a StatefulSet.
  - Shows how to use `POD_INDEX` env var for `--node-rank` (multi-node setup).
  - Separate Services for the dist-init port (5000) and the serving port (8000).
  - Reference for how SGLang's own team structures K8s manifests.

### SGLang Router HA: gRPC Mesh Design — Issue #10839

- **GitHub:** https://github.com/sgl-project/sglang/issues/10839
- **Level:** L4–L5
- **Published:** September 2025 (closed November 2025, implementation PR #14108)
- **What it contributes:**
  - Design document for the router's HA state-layer: how multiple router replicas eventually converge on the same radix tree without Redis or etcd.
  - CRDT-based approach (`rust-crdt`) with gRPC bidirectional streaming mesh; eventual consistency, not strong consensus.
  - State types being synced: worker registry, per-model radix trees, rate-limit buckets, and router membership.
  - Explains why Layer 16's lesson/08 describes multi-replica cache degradation: radix-tree sync was not yet stable as of early 2026 — confirmed by issue #18058 (Feb 2026): "Is multi-router replica data synchronization available already? No."
  - Key design constraint: no external datastore; all sync happens in-process via gRPC mesh on port configurable via `--router-mesh-port`.

### SGLang Router Roadmap — Issue #10341

- **GitHub:** https://github.com/sgl-project/sglang/issues/10341
- **Level:** L3–L4
- **Published:** September 2025 (closed November 2025)
- **What it contributes:**
  - Official roadmap for the SGLang router: multi-model support, gRPC mesh HA, worker management API, data-parallel aware routing, and semantic model selection.
  - Confirms the items that *are* checked (service discovery with new worker management API, policy registry per model family) vs items still pending (radix-tree sync across replicas, data mesh component).
  - Good orientation document for understanding what the router is evolving toward beyond Layer 16.

---

## Source code: the service discovery implementation

### `service_discovery.rs` — SGLang Model Gateway

- **File:** `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs`
- **Level:** L5 (source study)
- **What it contributes:**
  - The production Rust implementation of K8s service discovery used by Layer 16.
  - Key functions and their line numbers (as read in lesson/06):
    - `ServiceDiscoveryConfig` struct (line 33): holds `selector`, `namespace`, `port`, `check_interval`.
    - `matches_selector()` (line 88): AND logic across all label key=value pairs; empty selector returns `false` (safety guard).
    - `is_healthy()` (line 196): `is_ready && status == "Running"` — both conditions required.
    - `worker_url()` (line 200): `format!("http://{}:{}", self.ip, port)` — direct pod IP, no Service.
    - `start_service_discovery()` (line 206): `Client::try_default()` reads in-cluster ServiceAccount token (line 223); `Api::namespaced()` scopes to namespace (line 276).
    - Watcher loop (line 311–393): `watcher(pods, config).applied_objects()` → `filter_map(should_include)` → `try_for_each(handle_event)`.
    - Deletion detection (line 349): `deletion_timestamp.is_some()` → `handle_pod_deletion()`.
    - Exponential backoff (line 308–384): 1s → 2s → 4s → … → 300s max on API disconnect.
    - `handle_pod_event()` (line 399): dedup via `tracked_pods: HashSet` → `Job::AddWorker`.
    - `handle_pod_deletion()` (line 533): `tracked_pods.remove()` → `Job::RemoveWorker`.

---

## Multi-node GPU deployment: LeaderWorkerSet

### LeaderWorkerSet (LWS) — Overview

- **Docs:** https://lws.sigs.k8s.io/docs/overview/
- **GitHub:** https://github.com/kubernetes-sigs/lws
- **Level:** L3–L4
- **What it contributes:**
  - The Kubernetes-native API for deploying a group of pods as a single replicated unit — the correct primitive for multi-node tensor-parallel inference (e.g., a Llama-3.1 405B that spans 2 nodes × 8 GPUs each).
  - Key concept: "super-pod" — a `LeaderWorkerSet` replica consists of a leader pod (runs the model server) and N−1 worker pods (join the Ray/NCCL collective); all pods in the group are created and destroyed atomically.
  - `restartPolicy: RecreateGroupOnPodRestart` — if any pod in the group fails, the entire group is recreated; prevents partial tensor-parallel hangs.
  - Adopters include: vLLM, SGLang, NVIDIA NIM, NVIDIA Dynamo, llm-d, Amazon EKS — effectively the industry standard for multi-node LLM deployment on K8s.
  - Layer 16 uses single-GPU-per-pod workers and does not need LWS; this reference marks the line where Layer 16 ends and multi-node deployment begins.

### LeaderWorkerSet + vLLM — Deployment Guide

- **Docs:** https://docs.vllm.ai/en/stable/deployment/frameworks/lws/
- **Level:** L3–L4
- **What it contributes:**
  - Minimal YAML for deploying vLLM across 2 nodes × 8 GPUs using LWS: leader pod runs `vllm serve` with `--tensor-parallel-size 8 --pipeline_parallel_size 2`; worker pods run the Ray worker script.
  - `LWS_GROUP_SIZE` and `LWS_LEADER_ADDRESS` env vars are injected automatically by the LWS controller — no manual pod IP wiring needed.
  - Leader-only ClusterIP Service: the Service selector targets `role: leader` pods, so only the leader exposes port 8080; worker pods are unreachable directly.
  - Shared memory volume (`/dev/shm: Memory, 15Gi`) is required; same pattern as Layer 16's single-GPU worker YAML.

---

## GPU support on Kubernetes

### NVIDIA GPU Operator — Getting Started

- **Docs:** https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/25.9.2/getting-started.html
- **Level:** L2–L3
- **What it contributes:**
  - The standard way to enable `nvidia.com/gpu` as a schedulable resource in a Kubernetes cluster.
  - Helm install command: `helm install --wait --generate-name -n gpu-operator --create-namespace nvidia/gpu-operator`.
  - Explains Node Feature Discovery (NFD) dependency for GPU node labelling (`feature.node.kubernetes.io/pci-10de.present=true`).
  - GPU taint/toleration patterns: GPU nodes are typically tainted to prevent CPU workloads landing on them; worker pods must add the corresponding toleration.
  - Verification: `kubectl get nodes -o json | jq '.items[].metadata.labels | keys | any(startswith("feature.node.kubernetes.io"))'`.
  - Current version: v25.9.2 (as of 2026).

---

## Production reference: vLLM on Kubernetes

### vLLM Kubernetes Deployment Guide

- **Docs:** https://docs.vllm.ai/en/stable/deployment/k8s.html
- **Level:** L2–L3
- **What it contributes:**
  - Production-grade YAML for deploying vLLM with GPU resources, PVC, Secret (HF token), readiness/liveness probes.
  - Best practice note on readiness probe `failureThreshold`: "measure how much time it takes for the model server to show it's ready to serve" — same methodology recommended in lesson/04 for `initialDelaySeconds`.
  - Shared memory (`/dev/shm`) volume configuration (from `REPOS/vllm/docs/deployment/k8s.md`) — directly referenced in lesson/04_worker_deployment.md.
  - Local file: `REPOS/vllm/docs/deployment/k8s.md` (read during Layer 16 chapter creation).

### vLLM Production Stack

- **Docs:** https://docs.vllm.ai/projects/production-stack
- **GitHub:** https://github.com/vllm-project/production-stack
- **Level:** L3–L4
- **What it contributes:**
  - Official K8s-native reference stack for vLLM: Helm chart, Prometheus/Grafana observability, KV-cache-aware routing, HPA with custom metrics.
  - Architecture directly parallel to Layer 16: serving engine pods + router pod + observability stack, all deployed via Helm.
  - Prometheus Adapter configuration to expose `vllm_num_requests_waiting` as a custom K8s metric for HPA — same approach used in `k8s/sglang/templates/hpa.yaml`.
  - Grafana dashboard with KV cache usage, request throughput, and queue depth panels.
  - Released January 22, 2025 (Berkeley-UChicago collaboration, now official vLLM project).

### vLLM Production Stack RBAC Role

- **GitHub:** https://github.com/vllm-project/production-stack/blob/99ab33ab/operator/config/rbac/role.yaml
- **Level:** L4–L5
- **What it contributes:**
  - Shows the minimal RBAC role for the vLLM operator: `pods: get, list, watch` is the only core resource needed for service discovery (same as Layer 16's `Role`).
  - Confirms the Layer 16 design choice: pods need only `get/list/watch`, not `create/delete/patch`.
  - PR #647 (Aug 2025): "Reduce RBAC permissions for secrets to least privilege" — real-world confirmation that minimising RBAC scope is an active production concern.

### vLLM Helm Chart (examples)

- **GitHub:** `REPOS/vllm/examples/online_serving/chart-helm/`
- **Level:** L4–L5
- **What it contributes:**
  - Full Helm chart for vLLM: `templates/deployment.yaml`, `values.yaml`, probes, GPU resources, shared memory volume.
  - Directly used as reference when building `k8s/sglang/templates/worker-deployment.yaml`.
  - Shows production Helm chart structure: `_helpers.tpl`, `values.yaml` defaults, conditional templates (`{{- if .Values.ingress.enabled }}`).

---

## NVIDIA NIM on Kubernetes (Helm reference)

### NVIDIA NIM LLM Helm Chart Deployment

- **Docs:** https://docs.nvidia.com/nim/large-language-models/latest/deployment/kubernetes-deployment/helm-k8s.html
- **Level:** L3–L4
- **What it contributes:**
  - The industry reference for Helm-based LLM serving on K8s from NVIDIA.
  - Key values: `image.repository`, `image.tag`, `model.ngcAPISecret`, `persistence` (PVC), `resources.limits.nvidia.com/gpu`.
  - PVC sizing guidance: `persistence.size` should be adjusted based on model size and cache usage — same advice in Layer 16's `values.yaml` comments.
  - Multi-node models via LeaderWorkerSet (LWS) — beyond Layer 16 scope but useful context for what comes after single-GPU-per-pod setups.
  - Good comparison point: NIM uses a similar pattern (single GPU pod + PVC + secret) but hides it behind higher-level abstractions.

---

## GPU metrics: NVIDIA DCGM Exporter

### NVIDIA DCGM Exporter — GitHub and Docs

- **GitHub:** https://github.com/nvidia/dcgm-exporter
- **Docs:** https://docs.nvidia.com/datacenter/dcgm/3.1/gpu-telemetry/dcgm-exporter.html
- **Helm chart:** https://nvidia.github.io/dcgm-exporter/
- **Level:** L3–L4
- **What it contributes:**
  - The standard DaemonSet for exporting GPU hardware metrics (utilization, memory, temperature, SM clock) to Prometheus from every GPU node.
  - Key metric: `DCGM_FI_DEV_GPU_UTIL` (GPU compute utilization, 0–100%); `DCGM_FI_DEV_FB_USED` (framebuffer / VRAM used in MiB).
  - Deployed as a DaemonSet — one exporter pod per GPU node; metrics are labelled with `pod`, `namespace`, and `container` so per-pod GPU usage is visible in Prometheus.
  - The GPU Operator (referenced above) deploys dcgm-exporter automatically when monitoring is enabled — for clusters already using the GPU Operator, no separate install is needed.
  - Official Grafana dashboard: https://grafana.com/grafana/dashboards/12239
  - Current version: 4.5.2-4.8.1 (February 2026). Installed via: `helm install --generate-name gpu-helm-charts/dcgm-exporter`.
  - Why Layer 16 does not include it in the default stack: lesson/07 scrapes the router's own metrics (40+ SMG metrics), not hardware GPU metrics; DCGM exporter is the next layer of observability added when GPU utilization alerting is needed.

---

## Observability: Prometheus + Grafana for GPU LLM workloads

### GPU Autoscaling on Kubernetes: Prometheus Metrics to HPA with vLLM on GKE

- **URL:** https://medium.com/@akrem.issaoui1/gpu-autoscaling-on-kubernetes-from-prometheus-metrics-to-hpa-with-vllm-on-gke-48578e5753d1
- **Level:** L3
- **Published:** April 2026
- **What it contributes:**
  - End-to-end walkthrough: `kube-prometheus-stack` → ServiceMonitor → Prometheus → custom metrics → HPA.
  - The three-part alignment problem for cross-namespace ServiceMonitors: named ports + namespace selector + release label must all match.
  - Real metric names for vLLM (applicable to SGLang): `kv_cache_usage_perc` (not `gpu_cache_usage_perc`) — the "colon vs underscore trap."
  - Key insight directly quoted: "CPU-based autoscaling doesn't work for LLM workloads. A GPU can be 100% busy while CPU sits idle."
  - HPA using `vllm:num_requests_waiting` as the custom metric — same pattern used in `k8s/sglang/templates/hpa.yaml`.
  - KV cache pressure and queue depth alerts via PrometheusRule — mirrors Layer 16's alert design.

### Monitor LLM Inference in Production (2026): Prometheus & Grafana

- **URL:** https://glukhov.org/observability/monitoring-llm-inference-prometheus-grafana/
- **Level:** L2–L3
- **Published:** 2026
- **What it contributes:**
  - Covers vLLM, TGI, and llama.cpp — the three main open-source inference engines.
  - ServiceMonitor pattern for Kubernetes Prometheus Operator: create a named port on the Service, reference it in the `endpoints[].port` field.
  - SLO-style alert examples: high p95 latency (burn rate), queue time p99, error rate >1%, KV cache usage >90%.
  - Troubleshooting guide for "Prometheus target is DOWN" — useful when ServiceMonitor is configured but targets don't appear.

### vLLM Production Stack Observability README

- **GitHub:** https://github.com/vllm-project/production-stack/blob/99ab33ab/observability/README.md
- **Level:** L3–L4
- **What it contributes:**
  - Reference for deploying `kube-prom-stack` alongside the inference stack.
  - How to use Prometheus Adapter to expose `vllm_num_requests_waiting` as a K8s custom metric for HPA.
  - Grafana dashboard screenshots for inference stack monitoring.
  - Port-forward commands for accessing Prometheus (9090) and Grafana (3000) during development.

---

## Autoscaling beyond HPA: KEDA

### KEDA — Kubernetes Event-Driven Autoscaling (Prometheus Scaler)

- **Docs:** https://keda.sh/docs/latest/scalers/prometheus/
- **GitHub:** https://github.com/kedacore/keda
- **Level:** L3
- **What it contributes:**
  - KEDA sits between a metric source (Prometheus, Redis, Azure Service Bus, etc.) and the Kubernetes HPA controller, enabling scale-to-zero and scale-from-zero on arbitrary application metrics.
  - The Prometheus scaler: point `serverAddress` at any Prometheus endpoint, write a PromQL `query`, set a `threshold`, and KEDA drives the HPA accordingly — no prometheus-adapter required.
  - For GPU LLM workloads: scale workers on `smg_worker_requests_active` (SGLang) or `vllm:num_requests_waiting` (vLLM) rather than CPU; KEDA exposes the metric through the Kubernetes External Metrics API, which HPA reads natively.
  - `cooldownPeriod` and `stabilizationWindowSeconds` in `advanced.horizontalPodAutoscalerConfig.behavior` prevent flapping — critical for GPU pods that take 60–120s to become ready.
  - Key difference from prometheus-adapter: prometheus-adapter translates Prometheus metrics into the *Custom Metrics API* (per-object metrics); KEDA uses the *External Metrics API* (cluster-scoped). Both work with HPA; KEDA is simpler to configure and supports scale-to-zero.
  - Layer 16 mentions KEDA in lesson/08 as the recommended autoscaling path for GPU workloads; this is the reference for its configuration.

### Auto-Scaling GPU Inference Pods in Kubernetes: KEDA, Custom Metrics, and Cost Guards

- **URL:** https://markaicode.com/auto-scaling-gpu-inference-kubernetes/
- **Level:** L3
- **What it contributes:**
  - End-to-end walkthrough: KEDA install via Helm → `ScaledObject` targeting inference queue depth → `behavior` tuning for conservative scale-down → cost circuit-breaker pattern.
  - ScaledObject with `pollingInterval: 30`, `cooldownPeriod: 600`, `stabilizationWindowSeconds: 300` — production-safe values that prevent thrashing during load spikes.
  - Cost guard pattern: second KEDA trigger based on cloud billing API metric; if hourly cost exceeds budget, the trigger prevents further scale-up (complementary to `maxReplicaCount`).
  - Scaling latency benchmark: time from traffic spike to N-th pod serving traffic; dominated by GPU pod startup time (image pull + model load), not KEDA reaction time.

### GPU-Aware Autoscaling for GenAI APIs with KEDA + NVIDIA DCGM

- **URL:** https://towardsaws.com/gpu-aware-autoscaling-for-genai-apis-with-keda-nvidia-dcgm-bbaeee33330e
- **Level:** L3
- **Published:** September 2025
- **What it contributes:**
  - Walkthrough: DCGM exporter → Prometheus → KEDA `ScaledObject` using `DCGM_FI_DEV_GPU_UTIL` as the scaling signal.
  - Shows `TriggerAuthentication` CRD for secure Prometheus connection from KEDA.
  - GPU utilization as a scaling metric: unlike request queue depth, GPU utilization is a hardware signal independent of the inference framework — works with any inference server.
  - Caveat: GPU utilization lags queue depth by seconds; queue depth is more responsive for latency-sensitive workloads. Use GPU utilization when queue depth metrics are unavailable.

---

## High Availability: Session affinity and ingress routing

### Ingress-NGINX: Sticky Sessions with Cookie Affinity

- **Docs:** https://kubernetes.github.io/ingress-nginx/examples/affinity/cookie/
- **Level:** L3
- **What it contributes:**
  - Official Ingress-NGINX documentation for cookie-based session affinity.
  - The `INGRESSCOOKIE` mechanism: randomly generated key → consistent hash → same upstream pod.
  - `affinity-mode: balanced` vs `persistent` — Layer 16 uses the default (`balanced`) to allow redistribution when router pods are rebalanced.
  - Caveat: "if the backend pool grows NGINX will keep sending requests through the same server of the first request, even if it's overloaded" — confirms that `balanced` mode is preferred for router pods.

### Revisiting Session Affinity in Kubernetes

- **URL:** https://medium.com/@rajeshlagishetty/session-affinity-in-kubernetes-899e243f1ead
- **Level:** L2
- **Published:** January 2025
- **What it contributes:**
  - Clear comparison: `sessionAffinity: ClientIP` (K8s Service) vs Nginx cookie affinity vs IPVS source hashing.
  - Critical limitation directly quoted: "both approaches of Affinity/Source Hashing will fail if source IPs are NATed" — explains why `X-Session-ID` header hashing is more reliable for users behind corporate NAT or VPN.
  - Maps directly to Layer 16's decision guide (lesson/08): use ClientIP for simple setups, header-based hashing when clients share IPs.

### Nginx `upstream-hash-by` PR (original implementation)

- **GitHub PR:** https://github.com/kubernetes/ingress-nginx/pull/1490
- **Level:** L5
- **What it contributes:**
  - The original PR (merged 2017) that added `nginx.ingress.kubernetes.io/upstream-hash-by` annotation.
  - Shows the underlying Nginx config generated: `hash $request_uri consistent;` in the upstream block.
  - Important caveat from Stack Overflow (2025): `upstream-hash-by` is only deterministic *within* a single ingress-nginx replica. Multiple ingress controller replicas may have different endpoint orderings, causing the same hash to map to different backends. For Layer 16, this means: keep ingress-nginx at 1 replica OR use cookie-based affinity for true stickiness.

---

## Where the ecosystem is going: inference-aware K8s routing

### Kubernetes Gateway API Inference Extension (GIE)

- **K8s blog:** https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/
- **Docs:** https://gateway-api-inference-extension.sigs.k8s.io/
- **GitHub:** https://github.com/kubernetes-sigs/gateway-api-inference-extension
- **Level:** L3–L4
- **Published:** June 2025
- **What it contributes:**
  - The official Kubernetes project that adds LLM-aware routing on top of the standard Gateway API — without requiring a custom router like SGLang's.
  - Two new CRDs: `InferenceModel` (logical model endpoint, criticality level) and `InferencePool` (backend pods + routing policy).
  - Endpoint Picker (EPP): a pluggable sidecar that selects the best backend pod for each request based on KV cache load, queue depth, loaded LoRA adapters, and criticality — the same problem SGLang's cache-aware policy solves, but as a K8s-native standard.
  - Supported implementations (as of 2025–2026): GKE, Istio, NGINX Gateway Fabric, Agentgateway (used by llm-d). SGLang is listed as a supported model server.
  - Why this matters for Layer 16: Layer 16 routes at the application layer (SGLang's own router); GIE routes at the L7 gateway layer and is the direction the K8s community is standardizing on. Understanding GIE explains where SGLang's `--service-discovery` flag fits in the broader ecosystem.

### llm-d — Kubernetes-Native Distributed LLM Inference

- **Website:** https://llm-d.ai/
- **GitHub:** https://github.com/llm-d/llm-d/
- **CNCF blog:** https://www.cncf.io/blog/2026/03/24/welcome-llm-d-to-the-cncf-evolving-kubernetes-into-sota-ai-infrastructure/
- **Level:** L4
- **Published:** May 2025 (launched); CNCF Sandbox March 2026
- **What it contributes:**
  - The "what comes after" reference for Layer 16: llm-d is the middleware layer between a single inference engine (vLLM/SGLang) and cluster-level orchestration (KServe), adding disaggregated serving, hierarchical KV cache offloading, and inference-aware routing.
  - Uses LeaderWorkerSet (LWS) for multi-node replicas, Kubernetes Gateway API Inference Extension for routing, and vLLM as the model server — all three are references in this file.
  - Key features beyond Layer 16 scope: Prefill/Decode disaggregation (separate prefill and decode node pools), hierarchical KV cache offloading (GPU → CPU → NVMe), prefix-cache-aware routing that maintains near-zero TTFT under high QPS, and scale-to-zero autoscaling.
  - Benchmark (v0.5, Feb 2026): ~3.1k tok/s per B200 decode GPU; 16×16 B200 prefill/decode topology achieves ~50k output tok/s with order-of-magnitude TTFT reduction vs round-robin baseline.
  - Donated to CNCF Sandbox March 2026 by IBM Research, Red Hat, and Google Cloud; backed by NVIDIA, AMD, CoreWeave, Hugging Face, Intel.
  - Layer 16 covers the foundation (one router, service discovery, cache-aware routing); llm-d is the production-scale evolution of the same architecture.

---

## Kubernetes runtime reference: the `kube` Rust crate

### kube-rs — Rust Kubernetes Client and Controller Runtime

- **GitHub:** https://github.com/kube-rs/kube
- **Docs:** https://docs.rs/kube
- **Crates.io:** https://crates.io/crates/kube-client (v3.1.0, March 2026)
- **Level:** L4–L5
- **What it contributes:**
  - The Rust library used by `service_discovery.rs` to interact with the Kubernetes API.
  - `Client::try_default()`: tries in-cluster config first (ServiceAccount token), falls back to kubeconfig — exactly what line 223 of `service_discovery.rs` uses.
  - `Api::namespaced(client, "production")`: scopes all API calls to one namespace (line 277 of `service_discovery.rs`).
  - `watcher(api, Config::default()).applied_objects()`: the streaming watcher that drives the event loop (line 313 of `service_discovery.rs`). The runtime handles relisting and reconnection automatically under the hood.
  - `pod_watcher.rs` example: https://github.com/kube-rs/kube/blob/main/examples/pod_watcher.rs — the canonical usage pattern, nearly identical to what `service_discovery.rs` implements.
  - CNCF Sandbox project. Tested against Kubernetes v1.31+. Current version: 3.1.0.

---

## How this maps to the hierarchical model

| Cluster | Key references |
|---------|---------------|
| **16a why** — static URLs break | SGLang K8s docs (L2); SGLang router CLI issue #3073 (L3) |
| **16b architecture** — K8s components | vLLM K8s guide (L2); SGLang deployment guide (L2); NVIDIA NIM Helm (L3) |
| **16c rbac** — ServiceAccount, Role, RoleBinding | vLLM production-stack `role.yaml` (L4); RBAC PR #647 (L4) |
| **16d worker deployment** — GPU pods, probes, PVC | NVIDIA GPU Operator (L3); vLLM K8s guide probes section (L3); SGLang distributed StatefulSet (L4); LWS overview (L3, multi-node context) |
| **16e router deployment** — `--service-discovery` flags | SGLang Router CLI reference (L3); SGLang Model Gateway docs §13 (L4) |
| **16f service discovery internals** — `service_discovery.rs` | `service_discovery.rs` source (L5); kube-rs `watcher` docs (L4); `pod_watcher.rs` example (L5) |
| **16g observability** — Prometheus, ServiceMonitor, alerts | GKE autoscaling article (L3); LLM monitoring blog (L2); vLLM production stack observability (L3); DCGM Exporter (L3, GPU hardware metrics) |
| **16h high availability** — session affinity, X-Session-ID, HPA | Ingress-NGINX sticky sessions docs (L3); Session affinity medium article (L2); `upstream-hash-by` PR (L5); KEDA Prometheus scaler (L3); KEDA GPU article (L3); SGLang Router HA issue #10839 (L4, radix tree sync roadmap) |
| **Beyond Layer 16** — multi-node, inference-aware routing | LWS + vLLM (L3); Gateway API Inference Extension (L3); llm-d (L4) |

---

## See also

- `lesson/00_outline.md` — full section list for Layer 16.
- `lesson/summary.md` — narrative overview of all 8 sections.
- `k8s/README.md` — Helm chart quickstart and values reference.
- `layer15_router/hierarchical/REFERENCES.md` — routing theory references (Preble, Mitzenmacher, SGLang gateway policies). Layer 16 inherits all of those — this file covers only the K8s-specific additions.
- `WRITING_GUIDE/PERSONAS.md` — which reference depth fits which reader level.
