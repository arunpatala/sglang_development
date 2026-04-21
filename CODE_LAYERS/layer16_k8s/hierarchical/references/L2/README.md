# L2 References: LLM Inference on Kubernetes

**Level:** L2 — Practitioner deployment (configuration, YAML, cluster prerequisites)

**Reader profile:** Knows Kubernetes basics (pods, deployments, services). Has deployed web apps on K8s. Wants to understand how GPU inference pods differ from regular workloads. Satisfied when they can write a working GPU worker Deployment YAML and understand what the GPU Operator does.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_vllm_kubernetes_deployment.md` | vLLM Docs | Production GPU deployment YAML: PVC, Secret, GPU resource requests, readiness/liveness probes, shared memory volume (`/dev/shm`). The foundational pattern Layer 16 follows. |
| 02 | `02_nvidia_gpu_operator_getting_started.md` | NVIDIA | Cluster prerequisite: enables `nvidia.com/gpu` as schedulable K8s resource. NFD, driver container, device plugin. Without this, `nvidia.com/gpu: "1"` in pod specs fails. |
| 03 | `03_monitor_llm_inference_prometheus_grafana.md` | glukhov.org | Best general guide: golden signals for LLM workloads, ServiceMonitor pattern, key metric names for vLLM/TGI/SGLang, PromQL examples, alert rules, troubleshooting. |
| 04 | `04_session_affinity_kubernetes.md` | Medium | Session affinity options: ClientIP, IPVS Source Hashing, Nginx cookie. Explains why NATed clients break ClientIP affinity → motivates `X-Session-ID` header hashing. |

---

## Recommended reading order

**Fast path (30 min):** 01 → 02
- 01 for the GPU pod YAML pattern used throughout Layer 16.
- 02 to understand the cluster prerequisite (`nvidia.com/gpu` scheduling).

**Thorough path (60 min):** 01 → 02 → 03 → 04
- 03 for observability: ServiceMonitor, PromQL, alerts for LLM workloads.
- 04 for HA context: why session affinity is needed and which approach to use.

---

## How these map to Layer 16

| Layer 16 lesson | Most relevant L2 reference |
|---|---|
| `01_static_urls_break.md` — why we need K8s | Implied by 01 (static deployments, scaling) |
| `02_k8s_architecture.md` — K8s components | 01 (Deployment, Service, PVC pattern), 02 (GPU Operator) |
| `03_prerequisites_rbac.md` — cluster setup | 02 (GPU Operator prerequisites) |
| `04_worker_deployment.md` — GPU pods | 01 (complete GPU pod YAML with probes, PVC, /dev/shm) |
| `07_observability.md` — Prometheus, ServiceMonitor | 03 (ServiceMonitor, metric names, PromQL) |
| `08_high_availability.md` — session affinity | 04 (ClientIP vs cookie vs header hashing) |

---

## Common L2 limits to name for readers

These articles **do not explain:**
- How the SGLang service discovery watcher loop (`service_discovery.rs`) works internally.
- Why the router needs RBAC (the connection between pod watching and load balancing).
- How KEDA differs from HPA with prometheus-adapter.
- What LWS adds for multi-node models.

Those live in L3 (technical) and L4 (advanced) references.
