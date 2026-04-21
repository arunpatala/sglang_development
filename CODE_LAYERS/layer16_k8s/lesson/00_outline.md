# Layer 16 — Lesson Outline

## What This Lesson Covers

Layer 15 introduced a gateway process — `router.py` — that routes requests across multiple engine instances using a configurable policy. That gateway was configured with a static list of backend URLs in `config.yml`. In Kubernetes, static URLs break: pods get a new IP every restart, `kubectl scale` adds new pods silently, and the router has no mechanism to discover them. Layer 16 makes the router Kubernetes-aware by replacing the static URL list with Kubernetes service discovery.

No new routing policies are added. The `Worker`, `LoadBalancingPolicy`, and `Router` classes from Layer 15 carry forward unchanged. The new work is entirely in the infrastructure layer: how worker pods are labelled and deployed, how the router discovers them at runtime via the Kubernetes API, how RBAC grants the minimum required permissions, and how Prometheus metrics and health probes connect to cluster-level observability.

The lesson follows the deployment path: why static URLs are insufficient → what changes in the K8s architecture → prerequisites and permissions → deploying GPU worker pods → deploying the router with service discovery enabled → how the discovery loop works under the hood → observability → router high availability and the cache-tree consistency problem.

The single new concept introduced is the **Kubernetes watcher loop**: a long-lived connection to the K8s API server that delivers pod events (Added, Modified, Deleted) filtered by label selector, allowing the router to add and remove workers in real time without polling or configuration changes.

---

## Sections

### 01 — The Problem: Static URLs Break in Kubernetes (`01_static_urls_break.md`)
- Layer 15's `config.yml` lists backend URLs explicitly: `http://localhost:8114`, `http://localhost:8115`; works for local development, fails in Kubernetes
- Pod IP assignment in Kubernetes: every new pod gets a fresh IP from the cluster CIDR; a pod restart changes its IP silently; no DNS record for the pod IP survives the restart
- The scaling gap: `kubectl scale deployment sglang-worker --replicas=6` creates four new pods with four new IPs — the router's static list knows nothing about them
- The discovery contract: Kubernetes offers a `watch` API on any resource (pods, endpoints, services) that streams events as they happen; the router subscribes to pod events filtered by label selector and updates its worker list accordingly
- What Layer 16 adds: a `--service-discovery` flag, RBAC objects (ServiceAccount + Role + RoleBinding), labelled worker Deployments, and a router Deployment that starts with zero static URLs and builds its worker list from the cluster

### 02 — The Architecture in Kubernetes (`02_k8s_architecture.md`)
- How the call path changes: client → K8s Service (stable ClusterIP DNS) → Router Pod → Worker Pods (discovered dynamically); no static URL anywhere in the routing path after startup
- Component roles: K8s Service provides the stable hostname; Router Deployment runs one or more router replicas (no GPU, CPU-only); Worker Deployment runs N GPU pods each serving one engine; PVC stores model weights shared across pod restarts
- Label conventions: `app=sglang-worker`, `component=inference`, `model=llama-3-8b`; the router's `--selector` must match these labels; additional labels (e.g. `env=prod`, `team=platform`) can narrow scope
- Namespace isolation: the router watches pods in one namespace; `--service-discovery-namespace production` scopes discovery to `production`; cross-namespace watching requires a `ClusterRole` instead of a `Role` (avoid in production)
- The startup sequence: router pod starts → connects to K8s API → lists all matching pods currently Running + Ready → adds them as workers → begins watching for future changes; existing traffic is served from the initial list immediately

### 03 — Prerequisites and RBAC (`03_prerequisites_rbac.md`)
- GPU Operator or NVIDIA device plugin: required on each GPU node so that `nvidia.com/gpu: 1` resource requests are schedulable; verify with `kubectl describe nodes | grep -A5 "Allocatable"` and look for `nvidia.com/gpu`
- Kubernetes version ≥ 1.26: required for stable `apps/v1` Deployment API and modern Pod conditions used by the health check
- ServiceAccount: the router pod must run under a dedicated `ServiceAccount` (not `default`); `serviceAccountName: sglang-router` in the pod spec
- Role (namespace-scoped, least privilege):
  ```yaml
  rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
  ```
  The router only needs to observe pods — never create, delete, or modify them; using `ClusterRole` widens the blast radius unnecessarily
- RoleBinding: binds the `Role` to the `ServiceAccount` within the target namespace; if the router and workers are in different namespaces, a second RoleBinding is needed in the workers' namespace
- Why `list` in addition to `watch`: the `kube` Rust crate used in `service_discovery.rs` (line 223) calls `Client::try_default()` which detects in-cluster config from the mounted service account token; on startup it `list`s existing pods to bootstrap state before the `watch` stream begins
- HuggingFace token secret: `kubectl create secret generic hf-token-secret --from-literal=token=<HF_TOKEN>`; injected into worker pods via `secretKeyRef`; never put in ConfigMap or baked into the image

### 04 — Worker Deployment YAML (`04_worker_deployment.md`)
- Deployment structure: `replicas: 4`, `selector.matchLabels` must match `template.metadata.labels` exactly; the router's `--selector` matches those same labels
- GPU resource request: `nvidia.com/gpu: "1"` in both `requests` and `limits`; K8s scheduler places the pod on a node with a free GPU slot; omitting `limits` can cause over-scheduling
- Shared memory volume (required for tensor parallelism): `emptyDir: {medium: Memory, sizeLimit: "2Gi"}` mounted at `/dev/shm`; without it, tensor parallel engines fail to allocate shared memory segments; drawn from vLLM K8s guide (`REPOS/vllm/docs/deployment/k8s.md`)
- Liveness and readiness probes: both hit `/health` on port 8000; `initialDelaySeconds: 120` (or more for large models — profile once); the readiness probe controls when the pod's `Ready` condition becomes `True`; this is the gate the router's `is_healthy()` check reads
- Why the readiness probe is the critical path: `service_discovery.rs` line 121–127 defines `is_healthy()` as `self.is_ready && self.status == "Running"`; `is_ready` comes from the pod's `Ready` condition; a pod that is Running but not Ready is **never added** to the router's worker pool
- Model weights via PVC: `PersistentVolumeClaim` mounted at `/root/.cache/huggingface`; avoids re-downloading the model on every pod restart; `accessMode: ReadWriteOnce` sufficient for single-node workers; for multi-node use `ReadWriteMany` with a shared filesystem
- Node affinity and tolerations: `nodeSelector: {gpu-type: nvidia-a100}` schedules only on labelled GPU nodes; `tolerations` for the GPU taint (`nvidia.com/gpu: NoSchedule`) allows the pod to land on dedicated GPU nodes that are tainted for general workloads
- SGLang engine flags: `--enable-prefix-caching` activates `RadixCache` (from Layer 12) inside each engine; without it, the router's prefix-aware routing still routes correctly but the engine gains nothing from cache hits

### 05 — Router Deployment YAML (`05_router_deployment.md`)
- Router pod needs no GPU: CPU-only; resource requests `cpu: "2", memory: "4Gi"` are sufficient for the Rust routing process at typical concurrency
- The critical CLI change — drop `--worker-urls`, add `--service-discovery`:
  ```bash
  python -m sglang_router.launch_router \
    --service-discovery \
    --selector app=sglang-worker component=inference \
    --service-discovery-namespace production \
    --service-discovery-port 8000 \
    --policy cache_aware \
    --cache-threshold 0.5 \
    --balance-abs-threshold 32 \
    --host 0.0.0.0 --port 30000 \
    --prometheus-port 29000
  ```
- ServiceAccount attachment: `spec.serviceAccountName: sglang-router`; without this, the pod runs as `default` which has no permission to list/watch pods; the K8s API call fails and the router starts with zero workers
- K8s Service for the router: `type: ClusterIP` gives the router a stable DNS name (`sglang-router.production.svc.cluster.local:30000`) regardless of which router pod is currently running; use `type: LoadBalancer` or an `Ingress` for external access
- ConfigMap for router flags: pass the CLI args via a ConfigMap-backed environment variable or a command args list in the Deployment spec; avoids rebuilding the image to change thresholds
- Separate Prometheus Service: expose port 29000 on a separate K8s Service with label `monitoring: enabled` for Prometheus Operator `ServiceMonitor` scraping; do not expose it externally
- Router startup check: the router's `/readiness` endpoint returns 503 until at least one healthy worker is registered; use this as the readiness probe for the router pod so traffic is only routed after the first worker is discovered

### 06 — How Service Discovery Works (`06_service_discovery_internals.md`)
- Source file: `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs` — the production Rust implementation; Layer 16 uses this directly via `python -m sglang_router.launch_router --service-discovery`
- Kubernetes client setup (line 223): `Client::try_default()` detects in-cluster config from the service account token mounted at `/var/run/secrets/kubernetes.io/serviceaccount/`; no kubeconfig file needed inside the cluster
- The watcher stream (line 313): `watcher(pods, watcher_config).applied_objects()` opens a long-lived connection to the K8s API server that pushes pod events; this is a server-sent stream — not polling
- Label filtering (lines 88–96): `selector.iter().all(|(k, v)| labels.get(k) == Some(v))` — all key=value pairs in `--selector` must match; an `app=sglang-worker` selector matches any pod with that label; `app=sglang-worker component=inference` ANDs the two conditions
- Pod health gate (lines 121–127): a pod passes `is_healthy()` only when both `is_ready == true` (Ready condition = "True") and `status == "Running"` (phase); Pending, Terminating, and CrashLoopBackOff pods are excluded
- Worker addition (lines 427–530): when a healthy matching pod appears, `Job::AddWorker` is submitted to the job queue; the job validates the worker, starts health checks, and registers it in the policy's worker list; Prometheus records `smg_discovery_registrations_total{source="kubernetes", result="success"}`
- Worker removal (lines 533–596): when a pod's `deletionTimestamp` is set (pod is terminating), `Job::RemoveWorker` is submitted; the worker is removed from the active pool before the pod shuts down; in-flight requests to that worker complete normally (load counter ensures the `finally` decrement fires)
- Reconnection with exponential backoff (lines 308–393): if the K8s API connection drops, the watcher retries: 1s → 2s → 4s → … → 300s maximum; the router continues serving existing workers during reconnection; the tracked-pods `HashSet` ensures no duplicate additions when the watcher reconnects and replays events

### 07 — Observability: Prometheus and Alerting (`07_observability.md`)
- The router emits 40+ Prometheus metrics at `--prometheus-port 29000`; mount a `ServiceMonitor` (Prometheus Operator) or add a scrape job to `prometheus.yml` targeting the metrics port
- Service discovery specific metrics to watch:
  - `smg_discovery_registrations_total{source="kubernetes", result="success|failed|duplicate"}` — tracks how many pods have been registered; `duplicate` means a pod appeared twice (K8s can replay events on reconnect)
  - `smg_discovery_workers_discovered{source="kubernetes"}` — gauge of currently known workers; alert if this drops to 0
  - `smg_discovery_sync_duration_seconds` — time for a full reconciliation cycle
- Routing and latency:
  - `smg_http_requests_total` — request rate by path and status
  - `smg_http_request_duration_seconds` — latency histogram; `histogram_quantile(0.99, ...)` for P99
  - `smg_worker_requests_active` — per-worker in-flight count; mirrors `Worker.load` from Layer 15
- Worker health and circuit breaker:
  - `smg_worker_pool_size` — count of healthy workers; `sum(smg_worker_pool_size) == 0` → page immediately
  - `smg_worker_cb_state` — 0=Closed (normal), 1=Open (failing), 2=HalfOpen (recovering)
  - `smg_worker_health_checks_total{result="failure"}` — rising rate indicates persistent worker issues
- Essential alerting rules:
  ```yaml
  - alert: NoHealthyWorkers
    expr: sum(smg_worker_pool_size) == 0
    for: 1m
    severity: critical
  - alert: DiscoveryZeroWorkers
    expr: smg_discovery_workers_discovered{source="kubernetes"} == 0
    for: 2m
    severity: warning   # triggers before NoHealthyWorkers if discovery fails silently
  - alert: HighErrorRate
    expr: >
      sum(rate(smg_http_responses_total{status=~"5.."}[5m]))
      / sum(rate(smg_http_responses_total[5m])) > 0.05
    for: 5m
    severity: critical
  - alert: CircuitBreakerOpen
    expr: count(smg_worker_cb_state == 1) > 0
    for: 2m
    severity: warning
  ```
- Worker pod metrics: worker pods expose their own `/metrics` endpoint (vLLM / SGLang engine metrics); scrape separately; key: `vllm:cache_config_info` for KV cache utilisation, `vllm:gpu_cache_usage_perc` for memory pressure

### 08 — High Availability: Multiple Router Replicas and the Cache-Tree Problem (`08_high_availability.md`)
- Why a single router replica is a single point of failure: if the router pod crashes, all traffic fails until K8s restarts it (~30s); production deployments should run 2–3 replicas
- What changes with multiple replicas: each replica independently runs the service discovery loop, builds its own worker list, and maintains its own radix tree; trees are **not synchronised** across replicas (confirmed by SGLang GitHub issue #12700, filed Nov 2025)
- The cache-tree consistency problem: with 3 router replicas behind a ClusterIP Service, a request from user A may hit Router-1 on turn 1 and Router-2 on turn 2; Router-2 has no record of what Router-1 routed for turn 1, so the best-match worker may differ; cache hit rate drops 10–20% compared to a single router
- Mitigation option 1 — accept the loss (recommended): run 2–3 replicas, tolerate 10–20% cache degradation; HA benefit outweighs cache efficiency loss for most workloads; the simplest production choice
- Mitigation option 2 — session affinity at the load balancer: configure the K8s Service or Ingress to hash `x-user-id` (or client IP) to a consistent replica; the same user always hits the same router replica; preserves cache locality at the cost of uneven load if some users are much heavier
- Mitigation option 3 — single router with restart tolerance: `replicas: 1` with a fast restart (small image, no model weights); router is unavailable for the K8s restart window (~30s); acceptable for internal workloads with retry logic; not appropriate for user-facing services
- Horizontal Pod Autoscaler (HPA): scale worker pods on custom metrics (e.g. `smg_worker_requests_active > 20`); use KEDA for GPU-specific metrics; HPA adds new pods which the router discovers automatically via the watcher — no configuration change needed
- What is explicitly not in Layer 16: router-to-router radix tree synchronisation (open problem), global KV cache index (llm-d territory), cross-region routing (SkyWalker)

---

## Supporting Files

- `summary.md` — narrative walkthrough covering all eight sections with YAML examples and source references
- `sglang_reference.md` — maps Layer 16 K8s concepts to SGLang source: `ServiceDiscoveryConfig` → `src/service_discovery.rs:33`; `PodInfo::is_healthy()` → `src/service_discovery.rs:196`; `handle_pod_event` → `src/service_discovery.rs:399`; `handle_pod_deletion` → `src/service_discovery.rs:533`; watcher reconnection loop → `src/service_discovery.rs:311`
- `k8s_manifests/` — reference YAML files: `00_namespace.yaml`, `01_rbac.yaml`, `02_secret.yaml`, `03_pvc.yaml`, `04_worker_deployment.yaml`, `05_router_deployment.yaml`, `06_services.yaml`, `07_prometheus_rules.yaml`

---

## Key Code Anchors

| Concept | Location |
|---|---|
| `ServiceDiscoveryConfig` struct | `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs:33` |
| `PodInfo::from_pod()` | `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs:116` |
| `PodInfo::is_healthy()` | `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs:196` |
| `PodInfo::worker_url()` | `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs:200` |
| `start_service_discovery()` | `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs:206` |
| K8s client init (in-cluster) | `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs:223` |
| Watcher stream start | `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs:313` |
| Label selector filter | `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs:318` |
| Deletion timestamp check | `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs:349` |
| `handle_pod_event()` — worker addition | `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs:399` |
| `handle_pod_deletion()` — worker removal | `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs:533` |
| Exponential backoff reconnect | `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs:308` |
| vLLM GPU Deployment reference | `REPOS/vllm/docs/deployment/k8s.md` |
| vLLM Helm chart | `REPOS/vllm/examples/online_serving/chart-helm/` |
