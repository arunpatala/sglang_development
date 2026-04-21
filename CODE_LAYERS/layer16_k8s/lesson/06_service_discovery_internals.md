# 06 — How Service Discovery Works Internally

## What This Section Covers

When you pass `--service-discovery` to the router, a background task starts inside the router process. That task connects to the Kubernetes API, watches for pod events, and updates the router's worker pool in real time. This section walks through the actual production Rust code that does this.

**Source file:** `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs`

We'll read the code from top to bottom, explaining each piece simply.

---

## The Config Struct (Lines 33–48)

```rust
// service_discovery.rs line 33
pub struct ServiceDiscoveryConfig {
    pub enabled: bool,
    pub selector: HashMap<String, String>,   // e.g. {"app": "sglang-worker", "component": "inference"}
    pub check_interval: Duration,            // how long to wait before restarting after watcher exits (default: 60s)
    pub port: u16,                           // which port to connect to on worker pods (default: 8000)
    pub namespace: Option<String>,           // limit watch to this namespace (None = all namespaces)
    ...
}
```

This is filled in from the CLI flags when the router starts:

```
--service-discovery                       → enabled: true
--selector app=sglang-worker              → selector: {"app": "sglang-worker"}
--selector component=inference            → selector: {"component": "inference"} (merged with above)
--service-discovery-port 8000             → port: 8000
--service-discovery-namespace production  → namespace: Some("production")
```

---

## The PodInfo Struct (Lines 76–85)

```rust
// service_discovery.rs line 76
pub struct PodInfo {
    pub name: String,       // pod name, e.g. "sglang-worker-abc12"
    pub ip: String,         // pod IP, e.g. "10.244.2.14"
    pub status: String,     // Kubernetes phase: "Running", "Pending", "Succeeded", etc.
    pub is_ready: bool,     // whether the readiness probe is passing
    pub pod_type: Option<PodType>,  // Regular (normal mode)
    ...
}
```

This is a simplified view of a Kubernetes pod, containing only what the router needs. Everything else in the K8s pod (spec, resource usage, node name, etc.) is ignored.

---

## Step 1: Label Matching — `matches_selector` (Lines 88–96)

```rust
// service_discovery.rs line 88
fn matches_selector(pod: &Pod, selector: &HashMap<String, String>) -> bool {
    if selector.is_empty() {
        return false;   // safety: empty selector matches nothing, not everything
    }

    pod.metadata
        .labels
        .as_ref()
        .is_some_and(|labels| {
            selector.iter().all(|(k, v)| labels.get(k) == Some(v))
            //                   ↑ ALL selector entries must match (AND logic)
        })
}
```

**What this does:** For a pod to match, it must have **every** key-value pair in the selector as a label. If the selector is `{"app": "sglang-worker", "component": "inference"}`, a pod must have **both** labels. Missing either one → `false`.

**The safety check at line 89:** If the selector is empty (someone forgot to set it), the function returns `false` — matching nothing, rather than everything. Without this guard, an empty selector would match every pod in the cluster and the router would try to send inference traffic to the control plane, monitoring pods, etc.

---

## Step 2: Health Check — `is_healthy` (Lines 196–198)

```rust
// service_discovery.rs line 196
pub fn is_healthy(&self) -> bool {
    self.is_ready && self.status == "Running"
}
```

Two conditions, both required:

1. **`is_ready`**: The readiness probe is passing. Set from:

```rust
// service_discovery.rs line 121
let is_ready = if let Some(conditions) = &status.conditions {
    conditions
        .iter()
        .any(|condition| condition.type_ == "Ready" && condition.status == "True")
        //   ↑ looks for the "Ready" condition with value "True"
} else {
    false   // no conditions = not ready
};
```

This is the same `Ready` condition that `kubectl get pods` shows in the `READY` column. It becomes `True` only after the readiness probe passes. During model loading, it stays `False`.

2. **`status == "Running"`**: The pod phase from Kubernetes. Possible phases: `Pending`, `Running`, `Succeeded`, `Failed`, `Unknown`. Only `Running` pods get traffic.

**Practical meaning:** A pod must be **both** running (not initializing or terminating) **and** have its readiness probe passing. This double check means a pod that is `Running` but not yet `Ready` (still loading the model) is never added to the worker pool.

---

## Step 3: Worker URL Construction (Lines 200–203)

```rust
// service_discovery.rs line 200
pub fn worker_url(&self, port: u16) -> String {
    format!("http://{}:{}", self.ip, port)
    // e.g. "http://10.244.2.14:8000"
}
```

Simple. The router uses this URL to send inference requests to the worker.

---

## Step 4: The Main Discovery Function (Lines 206–397)

```rust
// service_discovery.rs line 206
pub async fn start_service_discovery(
    config: ServiceDiscoveryConfig,
    app_context: Arc<AppContext>,
    ...
) -> Result<task::JoinHandle<()>, kube::Error> {
```

This function is called once at router startup. It returns a `JoinHandle` — a handle to a background async task. The task runs forever (or until the router process exits).

### 4a. Connect to Kubernetes (Lines 221–223)

```rust
// service_discovery.rs line 221
let _ = rustls::crypto::ring::default_provider().install_default();  // set up TLS

let client = Client::try_default().await?;
//           ↑ reads credentials from the pod's mounted ServiceAccount token
//           file: /var/run/secrets/kubernetes.io/serviceaccount/token
//           CA cert: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
//           API server: KUBERNETES_SERVICE_HOST env var
```

No configuration needed inside the pod. Kubernetes automatically injects the service account token and the API server address as environment variables. `Client::try_default()` finds and uses them.

If the ServiceAccount doesn't have permission to list pods (Section 03 RBAC was skipped), this call succeeds but the first API call will return `403 Forbidden`.

### 4b. Scope the Watch to a Namespace (Lines 276–280)

```rust
// service_discovery.rs line 276
let pods: Api<Pod> = if let Some(namespace) = &config.namespace {
    Api::namespaced(client, namespace)  // watch only this namespace
} else {
    Api::all(client)                    // watch all namespaces (needs ClusterRole)
};
```

`Api::namespaced(client, "production")` means the watcher only gets events from pods in the `production` namespace. This is why `--service-discovery-namespace production` is important — without it, the router would need `ClusterRole` permissions and would see every pod in the cluster.

### 4c. The Retry Loop (Lines 308–393)

```rust
// service_discovery.rs line 308
let mut retry_delay = Duration::from_secs(1);
const MAX_RETRY_DELAY: Duration = Duration::from_secs(300);  // 5 minutes max

loop {
    // ... set up watcher ...
    match filtered_stream.try_for_each(...).await {
        Ok(_) => {
            retry_delay = Duration::from_secs(1);  // success: reset backoff
        }
        Err(err) => {
            error!("Error in Kubernetes watcher: {}", err);
            time::sleep(retry_delay).await;
            retry_delay = std::cmp::min(retry_delay * 2, MAX_RETRY_DELAY);
            //                         ↑ double on each failure, cap at 5 minutes
        }
    }
}
```

The watcher is wrapped in an **exponential backoff retry loop**. If the K8s API server is temporarily unreachable (network blip, API server restart), the watcher fails. The router sleeps 1s, then 2s, then 4s, ... up to 5 minutes, then retries.

This means the router is resilient to:
- Kubernetes API server restarts (common during cluster upgrades)
- Network partitions
- K8s API server rate limiting (429 errors)

Worker pods already in the pool continue to receive traffic during an API disconnect — the router only loses the ability to **discover new pods**. Existing workers are unaffected.

### 4d. The Watcher — Subscribing to Pod Events (Lines 312–313)

```rust
// service_discovery.rs line 312
let watcher_config = Config::default();
let watcher_stream = watcher(pods.clone(), watcher_config).applied_objects();
//                   ↑ from the kube-runtime crate
//                   ↑ returns a stream of pod change events
```

`watcher(pods, config)` creates a streaming subscription to pod events in the `pods` API (scoped to the namespace). `.applied_objects()` transforms the raw watch events into pod objects.

Internally, the kube-runtime watcher does:
1. `GET /api/v1/namespaces/production/pods` — lists all existing pods and their **resource version**
2. `GET /api/v1/namespaces/production/pods?watch=true&resourceVersion=12345` — subscribes to all changes since that resource version

Every time a pod is created, modified (e.g. readiness probe changes from False to True), or deleted, the watch stream delivers a new event to the router.

### 4e. Filtering the Stream (Lines 318–333)

```rust
// service_discovery.rs line 318
let filtered_stream = watcher_stream.filter_map(move |obj_res| {
    let config_inner = Arc::clone(&config_clone);
    async move {
        match obj_res {
            Ok(pod) => {
                if PodInfo::should_include(&pod, &config_inner) {
                    Some(Ok(pod))   // ← included: pass through the stream
                } else {
                    None            // ← excluded: drop this event
                }
            }
            Err(e) => Some(Err(e)), // ← errors always pass through
        }
    }
});
```

The watcher sees **all** pod events in the namespace — including the router pod itself, any monitoring sidecars, etc. The `filter_map` drops events for pods that don't match the label selector. Only pods matching `app=sglang-worker,component=inference` (or whatever the selector is) reach the event handler.

### 4f. Handling Each Event (Lines 339–370)

```rust
// service_discovery.rs line 339
.try_for_each(move |pod| {
    async move {
        let pod_info = PodInfo::from_pod(&pod, Some(&config_inner));

        if let Some(pod_info) = pod_info {
            if pod.metadata.deletion_timestamp.is_some() {
                // pod is being deleted → remove from worker pool
                handle_pod_deletion(&pod_info, ...).await;
            } else {
                // pod was created or modified → maybe add to worker pool
                handle_pod_event(&pod_info, ...).await;
            }
        }
        Ok(())
    }
})
```

Two cases:
1. `deletion_timestamp.is_some()` — Kubernetes sets this field on a pod as soon as you run `kubectl delete pod` or when a Deployment terminates it. The pod is not gone yet (it's still running its shutdown hooks) but the timestamp appearing is the signal to remove it from the router's pool.
2. Otherwise — the pod was created, became Ready, became NotReady, or had some other change. Call `handle_pod_event` to decide whether to add it.

---

## Step 5: Adding a Worker (Lines 399–531)

```rust
// service_discovery.rs line 399
async fn handle_pod_event(pod_info: &PodInfo, tracked_pods: ..., app_context: ..., port: u16, ...) {
    let worker_url = pod_info.worker_url(port);  // e.g. "http://10.244.2.14:8000"

    if pod_info.is_healthy() {  // Running AND Ready
```

First gate: only add healthy pods (line 408).

```rust
        let (should_add, tracked_count) = {
            let mut tracker = tracked_pods.lock().unwrap();
            if tracker.contains(pod_info) {
                (false, tracker.len())   // already tracked, skip
            } else {
                tracker.insert(pod_info.clone());
                (true, tracker.len())    // new pod, add it
            }
        };
```

Second gate (lines 410–425): deduplication. The `tracked_pods` HashSet remembers which pods have already been added to the router. The watcher may fire multiple events for the same pod (label changes, annotation changes, etc.). The `contains` check ensures we only call `AddWorker` once per pod.

```rust
        if should_add {
            let config = WorkerConfigRequest {
                url: worker_url.clone(),    // "http://10.244.2.14:8000"
                model_id: None,
                api_key: app_context.router_config.api_key.clone(),
                health_check_timeout_secs: ...,
                health_check_interval_secs: ...,
                ...
            };

            let job = Job::AddWorker { config: Box::new(config) };

            job_queue.submit(job).await   // sends to the router's control plane
```

When a new pod passes both gates (healthy + not already tracked), the code creates a `WorkerConfigRequest` and submits a `Job::AddWorker` to the router's job queue (lines 479–484). The router's control plane processes this job asynchronously — it connects to the new worker, runs health checks, and adds it to the routing policy's worker list.

---

## Step 6: Removing a Worker (Lines 533–597)

```rust
// service_discovery.rs line 533
async fn handle_pod_deletion(pod_info: &PodInfo, tracked_pods: ..., app_context: ..., port: u16) {
    let worker_url = pod_info.worker_url(port);

    let (was_tracked, remaining_count) = {
        let mut tracked = tracked_pods.lock().unwrap();
        let removed = tracked.remove(pod_info);  // remove from dedup set
        (removed, tracked.len())
    };

    if was_tracked {
        let job = Job::RemoveWorker { url: worker_url.clone() };
        job_queue.submit(job).await
        //        ↑ tells the router to stop sending traffic to this URL
```

When a pod's `deletion_timestamp` appears:
1. Remove from `tracked_pods` (dedup set)
2. Submit `Job::RemoveWorker` with the pod's URL
3. The router's control plane removes the URL from the routing policy's active worker list

Future requests are not sent to the removed URL. In-flight requests that were already in progress may still complete (the pod is still running its graceful shutdown period).

---

## The Complete Flow, Annotated

Here's the complete sequence from pod start to routing, with line numbers:

```
1. Router starts. CLI parses --service-discovery, --selector, etc.
   → ServiceDiscoveryConfig populated.

2. start_service_discovery() called (line 206).
   → Client::try_default() reads ServiceAccount token (line 223).
   → Api::namespaced(client, "production") scopes to namespace (line 277).

3. Background task spawned (line 273).
   → tracked_pods = {} (empty HashSet, line 274).

4. Watcher starts (line 312-313).
   → First: LIST all pods in production namespace (K8s API).
   → Delivers each existing pod as an event to the stream.
   → Then: WATCH for changes (subscribes to event stream).

5. For each pod event:
   a. filter_map checks should_include (line 324).
      → pods without app=sglang-worker,component=inference are dropped.

   b. deletion_timestamp check (line 349).
      → pod terminating? → handle_pod_deletion.
      → pod created/modified? → handle_pod_event.

   c. handle_pod_event checks is_healthy (line 408).
      → is_ready=false? → skip (model still loading).
      → status != "Running"? → skip.

   d. Deduplication check (line 419).
      → already in tracked_pods? → skip (duplicate event).
      → new pod? → insert into tracked_pods, submit Job::AddWorker.

6. Router control plane receives Job::AddWorker.
   → Connects to http://10.244.2.14:8000.
   → Runs HTTP health check.
   → Adds to routing policy worker list.
   → Policy can now send traffic to this pod.

7. Pod deleted (kubectl delete pod or Deployment scales down).
   → deletion_timestamp appears on pod object.
   → Watcher fires event.
   → handle_pod_deletion: remove from tracked_pods, submit Job::RemoveWorker.
   → Router control plane removes URL from routing policy.
   → No new requests go to this pod.
```

---

## What Happens to In-Flight Requests When a Pod Is Removed?

The router submits `Job::RemoveWorker` to its job queue. The control plane processes this asynchronously. Between the deletion event and the control plane removing the worker, a small number of requests may still be routed to the terminating pod.

Kubernetes gives terminating pods a **grace period** (default 30 seconds, configurable with `terminationGracePeriodSeconds`). During this period:
- The pod is still running and accepting connections
- Kubernetes removes the pod from any Service endpoints (so K8s-routed traffic stops)
- The router removes the pod from its pool when it receives the deletion event

In practice, this means:
- Zero dropped requests in normal scale-down (router removes pod before it stops accepting)
- Possible brief errors if the pod is killed suddenly (OOM, node failure) before the router removes it

---

## Verifying Service Discovery Is Working

```bash
# Watch router logs for discovery events
kubectl logs -n production -l app=sglang-router -f | grep -E "Adding|Removing|watcher"

# Expected on startup:
# [INFO] Starting K8s service discovery | selector: 'app=sglang-worker,component=inference'
# [INFO] Adding pod: sglang-worker-abc12 | type: Some(Regular) | url: http://10.244.2.14:8000
# [INFO] Adding pod: sglang-worker-xyz78 | type: Some(Regular) | url: http://10.244.3.7:8000

# Scale up and watch discovery in real time:
kubectl scale deployment sglang-worker -n production --replicas=3
# Expected (within ~10s of new pod becoming Ready):
# [INFO] Adding pod: sglang-worker-new99 | type: Some(Regular) | url: http://10.244.1.5:8000

# Delete a pod and watch removal:
kubectl delete pod sglang-worker-abc12 -n production
# Expected:
# [INFO] Removing pod: sglang-worker-abc12 | type: Some(Regular) | url: http://10.244.2.14:8000

# Check metrics for discovery counts (Section 07):
kubectl port-forward svc/sglang-router -n production 29000:29000 &
curl http://localhost:29000/metrics | grep discovery
# discovery_workers_discovered{source="kubernetes"} 3
# discovery_registrations_total{source="kubernetes",status="success"} 3
# discovery_deregistrations_total{source="kubernetes",reason="pod_deleted"} 1
```

---

## How Pod Deletion Works (Step by Step)

### What Kubernetes Does

Kubernetes never kills a pod instantly. When you run `kubectl delete pod` (or a Deployment scales down), Kubernetes takes two steps:

```
Step 1: Set deletionTimestamp on the pod object
        → pod is still RUNNING and accepting requests
        → but it now appears "Terminating" in kubectl get pods

Step 2: Wait for terminationGracePeriodSeconds (default: 30s)
        → sends SIGTERM to the container process
        → after grace period, sends SIGKILL
        → pod is fully gone
```

The router acts on **Step 1** — the moment `deletionTimestamp` appears — not Step 2. This is why the router removes the pod from its pool *before* the pod actually stops serving. In-flight requests that were already sent to the pod can still complete during the grace period.

### What the Router Does (service_discovery.rs)

The watcher stream fires whenever the pod object changes. The event handler checks:

```rust
// service_discovery.rs line 349
if pod.metadata.deletion_timestamp.is_some() {
    // Pod is terminating — remove immediately, don't wait for it to die
    handle_pod_deletion(&pod_info, ...).await;
}
```

`handle_pod_deletion` (line 533–597) does two things:

```rust
// 1. Remove from the deduplication set
tracked.remove(pod_info);

// 2. Submit Job::RemoveWorker to the control plane
let job = Job::RemoveWorker { url: worker_url.clone() };
job_queue.submit(job).await
```

After `Job::RemoveWorker` is processed, the routing policy no longer includes this URL. All future requests go to the remaining workers.

### Timeline for a Deleted Pod

```
t=0s   kubectl delete pod sglang-worker-abc12
         → K8s sets deletionTimestamp on pod object
         → watcher stream fires event

t=~0s  Router receives event
         → deletion_timestamp.is_some() = true
         → handle_pod_deletion() called
         → tracked_pods.remove(pod_info)
         → Job::RemoveWorker submitted
         → router stops sending NEW requests to this pod

t=~1s  Job processed by control plane
         → 10.244.2.14:8000 removed from routing policy's worker list

t=30s  K8s grace period expires
         → SIGKILL sent, pod is fully gone
         → Any in-flight requests that started before t=0s
           may still be completing (pod accepted them, was still running)
```

The critical insight: **the router removes the pod at t≈0, the pod dies at t=30**. This 30-second window is intentional — in-flight long-running LLM requests (which can take 10–60s) can complete normally. No new requests are sent, but existing ones finish.

---

## How the Radix Tree Is Updated After Deletion

The radix tree is **not explicitly cleaned** when a pod is deleted. There is no "delete all entries pointing to Worker A" operation. Instead, it self-heals lazily.

### What the Radix Tree Contains

The router's radix tree maps prompt prefixes to workers:

```
"You are a helpful assistant. User: What..." → Worker A (10.244.2.14:8000)
"You are a helpful assistant. User: How..."  → Worker A (10.244.2.14:8000)
"System: Respond in JSON. User: List..."     → Worker B (10.244.3.7:8000)
```

When Worker A is deleted, these entries still exist in the tree. They are now **stale** — they point to a URL that is no longer in the active worker pool.

### What Happens on the Next Request

The cache-aware routing policy runs a two-guard check for every request:

```
1. Prefix match: find the worker with the longest matching prefix
   → radix tree says: Worker A (10.244.2.14:8000)

2. Availability check: is Worker A still in the active pool?
   → Worker A was removed by Job::RemoveWorker
   → Worker A is NOT in the pool

3. Fallback: ignore the stale cache suggestion
   → pick the least-loaded worker from the remaining pool (Worker B or C)
   → route to that worker

4. Insert new entry: "this prefix is now on Worker B"
   → radix tree updated: "You are a helpful assistant..." → Worker B
```

### The Result: Lazy Self-Healing

```
Before deletion:
  Radix tree: prefix → Worker A (valid)
  Worker pool: [Worker A, Worker B, Worker C]

After deletion:
  Radix tree: prefix → Worker A (stale)
  Worker pool: [Worker B, Worker C]           ← Worker A removed

First request with that prefix:
  Radix tree lookup → Worker A → not in pool → fallback to Worker B
  Radix tree updated → prefix → Worker B

Second request with same prefix:
  Radix tree lookup → Worker B → in pool → cache hit on Worker B ✓
```

The tree repairs itself after **one cache miss per prefix**. This miss is unavoidable — the KV cache for that prefix was on Worker A's GPU, which is gone. The request must be recomputed on Worker B regardless. After that, the prefix is correctly mapped to Worker B.

### No Explicit Cleanup Needed

This design is intentional:
- **Simpler**: no need to track which prefixes are on which worker at deletion time
- **Correct**: stale entries cause at most one cache miss, never a routing error (the pool check prevents routing to a removed worker)
- **Fast**: deletion path is just "remove from set + submit job" — no tree traversal needed
