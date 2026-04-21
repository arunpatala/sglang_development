# 01 — The Problem: Static URLs Break in Kubernetes

## What We Had in Layer 15

In Layer 15, the router was told exactly which backends to talk to by listing their addresses in `config.yml`:

```yaml
# Layer 15 config.yml
router:
  workers:
    - url: "http://localhost:8114"
      name: "engine-a"
    - url: "http://localhost:8115"
      name: "engine-b"
```

This works perfectly on your laptop. You start two engines manually, start the router, and everything connects.

But "localhost" and fixed port numbers only work because you started both processes yourself and they stay running at those addresses forever.

---

## What Kubernetes Does Differently

In Kubernetes, you don't start processes manually. You tell Kubernetes *what* you want (a description called a **Deployment**), and Kubernetes starts the pods for you, on whatever machine has capacity.

This creates a fundamental problem: **you don't know the IP address ahead of time.**

### Pod IP Assignment

Every pod in Kubernetes gets an IP address from the cluster's internal network. You don't pick it — Kubernetes assigns it from a pool.

```
Pod starts  →  Kubernetes assigns IP: 10.244.2.14
Pod runs    →  Serving at 10.244.2.14:8000
Pod crashes →  Kubernetes restarts it
New pod     →  New IP assigned: 10.244.3.7   ← different!
```

The old IP (`10.244.2.14`) is gone. Any connection to it fails immediately. If you had hardcoded that IP in `config.yml`, your router is now pointing at nothing.

### A Concrete Example

Imagine you deploy a worker and note its IP:

```bash
$ kubectl get pods -o wide
NAME                     READY   STATUS    IP
sglang-worker-abc123     1/1     Running   10.244.2.14
```

You put `http://10.244.2.14:8000` in your router config. Everything works.

Then the worker pod crashes (GPU ran out of memory, kernel panic, node rebooted). Kubernetes automatically restarts it:

```bash
$ kubectl get pods -o wide
NAME                     READY   STATUS    IP
sglang-worker-xyz789     1/1     Running   10.244.3.7   ← new pod, new IP
```

Your router is still configured with `10.244.2.14`. Every request fails with "connection refused" until you manually update the config and restart the router.

**This is not an edge case.** Pod restarts happen constantly in production: rolling deployments, out-of-memory kills, node maintenance, hardware failures.

---

## The Scaling Problem

Static URLs also break when you scale.

In Layer 15, you have two workers. Traffic grows. You want four workers. In Kubernetes you run:

```bash
kubectl scale deployment sglang-worker --replicas=4
```

Kubernetes creates two more pods. They get IPs. They start serving requests. But your router still has only two entries in `config.yml`. The two new pods receive **zero traffic** — they are completely invisible to the router.

You now have four GPUs running but only two doing work. You're paying for four and getting two.

```
Layer 15 config.yml says: [engine-a, engine-b]
                                   ↓           ↓
                              Worker Pod A  Worker Pod B   ← gets traffic
                              Worker Pod C  Worker Pod D   ← idle, no traffic
```

To add the new workers, you'd have to:
1. Get the IPs of the new pods manually
2. Edit `config.yml`
3. Restart the router (dropping all in-flight requests)
4. Repeat every time you scale

This is operationally untenable.

---

## The Rolling Update Problem

When you push a new model version, Kubernetes does a **rolling update**: it replaces pods one at a time so there is always some capacity available.

```
Start: [v1-pod-a, v1-pod-b, v1-pod-c, v1-pod-d]

Step 1: kill v1-pod-a, start v2-pod-e  → [v2-pod-e, v1-pod-b, v1-pod-c, v1-pod-d]
Step 2: kill v1-pod-b, start v2-pod-f  → [v2-pod-e, v2-pod-f, v1-pod-c, v1-pod-d]
Step 3: ...
```

During this process, the live pod list changes every few seconds. Static config cannot keep up.

---

## The Solution: Let the Router Watch the Cluster

The fix is to stop hardcoding IP addresses and instead let the router **ask Kubernetes** which pods are currently running and healthy.

Kubernetes provides a `watch` API: any process with the right permissions can subscribe to a stream of pod events. The router gets notified immediately when:
- A new pod becomes Ready (add it to the worker pool)
- A pod starts terminating (remove it from the worker pool)
- A pod fails its health check (exclude it from routing)

The router no longer needs to know IP addresses ahead of time. It learns them live from the cluster.

```
Layer 16: Router asks Kubernetes → "which pods have label app=sglang-worker?"
Kubernetes answers: "10.244.2.14:8000, 10.244.3.7:8000, 10.244.1.9:8000"
New pod starts → Kubernetes tells router → router adds it
Pod crashes → Kubernetes tells router → router removes it
```

---

## What Changes in the Config

The entire `workers:` list is removed. In its place, a discovery selector:

```yaml
# Layer 15: static list
router:
  workers:
    - url: "http://localhost:8114"
    - url: "http://localhost:8115"
  policy: prefix_cache_aware

# Layer 16: dynamic discovery
router:
  service_discovery:
    enabled: true
    selector:
      app: sglang-worker
      component: inference
    namespace: production
    port: 8000
  policy: cache_aware
```

Or equivalently as CLI flags (what the production router actually uses):

```bash
# Layer 15
python -m sglang_router.launch_router \
  --worker-urls http://localhost:8114 http://localhost:8115 \
  --policy cache_aware

# Layer 16
python -m sglang_router.launch_router \
  --service-discovery \
  --selector app=sglang-worker component=inference \
  --service-discovery-namespace production \
  --service-discovery-port 8000 \
  --policy cache_aware
```

The routing policies (`cache_aware`, `round_robin`, `power_of_two`) are completely unchanged. The only difference is how the router learns which workers exist.

---

## The Label Selector Contract

The discovery mechanism works through **pod labels**. Labels are key=value pairs attached to Kubernetes objects. You put them on your worker pods:

```yaml
# Worker pod template labels
labels:
  app: sglang-worker          # what it is
  component: inference        # what role it plays
  model: llama-3-8b          # which model (useful for filtering)
```

And you tell the router which labels to match:

```bash
--selector app=sglang-worker component=inference
```

The router finds all pods where **all** specified labels match. A pod must have `app=sglang-worker` **AND** `component=inference` to be included. Pods with only one of the two labels are ignored.

This selector is the entire "configuration" for worker discovery. Add a pod with the right labels and it gets traffic. Remove the pod (or change its labels) and it stops getting traffic.

---

## Summary

| Problem | Layer 15 | Layer 16 |
|---|---|---|
| Pod restarts (new IP) | Router points at dead address | Router is notified, updates automatically |
| `kubectl scale` adds pods | New pods get no traffic | New pods discovered, added within seconds |
| Rolling updates | Config stale during rollout | Router tracks live pod state continuously |
| Worker IPs | Hardcoded in `config.yml` | Learned from Kubernetes API at runtime |

The router's routing logic (which policy runs, how the radix trie works, the two-guard algorithm) does not change at all. Only *how the router learns about workers* changes.
