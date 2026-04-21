# 03 — Prerequisites and RBAC

## What This Section Covers

Before you can deploy anything, two things need to be set up:

1. **GPU support** in Kubernetes — so that pods can request and use GPUs
2. **RBAC** (Role-Based Access Control) — so that the router pod has permission to call the Kubernetes API and watch for worker pods

Neither of these is complicated, but both are required. Skipping either one results in either GPU pods that can't schedule, or a router that starts with zero workers and never discovers any.

---

## Part 1: GPU Support

### Why Kubernetes Needs to Know About GPUs

By default, Kubernetes schedules pods based on CPU and memory. GPUs are a special resource. Without a GPU plugin, Kubernetes doesn't know the cluster has GPUs, and pods requesting `nvidia.com/gpu: 1` will sit in `Pending` forever with the error "Insufficient nvidia.com/gpu".

### Installing the NVIDIA GPU Operator

The GPU Operator is the simplest way to enable NVIDIA GPU support. It installs as a single Kubernetes operator that manages:
- The NVIDIA device plugin (exposes GPUs as K8s resources)
- CUDA drivers on GPU nodes
- Monitoring (DCGM exporter for GPU metrics)

```bash
# Install via Helm (the standard approach)
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace
```

After a few minutes, you should see GPU resources available:

```bash
kubectl describe nodes | grep -A 5 "Allocatable:"

# Expected output on a GPU node:
Allocatable:
  cpu:                 32
  memory:              251Gi
  nvidia.com/gpu:      8   ← this means 8 GPUs are available on this node
  pods:                110
```

If `nvidia.com/gpu` does not appear, the operator is not running or the node is not a GPU node.

### Verifying GPU Nodes Are Labelled

GPU nodes typically have labels that let you target them:

```bash
kubectl get nodes --show-labels | grep gpu

# Or check for the NVIDIA label:
kubectl get nodes -l nvidia.com/gpu.present=true
```

If your cloud provider (GKE, EKS, AKS) manages GPU nodes, they usually label them automatically. You can add your own labels for more targeted scheduling:

```bash
kubectl label node my-gpu-node-1 gpu-type=nvidia-a100
```

---

## Part 2: RBAC — What It Is and Why You Need It

### The Problem

The router needs to call the Kubernetes API to watch for worker pods. The Kubernetes API is protected — not any process can call it. Pods need **explicit permission** to make API calls.

RBAC (Role-Based Access Control) is Kubernetes's permission system. It answers the question: "Which actions can this service account perform on which resources?"

### The Three RBAC Objects

You need three RBAC objects for the router:

```
ServiceAccount   →   "this is who the router is"
Role             →   "these are the permissions"
RoleBinding      →   "connect the ServiceAccount to the Role"
```

Think of it like:
- **ServiceAccount** = an ID card for the pod
- **Role** = a list of what someone with that ID can do
- **RoleBinding** = give this ID card these permissions

### Object 1: ServiceAccount

A ServiceAccount is an identity for a pod. Every pod runs under a ServiceAccount. If you don't specify one, pods run under the `default` ServiceAccount — which has no permissions to call the Kubernetes API.

```yaml
# 01_rbac.yaml (part 1)
apiVersion: v1
kind: ServiceAccount
metadata:
  name: sglang-router        # the router pod will reference this name
  namespace: production
```

```bash
kubectl apply -f 01_rbac.yaml
```

### Object 2: Role

A Role defines what actions are allowed on what resources, within a specific namespace.

The router needs exactly three verbs on the `pods` resource:
- `get` — fetch a specific pod by name
- `list` — list all pods matching a selector (needed at startup)
- `watch` — subscribe to a stream of pod events (needed for live updates)

```yaml
# 01_rbac.yaml (part 2)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: sglang-router
  namespace: production
rules:
- apiGroups: [""]            # "" means core API group (pods are in the core group)
  resources: ["pods"]        # only pods, not deployments/services/etc
  verbs: ["get", "list", "watch"]
```

**What this explicitly does NOT allow:**
- `create` — cannot create new pods
- `delete` — cannot delete pods
- `update` or `patch` — cannot modify pods
- Any other resource (deployments, services, secrets, configmaps)

This is the minimum required. The router can see pods but cannot touch anything else.

> **Why not ClusterRole?** A `ClusterRole` would let the router watch pods in *all* namespaces — including `kube-system`, `monitoring`, and any other team's namespace. A `Role` scoped to `production` means the router can only see pods in its own namespace. Always use the narrowest scope possible.

### Object 3: RoleBinding

The RoleBinding connects the ServiceAccount to the Role:

```yaml
# 01_rbac.yaml (part 3)
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: sglang-router
  namespace: production
subjects:
- kind: ServiceAccount
  name: sglang-router          # must match ServiceAccount name
  namespace: production        # must match ServiceAccount namespace
roleRef:
  kind: Role
  name: sglang-router          # must match Role name
  apiGroup: rbac.authorization.k8s.io
```

### Complete RBAC File

Putting all three together in one file (easier to apply in one command):

```yaml
# k8s_manifests/01_rbac.yaml
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: sglang-router
  namespace: production

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: sglang-router
  namespace: production
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: sglang-router
  namespace: production
subjects:
- kind: ServiceAccount
  name: sglang-router
  namespace: production
roleRef:
  kind: Role
  name: sglang-router
  apiGroup: rbac.authorization.k8s.io
```

```bash
kubectl apply -f k8s_manifests/01_rbac.yaml
```

### How the Router Uses the ServiceAccount

When you specify `serviceAccountName: sglang-router` in the router pod spec, Kubernetes automatically:
1. Mounts the service account token at `/var/run/secrets/kubernetes.io/serviceaccount/token`
2. Mounts the CA certificate at `/var/run/secrets/kubernetes.io/serviceaccount/ca.crt`
3. Sets the environment variable `KUBERNETES_SERVICE_HOST`

The router's Rust code (in `service_discovery.rs`) calls `Client::try_default()` which reads these mounted files automatically. You don't need to pass any credentials explicitly.

```rust
// From service_discovery.rs line 223
let client = Client::try_default().await?;
// ↑ Reads token from /var/run/secrets/kubernetes.io/serviceaccount/token
// ↑ Reads CA cert from /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
// ↑ Reads API server address from KUBERNETES_SERVICE_HOST env var
// No kubeconfig, no extra configuration needed inside the cluster
```

---

## Part 3: The HuggingFace Token Secret

Worker pods need to download model weights from HuggingFace if the PVC is empty (first startup). Gated models (like Llama) require authentication.

**Never put tokens in a ConfigMap or container image.** Use a Kubernetes Secret:

```bash
# Create the secret from command line (token never goes in a file)
kubectl create secret generic hf-token-secret \
  --namespace production \
  --from-literal=token=hf_xxxxxxxxxxxxxxxxxxxx
```

Or as YAML (for GitOps — note: secrets in YAML are only base64-encoded, not encrypted):

```yaml
# k8s_manifests/02_secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: hf-token-secret
  namespace: production
type: Opaque
stringData:
  token: "hf_xxxxxxxxxxxxxxxxxxxx"   # will be base64-encoded by K8s
```

Worker pods reference this secret as an environment variable:

```yaml
env:
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: hf-token-secret   # the secret name
      key: token               # the key within the secret
```

HuggingFace's libraries check `HF_TOKEN` automatically when downloading models.

---

## Part 4: The Namespace

Create the namespace before deploying anything:

```yaml
# k8s_manifests/00_namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    environment: production
    team: ml-platform
```

```bash
kubectl apply -f k8s_manifests/00_namespace.yaml
```

All subsequent resources use `namespace: production`.

---

## Verification Checklist

After applying the RBAC objects, verify everything is in place:

```bash
# 1. Namespace exists
kubectl get namespace production

# 2. ServiceAccount exists
kubectl get serviceaccount sglang-router -n production

# 3. Role exists with correct verbs
kubectl describe role sglang-router -n production
# Should show: pods, get list watch

# 4. RoleBinding connects them
kubectl describe rolebinding sglang-router -n production
# Should show: Subject: sglang-router (ServiceAccount), Role: sglang-router

# 5. GPU nodes are ready
kubectl get nodes -l nvidia.com/gpu.present=true
kubectl describe nodes | grep "nvidia.com/gpu:" | grep -v "0"

# 6. Secret exists
kubectl get secret hf-token-secret -n production
```

### Testing RBAC Manually

You can test whether the ServiceAccount has the right permissions before deploying the router:

```bash
# Can the sglang-router ServiceAccount list pods?
kubectl auth can-i list pods \
  --as=system:serviceaccount:production:sglang-router \
  -n production
# Expected: yes

# Can it delete pods? (should be no)
kubectl auth can-i delete pods \
  --as=system:serviceaccount:production:sglang-router \
  -n production
# Expected: no

# Can it watch pods in a different namespace? (should be no)
kubectl auth can-i watch pods \
  --as=system:serviceaccount:production:sglang-router \
  -n kube-system
# Expected: no
```

---

## Common RBAC Mistakes

**Mistake 1: Forgetting to attach the ServiceAccount to the pod**

```yaml
# WRONG: pod runs as "default" ServiceAccount, has no permissions
spec:
  containers:
  - name: router
    image: ...

# CORRECT: explicitly attach
spec:
  serviceAccountName: sglang-router
  containers:
  - name: router
    image: ...
```

**Mistake 2: ServiceAccount in wrong namespace**

The ServiceAccount, Role, and RoleBinding must all be in the same namespace as the router pod. If the router pod is in `production` but the ServiceAccount is in `default`, K8s uses the `default` account (which has no permissions).

**Mistake 3: Using ClusterRole when a Role is sufficient**

```yaml
# AVOID: gives router visibility into all pods cluster-wide
kind: ClusterRole
...
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]

# PREFER: scoped to one namespace
kind: Role
metadata:
  namespace: production
```

**Mistake 4: Missing `list` verb**

The `watch` verb alone is not enough. At startup, the router `list`s all existing pods to get the initial state, then switches to `watch` for live updates. If `list` is missing, the router starts with zero workers and the initial bootstrap fails.

```yaml
# WRONG: missing list
verbs: ["get", "watch"]

# CORRECT:
verbs: ["get", "list", "watch"]
```
