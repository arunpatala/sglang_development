# Installing the NVIDIA GPU Operator

**Source:** https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/25.9.2/getting-started.html
**Author:** NVIDIA
**Version:** v25.9.2 (2026)
**Level:** L2–L3 — Cluster prerequisite
**Why here:** Prerequisite for Layer 16. The GPU Operator enables `nvidia.com/gpu` as a schedulable Kubernetes resource. Without it, `nvidia.com/gpu: "1"` resource requests in worker pod specs fail. Layer 16's `03_prerequisites_rbac.md` links to this.

---

## Prerequisites

- `kubectl` and `helm` CLIs available on a client machine.

Install Helm CLI:

```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 \
    && chmod 700 get_helm.sh \
    && ./get_helm.sh
```

All GPU worker nodes must run the same OS version (for the NVIDIA GPU Driver container). Nodes must use CRI-O or containerd as the container engine.

If using Pod Security Admission (PSA), label the namespace:

```bash
kubectl create ns gpu-operator
kubectl label --overwrite ns gpu-operator pod-security.kubernetes.io/enforce=privileged
```

---

## Node Feature Discovery (NFD)

NFD is a dependency. By default the GPU Operator deploys NFD automatically. Check if it's already running:

```bash
kubectl get nodes -o json | jq '.items[].metadata.labels | keys | any(startswith("feature.node.kubernetes.io"))'
```

Output `true` means NFD is already running — disable it when installing the Operator with `--set nfd.enabled=false`.

NFD labels GPU nodes with `feature.node.kubernetes.io/pci-10de.present=true` (NVIDIA PCI device). This is how you know a node has a GPU before the Operator runs.

---

## Installation

Add the NVIDIA Helm repository:

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
    && helm repo update
```

Install with default configuration:

```bash
helm install --wait --generate-name \
    -n gpu-operator --create-namespace \
    nvidia/gpu-operator \
    --version=v25.9.2
```

Or with custom options:

```bash
helm install --wait --generate-name \
    -n gpu-operator --create-namespace \
    nvidia/gpu-operator \
    --version=v25.9.2 \
    --set <option-name>=<option-value>
```

---

## Verification

After installation, verify GPU resources are available on nodes:

```bash
kubectl describe nodes | grep -A5 "Allocatable"
# Look for: nvidia.com/gpu: 8  (or however many GPUs the node has)
```

Verify all GPU Operator pods are running:

```bash
kubectl get pods -n gpu-operator
```

You should see pods for: `gpu-operator`, `nvidia-driver-daemonset`, `nvidia-device-plugin-daemonset`, `dcgm-exporter`, `node-feature-discovery`.

---

## GPU Taint/Toleration Pattern

GPU nodes are typically tainted to prevent CPU workloads from landing on them:

```bash
kubectl taint nodes <gpu-node-name> nvidia.com/gpu=present:NoSchedule
```

Worker pods must declare the corresponding toleration:

```yaml
spec:
  tolerations:
    - key: "nvidia.com/gpu"
      operator: "Exists"
      effect: "NoSchedule"
```

This is the pattern used in Layer 16's `04_worker_deployment.md` to ensure worker pods can schedule on GPU nodes.

---

## What the GPU Operator installs

| Component | Purpose |
|---|---|
| NVIDIA GPU Driver container | Installs/manages NVIDIA drivers on nodes |
| NVIDIA Device Plugin | Registers `nvidia.com/gpu` as a K8s schedulable resource |
| DCGM Exporter | Exports GPU metrics (utilization, memory, temperature) to Prometheus |
| Node Feature Discovery | Labels nodes with hardware capabilities |
| GPU Feature Discovery | Labels nodes with GPU-specific capabilities (compute capability, driver version) |
| NVIDIA Container Toolkit | Enables GPU access in containers via `--gpus` flag or K8s resource request |

Note: DCGM Exporter is deployed by the GPU Operator automatically when monitoring is enabled. See `L3/05_nvidia_dcgm_exporter.md` for details on scraping GPU metrics.

---

## Common Chart Customization Options

| Parameter | Description | Default |
|---|---|---|
| `driver.enabled` | Deploy NVIDIA GPU Driver container | `true` |
| `devicePlugin.enabled` | Deploy NVIDIA K8s Device Plugin | `true` |
| `dcgmExporter.enabled` | Deploy DCGM Exporter for GPU metrics | `true` |
| `nfd.enabled` | Deploy Node Feature Discovery | `true` |
| `toolkit.enabled` | Deploy NVIDIA Container Toolkit | `true` |
| `operator.defaultRuntime` | Container runtime (`docker`, `crio`, `containerd`) | `containerd` |

To view all options: `helm show values nvidia/gpu-operator`
