# Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

**Source:** https://arxiv.org/abs/2407.00079
**Paper PDF:** https://arxiv.org/pdf/2407.00079
**GitHub:** https://github.com/kvcache-ai/Mooncake
**Docs:** https://kvcache-ai.github.io/Mooncake/
**Transfer Engine Design:** https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html
**Venue:** USENIX FAST 2025 (File and Storage Technologies)
**Authors:** Moonshot AI (Kimi team) + MadSys @ Tsinghua University
**Level:** L3 — Production KV transfer engine; RDMA design, multi-NIC, topology-aware path selection
**Why here:** Mooncake is both a research paper and a production system powering Kimi (Moonshot AI's flagship LLM service). It is the **primary KV transfer engine used by SGLang for PD disaggregation** and is also integrated into vLLM v1, TensorRT-LLM, LMDeploy, and others. Understanding Mooncake's Transfer Engine explains the RDMA mechanics behind `--disaggregation-ib-device`, why NVLink transport is recommended for NVL72, and how multi-NIC pooling achieves the throughput required for large-scale PD disaggregation.

**BibTeX:**
```bibtex
@inproceedings{moonshot2025mooncake,
  title     = {Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving},
  author    = {{Moonshot AI} and {MadSys Group, Tsinghua University}},
  booktitle = {23rd USENIX Conference on File and Storage Technologies (FAST 25)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2407.00079}
}
```

---

## What Mooncake Is

Mooncake is the serving platform for **Kimi** — not a research prototype but a production system that has handled exponential traffic growth. Its architecture has two levels of disaggregation:

1. **Phase disaggregation**: separate prefill and decode clusters (same as DistServe/Splitwise)
2. **Cache disaggregation**: a distributed KVCache pool built from underutilised CPU DRAM, SSD, and RDMA resources across the GPU cluster — sitting between the two clusters

This second level is what makes Mooncake distinct: it enables **near-GPU prefix caching** that survives node failures and can be shared across prefill and decode instances.

---

## The Transfer Engine: Core Component

The **Mooncake Transfer Engine (TE)** is the RDMA data-movement library that enables efficient KV cache transfer. It is open-sourced separately from the Mooncake paper and is the component integrated into SGLang, vLLM, TensorRT-LLM, and NIXL.

### Two Core Abstractions

**Segment**: a contiguous address space that can be remotely read and written. Can be GPU VRAM (RDMA registered), CPU DRAM (pinned), or a file on NVMe.

**Transfer Agent**: the runtime that manages buffers, network cards, and transfer requests. One agent runs per inference server process.

### Supported Transfer Protocols

| Protocol | Use case |
|---|---|
| RDMA (InfiniBand / RoCEv2) | Inter-node GPU VRAM → GPU VRAM (highest bandwidth) |
| GPUDirect RDMA | Zero-copy: bypasses CPU, writes directly from NIC to GPU HBM |
| NVLink (intra-node) | Intra-node GPU → GPU transfers without PCIe |
| NVLink MNNVL (NVL72) | Multi-node NVLink (GB200 NVL72 rack-scale) |
| NVMe-oF | Storage → GPU (for long-context KV cache loading from disk) |
| TCP | Fallback; used for auxiliary metadata even when RDMA is available |
| CXL/shared-memory | Emerging rack-scale shared DRAM path |

### GPUDirect RDMA: The Key to Throughput

Without GPUDirect, KV cache transfer follows:
```
Prefill GPU VRAM → CPU RAM (PCIe) → NIC → network → NIC → CPU RAM (PCIe) → Decode GPU VRAM
```

With GPUDirect RDMA:
```
Prefill GPU VRAM → NIC → network → NIC → Decode GPU VRAM
```

CPU is entirely bypassed. This eliminates two PCIe crossings (one on each end) and removes the CPU as the bottleneck. For 40 GB of KV data (LLaMA3-70B at 128K context), GPUDirect RDMA reduces transfer time from ~15 seconds (PCIe double-copy path) to the NIC line-rate limit.

---

## Topology-Aware Path Selection

Modern inference servers have multiple CPU sockets, DRAM banks, GPUs, and RDMA NICs. Data can be transferred from any NIC to any GPU but with different bandwidths depending on the PCIe/UPI path.

Mooncake implements **topology-aware path selection**:
1. On startup, each server generates a topology matrix (GPU-NIC affinity, NUMA distances, PCIe bandwidth).
2. The matrix is broadcast to all cluster members.
3. For each KV transfer request, the TE selects the NIC(s) with the highest bandwidth path to the source/destination GPU.

This avoids the common failure mode where data crosses a PCIe/UPI bridge unnecessarily, which can halve effective bandwidth.

---

## Multi-NIC Pooling

A single RDMA NIC on an H100 server provides ~200 Gbps. Servers typically have 4–8 NICs. Mooncake's TE supports using **multiple RDMA NICs simultaneously** for a single transfer:

- Large transfers (>1 GB) are internally split into slices.
- Each slice is assigned to a different NIC based on topology affinity.
- All slices are submitted in parallel; completion is tracked independently.
- If one NIC fails or becomes congested, the TE retries on other NICs.

For 8-NIC servers, this can aggregate up to ~1.6 Tbps of theoretical RDMA bandwidth per node — sufficient to transfer even very large KV caches (e.g., 128K-token DeepSeek-V3 context) in a few seconds.

---

## NVLink Transport (Recommended for NVL72)

The NVIDIA GB200 NVL72 rack interconnects 72 GPUs with rack-scale NVLink (MNNVL). Mooncake supports NVLink as a transfer protocol, bypassing InfiniBand entirely for within-rack KV transfers:

```bash
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK
export MC_FORCE_MNNVL=True
```

NVLink provides ~10× higher bandwidth than InfiniBand (900 GB/s aggregate vs ~800 Gbps per port) and lower latency. For NVL72 deployments, NVLink transfer is the recommended configuration.

---

## Production Results

### Kimi Service (Moonshot AI)

- Mooncake enables Kimi to handle **75% more requests** vs the baseline architecture under real production workloads.
- In simulated long-context scenarios, achieves **up to 525% increase in throughput** while adhering to SLOs.
- **Prediction-based early rejection**: in highly overloaded conditions, Mooncake predicts which incoming requests cannot be served within SLO and rejects them early, preserving headroom for requests that can be served.

### Kimi K2 (July 2025)

- 128 H200 GPUs, PD disaggregation, large-scale expert parallelism
- **224,000 tokens/second prefill** throughput
- **288,000 tokens/second decode** throughput

---

## Mooncake's Relationship to Tiered KV Cache (Layer 17)

Mooncake combines PD disaggregation with a disaggregated KV cache pool — the same idea as HiCache but at the cluster level:

| Layer 17 HiCache | Layer 19 Mooncake |
|---|---|
| GPU VRAM (L1) | Prefill GPU VRAM |
| CPU DRAM (L2) | Distributed CPU DRAM across cluster nodes |
| Storage backend (L3) | SSD resources of the GPU cluster |
| Local prefix cache | Cross-cluster shared KV cache pool |
| `HiCacheStorage.batch_get_v2()` | Mooncake Transfer Engine `BatchTransfer` API |

The **Mooncake Store** (the KV cache pool) and the **Mooncake Transfer Engine** (the RDMA transport) are **separate components**. SGLang uses the Transfer Engine for PD KV transfer. HiCache uses the Mooncake Store as an L3 storage backend.

---

## Key Takeaways for Layer 19

- Mooncake is the production transfer engine behind SGLang's `--disaggregation-ib-device` flag — understanding it explains why multi-NIC RDMA configuration matters.
- **GPUDirect RDMA** is the key to achieving network-line-rate KV transfer without CPU involvement.
- **Topology-aware path selection** is critical in multi-NIC, multi-NUMA server configurations — naive NIC selection can halve transfer bandwidth.
- **NVLink** (`SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK`) is recommended for NVL72 racks where rack-scale NVLink is available.
- Mooncake's disaggregated KV cache pool extends the PD concept: not just separate prefill/decode, but also a shared, persistent KV store that enables prefix caching across all nodes in the cluster.
- **FAST 2025** — top-tier storage systems venue, validating that disaggregated serving is also a storage systems research problem.
