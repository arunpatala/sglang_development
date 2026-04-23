# NVIDIA Dynamo and NIXL: Enterprise-Scale Disaggregated Serving

**Sources:**
- **Dynamo Blog:** https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models (March 18, 2025)
- **Dynamo Docs:** https://docs.nvidia.com/dynamo/design-docs/overall-architecture
- **Dynamo GitHub:** https://github.com/ai-dynamo/dynamo
- **NIXL GitHub:** https://github.com/ai-dynamo/nixl
- **NIXL Blog:** https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library/
- **NIXL Docs:** https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md

**Level:** L3 — Enterprise orchestration framework + KV transfer library
**Why here:** NVIDIA Dynamo is the enterprise-scale orchestration layer that treats disaggregated serving as a first-class primitive, and NIXL is the KV transfer library it uses — also supported natively by SGLang, vLLM, TensorRT-LLM, and Anyscale Ray. Together they represent NVIDIA's production answer to the PD disaggregation problem. Understanding Dynamo's architecture explains the routing, control, and discovery planes that a production disaggregated cluster needs beyond just the transfer engine.

---

## NVIDIA Dynamo: What It Is

NVIDIA Dynamo is a **high-throughput, low-latency, open-source inference serving framework** announced at GTC 2025. It is backend-agnostic (supports vLLM, SGLang, TensorRT-LLM) and designed to scale from single-GPU to thousands-of-GPU deployments.

Dynamo's central thesis: **disaggregated serving requires not just a transfer engine but a full orchestration layer** with separate planes for requests, control, discovery, and events.

### Design Goals

1. **Latency stability**: keep TTFT and ITL predictable under bursty and mixed-length traffic.
2. **GPU efficiency**: disaggregate prefill and decode so each can scale independently.
3. **Compute reuse**: minimise KV recomputation through KV-aware routing and cache lifecycle management.

---

## Dynamo's 4-Plane Architecture

### Request Plane

Handles the data path for user requests. In disaggregated mode:
1. Client sends request to Frontend.
2. Frontend validates/preprocesses and forwards to Router.
3. Router chooses a Prefill worker.
4. Prefill computes KV and returns transfer metadata.
5. Router chooses a Decode worker.
6. Decode receives KV state via NIXL transfer path.
7. Decode streams output tokens back through Frontend to Client.

### Control Plane

Manages the configuration and state of workers. The Planner component adjusts the prefill/decode pool size dynamically in response to SLO signals — if TPOT is violating SLO, scale up the decode pool; if TTFT is violating SLO, scale up the prefill pool.

### Discovery Plane

Workers register themselves with an etcd-based lease mechanism. The Router queries discovery to find available prefill and decode workers. Stale workers (lease expired) are automatically removed; traffic reroutes to healthy workers.

### Event Plane

Asynchronous event bus for propagating state changes (cache hit/miss signals, worker availability changes, SLO violation alerts) without blocking the request path.

---

## KV-Aware Routing: Dynamo's Key Innovation

Unlike simple load-balanced routing, Dynamo's Router tracks **KV cache state** across workers:

- Each prefill worker maintains a local KV cache (similar to SGLang's RadixCache).
- The Router tracks which prefixes are cached on which workers.
- For each incoming request, the Router routes to the prefill worker with the **highest prefix overlap** — avoiding KV recomputation.

**Result**: in a test with 100K requests to DeepSeek R1-Distilled Llama-70B FP8 on 2 H100 nodes, KV-aware routing reduces TTFT by routing repeat prefixes to workers that already have the KV cached.

**Comparison to SGLang**: SGLang's router currently uses round-robin; cache-aware routing is a planned enhancement. Dynamo has this built into its Router design from the start.

---

## KV Block Manager (KVBM)

KVBM extends effective KV cache capacity using multi-tier memory offload/recall — essentially HiCache (Layer 17) integrated into the Dynamo control plane. When a prefill worker's GPU VRAM fills, KVBM offloads KV blocks to CPU DRAM or NVMe, and restores them on cache hit.

---

## Benchmark Results

| Metric | Dynamo (disaggregated) vs baseline |
|---|---|
| Requests served (DeepSeek-R1, GB200 NVL72) | **Up to 30× more requests** vs collocated |
| Throughput on H100 (disaggregated) | Significantly higher than collocated for TPOT-constrained workloads |

---

## NIXL: The KV Transfer Library

NIXL (NVIDIA Inference Xfer Library) is the open-source, vendor-agnostic data movement library used by Dynamo for KV transfer. It is also supported directly by SGLang (`--disaggregation-transfer-backend nixl`) and vLLM (`NixlConnector`).

### Core Abstraction: Transfer Agent

Each server process runs a NIXL **Transfer Agent**. The agent manages three components:

1. **Memory Section**: a unified view of all registered memory — GPU VRAM (via GPUDirect RDMA), CPU DRAM (pinned), local NVMe, remote storage (via NVMe-oF or S3). All exposed through the same buffer-list API.

2. **Transfer Backend Interface**: pluggable transport backends — UCX (RDMA InfiniBand/RoCEv2), NVIDIA Magnum IO GPUDirect Storage (GDS), TCP, NVLink. Automatically selects the best backend based on source/destination memory types.

3. **Metadata Handler**: exchanges registration metadata between agents on different nodes. Uses ETCD for distributed metadata exchange. Metadata is cached to avoid per-transfer latency.

### Three Use Cases in NIXL

| Use case | Description |
|---|---|
| **Disaggregation** | KV blocks from prefill GPU VRAM to decode GPU VRAM (RDMA zero-copy) |
| **Long-context KV loading** | KV cache from NVMe/object storage to GPU VRAM (GDS path) |
| **Expert parallelism** | MoE all-to-all expert activations across GPUs (NVLink/RDMA) |

### Async Transfer API

```python
# Submit async write (non-blocking)
handle = agent.transfer_submit_write(src_addr, dst_addr, size)

# Check completion without blocking
status = agent.transfer_check_status(handle)
```

This async model allows compute and transfer to overlap — the decode server can begin processing the already-transferred layers while later layers are still being transferred over the network.

### NIXL in SGLang

```bash
# Single-node NIXL
python -m sglang.launch_server \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --port 30000

# Switch NIXL backend to LIBFABRIC
export SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC
```

### NIXL Benchmarking Tools

- **NIXLBench**: model-agnostic bandwidth benchmark; sweeps block sizes and batch sizes; reports bandwidth + latency percentiles.
- **KVBench**: LLM-aware profiler; auto-calculates exact KV I/O size for supported models (LLaMA, Mistral, DeepSeek) and generates ready-to-run NIXLBench commands.

---

## Dynamo vs SGLang PD Disaggregation

| Feature | SGLang PD Disagg | NVIDIA Dynamo |
|---|---|---|
| KV transfer engine | Mooncake or NIXL | NIXL (primary) |
| Routing | Round-robin (router) | KV-aware prefix routing |
| Backend support | SGLang only | vLLM, SGLang, TensorRT-LLM |
| Discovery | Manual URL list | etcd lease-based dynamic discovery |
| Control plane | Manual scaling | Planner (auto-scaling per SLO) |
| Multi-tier cache | HiCache integration | KVBM (built-in) |
| Fault tolerance | Manual restart | Automatic rerouting via discovery |

**Summary**: SGLang's disaggregation is purpose-built and optimised for SGLang; Dynamo is a general orchestration layer that adds KV-aware routing, dynamic scaling, and multi-framework support on top.

---

## Key Takeaways for Layer 19

- **Dynamo** shows that production disaggregation needs more than a transfer engine: KV-aware routing, dynamic scaling (Planner), distributed discovery (etcd leases), and multi-tier cache management are all required components.
- **NIXL** is the vendor-agnostic alternative to Mooncake: uses UCX/GDS for RDMA, supports all backends through a plugin architecture, and integrates with every major inference framework.
- **KV-aware routing** (routing to the worker with the highest prefix overlap) is the next evolution from SGLang's current round-robin router — reduces TTFT by avoiding recomputation.
- The 30× throughput improvement on GB200 NVL72 comes from combining disaggregation with NVLink-speed KV transfer — the same principle behind SGLang's NVLink Mooncake transport.
- NIXL's async transfer API (`transfer_submit_write` + `transfer_check_status`) enables compute-transfer overlap — the same pattern used by HiCache's per-layer loading in Layer 17.
