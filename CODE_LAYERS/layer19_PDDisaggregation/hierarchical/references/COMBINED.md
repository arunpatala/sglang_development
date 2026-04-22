# PD Disaggregation — Combined Reference

**What this file is:** A synthesis of all Layer 19 reference material into a single progressive narrative. The reading order moves from "why collocated prefill-decode scheduling fails" → "what goodput-optimised disaggregation solves" → "how KV cache is transferred between workers" → "what the transfer engines (Mooncake, NIXL) do under the hood" → "how Dynamo orchestrates this at scale" → "when not to disaggregate" → "how to launch it in SGLang."

**Sources synthesized:**

| Level | File | Source | Key contribution |
|---|---|---|---|
| L2 | `01_sglang_pd_docs.md` | SGLang Official Docs | Launch commands, all flags and env vars, Mooncake/NIXL backends, staging buffer |
| L2 | `02_lmsys_deepseek_96h100.md` | LMSYS Blog (May 2025) | Production recipe: 52.3k/22.3k tokens/sec on 96 H100 GPUs with DeepSeek-V3 |
| L3 | `01_distserve_osdi24.md` | USENIX OSDI 2024 | Foundational paper: goodput metric, interference quantification, 7.4× improvement |
| L3 | `02_splitwise_isca24.md` | ISCA 2024 | Production trace characterisation; hardware heterogeneity; 1.4× throughput at 20% lower cost |
| L3 | `03_mooncake_fast25.md` | USENIX FAST 2025 | Mooncake production KV transfer engine: GPUDirect RDMA, multi-NIC, topology-aware |
| L3 | `04_nvidia_dynamo_nixl.md` | NVIDIA GTC 2025 | Dynamo enterprise orchestration; NIXL connector library; KV-aware routing |
| L4 | `01_sarathi_osdi24.md` | OSDI 2024 | SARATHI chunked prefill — the aggregation baseline; what disaggregation must outperform |
| L4 | `02_taichi_agg_vs_disagg.md` | arXiv Aug 2025 | TaiChi — SLO-regime analysis: when to disaggregate vs aggregate |
| L4 | `03_vllm_disagg_connector.md` | vLLM Docs + GitHub | vLLM Connector/LookupBuffer abstraction; 6 connector types |

---

## 1. The Problem: Why Collocated Scheduling Fails

### What the two phases are

Every LLM inference request has two phases with fundamentally different resource profiles:

**Prefill** processes the entire prompt in one parallel forward pass. All tokens attend to all other tokens simultaneously via causal attention — a large matrix multiplication that saturates GPU compute. Arithmetic intensity scales with sequence length: a 10K-token prompt generates ~10× more FLOP than a 1K-token prompt. Prefill is **compute-bound**.

**Decode** generates one output token at a time. Each step loads all model weights and all active KV tensors from HBM to compute one new token. The arithmetic intensity per decode step is roughly constant regardless of sequence length — each step is `O(1 × model_size)` computation for `O(model_size + KV_size)` memory reads. Decode is **memory-bandwidth-bound**.

This is not a subtle distinction. On a roofline analysis, prefill sits far above the memory-bandwidth ceiling (compute-bound), while decode sits below it (memory-bandwidth-bound). The optimal GPU configuration for each phase is different.

### The two failure modes of colocation

**Failure mode 1: Head-of-line blocking (TPOT impact)**

When a large prefill request (say, a 10K-token coding prompt) runs on a collocated GPU, it monopolises the GPU's compute units for hundreds of milliseconds. Any decode requests scheduled concurrently are blocked until the prefill completes. From the decode requests' perspective, they experience a sudden spike in TPOT — the output tokens stop arriving. This violates TPOT SLOs.

SARATHI (L4/01) mitigates this with chunked prefill: split the large prefill into small chunks and interleave chunks with decode steps. This reduces the maximum block duration (from "entire prefill" to "one chunk"), but does not eliminate it. Decode requests still wait for each chunk. At high prefill rates, residual interference remains significant.

**Failure mode 2: Resource coupling (resource allocation impact)**

Even without interference, colocation forces the system to use the same GPU configuration (TP, PP) for both phases. But:

- Prefill benefits from high-compute TP configurations with large FLOP throughput.
- Decode benefits from configurations that maximise HBM bandwidth per step.

These are different hardware optima. A single configuration that works for both is suboptimal for each. DistServe (L3/01) measures that this coupling reduces goodput by up to 7× compared to a system that independently optimises each phase.

---

## 2. Goodput: The Correct Metric

DistServe introduced **goodput** as the metric to optimise for production LLM serving:

> **Goodput** = number of requests completed per second such that both TTFT ≤ TTFT_SLO and TPOT ≤ TPOT_SLO for >90% of requests.

Throughput (tokens/sec or requests/sec) ignores SLO compliance — a system can achieve high throughput while failing 40% of latency constraints. Goodput captures what users actually experience.

### The goodput-optimisation problem

Given a fixed GPU budget and SLO constraints (TTFT_SLO, TPOT_SLO), maximise goodput.

DistServe shows that with collocated serving:
- Satisfying TTFT_SLO requires limiting decode interference → low prefill utilisation.
- Satisfying TPOT_SLO requires limiting prefill interference → low decode utilisation.
- Trying to satisfy both simultaneously over-provisions one or both phases.

With disaggregation:
- Prefill pool optimises purely for TTFT_SLO (scale up prefill GPUs if TTFT is violated).
- Decode pool optimises purely for TPOT_SLO (scale up decode GPUs if TPOT is violated).
- Each pool scales independently.

**DistServe's result**: 7.4× more requests served under the same SLO constraints on a fixed GPU budget, or 12.6× tighter SLOs achievable at the same request rate.

---

## 3. The Disaggregated Architecture

### Phase-specific workers

PD disaggregation assigns each phase to a dedicated pool of GPU workers:

```
                  ┌─────────────────────────────┐
                  │         ROUTER              │
                  │  (distributes requests)     │
                  └──────────┬──────────────────┘
                             │
               ┌─────────────▼─────────────┐
               │      PREFILL POOL          │
               │  (compute-bound, high TP)  │
               │  Runs forward pass         │
               │  Produces KV cache         │
               └─────────────┬─────────────┘
                             │ KV transfer (RDMA / NIXL)
               ┌─────────────▼─────────────┐
               │      DECODE POOL           │
               │  (memory-bound, high BW)   │
               │  Receives KV cache         │
               │  Runs generation loop      │
               └─────────────┬─────────────┘
                             │ output tokens
                           client
```

### The handshake protocol (SGLang implementation)

```
Step 1: Request arrives at router → dispatched to a prefill worker and a decode worker simultaneously.
Step 2: Decode worker PRE-ALLOCATES KV cache pages for the expected KV size.
         → sends KV page indices to prefill worker (bootstrap/handshake).
Step 3: Prefill worker runs the forward pass (model inference on input tokens).
         → writes KV tensors directly into decode worker's pre-allocated pages (via RDMA).
         → sends completion notification.
Step 4: Decode worker receives completion notification.
         → begins autoregressive generation loop.
         → streams output tokens back to router → client.
```

The critical correctness requirement: decode **pre-allocates** before prefill runs. This prevents a race condition where the KV cache is ready but the decode worker has no space.

### Independent resource allocation

```bash
# Prefill workers: optimise for compute throughput
python -m sglang.launch_server \
  --disaggregation-mode prefill \
  --tp-size 4 --dp-size 8 \          # Smaller TP = more FLOP per token
  --enable-dp-attention \
  --chunked-prefill-size 32768        # Limit individual step duration

# Decode workers: optimise for memory bandwidth
python -m sglang.launch_server \
  --disaggregation-mode decode \
  --tp-size 8 --dp-size 4 \          # Larger TP = more HBM bandwidth
  --max-running-requests 128          # Fill HBM with active sequences
```

---

## 4. KV Transfer: The Infrastructure Layer

### Why the transfer layer is the hard part

After disaggregation, the bottleneck shifts from GPU compute to KV cache transfer. For a large model and long prompt:

- LLaMA-70B, 4K-token prompt, TP=8: ~8 GB of KV cache per request.
- DeepSeek-V3, 8K-token prompt, TP=16: ~40 GB of KV cache per request.

At 200 Gbps RDMA bandwidth (one 200 Gbps NIC), 40 GB takes 1.6 seconds. With 8 × 200 Gbps NICs in parallel, it takes 200 ms. This is the transfer latency floor — the minimum time the decode worker must wait before it can start generating.

For the system to achieve good goodput, the KV transfer must be:
1. **Fast**: saturate all available NIC bandwidth simultaneously.
2. **Zero-copy**: no CPU involvement in the data path.
3. **Topology-aware**: avoid PCIe/UPI crossings that cut effective bandwidth in half.
4. **Reliable**: retry on NIC failure without corrupting the KV state.

### Mooncake Transfer Engine

The **Mooncake Transfer Engine (TE)** provides all four properties:

**Fast via multi-NIC pooling**: registers all RDMA NICs on each node; distributes a large transfer across NICs based on their topology affinity. For 8-NIC nodes: up to 1.6 Tbps aggregate theoretical bandwidth.

**Zero-copy via GPUDirect RDMA**: the NIC reads directly from the prefill GPU's HBM over PCIe and writes directly into the decode GPU's HBM on the other end. CPU is not involved in the data path.

**Topology-aware path selection**: each node broadcasts a topology matrix (GPU-NIC affinity, NUMA distances). For each transfer, the TE selects the NIC with the shortest electrical path to the source GPU.

**Reliable via NIC retry**: if a NIC becomes temporarily unavailable, the TE automatically retries the affected slices on other NICs.

### NIXL: The Vendor-Agnostic Alternative

NIXL (NVIDIA Inference Xfer Library) provides the same capabilities through a plugin architecture:

- **UCX backend**: RDMA InfiniBand/RoCEv2, GPUDirect RDMA (same as Mooncake's RDMA path).
- **GDS backend**: NVIDIA GPUDirect Storage for loading KV from NVMe directly to GPU.
- **TCP backend**: fallback for non-RDMA clusters.
- **ETCD-based metadata exchange**: agents on different nodes exchange memory registration metadata without blocking the data path.

NIXL is backend-agnostic — the same API works whether the cluster uses InfiniBand, RoCEv2, or EFA (AWS). Mooncake is RDMA-native but also adds NVLink support for NVL72.

### NVLink for Rack-Scale Transfer (NVL72)

On GB200 NVL72 racks, 72 GPUs are interconnected with rack-scale NVLink (MNNVL) at ~900 GB/s aggregate. Mooncake supports this as a transfer protocol:

```bash
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK
export MC_FORCE_MNNVL=True
```

NVLink transport eliminates InfiniBand entirely for within-rack transfers. At NVL72 scale with full disaggregation, even 40 GB KV caches transfer in sub-100ms.

---

## 5. Enterprise-Scale Orchestration (Dynamo)

For deployments with many prefill and decode workers, a bare KV transfer engine is insufficient. Dynamo (NVIDIA) adds the orchestration layer:

### KV-Aware Routing

Dynamo's router tracks prefix cache state across prefill workers. For each incoming request, it routes to the prefill worker with the **highest prefix overlap** — avoiding recomputation of already-cached prefixes.

Example: a 10K-token shared system prompt is cached on prefill worker 2. The next request with that same system prompt is routed to worker 2 → zero prefill computation for the shared prefix → TTFT for that request drops from seconds to milliseconds.

SGLang's current router uses round-robin. Cache-aware routing (Dynamo-style) is the next evolution and reduces effective TTFT by eliminating redundant prefill for repeated prefixes.

### Dynamic Scaling

Dynamo's Planner monitors SLO violation signals and adjusts pool sizes:
- TTFT violation → scale up prefill pool (add more prefill workers).
- TPOT violation → scale up decode pool (add more decode workers).

At steady state: minimum GPUs in each pool sufficient to satisfy both SLOs simultaneously.

---

## 6. When Not to Disaggregate

TaiChi (L4/02) defines the exact SLO regimes where disaggregation helps vs hurts:

### Disaggregation wins when TPOT constraint is tight

Dedicated decode GPUs are never interrupted by prefill → stable, predictable TPOT. If TPOT SLO is the binding constraint, disaggregation is the right choice.

### Aggregation wins when TTFT constraint is tight

All GPUs can process prefill → lower TTFT (more compute capacity for prompt processing). Disaggregation reduces the number of GPUs handling prefill (only the prefill pool), which increases TTFT. If TTFT SLO is tight and TPOT SLO is relaxed, aggregation achieves higher goodput.

### Neither is optimal for balanced SLOs

Under tight TTFT + tight TPOT, a hybrid (TaiChi-style dynamic worker assignment) achieves 77% higher goodput than either pure approach.

### MoE models: disaggregation is mandatory

For DeepSeek-V3 and similar MoE models, PD disaggregation is required regardless of SLO regime because the expert parallelism dispatch patterns for prefill (compute-intensive, large-batch) and decode (memory-intensive, single-token) are fundamentally incompatible on shared hardware.

---

## 7. Production Deployment: DeepSeek-V3 at 96 H100 GPUs

The canonical production deployment (LMSYS Blog, May 2025):

**Cluster**: 12 nodes × 8 H100 GPUs = 96 GPUs total.
**Split**: 3 nodes (24 GPUs) for prefill + 9 nodes (72 GPUs) for decode.
**Transfer**: Mooncake over InfiniBand.
**Expert parallelism**: DeepEP (normal dispatch on prefill, low-latency dispatch on decode) + EPLB.
**Results**:

| Metric | Value |
|---|---|
| Input throughput per node | 52,300 tokens/second |
| Output throughput per node | 22,300 tokens/second |
| Input sequence length | 2,000 tokens |

**Why 3:9 prefill-to-decode ratio?** DeepSeek-V3 is decode-heavy: it generates long outputs for complex reasoning tasks. Decode is the sustained steady-state computation; prefill is bursty. More decode GPUs means more concurrent decode sequences, maximising output throughput.

---

## 8. Reference Architecture Summary

```
[Client] → [Router (round-robin or cache-aware)]
                  ↓
         [Prefill Pool — compute-optimised]
         ┌────────────────────────────────┐
         │ --disaggregation-mode prefill  │
         │ --tp-size N                    │
         │ --chunked-prefill-size 32768   │
         │ --disaggregation-ib-device     │
         │ --moe-a2a-backend deepep       │
         └──────────────┬─────────────────┘
                        │ KV transfer (Mooncake RDMA / NIXL)
         [Decode Pool — memory-bandwidth-optimised]
         ┌────────────────────────────────┐
         │ --disaggregation-mode decode   │
         │ --tp-size M (M may ≠ N)        │
         │ --max-running-requests 128     │
         │ --disaggregation-ib-device     │
         └──────────────┬─────────────────┘
                        ↓
                   [Output tokens → Client]
```

**All references in Layer 19 map to one of three concerns:**
1. **Why** (DistServe, Splitwise, TaiChi): the interference problem, goodput metric, SLO-regime analysis.
2. **How** (Mooncake, NIXL, Dynamo, vLLM connector): KV transfer mechanics, orchestration, connector abstractions.
3. **What** (SGLang docs, LMSYS blog): launch commands, flags, production recipes.
