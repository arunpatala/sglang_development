# PD Disaggregation — Material Omitted from COMBINEDL1L2.md

**What this file is:** The full text of every section omitted from `COMBINEDL1L2.md`. The "What Is Left Out and Why" appendix of `COMBINEDL1L2.md` names each omission and explains why it was excluded. This file preserves the complete original text so no source material is lost.

**Parent file:** `COMBINEDL1L2.md` (L1 + L2 synthesis)
**Sources:** L3/01, L3/02, L3/03, L3/04, L4/01, L4/02, L4/03, L2/02 (MoE internals)

---

## Omission 1: DistServe — Formal Goodput Model and Interference Quantification

**Source:** `L3/01_distserve_osdi24.md`
**Venue:** USENIX OSDI 2024
**Why omitted from COMBINEDL1L2.md:** COMBINEDL1L2.md introduces goodput as a concept and cites the 4–7× improvement number. The formal model — how goodput is derived as a function of TTFT/TPOT SLO constraints, how interference is quantified with queuing theory, and how resource allocation is optimised per-phase — requires engaging with the paper's formal framework. The concept belongs at L1; the derivation belongs at L3.

---

### The Core Problem: Why Throughput Is the Wrong Metric

Traditional LLM serving optimises for **throughput** — tokens per second or requests per second. But LLM applications care about two distinct latency dimensions:

- **TTFT** (Time to First Token): latency from request submission to the first output token. Dominated by the prefill phase. Users perceive this as "how long before the model starts responding."
- **TPOT** (Time Per Output Token): latency between consecutive output tokens during decode. Users perceive this as "how fast the model is typing."

**Goodput** is the number of requests completed per second that satisfy **both** TTFT and TPOT SLO constraints simultaneously. A system can have high throughput while having poor goodput if many requests violate one of the SLOs.

### The Colocation Coupling Problem

When prefill and decode share the same GPUs:

1. **Prefill-decode interference**: a large prefill computation wave (processing a 10K-token prompt) runs for hundreds of milliseconds on the GPU. Any decode step scheduled during this window is delayed by the full prefill duration. This spikes TPOT for all currently-decoding requests.

2. **Resource coupling**: the same GPU must handle both compute-bound prefill (needs high FLOP throughput) and memory-bandwidth-bound decode (needs high HBM bandwidth). The parallelism strategy (TP, PP) that optimises prefill degrades decode, and vice versa. There is no single configuration that is optimal for both.

3. **Over-provisioning trap**: to meet both TTFT and TPOT SLOs simultaneously with a collocated system, you must over-provision GPUs — paying for hardware that runs at low utilisation.

### DistServe's Solution: Key Design Choices

**Independent resource allocation**: each phase is provisioned separately based on its own SLO requirements:
- TTFT SLO tight → scale prefill pool (more GPUs per instance, or more instances).
- TPOT SLO tight → scale decode pool.
- Both tight → scale both independently.

**Per-phase parallelism**: the prefill pool can use tensor parallelism optimised for compute throughput (TP-4, high FLOP utilisation). The decode pool can use a configuration optimised for memory bandwidth (TP-8, maximise HBM reads per step). These can be different.

**Bandwidth-aware placement**: DistServe places the two phases according to the serving cluster's bandwidth. If prefill and decode are on the same physical rack (high NVLink/InfiniBand bandwidth), the KV transfer cost is amortised. If on different pods, DistServe reduces transfer cost by adjusting the parallelism to minimise KV cache size.

### Experimental Results

Evaluated on OPT-13B, OPT-66B, OPT-175B across three application workloads (chatbot, document summarisation, coding). Compared against vLLM-continuous-batching (state-of-the-art collocated baseline):

| Metric | DistServe vs vLLM |
|---|---|
| Max requests served at SLO (goodput) | **7.4× more requests** |
| Tightest SLO achievable at same rate | **12.6× tighter** |
| Requests within SLO (>90% guarantee) | Maintained across all workloads |

**Why 7.4×?** The prefill pool handles 100% compute-bound workloads with optimal TP configuration; the decode pool handles 100% memory-bandwidth-bound workloads with optimal configuration. Neither is compromised by the other's requirements.

### Concept-to-SGLang Mapping

| DistServe Concept | SGLang Equivalent |
|---|---|
| Prefill instance (KV producer) | `--disaggregation-mode prefill` server |
| Decode instance (KV consumer) | `--disaggregation-mode decode` server |
| KV cache transfer | Mooncake RDMA or NIXL |
| Router / orchestration layer | `sglang_router.launch_router --pd-disaggregation` |
| Per-phase resource allocation | Separate `--tp-size`, `--dp-size` per mode |
| Bandwidth-aware placement | `--disaggregation-ib-device` NIC selection |

**BibTeX:**
```bibtex
@inproceedings{zhong2024distserve,
  title  = {DistServe: Disaggregating Prefill and Decoding for
             Goodput-optimized Large Language Model Serving},
  author = {Yinmin Zhong and Shengyu Liu and Junda Chen and Jianbo Hu
            and Yibo Zhu and Xuanzhe Liu and Xin Jin and Hao Zhang},
  booktitle = {18th USENIX Symposium on Operating Systems Design and
               Implementation (OSDI 24)},
  year   = {2024},
  url    = {https://arxiv.org/abs/2401.09670}
}
```

---

## Omission 2: Splitwise — Production Trace Characterisation and Hardware Heterogeneity

**Source:** `L3/02_splitwise_isca24.md`
**Venue:** ISCA 2024 (Best Paper Award)
**Why omitted from COMBINEDL1L2.md:** The hardware heterogeneity insight (prefill benefits from FLOPS, decode benefits from HBM bandwidth) is carried in COMBINEDL1L2.md's cost arithmetic section. The production trace analysis from Azure, the SplitwiseSim simulator, and the vLLM prototype PR are all L3 material — they validate the disaggregation hypothesis empirically and provide the cost/throughput tradeoff data that motivated real deployments.

---

### Production Workload Characterisation (Azure Traces)

Splitwise's most important contribution is its use of **real Azure production traces** from two LLM serving services. Key quantified findings:

| Characteristic | Prefill | Decode |
|---|---|---|
| Compute profile | Compute-intensive, saturates GPU FLOP throughput | Memory-bandwidth-bound, loads weights + KV per step |
| GPU utilisation | High (matrix multiplications, full batch parallelism) | Low (autoregressive, one token per step) |
| Memory pressure | Low (transient activations only) | High (KV cache grows with sequence length) |
| Power draw | High (compute-bound) | Lower (memory-bandwidth-bound) |
| Duration | Short (milliseconds to seconds) | Long (seconds to minutes) |

**Key insight quantified from production**: token generation phases **do not require the compute capability of the latest GPUs** and can be run on lower-power, lower-cost hardware with equivalent quality. Even with continuous batching, decode-phase GPU arithmetic intensity is too low to saturate high-FLOP GPUs.

### Hardware Heterogeneity: The Novel Dimension

Splitwise proposes running prefill and decode on **different hardware tailored to each phase**:

| Phase | Optimal hardware | Reason |
|---|---|---|
| Prefill (prompt computation) | Latest GPUs (H100, B200) | Highest FLOP throughput per dollar |
| Decode (token generation) | Older or memory-optimised GPUs (A100, H20) | High HBM bandwidth per dollar; FLOP budget is not the constraint |

**Measured result**: by using H100 for prefill + A100 for decode vs all-H100:
- **1.4× higher throughput at 20% lower cost**
- OR **2.35× more throughput at the same cost and power budget**

This hardware heterogeneity insight influenced real cloud deployments — some providers run prefill on latest-gen GPUs while decode runs on previous-generation hardware, reducing serving cost without quality loss.

### The KV State Transfer

Splitwise analyses the KV state transfer using fast back-plane interconnects:
- Within a rack: NVLink or PCIe direct transfers (high bandwidth, low latency)
- Across racks: InfiniBand (lower bandwidth, higher latency)

**Amortisation threshold from the paper**: for prompts longer than ~500 tokens generating more than ~50 output tokens, disaggregation is always net-positive in latency.

### SplitwiseSim: Cluster-Level Simulation

SplitwiseSim (https://github.com/Mutinifni/splitwise-sim) is a discrete-event simulator for evaluating cluster-level PD disaggregation policies:
- Models heterogeneous hardware pools, variable prompt/output length distributions (including the two Azure traces), routing policies, and KV transfer latency as a function of payload size and network topology.
- Used in research to prototype new routing strategies without a physical cluster.
- Useful for capacity planning: determine optimal prefill/decode GPU count ratio for a given workload distribution.

### The vLLM Prototype

Splitwise includes a prototype implementation of its KV-cache transfer mechanism in vLLM (**GitHub PR #2809** — the first public implementation of inter-instance KV transfer). This PR became the direct ancestor of vLLM's `vllm/distributed/kv_transfer/` module that all vLLM disaggregation connectors build on today.

**BibTeX:**
```bibtex
@inproceedings{patel2024splitwise,
  title     = {Splitwise: Efficient Generative LLM Inference Using Phase Splitting},
  author    = {Pratyush Patel and Esha Choukse and Chaojie Zhang and Aashaka Shah
               and {\'I}{\~n}igo Goiri and Saeed Maleki and Ricardo Bianchini},
  booktitle = {Proceedings of the 51st Annual International Symposium on
               Computer Architecture (ISCA '24)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2311.18677}
}
```

---

## Omission 3: Mooncake — Transfer Engine Internals

**Source:** `L3/03_mooncake_fast25.md`
**Venue:** USENIX FAST 2025
**Why omitted from COMBINEDL1L2.md:** COMBINEDL1L2.md names Mooncake as a transfer backend and cites the 27–107ms transfer window numbers. The mechanics that produce those numbers — GPUDirect RDMA path, multi-NIC pooling, topology-aware NIC selection, NVLink transport — are L3 detail that is not necessary for understanding how to launch or evaluate a disaggregated cluster.

---

### GPUDirect RDMA: The Key to Throughput

Without GPUDirect, KV cache transfer follows a CPU-mediated path:

```
Prefill GPU VRAM → CPU RAM (PCIe) → NIC → network → NIC → CPU RAM (PCIe) → Decode GPU VRAM
```

With GPUDirect RDMA, the CPU is entirely bypassed:

```
Prefill GPU VRAM → NIC → network → NIC → Decode GPU VRAM
```

This eliminates two PCIe crossings and removes the CPU as the bottleneck. For 40 GB of KV data (LLaMA3-70B at 128K context), GPUDirect RDMA reduces transfer time from ~15 seconds (PCIe double-copy path) to the NIC line-rate limit.

### Topology-Aware Path Selection

Modern inference servers have multiple CPU sockets, DRAM banks, GPUs, and RDMA NICs. Data can be transferred from any NIC to any GPU, but with different bandwidths depending on the PCIe/UPI path.

Mooncake's topology-aware path selection:
1. On startup, each server generates a topology matrix (GPU-NIC affinity, NUMA distances, PCIe bandwidth).
2. The matrix is broadcast to all cluster members.
3. For each KV transfer request, the Transfer Engine selects the NIC(s) with the highest bandwidth path to the source/destination GPU.

This avoids the common failure mode where data crosses a PCIe/UPI bridge unnecessarily, which can halve effective bandwidth.

### Multi-NIC Pooling

A single RDMA NIC on an H100 server provides ~200 Gbps. Servers typically have 4–8 NICs. Mooncake's Transfer Engine supports using **multiple RDMA NICs simultaneously** for a single transfer:

- Large transfers (>1 GB) are internally split into slices.
- Each slice is assigned to a different NIC based on topology affinity.
- All slices are submitted in parallel; completion is tracked independently.
- If one NIC fails or becomes congested, the TE retries on other NICs.

For 8-NIC servers, this can aggregate up to ~1.6 Tbps of theoretical RDMA bandwidth per node — sufficient to transfer even very large KV caches (128K-token DeepSeek-V3 context) in a few seconds.

### NVLink Transport for NVL72

The NVIDIA GB200 NVL72 rack interconnects 72 GPUs with rack-scale NVLink (MNNVL). Mooncake supports NVLink as a transfer protocol, bypassing InfiniBand entirely for within-rack KV transfers:

```bash
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK
export MC_FORCE_MNNVL=True
```

NVLink provides ~10× higher bandwidth than InfiniBand (900 GB/s aggregate vs ~800 Gbps per port) and lower latency. For NVL72 deployments, NVLink transfer is the recommended configuration.

### Disaggregated KV Cache Pool (Beyond the Transfer Engine)

Mooncake adds a second tier of disaggregation: beyond separating prefill/decode clusters, it builds a **distributed KVCache pool** from underutilised CPU DRAM and SSD resources across the GPU cluster. This enables prefix caching that survives node failures and can be shared across all prefill and decode instances — functionally similar to SGLang's HiCache but at cluster scale.

**Prediction-based early rejection**: in highly overloaded conditions, Mooncake predicts whether an incoming request can be served within SLO before accepting it, rejecting those it cannot serve early to preserve headroom. SGLang's current disaggregation mode does not implement this.

### Production Results

- Kimi service: 75% more requests vs baseline; up to 525% throughput increase in long-context scenarios.
- Kimi K2 (128 H200, July 2025): 224,000 tokens/second prefill, 288,000 tokens/second decode.

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

## Omission 4: NVIDIA Dynamo — 4-Plane Architecture and KV-Aware Routing

**Source:** `L3/04_nvidia_dynamo_nixl.md`
**Why omitted from COMBINEDL1L2.md:** COMBINEDL1L2.md names Dynamo as a production framework built around disaggregation. The 4-plane architecture (request, control, discovery, event planes) and KV-aware routing internals are enterprise orchestration concerns beyond what someone needs to understand or launch a disaggregated SGLang cluster.

---

### Dynamo's Central Thesis

> "Disaggregated serving requires not just a transfer engine but a full orchestration layer" with separate planes for requests, control, discovery, and events.

### The 4-Plane Architecture

**Request Plane** — the data path:
1. Client → Frontend → Router
2. Router chooses a Prefill worker
3. Prefill computes KV, returns transfer metadata
4. Router chooses a Decode worker
5. Decode receives KV via NIXL transfer
6. Decode streams output tokens back through Frontend to Client

**Control Plane (Planner)** — dynamic pool sizing:
- Monitors real-time SLO violation rates: TTFT P95 > TTFT_SLO → scale prefill pool.
- Issues scale-up/scale-down commands to Kubernetes or bare-metal scheduler.
- Uses a feedback control loop with configurable reaction time and cooldown.
- SGLang's disaggregation currently has no equivalent; scale-out/scale-in is manual.

**Discovery Plane** — worker registration via etcd leases:
- Each worker publishes its endpoint, role (prefill/decode), and current load state.
- Lease TTL: 10 seconds (configurable). Worker death → lease expires → router removes it.
- New workers start serving within one lease period after registration — no manual router reconfiguration.
- SGLang's router uses a static list of prefill/decode endpoints via CLI flags.

**Event Plane** — asynchronous state propagation:
- KV cache hit/miss signals, SLO violation alerts, worker availability changes.
- Router subscribes to cache state change events rather than polling each worker.
- This is what makes KV-aware routing scalable: the routing table updates via events, not polling.

### KV-Aware Routing: Dynamo's Key Innovation

Unlike round-robin, Dynamo's Router tracks **KV cache state** across workers:
- Each prefill worker maintains a local KV cache (similar to SGLang's RadixCache).
- The Router tracks which prefixes are cached on which workers.
- For each incoming request, the Router routes to the prefill worker with the **highest prefix overlap** — avoiding KV recomputation.

SGLang's router currently uses round-robin; KV-aware routing is a planned enhancement. Dynamo has this built into its Router design from the start.

### NIXL: The KV Transfer Library

NIXL (NVIDIA Inference Xfer Library) is Dynamo's KV transfer library, also supported natively by SGLang (`--disaggregation-transfer-backend nixl`) and vLLM (`NixlConnector`).

**Core abstraction — Transfer Agent**: each server process runs one. The agent manages:
1. **Memory Section**: unified view of all registered memory — GPU VRAM (GPUDirect RDMA), CPU DRAM (pinned), local NVMe, remote storage (NVMe-oF or S3).
2. **Transfer Backend Interface**: pluggable backends — UCX (RDMA InfiniBand/RoCEv2), NVIDIA Magnum IO GPUDirect Storage (GDS), TCP, NVLink. Automatically selects the best backend based on source/destination memory types.
3. **Metadata Handler**: exchanges registration metadata between agents via etcd. Metadata is cached to avoid per-transfer latency.

**Async transfer API**:
```python
# Submit non-blocking write
handle = agent.transfer_submit_write(src_addr, dst_addr, size)

# Check completion without blocking
status = agent.transfer_check_status(handle)
```

This async model allows compute and transfer to overlap — the decode server can begin processing already-transferred layers while later layers are still in transit.

**NIXL benchmarking tools:**
- **NIXLBench**: model-agnostic bandwidth/latency benchmark; sweeps block sizes and batch sizes; reports bandwidth + latency percentiles (P50/P95/P99).
- **KVBench**: LLM-aware profiler; auto-calculates exact KV I/O size for supported models (LLaMA, Mistral, DeepSeek) and generates ready-to-run NIXLBench commands.

### Dynamo Benchmark Result

| Metric | Result |
|---|---|
| Requests served (DeepSeek-R1, GB200 NVL72) | Up to **30× more requests** vs collocated |

The 30× improvement comes from combining disaggregation with NVLink-speed KV transfer on GB200 NVL72.

---

## Omission 5: SARATHI — Chunked Prefill Mechanics and Pipeline Analysis

**Source:** `L4/01_sarathi_osdi24.md`
**Venue:** USENIX OSDI 2024
**Why omitted from COMBINEDL1L2.md:** COMBINEDL1L2.md describes chunked prefill as "the scheduling workaround" and "the prior art" that reduces but cannot eliminate interference. The mechanics of how chunks are constructed, scheduled, and interleaved with decode steps — and the pipeline bubble reduction analysis — are L4 material needed only by practitioners building or extending a scheduler.

---

### The Problem SARATHI Addresses

Standard LLM inference suffers from two inefficiencies:
1. **Head-of-line blocking**: a single long prefill (10K-token prompt) monopolises the GPU for hundreds of milliseconds; all decode requests are blocked.
2. **Pipeline parallelism bubbles**: large prefill microbatches have variable duration, creating imbalanced stage times.

### Chunked Prefill Mechanics

Instead of one large prefill step, SARATHI splits prefill into **fixed-size chunks** scheduled across multiple steps:

```
Standard: [PPPPPPPPPPPP | D D D D D D D D D]
           ← 12 prefill tokens →  ← decode →
           One prefill blocks all decodes for entire duration.

SARATHI:  [PPPP | D D D D] → [PPPP | D D D D] → [PPPP | D D D D]
          Chunk 1 + decodes   Chunk 2 + decodes   Chunk 3 + decodes
```

**Decode-maximal batching**: each batch contains one prefill chunk (saturates GPU FLOP throughput during the chunk) + as many decode requests as fit in remaining KV memory. Decode requests "piggyback" on the compute wave — their incremental compute cost is an order of magnitude lower.

### Pipeline Bubble Reduction

Chunked prefill ensures every stage in a pipeline-parallel setup receives a uniformly-sized chunk, eliminating microbatch duration imbalance:
- **6.29× pipeline bubble reduction** on GPT-3 (175B) with PP=8
- **1.91× end-to-end throughput improvement** from the bubble reduction alone

### Results

| Model | Metric | Improvement |
|---|---|---|
| LLaMA-13B (A6000) | Decode throughput | **Up to 10×** |
| LLaMA-13B (A6000) | End-to-end throughput | **Up to 1.33×** |
| LLaMA-33B (A100) | Decode throughput | **4.25×** |
| LLaMA-33B (A100) | End-to-end throughput | **1.25×** |
| GPT-3 (pipeline parallel) | Pipeline bubbles | **6.29× reduction** |

### What Chunked Prefill Cannot Fix

| Problem | SARATHI (chunked prefill) | PD Disaggregation |
|---|---|---|
| Head-of-line blocking | Reduced (chunk duration, not full prefill) | Eliminated (prefill on separate GPU) |
| Decode TPOT increase during prefill | Reduced but non-zero | Zero (decode GPU never runs prefill) |
| Resource coupling | Still coupled (same GPU, same TP config) | Decoupled (independent TP per phase) |
| Hardware optimisation | Cannot use different hardware per phase | Can use compute-optimised vs memory-optimised GPUs |
| TPOT variance | Still variable (chunk scheduling adds jitter) | Stable (decode GPU is dedicated) |

**The key limit**: even a 1-token prefill chunk requires exclusive GPU access for one forward step; decode is blocked during that step. DistServe measures that with chunked prefill, collocated systems still show **3–5× higher TPOT variance** vs disaggregated systems under production SLO constraints.

**In SGLang's PD disaggregation**: the prefill server uses chunked prefill via `--chunked-prefill-size` to limit step duration, ensuring the prefill server doesn't take arbitrarily long on very large prompts before KV transfer can begin.

**BibTeX:**
```bibtex
@inproceedings{agrawal2024sarathi,
  title     = {{SARATHI}: Efficient LLM Inference by Piggybacking Decodes
               with Chunked Prefills},
  author    = {Amey Agrawal and Ashish Panwar and Jayashree Mohan
               and Nipun Kwatra and Bhargav S. Gulavani and Ramachandran Ramjee},
  booktitle = {18th USENIX Symposium on Operating Systems Design and
               Implementation (OSDI 24)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2308.16369}
}
```

---

## Omission 6: TaiChi — SLO-Regime Framework and Hybrid Switching

**Source:** `L4/02_taichi_agg_vs_disagg.md`
**Why omitted from COMBINEDL1L2.md:** COMBINEDL1L2.md section 11 ("When Disaggregation Makes Things Worse") covers the failure modes at L1 level: short prompts, high prefix cache hit rates, and too few GPUs. The SLO-regime framework — when tight TTFT favours aggregation and tight TPOT favours disaggregation — and TaiChi's hybrid switching mechanism are L4 material. The framework is also the most rigorous answer to "when not to disaggregate", going deeper than the L1 decision heuristics.

---

### The Central Question

Since 2024, two camps have produced competing optimisations:
- **PD Aggregation** (Orca, SARATHI/Sarathi-Serve): co-locate, manage interference through chunked prefill.
- **PD Disaggregation** (DistServe, Splitwise, Mooncake, NVIDIA Dynamo): physically separate, eliminate interference entirely.

TaiChi settles the debate empirically and theoretically.

### The SLO-Regime Framework

**When PD Aggregation is Optimal — tight TTFT + relaxed TPOT:**
- All GPUs contribute to prefill simultaneously → low TTFT.
- TPOT violations from decode interference are tolerable.
- Aggregation achieves maximum GPU utilisation by batching prefill and decode together.
- Disaggregation would hurt TTFT because fewer GPUs handle prefill (only the prefill pool).

**When PD Disaggregation is Optimal — tight TPOT + relaxed TTFT:**
- Dedicated decode pool is never interrupted by prefill → stable TPOT.
- Higher TTFT is acceptable.
- Aggregation causes prefill interference to spike TPOT, violating the strict TPOT SLO.

**The Balanced SLO Problem — tight TTFT + tight TPOT:**
- PD aggregation: TPOT violations due to prefill interference.
- PD disaggregation: TTFT violations because fewer instances handle prefill.
- Neither can satisfy both SLOs at the same request rate.

### TaiChi's Solution: Hybrid-Mode Inference

TaiChi divides the GPU pool into dynamic categories:
- **P-heavy workers**: primarily process prefill batches.
- **D-heavy workers**: primarily process decode batches.

In **aggregation mode**: all workers process mixed P+D batches (SARATHI-style).
In **disaggregation mode**: fixed split; KV transferred between pools.
In **hybrid mode** (TaiChi's contribution): the P-heavy/D-heavy split is **dynamic** — adjustable in response to real-time SLO violation signals.

**SLO monitoring loop:**
```
Monitor TTFT violations → too many → shift some D-heavy workers to P-heavy
Monitor TPOT violations → too many → shift some P-heavy workers to D-heavy
At equilibrium: minimum workers of each type for both SLOs to be satisfied
```

### Results

| Metric | TaiChi vs State-of-the-Art |
|---|---|
| Goodput improvement | **Up to 77% over SOTA** |
| TTFT reduction vs PD disaggregation | **Up to 13.2×** (disaggregation has too-high TTFT when TTFT SLO is tight) |
| TPOT reduction vs PD aggregation | **Up to 1.69×** |

### The SLO Decision Matrix

| Workload type | TTFT constraint | TPOT constraint | Recommended approach |
|---|---|---|---|
| Chatbot (responsiveness matters) | Tight | Relaxed | PD Aggregation (SARATHI-style) |
| Code streamer (smooth output) | Relaxed | Tight | PD Disaggregation |
| Agentic AI (long chains, low latency) | Tight | Tight | TaiChi hybrid or large disaggregation cluster |
| Batch inference (offline) | Relaxed | Relaxed | Aggregation (maximise throughput) |
| RAG with long prompts | Moderate | Tight | PD Disaggregation |
| Multi-turn chat | Moderate | Moderate | TaiChi or cache-aware aggregation |

### PPD Disaggregation: Three-Way Split for Multi-Turn

TaiChi also introduces PPD (Prefill-Prompt-Decode) disaggregation for multi-turn serving — a **third worker type** (Prompt worker) specifically for context history KV loading/computation:

```
P worker (Prompt)  → handles context history KV loading/computation
P worker (Prefill) → handles new-token KV computation (append prefill)
D worker (Decode)  → handles autoregressive generation
```

In long multi-turn conversations, the context prefill (re-processing thousands of historical tokens) dominates prefill compute. Offloading it to a dedicated worker prevents it from blocking new-request prefills. This is not yet implemented in SGLang production mode.

**BibTeX:**
```bibtex
@article{wang2025taichi,
  title   = {Prefill-Decode Aggregation or Disaggregation?
             Unifying Both for Goodput-Optimized LLM Serving},
  author  = {Chao Wang and others},
  journal = {arXiv preprint arXiv:2508.01989},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.01989}
}
```

---

## Omission 7: vLLM — Connector Architecture and KV Transfer Abstraction

**Source:** `L4/03_vllm_disagg_connector.md`
**Why omitted from COMBINEDL1L2.md:** COMBINEDL1L2.md covers SGLang's launch commands (`--disaggregation-mode`, `--disaggregation-transfer-backend`) and the handshake protocol in enough detail to deploy and operate a cluster. vLLM's 6-connector design, `BaseKVConnector` abstraction, `KVLookupBufferBase`, and `--kv-transfer-config` schema are reference material for engineers writing or extending a connector — not for operators deploying one.

---

### Why vLLM's Implementation Matters

vLLM's disaggregated prefilling was the first open-source implementation of inter-instance KV transfer (building on Splitwise's prototype PR #2809). Its Connector/LookupBuffer abstraction is the reference design for how any inference framework should expose KV transfer. SGLang's PD disaggregation follows the same conceptual architecture (prefill node = KV producer, decode node = KV consumer) but with tighter SGLang-specific integration.

> **Important note from vLLM docs:** "Disaggregated prefill DOES NOT improve throughput. It improves latency SLO compliance and decouples TTFT from ITL." — This is the clearest canonical statement of what disaggregation optimises.

### The 6 Supported Connectors

| Connector | Transport | Use case |
|---|---|---|
| **NixlConnector** | RDMA InfiniBand/RoCEv2 via UCX | Default high-performance GPU-to-GPU transfer in production |
| **MooncakeConnector** | RDMA, NVLink, TCP via Mooncake TE | Multi-NIC pooling and topology-aware path selection |
| **P2pNcclConnector** | NCCL P2P (PCIe or NVLink) | Clusters without RDMA NICs; requires a proxy process |
| **LMCacheConnectorV1** | NIXL + LMCache storage | Cross-engine KV sharing + disaggregated prefilling unified |
| **ExampleConnector** | Reference implementation | Starting template for custom connectors |
| **MultiConnector** | Chains multiple connectors | RDMA → NCCL fallback; NIXL + LMCache combined |

### The BaseKVConnector Abstraction

```python
class BaseKVConnector(ABC):
    def send_kv_caches_and_hidden_states(
        self, model_executable, model_input, kv_caches, hidden_or_intermediate_states
    ) -> None: ...

    def recv_kv_caches_and_hidden_states(
        self, model_executable, model_input, kv_caches
    ) -> Tuple[torch.Tensor, bool]: ...

    def close(self) -> None: ...
```

- **`send_kv_caches`**: called on the prefill instance after the forward pass. Writes KV tensors to the transfer buffer.
- **`recv_kv_caches`**: called on the decode instance before its forward pass. Returns `(hidden_states, bypass_model_exec)` — if `bypass_model_exec=True`, decode skips its own prefill computation (it already has the KV).

### The KVLookupBufferBase Abstraction

```python
class KVLookupBufferBase(ABC):
    def insert(self, input_tokens, roi, key, value, hidden) -> None: ...
    def drop_select(self, input_tokens, roi) -> Tuple[...]: ...
    def close(self) -> None: ...
```

`insert`: prefill instance inserts KV cache into the buffer.
`drop_select`: decode instance **atomically** selects and removes the KV cache matching its request. "Drop" ensures each KV cache is consumed exactly once — prevents double consumption.

### The `--kv-transfer-config` JSON Schema

```python
class KVTransferConfig(BaseModel):
    kv_connector: str               # Connector class name
    kv_role: str                    # "kv_producer", "kv_consumer", or "kv_both"
    kv_rank: int = 0                # Rank within the transfer group
    kv_parallel_size: int = 2       # Total size of transfer group
    kv_buffer_size: float = 1e9    # Transfer buffer size in bytes
    kv_port: str = "14579"          # Port for connector communication
    kv_connector_extra_config: dict # Connector-specific configuration
```

**`kv_role`** is the most important field:
- `kv_producer`: this instance is the prefill worker (sends KV)
- `kv_consumer`: this instance is the decode worker (receives KV)
- `kv_both`: runs both phases — useful for testing or multi-turn scenarios

### vLLM vs SGLang Disaggregation Architecture

| Aspect | vLLM | SGLang |
|---|---|---|
| Transfer abstraction | `BaseKVConnector` (6 implementations) | `DisaggTransferBackend` (Mooncake, NIXL) |
| Configuration | JSON via `--kv-transfer-config` | CLI flags `--disaggregation-mode`, `--disaggregation-ib-device` |
| Router | External (Ray Serve, Dynamo, custom) | Built-in `sglang_router` |
| MoE support | General MoE | DeepEP + EPLB integration |
| Multi-connector chaining | MultiConnector | Not yet supported |

---

## Omission 8: Expert Parallelism Dispatch Internals (MoE Models)

**Source:** `L2/02_lmsys_deepseek_96h100.md`
**Why omitted from COMBINEDL1L2.md:** COMBINEDL1L2.md states that "PD disaggregation is mandatory for MoE models at this scale" and explains the core reason (different dispatch patterns for prefill vs decode). The DeepEP dispatch mode mechanics and EPLB load balancing algorithm are L3 material needed only by engineers adapting disaggregation for other MoE models.

---

### DeepEP Dispatch Mode Conflict

DeepSeek-V3 uses Mixture-of-Experts (MoE) with 256 experts per layer, only 8 activated per token. The expert parallelism communication library (DeepEP) uses two fundamentally different dispatch patterns:

**Normal mode** (required for prefill):
- High-throughput all-to-all for large batches
- Maximises expert computation overlap with communication
- Designed for compute-intensive, large-batch processing
- Incompatible with single-token decode steps

**Low-latency mode** (required for decode):
- Minimises dispatch latency for small batches (often single-token)
- Sacrifices throughput for responsiveness
- Designed for autoregressive generation
- Incompatible with the large prefill batches that benefit from "normal" mode

When prefill and decode run on the same GPU workers, the server must switch between modes per batch — which cannot be done optimally when the two types of batches are interleaved. The dispatch mode chosen for one phase degrades the other.

**PD disaggregation resolves this**: prefill nodes use "normal" mode permanently; decode nodes use "low-latency" mode permanently. Neither compromises the other.

For MoE models, this is not an optimisation — it is a **correctness requirement** for achieving optimal throughput.

### EPLB: Expert Parallelism Load Balancing

Expert load is inherently uneven — some experts are activated far more frequently than others. Without load balancing, the GPUs hosting popular experts become bottlenecks.

EPLB (Expert Parallelism Load Balancing) redistributes expert computation to balance GPU utilisation:
- Analyses the expert activation distribution from observed input/output data.
- Reassigns experts to GPUs to equalise activation frequency across workers.
- The SGLang DeepSeek deployment configures EPLB using a distribution matching the observed input/output data pattern.

Without EPLB, some GPUs in the expert parallel group would be idle while others are saturated — reducing the effective throughput of the expert parallelism configuration.

### The 3:9 Prefill-to-Decode Ratio

The DeepSeek-V3 deployment uses 3 prefill nodes (24 GPUs) and 9 decode nodes (72 GPUs). This 1:3 ratio is the production tuning point for this model at this scale.

**Why decode needs more resources than prefill:**
- Decode runs continuously — every token generation step runs on the decode pool.
- Prefill is bursty — processes a prompt once, then hands off.
- At steady state, far more GPU time is consumed by decode than by prefill.
- The ratio depends on the input/output token length ratio of actual workload: longer outputs → more decode GPU time → higher decode/prefill ratio.

For a workload with 2,000-token inputs and 500-token outputs (250% decode/prefill ratio by time), a 1:3 GPU ratio matches the compute demands roughly.

### SGLang CLI Flags for MoE Disaggregation

```bash
# Prefill node (normal DeepEP mode)
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-mode prefill \
  --moe-a2a-backend deepep \
  --enable-dp-attention \
  --tp-size 16 --dp-size 8 \
  ...

# Decode node (low-latency DeepEP mode)
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-mode decode \
  --moe-a2a-backend deepep \
  --enable-dp-attention \
  --tp-size 16 --dp-size 8 \
  --max-running-requests 128 \
  ...
```

The `--moe-a2a-backend deepep` flag activates DeepEP; `--enable-dp-attention` activates data-parallel attention. Both are required for the DeepSeek-V3 deployment configuration.
