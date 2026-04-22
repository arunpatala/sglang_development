# Prefill-Decode Disaggregation: From the Problem to the Production Stack

**Level:** L3 + L4 — concept-first narrative. Systems (DistServe, Splitwise, SARATHI, Mooncake, NIXL, Dynamo, TaiChi, vLLM) appear as examples of concepts, not as the organizing principle.

**Reading contract:** Each section creates a question the next section answers. The structure follows the problem chain, not the publication timeline.

---

## Section Map

| § | The question being answered | Key systems referenced |
|---|---|---|
| 1 | What are the two phases and why are they fundamentally different? | Splitwise traces |
| 2 | What goes wrong when you run both on the same GPU? | DistServe measurements |
| 3 | Can scheduling alone fix the interference? | SARATHI |
| 4 | What is the right metric to measure the damage? | DistServe (goodput) |
| 5 | What does separation actually give you? | DistServe, Splitwise |
| 6 | Can you use different hardware per phase to save money? | Splitwise |
| 7 | What new problem does separation introduce? | Mooncake, NIXL |
| 8 | How do you solve the transfer problem in practice? | Mooncake, NIXL, vLLM connectors |
| 9 | At scale, transfer isn't enough — what else breaks? | NVIDIA Dynamo |
| 10 | Is disaggregation always the right answer? | TaiChi |
| 11 | How does a real framework implement all of this? | SGLang PD mode |
| 12 | Does it actually work at production scale? | Perplexity, DeepSeek, Kimi |
| 13 | When should you disaggregate? The decision framework | Splitwise, TaiChi |

---

## 1. The Two Phases: Structurally Different Hardware Workloads

Every LLM request passes through two phases. They share the same model weights, the same GPU, and the same CUDA kernels — but they are on opposite ends of the hardware utilisation spectrum.

### Prefill: Compute-Bound

When a prompt arrives, the model processes the **entire input sequence in parallel**. For each token, in each attention layer, it computes Query, Key, and Value vectors. The K and V vectors for every input token are saved to GPU memory as the **KV cache**.

Prefill is dominated by large matrix multiplications across the full sequence length. For a 4,096-token prompt:

```
Prefill arithmetic intensity ≈ 200–400 FLOP/byte
GPU compute utilisation: 90–95%
GPU HBM bandwidth utilisation: low
```

The GPU's tensor cores are fully saturated. The HBM bus is barely touched. Prefill is **compute-bound** — it scales with FLOP throughput, not memory bandwidth.

### Decode: Memory-Bandwidth-Bound

After prefill, the model generates output tokens one at a time. Each decode step:
1. Reads the **entire KV cache** from HBM (grows with every token generated)
2. Computes attention over a batch of size 1 per active request
3. Appends the new token's K and V to the cache
4. Repeats

The attention computation per decode step is tiny (shape [1 × d_model]). But it requires reading potentially gigabytes of KV cache on every single step:

```
Decode arithmetic intensity ≈ 1 FLOP/byte (constant, regardless of sequence length)
GPU compute utilisation: 20–40%
GPU HBM bandwidth utilisation: maxed out
```

The GPU's tensor cores sit idle most of the time. Decode is **memory-bandwidth-bound** — it scales with HBM bandwidth, not FLOP throughput.

### Why This Matters (Validated in Production)

This asymmetry is not a theoretical observation. Splitwise (ISCA 2024) characterised both phases from **real Azure production traces** across two LLM services:

| | Prefill | Decode |
|---|---|---|
| Primary bottleneck | Compute (tensor cores) | Memory bandwidth (HBM reads) |
| GPU compute utilisation | High | Low — even with full continuous batching |
| Memory pressure | Low (transient activations only) | High (KV cache held for entire generation) |
| Duration | Short (ms to seconds) | Long (seconds to minutes) |

> "Token generation phases do not require the compute capability of the latest GPUs and can be run on lower-power, lower-cost hardware with equivalent quality." — Splitwise, ISCA 2024

This is the physical foundation of everything that follows. Two workloads, opposite hardware requirements, running on the same silicon. The question is: *what happens when you make them share?*

---

## 2. What Happens When You Mix Them: The Interference Problem

When prefill and decode share the same GPU pool, the structural asymmetry becomes a **structural conflict**.

### The Mechanism

When a new request arrives and prefill begins:
1. The GPU's tensor cores are saturated — hundreds of milliseconds of compute-intensive work
2. All active decode steps stall: they can't run during prefill
3. Users watching a streaming response see the text **pause mid-sentence**

When many decode requests fill the queue:
1. New prefill requests queue behind them
2. Even though the server isn't at capacity, new users wait
3. TTFT climbs even at moderate load

The two SLOs **pull against each other**. Making one better makes the other worse:

```
To reduce TTFT → prioritise prefill → decode requests stall → ITL spikes
To reduce ITL  → throttle prefill  → new requests queue  → TTFT climbs
```

No scheduling policy resolves this — both phases are competing for the same hardware resource at the same time.

### The Quantified Damage

DistServe (OSDI 2024) measured the interference effect directly under production-representative SLOs:

- **TPOT variance**: collocated systems show **3–5× higher TPOT variance** vs disaggregated, even when chunked prefill (SARATHI) is applied
- **ITL spike correlation**: decode ITL spikes with prefill batch size — a clear positive linear correlation measured in production-representative workloads
- **TTFT drift**: as the decode queue fills, TTFT rises even below full server capacity — the prefill can't get CPU scheduling time

### The Second Problem: Resource Coupling

Interference is a scheduling problem. Resource coupling is a hardware problem. They are separate, and scheduling cannot fix the second one.

A single GPU has a fixed ratio of FLOP throughput to HBM bandwidth. The optimal **tensor parallelism degree** (TP) that minimises per-request TTFT is different from the TP that maximises decode throughput. In a collocated system, you choose one TP configuration and both phases are suboptimal:

- Low TP → faster individual prefills, but decode workers don't fully utilise HBM bandwidth
- High TP → better decode throughput, but each prefill request incurs more all-reduce communication overhead

You cannot have both simultaneously on the same pool.

---

## 3. Can Scheduling Alone Fix It? The Aggregation Baseline

Before reaching for a full architectural redesign, the natural question is: *can we fix this with better scheduling?* SARATHI (OSDI 2024) represents the most rigorous answer to that question.

### The Idea: Chunked Prefill

Instead of processing a large prefill in one uninterrupted step (which blocks all decode requests for its entire duration), SARATHI splits prefill into **fixed-size chunks** and interleaves them with decode steps:

```
Standard batching:
  [PPPPPPPPPPPP ← 12 prefill tokens blocking everything → ] [D][D][D][D][D][D]

SARATHI chunked prefill:
  [PPPP + D D D D] → [PPPP + D D D D] → [PPPP + D D D D]
   chunk 1 + decodes    chunk 2 + decodes    chunk 3 + decodes
```

Each batch contains one prefill chunk (enough to saturate GPU compute for that step) plus as many concurrent decode requests as fit in the remaining GPU capacity. Decode requests "piggyback" on the gaps.

**What this achieves:**
- Head-of-line blocking is reduced from (full prefill duration) to (one chunk duration)
- Pipeline bubbles from variable-length prefill microbatches are nearly eliminated
- Measured: **up to 10× decode throughput improvement** for LLaMA-13B; **1.91× pipeline bubble reduction** at PP=8

### What It Cannot Fix

SARATHI is the right solution for a specific regime. But it has hard limits:

| Problem | SARATHI | Verdict |
|---|---|---|
| Head-of-line blocking | Reduced to one chunk duration | Partial fix |
| Decode ITL during prefill | Still exists — decode waits for each chunk | **Cannot eliminate** |
| Resource coupling (FLOP/HBM ratio) | Unchanged — same GPU, same TP config | **Cannot fix** |
| Per-phase hardware optimisation | Impossible — one pool, one config | **Cannot fix** |
| TPOT variance | Reduced but non-zero (chunk scheduling adds jitter) | Partial fix |

DistServe measured **3–5× higher TPOT variance** with chunked prefill vs disaggregated systems. Even a 1-token prefill chunk requires exclusive GPU access for one forward step — at high prefill rates, residual interference remains.

**The conclusion:** SARATHI is the best aggregation can do. If it's not enough — and for many production workloads it isn't — the problem is architectural, not schedulable.

> Note: SARATHI and disaggregation are **complementary**. SGLang's prefill server uses `--chunked-prefill-size` alongside disaggregation to prevent large individual prompts from delaying KV transfers.

---

## 4. The Right Metric: Why Throughput Lies

Before examining the disaggregated solution, it's worth being precise about what we're optimising. The standard metric — *requests per second* — is the wrong target. It hides exactly the problem we're trying to solve.

### The Problem with Raw Throughput

A system can serve many requests per second while violating the latency SLOs that users actually experience. Consider:
- System A: 100 req/s, but 30% of requests exceed TTFT SLO or ITL SLO
- System B: 60 req/s, all requests within both SLOs

System A has higher throughput. System B has higher **goodput** — the metric DistServe introduced.

### Goodput: The Correct Target

> "Goodput = number of requests that satisfy **both** the TTFT SLO **and** the TPOT SLO per unit time."

A request contributes to goodput only if:
- TTFT ≤ TTFT\_SLO (response started fast enough), **AND**
- TPOT ≤ TPOT\_SLO (each token streamed fast enough)

Both conditions simultaneously. Either violation disqualifies the request from goodput.

```
                  TPOT_SLO
                      │   ┌──────────────────────────────┐
                      │   │                              │
                      │   │   Requests that count toward │
                      │   │         GOODPUT              │
                      │   │                              │
                      ├───┤                              │
                      │   └──────────────────────────────┘
                      └──────────────────────────────────── TTFT_SLO
```

In a collocated system under load, TTFT and TPOT SLO violations happen simultaneously — interference pushes requests out of the box in both dimensions at once. Disaggregation eliminates the interference, fitting many more requests inside the box.

**This is why DistServe's 7.4× result is measured in goodput, not throughput.** Under raw throughput comparison, the improvement is smaller. Goodput under strict SLO constraints tells the story users actually experience.

---

## 5. The Solution: Separate the Phases

Disaggregation is the architectural response to a coupling problem: if two workloads have incompatible hardware requirements and share resources, separate the resources.

### The Architecture

Three components replace the monolithic serving pool:

**The Router** — single client-facing entry point. Routes each request to an available prefill worker; after prefill completes, routes the KV cache to a decode worker. Keeps no model state.

**The Prefill Pool** — GPUs configured for high FLOP throughput. Process prompts, fill KV caches, hand them off. Never generate output tokens. Can batch many prompts simultaneously.

**The Decode Pool** — GPUs configured for HBM bandwidth and capacity. Receive KV caches, run autoregressive generation. Maintain KV caches for the full duration of each request.

```
Client → Router → Prefill Worker → [KV cache over network] → Decode Worker → tokens → Client
```

### What Separation Unlocks

**Independent TP configuration:** prefill workers use a TP degree optimised for compute throughput (fewer all-reduce calls → lower per-request TTFT). Decode workers use a TP degree optimised for HBM utilisation (more shards → each shard reads less HBM per step → lower ITL). These are different numbers. Impossible to optimise simultaneously in a collocated pool.

**Independent scaling:** TTFT SLO violated? Add prefill workers. TPOT SLO violated? Add decode workers. In a collocated system, adding capacity addresses both SLOs simultaneously — expensive overprovisioning.

**Independent batching strategy:** prefill workers batch similar-length prompts for efficient compute utilisation. Decode workers maximise concurrent request count to amortise HBM reads.

### The Measured Payoff

DistServe evaluated against vLLM with continuous batching (the collocated baseline), across OPT-13B to OPT-175B, on chatbot, summarisation, and coding workloads:

| Metric | Result |
|---|---|
| Goodput at tight SLO | **7.4× more requests/sec** |
| Tightest SLO achievable at same rate | **12.6× tighter** |

The 7.4× comes entirely from removing the coupling constraint — each pool now runs its optimal configuration 100% of the time, with no compromise.

---

## 6. Separation Unlocks Hardware Heterogeneity

Once you have separate pools, you are no longer required to use the same GPU SKU for both phases. This is Splitwise's insight — and it has direct cost consequences.

### The Mismatch

An H100 SXM costs ~$30k and delivers 80 TFLOPS FP16 and 3.35 TB/s HBM3 bandwidth. During decode, the 80 TFLOPS goes almost entirely unused — decode arithmetic intensity is ~1 FLOP/byte, nowhere near what justifies H100's compute price. You are paying for 80 TFLOPS and using 3.35 TB/s.

An A100 costs ~$15k, delivers 77 TFLOPS FP16 and 2.0 TB/s HBM2e. For a decode-only worker, its utilisation profile is nearly identical to H100. You pay half, get ~60% of the decode throughput. The ratio is favourable.

### The Hardware Assignment

| Phase | Optimal choice | Why |
|---|---|---|
| Prefill | H100, B200 — highest FLOP/dollar | Compute-bound; tensor cores are the bottleneck |
| Decode | A100, H20 — highest HBM BW/dollar at lower cost | Memory-bandwidth-bound; HBM bandwidth is the bottleneck, FLOP budget is irrelevant |

**Validated by experiment** (H100 prefill + A100 decode vs all-H100, same total cost budget):

| Scenario | Result |
|---|---|
| Same cost budget | **2.35× more throughput** |
| Same throughput target | **1.4× more throughput at 20% lower cost** |

SPAD (UT Austin) validated the hardware insensitivity directly:
- Reduce memory bandwidth by 40% on the prefill chip → only **17% TTFT increase** (prefill doesn't need bandwidth)
- Reduce compute capacity by 50% on the decode chip → only **22% ITL increase** (decode doesn't need compute)

### In Practice (AWS)

| Pool | Instance | GPU | What you're paying for |
|---|---|---|---|
| Prefill | p5.48xlarge | 8× H100 (HBM3) | 8× 3,200 Gbps EFA, highest FP8 FLOPS |
| Decode | p4de.24xlarge | 8× A100 80GB (HBM2e) | 640 GB total KV capacity, 2.0 TB/s BW, half the cost/GPU |

The Splitwise insight maps directly to a practical AWS cluster design — and it saves real money at scale.

---

## 7. The New Problem: Moving the KV Cache

Disaggregation introduces a problem that didn't exist before: the KV cache must travel across a network from the prefill GPU's VRAM to the decode GPU's VRAM. This is the **transfer tax** — the price of separation.

### How Large Is It?

```python
kv_bytes_per_token = n_layers × n_kv_heads × head_dim × 2 (K and V) × bytes_per_element

# LLaMA-3.1-70B (GQA), FP16, 4,096-token prompt:
# 80 layers × 8 heads × 128 dim × 2 × 2 bytes = 327,680 bytes/token
# × 4,096 tokens = 1.34 GB per request

# LLaMA-3.1-70B at 128K context: ~40 GB per request
# DeepSeek-V3 at 8K context: ~2.1 GB per request (MLA architecture is smaller)
```

For a 500 ms TTFT budget with 200 ms prefill time, you have 300 ms to transfer the KV cache. That sets your minimum required network bandwidth:

```
Required bandwidth = 1.34 GB / 0.3 s ≈ 4.5 GB/s (minimum, 4K-token LLaMA-70B)
```

### The Network Speed Ladder

| Technology | Bandwidth | Transfer Time (1.34 GB) | Viable? |
|---|---|---|---|
| 10 GbE TCP | 1.25 GB/s | 1.07 s | No |
| 100 GbE TCP (CPU-mediated) | ~6 GB/s | **220 ms** | Borderline |
| InfiniBand HDR (GPUDirect) | 25 GB/s | **54 ms** | Yes |
| InfiniBand NDR (GPUDirect) | 50 GB/s | **27 ms** | Yes |
| NVLink (within one server) | 900 GB/s | **1.5 ms** | Free |
| NVLink (NVL72 rack) | ~1,000 GB/s | **~1 ms** | Free |

**The critical enabler: GPUDirect RDMA.** Without it, the KV transfer path crosses the CPU twice:

```
Without GPUDirect:
  Prefill GPU VRAM → PCIe → CPU RAM → NIC → network → NIC → CPU RAM → PCIe → Decode GPU VRAM
  Effective bandwidth on 100 GbE: ~6 GB/s (CPU and PCIe become bottlenecks)

With GPUDirect RDMA:
  Prefill GPU VRAM → NIC → network → NIC → Decode GPU VRAM
  Effective bandwidth: wire rate (~12.5 GB/s on 100 GbE, ~50 GB/s on NDR IB)
```

GPUDirect RDMA is not optional — it's what makes disaggregation fast enough to be worth the transfer cost. This is why the entire transfer engineering layer (Mooncake, NIXL) is built on top of it.

### The Amortisation Threshold

The transfer overhead is a fixed cost per request. The benefit (eliminating interference) scales with output length. Splitwise derived the break-even point: **prompts longer than ~500 tokens generating more than ~50 output tokens** — disaggregation is always net-positive. Below this threshold, the transfer overhead exceeds the interference-elimination gain.

---

## 8. Solving the Transfer Problem: Three Approaches

The transfer problem has two sub-problems:
1. **Physical transport**: how to move data across the network as fast as possible (GPUDirect, NIC selection, multi-NIC)
2. **Framework abstraction**: how to expose KV transfer as a reusable interface for inference frameworks (vLLM, SGLang, TensorRT-LLM)

The ecosystem has produced three complementary answers.

### Approach A: The Production Transfer Engine (Mooncake)

Mooncake (FAST 2025) is Moonshot AI's production transfer engine for Kimi. It treats KV transfer as a first-class systems problem and solves each sub-problem explicitly.

**Problem: NIC–GPU affinity is not uniform.**
In modern inference servers, not all NIC–GPU paths are equal. A NIC on the same PCIe root complex as the GPU provides full bandwidth; a NIC behind a PCIe/UPI bridge provides half or less.

Mooncake's solution: **topology-aware NIC selection**. On startup, each server generates a topology matrix mapping GPU–NIC affinity and NUMA distances. For each transfer, the engine selects the NIC with the best path to the source GPU. Skipping this (naive NIC selection) can halve effective bandwidth — a common deployment error in early disaggregation setups.

**Problem: A single NIC is a single bandwidth limit.**
One NDR InfiniBand NIC provides ~50 GB/s. Modern servers have 4–8 NICs.

Mooncake's solution: **multi-NIC pooling**. Large transfers are internally sliced; each slice is assigned to a different NIC based on topology affinity. All slices run in parallel. For 8-NIC servers, effective aggregate bandwidth approaches 8× the per-NIC limit — enough for even very large KV caches.

**Problem: InfiniBand exists alongside NVLink — different technologies for different topologies.**
For GB200 NVL72 rack-scale deployments, 72 GPUs are interconnected at rack-scale NVLink speeds (~1,000 GB/s), far exceeding any InfiniBand NIC.

Mooncake's solution: pluggable transport backends.

```bash
# Select InfiniBand (default, inter-node)
--disaggregation-ib-device mlx5_0

# Switch to NVLink for NVL72 intra-rack transfers
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK
export MC_FORCE_MNNVL=True
```

**Production results:**
- Kimi service: 75% more requests under SLO adherence; up to **525% throughput increase** for long-context workloads
- Kimi K2 (128 H200 GPUs, July 2025): **224,000 tokens/second prefill**, **288,000 tokens/second decode**

### Approach B: The Vendor-Agnostic Library (NIXL)

NIXL (NVIDIA Inference Xfer Library) takes a different design goal: work with any high-speed interconnect, expose a single unified API, and let the framework not care about the underlying transport.

Each server runs one NIXL **Transfer Agent** managing:
- A unified memory section (GPU VRAM, CPU DRAM, NVMe — same buffer-list API regardless of location)
- A pluggable transport backend (UCX for RDMA, GDS for NVMe→GPU, TCP as fallback, NVLink for within-rack)
- A metadata handler (etcd-cached endpoint registration, avoiding per-transfer round-trips)

The key capability that Mooncake also implements but NIXL exposes explicitly: **async non-blocking transfer**.

```python
# Submit transfer, don't wait
handle = agent.transfer_submit_write(src_addr, dst_addr, size)

# Check later, while compute continues
status = agent.transfer_check_status(handle)
```

This allows the decode worker to begin processing layers that have already arrived while the remaining layers are still in transit — the compute-transfer overlap that Perplexity implements manually with layer-pipelined KV transfer, now available as a library primitive.

Three use cases, one library:
- Disaggregation: KV blocks, prefill GPU VRAM → decode GPU VRAM (RDMA zero-copy)
- Long-context loading: KV blocks from NVMe/object storage → GPU VRAM (GDS path)
- MoE expert parallelism: all-to-all expert activations across GPUs (NVLink/RDMA)

### Approach C: The Framework Abstraction (vLLM Connectors)

If Mooncake is the production engine and NIXL is the library, vLLM's connector architecture is the **interface specification** — the answer to "what must a KV transfer system expose to an inference framework?"

The key insight from vLLM's design:

```python
class BaseKVConnector(ABC):
    def send_kv_caches_and_hidden_states(...) -> None: ...        # called on prefill worker
    def recv_kv_caches_and_hidden_states(...) -> (tensor, bool):  # called on decode worker
```

The `recv` return value `bypass_model_exec: bool` is the critical design decision. When `True`, the decode worker **skips its own prefill computation entirely** — it has the KV from the transfer. The framework doesn't need to know which connector is underneath; it just checks this flag.

vLLM ships 6 concrete connectors, each answering the transfer problem for a different infrastructure:

| Connector | Transport | When to choose |
|---|---|---|
| NixlConnector | RDMA (UCX/InfiniBand) | Production clusters with RDMA NICs |
| MooncakeConnector | RDMA + NVLink + multi-NIC topology | Multi-NIC pooling + topology-aware routing |
| P2pNcclConnector | NCCL P2P (PCIe/NVLink) | No RDMA NICs — use NCCL as fallback |
| LMCacheConnectorV1 | NIXL + persistent KV storage | Cross-engine KV sharing + storage |
| MultiConnector | Chains any combination | RDMA primary + NCCL fallback |
| ExampleConnector | Reference implementation | Starting point for custom connectors |

The 6 connectors collectively define the full design space: RDMA, NVLink, NCCL, storage-backed, chained, and custom. Any inference framework needing to add disaggregation support can start from this interface.

> **The canonical statement from vLLM docs:** "Disaggregated prefill DOES NOT improve throughput. It improves latency SLO compliance and decouples TTFT from ITL."

---

## 9. Beyond Transfer: The Orchestration Problem at Scale

With transfer solved, a new class of problems emerges at production scale: **orchestration**. How does the system route requests intelligently? How does it know a worker died? How does it react when TTFT SLOs start slipping without a human restarting processes?

These are not transfer problems — they are coordination problems. And they require separate infrastructure.

NVIDIA Dynamo (GTC 2025) is the clearest statement of what this infrastructure needs. Its central claim: disaggregated serving requires not just a transfer engine but **a separate plane for each concern**.

### The Four-Plane Architecture

**Request Plane** (the data path — what every disaggregated system already has):
```
Client → Router → Prefill Worker → [KV] → Decode Worker → tokens → Client
```

**Discovery Plane** (etcd lease-based worker registration):

Without dynamic discovery, every time you add a worker or one crashes, you manually restart the router. At 50+ workers, this is operationally untenable.

Dynamo's solution: workers publish to etcd at startup with a 10-second TTL lease. Worker death → lease expires → router automatically removes it. New workers serve within one lease period. No manual reconfiguration.

**Event Plane** (asynchronous state propagation):

The router needs to know which prefill worker has which prefix cached. Workers need to know when SLO violations are occurring. Neither can poll every peer continuously at scale.

Dynamo's solution: a pub/sub event bus. Prefill workers publish KV cache state; the router subscribes. Monitoring publishes SLO violation rates; the Planner subscribes. Changes propagate without polling.

**Control Plane (the Planner)** (dynamic pool sizing based on live SLO signals):

TTFT P95 rises above the SLO threshold → the Planner issues a scale-up command to the prefill pool. TPOT violations appear → the Planner grows the decode pool. A feedback control loop with configurable reaction time and cooldown handles the dynamics.

SGLang's current disaggregation has no equivalent — scaling is manual. Dynamo is what production-grade autoscaling looks like.

### KV-Aware Routing: The Router's Hidden Leverage

The Request Plane's router is not just a load balancer. Dynamo's router tracks **which token prefixes are cached on which prefill workers** and routes each incoming request to the worker with the highest prefix overlap — skipping recomputation of already-cached context.

In a test with 100K requests to DeepSeek R1-Distilled Llama-70B FP8 on 2 H100 nodes, KV-aware routing measurably reduces TTFT by routing repeat prefixes to workers that already have the KV cached. SGLang's router currently uses round-robin; prefix-aware routing is a planned upgrade.

The combined result (disaggregation + NVLink-speed transfer + KV-aware routing + dynamic scaling on GB200 NVL72): **up to 30× more requests** vs collocated H100.

---

## 10. Is Disaggregation Always the Right Answer?

Everything so far has argued for disaggregation. But TaiChi (arXiv, August 2025) asks the question the other papers don't: *when is aggregation better?*

The answer turns out to be well-defined: **it depends on which SLO is the binding constraint.**

### The SLO Regime Analysis

**Tight TTFT + relaxed TPOT (e.g., a chatbot that must start responding within 1 second):**

Disaggregation hurts TTFT here. With separate pools, only the prefill pool handles new requests. In a collocated system, all GPUs can contribute to prefill, yielding lower TTFT at lower request rates.

→ **Aggregation is better**

**Tight TPOT + relaxed TTFT (e.g., a code generator streaming long outputs, or a batch API):**

Disaggregation eliminates all prefill interference on the decode pool. TPOT is perfectly stable. Users or systems can tolerate waiting slightly longer for the first token.

→ **Disaggregation is better**

**Tight TTFT + tight TPOT (agentic workflows, real-time coding assistants):**

Aggregation causes TPOT violations from interference. Disaggregation causes TTFT violations from smaller effective prefill capacity. Neither approach satisfies both SLOs at the same request rate.

→ **Neither is optimal; a hybrid is required**

### TaiChi's Solution: Dynamic Worker Reassignment

TaiChi implements a continuous monitoring loop that reassigns workers between P-heavy (prefill-focused) and D-heavy (decode-focused) categories based on live SLO violation signals:

```
Monitor TTFT violations → too many → shift some D-heavy workers to P-heavy
Monitor TPOT violations → too many → shift some P-heavy workers to D-heavy
At equilibrium: minimum workers of each type to satisfy both SLOs simultaneously
```

This is not a static configuration — it's a feedback controller that responds to traffic patterns in real time.

**Measured results** (on DeepSeek-R1, Llama-70B):

| Scenario | TaiChi vs state-of-the-art |
|---|---|
| Tight TPOT SLO | **Up to 77% better goodput** over SOTA |
| Tight TTFT SLO | **Up to 13.2× lower TTFT** vs pure disaggregation |
| Tight TPOT SLO | **Up to 1.69× lower TPOT** vs pure aggregation |

### The SLO Decision Map

| Workload | TTFT | TPOT | Right choice |
|---|---|---|---|
| Responsive chatbot | Tight | Relaxed | Aggregation (SARATHI-style) |
| Code streamer | Relaxed | Tight | **Disaggregation** |
| Agentic AI | Tight | Tight | TaiChi hybrid or large disagg cluster |
| Batch inference | Relaxed | Relaxed | Aggregation (maximise throughput) |
| RAG with long prompts | Moderate | Tight | **Disaggregation** |
| MoE models (DeepSeek-V3) | — | — | **Disaggregation** (mandatory — see §12) |

---

## 11. How SGLang Implements It: The Concrete Picture

The preceding sections describe concepts. Here is how those concepts materialise in SGLang's disaggregated serving mode.

### The Handshake Protocol (Correctness First)

The naive implementation has a race condition: the prefill worker finishes and tries to write KV to the decode worker, but the decode worker has no pre-allocated space. SGLang solves this with a **pre-allocation handshake**:

```
Step 1: Decode server pre-allocates KV cache pages for this request
         → sends KV page indices to Prefill server

Step 2: Prefill server receives indices
         → runs model forward pass
         → writes KV cache directly into Decode server's pre-allocated pages (RDMA)
         → sends completion notification

Step 3: Decode server receives KV (already in its VRAM at the pre-allocated indices)
         → begins autoregressive generation immediately
         → streams output tokens to client
```

The decode server allocates **before** prefill runs. No race, no retry, no buffer overflow.

### Launching a Cluster (Minimal Setup)

```bash
# 1. Prefill worker (GPU 0)
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_roce0 \
  --port 30000

# 2. Decode worker (GPU 1)
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --disaggregation-ib-device mlx5_roce0 \
  --port 30001 --base-gpu-id 1

# 3. Router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 8000
```

### Critical Configuration Knobs

| Flag / Variable | What it controls | Default |
|---|---|---|
| `--disaggregation-mode` | `prefill` or `decode` — which role this worker plays | — |
| `--disaggregation-transfer-backend` | `mooncake` or `nixl` — which transfer engine | `mooncake` |
| `--disaggregation-ib-device` | Which RDMA NIC to use (`mlx5_0`, `mlx5_roce0`) | Auto |
| `--max-running-requests` | Cap on concurrent decode requests — prevents KV OOM | Unlimited |
| `--moe-a2a-backend deepep` | Activates DeepEP for MoE models (DeepSeek-V3/V2) | Disabled |
| `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK` | Switch transfer to NVLink (NVL72 intra-rack) | RDMA |
| `SGLANG_DISAGG_STAGING_BUFFER=1` | Enable staging buffer for heterogeneous TP (non-MLA) | Disabled |

### Why MoE Models Require Disaggregation

For DeepSeek-V3 and other MoE models, disaggregation is not optional — it's mandated by the expert parallelism library (DeepEP). The prefill phase requires "normal mode" dispatch (high-throughput, large batches). The decode phase requires "low-latency mode" dispatch (minimal latency, single-token batches). These modes are **mutually exclusive** — a single worker cannot optimally serve both phases. Only by running prefill and decode on separate servers can each phase use its optimal dispatch mode.

`--disaggregation-mode prefill` → DeepEP automatically uses normal mode
`--disaggregation-mode decode` → DeepEP automatically uses low-latency mode

No additional flag needed. The mode is selected from the worker's role.

---

## 12. Does It Work at Production Scale?

### Perplexity AI: 435M Queries/Month (Dec 2024)

Perplexity AI deploys disaggregated serving for 435M+ search queries per month across 20+ AI models on H100 SXM GPUs. This is the largest publicly confirmed production deployment of PD disaggregation.

> "This technique significantly boosts overall system throughput while meeting SLAs, translating to lower cost per token. Additionally, this technique gives Perplexity the flexibility to use **different NVIDIA GPU products for each inference phase** given its specific hardware resource requirements." — NVIDIA Spotlight, December 2024

Both major insights — interference elimination and hardware heterogeneity — are confirmed in production at scale. Estimated cost savings on the Related-Questions feature alone: ~$1M/year.

### DeepSeek-V3 on 96 H100 GPUs (May 2025)

The LMSYS/SGLang team deployed DeepSeek-V3 (671B MoE) disaggregated across 12 nodes × 8 H100s, with a **3:9 prefill:decode ratio** (24 prefill GPUs, 72 decode GPUs).

| Metric | Value |
|---|---|
| Prefill pool | 3 nodes (24 H100 GPUs) |
| Decode pool | 9 nodes (72 H100 GPUs) |
| Transfer | Mooncake + InfiniBand |
| Input throughput | **52,300 tokens/second per node** |
| Output throughput | **22,300 tokens/second per node** |

> "To the best of our knowledge, this is the highest reported throughput for DeepSeek-V3 serving at that time." — SGLang Team

The 3:9 ratio reflects the arithmetic: for this workload, decode GPUs were the throughput-limiting resource. Adding more decode capacity (9 nodes) while keeping prefill smaller (3 nodes) matched the observed workload distribution.

### Moonshot AI — Kimi K2 (July 2025)

| Metric | Value |
|---|---|
| Hardware | 128 H200 GPUs |
| Transfer | Mooncake Transfer Engine + InfiniBand |
| Prefill | **224,000 tokens/second** |
| Decode | **288,000 tokens/second** |

The Kimi service (predecessor to Kimi K2) measured **75% more requests** handled under SLO adherence and **up to 525% throughput increase** for long-context workloads vs the collocated baseline.

---

## 13. The Decision: When to Disaggregate and at What Cost

With the full problem chain understood, the decision reduces to 5 checks and a cost formula.

### The 5 Checks

**Check 1 — Measure your decode/prefill time ratio**
What fraction of wall-clock GPU time is spent in decode vs prefill? If decode < 70%: smaller payoff from disaggregation. If decode > 85%: you're paying for idle tensor cores for most of each request's lifetime.

**Check 2 — Check your SLO regime (TaiChi's insight)**
Is your binding constraint TTFT, TPOT, or both? Tight TTFT only → disaggregation may hurt you (fewer GPUs handle prefill). Tight TPOT → disaggregation is the right fix. Both tight → hybrid or larger disaggregated cluster.

**Check 3 — Calculate your transfer cost**
Median prompt length → KV cache size (formula in §7) → required bandwidth for your TTFT budget. Do you have RDMA-capable NICs? Run KVBench against your actual cluster. If bandwidth < required → not viable without better network hardware.

**Check 4 — Audit your scale**
Below ~16 GPUs: scheduling overhead typically exceeds utilisation gain. Above 32 GPUs with sustained traffic: cost savings compound and the case strengthens.

**Check 5 — Check your prefix cache hit rate**
If prefix cache hit rate > 80% (multi-turn conversations, repeated system prompts), the decode worker already holds most of the KV from previous turns. Running prefill locally on the decode worker for high-hit requests may outperform routing every request through the prefill pool.

### The Cost Arithmetic

Disaggregation reduces per-token cost through two mechanisms:

1. **Interference elimination** → more requests complete within SLO → higher goodput at same hardware → lower effective cost/request
2. **Hardware heterogeneity** → decode pool runs on cheaper GPUs (A100 vs H100) → 20% lower cluster cost at same throughput (Splitwise)

Combined, Splitwise shows **2.35× more throughput at same cost** or **20% cost reduction at same throughput**. InfoQ's cluster-level analysis puts the total infrastructure savings at **15–40%** at scale.

The adoption timeline confirms the economics: DistServe published in 2024; within 18 months, disaggregated serving was in production at Perplexity, Meta, LinkedIn, Mistral, Moonshot AI, and HuggingFace. NVIDIA built Dynamo around it as a first-class serving architecture. Teams that disaggregated early and designed their network topology accordingly are measurably paying less per token.

---

## Key Quotes

> "Since prefill primarily determines the TTFT and decode impacts ITL, collocating them makes it difficult to optimize both metrics simultaneously." — BentoML

> "A Llama 70B model running inference on an H100 GPU hits 92% compute utilization during prefill. Thirty milliseconds later, during decode, that same GPU drops to 30%." — TDS, April 2026

> "Goodput = number of requests that meet both the TTFT SLO and the TPOT SLO per unit time." — DistServe / Hao Zhang, CMU

> "Token generation phases do not require the compute capability of the latest GPUs." — Splitwise, ISCA 2024

> "Disaggregated prefill DOES NOT improve throughput. It improves latency SLO compliance and decouples TTFT from ITL." — vLLM docs

> "To the best of our knowledge, this is the highest reported throughput for DeepSeek-V3 serving at that time." — SGLang Team, May 2025
