# PD Disaggregation — Omitted Material

**What this file is:** The full text of every section omitted from `COMBINED.md`. The appendix of `COMBINED.md` names each omission and explains why it was excluded. This file preserves the complete original text so no source material is lost.

**Sources:** L2/01, L2/02, L3/01, L3/02, L3/03, L3/04, L4/01, L4/02, L4/03

---

## Omission 1: Mooncake — Disaggregated KV Cache Pool (Beyond the Transfer Engine)

**Source:** `L3/03_mooncake_fast25.md`
**Why omitted from COMBINED.md:** COMBINED.md covers Mooncake as a KV transfer engine (the component directly relevant to SGLang's `--disaggregation-ib-device`). Mooncake's second major contribution — the disaggregated KV cache pool across CPU DRAM and SSD — overlaps with Layer 17 (Tiered KV Cache). Including it in COMBINED.md would blur the boundary between Layer 17 and Layer 19.

### Original text: The Disaggregated KVCache Pool

Mooncake adds a second tier of disaggregation: beyond separating prefill/decode clusters, it builds a **distributed KVCache pool** from underutilised CPU DRAM and SSD resources across the GPU cluster.

The pool is accessed by both prefill and decode instances. Its primary use is persistent KV prefix caching across requests:

1. Prefill instance computes KV for a common prefix (system prompt).
2. KV is written to the pool.
3. Next request with the same prefix: prefill instance reads from the pool → skips re-computation.

This is functionally identical to SGLang's HiCache (`--hicache-storage-backend mooncake`), but implemented as a cluster-wide service rather than a node-local cache.

**Key difference from HiCache:**

| Aspect | HiCache (Layer 17) | Mooncake KVCache Pool (Layer 19) |
|---|---|---|
| Scope | Single node (GPU + CPU + storage) | Cluster-wide (across all nodes) |
| Access pattern | Prefix-tree lookup (RadixTree) | Hash-based key-value lookup |
| KV granularity | Page (configurable, default 16 tokens) | Page (similar) |
| Sharing | Local to one SGLang instance | Shared across all prefill and decode instances |
| Network | Local DRAM + local NVMe | RDMA across nodes |

### Prediction-Based Early Rejection

Under highly overloaded conditions, Mooncake's KVCache-centric scheduler uses a prediction model to estimate whether an incoming request can be served within its SLO before accepting it. If the prediction indicates the SLO will be violated, the request is rejected immediately (429 response) — freeing headroom for requests that can be served.

This prediction uses:
- Current queue depths in prefill and decode clusters.
- Estimated transfer bandwidth to the decode cluster.
- Historical TTFT/TPOT distributions.

This is not implemented in SGLang's disaggregation mode — SGLang currently accepts all requests and queues them.

---

## Omission 2: Splitwise — SplitwiseSim Simulator Details

**Source:** `L3/02_splitwise_isca24.md`
**Why omitted from COMBINED.md:** SplitwiseSim is a research tool for evaluating routing policies offline. COMBINED.md focuses on production deployment architecture. The simulator is relevant for researchers designing new routing algorithms but not for practitioners deploying disaggregated serving.

### Original text: SplitwiseSim

SplitwiseSim (https://github.com/Mutinifni/splitwise-sim) is a discrete-event simulator for cluster-level PD disaggregation evaluation. It models:

- **Heterogeneous hardware pools**: configure different GPU types per phase (e.g., H100 for prefill, A100 for decode).
- **Workload distributions**: replay real or synthetic prompt/output length distributions (Splitwise's two Azure traces are included).
- **Routing policies**: evaluate round-robin, load-aware, and cache-aware routing at cluster scale without a physical cluster.
- **KV transfer latency**: parameterised by payload size and network topology (NVLink / InfiniBand / PCIe).

Use cases:
- **Cluster design**: determine optimal prefill/decode GPU count ratio for a given workload distribution.
- **Hardware heterogeneity evaluation**: compare all-H100 vs H100-prefill/A100-decode cost/throughput tradeoffs.
- **Routing algorithm prototyping**: test new routing strategies before implementing them in a real framework.

---

## Omission 3: Dynamo — 4-Plane Architecture Details (Control + Discovery + Event Planes)

**Source:** `L3/04_nvidia_dynamo_nixl.md`
**Why omitted from COMBINED.md:** COMBINED.md covers Dynamo's KV-aware routing and NIXL transfer — the two components directly relevant to SGLang's evolution path. The full 4-plane architecture (request, control, discovery, event planes) is an enterprise orchestration concern that goes beyond what most SGLang practitioners need.

### Original text: Control Plane

Dynamo's Planner component adjusts the prefill/decode pool size dynamically:

- Monitors real-time SLO violation rates (TTFT P95 > TTFT_SLO? → scale prefill pool).
- Issues scale-up/scale-down commands to Kubernetes or bare-metal scheduler.
- Uses a feedback control loop with configurable reaction time and cooldown.

The Planner is Dynamo-specific — SGLang's disaggregation currently has no equivalent. Scale-out/scale-in is manual.

### Original text: Discovery Plane (etcd-based)

Workers register themselves with an etcd service using a lease mechanism:
- Each worker publishes its endpoint, role (prefill/decode), and current load state.
- Lease TTL: 10 seconds (configurable). If a worker dies, its lease expires and etcd removes it.
- Router polls etcd for the current worker list; stale entries are automatically removed.
- New workers start serving within one lease period after registration — no manual router reconfiguration.

SGLang's router uses a static list of prefill/decode endpoints (`--prefill` and `--decode` flags). Dynamic discovery requires manual restart if workers change.

### Original text: Event Plane

Asynchronous event bus for cross-component state propagation:
- KV cache hit/miss signals: prefill workers publish prefix cache state; router subscribes.
- SLO violation alerts: monitoring publishes violation events; Planner subscribes.
- Worker availability changes: discovery plane publishes; router subscribes.

The event plane is what makes KV-aware routing possible at scale: the router doesn't need to poll each prefill worker for cache state — it subscribes to state change events and updates its routing table in response.

---

## Omission 4: SARATHI — Pipeline Parallelism Bubble Analysis

**Source:** `L4/01_sarathi_osdi24.md`
**Why omitted from COMBINED.md:** SGLang's PD disaggregation does not currently use pipeline parallelism for the KV computation (it uses tensor parallelism). The pipeline bubble reduction from chunked prefill is a SARATHI-specific result for models served with PP. Including it in COMBINED.md would suggest PP is relevant to Layer 19's architecture, which it is not for current SGLang deployments.

### Original text: Pipeline Bubble Reduction

In pipeline-parallel LLM inference, the model layers are divided across multiple GPUs in a pipeline. Each "microbatch" (slice of the input batch) flows through the pipeline stages sequentially. If microbatches have different compute durations (as they do when large prefill microbatches and small decode microbatches are mixed), some pipeline stages will be idle waiting for the previous stage — creating "pipeline bubbles."

SARATHI's chunked prefill creates uniformly-sized microbatches:
- Every microbatch contains one prefill chunk (fixed token count) + some decode requests.
- The prefill chunk size is chosen to saturate one pipeline stage's compute budget.
- Result: all pipeline stages complete each microbatch in roughly the same time.

**Measured result**: 6.29× pipeline bubble reduction on GPT-3 (175B) with PP=8, leading to 1.91× end-to-end throughput improvement.

This result is important for PP deployments but does not apply to TP-only disaggregated deployments.

---

## Omission 5: vLLM — P2pNcclConnector Proxy Architecture

**Source:** `L4/03_vllm_disagg_connector.md`
**Why omitted from COMBINED.md:** COMBINED.md focuses on the RDMA-based connectors (NIXL, Mooncake) as the production path. The P2pNcclConnector's proxy architecture is a specific implementation detail for clusters without RDMA, which is an edge case for the high-performance serving deployments Layer 19 targets.

### Original text: P2pNcclConnector Proxy

The P2pNcclConnector uses NCCL for direct P2P GPU transfers. It requires a proxy process to coordinate the NCCL communication group between the prefill and decode instances:

```
[Prefill Instance] → [Proxy Process] → [Decode Instance]
   (NCCL send)         (coordinates)      (NCCL recv)
```

The proxy maintains a TCP connection to both instances and sends control messages to initiate NCCL operations. The actual KV data travels NCCL P2P (PCIe or NVLink, depending on hardware), bypassing the proxy for the data path.

**Configuration**:

```json
{
  "kv_connector": "P2pNcclConnector",
  "kv_role": "kv_producer",
  "kv_rank": 0,
  "kv_parallel_size": 2,
  "kv_buffer_size": "1e9",
  "kv_port": "14579",
  "kv_connector_extra_config": {
    "proxy_ip": "127.0.0.1",
    "proxy_port": "30001",
    "send_type": "PUT_ASYNC"
  }
}
```

**Bandwidth limitation**: on PCIe clusters, NCCL P2P is limited to ~32 GB/s (PCIe 4.0 × 16). For 40 GB of KV data, this means ~1.25 seconds per transfer — acceptable only for long decode sequences where the transfer latency is amortised.

---

## Omission 6: TaiChi — Multi-Turn Serving and PPD Disaggregation

**Source:** `L4/02_taichi_agg_vs_disagg.md`
**Why omitted from COMBINED.md:** TaiChi also introduces PPD (Prefill-Prompt-Decode) disaggregation for multi-turn serving. This is a three-way split rather than a two-way split. While theoretically interesting, PPD disaggregation adds significant complexity and is not yet implemented in SGLang or vLLM in a production-ready form.

### Original text: PPD Disaggregation for Multi-Turn Serving

In multi-turn conversations, each new turn adds tokens to a growing history. The new turn's prefill computation can be split into:

- **Append prefill**: computing KV for the new tokens (the human's latest message).
- **Context prefill**: re-computing (or loading from cache) KV for the conversation history so far.

PPD disaggregation introduces a **third worker type** — the Prompt worker — specifically for context prefill (loading/re-computing conversation history KV). The three-way split:

```
P worker (Prompt) → handles context history KV loading/computation
P worker (Prefill) → handles new-token KV computation (append prefill)
D worker (Decode) → handles autoregressive generation
```

**Why this matters**: in long multi-turn conversations, the context prefill (re-processing thousands of historical tokens) dominates prefill compute. Offloading it to a dedicated worker prevents it from blocking new-request prefills.

**Why it is omitted from COMBINED.md**: SGLang's HiCache (Layer 17) handles context KV via prefix caching — if the conversation history KV is cached in GPU or CPU memory, re-computation is avoided entirely. The PPD disaggregation approach is complementary to but distinct from prefix caching, and implementing PPD requires significant scheduler changes not yet in SGLang.

---

## Omission 7: NIXL — NIXLBench and KVBench Tool Details

**Source:** `L3/04_nvidia_dynamo_nixl.md`
**Why omitted from COMBINED.md:** Benchmarking tools are useful for infrastructure teams evaluating NIXL performance but not for typical users deploying disaggregated serving with SGLang. The existence of these tools is noted in the full reference; the technical details are preserved here.

### Original text: NIXLBench

NIXLBench is a model-agnostic KV transfer bandwidth benchmark:

```bash
# Install
pip install nixl

# Run bandwidth sweep
nixl_bench \
  --src-type GPU --dst-type GPU \
  --block-sizes 1MB,4MB,16MB,64MB \
  --batch-sizes 1,4,16,64 \
  --iterations 100 \
  --report bandwidth_latency.csv
```

Output: bandwidth (GB/s) and latency (μs, P50/P95/P99) for each (block_size, batch_size) combination. Use this to characterise the effective KV transfer rate for your specific hardware configuration before deploying disaggregated serving.

### Original text: KVBench

KVBench extends NIXLBench with LLM-model awareness:

```bash
# Auto-calculate KV sizes for LLaMA-3.1-70B, then benchmark
nixl_kvbench --model llama-3.1-70b --seq-len 4096 --batch 8 --tp 8
```

KVBench automatically calculates the KV cache size for a given model configuration and generates the corresponding NIXLBench command. Supported model families: LLaMA, Mistral, DeepSeek, and others.

This tool is particularly useful for capacity planning: given a target TTFT budget and maximum acceptable KV transfer latency, KVBench determines whether your cluster's RDMA bandwidth is sufficient.
