# PD Disaggregation — Material Omitted from COMBINEDL3L4.md

**What this file is:** The full text of every section omitted from `COMBINEDL3L4.md`. The "What Is Left Out and Why" appendix of `COMBINEDL3L4.md` names each omission and explains why it was excluded. This file preserves the complete material so nothing is lost.

**Parent file:** `COMBINEDL3L4.md` (L1 + L2 + L3 + L4 synthesis)
**Sources:** L3/02 (SplitwiseSim), L3/03 (Mooncake prediction rejection), L3/04 (NIXL benchmarking tools, Dynamo control plane params), L4/02 (TaiChi PPD), L4/03 (vLLM full config schema), L2/02 (MoE dispatch internals)

---

## Omission 1: SplitwiseSim — Cluster-Level Simulation Tool

**Source:** `L3/02_splitwise_isca24.md`
**Why omitted from COMBINEDL3L4.md:** SplitwiseSim is a research-level offline evaluation tool. Its purpose and scope are described in COMBINEDL3L4 §7 ("Splitwise: Production Traces and Hardware Heterogeneity"). The tool's configuration format, trace input format, and example commands are operator tooling — relevant for researchers designing new routing algorithms, but not for production deployment decisions.

---

### SplitwiseSim: What It Is

SplitwiseSim (https://github.com/Mutinifni/splitwise-sim) is a **discrete-event simulator** for evaluating cluster-level PD disaggregation policies offline — without requiring a physical cluster.

### What the Simulator Models

- **Heterogeneous hardware pools**: configure different GPU types per phase (e.g., H100 for prefill, A100 for decode); the simulator models FLOP throughput and HBM bandwidth separately per GPU type.
- **Workload distributions**: replay real or synthetic prompt/output length distributions. The Splitwise paper includes two Azure production traces (one code-completion workload, one conversational workload).
- **Routing policies**: evaluate round-robin, load-aware, and cache-aware routing across the cluster at simulated scale (hundreds of GPUs), without the cost of a physical experiment.
- **KV transfer latency**: parameterised by payload size and network topology (NVLink, InfiniBand, PCIe). The simulator models network as a first-class resource with configurable bandwidth and latency.

### Use Cases

**Cluster design**: determine the optimal prefill/decode GPU count ratio for a target workload distribution. The simulator sweeps prefill:decode ratios and reports goodput at each configuration — avoiding the need to physically deploy each ratio to find the optimum.

**Hardware heterogeneity evaluation**: compare all-H100 vs H100-prefill/A100-decode vs other combinations. The simulator's throughput-at-SLO metric (equivalent to goodput) shows the cost-per-token improvement from using cheaper hardware for decode.

**Routing algorithm prototyping**: test new routing strategies (e.g., a prefix-cache-aware routing policy) before implementing them in vLLM, SGLang, or Dynamo. The simulator replays the workload distribution against the routing policy and reports TTFT, TPOT, and goodput distributions.

### Example: Evaluating Prefill/Decode Ratio

A typical SplitwiseSim experiment for ratio evaluation:
1. Fix the total cluster size (e.g., 64 GPUs, all H100 SXM).
2. Sweep the prefill fraction from 10% to 90% (6 to 58 GPUs for prefill; remainder for decode).
3. For each fraction, replay the target workload and record goodput at (TTFT_SLO = 2s, TPOT_SLO = 100ms).
4. The optimal fraction minimises cost while meeting the SLO at the target request rate.

For the DeepSeek-V3 deployment (96 H100 GPUs), a similar analysis led to the 3:9 (25% prefill / 75% decode) ratio used in production.

---

## Omission 2: Mooncake — Prediction-Based Early Rejection

**Source:** `L3/03_mooncake_fast25.md`
**Why omitted from COMBINEDL3L4.md:** Mooncake's prediction-based early rejection is mentioned in §10 as a mechanism for handling overloaded conditions. The prediction model details, the features it uses, and the rejection threshold tuning are operational specifics not yet implemented in SGLang's disaggregation mode.

---

### The Problem: Overload Without Rejection

In highly overloaded conditions, a serving system that accepts all requests creates a cascading backlog: every accepted request waits in queue, extends the queue for subsequent requests, and causes widespread TTFT violations. Without active rejection, the system degrades gracefully in throughput but catastrophically in SLO compliance.

### Mooncake's Solution: Admit-Time SLO Prediction

Before accepting a request, Mooncake's scheduler runs a lightweight prediction:

**Prediction inputs:**
- Current queue depth in the prefill cluster (number of waiting requests)
- Current queue depth in the decode cluster
- Estimated KV transfer bandwidth to the decode cluster (from recent measurements)
- Historical TTFT and TPOT distribution for similar requests
- Request metadata: estimated prompt length, expected output length

**Prediction output**: probability that this request will complete within its SLO constraints if accepted.

**Decision rule**: if `P(SLO satisfied | current system state) < threshold`, return HTTP 429 (Too Many Requests) immediately — before any GPU resources are allocated to the request.

### Why This Works

By rejecting requests at admission time, Mooncake preserves headroom for the requests it does accept. The accepted requests see shorter queues, lower transfer latency, and higher SLO compliance — at the cost of a higher rejection rate for the marginal requests.

**Production result (Kimi service)**: under simulated high-overload conditions, early rejection allows Mooncake to maintain SLO compliance for 75% of requests (those it accepts), vs vLLM collocated which maintains compliance for much lower fractions under the same load.

### Why SGLang Does Not Implement This

SGLang's current disaggregation mode accepts all requests and queues them. Admission control requires:
1. A real-time model of current system state (queue depths, transfer bandwidth)
2. A prediction model trained on the target workload
3. A configurable rejection threshold (and a mechanism to tune it)

These are non-trivial operational requirements. The correct design is similar to Dynamo's Planner feedback loop — monitoring SLO violation rates and adjusting admission thresholds accordingly.

---

## Omission 3: NIXL — NIXLBench and KVBench Tooling

**Source:** `L3/04_nvidia_dynamo_nixl.md`
**Why omitted from COMBINEDL3L4.md:** NIXLBench and KVBench are operator tools for characterising KV transfer bandwidth before deploying a disaggregated cluster. Their purpose is mentioned in COMBINEDL3L4 §12 and §18 (Check 5: "Run KVBench against your actual cluster topology"). The command-line details belong in an operator runbook.

---

### NIXLBench: Model-Agnostic Bandwidth Benchmark

NIXLBench is a bandwidth and latency benchmark for NIXL transfer paths — independent of any specific LLM model.

```bash
# Install NIXL
pip install nixl

# Run bandwidth sweep
nixl_bench \
  --src-type GPU --dst-type GPU \
  --block-sizes 1MB,4MB,16MB,64MB,256MB,1GB \
  --batch-sizes 1,4,16,64 \
  --iterations 100 \
  --report bandwidth_latency.csv
```

**Output per (block_size, batch_size) combination:**
- Bandwidth (GB/s)
- Latency: P50, P95, P99 (microseconds)

**What to look for:**
- Bandwidth should approach the theoretical NIC line-rate for large block sizes.
- P99 latency should be stable — large variance indicates NIC contention or PCIe/NUMA routing issues.
- Bandwidth for `--src-type CPU --dst-type GPU` vs `--src-type GPU --dst-type GPU` shows the GPUDirect RDMA speedup.

**Using NIXLBench for cluster validation**: before deploying a disaggregated cluster, run NIXLBench between all prefill-decode node pairs. Verify that bandwidth meets the minimum required for your TTFT budget (use the formula in COMBINEDL3L4 §9: `required_bandwidth = kv_cache_size / transfer_time_budget`).

### KVBench: LLM-Aware Profiler

KVBench extends NIXLBench with awareness of LLM model architectures. It automatically calculates the exact KV cache size for a given model and sequence length.

```bash
# Benchmark KV I/O for LLaMA-3.1-70B, 4K-token sequence, batch of 8, TP=8
nixl_kvbench --model llama-3.1-70b --seq-len 4096 --batch 8 --tp 8

# DeepSeek-V3 (MLA architecture), 2K sequence, batch 32, TP=16
nixl_kvbench --model deepseek-v3 --seq-len 2048 --batch 32 --tp 16
```

**What KVBench generates:** the exact NIXLBench command with block sizes and batch sizes corresponding to the model's actual KV tensor shapes. This avoids misconfiguration where the benchmark block size doesn't match the actual transfer unit.

**Supported models**: LLaMA (all sizes), Mistral, DeepSeek-V2/V3 (MLA architecture, smaller KV due to latent attention), Qwen, and others in NIXL's model registry.

**Capacity planning use case**: given a target TTFT budget and maximum acceptable KV transfer latency, KVBench determines whether your cluster's RDMA bandwidth is sufficient for your model and typical prompt length. If not sufficient, it identifies which bottleneck (NIC bandwidth, PCIe routing, or network topology) is limiting.

---

## Omission 4: NVIDIA Dynamo — Control Plane Parameters and Planner Configuration

**Source:** `L3/04_nvidia_dynamo_nixl.md`
**Why omitted from COMBINEDL3L4.md:** COMBINEDL3L4 §11 describes the Planner's role (monitoring SLO violations, issuing scale-up/scale-down commands). The Planner's configuration parameters, YAML schema, and feedback loop tuning guidance are enterprise deployment specifics beyond the conceptual architecture.

---

### Planner Configuration Parameters

The Dynamo Planner is configured via a YAML file specifying:

```yaml
planner:
  ttft_slo_ms: 2000          # TTFT SLO threshold in milliseconds
  tpot_slo_ms: 100           # TPOT SLO threshold in milliseconds
  slo_violation_window_s: 60  # Measurement window for computing violation rates
  scale_up_threshold: 0.05   # Trigger scale-up when violation rate > 5%
  scale_down_threshold: 0.01 # Allow scale-down when violation rate < 1%
  reaction_time_s: 30        # Minimum time between consecutive scaling actions
  cooldown_s: 120            # Minimum time after scale-up before scale-down allowed
  min_prefill_workers: 1
  max_prefill_workers: 32
  min_decode_workers: 1
  max_decode_workers: 64
```

**Tuning guidance (from Dynamo docs):**
- `reaction_time_s`: too short → over-reactive, oscillating pool sizes; too long → slow response to traffic bursts.
- `cooldown_s`: prevents scale-down immediately after scale-up (workers need time to warm up their KV caches after scaling).
- `scale_up_threshold` vs `scale_down_threshold`: the asymmetric gap (5% vs 1%) creates a hysteresis band — the system doesn't oscillate around the boundary.

### etcd Discovery Plane Details

Each Dynamo worker publishes a lease record at startup:

```json
{
  "endpoint": "http://10.0.1.42:30000",
  "role": "prefill",
  "model": "deepseek-v3",
  "tp_size": 16,
  "load_state": {
    "queue_depth": 3,
    "kv_cache_utilisation": 0.72
  }
}
```

- **Lease TTL**: 10 seconds (configurable via `DYNAMO_ETCD_LEASE_TTL`).
- **Heartbeat interval**: workers refresh their lease every `TTL / 3` seconds.
- **Router update latency**: the router polls etcd every `TTL / 2` seconds — maximum staleness before a dead worker is removed from routing is `~1.5 × TTL` (15 seconds at default settings).

### KV Block Manager (KVBM): HiCache Integration

KVBM extends effective KV cache capacity using multi-tier memory offload/recall:

```
GPU HBM (hot cache) → CPU DRAM (warm cache) → NVMe (cold cache)
```

When a prefill worker's GPU VRAM fills:
1. KVBM offloads the least-recently-used KV blocks to CPU DRAM.
2. If CPU DRAM fills, offloads further to NVMe.
3. On a cache hit for an offloaded prefix: restores to GPU HBM before the request is served.

This is functionally identical to SGLang's HiCache (Layer 17), integrated into the Dynamo control plane. The KVBM enables a prefill worker to maintain a much larger effective prefix cache than GPU VRAM allows — at the cost of restoration latency for offloaded prefixes.

**KVBM configuration (Dynamo YAML):**

```yaml
kvbm:
  cpu_dram_capacity_gb: 256   # CPU DRAM available for KV offload
  nvme_capacity_gb: 4096      # NVMe capacity for cold KV storage
  eviction_policy: lru        # least-recently-used
  prefetch_enabled: true      # Prefetch likely-needed KV blocks from DRAM/NVMe before request arrives
```

---

## Omission 5: TaiChi — PPD Three-Way Disaggregation

**Source:** `L4/02_taichi_agg_vs_disagg.md`
**Why omitted from COMBINEDL3L4.md:** TaiChi's SLO-regime framework and hybrid switching (COMBINEDL3L4 §13) are the actionable results for production deployments. PPD disaggregation is a theoretical extension for multi-turn serving not yet implemented in any production framework.

---

### The Multi-Turn Problem

In multi-turn conversations, each new turn adds tokens to a growing history. The new turn's prefill computation splits into two parts:

- **Append prefill**: computing KV for the new tokens (the user's latest message only — typically 10–100 tokens).
- **Context prefill**: computing or loading KV for the conversation history so far (potentially thousands of tokens from previous turns).

In a standard PD disaggregated system, both parts of the new turn's prefill run on the same prefill worker. For long conversations, context prefill can dominate prefill compute — monopolising the prefill pool for historical tokens, blocking new requests' prefills.

### PPD: The Three-Way Split

TaiChi introduces a **third worker type** — the Prompt worker — specifically for context history KV loading and computation:

```
Client request
       ↓
P worker (Prompt)   → handles context history KV loading/computation
                        (re-loading or re-computing all previous turn KV)
       ↓ (context KV)
P worker (Prefill)  → handles new-token KV computation (append prefill)
                        (only the new tokens from this turn)
       ↓ (full KV: context + new tokens)
D worker (Decode)   → handles autoregressive generation
       ↓
Output stream
```

**Why this matters**: in a long conversation (e.g., 20 turns of 200 tokens each = 4,000-token history), the "context prefill" for the 21st turn processes those 4,000 historical tokens. Offloading this to a dedicated Prompt worker prevents blocking the main Prefill pool for 4,000 tokens worth of computation.

### Relationship to Prefix Caching (SGLang HiCache)

PPD disaggregation and prefix caching are complementary:
- **Prefix caching (HiCache, Layer 17)**: if the conversation history KV is cached in GPU or CPU memory, context prefill is avoided entirely (a cache hit → no recomputation).
- **PPD disaggregation**: handles the case where the cache is cold (no hit) — the Prompt worker re-computes context KV without blocking the Prefill pool.

For workloads with high prefix cache hit rates, PPD adds complexity without benefit. For workloads with cold caches (fresh conversations, no reuse), PPD provides throughput improvement.

### Current Status

PPD disaggregation requires:
- A three-way scheduler routing context-prefill work to Prompt workers.
- A new KV transfer path from Prompt → Prefill (context KV) → Decode.
- Coordination between three worker types instead of two.

**Not yet implemented** in SGLang, vLLM, or Dynamo in a production-ready form. Active research direction.

---

## Omission 6: vLLM — Full `KVTransferConfig` JSON Schema

**Source:** `L4/03_vllm_disagg_connector.md`
**Why omitted from COMBINEDL3L4.md:** COMBINEDL3L4 §14 covers the most important field (`kv_role`) and describes each connector. The full configuration schema with all fields, types, defaults, and per-connector extra-config options is reference documentation for connector writers, not for operators deploying one of the six standard connectors.

---

### Full `KVTransferConfig` Schema

```python
class KVTransferConfig(BaseModel):
    kv_connector: str                  # Connector class name (required)
    kv_role: str                       # "kv_producer" | "kv_consumer" | "kv_both" (required)
    kv_rank: int = 0                   # Rank within the transfer group
    kv_parallel_size: int = 2          # Total group size (producers + consumers)
    kv_buffer_size: float = 1e9       # Transfer buffer size in bytes (1 GB default)
    kv_port: str = "14579"             # Port for connector control communication
    kv_connector_extra_config: dict    # Connector-specific; see per-connector below
```

### Per-Connector Extra-Config Fields

**NixlConnector:**
```json
{
  "NIXL_Backend": "UCX"       // "UCX" (default) or "LIBFABRIC"
}
```

**MooncakeConnector:**
```json
{
  "mooncake_config_path": "/path/to/mooncake.json"   // Mooncake cluster topology file
}
```

**P2pNcclConnector:**
```json
{
  "proxy_ip": "127.0.0.1",
  "proxy_port": "30001",
  "send_type": "PUT_ASYNC"    // "PUT_ASYNC" (non-blocking) or "PUT_SYNC"
}
```

**MultiConnector:**
```json
{
  "connectors": [
    { "kv_connector": "NixlConnector", ... },
    { "kv_connector": "LMCacheConnectorV1", ... }
  ]
}
```

Connectors are tried in order; the first that succeeds for a given transfer is used. Useful for RDMA → NCCL fallback.

### `kv_parallel_size` and `kv_rank` Semantics

For a deployment with 2 prefill workers and 4 decode workers:
- `kv_parallel_size = 6` (total workers in the transfer group)
- Prefill workers: `kv_rank = 0, 1` (producers)
- Decode workers: `kv_rank = 2, 3, 4, 5` (consumers)

The rank determines which buffer slots each worker uses — preventing producers from writing to the same buffer slots as other producers.

### vLLM Directory Structure

```
vllm/distributed/kv_transfer/
├── kv_connector/
│   ├── base.py                   # BaseKVConnector abstract class
│   ├── factory.py                # ConnectorFactory — maps name string to class
│   ├── simple_connector.py       # Wraps LookupBuffer into Connector interface
│   └── worker_connector.py       # Per-worker connector state management
├── kv_lookup_buffer/
│   ├── base.py                   # KVLookupBufferBase abstract class
│   └── simple_buffer.py          # In-memory dictionary implementation
├── kv_transfer_agent.py          # Manages connector lifecycle per process
└── kv_transfer_config.py         # KVTransferConfig dataclass definition
```

Each connector lives in `vllm/distributed/kv_transfer/<connector_name>/`:
- `nixl/` — NixlConnector implementation
- `mooncake/` — MooncakeConnector implementation
- `p2p/` — P2pNcclConnector + proxy process
- `lmcache_integration/` — LMCacheConnectorV1

---

## Omission 7: DeepEP/EPLB — MoE Dispatch Internals

**Source:** `L2/02_lmsys_deepseek_96h100.md`
**Why omitted from COMBINEDL3L4.md:** COMBINEDL3L4 §16 states that PD disaggregation is mandatory for MoE models (DeepSeek-V3) and identifies the reason (incompatible DeepEP dispatch modes for prefill vs decode). The dispatch mode mechanics and EPLB load balancing algorithm are internals for engineers adapting disaggregation to other MoE architectures.

---

### DeepEP: Two Dispatch Modes

DeepSeek-V3 uses Mixture-of-Experts (MoE) with 256 experts per layer (8 activated per token). The expert parallelism communication library (DeepEP) implements two fundamentally different all-to-all dispatch patterns:

**Normal mode** (required for prefill):
- High-throughput all-to-all for large batches
- Maximises compute-communication overlap: expert compute on GPU N runs while expert tokens for GPU N+1 are still in transit
- Batch sizes: typically 100–1000+ tokens per dispatch
- Latency: high (designed for throughput, not low latency)

**Low-latency mode** (required for decode):
- Minimises dispatch latency for small batches (often single-token per request)
- Sacrifices throughput for latency: no compute-communication overlap; waits for all dispatches to complete before compute
- Batch sizes: 1–8 tokens per dispatch in typical decode steps
- Latency: low (designed for autoregressive latency)

**The conflict**: when prefill and decode are interleaved on the same GPU workers, the server must switch between modes per batch. This is technically possible but cannot be done optimally: "normal mode" batches are delayed when the scheduler is in "low-latency mode", and vice versa. The overhead of mode-switching and the inability to sustain the optimal per-mode configuration reduces throughput significantly.

**PD disaggregation resolves this permanently**: prefill nodes run "normal mode" exclusively; decode nodes run "low-latency mode" exclusively. No mode switching, no compromise.

### EPLB: Expert Parallelism Load Balancing

Expert activation is inherently uneven — some experts are activated far more frequently than others. Without intervention, the GPUs hosting popular experts become bottlenecks while GPUs hosting rare experts sit idle.

**EPLB algorithm:**
1. Measure expert activation frequency over a calibration window (e.g., 1,000 requests from the target workload distribution).
2. Compute a target activation budget per GPU: `target = total_activations / n_gpus`.
3. Solve a bin-packing assignment: reassign experts to GPUs to minimise the maximum deviation from target.
4. Apply the assignment: update the expert-to-GPU mapping in the model's EP configuration.

**EPLB in the SGLang DeepSeek deployment**: the blog uses an expert activation distribution matching the observed input/output data pattern. The EPLB assignment is precomputed offline and baked into the server's launch configuration.

**Without EPLB**: some GPUs in the expert parallel group are fully saturated (high-activation experts) while others are idle (rare experts). Effective throughput is limited by the busiest GPU rather than the average.

### SGLang CLI for MoE Disaggregation

```bash
# Prefill node — normal DeepEP mode
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-mode prefill \
  --disaggregation-ib-device ${device_name} \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 --node-rank 0 \
  --tp-size 16 --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \     # Activates DeepEP; normal mode used automatically for prefill
  --mem-fraction-static 0.8

# Decode node — low-latency DeepEP mode
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-mode decode \
  --disaggregation-ib-device ${device_name} \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 --node-rank 0 \
  --tp-size 16 --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \     # DeepEP; low-latency mode used automatically for decode
  --mem-fraction-static 0.8 \
  --max-running-requests 128
```

The `--disaggregation-mode` flag determines which DeepEP dispatch mode is selected — no additional flag is needed. The mode is chosen automatically based on the worker's role.
