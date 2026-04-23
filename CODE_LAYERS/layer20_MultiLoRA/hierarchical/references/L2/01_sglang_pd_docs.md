# SGLang PD Disaggregation: Official Documentation

**Source:** https://docs.sglang.io/advanced_features/pd_disaggregation.html
**Author:** SGLang Team
**Level:** L2 — Practitioner launch reference
**Why here:** The canonical deployment reference for PD disaggregation in SGLang. Contains every CLI flag, all environment variables for prefill and decode server tuning, router setup, and copy-paste launch commands for Mooncake and NIXL backends on both single-node and multi-node DeepSeek configurations. Layer 19's configuration section draws its launch recipes directly from this document.

---

## Why and What is PD Disaggregation?

LLM inference comprises two distinct phases: **Prefill** (computation-intensive, processes the entire input) and **Decode** (memory-intensive, manages the KV cache for token generation). Traditionally handled in a unified engine, combined scheduling introduces two structural problems:

1. **Prefill Interruption**: Incoming prefill batches frequently interrupt ongoing decode batches, causing substantial delays in token generation.
2. **DP Attention Imbalance**: In data-parallel attention, one DP worker may process a prefill batch while another handles a decode batch simultaneously, leading to increased decode latency.

PD Disaggregation resolves these by separating the two stages into dedicated server processes, enabling tailored optimizations for each phase.

**Supported transfer backends**: Mooncake (RDMA, recommended for production) and NIXL (UCX/LIBFABRIC, network-agnostic).

---

## Mooncake Backend

### Installation

```bash
uv pip install mooncake-transfer-engine
```

### Single-Node Launch (Llama)

```bash
# Start prefill worker
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-ib-device mlx5_roce0

# Start decode worker (on same node, different GPU)
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-ib-device mlx5_roce0

# Start the router (client-facing entry point)
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 8000
```

### Multi-Node Launch (DeepSeek)

```bash
# prefill node 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-mode prefill \
  --host ${local_ip} --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 --node-rank 0 \
  --tp-size 16 --dp-size 8 \
  --enable-dp-attention --moe-a2a-backend deepep \
  --mem-fraction-static 0.8

# decode node 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-mode decode \
  --host ${local_ip} --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 --node-rank 0 \
  --tp-size 16 --dp-size 8 \
  --enable-dp-attention --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128
```

### NVLink Transport (NVL72 Deployments)

```bash
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK
export MC_FORCE_MNNVL=True
```

Supported values: `NVLINK`, `BAREX`, `INTRA_NODE_NVLINK`. Auxiliary data still uses TCP as a temporary workaround.

---

## Prefill Server Environment Variables

| Variable | Description | Default |
|---|---|---|
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` | Worker threads for KV transfer per TP rank | `max(4, min(12, int(0.75 × cpu_count()) // 8))` |
| `SGLANG_DISAGGREGATION_QUEUE_SIZE` | Parallel transfer queues (shards requests from multiple decode instances) | `4` |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | Timeout (sec) for receiving destination KV indices | `300` |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL` | Interval (sec) between cleanup of bootstrap entries | `120` |

> Increase `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600` if higher mean TTFT is acceptable — needed when decode nodes have high latency connecting to prefill nodes.

## Decode Server Environment Variables

| Variable | Description | Default |
|---|---|---|
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` | Health check interval to prefill servers (sec) | `5.0` |
| `SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE` | Consecutive failures before marking prefill offline | `2` |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | Timeout (sec) for receiving KV cache after initialization | `300` |

---

## NIXL Backend

### Installation

```bash
pip install nixl
# Or build from source if UCX is already installed:
git clone https://github.com/ai-dynamo/nixl.git
cd nixl && pip install . --config-settings=setup-args="-Ducx_path=/path/to/ucx"
```

### Single-Node Launch (Llama)

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-transfer-backend nixl

python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 --base-gpu-id 1 \
  --disaggregation-transfer-backend nixl

python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 8000
```

### NIXL Backend Selection

```bash
# Default is UCX; switch to LIBFABRIC if needed
export SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC
```

Available: `UCX` (default), `LIBFABRIC`, or any installed NIXL plugin.

---

## Heterogeneous TP with GPU Staging Buffer

When prefill and decode use different TP sizes (e.g., prefill TP=4, decode DP-attention TP=1), the KV memory layout differs between sides. The GPU staging buffer solves this:

- **Prefill side**: gathers KV head slices into a contiguous staging buffer, then performs bulk RDMA transfer.
- **Decode side**: scatters received data into the correct KV cache pages.
- **Result**: 2–5× throughput improvement over default per-token slice approach at high concurrency.

> **Note:** Only for non-MLA models (GQA/MHA). MLA models (DeepSeek-V2/V3) should **not** enable this flag.

```bash
export SGLANG_DISAGG_STAGING_BUFFER=1
export SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=64  # per-worker, prefill side
export SGLANG_DISAGG_STAGING_POOL_SIZE_MB=4096   # ring buffer, decode side
```

---

## Key Takeaways for Layer 19

- `--disaggregation-mode prefill` / `decode`: the single flag that converts a standard SGLang server into a phase-specific worker.
- `--disaggregation-ib-device`: which RDMA NIC to use; must be set on both prefill and decode sides.
- **Mooncake** is the production recommendation for RDMA clusters; **NIXL** is the vendor-agnostic alternative.
- The **router** is the client-facing entry point — clients send requests to the router, which dispatches to prefill then relays to decode.
- For MoE models (DeepSeek), combine `--moe-a2a-backend deepep` with `--enable-dp-attention` on both prefill and decode workers.
- `SGLANG_DISAGGREGATION_QUEUE_SIZE=4`: allows the prefill server to handle KV transfers to multiple decode instances concurrently.
