# Inside vLLM's New KV Offloading Connector: Smarter Memory Transfer for Maximizing Inference Throughput

**Source:** https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html
**Author:** vLLM Team, January 8, 2026
**Level:** L2 — Practitioner / design comparison
**Why here:** vLLM independently solved the same GPU KV cache capacity problem with a CPU offloading connector. Reading this alongside the SGLang HiCache blog reveals both what the two systems share (pinned memory, DMA, async transfers, page-size tuning) and where they diverge (embedded vs connector-API approach, GPU-assisted kernels vs DMA-only). The DMA vs custom kernel analysis directly explains why HiCache's `--hicache-io-backend kernel` option exists.

---

## Motivation

Serving LLM models is computationally complex, centered on computing Key-Value (KV) data. The prefill stage — where KV values are calculated per prompt — is expensive. KV cache offloading provides two benefits:

1. **Prefix cache reuse** across requests with shared prefixes → lower TTFT, higher throughput.
2. **Preemption avoidance** when GPU memory is full → instead of discarding and recomputing, offload to CPU DRAM.

CPU RAM is preferred as a staging tier because:
- Capacity typically exceeds GPU VRAM
- Low latency, high throughput transfers over PCIe
- Good staging area for further offloading to disk/remote storage

---

## The vLLM Offloading Connector

### Usage

```bash
# vLLM 0.14.0+
vllm serve <model> --kv_offloading_backend native --kv_offloading_size <size_in_GB>

# Older releases
vllm serve <model> --kv-transfer-config \
  '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks": <num_cpu_blocks>}}'
```

### Connector API Design

The connector API is called **before** each request (load from external source) and **after** computation (store KV data). As of v0.9.0, it supports **asynchronous** load/store, enabling offloading to run in parallel with model computation.

---

## Benefits

### Single-request TTFT (Llama-3.1-8B, H100)
Loading KV from CPU reduces TTFT by **2×–22×** depending on prompt length.

### Concurrent request throughput (10,000 × 512-token prefill requests, H100)
Throughput increases with CPU KV cache hit rate. At 80% hit rate: **up to 9× throughput improvement**.

> "The major gain in KV cache offloading is throughput maximization."

### Version improvements
- **v0.12.0**: Up to 4× TTFT reduction, 5× throughput increase vs v0.11.0 (due to physical block layout change)
- **v0.14.0** (upcoming): preempted requests can load back from CPU; race condition fix

---

## DMA vs Custom Kernel: The Core Transfer Decision

### Background
When copying KV data from GPU to CPU (or back), there are two mechanisms:
- **DMA** (`cudaMemcpyAsync`): uses hardware DMA engine, minimal CPU/GPU core overhead.
- **Custom CUDA kernel**: uses GPU cores directly to copy 16-byte words via raw pointers; high parallelism but interferes with model computation.

### Single-direction benchmark (1000 blocks, H100)
- Small blocks (< ~1 MB): **custom kernel wins** significantly.
- Large blocks (> ~2 MB): **DMA is competitive** and less noisy.
- Bi-directional peak: DMA = **83.4 GB/s**, kernel = **68.5 GB/s**.

### The Physical Block Size Problem
vLLM's default layout stores each layer's KV cache separately, fragmenting a logical block into `num_layers × 2` sub-blocks. Effective block size for Llama-3.1-8B at default settings: **32 KB** (old) → **2 MB** (new after v0.12.0 layout change).

### Physical block sizes for common models (16-token blocks)

| Model | Old size | New size |
|---|---|---|
| Llama-3.1-8B-Instruct | 32 KB | 2 MB |
| Llama-3.1-70B-Instruct | 8 KB | 1.25 MB |
| DeepSeek-R1-Distill-Qwen-32B (TP=2) | 16 KB | 2 MB |
| Mistral-7B-Instruct-v0.2 | 32 KB | 2 MB |
| Qwen2.5-7B-Instruct | 16 KB | 0.87 MB |

The new layout increased physical block size by `2 × num_layers`, putting all models in the DMA-efficient range.

### End-to-end: DMA vs Kernel (concurrent requests)
- DMA achieves **5–32% better throughput** than the custom kernel, because the kernel interferes with GPU model computation.
- For cache misses: the custom kernel actually **reduces throughput by 6%** vs baseline (no offloading).

**Conclusion**: DMA is preferred for concurrent workloads due to non-interference with GPU cores. This is why HiCache's `--hicache-io-backend kernel` (which uses GPU-assisted kernels on top of DMA) is positioned as higher-performance for its specific transfer patterns.

---

## Key Takeaways for Layer 17

- **vLLM independently validated** the same design decisions as HiCache: async transfers, pinned memory, large contiguous block transfers.
- **Physical block layout is critical**: fragmented layouts kill DMA performance. HiCache's `--hicache-mem-layout page_first` solves the same problem.
- **GPU-core-based kernels** interfere with model computation in concurrent scenarios; HiCache's custom kernels are tuned to minimize this.
- vLLM's connector is a **middleware approach** (pluggable API); HiCache is **embedded** in the SGLang memory manager — different tradeoffs for integration depth and flexibility.
- Both systems confirm the **2–22× TTFT improvement** and **up to 9× throughput gain** from CPU offloading.
