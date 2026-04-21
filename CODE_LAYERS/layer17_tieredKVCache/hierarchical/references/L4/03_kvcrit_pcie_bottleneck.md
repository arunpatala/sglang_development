# Understanding Bottlenecks for Efficiently Serving LLM Inference With KV Offloading

**Source:** https://arxiv.org/abs/2601.19910
**Paper PDF:** https://arxiv.org/pdf/2601.19910
**Venue:** arXiv preprint (submitted December 16, 2025)
**Authors:** William Meng (UPenn, Intel), Benjamin Lee (UPenn), Hong Wang (Intel)
**Level:** L4 — Advanced analytical framework; PCIe bottleneck theory
**Why here:** This paper derives **κ_crit** — the critical cached-to-prefill token ratio at which KV-offloaded inference transitions from compute-bound to memory-bound (PCIe-bound). It provides the theoretical foundation for why all of HiCache's data-plane optimizations (GPU-assisted kernels, page-first layout, zero-copy, compute-transfer overlap) matter in production, and what limits they run into. The `load_back_duration_seconds` metric in HiCache's Prometheus output is best interpreted through the lens of this analysis.

**BibTeX:**
```bibtex
@article{meng2025kvcrit,
  title = {Understanding Bottlenecks for Efficiently Serving {LLM} Inference
           With {KV} Offloading},
  author = {William Meng and Benjamin Lee and Hong Wang},
  journal = {arXiv preprint arXiv:2601.19910},
  year = {2025},
  url = {https://arxiv.org/abs/2601.19910}
}
```

---

## Problem

KV cache offloading enables long-context LLM inference by storing caches in CPU DRAM. But:

1. **PCIe bandwidth is limited**: H100 ↔ CPU: ~64 GB/s bidirectional peak; in practice 40–50 GB/s sustained.
2. **KV data is large**: A single layer's KV for 100K tokens at FP16 ≈ several GB.
3. **Typical workloads vastly exceed the PCIe budget**: the ratio of cached tokens to new tokens is orders of magnitude higher than what PCIe can service within one prefill budget.

The result: **99% of inference latency is spent on PCIe transfers**, not GPU computation. The GPU sits at 28% of rated TDP — mostly idle, waiting for data.

---

## The κ_crit Framework

### Definition

Let:
- `C` = number of **cached** tokens to load (from CPU DRAM)
- `P` = number of **new** tokens to prefill (compute KV from scratch on GPU)
- `κ` = C / P = the cached-to-prefill ratio

The **critical ratio κ_crit** is the threshold at which the system transitions:
- `κ < κ_crit`: **compute-bound** — GPU is the bottleneck; KV loading fits within compute time
- `κ > κ_crit`: **memory-bound** — PCIe is the bottleneck; GPU waits for data transfers

### Derivation (simplified)

```
κ_crit = (GPU_FLOPS × bytes_per_KV_entry) / (PCIe_bandwidth × FLOPs_per_token)
```

Where:
- `GPU_FLOPS` = peak GPU throughput (e.g., H100: 989 TFLOPS for FP16)
- `PCIe_bandwidth` = effective transfer rate (e.g., PCIe 5.0 ×16: ~64 GB/s)
- `bytes_per_KV_entry` = size of one token's KV across all layers (model-dependent)
- `FLOPs_per_token` = attention FLOPs per new token

### Example values

| Model | KV size per token | κ_crit (H100, PCIe 5.0) |
|---|---|---|
| Llama-3.1-8B | ~1.0 MB | ~8 |
| Llama-3.1-70B | ~3.5 MB | ~3 |
| DeepSeek-R1-671B (FP8) | ~6.0 MB | ~2 |

For typical RAG or multi-turn workloads, the observed κ ratio is often **100–10,000** — orders of magnitude above κ_crit. This means the system is **severely memory-bound** for any non-trivial amount of cached context.

---

## Empirical Measurements

### Latency breakdown
- **99% of latency** in offloaded inference is PCIe transfers (empirically measured)
- GPU utilization (TDP): **28%** of rated capacity during offloaded inference
- Conclusion: GPU is massively underutilized; the bottleneck is the interconnect

### Transfer efficiency
- Theoretical PCIe 5.0 ×16 peak: 64 GB/s
- Sustained measured rate (pinned memory, large blocks): ~48–55 GB/s
- For small blocks (< 512 KB per transfer): drops to 10–20 GB/s due to overhead

---

## Proposed Optimizations (from the paper)

### 1. Hardware Interconnects
- **CXL memory** (DDR5 over CXL): memory-semantics access to pooled DRAM; avoids PCIe protocol overhead
- **NVLink-C2C** (Grace-Hopper): 900 GB/s CPU↔GPU bandwidth vs 64 GB/s PCIe
- **HBM expansion**: stacked memory on the package, no PCIe involved

### 2. Model Architecture
- **Grouped Query Attention (GQA)**: reduces KV size by num_heads/num_kv_heads — directly reduces PCIe load
- **MLA (Multi-head Latent Attention)**: extreme KV compression (DeepSeek-V2/R1); κ_crit effectively raised by 10–20×
- **Hybrid sparse attention**: some layers use full KV, others use sliding window — reduces average KV size

### 3. Scheduling Algorithms
- **Selective prefetch**: only load KV for tokens with high attention importance (cf. InfiniGen, IMPRESS)
- **Request batching**: batch multiple requests that share the same cached prefix → amortize PCIe cost
- **Pipelining**: overlap KV loading with computation for different requests in the batch

---

## Implications for HiCache

### κ_crit tells you when HiCache L2 (CPU DRAM) helps

| Scenario | κ ratio | Memory-bound? | HiCache L2 helps? |
|---|---|---|---|
| 1K cached, 1K new | 1 | No | Marginal (κ < κ_crit) |
| 10K cached, 100 new | 100 | Yes | Yes (reduces re-prefill cost) |
| 100K cached, 10 new | 10,000 | Severely | Yes, but PCIe is the new bottleneck |

When κ ≫ κ_crit:
- Loading from L2 (CPU DRAM) is still faster than re-prefill (fewer FLOPs needed)
- But the PCIe transfer itself takes time proportional to cache size
- HiCache's compute-transfer overlap (`load_to_device_per_layer`) hides this latency

### What κ_crit means for HiCache L3 (Disk/Remote Storage)

For L3 backends (file, 3FS, Mooncake):
- Latency is **much higher** than PCIe (NVMe: ~100µs; network: 1–10ms)
- κ must be **very high** (1000+) to justify L3 loading instead of re-prefill
- HiCache's `prefetch_threshold` (default 256 tokens) is a heuristic for this — only prefetch from L3 if there are enough cached tokens to justify the I/O latency

### Monitoring κ in production

Use the Prometheus metrics:
```promql
# Approximate κ ratio
sum(rate(sglang:load_back_tokens_total[5m]))
  / sum(rate(sglang:num_tokens_generated_total[5m]))
```

When this ratio is high and `sglang:load_back_duration_seconds` P99 is also high, you are in the PCIe-bound regime. Options:
1. Increase `--hicache-io-backend kernel` (use GPU-assisted kernels for higher throughput)
2. Increase `--page-size` (larger contiguous transfers → higher DMA efficiency)
3. Use `--hicache-mem-layout page_first_direct` (reduces transfer staging overhead)
4. Consider hardware upgrade (CXL, NVLink-C2C) if PCIe is the hard limit

---

## Key Takeaways for Layer 17

- **κ_crit quantifies when KV offloading helps**: only when the cached-to-prefill ratio exceeds the GPU compute-to-PCIe bandwidth ratio.
- **99% of latency is PCIe transfers** in typical offloaded workloads — GPU is idle 72% of the time.
- **HiCache's data-plane optimizations** (kernels, page-first, zero-copy) directly address this bottleneck.
- **MLA and GQA models** have fundamentally better κ_crit because their KV is smaller — HiCache is more effective on these models.
- **`load_back_duration_seconds` P99** is the most important Prometheus metric for detecting the PCIe-bound regime.
- **Future hardware** (CXL, NVLink-C2C, HBM expansion) will raise the effective PCIe bandwidth, improving κ_crit and making HiCache even more effective.
