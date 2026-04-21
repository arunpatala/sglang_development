# SGLang HiCache: Fast Hierarchical KV Caching with Your Favorite Storage Backends

**Source:** https://lmsys.org/blog/2025-09-10-sglang-hicache/
**Author:** Zhiqiang Xie, SGLang Team (LMSYS Org), September 10, 2025
**Level:** L2 — Practitioner motivation and launch recipes
**Why here:** The primary launch post for SGLang HiCache. Contains real production testimonials (Novita AI, Ant Group), the authoritative benchmark numbers (6× throughput, 80% TTFT reduction), and copy-paste launch commands for the 3FS and Mooncake backends. Layer 17's `06_configuration.md` draws its canonical launch recipes directly from this post.

---

## From the Community

> In a coding agent scenario using Qwen3-Coder-480B, the observed dialogues often stretched past 25K tokens around 8 turns per session. Without full KV cache retention, nearly every request required costly re-computation. By integrating SGLang HiCache with DeepSeek 3FS KVStore for large-scale historical KV caching, the session's average TTFT dropped by **56%**, inference throughput **doubled**, and the cache hit rate jumped from **40% to 80%**.
> – Novita AI

> Integrating SGLang HiCache with the Mooncake service enables scalable KV cache retention and high-performance access. In our evaluation, we tested the DeepSeek-R1-671B model under PD-disaggregated deployment using in-house online requests sampled from a general QA scenario. On average, cache hits achieved an **84% reduction in TTFT** compared to full re-computation.
> – Ant Group

SGLang HiCache achieves up to **6× throughput improvement** and up to **80% reduction in TTFT** in provided benchmarks.

---

## Why Hierarchical KV Caching Matters

[RadixAttention](https://arxiv.org/abs/2312.07104) achieves state-of-the-art performance by reusing KV caches stored in GPU memory. However, the caching benefit is inevitably limited by a **capacity bottleneck**: as contexts grow longer and more clients engage in more rounds of conversations, the cache hit rate declines because most historical KV caches must be evicted to make room for new data.

To address this, SGLang HiCache extends RadixAttention with a **HiRadixTree** that acts as a page table for referencing KV caches residing locally in GPU and CPU memory. Alongside, a cache controller automatically manages loading and backing up KV cache data across hierarchies, including GPU and CPU memory pools as well as external layers such as disks and remote memory.

---

## Design of SGLang HiCache

### Optimized Data Plane

The key bottleneck in hierarchical memory systems is the latency of moving data from slower to faster tiers. Beyond the standard `cudaMemcpyAsync`, SGLang developed a set of [GPU-assisted I/O kernels](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/kvcacheio/transfer.cu) that deliver **up to 3× higher throughput** for CPU–GPU transfers.

HiCache uses a **"page-first" layout** for CPU and storage layers (instead of GPU's "layer-first" layout), enabling larger transfer sizes per transaction. When combined with a **zero-copy** mechanism, this achieves up to **2× higher throughput** in typical deployments.

### Versatile Control Plane

- **CPU hit**: Uses layer-wise overlapping to load KV cache of layer N+1 while layer N is executing — hiding transfer latency behind computation.
- **Storage hit**: Opportunistically prefetches data from storage into host memory once a cache hit is detected. The prefetch strategy is configurable: `best_effort`, `wait_complete`, or `timeout`.
- **Write policies**: `write_through` (strongest caching), `write_through_selective` (backs up only hot spots), `write_back` (mitigates capacity pressure).

### Pluggable Storage Backends

The storage backend interface requires only three methods: `get(key)`, `exist(key)`, `set(key, value)`. Built-in backends:

- **Mooncake** — RDMA-based distributed shared memory
- **3FS (HF3FS)** — DeepSeek's distributed filesystem
- **NIXL** — Unified API for GDS, 3FS, S3-compatible object storage
- **File** — Local disk (reference implementation)
- **AIBrix KVCache** — External KVCache service

---

## Benchmark Launch Commands

### DeepSeek R1 on 8× H20 with 3FS

```bash
python3 -m sglang.launch_server \
  --model-path /DeepSeek-R1/ --tp 8 --page-size 64 \
  --context-length 65536 --chunked-prefill-size 6144 \
  --mem-fraction-static 0.85 \
  --enable-hierarchical-cache --hicache-ratio 2 \
  --hicache-io-backend kernel --hicache-mem-layout page_first \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete
```

### Qwen3-235B on 8× H800 with Mooncake (RDMA)

```bash
MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata" \
MOONCAKE_GLOBAL_SEGMENT_SIZE=816043786240 \
MOONCAKE_PROTOCOL="rdma" \
MOONCAKE_DEVICE="$DEVICE_LIST" \
MOONCAKE_MASTER=127.0.0.1:50051 \
python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 8 --page-size 64 \
  --enable-hierarchical-cache --hicache-ratio 2 \
  --hicache-storage-prefetch-policy timeout \
  --hicache-storage-backend mooncake
```

---

## Benchmark Instructions

```bash
# Long-context benchmark
python3 benchmark/hicache/bench_long_context.py \
  --model-path /DeepSeek-R1/ \
  --dataset-path loogle_wiki_qa.json

# Multi-turn benchmark
python3 benchmark/hicache/bench_multiturn.py \
  --model-path $MODEL_PATH --disable-random-sample \
  --output-length 1 --request-length 2048 \
  --num-clients 80 --num-rounds 10 \
  --max-parallel 4 --request-rate 16
```

Full benchmark scripts: https://github.com/sgl-project/sglang/tree/main/benchmark/hicache

---

## Key Takeaways for Layer 17

- The **capacity bottleneck** of GPU-only RadixCache is the core motivation for HiCache.
- The **HiRadixTree** is the metadata layer — it tracks token-prefix to tier-location mappings.
- **GPU-assisted I/O kernels** and **page-first layout** are the two main data-plane optimizations.
- **Three prefetch policies** offer different TTFT/throughput tradeoffs.
- Real deployments confirm 2–6× throughput gains and 56–84% TTFT reductions.
