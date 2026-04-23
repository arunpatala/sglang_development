# Understanding Prefill-Decode Disaggregation in LLM Inference Optimization

**Source:** https://naddod.medium.com/understanding-the-prefill-decode-disaggregation-in-llm-inference-optimization-5c11223a5360
**Author:** NADDOD (The Interconnect Engine for the AI Era)
**Published:** August 22, 2025 — 5-minute read
**Level:** L1 — Structured 5-minute overview; advantages and challenges with solutions
**Why here:** The most structured L1 reference — covers all four sections (what are the phases, what is disaggregation, advantages, challenges) in a single 5-minute read. Particularly useful for explaining why different parallelism strategies are optimal for each phase (TP for prefill, PP for decode) and for listing the 4 key implementation challenges with corresponding solutions.

---

## 1. What Are the Two Phases?

### Prefill Phase (Input Processing)

The model reads and processes the entire input text **in parallel**, breaking it into tokens and building a holistic understanding of the context. It generates the **KV (Key-Value) cache** to support subsequent generation.

**Characteristics:**
- Centered around highly parallelised matrix operations
- Computationally intensive but relatively low in memory usage
- Well-suited for high-throughput GPUs or multi-GPU systems
- Efficiently processes input data when parallelised

### Decode Phase (Output Generation)

The model enters an **autoregressive generation** mode: predicts and outputs tokens one by one based on the KV cache from prefill and previously generated content. Each new token depends on the previous result — making the process strictly sequential.

**Characteristics:**
- Completely different computational model from prefill
- Like "word-by-word writing" — one token at a time
- Highly sensitive to latency
- As context expands, KV cache grows → GPU memory usage rapidly increases
- Typically much slower than prefill
- Often underutilises the GPU (memory-bound, not compute-bound)

---

## 2. What Is Prefill-Decode Disaggregation?

Deploy the two stages in **different hardware resource pools** to better match their respective computing characteristics:

- **Prefill pool**: High computing power, suitable for large-scale parallel computing → high-throughput GPUs.
- **Decode pool**: Latency-sensitive, requires larger video memory to store growing KV cache → GPUs with larger video memory and faster memory response.

```
[Prefill Pool: High-compute GPUs]
       ↓ (KV cache transfer)
[Decode Pool: Large-memory-bandwidth GPUs]
       ↓
[Output Response]
```

---

## 3. Advantages of Disaggregation

In traditional architectures, prefill and decode share the same device → prone to interference:

> "Prefill requests can occupy decode resources, leading to increased latency and output lag. Furthermore, the two stages are suited to different parallelization strategies: prefill is suited to tensor parallelism to reduce latency, while decode is suited to pipeline parallelism to increase throughput. This mixed operation is inefficient."

**Key advantages:**

1. **Dedicated Resource Allocation**: Prefill and decode can be independently scheduled and scaled. For multi-turn or agentic workloads with high KV cache reuse, large portions can be reused → less compute demand on prefill → more resources freed for decode.

2. **Parallel Execution**: Prefill and decode no longer block each other → higher concurrency and overall throughput.

3. **Independent Optimisation**: Different strategies (tensor parallelism for prefill, pipeline parallelism for decode) can be applied to each phase independently — matching the goals of low TTFT and low ITL respectively.

---

## 4. The Challenges of Disaggregation (and Solutions)

### Challenge 1: Additional Communication Overhead

KV cache must transfer between prefill and decode GPUs → network latency and bandwidth become bottlenecks.

**Solutions:**
- **High-performance network**: Ultra-low latency, high-bandwidth networks (InfiniBand, NVLink) to reduce KV cache transmission latency.
- **KV Cache partitioning**: Store data in token-based shards, maximising data proximity to decode nodes to reduce cross-node transmission.

### Challenge 2: Increased Scheduling Complexity

The system must coordinate task allocation between the two pools in real time. Risk: decode phase waits in vain for prefill results, or prefill resources are blocked by the decode phase.

**Solutions:**
- **Priority Scheduling**: Prefill tasks prioritised to ensure low TTFT. Decode tasks fairly scheduled to maintain stable throughput (TPOT).
- **Pipeline Parallelism**: Prefill and decode executed in parallel to ensure decode is not idle and prefill resources are not blocked.

### Challenge 3: Complex Implementation

The framework and task scheduler must support pipelining, KV cache output, and asynchronous execution — harder to maintain than single-machine inference.

**Solutions:**
- **Pipelining and async execution**: Task scheduler manages overlapping execution of prefill and decode, avoiding idle waiting and blocking.
- **KV Cache output management**: Ensures correct transmission and use of cached data, improving stability across GPUs and nodes.

### Challenge 4: High Memory Usage

The prefill phase may process multiple requests simultaneously; the decode phase must retain the entire KV cache. Significant memory pressure on both sides.

**Solutions:**
- **Tiered storage**: Hot data in GPU memory, warm/cold data in host memory or NVMe → reduces GPU memory pressure.
- **Memory pool management**: Reuses memory blocks to reduce dynamic allocation overhead.
- **RDMA acceleration**: Fast KV cache transfers in distributed scenarios, alleviating memory pressure during both phases.

---

## Key Takeaways for Layer 19

- The **parallelism strategy mismatch** is the key insight from this article: prefill benefits from tensor parallelism (reduces per-token computation latency); decode benefits from pipeline parallelism (increases throughput by overlapping steps). Trying to use the same parallelism for both is always a compromise.
- **RDMA acceleration** appears in the "solutions" list for three of the four challenges — confirming that high-speed interconnects are not optional for production disaggregation.
- The **tiered storage** solution in Challenge 4 foreshadows Layer 17 (HiCache) — the decode pool's KV cache pressure is what motivates HiCache's CPU/SSD offload.
- The 5-minute structure of this article (definition → disaggregation → advantages → challenges) mirrors the lesson outline for Layer 19 — useful as a quick orientation read before the longer lesson files.
