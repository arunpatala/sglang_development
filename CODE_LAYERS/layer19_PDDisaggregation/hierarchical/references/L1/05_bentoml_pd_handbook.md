# Prefill-Decode Disaggregation — BentoML LLM Inference Handbook

**Source:** https://bentoml.com/llm/inference-optimization/prefill-decode-disaggregation
**Author:** BentoML Team
**Level:** L1–L2 — Handbook-style deployment reference; silver-bullet caveats; KV transfer methods
**Why here:** The most concise handbook-style reference that directly names the failure modes of disaggregation from real-world testing (20–30% performance drop in unsuitable workloads). Also provides the clearest one-sentence answer to "why does co-location make optimisation hard": "since prefill primarily determines the TTFT and decode impacts ITL, collocating them makes it difficult to optimize both metrics simultaneously." Lists SGLang, vLLM, Dynamo, and llm-d as active framework implementations.

---

## Phase Definitions

**Prefill**: Processes the entire sequence in parallel and stores key and value vectors from the attention layers in a KV cache. Because it handles all tokens at once, prefill is **compute-bound**, but not too demanding on GPU memory.

**Decode**: Generates output tokens one at a time, reusing the KV cache built during prefill. Decode requires **fast memory access** but lower compute.

---

## Why Co-location Is the Problem

> "Since prefill primarily determines the TTFT and decode impacts ITL, collocating them makes it difficult to optimize both metrics simultaneously."

In practice, with multiple concurrent requests:
- Each request has its own prefill and decode needs.
- When the GPU is occupied with compute-heavy prefill tasks, decode tasks must wait → increases ITL.
- When memory bus is saturated by decode tasks, prefill is starved of memory reads → increases TTFT.
- Only **one phase can run at a time** under standard scheduling — they compete for the same resources.

---

## Why Disaggregation Makes Sense

The idea: separate these two very different tasks so they don't get in each other's way.

### Key Benefits

1. **Dedicated Resource Allocation**: Prefill and decode can be scheduled and scaled independently on different hardware. For workloads with lots of prompt overlap (multi-turn conversations, agentic workflows), much of the KV cache can be reused → less compute demand on prefill → more resources for decode.

2. **Parallel Execution**: Prefill and decode no longer interfere. Run them in parallel → better concurrency and throughput.

3. **Independent Tuning**: Different optimisation techniques (tensor parallelism vs pipeline parallelism) can be applied to each phase to better meet TTFT and ITL goals.

**Frameworks with active PD disaggregation support:**
- [SGLang](https://github.com/sgl-project/sglang/issues/4655)
- [vLLM](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)
- [NVIDIA Dynamo](https://docs.nvidia.com/dynamo/latest/architecture/disagg_serving.html)
- [llm-d](https://docs.google.com/document/d/1FNN5snmipaTxEA1FGEeSH7Z_kEqskouKD1XYhVyTHr8/)

---

## Disaggregation Isn't Always a Silver Bullet

> "If your workload is too small, or your GPU setup isn't tuned for this approach, performance can drop (by 20–30% in our tests)."

### Three Failure Modes

**1. Thresholds matter — small workloads degrade:**
Performance can drop 20–30% when disaggregation is applied to workloads that don't need it. If the workload is small (short prompts, short outputs), the KV transfer overhead exceeds the interference-elimination gain.

**2. Local prefill can be faster:**
For shorter prompts, or when the decode engine has a high prefix cache hit rate, running prefill locally on the decode worker is often faster and simpler than a full round-trip through the prefill pool.

**3. Data transfer cost:**
Disaggregation requires moving KV caches rapidly and reliably between prefill and decode workers. This means your solution must support fast, low-latency communication protocols that are both hardware- and network-agnostic. Unless the performance gains from disaggregation outweigh the data transfer cost, overall performance can actually degrade.

---

## KV Transfer Methods (Reference List)

Current methods for inter-worker KV cache transfer:
- **NVIDIA Inference Xfer Library (NIXL)** — vendor-agnostic, UCX/GDS backends, integrated with SGLang and vLLM
- **Mooncake Transfer Engine** — RDMA-based, production system for Kimi (Moonshot AI)
- **CXL** — rack-scale shared memory alternative to RDMA (emerging)
- **NVMe-oF** — for cold KV cache loading from storage (not primary disaggregation transfer)

---

## Key Takeaways for Layer 19

- The clearest one-sentence statement of the co-location problem: "Since prefill primarily determines the TTFT and decode impacts ITL, collocating them makes it difficult to optimize both metrics simultaneously."
- **20–30% performance degradation** is the measured cost of applying disaggregation to unsuitable workloads (BentoML's own production tests) — this is the key quantifier for "when not to disaggregate."
- Local prefill (running prefill on the decode worker for short prompts) is not a fallback — it's often the correct choice for prefix-cache-hot or short-prompt workloads.
- The three-item KV transfer method list (NIXL, CXL, NVMe-oF) maps directly to SGLang's `--disaggregation-transfer-backend` options and the broader KV transfer infrastructure landscape.
- This article is the most appropriate response to the question "should we use PD disaggregation?" — it gives the answer (yes in some cases, no in others) without requiring the reader to read DistServe first.
