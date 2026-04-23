# Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs

**Source:** https://lmsys.org/blog/2025-05-05-large-scale-ep
**Author:** SGLang Team (LMSYS Org), May 5, 2025
**Level:** L2–L3 — Production deployment case study
**Why here:** The canonical large-scale production deployment of SGLang PD disaggregation. Describes how DeepSeek-V3 (671B MoE) was deployed across 12 nodes × 8 H100 GPUs using PD disaggregation + full expert parallelism (EP), achieving 52.3k input and 22.3k output tokens/second per node. Layer 19's production deployment section draws its multi-node recipe and throughput numbers from this post.

---

## Cluster Configuration

- **Hardware**: 12 nodes × 8 H100 GPUs = 96 H100 GPUs total
- **Model**: DeepSeek-V3 (671B parameters, MoE architecture — MLA + 256 experts)
- **Deployment split**: 3 prefill nodes (24 GPUs) + 9 decode nodes (72 GPUs)
- **Network**: InfiniBand interconnect between nodes; Mooncake as the KV transfer engine

---

## Results

| Metric | Value |
|---|---|
| Input throughput per node | **52,300 tokens/second** |
| Output throughput per node | **22,300 tokens/second** |
| Input sequence length | 2,000 tokens |
| Configuration | PD disaggregation + DeepEP + EPLB |

> "To the best of our knowledge, this is the highest reported throughput for DeepSeek-V3 serving at that time."

---

## How PD Disaggregation Handles DeepSeek-V3's MoE

DeepSeek-V3 uses Mixture-of-Experts (MoE) with 256 experts per layer, only 8 activated per token. Without PD disaggregation, **DeepEP's `auto` mode faces a fundamental limitation**: it cannot simultaneously support all dispatch patterns needed for both prefill and decode phases. Specifically:

- **Prefill phase** benefits from a "normal" dispatch pattern (compute-intensive, large batch, high-throughput expert routing).
- **Decode phase** benefits from a "low-latency" dispatch pattern (memory-intensive, small batch, latency-sensitive routing).

These two patterns conflict when sharing the same GPU workers. PD disaggregation resolves this by assigning each phase to dedicated workers — prefill nodes use the normal pattern, decode nodes use the low-latency pattern.

---

## Implementation: The Handshake Protocol

The PD disaggregation design in SGLang interleaves execution between a Prefill Server and a Decode Server:

```
Step 1: Decode Server pre-allocates KV cache pages for the incoming request
         → signals Prefill Server with KV page indices

Step 2: Prefill Server receives KV indices
         → runs model forward pass
         → writes KV cache directly into Decode Server's pre-allocated pages (via RDMA)
         → sends completion notification

Step 3: Decode Server receives KV cache
         → begins autoregressive generation loop
         → streams output tokens back to client
```

The critical design choice: the **Decode Server pre-allocates before prefill begins**. This avoids a race condition where the decode server might not have space when the prefill server tries to write.

---

## Expert Parallelism Integration

### DeepEP (Expert Parallelism Communication Library)

DeepEP is DeepSeek's communication library for streamlining expert parallelism in MoE models. SGLang integrates it via `--moe-a2a-backend deepep`.

- **Normal mode** (used in prefill): high-throughput all-to-all for large batches; maximizes expert computation overlap with communication.
- **Low-latency mode** (used in decode): minimizes dispatch latency for small batches (often single-token); sacrifices throughput for responsiveness.

Without PD disaggregation, the server must switch between modes per batch — which cannot be done optimally when prefill and decode batches are interleaved.

### EPLB (Expert Parallelism Load Balancing)

Expert load is inherently uneven — some experts are activated much more frequently than others. EPLB redistributes expert computation to balance GPU utilization. The SGLang blog uses a distribution matching the observed input/output data pattern for EPLB configuration.

---

## Benchmark Configurations Compared

| Configuration | Description |
|---|---|
| SGLang TP16 × 6 | Baseline: every 2 nodes run DeepSeek-V3 independently (TP=16, DP attention) |
| SGLang PD Disaggregation | PD disagg + full EP optimization; 3 prefill + 9 decode nodes |
| SGLang PD Disagg + simulated MTP | Simulates multi-token prediction by doubling batch size + halving KV length |
| Aggregated baseline | Traditional collocated serving |

PD disaggregation with full EP significantly outperforms all baseline configurations on combined throughput (input + output tokens/sec/node).

---

## Key Takeaways for Layer 19

- PD disaggregation is **essential** for MoE models like DeepSeek-V3: expert parallelism dispatch patterns are incompatible between prefill and decode phases when colocated.
- The **pre-allocation handshake** is the key correctness mechanism: decode allocates pages first, then prefill writes to them.
- A 3:9 prefill-to-decode GPU ratio is the production tuning point for DeepSeek-V3 at this scale — decode is more resource-hungry because it runs continuously while prefill is bursty.
- **Mooncake + InfiniBand** is the production-recommended transfer stack for multi-node DeepSeek deployments.
- The throughput gains (52.3k input tokens/sec) are only achievable with both PD disagg **and** expert parallelism — neither alone achieves this.
