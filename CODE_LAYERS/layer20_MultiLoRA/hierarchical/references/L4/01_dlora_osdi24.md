# dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving

**Source:** https://www.usenix.org/conference/osdi24/presentation/wu-bingyang
**Paper PDF:** https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf
**Authors:** Bingyang Wu, Ruidong Zhu, Zili Zhang (Peking University); Peng Sun (Shanghai AI Lab); Xuanzhe Liu, Xin Jin (Peking University)
**Venue:** USENIX OSDI 2024 (Operating Systems Design and Implementation), pages 911–927
**Published:** July 2024
**Level:** L4 — Advanced research system; dynamic merge/unmerge + cross-worker request migration
**Why here:** dLoRA shows that the binary choice between "merge adapter into base model" vs "keep separate for batching" is false — the optimal decision depends on workload characteristics and should be made dynamically. This is the key insight that S-LoRA missed: skewed adapter distributions can make merging *more* efficient than batching. dLoRA's credit-based algorithm is the production-grade version of a decision rule that Layer 20 bakes in at startup (always keep separate).

**BibTeX:**
```bibtex
@inproceedings{wu2024dlora,
  title  = {{dLoRA}: Dynamically Orchestrating Requests and Adapters for {LoRA} {LLM} Serving},
  author = {Bingyang Wu and Ruidong Zhu and Zili Zhang and Peng Sun
            and Xuanzhe Liu and Xin Jin},
  booktitle = {18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)},
  pages  = {911--927},
  year   = {2024},
  url    = {https://www.usenix.org/conference/osdi24/presentation/wu-bingyang}
}
```

---

## Abstract

Low-rank adaptation (LoRA) is a popular approach to fine-tune pre-trained large language models (LLMs) to specific domains. This paper introduces dLoRA, an inference serving system for LoRA models. dLoRA achieves high serving efficiency by dynamically orchestrating requests and LoRA adapters in terms of two aspects:

1. **Dynamically merge and unmerge adapters** with the base model
2. **Dynamically migrate requests and adapters** between different worker replicas

These capabilities are designed based on two insights:
- Despite the allure of batching without merging, it is **not always beneficial to unmerge**, especially when the types of requests are skewed.
- The autoregressive nature of LLM requests introduces **load imbalance between worker replicas** due to varying input and output lengths, even if input requests are distributed uniformly.

**Results:** dLoRA improves throughput by up to **57.9× over vLLM** and **26.0× over HuggingFace PEFT**. Compared to S-LoRA, dLoRA achieves up to **1.8× lower average latency**.

---

## The Merge vs. Unmerge Decision

S-LoRA and Punica always serve adapters in the *unmerged* mode:
```
h = W₀x + BAx · scaling   # compute separately, batch with SGMV
```

An alternative is to *merge* the adapter into the base weights and serve as a single model:
```
W = W₀ + BA · scaling
h = Wx                     # single GEMM, no overhead
```

### When is merging better?

**Merged serving** benefits:
- Zero overhead: one GEMM instead of two
- Optimal for batches where all requests use the same adapter (highly skewed distribution)
- No SGMV kernel required

**Merged serving** costs:
- Must merge/unmerge when adapter changes (takes time proportional to number of layers × parameter size)
- Only serves one adapter at a time efficiently (no mixed batching)

**Unmerged serving** benefits:
- Mixed batches: base model + LoRA requests in same batch
- Multiple adapters in same batch (Punica/SGMV)

**Unmerged serving** costs:
- Extra SGMV computation for every LoRA-token
- Memory for separate A and B matrices

**dLoRA's insight:** In real workloads, adapter request distribution is often highly skewed (some adapters are "hot" with many requests, others are "cold"). For hot adapters, merging and dedicating a worker replica is more efficient than unmerged batching with SGMV overhead.

---

## The Credit-Based Batching Algorithm

dLoRA uses a **credit system** to decide when to merge/unmerge:

```python
# Simplified credit-based logic
credits[adapter_id] += 1 per request arriving for this adapter
credits[adapter_id] -= decay_rate per time unit

if credits[adapter_id] > MERGE_THRESHOLD:
    # This adapter has high enough traffic → merge and dedicate a replica
    merge(adapter_id, base_model)
    route_all_requests_for(adapter_id) → merged_replica

elif credits[adapter_id] < UNMERGE_THRESHOLD:
    # Traffic dropped → unmerge and return to shared SGMV serving
    unmerge(adapter_id, base_model)
    route_requests_via_sgmv()
```

This adapts to workload shifts in real time. A credit accumulates when requests arrive and decays when they don't — analogous to a leaky bucket algorithm.

---

## Request-Adapter Co-Migration

**Problem:** With multiple worker replicas (each holding a copy of the base model), requests are initially distributed uniformly. However, due to varying output lengths (some requests generate 10 tokens, others 1000), replicas become load-imbalanced over time.

**Solution:** dLoRA implements **request-adapter co-migration** — moving both the in-progress request state (KV cache) and the adapter weights together from an overloaded replica to an underloaded one.

### Migration cost analysis

- KV cache size: proportional to sequence length (can be large)
- Adapter size: small (typically 10–200MB for LoRA adapters)
- Total migration cost: dominated by KV cache for long sequences

dLoRA includes a **migration decision function** that triggers migration only when the expected latency reduction exceeds the migration cost, preventing thrashing.

---

## Architecture

```
                   Request Router
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
   [Replica 0]    [Replica 1]    [Replica 2]
   base + A₁,B₁   base (SGMV)   base + A₂,B₂
   (merged)        (unmerged)     (merged)
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                 Co-Migration Manager
                 (moves requests + KV caches
                  between replicas as needed)
```

---

## Comparison to S-LoRA

| Aspect | S-LoRA | dLoRA |
|---|---|---|
| Serving mode | Always unmerged + SGMV | Dynamic: merge or unmerge based on traffic |
| Load balancing | Static routing | Dynamic request-adapter co-migration |
| Workload assumption | Uniform adapter access | Skewed access patterns |
| Average latency | Higher (SGMV overhead on hot adapters) | Up to 1.8× lower |
| Throughput | 4× vs naive | 57.9× vs vLLM |

---

## Relevance to Layer 20

Layer 20 hardcodes the "always unmerged" decision — which is the choice dLoRA would make for a system with one static adapter and mixed base/LoRA traffic. dLoRA generalizes this to N adapters with dynamic switching.

The credit-based merge/unmerge decision is a natural extension of Layer 20's binary `lora_mask` (0 or 1 per token):

```
Layer 20: lora_mask per token (0.0 or 1.0)  →  static unmerged serving
dLoRA:    credits per adapter               →  dynamic merge decision
```

For Layer 20's use case (one adapter, small batch, mixed traffic), the masked unmerged approach is correct and simple. dLoRA matters when scaling to hundreds of adapters with real traffic patterns.
