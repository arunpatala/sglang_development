# DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving

**Source:** https://arxiv.org/abs/2401.09670
**Paper PDF:** https://arxiv.org/pdf/2401.09670
**Conference page:** https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
**Blog post:** https://hao-ai-lab.github.io/blogs/distserve/
**Venue:** USENIX OSDI 2024 (Operating Systems Design and Implementation)
**Authors:** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang (Peking University + UC San Diego)
**Level:** L3 — Foundational systems paper; goodput framing, interference quantification, disaggregation design
**Why here:** DistServe is the paper that defined the disaggregation paradigm for LLM serving. It introduced **goodput** as the correct optimization target, proved that collocated prefill-decode scheduling creates unavoidable interference, and showed that assigning each phase to dedicated GPUs with independent resource allocation can serve 7.4× more requests under latency constraints. Every subsequent disaggregation paper (Splitwise, Mooncake, NVIDIA Dynamo) cites DistServe as the foundational reference.

**BibTeX:**
```bibtex
@inproceedings{zhong2024distserve,
  title  = {DistServe: Disaggregating Prefill and Decoding for
             Goodput-optimized Large Language Model Serving},
  author = {Yinmin Zhong and Shengyu Liu and Junda Chen and Jianbo Hu
            and Yibo Zhu and Xuanzhe Liu and Xin Jin and Hao Zhang},
  booktitle = {18th USENIX Symposium on Operating Systems Design and
               Implementation (OSDI 24)},
  year   = {2024},
  url    = {https://arxiv.org/abs/2401.09670}
}
```

---

## The Core Problem: Why Throughput Is the Wrong Metric

Traditional LLM serving optimises for **throughput** — tokens per second or requests per second. But LLM applications care about two distinct latency dimensions:

- **TTFT** (Time to First Token): latency from request submission to the first output token. Dominated by the prefill phase. Users perceive this as "how long before the model starts responding."
- **TPOT** (Time Per Output Token): latency between consecutive output tokens during decode. Users perceive this as "how fast the model is typing."

**Goodput** is the number of requests completed per second that satisfy **both** TTFT and TPOT SLO constraints simultaneously. A system can have high throughput while having poor goodput if many requests violate one of the SLOs.

### The Colocation Coupling Problem

When prefill and decode share the same GPUs:

1. **Prefill-decode interference**: a large prefill computation wave (processing a 10K-token prompt) runs for hundreds of milliseconds on the GPU. Any decode step scheduled during this window is delayed by the full prefill duration. This spikes TPOT for all currently-decoding requests.

2. **Resource coupling**: the same GPU must handle both compute-bound prefill (needs high FLOP throughput) and memory-bandwidth-bound decode (needs high HBM bandwidth). The parallelism strategy (TP, PP) that optimises prefill degrades decode, and vice versa. There is no single configuration that is optimal for both.

3. **Over-provisioning trap**: to meet both TTFT and TPOT SLOs simultaneously with a collocated system, you must over-provision GPUs — paying for hardware that runs at low utilisation.

---

## DistServe's Solution: Disaggregation

DistServe assigns prefill and decoding computation to **different GPUs**. After the first token is computed, the KV cache is transferred from the prefill instance to the decode instance via a point-to-point transfer.

### Key Design Choices

**Independent resource allocation**: each phase is provisioned separately based on its own SLO requirements:
- TTFT SLO tight → scale prefill pool (more GPUs per instance, or more instances).
- TPOT SLO tight → scale decode pool.
- Both tight → scale both independently.

**Per-phase parallelism**: the prefill pool can use tensor parallelism optimised for compute throughput (TP-4, high FLOP utilisation). The decode pool can use a configuration optimised for memory bandwidth (TP-8, maximise HBM reads per step). These can be different.

**Bandwidth-aware placement**: DistServe places the two phases according to the serving cluster's bandwidth. If prefill and decode are on the same physical rack (high NVLink/InfiniBand bandwidth), the KV transfer cost is amortised. If on different pods, DistServe reduces transfer cost by adjusting the parallelism to minimise KV cache size.

### The KV Transfer

Each KV cache entry is a 2D tensor: `[num_layers, num_heads × head_dim × 2]`. For large models and long prompts, this is hundreds of megabytes. DistServe transfers this after prefill completes, before the decode server begins generation. The transfer latency is a one-time cost per request, amortised over the full decode sequence.

---

## Experimental Results

Evaluated on OPT-13B, OPT-66B, OPT-175B across three application workloads (chatbot, document summarisation, coding). Compared against vLLM-continuous-batching (state-of-the-art collocated baseline):

| Metric | DistServe vs vLLM |
|---|---|
| Max requests served at SLO (goodput) | **7.4× more requests** (updated v3 result) |
| Tightest SLO achievable at same rate | **12.6× tighter** |
| Requests within SLO (>90% guarantee) | Maintained across all workloads |

**Why 7.4×?** The prefill pool handles 100% compute-bound workloads with optimal TP configuration; the decode pool handles 100% memory-bandwidth-bound workloads with optimal configuration. Neither is compromised by the other's requirements.

---

## Connection to SGLang PD Disaggregation

| DistServe Concept | SGLang Equivalent |
|---|---|
| Prefill instance (KV producer) | `--disaggregation-mode prefill` server |
| Decode instance (KV consumer) | `--disaggregation-mode decode` server |
| KV cache transfer | Mooncake RDMA or NIXL |
| Router / orchestration layer | `sglang_router.launch_router --pd-disaggregation` |
| Per-phase resource allocation | Separate `--tp-size`, `--dp-size` per mode |
| Bandwidth-aware placement | `--disaggregation-ib-device` NIC selection |
| Goodput optimisation | `--max-running-requests` on decode server |

---

## Key Takeaways for Layer 19

- **Goodput is the right metric**: throughput ignores TTFT/TPOT SLOs; goodput captures what users actually experience.
- **Interference is structural**: it cannot be eliminated by better scheduling in a collocated system — only by physical separation.
- **Resource coupling is the second problem**: even without interference, a single parallelism configuration cannot be optimal for both compute-bound prefill and memory-bandwidth-bound decode.
- DistServe's 7.4× goodput improvement shows the magnitude of the gain available from disaggregation — not a marginal improvement, but an order-of-magnitude change.
- **OSDI 2024** — the top-tier systems conference. This result was peer-reviewed and replicated.
