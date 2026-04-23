# Splitwise: Efficient Generative LLM Inference Using Phase Splitting

**Source:** https://arxiv.org/abs/2311.18677
**Paper PDF:** https://homes.cs.washington.edu/~patelp1/papers/splitwise-isca24.pdf
**Simulator:** https://github.com/Mutinifni/splitwise-sim
**Venue:** ISCA 2024 (International Symposium on Computer Architecture)
**Authors:** Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, Ricardo Bianchini (University of Washington + Microsoft)
**Level:** L3 — Technical systems paper; production trace characterisation, hardware heterogeneity, cluster-level optimisation
**Why here:** Splitwise is the independent co-discovery of prefill-decode disaggregation, published simultaneously with DistServe (both at top-tier venues in 2024). Its unique contribution is grounding the design in **real production traces from two Microsoft Azure LLM serving services**, giving a workload-distribution-aware analysis that DistServe's synthetic benchmarks cannot provide. It also introduces the insight that different GPU hardware SKUs may be optimal for each phase — enabling cost-optimal cluster design.

**BibTeX:**
```bibtex
@inproceedings{patel2024splitwise,
  title     = {Splitwise: Efficient Generative LLM Inference Using Phase Splitting},
  author    = {Pratyush Patel and Esha Choukse and Chaojie Zhang and Aashaka Shah
               and {\'I}{\~n}igo Goiri and Saeed Maleki and Ricardo Bianchini},
  booktitle = {Proceedings of the 51st Annual International Symposium on
               Computer Architecture (ISCA '24)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2311.18677}
}
```

---

## Production Workload Characterisation

Splitwise's most important contribution is its use of real Azure production traces. Key findings:

### Phase Asymmetry (quantified from production)

| Characteristic | Prompt Computation (Prefill) | Token Generation (Decode) |
|---|---|---|
| Compute profile | Compute-intensive, saturates GPU FLOP throughput | Memory-bandwidth-bound, loads weights + KV per step |
| GPU utilisation | High (matrix multiplications, full batch parallelism) | Low (autoregressive, one token per step) |
| Memory pressure | Low (transient activations only) | High (KV cache grows with sequence length) |
| Power draw | High (compute-bound) | Lower (memory-bandwidth-bound) |
| Duration | Short (single forward pass, milliseconds to seconds) | Long (one step per output token, seconds to minutes) |

### Why Token Generation Underutilises GPU Compute

During decode, each step processes exactly one new token per request. The compute per step scales as `O(1 × model_size)` — independent of sequence length. Even with continuous batching across many requests, the GPU's FLOP capacity is largely idle during decode because the arithmetic intensity is too low to saturate it.

Splitwise quantifies this: **token generation phases do not require the compute capability of the latest GPUs** and can be run on lower-power, lower-cost hardware with equivalent quality.

---

## The Phase Splitting Design

Splitwise splits the two phases of a LLM inference request onto **separate machines**:

```
Client → Router → Prompt Machine → [KV state transfer] → Token Machine → Client
                  (prefill runs)                           (decode runs)
```

The state transfer is the generated KV cache: all K and V tensors for all layers, for all input tokens. For a LLaMA-70B model with a 4K-token prompt and TP=8, this is roughly 8 GB of data per request.

### KV State Transfer Optimisation

Splitwise implements and optimises the KV state transfer using **fast back-plane interconnects** in GPU clusters:
- Within a rack: NVLink or PCIe direct transfers (high bandwidth, low latency)
- Across racks: InfiniBand (lower bandwidth, higher latency)

The paper provides an analysis of when the transfer cost is amortised: for prompts longer than ~500 tokens generating more than ~50 output tokens, disaggregation is always net-positive in latency.

---

## Hardware Heterogeneity Insight

**Splitwise's novel design dimension**: the prompt and token phases can run on **different hardware** tailored to each phase:

| Phase | Optimal hardware | Reason |
|---|---|---|
| Prompt computation (prefill) | Latest GPUs (H100, B200) | Highest FLOP throughput per dollar |
| Token generation (decode) | Older/lower-cost GPUs (A100, H20) or memory-optimised GPUs | High HBM bandwidth per dollar; compute FLOP budget is not the constraint |

**Result from the paper**: by using H100 for prefill + A100 for decode vs all-H100:
- **1.4× higher throughput at 20% lower cost**
- OR **2.35× more throughput at the same cost and power budget**

This hardware heterogeneity insight influenced real cloud deployments — some providers run prefill on latest-gen GPUs while decode runs on previous-generation hardware, reducing serving cost without quality loss.

---

## SplitwiseSim: Cluster-Level Simulation

Splitwise open-sourced SplitwiseSim (https://github.com/Mutinifni/splitwise-sim): a discrete-event simulator for evaluating cluster-level PD disaggregation policies.

The simulator models:
- Heterogeneous hardware pools (different GPU types per phase)
- Variable prompt/output length distributions (from real traces)
- Routing policies (round-robin, load-aware, cache-aware)
- KV transfer latency as a function of payload size and network topology

SplitwiseSim is still used in research to evaluate new routing and scheduling strategies without needing a physical cluster.

---

## The First vLLM KV Transfer Implementation

Splitwise includes a prototype implementation of its KV-cache transfer mechanism in vLLM (**GitHub PR #2809** — the first public implementation of inter-instance KV transfer). This PR became the direct ancestor of vLLM's `vllm/distributed/kv_transfer/` module that all vLLM disaggregation connectors build on today.

---

## Connection to SGLang PD Disaggregation

| Splitwise Concept | SGLang / Production Equivalent |
|---|---|
| Prompt machine (compute-bound) | Prefill worker; can use higher-compute GPU SKUs |
| Token machine (memory-bound) | Decode worker; can use different GPU SKUs |
| KV state transfer (back-plane) | Mooncake RDMA / NIXL transfers |
| Hardware heterogeneity | Different `--mem-fraction-static`, `--max-running-requests` per mode |
| SplitwiseSim routing analysis | `sglang_router` routing policy design |
| Production traces (Azure) | Validates that real traffic distributions benefit from disaggregation |

---

## Key Takeaways for Layer 19

- **Production traces validate disaggregation**: Splitwise's real Azure data shows that prompt/token phase asymmetry is a consistent production phenomenon, not a synthetic benchmark artifact.
- **Token generation is compute-underutilised**: even with continuous batching, decode-phase GPUs are memory-bandwidth-bound, not compute-bound — making latest-gen high-FLOP GPUs overkill for decode.
- **Hardware heterogeneity enables cost optimisation**: running prefill on compute-optimised GPUs and decode on memory-optimised GPUs can halve per-token cost.
- **ISCA 2024** — top-tier computer architecture conference. The paper received the Best Paper Award.
- The vLLM KV transfer prototype in this paper seeded all subsequent connector implementations.
