# CaraServe: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference

**Source:** https://arxiv.org/abs/2401.11240
**Paper PDF:** https://arxiv.org/pdf/2401.11240
**Authors:** Suyi Li, Hanfeng Lu, Tianyuan Wu, Minchen Yu, Ding Ding, Hao Feng, Tingwei Lu, Zhipeng Di, Chao Yu, Bin Dong, Wei Wang, Ruichuan Chen
**Submitted:** January 20, 2024
**Level:** L4 — Advanced research system; CPU-assisted cold-start elimination and rank-aware scheduling
**Why here:** CaraServe addresses the cold-start problem that S-LoRA and Punica ignored: when an adapter must be loaded from CPU to GPU, there is a substantial delay before token generation can begin. CaraServe solves this by using the CPU to run prefill *while* the adapter is loading to GPU. This is a different dimension of the serving challenge than Layer 20 addresses, but is critical for production systems where adapter catalogs are large and cold starts are frequent.

**BibTeX:**
```bibtex
@article{li2024caraserve,
  title  = {{CaraServe}: {CPU}-Assisted and Rank-Aware {LoRA} Serving
            for Generative {LLM} Inference},
  author = {Suyi Li and Hanfeng Lu and Tianyuan Wu and Minchen Yu and Ding Ding
            and Hao Feng and Tingwei Lu and Zhipeng Di and Chao Yu and Bin Dong
            and Wei Wang and Ruichuan Chen},
  journal = {arXiv preprint arXiv:2401.11240},
  year   = {2024},
  url    = {https://arxiv.org/abs/2401.11240}
}
```

---

## Abstract

Pre-trained large language models often need specialization for domain-specific tasks. CaraServe presents a system that efficiently serves many LoRA adapters derived from a common base model. CaraServe maintains the base model on GPUs and dynamically loads activated LoRA adapters from main memory. As GPU loading results in a **cold-start that substantially delays token generation**, CaraServe employs a **CPU-assisted approach**: it early starts the activated adapters on CPUs for prefilling as they are being loaded onto GPUs; after loading completes, it then switches to the GPUs for generative LoRA inference. CaraServe also employs a **rank-aware scheduling algorithm** to optimally schedule heterogeneous LoRA requests for maximum SLO attainment. Results demonstrate that CaraServe can speed up the average request serving latency by up to **1.4× and achieve SLO attainment of up to 99%**.

---

## The Cold-Start Problem

### What happens without CaraServe

When a request arrives for an adapter not currently in GPU VRAM:

```
Time →
|──────────────────|──────────────────|──────────────────|
| Adapter loading  | GPU prefill      | GPU decode       |
| (CPU→GPU xfer)   | (starts AFTER    | ...              |
| 100-500ms        |  loading done)   |                  |
|──────────────────|──────────────────|──────────────────|

TTFT = adapter_load_time + prefill_time
     = (potentially large cold start)
```

For large adapters (high rank, many target modules) or slow PCIe bandwidth, adapter loading can dominate TTFT.

### CaraServe's approach

```
Time →
|──────────────────|──────────────────|
| Adapter loading  | GPU prefill      |  GPU decode...
| (CPU→GPU xfer)   |                  |
|──────────────────|                  |
| CPU prefill      |──────────────────|
| (starts concurr- | switch to GPU    |
|  ently with load)|                  |
|──────────────────|──────────────────|

TTFT = max(adapter_load_time, cpu_prefill_time) ≈ reduced
```

The CPU runs prefill on its own compute while the GPU transfers weights. Since CPU prefill is slower than GPU prefill, the overlap is only beneficial if the adapter loading takes longer than the time it would take the GPU to do prefill alone. CaraServe's scheduling algorithm accounts for this.

---

## CPU-Assisted Prefill Design

### Architecture

- **CPU DRAM:** stores all adapter A and B matrices (full adapter catalog)
- **GPU VRAM:** hot adapter pool (recently used adapters)
- **PCIe bus:** transfers adapters CPU→GPU

For a new request with a cold adapter:
1. Start H2D transfer (CPU→GPU, ~100MB/s on PCIe 4.0 for a typical LoRA)
2. *Simultaneously* run prefill on CPU using CPU-resident adapter weights
3. When H2D completes, hand off intermediate KV states to GPU
4. GPU continues with decode

### Challenges and solutions

**Challenge 1: CPU prefill is slow**
- CPU is ~10× slower at attention computation
- Solution: only use CPU prefill if `adapter_load_time > cpu_prefill_time`. CaraServe profiler determines this threshold per adapter rank and prompt length.

**Challenge 2: KV state transfer CPU→GPU**
- After CPU prefill, the KV cache (for all prefill tokens) must be transferred to GPU
- Solution: CaraServe pre-allocates pinned (page-locked) CPU memory for the KV cache to enable fast DMA transfers

**Challenge 3: Synchronization**
- CPU prefill may finish before or after H2D completes
- Solution: fine-grained synchronization barrier; GPU decode starts as soon as both (a) adapter is loaded AND (b) KV prefill data is available

---

## Rank-Aware Scheduling

LoRA adapters have heterogeneous ranks (`r = 4, 8, 16, 64, 128, ...`). This creates variable computation and memory costs per adapter.

### The scheduling problem

Given a batch of requests with adapters of different ranks, what order should they be processed to maximize the fraction meeting their TTFT SLO?

### CaraServe's rank-aware priority algorithm

```python
def compute_priority(request):
    adapter_rank = get_rank(request.adapter_id)
    slo_deadline = request.arrival_time + request.ttft_slo
    time_remaining = slo_deadline - current_time

    # Rank-aware urgency: high rank adapters need more compute time
    # and may need longer loading time → schedule them earlier
    compute_cost_estimate = estimate_cost(adapter_rank, request.prompt_length)
    urgency = compute_cost_estimate / time_remaining

    return urgency  # higher urgency → scheduled first
```

Key insight: a rank-64 adapter request arriving at the same time as a rank-8 request requires more processing. If both have the same SLO deadline, the rank-64 request should be scheduled first because it needs more lead time.

---

## Evaluation Results

On A100 GPU with LLaMA-7B:

| Metric | S-LoRA | CaraServe |
|---|---|---|
| Average latency speedup | 1× | **1.4×** |
| SLO attainment (strict TTFT) | 85% | **99%** |
| Cold-start handling | Request queued | CPU-assisted immediate start |

The 99% SLO attainment is particularly impressive for a system serving thousands of adapters, where cold starts are frequent.

---

## Memory Architecture Comparison

| System | Adapter storage | Cold start handling | KV cache management |
|---|---|---|---|
| Layer 20 | GPU only (1 adapter) | None (no cold starts) | Not paged |
| S-LoRA | CPU→GPU on demand | Request queued during load | Unified Paging |
| CaraServe | CPU→GPU on demand | CPU-assisted prefill | Pinned memory |
| dLoRA | GPU pool (replica per merged adapter) | Migration-aware | Standard paged |

---

## Relevance to Layer 20

Layer 20 has zero cold-start problem: the single adapter is loaded at startup and permanently resident in GPU VRAM. CaraServe is relevant when extending Layer 20 to support a dynamic adapter pool, where cold starts become a real problem.

The CPU-assisted approach is particularly relevant for:
- Large adapter pools (>100 adapters) where VRAM cannot hold all adapters simultaneously
- Low-rank adapters with long prompts (where CPU prefill can keep up with GPU loading)
- Production deployments where SLO guarantees are required

For Layer 20's minimal design, the equivalent of CaraServe's optimization would be startup pre-loading, which is already done by `LoRAAdapter.__init__()`.
