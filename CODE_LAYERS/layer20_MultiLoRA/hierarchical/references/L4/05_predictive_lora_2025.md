# Predictive-LoRA: A Proactive and Fragmentation-Aware Serverless Inference System for LLMs

**Source:** https://arxiv.org/abs/2512.20210
**Paper PDF:** https://arxiv.org/pdf/2512.20210
**Authors:** Yuqi Tang and others
**Submitted:** December 23, 2025
**Level:** L4 — Advanced research system; predictive adapter prefetching + page-based memory management
**Why here:** Predictive-LoRA addresses two specific problems in serverless LoRA serving that ServerlessLoRA did not solve: (1) cold-start latency from *reactive* adapter loading (you load after a request arrives, not before), and (2) GPU memory fragmentation from adapters of different ranks. P-LoRA's LSTM traffic predictor and page-based memory management are the most sophisticated memory management approach in the surveyed literature, achieving 1.52× throughput over S-LoRA while reducing TTFT by 35%.

**BibTeX:**
```bibtex
@article{tang2025predictive,
  title  = {{Predictive-LoRA}: A Proactive and Fragmentation-Aware
            Serverless Inference System for {LLMs}},
  author = {Yuqi Tang and others},
  journal = {arXiv preprint arXiv:2512.20210},
  year   = {2025},
  url    = {https://arxiv.org/abs/2512.20210}
}
```

---

## Abstract

The serverless computing paradigm offers compelling advantages for deploying LLM inference services, including elastic scaling and pay-per-use billing. However, serving multiple fine-tuned LLMs via LoRA in serverless environments faces critical challenges:

1. **Reactive adapter loading** causes significant cold-start latency
2. **Frequent adapter swapping** leads to severe GPU memory fragmentation

We present Predictive-LoRA (P-LoRA), a proactive and fragmentation-aware serverless inference system. P-LoRA introduces:

1. A **lightweight LSTM-based traffic predictor** that forecasts adapter demand and proactively prefetches hot adapters from host memory to GPU, reducing cold start latency by up to **68%**
2. A **page-based adapter memory management mechanism** inspired by OS virtual memory, which keeps GPU memory utilization above **87%** even under heterogeneous adapter ranks

Results: P-LoRA achieves **1.52× higher throughput than S-LoRA** while reducing the average TTFT by **35%** under high concurrency scenarios.

---

## Problem 1: Reactive Loading Causes Unnecessary Cold Starts

S-LoRA and similar systems use a reactive loading strategy:

```
Request for adapter_A arrives
    → Check GPU VRAM: not present (cold)
    → Load adapter_A from CPU/disk to GPU    ← 100-500ms wait
    → Process request
```

This is reactive — the system only starts loading when a request arrives. If adapter access patterns are predictable (e.g., adapter_A is popular every morning), the system can predict the need and prefetch proactively:

```
Traffic predictor: "adapter_A will be needed soon"
    → Prefetch adapter_A to GPU VRAM now (during idle time)
    ...
Request for adapter_A arrives
    → Check GPU VRAM: PRESENT (warm hit!)
    → Process request immediately
```

---

## The LSTM Traffic Predictor

P-LoRA uses a **lightweight LSTM (Long Short-Term Memory)** neural network to predict future adapter demand.

### Input features

For each time window `t`:
- Request arrival rate per adapter: `r_i(t)` for each adapter `i`
- Cumulative request count per adapter: `c_i(t)`
- Time-of-day features: hour, day-of-week (captures daily/weekly patterns)
- Recent access history: last `k` time windows

### Output

- Predicted request rate for each adapter in the next time window: `r̂_i(t+1)`
- Adapters predicted to be "hot" (above a threshold) are prefetched

### Model size

The LSTM is **intentionally small** (few hundred parameters) to minimize overhead:
- Inference takes ~1ms on CPU
- Model is retrained periodically using online learning from recent traffic data

### Why LSTM over simpler predictors?

| Predictor | Can model | Limitation |
|---|---|---|
| LRU (recency) | Temporal locality | Cannot predict future demand |
| Frequency (LFU) | Access frequency | Cannot adapt to changing patterns |
| Moving average | Short-term trends | Cannot capture long-term cycles |
| **LSTM** | **Long-range dependencies, cyclical patterns, trend shifts** | Slight training overhead |

The LSTM can model patterns like "adapter_A is popular 9am–5pm on weekdays" and "adapter_B spikes after marketing emails."

---

## Problem 2: Memory Fragmentation from Variable-Rank Adapters

Different adapters have different ranks (`r = 4, 8, 16, 32, 64, 128`). With naive allocation:

```
GPU VRAM after loading/evicting adapters:

|──────|  ←  adapter_A, rank=8  (100MB)
|      |  ←  FREE (after adapter_B rank=64 evicted)  (800MB)
|──────|  ←  adapter_C, rank=16  (200MB)
|      |  ←  FREE (after adapter_D rank=32 evicted)  (400MB)
|──────|  ←  adapter_E, rank=4   (50MB)

Total free: 1.2GB, but largest contiguous: 800MB
If adapter_F needs 1GB: OOM despite having enough total free memory!
```

This is the classic memory fragmentation problem.

---

## Page-Based Adapter Memory Management

Inspired by OS virtual memory, P-LoRA uses **fixed-size pages** to store adapter weights:

```python
PAGE_SIZE = 64MB  # fixed, chosen to balance fragmentation vs overhead

class AdapterMemoryPool:
    def __init__(self, total_vram):
        self.pages = [Page(i) for i in range(total_vram // PAGE_SIZE)]
        self.free_pages = deque(range(len(self.pages)))
        self.adapter_page_map = {}  # adapter_id → [page_ids]

    def allocate(self, adapter_id, adapter_size_bytes):
        n_pages = math.ceil(adapter_size_bytes / PAGE_SIZE)
        if len(self.free_pages) < n_pages:
            self.evict_lru_adapter()  # free pages via LRU eviction
        page_ids = [self.free_pages.popleft() for _ in range(n_pages)]
        self.adapter_page_map[adapter_id] = page_ids

    def evict(self, adapter_id):
        page_ids = self.adapter_page_map.pop(adapter_id)
        self.free_pages.extend(page_ids)  # pages returned to free pool
```

**Key benefit:** Pages are equal-size, so any page can be used for any adapter. No fragmentation — free pages can always be combined for larger allocations.

**VRAM utilization:** P-LoRA maintains >87% VRAM utilization even with heterogeneous ranks (rank-4 to rank-128 adapters in the same pool).

---

## Combined P-LoRA Architecture

```
                    LSTM Predictor
                         │
              predicts hot adapters
                         │
              ┌──────────▼──────────┐
              │  Prefetch Scheduler  │
              │  (proactive loading)  │
              └──────────┬──────────┘
                         │
                         ▼
         Page-Based GPU Memory Pool
         ┌──────────────────────────┐
         │ [page][page][page]...    │  ← adapter pages
         │ [page][page][page]...    │  ← KV cache pages
         │ [page][page][page]...    │  ← free pages
         └──────────────────────────┘
                         │
              requests served from pool
```

---

## Evaluation on Azure Functions Trace

Microsoft Azure Functions traffic traces (real serverless workload):

| System | TTFT | Throughput | VRAM utilization |
|---|---|---|---|
| S-LoRA | 1× (baseline) | 1× | 60-70% |
| P-LoRA | **0.65× (35% lower)** | **1.52×** | **87%+** |

Cold start reduction: 68% fewer cold starts thanks to LSTM-based prefetching.

---

## Relevance to Layer 20

P-LoRA's page-based memory management is the most sophisticated version of what Layer 20 avoids entirely:

| Feature | Layer 20 | P-LoRA |
|---|---|---|
| Adapter memory management | None (static GPU allocation) | Page-based pool with LRU eviction |
| Cold start handling | None (no cold starts) | LSTM predictor for proactive prefetch |
| Fragmentation | None (single adapter, fixed size) | Page-based (no fragmentation by design) |
| Prediction | None | LSTM traffic predictor |

The page-based memory management described here is essentially the same concept as S-LoRA's Unified Paging, but with explicit page size management (S-LoRA uses variable-size chunks). P-LoRA's contribution is the predictive prefetching layer on top.
