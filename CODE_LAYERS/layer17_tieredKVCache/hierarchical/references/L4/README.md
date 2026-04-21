# L4 References: Tiered KV Cache

**Level:** L4 — Advanced / research papers

**Reader profile:** Wants to understand the broader research context: what problems inspired HiCache's design, what alternatives exist, and what the theoretical limits of CPU offloading are. Comfortable reading systems papers.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_infinigen_osdi24.md` | USENIX OSDI 2024 | InfiniGen: speculative prefetching of only important KV pages from CPU — explains why prefetching all pages wastes PCIe bandwidth; 3× speedup over prior methods. |
| 02 | `02_impress_fast25.md` | USENIX FAST 2025 | IMPRESS: importance-informed three-tier GPU/CPU/SSD storage; KV reordering and score-based cache management; 2.8× TTFT improvement; strong systems-storage venue. |
| 03 | `03_kvcrit_pcie_bottleneck.md` | arXiv Dec 2025 | Analytical framework deriving κ_crit: the critical cached-to-prefill ratio where prefill becomes memory-bound. 99% of offloaded-inference latency is PCIe transfers. Explains why `load_back_duration_seconds` P99 is the most important metric to watch. |

---

## Recommended reading order

**Fast path (60 min):** 03 → 01
- 03 first: the PCIe bottleneck analysis provides the theoretical foundation for why all of HiCache's DMA optimizations matter.
- 01: InfiniGen shows how to go beyond simple CPU offloading by prefetching selectively.

**Thorough path (2–3 hours):** 03 → 01 → 02
- 02: IMPRESS adds the importance-aware selection angle and a strong storage-system perspective.

---

## How these map to Layer 17

| Layer 17 lesson | Most relevant L4 reference |
|---|---|
| `01_eviction_problem.md` — cost of re-prefill | 03 (κ_crit shows exactly when offloading helps vs hurts) |
| `03_host_pool.md` — why DMA optimizations matter | 03 (PCIe as the bottleneck; 99% latency in transfers) |
| `04_storage_backend.md` — prefetch policy design | 01 (InfiniGen's speculative prefetch; only fetch important pages) |
| `05_observability.md` — load-back latency P99 alert | 03 (analytical justification: PCIe dominates inference time when κ_ratio ≫ κ_crit) |
