# Layer 0 — Benchmark Results

**Date:** 2026-04-19  
**Model:** Qwen/Qwen3-0.6B  
**Hardware:** NVIDIA GeForce RTX 4060 Ti  
**Command:**
```bash
python benchmark.py --layer 0 --port 8100 --num-requests 20
```

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | ShareGPT (`anon8231489123/ShareGPT_Vicuna_unfiltered`) |
| Seed | 42 |
| Requests | 20 |
| Max new tokens | 128 (capped at reference completion length) |
| Mode | Sequential (one request at a time) |
| `use_cache` | `False` — Layer 0 defining constraint |

---

## Summary Metrics

| Metric | Value |
|--------|-------|
| Successful requests | 20 / 20 |
| Total input tokens | 8,461 |
| Total output tokens | 2,109 |
| Total wall time | 49.5 s |
| **Output throughput** | **42.6 tok/s** |
| Total throughput | 213.5 tok/s |
| Request rate | 0.404 req/s |
| Avg latency | 2475 ms |
| p50 latency | 1458 ms |
| p99 latency | 8938 ms |

---

## Per-Request Breakdown

| # | Prompt tokens | Output tokens | Latency | Status |
|---|--------------|---------------|---------|--------|
|  1 |   729 |    121 |   4.1s | ok |
|  2 |     9 |    128 |   1.5s | ok |
|  3 |   110 |    128 |   1.5s | ok |
|  4 |    59 |    128 |   1.5s | ok |
|  5 |    21 |    128 |   1.4s | ok |
|  6 |  1276 |    128 |   6.9s | ok |
|  7 |    57 |    128 |   1.4s | ok |
|  8 |    36 |     86 |   1.0s | ok |
|  9 |    78 |    128 |   1.5s | ok |
| 10 |   667 |    128 |   3.8s | ok |
| 11 |   414 |      6 |   0.1s | ok |
| 12 |   227 |    128 |   2.0s | ok |
| 13 |     7 |    128 |   1.4s | ok |
| 14 |   379 |     29 |   0.6s | ok |
| 15 |   401 |    128 |   2.7s | ok |
| 16 |  1098 |    128 |   6.0s | ok |
| 17 |    17 |    128 |   1.4s | ok |
| 18 |   743 |     42 |   1.3s | ok |
| 19 |  1648 |    128 |   8.9s | ok |
| 20 |   245 |     33 |   0.5s | ok |

---

## What the numbers reveal

**Latency scales with prompt length.**  
With `use_cache=False`, every decode step recomputes attention over the entire sequence — O(prompt_len) extra work per generated token.

**Output throughput is the baseline to beat: 42.6 tok/s.**  
Layer 1 writes the decode loop manually (same cost). Layer 2 adds KV cache and this number should jump significantly.
