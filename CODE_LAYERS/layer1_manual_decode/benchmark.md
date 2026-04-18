# Layer 1 — Benchmark Results

**Date:** 2026-04-18  
**Model:** Qwen/Qwen3-0.6B  
**Hardware:** NVIDIA GeForce RTX 4060 Ti  
**Command:**
```bash
python benchmark.py --layer 1 --port 8101 --num-requests 20
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
| Decode loop | Manual — `model.forward()` called per token, `use_cache=False` |

---

## Summary Metrics

| Metric | Value |
|--------|-------|
| Successful requests | 20 / 20 |
| Total input tokens | 8,461 |
| Total output tokens | 1,990 |
| Total wall time | 56.3 s |
| **Output throughput** | **35.3 tok/s** |
| Total throughput | 185.5 tok/s |
| Request rate | 0.355 req/s |
| Avg E2E latency | 2817 ms |
| p50 E2E latency | 1596 ms |
| p99 E2E latency | 10853 ms |
| **Avg TTFT** | **75 ms** |
| p50 TTFT | 22 ms |
| p99 TTFT | 965 ms |
| **Avg TPOT** | **27 ms/tok** |
| p50 TPOT | 19 ms/tok |
| p99 TPOT | 85 ms/tok |

---

## Per-Request Breakdown

| # | Prompt tokens | Output tokens | Latency | TTFT | TPOT | Status |
|---|--------------|---------------|---------|------|------|--------|
|  1 |   729 |     75 |   3.8s | 965 ms | 37 ms | ok |
|  2 |     9 |     49 |   0.6s | 13 ms | 12 ms | ok |
|  3 |   110 |    128 |   1.7s | 13 ms | 13 ms | ok |
|  4 |    59 |    128 |   1.6s | 12 ms | 12 ms | ok |
|  5 |    21 |    128 |   1.5s | 12 ms | 12 ms | ok |
|  6 |  1276 |    128 |   8.3s | 61 ms | 65 ms | ok |
|  7 |    57 |    128 |   1.6s | 12 ms | 12 ms | ok |
|  8 |    36 |     92 |   1.1s | 13 ms | 12 ms | ok |
|  9 |    78 |    128 |   1.6s | 12 ms | 12 ms | ok |
| 10 |   667 |    128 |   4.6s | 33 ms | 36 ms | ok |
| 11 |   414 |      6 |   0.2s | 23 ms | 37 ms | ok |
| 12 |   227 |    128 |   2.4s | 23 ms | 19 ms | ok |
| 13 |     7 |    128 |   1.6s | 14 ms | 12 ms | ok |
| 14 |   379 |     29 |   0.7s | 22 ms | 25 ms | ok |
| 15 |   401 |    128 |   3.1s | 22 ms | 24 ms | ok |
| 16 |  1098 |    128 |   7.3s | 54 ms | 57 ms | ok |
| 17 |    17 |    128 |   1.5s | 13 ms | 12 ms | ok |
| 18 |   743 |     42 |   1.6s | 35 ms | 37 ms | ok |
| 19 |  1648 |    128 |  10.9s | 81 ms | 85 ms | ok |
| 20 |   245 |     33 |   0.7s | 64 ms | 19 ms | ok |

---

## What the numbers reveal

**TTFT scales with prompt length.**  
The first forward pass covers the entire prompt (prefill). Longer prompts = longer TTFT.
With no KV cache, TPOT also grows with sequence length since every decode step
re-reads the full growing sequence.

**Output throughput: 35.3 tok/s.**  
Layer 2 will add `past_key_values` — one change in `model.py` — and TPOT should
drop to near-constant regardless of prompt length.
