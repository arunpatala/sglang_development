# Layer 2 — Benchmark Results

**Date:** 2026-04-18  
**Model:** Qwen/Qwen3-0.6B  
**Hardware:** NVIDIA GeForce RTX 4060 Ti  
**Command:**
```bash
python benchmark.py --layer 2 --port 8102 --num-requests 20
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
| Decode loop | Manual — prefill once, decode with `past_key_values` reuse |

---

## Summary Metrics

| Metric | Value |
|--------|-------|
| Successful requests | 20 / 20 |
| Total input tokens | 8,461 |
| Total output tokens | 2,045 |
| Total wall time | 26.1 s |
| **Output throughput** | **78.4 tok/s** |
| Total throughput | 402.8 tok/s |
| Request rate | 0.767 req/s |
| Avg E2E latency | 1304 ms |
| p50 E2E latency | 1535 ms |
| p99 E2E latency | 2015 ms |
| **Avg TTFT** | **40 ms** |
| p50 TTFT | 18 ms |
| p99 TTFT | 313 ms |
| **Avg TPOT** | **12 ms/tok** |
| p50 TPOT | 12 ms/tok |
| p99 TPOT | 15 ms/tok |

---

## Per-Request Breakdown

| # | Prompt tokens | Output tokens | Latency | TTFT | TPOT | Status |
|---|--------------|---------------|---------|------|------|--------|
|  1 |   729 |     78 |   1.3s | 313 ms | 13 ms | ok |
|  2 |     9 |     89 |   1.1s | 13 ms | 12 ms | ok |
|  3 |   110 |    128 |   1.5s | 13 ms | 12 ms | ok |
|  4 |    59 |    128 |   1.5s | 13 ms | 12 ms | ok |
|  5 |    21 |    128 |   1.5s | 18 ms | 12 ms | ok |
|  6 |  1276 |    128 |   1.6s | 63 ms | 12 ms | ok |
|  7 |    57 |    128 |   1.5s | 12 ms | 12 ms | ok |
|  8 |    36 |    104 |   1.2s | 13 ms | 12 ms | ok |
|  9 |    78 |    128 |   1.7s | 12 ms | 13 ms | ok |
| 10 |   667 |    128 |   1.6s | 33 ms | 12 ms | ok |
| 11 |   414 |      6 |   0.1s | 23 ms | 13 ms | ok |
| 12 |   227 |    128 |   1.6s | 14 ms | 12 ms | ok |
| 13 |     7 |    128 |   1.5s | 12 ms | 12 ms | ok |
| 14 |   379 |     29 |   0.4s | 22 ms | 12 ms | ok |
| 15 |   401 |    128 |   1.6s | 23 ms | 13 ms | ok |
| 16 |  1098 |    128 |   1.6s | 53 ms | 12 ms | ok |
| 17 |    17 |    128 |   1.7s | 13 ms | 13 ms | ok |
| 18 |   743 |     42 |   0.5s | 35 ms | 12 ms | ok |
| 19 |  1648 |    128 |   2.0s | 82 ms | 15 ms | ok |
| 20 |   245 |     33 |   0.4s | 18 ms | 12 ms | ok |

---

## What the numbers reveal

**TPOT is now near-constant regardless of prompt length.**  
Each decode step only processes one new token against the cached K/V — O(seq_len)
attention instead of O(seq_len²). The cache grows by one row per step but the
compute per step is dominated by the linear layers, not quadratic attention.

**TTFT still scales with prompt length** — the prefill pass must still attend over
the full prompt. That is unavoidable without chunked prefill (a later layer).

**Output throughput: 78.4 tok/s.**  
Compare TPOT here vs Layer 1: it should be near-constant (~12ms) vs growing
with prompt length in Layer 1. That is the KV cache speedup made visible.
