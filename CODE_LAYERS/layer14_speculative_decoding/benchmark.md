## Layer 13 — Speculative Decoding Benchmark

**Config**: 8 prompts · max_tokens=100 · N=5 draft tokens · page_size=16
**Models**: target=`Qwen/Qwen3-1.7B`  draft=`Qwen/Qwen3-0.6B`

| Metric | Target-only | Spec-decode |
|--------|-------------|-------------|
| Total wall time | 15.29s | 38.30s |
| Output tok/s | 52.3 | 21.3 |
| TTFT avg / p95 | 20ms / 21ms | 37ms / 41ms |
| Avg output tokens | 100.0 | 101.8 |
| Acceptance rate | — | 28.6% |
| Tokens per step | — | 2.43 (max=6) |
| **Speedup** | 1.00× | **0.41×** |