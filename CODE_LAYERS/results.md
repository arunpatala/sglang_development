# Layer Benchmark Results

All runs: 20 requests, sequential, ShareGPT dataset (seed=42), Qwen3-0.6B, RTX 4060 Ti.

Each layer has its own `benchmark.py`. Run it from within that layer's directory.

| Layer | Mode | OK/N | Input tok | Output tok | Wall time | Output tok/s | Total tok/s | Avg E2E (ms) | Avg TTFT (ms) | Avg TPOT (ms) |
|-------|------|------|-----------|------------|-----------|--------------|-------------|--------------|---------------|---------------|
| 0 — naive_inference | sequential | 20/20 | 8461 | 2109 | 53.1s | 39.7 | 198.9 | 2657 | — | — |
| 1 — manual_decode | sequential | 20/20 | 8461 | 1990 | 56.3s | 35.3 | 185.5 | 2817 | 75 | 27 |
| 2 — kv_cache | sequential | 20/20 | 8461 | 2045 | 26.1s | 78.4 | 402.8 | 1304 | 40 | 12 |
| 3 — static_batching | batch sweep | — | — | — | — | — | — | — | — | — |

## What to observe

- **Layer 0 → Layer 1**: Adding `use_cache=True` eliminates redundant KV recomputation. Expect output tok/s to jump noticeably.
- **Layer 1 → Layer 2**: Adding continuous batching drastically raises GPU utilisation. Multiple short requests no longer wait behind one long one.
- **Layer 2 → Layer 3**: Chunked prefill prevents prefill-starvation of the decode phase.

## How to reproduce

```bash
# Start the server (in one terminal)
cd CODE_LAYERS/layer0_naive_inference
python server.py

# Run benchmark (in another terminal)
cd CODE_LAYERS/layer0_naive_inference
python benchmark.py --num-requests 20
```
