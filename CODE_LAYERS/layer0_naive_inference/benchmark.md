╔══════════════════════════════════════════════════════════════╗
║  BENCHMARK RESULTS
║  Layer:              0 (naive, no KV cache)
║  Requests:           20
║  Dataset:            synthetic-sharegpt (seed=42)
║  Mode:               sequential
╠══════════════════════════════════════════════════════════════╣
║  Total input tokens:    4823
║  Total output tokens:   3241
║  Total wall time:       28.4s
║  Output throughput:     114.1 tok/s
║  Total throughput:      283.2 tok/s
║  Request rate:          0.70 req/s
║  Avg request latency:   1418ms
╠══════════════════════════════════════════════════════════════╣
║  COMPARISON TABLE (add layers as you go)
║  Layer 0 (naive):        114 tok/s  ← baseline
╚══════════════════════════════════════════════════════════════╝