## Layer 6 — Packed Batching (Ragged KV + FlashInfer)

| Metric | Value |
|--------|-------|
| Requests (total / concurrency) | 20 / 4 |
| Max output tokens | 128 |
| Total wall time | 7.69s |
| Output throughput | 204.8 tok/s |
| Total throughput | 337.8 tok/s |
| TTFT avg / p95 | 99ms / 369ms |
| Latency avg / p95 | 1316ms / 2282ms |

## Comparison vs Layer 5 (BatchedKVCache + F.sdpa)

| Metric | Layer 5 | Layer 6 | Δ |
|--------|---------|---------|---|
| Wall time | 7.52s | 7.69s | +2% |
| Output throughput | 208.4 tok/s | 204.8 tok/s | -2% |
| Total throughput | 344.4 tok/s | 337.8 tok/s | -2% |
| TTFT avg / p95 | 92ms / 282ms | 99ms / 369ms | +8% avg |
| Latency avg / p95 | 1302ms / 2400ms | 1316ms / 2282ms | ~same |

### Why Layer 6 is not faster here

At Qwen3-0.6B scale with short ShareGPT prompts (avg ~50 tokens, max KV ~180 tokens),
the **padding waste** in Layer 5 is small — the gap between shortest and longest KV
is only ~130 tokens, so very few zero-columns are being computed.

The FlashInfer kernel also has a fixed per-call overhead (kernel launch, indptr
processing) that shows up more when the actual compute is small.

Layer 6's ragged packing **does eliminate padding waste** — but the benefit only
appears at larger batch sizes and longer, more variable KV lengths (e.g. 1000+ token
contexts with high variance). The copy cost (gathering PerReqKVCache tensors into
the packed buffer each step) also offsets savings at this scale.

The real win from this layer is **correctness of the approach**: requests genuinely
attend only to their own tokens (enforced by kv_indptr, not by a mask that could
have bugs at padding boundaries). Layer 7 (paged KV cache) eliminates the copy cost
and is where the throughput improvement becomes measurable.
