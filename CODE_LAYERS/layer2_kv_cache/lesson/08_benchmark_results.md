# 07 — What the Numbers Show

## Running the Benchmark

The benchmark script is unchanged from Layer 1. It sends the same 20 ShareGPT conversations to the server, in the same order, with the same random seed, capping output at 128 tokens. The only difference is the port (8102 instead of 8101) and the layer label.

```bash
# Terminal 1 — start the server
python server.py

# Terminal 2 — run the benchmark
python benchmark.py
```

The server prints one log line per request showing prompt token count, latency, TTFT, and TPOT. After all 20 requests, the benchmark prints a summary table and writes `benchmark.md`. The numbers worth comparing against Layer 1 are TTFT and TPOT.

## What TTFT Should Look Like

TTFT — Time To First Token — measures the cost of the prefill forward pass plus the first sample. This is unchanged from Layer 1. The full prompt is still processed in one forward pass, the attention computation is still `O(L²)` in prompt length, and no cache is available to help.

What you should observe is that the TTFT distribution in Layer 2 closely matches the TTFT distribution in Layer 1. Requests with short prompts will have short TTFT; requests with long prompts will have longer TTFT. The ratio should be similar between layers because the prefill code path is identical — only the presence of the `KVCache` object differs, and populating it adds negligible overhead compared to the forward pass itself.

If TTFT is noticeably higher in Layer 2 than Layer 1, the most likely cause is GPU memory pressure: the model's weights and the newly allocated cache both need to fit in memory, and if that causes increased memory system contention, the forward pass takes longer. For a 0.6B model this is unlikely to be visible. For larger models it can matter.

## What TPOT Should Look Like

TPOT — Time Per Output Token — is where Layer 2 shows its improvement, and the effect should be dramatic and clear.

In Layer 1, TPOT is not constant. Each decode step processes a sequence one token longer than the previous step, so each step is slightly more expensive. For requests with long prompts, TPOT is noticeably higher than for requests with short prompts, because even the first decode step must attend over a long context. Across the 20 benchmark requests, the TPOT values vary with prompt length.

In Layer 2, TPOT should be near-constant. Each decode step processes exactly one new token regardless of how long the full context is. The attention computation reads `L+k` cached entries but has only one query, so its cost is `O(L+k)` — linear in context length — compared to `O((L+k)²)` in Layer 1. For Qwen3-0.6B at typical ShareGPT prompt lengths (a few hundred tokens), this is a significant difference.

In practice, TPOT is dominated not just by the attention operation but also by the linear layers in the transformer — the feed-forward network and the projection matrices — which have constant cost per token regardless of context length. So while attention is the piece that changes, the total observed TPOT improvement is a fraction of the attention improvement, depending on the relative weight of attention vs. linear layers in the overall step cost. For a small model like 0.6B, linear layers are a larger share of the total compute than they would be for a larger model.

What you should see is that TPOT is no longer correlated with prompt length. A request with a 600-token prompt should have similar TPOT to a request with a 50-token prompt.

## Reading the Output

The benchmark table prints per-request results including prompt token count, completion tokens, latency, TTFT, and TPOT:

```
[ 1/20] prompt_tokens= 432  max_new_tokens=128  ... done in 2.3s  (128 tokens  ttft=489ms  tpot=12ms)
[ 2/20] prompt_tokens=  87  max_new_tokens= 64  ... done in 0.9s  (64 tokens   ttft=115ms  tpot=11ms)
[ 3/20] prompt_tokens= 891  max_new_tokens=128  ... done in 2.8s  (128 tokens  ttft=921ms  tpot=13ms)
```

Notice that TPOT is consistent across requests (11–13ms here) while TTFT varies with prompt length (115ms for 87 tokens vs. 921ms for 891 tokens). This is the KV cache in action: decode cost is near-constant, prefill cost scales with prompt length.

The summary metrics at the end of the run:

```
Output throughput  : X.X tok/s
Avg TTFT           : Y ms
Avg TPOT           : Z ms/tok
```

Compare `Avg TPOT` against Layer 1's TPOT. The drop should be substantial. Compare `Avg TTFT` — it should be similar. Overall output throughput will improve because each request completes faster (lower TPOT means shorter total latency for the decode portion), even though `req/s` is still limited by the sequential server structure.

## The Memory Cost Made Visible

The server log prints cache memory after each prefill:

```
after prefill: KVCache(layers=28, seq_len=47, memory=6.5 MB)
```

For a 47-token prompt, the cache uses 6.5 MB. Each token in the cache for Qwen3-0.6B costs approximately 112 KB (28 layers × 2 tensors × 8 heads × 128 dimensions × 2 bytes). The cache grows throughout the request: by the end of generating 128 tokens on top of that 47-token prompt, the cache holds 175 token positions and occupies roughly 19 MB.

This number matters for understanding capacity: the GPU has a fixed amount of memory, most of which is occupied by the model weights (approximately 1.2 GB for Qwen3-0.6B in bfloat16). The remaining memory is available for KV caches. If you were serving many requests concurrently, the total cache memory across all active sequences would eventually exhaust the remaining GPU memory, forcing the system to queue or reject new requests. That constraint — and how to manage it efficiently — is what paged attention (a much later layer) addresses.

## Comparing Against Layer 1

The table below is the expected pattern when comparing Layer 1 and Layer 2 on the same benchmark:

| Metric | Layer 1 | Layer 2 | What changed |
|---|---|---|---|
| TTFT (short prompt) | ~120 ms | ~120 ms | Unchanged — same prefill |
| TTFT (long prompt) | ~900 ms | ~900 ms | Unchanged — same prefill |
| TPOT (short prompt) | ~12 ms | ~11 ms | Slight improvement |
| TPOT (long prompt) | ~80 ms | ~12 ms | Large improvement |
| Output throughput | ~X tok/s | ~Y tok/s | Higher, driven by TPOT drop |

The improvement in TPOT is largest for long prompts, because those are the cases where Layer 1 was doing the most redundant attention computation per decode step. For short prompts, the attention is already cheap and the linear layers dominate, so the improvement is smaller.
