# 06 — Padding Waste and GPU Utilisation

## The Utilisation Problem in Layer 2

Layer 2 processed one request per forward call. The decode step was a `[1, 1]` tensor — a single token fed through the full transformer. For Qwen3-0.6B, producing one output token requires loading approximately 1.2 GB of weights from GPU memory, running them over a single position, and writing the output. The arithmetic for one position is trivial: a few million multiplications. The GPU's compute units sit idle for the vast majority of each step, waiting on the memory subsystem to deliver the weights. This is what the model docstring means by "GPU ~5% utilised" — the arithmetic hardware is active for roughly 5% of the step's wall time.

Static batching changes this by making each decode step operate on `[B, 1]`. The same 1.2 GB of weights are loaded once and applied to B positions simultaneously. For B=16, the arithmetic is 16× larger. The memory load is the same. The ratio of arithmetic to memory access — called arithmetic intensity — is 16× higher. GPU utilisation climbs toward 80%.

---

## What Batching Actually Buys

The cost model is simple. Each decode step has two components: loading model weights from GPU memory (fixed cost, independent of B) and computing the forward pass over `[B, 1]` (scales with B). For small B, the fixed memory load dominates and the arithmetic is cheap. As B grows, the arithmetic component grows while the memory load stays constant, and the GPU becomes less idle.

This is why throughput climbs with batch size up to a point and then levels off. At B=1, the GPU finishes the arithmetic almost instantly after the memory load completes. At B=16, the arithmetic takes 16× as long, which means the memory load is no longer the sole bottleneck — the GPU is doing real work for a larger fraction of each step. At very large B, the arithmetic itself becomes the bottleneck and throughput plateaus.

The benchmark numbers from `benchmark.md` show this shape:

```
batch=1   →  68 tok/s    (memory bound, GPU idle most of each step)
batch=4   →  140 tok/s   (large improvement, moving toward compute)
batch=8   →  139 tok/s   (near-flat, memory and compute roughly balanced)
batch=16  →  149 tok/s   (peak throughput on this hardware)
batch=20  →  OOM         (KV caches for 20 requests exhaust GPU memory)
```

The improvement from B=1 to B=4 is 2×. From B=4 to B=16 it is only 6%. The curve is concave — gains diminish as batch size grows. For a 0.6B model with a relatively small feed-forward dimension, the compute-bound regime arrives at a lower batch size than it would for a larger model.

---

## The Padding Waste Problem

Throughput gains come with a cost: prefill padding waste. When prompts in a batch have different lengths, they are all padded to the length of the longest one. The prefill forward pass processes `[B, max_prompt_len]` — B rows, each of length `max_prompt_len` — regardless of how short the shortest prompt actually is.

Consider a batch of two requests: one 10-token prompt and one 1000-token prompt. After left-padding, `input_ids` is `[2, 1000]`. The prefill processes 2000 token positions. Of those 2000 positions, 990 belong to the padding of the short prompt — they are masked by `attention_mask = 0` and contribute nothing to the output. The model still runs its matrix multiplications over them. The wasted fraction is 990 / 2000 = 49.5%. For a batch of 16 requests with similarly variable lengths, the wasted fraction can easily exceed 70%.

This is the fundamental structural flaw of static batching. The batch is assembled before prefill begins and its padded shape is fixed. There is no mechanism to process the short prompts without computing over their padding rows.

TTFT reflects this directly. With a single short prompt, prefill is fast. With a batch that contains a long prompt, prefill is slow for everyone — including the short prompts that are waiting for their results:

```
batch=1   →  TTFT = 29 ms
batch=4   →  TTFT = 115 ms
batch=16  →  TTFT = 1306 ms
```

The TTFT at B=16 is 45× higher than at B=1. The improvement in throughput (2.2×) comes at the cost of every individual request waiting much longer for its first token.

---

## Head-of-Line Blocking

The second cost of static batching is head-of-line blocking. The decode loop runs until `finished.all()` — until the last active request emits EOS. A request that finishes in 5 tokens is stuck waiting in the batch until the request that needs 128 tokens completes. Its result is not returned until then.

In a server that serves many users, this means short requests experience latency that scales with the longest concurrent request rather than with their own output length. A user asking a simple yes/no question might wait five seconds for a response not because their answer is long but because someone else in the same batch requested a lengthy essay.

Static batching is the right starting point for a lesson because it demonstrates the GPU utilisation improvement cleanly and makes the structural problems of fixed-shape batching visible. The TTFT growth and head-of-line blocking numbers in the benchmark are not bugs — they are the honest measurement of what static batching costs. Continuous batching, which addresses both problems by allowing requests to enter and leave the batch dynamically, is what comes next.
