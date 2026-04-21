# 01 — The Memory Bottleneck: Why KV Cache Quantization Exists

## What This Section Covers

Before learning how KV cache quantization works, it is worth understanding **why it is necessary** — what constraint it is responding to. This section establishes the KV cache size formula, works through concrete numbers for modern models and serving workloads, and explains why the bottleneck is different from the weight-memory problem that weight quantization addresses.

---

## The KV Cache Size Formula

Every transformer attention layer maintains a Key tensor and a Value tensor for every token currently in the batch. The total GPU memory consumed is:

```
KV_bytes = num_layers × 2 × num_kv_heads × head_dim × num_tokens × bytes_per_element
```

Where:
- `num_layers` — number of transformer layers (depth of the model)
- `2` — one K tensor and one V tensor per layer
- `num_kv_heads` — number of key/value attention heads (for GQA models, fewer than query heads)
- `head_dim` — dimension per head (typically 128)
- `num_tokens` — sum of all token positions across the active batch: `sum(context_len + generated_len)` for every request currently being processed
- `bytes_per_element` — 2 for BF16, 1 for FP8, 0.5 for FP4

For Llama-3.1-70B (GQA-8, 80 layers, 8 KV heads, head_dim=128):

```
KV_per_token (BF16) = 80 × 2 × 8 × 128 × 2 bytes = 327,680 bytes ≈ 0.32 MB per token
KV_per_token (FP8)  = 0.16 MB per token
KV_per_token (FP4)  = 0.08 MB per token
```

This compounds quickly with batch size and context length:

| Config | Total tokens in flight | KV @ BF16 | KV @ FP8 | KV @ FP4 |
|---|---|---|---|---|
| Batch=32, ctx=2048 | 65,536 | 20.5 GB | 10.2 GB | 5.1 GB |
| Batch=32, ctx=8192 | 262,144 | 81.9 GB | 40.9 GB | 20.5 GB |
| Batch=8, ctx=32768 | 262,144 | 81.9 GB | 40.9 GB | 20.5 GB |
| Batch=1, ctx=131072 | 131,072 | 41.0 GB | 20.5 GB | 10.2 GB |

A single 8×H100 node (640 GB total VRAM) minus 140 GB for 70B FP8 weights leaves ~500 GB for KV. At BF16, a batch=32/ctx=8192 workload consumes 82 GB — about 16% of KV budget. At FP8, the same workload uses 41 GB, freeing room for either a larger batch or longer contexts.

---

## The Two Pressures That Make KV Cache the Binding Constraint

### Pressure 1: Throughput demands large batches

LLM serving economics reward **high token throughput**: packing more requests into a single forward pass amortizes the fixed cost of loading model weights. Throughput is maximized by maximizing batch size. Larger batch → more total tokens in flight → larger KV cache.

At some point the KV cache fills the GPU VRAM. Adding one more request means evicting someone else's KV tensors. If those tensors are not recoverable (Layer 12's `RadixCache` discards them), the evicted request must re-prefill from scratch on the next iteration — wasting compute, not just memory.

### Pressure 2: Long contexts need more tokens per request

Models like Llama-3.1 (128K context) and Qwen2.5 (1M context) serve document-QA, code analysis, and multi-turn chat workloads where a single request might carry 32K–128K tokens. A single 128K-token request consumes:

```
Llama-3.1-70B, 128K context, BF16:
80 × 2 × 8 × 128 × 131,072 × 2 bytes = 42.9 GB
```

One request alone takes 42.9 GB — over half a single H100's VRAM for just its KV cache. At FP8 this falls to 21.5 GB. The long-context capability of modern models is practically gated by this single number.

---

## Why This Is a Different Problem From HiCache (Layer 17)

HiCache (Layer 17) and KV cache quantization address the same symptom (VRAM exhaustion) from opposite directions:

| Approach | Mechanism | What It Preserves | Tradeoff |
|---|---|---|---|
| **HiCache** (Layer 17) | Moves BF16 KV tensors to CPU/disk when GPU is full | Full BF16 precision; no accuracy impact | Reload latency (ms–100ms) when cold tokens are needed again |
| **KV Quantization** (Layer 18) | Shrinks each tensor from 2 bytes → 1 byte before it is ever stored | Fits 2× more tokens in GPU VRAM permanently | Tiny accuracy degradation from reduced precision |

They are **orthogonal and composable**: you can run FP8 KV with HiCache simultaneously. FP8 KV makes each tensor smaller; HiCache moves smaller tensors across tiers when the (now smaller) pool still overflows. The PCIe bandwidth cost of a HiCache tier-2 load is also halved.

---

## The Decode Step: Memory Bandwidth, Not Compute

A subtler reason KV quantization matters beyond raw capacity: **decode is memory-bandwidth-bound**.

During the prefill phase, the GPU computes the full attention matrix for all input tokens simultaneously — this is FLOP-intensive and uses the tensor cores efficiently.

During the decode phase, the model generates one token at a time. Each decode step must:
1. Load Q, K, V weight matrices (constant cost — same each step)
2. Compute Q projection for the single new token
3. Load **all KV cache entries** for all tokens in the context
4. Compute attention between the new Q and all cached K, V

Step 3 is the bottleneck. For a 128K-token context, the GPU loads ~43 GB of KV data per decode step (BF16 Llama-70B). At H100 HBM bandwidth of 3.35 TB/s, that is 43/3350 = 12.8 ms per token of KV load time alone — independent of the compute cost.

At FP8: 21.5 GB / 3.35 TB/s = 6.4 ms per token. **Halved decode latency from memory bandwidth alone**, with no change in model accuracy for well-calibrated scales.

This is why FP8 KV quantization is not just a memory-saving trick — it is a throughput and latency optimization that operates at every single decode step.

---

## What Causes the Accuracy Loss?

KV quantization stores values at lower precision. Three sources of error:

**1. Rounding error:** BF16 has 7 mantissa bits; FP8 e4m3 has 3. Each stored value is rounded to the nearest FP8 value. For values well within the FP8 range with a good scale, this rounding is ≈ 1–2% relative error per value — small enough that the softmax and output projection absorb it.

**2. Saturation:** Values larger than the FP8 maximum (448 × scale) are clipped to the maximum. If the scale is too small (or was not calibrated on representative data), outlier activations saturate → attention scores become incorrect → model output degrades.

**3. Scale mismatch:** A single per-tensor scale must accommodate the largest value in the layer across all tokens. This inflates the scale, compressing small values into a narrow band of FP8 representable values. Per-channel or per-token scales (Section 02) mitigate this.

In practice: with calibrated per-layer scales from a representative dataset, FP8 e4m3 KV quantization produces < 0.5% degradation on standard LLM benchmarks (MMLU, HumanEval, LongBench) for Llama, Qwen, and Mistral model families.

---

## Summary

- KV cache grows with `num_layers × 2 × num_kv_heads × head_dim × num_tokens × element_size`
- At BF16, modern 70B models need 0.32 MB per token — filling VRAM fast at production batch sizes and long contexts
- FP8 halves this to 0.16 MB per token; FP4 halves again to 0.08 MB
- The decode step loads the entire KV cache every token: halving KV size directly halves decode memory-bandwidth cost and decode latency
- HiCache moves tensors across tiers; quantization shrinks each tensor — both composable, neither redundant
- Accuracy cost is small (< 0.5% on standard benchmarks) with calibrated scales; the main risk is scale saturation from mis-calibration

**Next section:** how float8 is represented and why the choice of scale granularity determines how much accuracy you lose.
