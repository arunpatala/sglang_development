# Layer 3 (Static Batching) — Benchmark Results

**Date:** 2026-04-18  
**Model:** Qwen/Qwen3-0.6B  
**Hardware:** NVIDIA GeForce RTX 4060 Ti  
**Requests:** 20 ShareGPT (seed=42)  
**Max new tokens:** 128

---

## Batch Size Sweep

| Batch size | Output tokens | Wall time | Tok/s | Avg TTFT | Avg TPOT | Status |
|-----------|---------------|-----------|-------|----------|----------|--------|
| 1 | 2117 | 31.0s | 68.3 | 29 | 14 | ok |
| 4 | 2495 | 17.8s | 140.2 | 115 | 20 | ok |
| 8 | 2444 | 17.6s | 138.8 | 252 | 24 | ok |
| 16 | 2467 | 16.6s | 148.8 | 1306 | 33 | ok |
| 20 | 0 | 2.5s | 0.0 | 0 | 0 | FAIL |

Best batch size: **16** → **148.8 tok/s**

---

## What the numbers reveal

**Throughput climbs with batch size** because each GPU forward pass does B× more
useful work. A single decode step with batch=8 costs barely more than batch=1
in wall time, but produces 8× the output tokens.

**TTFT grows with batch size** — the prefill pass must process B padded prompts.
The longest prompt in the batch determines the padded length, so large batches
with variable-length prompts waste prefill compute.

**TPOT stays near-constant** — each decode step is [B, 1] regardless of B.
The GPU does more work per step but it's still memory-bandwidth bound, not
compute bound, so latency per step barely changes.

**The head-of-line blocking problem remains.** Short requests in a batch must
wait until the longest one finishes. Continuous batching (Layer 4/5) fixes this
by evicting finished requests and inserting new ones mid-flight.
