# 07 — What Comes Next

## What Layer 4B Did Not Change

Layer 4B replaced the model implementation but left the scheduling primitive untouched. `generate_batch` still assembles a fixed set of B requests, runs prefill for all of them together, then runs a decode loop until the last one finishes. A request that generates 5 tokens and a request that generates 1000 tokens are locked together in the same batch. The 5-token request's result is ready after a handful of decode steps, but it is not returned to the caller until `finished.all()` is true — which means waiting for the 1000-token request to complete. This is head-of-line blocking, identified in Layer 3 and unchanged through Layers 4A and 4B.

The `model/` package and `kv_cache.py` from Layer 4B are exactly what Layer 5 builds on. The architecture, weight loading, RoPE, attention, and mask construction are all carried forward unchanged.

---

## Layer 5 — Continuous Batching

Layer 5 (`layer5_continuous_batching`) addresses head-of-line blocking by allowing requests to enter and leave the decode batch independently, at every step.

Three new files define the scheduling infrastructure. `request.py` introduces a `Req` dataclass that represents a single in-flight request: its tokenised prompt, its own `PerReqKVCache`, a status field (`WAITING`, `RUNNING`, `FINISHED`), and an `asyncio.Future` that the HTTP layer awaits for the result. `batch.py` introduces a `Batch` and a `ForwardMode` enum (`PREFILL` or `DECODE`) that describes the shape of the next forward pass to the model runner. `scheduler.py` runs a background daemon thread that pulls from a waiting queue, prefills new requests one at a time, and runs one decode step across all currently active requests on every iteration.

The model runner changes from `generate_batch` to two separate methods:

```python
# Layer 5 — model_runner.py
def prefill(self, req: Req) -> None:
    # B=1 prefill for one request; populates req.kv_cache
    ...

def decode_step(self, running_reqs: list[Req]) -> None:
    # B=N decode step across all running requests; appends to each req.kv_cache
    ...
```

The scheduler calls `prefill` once when a new request arrives, then calls `decode_step` on every loop iteration with the current set of running requests. When a request emits EOS, the scheduler resolves its future from the background thread using `loop.call_soon_threadsafe(req.future.set_result, result)` and removes it from the running list. The next iteration of the decode loop has one fewer request — no waiting.

---

## The KV Cache Problem

Continuous batching introduces a complication that Layer 4B's `KVCache` does not handle. In Layer 4B, all B requests share a single `[B, n_kv_heads, seq_len, head_dim]` tensor with a common sequence length. In Layer 5, requests arrive at different times and have different KV lengths. A request that has been decoding for 100 steps has a KV cache of length 100; a request that just started prefill has a KV cache of length equal to its prompt.

Layer 5 introduces `PerReqKVCache` — one per request, growing independently — and `BatchedKVCache`, which pads and stacks the per-request caches into a rectangular tensor for each decode step. After the decode forward pass, `write_back()` appends only the new single-token K/V to each `PerReqKVCache`, keeping per-request storage compact. The `position_ids` for each request during decode are `kv_len_i` — the individual cache length — not the shared `max_kv_len`, preserving the per-request RoPE correctness that Layers 3, 4A, and 4B all relied on.

---

## What Stays the Same

The `model/` package — `Qwen3Config`, `Qwen3ForCausalLM`, `Qwen3Model`, `Qwen3Attention`, `RMSNorm`, `RotaryEmbedding`, `Qwen3MLP`, `Qwen3DecoderLayer`, `_build_additive_mask` — is imported by Layer 5's `model_runner.py` without modification. The tokenizer is likewise unchanged. The forward call signature `self.model(ids, attention_mask=mask, kv_cache=kv, position_ids=pos)` is the same. Owning the model code in Layer 4B means Layer 5 can change the scheduler without touching the architecture at all. That separation is the architectural payoff of Layers 4A and 4B.
