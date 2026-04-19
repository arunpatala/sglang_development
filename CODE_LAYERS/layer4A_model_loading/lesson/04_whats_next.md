# 03 — What Comes Next

## What Layer 4A Did Not Change

Layer 4A introduced our own config reading and weight streaming pipeline, but the computation inside `forward` is still entirely HuggingFace's. `Qwen3ForCausalLM.forward` in this layer is three lines:

```python
def forward(self, input_ids, attention_mask, past_key_values, position_ids):
    out = self._model(input_ids=input_ids, attention_mask=attention_mask,
                      past_key_values=past_key_values, position_ids=position_ids,
                      use_cache=True)
    return out.logits, out.past_key_values
```

Everything that happens between the input token IDs and the output logits — the embedding lookup, all 28 decoder layers with their RMSNorm, RoPE, grouped-query attention, SwiGLU MLP, and the final LM head projection — runs inside `self._model`, which is still HuggingFace's `Qwen3ForCausalLM`. The `KVCache` from Layer 3 handles caching through HF's internal `DynamicCache` interface.

---

## Layer 4B — Model Layers

Layer 4B (`layer4B_model_layers`) replaces `self._model` with our own hand-written implementation. Five files in `model/` implement the Qwen3 architecture from scratch:

| File | What it implements |
|---|---|
| `norm.py` | `RMSNorm` — root-mean-square layer normalisation |
| `mlp.py` | `Qwen3MLP` — SwiGLU feed-forward with gate, up, and down projections |
| `rope.py` | `RotaryEmbedding` — precomputed sinusoidal frequencies, applied via complex rotation |
| `attention.py` | `Qwen3Attention` — grouped-query attention with `repeat_kv`, SDPA, and causal masking |
| `decoder_layer.py` | `Qwen3DecoderLayer` — one transformer block: pre-norm, attention, residual, pre-norm, MLP, residual |

These are composed into `Qwen3Model` (the stacked decoder) and the outer `Qwen3ForCausalLM` (embedding + decoder + LM head), both in `model/qwen3.py`.

The `KVCache` also changes in Layer 4B: instead of HF's `DynamicCache` interface, attention layers call `kv.update(layer_idx, key, value)` directly on our own cache object. `model_runner.py` passes it as `kv_cache=kv` rather than `past_key_values=kv`, and the return value becomes a plain logits tensor rather than a tuple — the cache is truly in-place with no returned object.

---

## Layer 5 — Continuous Batching (Now Layer 6)

Layer 4A and 4B both leave the scheduling primitive untouched. `generate_batch` still assembles a fixed set of B requests, runs prefill for all of them together, then runs a decode loop until the last one finishes. A request that generates 5 tokens and a request that generates 1000 tokens are locked together — the 5-token result is not returned until `finished.all()` is true. This is head-of-line blocking, present from Layer 1 through Layer 4B.

Layer 5 (continuous batching) addresses this by allowing requests to enter and leave the decode batch independently at every step. Three new files define the scheduling infrastructure:

- `request.py` — `Req` dataclass: tokenised prompt, its own per-request KV cache, a status field (`WAITING`, `RUNNING`, `FINISHED`), and an `asyncio.Future` the HTTP layer awaits for the result.
- `batch.py` — `Batch` and `ForwardMode` enum (`PREFILL` or `DECODE`) that describes the next forward pass.
- `scheduler.py` — background daemon thread that pulls from a waiting queue, prefills new requests one at a time, and runs one decode step across all currently active requests on every iteration.

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

When a request emits EOS, the scheduler resolves its future and removes it from the running list — the next decode step has one fewer request, with no waiting.

The `model/` package from Layer 4B is imported by Layer 5's `model_runner.py` without modification. Owning the model code means the scheduler can be redesigned independently of the architecture — that separation is the architectural payoff of Layer 4.
