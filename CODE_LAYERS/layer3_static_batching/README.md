# Layer 3 — Static Batching

## What changes from Layer 2

| | Layer 2 | Layer 3 |
|---|---|---|
| Requests per GPU pass | 1 | B (batch) |
| Input shape (prefill) | `[1, prompt_len]` | `[B, max_prompt_len]` (left-padded) |
| Input shape (decode) | `[1, 1]` | `[B, 1]` |
| New endpoint | `/generate` only | `/generate` + `/generate_batch` |
| `kv_cache.py` | unchanged | unchanged |

## Key ideas

**Left padding:** prompts of different lengths are padded from the left so every
row's last real token is at position `-1`. The `attention_mask` (0=pad, 1=real)
tells the model to ignore pad positions.

**Finished mask:** the decode loop tracks which requests have emitted `<eos>`.
Finished requests get `pad_token_id` as input and their output is discarded.
The loop ends when all B requests have finished.

**Padding waste:** a batch of [10-token, 1000-token] prompts spends 99% of
prefill compute on padding for the short request. This is the core cost of
static batching.

## File layout

```
layer3_static_batching/
  kv_cache.py   ← copied from layer2, unchanged (batch dim already supported)
  model.py      ← BatchedKVCacheModel with generate_batch()
  server.py     ← /generate + /generate_batch endpoints, port 8103
  benchmark.py  ← sweeps batch sizes [1, 4, 8, 16, 20]
  README.md
```

## Throughput curve

```
batch=1  → GPU ~5% utilised  (same as layer 2)
batch=4  → GPU ~20% utilised
batch=8  → GPU ~40% utilised
batch=16 → GPU ~80% utilised  ← sweet spot on RTX 4060 Ti
batch=20 → may OOM or plateau
```

## Run

```bash
# Terminal 1
cd CODE_LAYERS/layer3_static_batching
python server.py

# Terminal 2
cd CODE_LAYERS/layer3_static_batching
python benchmark.py                       # full sweep
python benchmark.py --batch-sizes 1 8    # just two sizes
```
