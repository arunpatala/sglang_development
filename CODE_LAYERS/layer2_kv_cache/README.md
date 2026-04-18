# Layer 2 — KV Cache

## What changes from Layer 1

Only **`model.py`**. `server.py` and `benchmark.py` are identical in structure.

| | Layer 1 | Layer 2 |
|---|---|---|
| Decode input | full growing sequence every step | single new token every step |
| `use_cache` | `False` | `True` |
| KV cache | none — recomputed from scratch | `past_key_values` reused |
| TPOT cost | O(seq_len) per step → grows with prompt | O(1) per step → near-constant |

## The two-line diff in model.py

```python
# Layer 1 — full sequence every step
out = model(input_ids=ids, use_cache=False)

# Layer 2 — one token + cached K/V
out = model(input_ids=current_token, past_key_values=past_kv, use_cache=True)
past_kv = out.past_key_values
```

## File layout

```
layer2/
  model.py       ← KVCacheModel: prefill once, decode with past_key_values
  server.py      ← identical to layer1/server.py except import + port 8102
  benchmark.py   ← identical to layer1/benchmark.py except port 8102 + label
  README.md
```

## What to observe in the benchmark

Compare TPOT per request vs Layer 1:

| Request | Prompt tokens | Layer 1 TPOT | Layer 2 TPOT |
|---------|--------------|--------------|--------------|
| short   | ~10          | ~12ms        | ~12ms        |
| medium  | ~400         | ~25ms        | ~12ms        |
| long    | ~1600        | ~85ms        | ~12ms        |

TPOT should be near-constant in Layer 2 regardless of prompt length.
TTFT still scales with prompt length (prefill is unchanged).

## Run

```bash
# Terminal 1 — start the server
cd CODE_LAYERS/layer2
python server.py

# Terminal 2 — run the benchmark
cd CODE_LAYERS/layer2
python benchmark.py --num-requests 20
```
