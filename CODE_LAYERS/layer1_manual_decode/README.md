# Layer 1 — Manual Decode Loop (No KV Cache)

## What changes from Layer 0

| | Layer 0 | Layer 1 |
|---|---|---|
| Decode | `model.generate(use_cache=False)` — black box | Manual `for` loop calling `model.forward()` — white box |
| Files | `server.py` only | `model.py` + `server.py` |
| Throughput | baseline | ~same (identical computation) |

Layer 1 does not improve performance. Its purpose is to **open the black box** so we can see and control every step of autoregressive decoding. Layer 2 will then modify only `model.py` to add KV cache.

## File layout

```
layer1/
  model.py       ← NaiveModel: loads weights, owns the decode loop
  server.py      ← thin FastAPI wrapper, no inference logic
  benchmark.py   ← same dataset/seed as layer0, port 8101
  README.md
```

## The decode loop (model.py)

```python
ids = input_ids                         # prompt tokens
for step in range(max_new_tokens):
    out = model(input_ids=ids,          # full forward pass from scratch
                use_cache=False)
    next_token = sample(out.logits[0, -1, :], temperature)
    if next_token == eos_id:
        break
    ids = torch.cat([ids, [[next_token]]], dim=1)   # grow sequence by 1
```

Every iteration re-reads the entire sequence. Prompt of length L, output of
length T → L + L+1 + L+2 + … + L+T attention computations = O(L·T + T²/2).

## What Layer 2 will change (only in model.py)

```python
# layer2/model.py — the surgical addition
past_kv = None
for step in range(max_new_tokens):
    out = model(input_ids=current_token,      # only ONE token per step
                past_key_values=past_kv,      # reuse cached K/V
                use_cache=True)
    past_kv = out.past_key_values             # save for next step
    ...
```

`server.py` and `benchmark.py` are unchanged in Layer 2.

## Run

```bash
# Terminal 1 — start the server
cd CODE_LAYERS/layer1
python server.py

# Terminal 2 — run the benchmark
cd CODE_LAYERS/layer1
python benchmark.py --num-requests 20
```

## Expected result

Output throughput should be within a few percent of Layer 0 — same computation,
different code path. If the numbers differ significantly, something is wrong.
The real speedup comes in Layer 2.
