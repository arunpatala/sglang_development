# 07 — What Comes Next

Layer 1 opened the black box. The generate loop is now in our code, every step is timed, and the two phases of generation — prefill and decode — are named and measured separately. The computation is identical to Layer 0: the same number of forward passes, the same cost per step, the same throughput numbers. What changed is visibility and control.

That control is exactly what Layer 2 needs. The improvement it makes is a single conceptual change to the decode loop: instead of passing the full growing sequence on every step, it saves the key and value tensors from the previous forward pass and passes only the one new token on each subsequent step. The model reads the saved tensors rather than recomputing them. Here is what that looks like in `model.py`:

```python
# Layer 1 — full sequence every step
for step in range(max_new_tokens):
    out = self.model(input_ids=ids, use_cache=False)
    ...

# Layer 2 — one token per decode step, saved tensors reused
past_kv = None
for step in range(max_new_tokens):
    if past_kv is None:
        out = self.model(input_ids=ids, use_cache=True)        # prefill: full prompt
    else:
        out = self.model(input_ids=new_token, past_key_values=past_kv, use_cache=True)  # decode: one token
    past_kv = out.past_key_values
    ...
```

`server.py` does not change. `benchmark.py` does not change. The API contract — `POST /generate`, same request schema, same response schema — stays identical. The only file that changes is `model.py`, and within it only the decode loop. This is the payoff of owning the loop.

The effect on TPOT will be significant. In Layer 1, each decode step processes a sequence that grows by one token every iteration. In Layer 2, each decode step processes exactly one token regardless of how long the sequence has become. TPOT drops and stays flat rather than creeping upward. TTFT stays roughly the same — the prefill pass still has to process the full prompt, and that cost has not changed.

The benchmark numbers from the same 20 ShareGPT conversations, same seed, same hardware, will show the difference directly.
