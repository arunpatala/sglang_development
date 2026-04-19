# SGLang Reference — Layer 1 Concepts in the Real Codebase

This file maps each concept introduced in the Layer 1 lesson to where and how it is implemented in the SGLang source tree. All paths are relative to `REPOS/sglang/python/sglang/srt/`.

The central difference from our implementation: Layer 1 has one Python function that calls `model()` and samples one token in a sequential loop, one request at a time. SGLang runs the same conceptual steps — forward pass, logit processing, sampling — but across a dynamically assembled batch of many requests simultaneously, managed by a scheduler running in its own subprocess.

---

## 1. The Forward Pass (`model(input_ids)`)

**Layer 1:** `self.model(input_ids=ids, use_cache=False)` — a direct HuggingFace model call, returns `out.logits`.

**SGLang:** `model_executor/model_runner.py` — `ModelRunner.forward_decode` (line 2722) and `ModelRunner.forward_extend` (line 2745)

SGLang distinguishes two separate forward methods at the runner level:

- `forward_extend` is the prefill path. It processes new tokens being added to the context — the full prompt for a new request, or a continuation in chunked prefill mode.
- `forward_decode` is the decode path. It processes one new token per active request across the entire batch.

Both ultimately call `self.model.forward(forward_batch.input_ids, forward_batch.positions, forward_batch)`, but they set up different attention metadata beforehand via `attn_backend.init_forward_metadata`. In Layer 1, one Python function handles both phases identically; in SGLang they are distinct code paths with different memory access patterns.

The input is not a simple `input_ids` tensor. It is a `ForwardBatch` object (`model_executor/forward_batch_info.py`, line 280) that carries the token IDs, position indices, sequence lengths, KV cache pointers, and a `ForwardMode` enum (`EXTEND` or `DECODE`) indicating which path to take.

---

## 2. Prefill and Decode as Explicit Modes

**Layer 1:** Both phases use the same `self.model(input_ids=ids, use_cache=False)` call. The distinction is conceptual, tracked only by step index.

**SGLang:** `model_executor/forward_batch_info.py` — `ForwardMode` enum (line 81)

```python
class ForwardMode(IntEnum):
    EXTEND = auto()   # prefill: processing new tokens
    DECODE = auto()   # decode: generating one token per request
    ...
```

`ForwardMode.EXTEND` (prefill) and `ForwardMode.DECODE` are first-class values in SGLang. The scheduler (`managers/scheduler.py`) determines the mode for each batch and sets it on the `ForwardBatch`. Attention backends use this to switch between different kernel implementations — the attention pattern for processing a full prompt differs from the attention pattern for generating a single token with cached keys and values.

In a continuous batching system, a single forward pass may contain some requests in `EXTEND` mode (new arrivals being prefilled) and others in `DECODE` mode (ongoing generation). The scheduler assembles these together, and the chunked prefill feature (`EXTEND` + `DECODE` in the same batch) allows further mixing.

---

## 3. Logits and the Last Position

**Layer 1:**
```python
out.logits          # shape: [1, seq_len, vocab_size]
out.logits[0, -1, :]  # last position only
```

**SGLang:** `model_executor/forward_batch_info.py` — `LogitsProcessorOutput`; `layers/logits_processor.py`

SGLang does not return the full `[batch, seq_len, vocab_size]` logit tensor from the forward pass. Instead, a `LogitsProcessor` attached to the model's output head extracts only the logits at the last position for each sequence in the batch — the equivalent of our `[0, -1, :]` slice — before returning. This avoids materialising the full tensor for every position in every sequence, which would be enormous for long sequences in a large batch.

The result is a `LogitsProcessorOutput` with a `next_token_logits` field of shape `[batch_size, vocab_size]` — one logit vector per request in the batch, already sliced to the relevant position.

---

## 4. Sampling — Greedy, Temperature, Top-k, Top-p

**Layer 1:** `_sample_next_token` in `model.py` — three cases: argmax, temperature + multinomial, temperature=1 + multinomial. One request at a time.

**SGLang:** `layers/sampler.py` — `Sampler` class (line 41), `sampling/sampling_params.py` — `SamplingParams` (line 31)

`SamplingParams` is the canonical container for all sampling configuration:

```python
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    max_new_tokens: int = 128
    stop: ...
    stop_token_ids: ...
```

`temperature=0` is normalised to `top_k=1` (greedy) during parameter validation, so the sampler never sees temperature=0 as a special case — it just sees top_k=1.

The `Sampler.forward` method operates on the entire batch at once. Its key branches mirror our three cases:

```python
if sampling_info.is_all_greedy:
    batch_next_token_ids = torch.argmax(logits, -1)          # greedy for all
else:
    logits.div_(sampling_info.temperatures)                  # temperature scaling
    logits[:] = torch.softmax(logits, dim=-1)                # → probabilities
    # then top-k / top-p filtering via FlashInfer or custom CUDA kernels
    batch_next_token_ids = top_k_top_p_sampling_from_probs(...)
```

The critical difference: `sampling_info.temperatures` is a tensor of shape `[batch_size]`, one temperature per request. Different requests in the same batch can have different temperatures, and the division is a single batched operation across all of them. Our implementation handles one request and one temperature at a time.

Top-k and top-p filtering (`top_k_renorm_prob`, `top_p_renorm_prob`) are applied as vectorised operations using FlashInfer kernels rather than Python loops.

---

## 5. TTFT and TPOT

**Layer 1:** Computed manually from `step_times`: `ttft_ms = step_times[0] * 1000`, `tpot_ms = mean(step_times[1:]) * 1000`. Returned in the response payload.

**SGLang:** `observability/req_time_stats.py` — `ReqTimeStats` class; `observability/metrics_collector.py`

SGLang tracks timing at the individual request level through `ReqTimeStats`, which records timestamps at key lifecycle events:

```python
def get_first_token_latency(self):
    return self.first_token_time - self.created_time      # TTFT

def get_decode_latency(self):
    return self.finished_time - self.first_token_time     # total decode time

# decode_throughput = completion_tokens / decode_latency  # ≈ 1 / TPOT
```

These are collected by the `TokenizerManager` when it receives the first token from the detokenizer (`ttft_observed` flag, line 2108 of `tokenizer_manager.py`) and exposed as Prometheus histograms via `MetricsCollector`:

```
sglang:time_to_first_token_seconds  (histogram)
```

The important architectural difference is that in a continuous batching system, "first token" has a subtler meaning. A request does not start prefill the moment it arrives — it waits in the scheduler queue until a batch slot is available. `created_time` is when the request arrived at the API server; `first_token_time` is when the detokenizer sent the first decoded token back. The TTFT therefore includes queue wait time, which our Layer 1 implementation does not have (requests are served immediately, sequentially).

There is no direct TPOT metric exposed as a per-step average in SGLang's standard metrics. Instead, decode throughput (`completion_tokens / decode_latency`) gives the equivalent aggregate measure. Individual step times are not tracked, because in continuous batching a "step" produces tokens for many requests simultaneously and attributing step time to a single request is not straightforward.

---

## 6. The EOS Check and Stop Conditions

**Layer 1:** `if next_token_id == self.eos_id: break` — a single check against the tokenizer's EOS token.

**SGLang:** `managers/scheduler.py` — `check_finished`, stop conditions in `sampling/sampling_params.py`

SGLang checks several stop conditions simultaneously for each request after sampling:

- EOS token ID (from tokenizer config)
- Custom `stop_token_ids` supplied in the request
- Stop strings: the detokenizer checks whether the decoded text ends with any string in the `stop` list
- `stop_regex`: regular expression match against decoded output
- `max_new_tokens` budget exhausted
- Context window limit reached

Our `self.eos_id` check is the simplest version of the first condition. In production, a request that generates a stop string mid-response needs the detokenizer to notice it in the decoded text, trigger a stop signal back to the scheduler, and have the scheduler mark the request as finished — a multi-process coordination that our single-process loop trivially handles with a `break`.

---

## Summary of Differences

| Concept | Layer 1 | SGLang |
|---|---|---|
| Forward pass | Single `model(input_ids)` call, one request | `ModelRunner.forward_decode` / `forward_extend`, whole batch |
| Prefill vs decode | Same code path, distinguished by step index | Explicit `ForwardMode.EXTEND` vs `ForwardMode.DECODE`, different attention kernels |
| Logit extraction | `out.logits[0, -1, :]` — slice last position manually | `LogitsProcessor` extracts last-position logits before returning |
| Sampling | Single request, Python `argmax` / `multinomial` | `Sampler` operates on `[batch_size, vocab_size]` tensor, batched temperature, FlashInfer kernels for top-k/top-p |
| Sampling params | `temperature` only | `SamplingParams`: temperature, top_k, top_p, repetition_penalty, stop strings, stop regexes |
| TTFT | `step_times[0]` — first decode step | `first_token_time - created_time` including queue wait, exposed as Prometheus histogram |
| TPOT | `mean(step_times[1:])` — average of pure decode steps | `completion_tokens / decode_latency` aggregate; no per-step average |
| EOS check | `next_token_id == self.eos_id` | Multi-condition: EOS, stop token IDs, stop strings, stop regex, max tokens, context limit |
