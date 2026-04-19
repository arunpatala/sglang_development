# 07 — TTFT and TPOT

## Beyond Total Latency

Layer 0 reported a single number: `latency_ms` — the total wall time for the entire request, from receiving the prompt to returning the response. That is a useful number, but it mixes together two fundamentally different costs. Layer 1 splits them apart.

The `GenerateResponse` in Layer 1 adds two new fields:

```python
class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    ttft_ms: float    # Time To First Token
    tpot_ms: float    # Time Per Output Token
```

These two metrics come from `step_times`, the list that records how long each iteration of the decode loop took.

## Time To First Token (TTFT)

```python
ttft_ms = round(step_times[0] * 1000, 1) if step_times else 0.0
```

TTFT is the time from when the request arrives to when the first generated token is ready. In the code that is `step_times[0]` — the duration of the very first iteration of the loop.

That first iteration is the prefill step. The model receives the full prompt, processes all tokens in a single forward pass, and produces the logits from which the first output token is sampled. TTFT therefore captures two things: how expensive the prompt was to process, and how long a user waits before anything starts appearing in a streaming response.

TTFT is sensitive to prompt length. A 20-token prompt and a 500-token prompt both complete in one forward pass, but the 500-token pass is much more expensive because attention scales quadratically with sequence length. If you are building a system where users send long contexts — documents, code files, long chat histories — TTFT is often the bottleneck you need to optimise first.

## Time Per Output Token (TPOT)

```python
decode_steps = step_times[1:]
tpot_ms = round((sum(decode_steps) / len(decode_steps)) * 1000, 1) \
           if decode_steps else ttft_ms
```

TPOT is the average duration of a single decode step, excluding the first step. It captures the steady-state cost of the generation loop once prefill is done.

Step 0 is excluded from the TPOT average because it includes the prefill cost — processing the full prompt — which makes it unrepresentative of what a normal decode step costs. Steps 1 onwards each process the prompt plus however many tokens have been generated so far. Their cost is more uniform and represents the "production speed" of the decoder.

TPOT is what determines how fast text appears to a user who is watching a streaming response. A TPOT of 20 ms means the server produces 50 tokens per second (`1000 / 20`). At roughly 4–5 tokens per word, that is about 10–12 words per second — fast enough to appear natural for most use cases.

## Why They Move Independently

In Layer 1, TTFT and TPOT are both expensive and both grow with sequence length — there is no caching. But in later layers, they will respond to optimisations differently, which is exactly why tracking them separately is valuable.

Adding a KV cache (Layer 2) dramatically reduces TPOT. Each decode step shrinks from processing the full growing sequence to processing just one new token. TTFT stays roughly the same — the prefill pass still has to process the full prompt.

Batching (a later layer) increases raw throughput but tends to increase TPOT for individual requests, since the GPU is now shared across many sequences. TTFT may also increase if the server has to wait to assemble a batch before processing.

A system can have excellent throughput but poor TPOT (bad for interactive users), or excellent TPOT but poor throughput (good for users, bad for the operator paying for the GPU). TTFT and TPOT together give you the information to understand which situation you are in.

## What the Numbers Look Like in Layer 1

Because Layer 1 has no KV cache, TPOT is not constant — it grows slightly with each decode step as the sequence gets longer and the forward pass has more tokens to process. In practice for short outputs (32–128 tokens) with a 0.6B model the growth is small enough to be visible only as a slight upward trend in per-step timings, not a dramatic increase. The average TPOT is still a reasonable characterisation of decode speed.

The benchmark in `benchmark.py` reports overall output throughput in tokens per second, which is `completion_tokens / (latency_ms / 1000)`. This is closely related to TPOT but averages over the entire request including the TTFT cost. For comparing layers, output throughput is the headline number. TTFT and TPOT are diagnostic tools for understanding where time is going within a single request.
