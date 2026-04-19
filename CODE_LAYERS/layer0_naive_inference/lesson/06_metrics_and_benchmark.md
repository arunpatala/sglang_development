# 06 — Metrics and Benchmarking

## What to Measure and Why

An LLM inference server has several performance dimensions, and it is easy to conflate them. Before looking at numbers it is worth being precise about what each metric actually captures.

**Latency** is the time from when a client sends a request to when it receives the complete response. For an LLM server this is dominated by two phases: the prefill (processing the prompt, a single parallel forward pass) and the decode (generating tokens one at a time). Latency is measured per request, and you typically care about the median (p50) and the tail (p99). A server with a p50 of 400 ms and a p99 of 8000 ms is behaving very differently from one with a p50 of 400 ms and a p99 of 450 ms, even though their median latencies are identical.

**Output throughput** (measured in output tokens per second) is the rate at which the server produces generated tokens across all active requests, summed over the measurement window. This is the primary capacity metric for LLM serving: it tells you how much generative work the system can sustain. A server with high output throughput can serve more users, generate longer responses in the same time, or do both. When comparing layers in this curriculum, output throughput is the number to track.

**Total throughput** adds input (prompt) tokens to output tokens in the denominator. It is useful when prompt processing is a significant part of the load, which it is for systems with very long context windows or retrieval-augmented generation. For the short prompts in the ShareGPT benchmark used here, output throughput and total throughput differ noticeably because prompts are substantially longer than completions.

**Request rate** (requests per second) counts how many complete requests the server finishes per unit time. For a sequential server like Layer 0, request rate is simply `1 / average_latency`. For a batched server, request rate and output throughput can move independently — you can serve fewer requests per second while producing more total tokens, if each request generates a longer response.

## The ShareGPT Benchmark

The benchmark script, `benchmark.py`, uses the ShareGPT dataset — a collection of real user conversations shared from the ChatGPT interface — to produce a workload that is representative of actual assistant usage. This matters because synthetic prompts (e.g. always generating from "Hello") do not stress the system in the same way: real prompts vary in length, topic, and the amount of output they naturally elicit.

The benchmark is designed to be reproducible across layers. It downloads the ShareGPT dataset via HuggingFace Hub (or uses a local cached copy), shuffles it with a fixed seed (`seed=42` by default), and samples the first `N` valid conversations. Each conversation is represented as a single user turn (the first human message) paired with a reference completion (the first assistant message). The reference completion length, capped at `max_new_tokens`, determines how many tokens the server is asked to generate for that request.

Running the benchmark against the Layer 0 server is straightforward:

```bash
# Terminal 1 — start the server
python server.py

# Terminal 2 — run the benchmark
python benchmark.py --layer 0 --num-requests 20
```

The `--layer` argument is a label only; it is used to annotate the output and the generated `benchmark.md` file. The benchmark always talks to whichever server is running on the configured port, so you can run the same command against Layer 1, Layer 2, and so on by simply changing the server that is running.

After all requests complete, the benchmark prints a summary and writes `benchmark.md`:

```
╔══════════════════════════════════════════════════════════════╗
║  BENCHMARK RESULTS
║  Layer              : 0
║  Requests sent      : 20
║  Dataset            : ShareGPT (seed=42)
║  Mode               : sequential
╠══════════════════════════════════════════════════════════════╣
║  Total input tokens : 4,823
║  Total output tokens: 3,241
║  Total wall time    : 28.4s
╠══════════════════════════════════════════════════════════════╣
║  Output throughput  : 114.1 tok/s
║  Total throughput   : 283.2 tok/s
║  Request rate       : 0.703 req/s
╠══════════════════════════════════════════════════════════════╣
║  Avg latency        : 1418 ms
╚══════════════════════════════════════════════════════════════╝
```

These are the baseline numbers on an RTX 4060 Ti with Qwen3-0.6B. Every subsequent layer in this curriculum is evaluated against the same 20 ShareGPT conversations, the same seed, the same hardware — so the numbers are directly comparable.

## Reading the Numbers

114 output tokens per second means the server produces roughly 114 new tokens for every second of wall-clock time across the full 20-request run. At 4–5 tokens per word, that is approximately 22–28 words per second of generated text. A single user looking at a streaming response would find this fast enough; 10 concurrent users would each experience the full 28-second wait in sequence.

The average latency of 1418 ms is dominated by the cost of running a forward pass through the model for every generated token. Prompt lengths in this ShareGPT sample average around 240 tokens, and for each of the (up to 128) decode steps the model processes the full sequence again from scratch. Prompts that happen to be shorter complete faster; longer prompts visibly drag.

The request rate of 0.70 req/s is simply 20 requests divided by 28.4 seconds. In a batched system this number can increase independently of output throughput; here, because the server is strictly sequential, the two are coupled.

## What the Benchmark Reveals About Layer 0

The benchmark exposes the two most visible costs of the naive design. First, because requests are processed one at a time, the GPU is underutilised during the decode phase: generating a single token from a single sequence is a very small operation that leaves most of the GPU idle. Second, the sequential benchmark does not even expose the head-of-line blocking behaviour — to see that, `test_client.py`'s concurrent test is the right tool, and it shows that N simultaneous clients experience roughly N × single-request latency.

The numbers in `benchmark.md` are the reference point. Write them down: **114 tok/s output throughput, 1418 ms average latency**. Every improvement in every subsequent layer will be measured against these figures.
