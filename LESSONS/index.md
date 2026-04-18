# SGLang Internals — Lesson Plan

For someone who already knows: SGLang as a user, PyTorch, HuggingFace Transformers.
Goal: understand the internals well enough to contribute code.

Each lesson is one focused concept, explained in textbook prose with code.

---

## Module 1 — The Problem (Why Serving is Hard)

| Lesson | Title | Core Question |
|---|---|---|
| 1 | **Why Naive Inference is Slow** | What happens inside `model.generate()` and why it cannot scale |
| 2 | **The KV Cache — What It Is and Why It Exists** | How attention reuses past computation and what "caching" actually means in memory |
| 3 | **Why Batching Naive Requests Breaks Everything** | Static batching, padding waste, and the problem of variable-length sequences |

## Module 2 — Continuous Batching

| Lesson | Title | Core Question |
|---|---|---|
| 4 | **Continuous Batching — The Core Idea** | What changes when you don't wait for all sequences to finish |
| 5 | **Prefill vs Decode — Two Very Different Problems** | Why these two phases have different compute characteristics and must be treated differently |
| 6 | **Chunked Prefill — Balancing New Requests and Running Ones** | How large prompts are split to prevent decode starvation |

## Module 3 — Memory Management

| Lesson | Title | Core Question |
|---|---|---|
| 7 | **The KV Cache Pool — How SGLang Allocates GPU Memory** | ReqToTokenPool, TokenToKVPool, and the slot-based design |
| 8 | **Paged Attention — Borrowing from OS Virtual Memory** | Why paging eliminates fragmentation |
| 9 | **The RadixCache Part 1 — The Prefix Tree Data Structure** | What a radix tree is and how token sequences map to tree paths |
| 10 | **The RadixCache Part 2 — Prefix Matching and Eviction** | How match_prefix works, LRU eviction, reference counting |

## Module 4 — The Scheduler

| Lesson | Title | Core Question |
|---|---|---|
| 11 | **The Scheduler's Main Loop** | The event_loop_normal and event_loop_overlap loops step by step |
| 12 | **Schedule Policy — Longest Prefix Match** | Why LPM maximizes cache reuse and how it is implemented |
| 13 | **Preemption and Retract** | What happens when memory runs out mid-generation |

## Module 5 — The Process Architecture

| Lesson | Title | Core Question |
|---|---|---|
| 14 | **Three Processes, One Server** | Why SGLang uses separate OS processes and ZeroMQ |
| 15 | **TokenizerManager — The Async Front Door** | How the HTTP server and tokenizer coexist in one process |
| 16 | **DetokenizerManager — Streaming Tokens Back** | Incremental decoding, surrogate offsets, stop-string trimming |

## Module 6 — The Batch Pipeline

| Lesson | Title | Core Question |
|---|---|---|
| 17 | **ScheduleBatch → ModelWorkerBatch → ForwardBatch** | The three-level data structure transformation from CPU to GPU |
| 18 | **ForwardMode: EXTEND, DECODE, MIXED** | How the mode drives attention kernel selection |

## Module 7 — GPU Execution

| Lesson | Title | Core Question |
|---|---|---|
| 19 | **The ModelRunner — How the Forward Pass Works** | Loading weights, attention layer hooks, KV writes |
| 20 | **Attention Backends — FlashInfer, Triton, FlashAttention** | What makes each backend different and when each is used |
| 21 | **CUDA Graphs — Eliminating Kernel Launch Overhead** | What CUDA graphs are, how SGLang captures and replays them |

## Module 8 — Sampling and Output

| Lesson | Title | Core Question |
|---|---|---|
| 22 | **Sampling — From Logits to Token IDs** | Temperature, top-p, top-k, penalties implemented as tensor ops |
| 23 | **The HTTP Layer — FastAPI and OpenAI Compatibility** | Route handlers, Pydantic models, how to add custom endpoints |

## Module 9 — Advanced Topics

| Lesson | Title | Core Question |
|---|---|---|
| 24 | **Tensor Parallelism in SGLang** | How model weights are sharded and how NCCL allreduce fits in |
| 25 | **Speculative Decoding** | Draft model, verification, and why it can be faster |
| 26 | **LoRA Multi-Adapter Serving** | How multiple adapters share one base model on the same GPU |

---

Start with Lesson 1 → `lesson1/lesson.md`
