# Lesson 1 — Why Naive Inference is Slow

**Prerequisites:** You can run a HuggingFace model with `model.generate()`. You know what
a transformer attention layer does. You are comfortable reading PyTorch code.

**What you will understand after this lesson:** Why running a language model one request
at a time is catastrophically inefficient, what the KV cache is and why it exists, and
why even with the KV cache you quickly run into a new problem that makes serving hard.
This is the "why" behind everything SGLang does.

---

## 1.1 — A Transformer Generates One Token at a Time

Let's start with what you already know. When you call a HuggingFace model to generate
text, you do something like this:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
model = model.cuda()

prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate up to 50 tokens
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0]))
```

This works. But what is `model.generate` actually doing? It is not running the transformer
once and getting 50 tokens out. A transformer is not built that way. Instead, it runs the
forward pass *once per output token*. Each forward pass takes the entire sequence so far
as input and produces a probability distribution over the vocabulary for the *next* token.
One token is sampled from that distribution, appended to the sequence, and then the whole
thing runs again. This loop — forward pass, sample, append, repeat — is called
**autoregressive decoding**.

For a prompt of 10 tokens and 50 output tokens, there are 50 forward passes. Each forward
pass processes a sequence that grows by one token per step: first 10 tokens, then 11,
then 12, all the way to 59. This is the fundamental rhythm of language model inference,
and understanding it is the foundation for understanding everything else.

---

## 1.2 — What Happens Inside One Forward Pass

Let's trace what happens in a single transformer forward pass. Consider a decoder-only
model like Qwen3. It has an embedding layer, then N identical transformer blocks stacked
on top of each other, then a final language model head that maps hidden states to
vocabulary logits.

Each transformer block contains two sub-layers: a **multi-head attention** layer and a
**feed-forward** (MLP) layer. The attention layer is what we care about most.

In the attention layer, the current sequence of hidden states — one vector per token — is
projected into three tensors: Query (Q), Key (K), and Value (V). Each is of shape
`[sequence_length, num_heads, head_dim]`. The attention scores are computed as:

```
scores = softmax(Q @ K.T / sqrt(head_dim))   # [seq_len, seq_len]
output = scores @ V                            # [seq_len, hidden_size]
```

This is the core quadratic operation: every query token attends to every key token. The
computation cost grows with the *square* of the sequence length. A sequence of 1000 tokens
requires roughly 1,000,000 pairwise attention computations per layer.

Here is the crucial insight for this lesson: in autoregressive decoding, we only need the
output for the **last** token in the sequence, because that is the one we are predicting.
But the attention operation computes outputs for all positions. Worse, the K and V tensors
for all the *previous* tokens are computed fresh every single step, even though they have
not changed.

To make this concrete: at decode step 40 (processing 49 tokens to predict token 50), you
recompute K and V for all 49 tokens even though tokens 1 through 48 were already fully
processed at step 39. You are throwing away and recomputing the same numbers, over and
over, for every new token.

This is the naive baseline. It is correct but wildly inefficient. For a 1000-token prompt
generating 100 tokens, you recompute the K and V for all 1000 prompt tokens 100 separate
times. Each recomputation requires passing 1000 vectors through two linear layers per
layer of the model — that is a lot of floating-point operations and a lot of GPU memory
bandwidth wasted on data that has not changed.

---

## 1.3 — The KV Cache: Save What You Already Computed

The solution is obvious once you see the problem: if the K and V tensors for past tokens
do not change, save them. This is the **KV cache**.

HuggingFace models implement this with `past_key_values`. When you call the model
forward with `use_cache=True`, each attention layer returns its K and V tensors for the
current step, and on the next step it receives those saved tensors as input:

```python
# Manual autoregressive loop showing KV cache explicitly
past_kv = None
input_ids = tokenizer.encode("What is the capital of France?", return_tensors="pt").cuda()

for step in range(50):
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_kv,
            use_cache=True,
        )
    
    # outputs.past_key_values contains cached K, V for all past tokens
    past_kv = outputs.past_key_values
    
    # Only feed the new token on subsequent steps
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    input_ids = next_token  # <-- just the one new token, not the full sequence!
    
    if next_token.item() == tokenizer.eos_token_id:
        break
```

With the KV cache, instead of processing the full sequence every step, you process exactly
one token (or the new chunk of tokens in the prefill phase). The K and V for all past
tokens are read from GPU memory rather than recomputed. The attention computation goes
from O(seq_len²) per step to O(seq_len) per step — a massive improvement.

The HuggingFace `model.generate()` function does exactly this internally. When you call it,
it runs a full forward pass on the prompt tokens once (the "prefill"), then enters a decode
loop where it processes one token per step while accumulating `past_key_values`.

---

## 1.4 — What the KV Cache Actually Is in Memory

It is worth being concrete about what `past_key_values` looks like in memory, because this
is exactly what SGLang manages at a much finer level.

For a model with `L` layers, `H` attention heads, and `D` head dimension, the KV cache
for a sequence of `T` tokens contains:

- `L` Key tensors, each of shape `[T, H, D]`
- `L` Value tensors, each of shape `[T, H, D]`

For Qwen3-0.6B specifically: 28 layers, 16 attention heads, head_dim=64. For a sequence
of 512 tokens and bfloat16 precision (2 bytes per element):

```
Memory per sequence = 2 × 28 × 512 × 16 × 64 × 2 bytes
                    = 2 × 28 × 512 × 2048 bytes
                    ≈ 58 MB
```

That is 58 MB for a single 512-token sequence. A modern GPU with 16 GB of VRAM has
about 13–14 GB available after loading the model weights. Simple division: you can hold
roughly 220 sequences at once. That is actually the theoretical maximum — in practice,
other memory usage and fragmentation reduce it further.

This is not a small number, but it is finite. KV cache memory is the primary bottleneck
in LLM serving. Everything SGLang does around memory management — the pool allocator, the
radix tree, paging, eviction — exists to use this finite memory as efficiently as possible.

---

## 1.5 — One Request at a Time: The GPU Utilization Problem

Now consider what happens when you receive requests one at a time, in sequence. A user
sends a request; you run prefill, then decode until done, then return the response, then
process the next request.

The problem is that the decode phase has an extremely poor compute-to-memory ratio. Modern
GPUs are designed for throughput: they can execute trillions of floating-point operations
per second. But they can only sustain this throughput if their arithmetic units are kept
busy. During the decode phase of a single request:

- The input at each step is a single token vector: a tensor of shape `[1, hidden_size]`.
- Multiplying a `[1, hidden_size]` vector by a weight matrix `[hidden_size, hidden_size]`
  has only `hidden_size²` multiply-add operations.
- For Qwen3-0.6B with hidden_size=1024, that is roughly 1 million operations per linear layer.
- But the GPU must read the entire weight matrix — 1024 × 1024 × 2 bytes = 2 MB — from
  memory just to do those 1 million operations.

The GPU is *memory-bandwidth-bound*, not compute-bound. Its arithmetic units sit idle
while waiting for data to arrive from GPU memory. Utilization is typically below 10%
during single-request decode. The GPU is capable of 10× more work but has nothing to do.

This is the core argument for batching: if you process 32 requests simultaneously instead
of 1, the input matrix becomes `[32, hidden_size]` instead of `[1, hidden_size]`. The
weight matrix is still only read once, but now it does 32× more useful work. GPU
utilization rises dramatically, and throughput (tokens per second across all users)
improves nearly linearly with batch size.

---

## 1.6 — Naive Batching Breaks Because Sequences Have Different Lengths

The natural response is: batch requests together. But there is an immediate problem.
Different users send prompts of different lengths and expect completions of different
lengths. Naive batching requires all sequences in a batch to have the same length, so you
pad shorter sequences with dummy tokens:

```python
# Padding-based batch (the naive approach)
prompts = [
    "Hi",                                     # 2 tokens
    "What is the capital of France?",         # 8 tokens
    "Explain the history of the Roman Empire" # 8 tokens (much shorter!)
]

# Tokenize with padding to match the longest sequence
inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
# inputs["input_ids"] shape: [3, 8] — all padded to length 8

outputs = model.generate(**inputs, max_new_tokens=200)
```

The KV cache in this batched setting has shape `[batch_size, num_layers, seq_len, num_heads, head_dim]`.
Every sequence holds a KV cache for the full padded length, even though "Hi" only needed 2 tokens.

But the padding problem during generation is worse. Every sequence in the batch must
generate tokens until *all* sequences are finished. If you batch a request that needs
5 output tokens with one that needs 200, you must wait for the 200-token request to finish
before returning *any* response. The 5-token request is essentially frozen in time,
occupying KV cache memory and being processed repeatedly even after it has logically
finished. This is called **head-of-line blocking**.

The GPU's memory is also wasted on padding. A batch with sequences of lengths
[2, 50, 100, 150] padded to length 150 wastes (150-2 + 150-50 + 150-100) / (150×4) ≈
49% of its KV cache memory on tokens that do not exist.

---

## 1.7 — The Two Phases Have Fundamentally Different Characters

There is one more important distinction that naive batching misses: prefill and decode
are not the same kind of operation, and trying to handle them uniformly is a mistake.

**Prefill** is compute-bound. You are running attention over a potentially large sequence
all at once. The Q, K, V tensors are large, the attention score matrix is large, and the
GPU arithmetic units are fully utilized. A long prompt might take hundreds of milliseconds
to prefill. This is fundamentally a matrix-multiplication-heavy workload.

**Decode** is memory-bandwidth-bound. You are processing one new token per step, reading
the large KV cache from GPU memory at each step. A single decode step might take only
5–20 milliseconds on a modern GPU, but the GPU is largely idle from a compute standpoint.

Mixing these two phases naively in one batch creates problems. If you try to prefill a
long prompt while also decoding other requests, the prefill computation takes so long that
the decode requests experience large latency spikes. This is called **decode starvation**.

The ideal system would:
1. Process prefill for new requests quickly and efficiently.
2. Keep decoding for existing requests running continuously with minimal interruption.
3. Batch decode steps across as many requests as possible to maximize GPU utilization.
4. Never pad or wait for unrelated requests to finish.

This is exactly what SGLang's continuous batching and scheduler achieve. But that is
the topic of Lesson 4.

---

## 1.8 — What SGLang Does Differently (Preview)

By the end of this lesson series you will understand each of these in detail. For now,
a preview of the answers to the problems raised in this lesson:

The answer to **recomputing K and V** is the KV cache (which you now understand in depth).

The answer to **finite KV cache memory** is the pool-based memory manager
(`srt/mem_cache/memory_pool.py`) — a flat pre-allocated GPU buffer of "slots", each holding
the KV pairs for one token, allocated and freed on demand rather than per-sequence.

The answer to **padding waste** is paged attention — storing KV pairs in fixed-size pages
that are allocated only when needed, like virtual memory pages in an OS. No padding, no
wasted slots.

The answer to **head-of-line blocking** is continuous batching — sequences enter and leave
the batch at any time, not in lockstep. A finished sequence immediately frees its memory
for a waiting one.

The answer to **decode starvation** is chunked prefill — new prompts are processed in
small chunks per scheduling step, so running decode requests are never blocked for more
than one chunk-worth of time.

---

## 1.9 — Connecting to the SGLang Source Code

Now that you understand the problem, here is where each concept lives in the SGLang
codebase. You do not need to read these files in depth yet — that is what later lessons
are for. This is just to orient you.

**The KV cache tensors** are allocated and managed in:
```
REPOS/sglang/python/sglang/srt/mem_cache/memory_pool.py
```
The class `ReqToTokenPool` maps a request ID to the list of slot indices it is using.
The class `KVCache` is the actual GPU tensor that stores K and V pairs.

**The attention layer** that reads and writes through this pool is `RadixAttention`:
```
REPOS/sglang/python/sglang/srt/layers/radix_attention.py
```
This replaces the standard `nn.MultiheadAttention`. Instead of computing attention on a
contiguous per-sequence tensor, it takes a list of slot indices and accesses the shared pool.

**The model using RadixAttention** — look at the Qwen3 attention layer:
```
REPOS/sglang/python/sglang/srt/models/qwen3.py
```
At line 144 you can see: `self.attn = RadixAttention(...)`. This is the drop-in
replacement for standard attention. It is what makes SGLang's memory management possible.

**The forward pass that drives it all** is in:
```
REPOS/sglang/python/sglang/srt/model_executor/model_runner.py
```
This is where `ForwardBatch` (the GPU-resident batch structure) is passed into the model.

---

## 1.10 — Exercises to Solidify Understanding

These are not assignments to be graded. They are experiments you can run right now in
your working environment to see these concepts directly.

**Exercise A — Measure the cost of not caching.**
Write a loop that calls `model.forward(full_sequence)` at each decode step without any
KV cache. Measure wall time for 50 steps with a 100-token prompt. Then repeat using
`past_key_values`. Compare the times.

```python
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/path/to/local/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()

prompt_ids = tokenizer.encode("What is the capital of France?", return_tensors="pt").cuda()
N_STEPS = 30

# --- Without KV cache ---
current_ids = prompt_ids.clone()
t0 = time.perf_counter()
for _ in range(N_STEPS):
    with torch.no_grad():
        out = model(input_ids=current_ids, use_cache=False)
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    current_ids = torch.cat([current_ids, next_tok], dim=-1)
t_no_cache = time.perf_counter() - t0

# --- With KV cache ---
current_ids = prompt_ids.clone()
past_kv = None
t0 = time.perf_counter()
for _ in range(N_STEPS):
    with torch.no_grad():
        out = model(input_ids=current_ids, past_key_values=past_kv, use_cache=True)
    past_kv = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    current_ids = next_tok  # only the new token
t_with_cache = time.perf_counter() - t0

print(f"No cache:   {t_no_cache:.2f}s  ({N_STEPS/t_no_cache:.1f} tok/s)")
print(f"With cache: {t_with_cache:.2f}s  ({N_STEPS/t_with_cache:.1f} tok/s)")
print(f"Speedup:    {t_no_cache/t_with_cache:.1f}x")
```

**Exercise B — Measure KV cache memory.**
Check how much GPU memory is consumed before and after the KV cache is created:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/path/to/local/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)

torch.cuda.synchronize()
mem_before = torch.cuda.memory_allocated() / 1024**2
print(f"After model load: {mem_before:.0f} MB")

prompt = "Tell me a very long story about " * 20  # make a long prompt
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
print(f"Prompt length: {inputs['input_ids'].shape[1]} tokens")

with torch.no_grad():
    out = model(**inputs, use_cache=True)

torch.cuda.synchronize()
mem_after = torch.cuda.memory_allocated() / 1024**2
kv_size = sum(
    k.nbytes + v.nbytes
    for k, v in out.past_key_values
) / 1024**2

print(f"After forward:    {mem_after:.0f} MB")
print(f"KV cache size:    {kv_size:.1f} MB")
```

**Exercise C — Read the RadixAttention class.**
Open `REPOS/sglang/python/sglang/srt/layers/radix_attention.py` and read the `forward`
method. Notice that it does not receive `past_key_values` like a HuggingFace model.
Instead, it receives a `ForwardBatch` object that contains GPU tensors pointing into the
shared KV pool. The key question to ask as you read: how does it know *where* in the pool
to read past KV pairs, and *where* to write the new ones? The answer is in `forward_batch.req_to_token`
and `forward_batch.out_cache_loc`. Hold that question — Lesson 7 explains it completely.

---

## Summary

A language model generates one token per forward pass, autoregressively. Without a KV
cache, every step recomputes the key and value tensors for all past tokens — an enormous
waste. The KV cache solves this by saving those tensors. But the KV cache itself
consumes substantial GPU memory, that memory is the primary bottleneck in serving, and
naive approaches to batching waste it through padding and head-of-line blocking. The
decode phase is memory-bandwidth-bound rather than compute-bound, so batching many
requests together is essential for GPU efficiency. SGLang's entire architecture is
designed to pack as many requests as possible into GPU memory without waste, while keeping
the GPU arithmetic units busy at all times. The next lesson goes deeper into what the KV
cache actually stores and how the SGLang memory pool differs from the HuggingFace
`past_key_values` approach.

---

**Next:** Lesson 2 — The KV Cache in Detail (`LESSONS/lesson2/lesson.md`)
