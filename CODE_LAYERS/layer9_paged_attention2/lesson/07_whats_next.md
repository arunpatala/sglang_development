# 07 — What Comes Next

Layer 9 eliminates three sources of per-step overhead that were present in Layer 8: the Python loop over historical slot lists, the CPU-to-GPU copy of the full `kv_indices` array, and the per-step reconstruction of the `BatchDecodeWithPagedKVCacheWrapper`. The decode step's CPU-side work is now O(B) — five small integer tensors, a `torch.cumsum` on the GPU, a Triton kernel launch, and one `begin_forward` call. The bottleneck has shifted.

---

## What Remains

Two structural costs are not yet eliminated.

The first is `torch.empty` for `kv_indices` each decode step. `total_pages_in_batch` changes every step as requests arrive and finish, so the size of `kv_indices` is not fixed at startup. The allocation itself is lightweight — PyTorch's caching allocator handles it in nanoseconds — but it introduces a dependency on the Python interpreter and the allocator's internal lock before the Triton kernel can be launched. SGLang's production runtime pre-allocates a `kv_indices` buffer at the maximum possible size and slices it, eliminating this allocation entirely.

The second is the Python `sum(num_pages_list)` that computes `total_pages_in_batch`. It is O(B) but runs in the Python interpreter. This could be replaced by reading `kv_indptr[B]` from the GPU after `cumsum` — but that would require a GPU-to-CPU synchronization (a `kv_indptr[-1].item()` call), which is typically slower than the Python sum for batch sizes below a few hundred. At very large batch sizes, a persistent host-mapped `kv_indptr` buffer that the GPU writes into directly would allow the CPU to read the total without a synchronization.

---

## Chunked Prefill

Layer 9 runs each prefill as a separate B=1 forward pass. Long prompts — 4096 tokens at maximum context — occupy a single, exclusive forward pass that blocks the decode loop. Production systems use chunked prefill: the prompt is split into chunks of up to `chunk_size` tokens (typically 512 or 2048), and each chunk is batched with the ongoing decode requests. The prefill tokens attend causally over the chunk; the decode tokens attend over the full pool. This requires a mixed-mode attention kernel that handles prefill and decode queries in the same forward pass, and a `ForwardBatch` that carries both `PREFILL` and `DECODE` context. FlashInfer's `BatchPrefillWithPagedKVCacheWrapper` is the kernel that handles this case.

---

## Tensor Parallelism

Every forward pass in all nine layers runs on a single GPU. For models that do not fit on one GPU — Qwen3-7B and above — the weight matrices are sharded across devices and Q/K/V projections run in parallel. This requires `all-reduce` calls after the attention output projection and after the MLP gate projection. The `ModelRunner` would need to initialize a distributed process group and wrap the linear layers in tensor-parallel shards. The KV cache design in Layer 9 is compatible with tensor parallelism: each GPU holds a shard of the K and V heads, and `ReqToTokenPool` is replicated on each device.

---

## Prefix Caching

If two requests share a common prefix — a system prompt, a few-shot template — they currently pay the full prefill cost independently. Prefix caching stores the KV pages for shared prefixes in the pool and reuses them across requests. When a new request arrives whose prefix matches a cached sequence, `prefill` skips the transformer forward pass for the shared portion and directly records the cached page indices in `ReqToTokenPool`. Because `ReqToTokenPool` stores page indices, prefix sharing requires no data copy — the cached pages are already in `KVPool`, and only their indices need to be written into the new request's row.

---

## Speculative Decoding

Speculative decoding uses a small draft model to propose K candidate tokens and a large target model to verify them in one forward pass. If all K tokens are accepted, the effective throughput is K tokens per target-model step. The paged KV cache in Layer 9 is compatible with speculative decoding: the draft tokens are treated as a mini-prefill (stored in the pool pages), and the verification pass runs with `q_len = K` over the existing history in the pool. `kv_last_page_lens` generalizes naturally — the last page of the draft tokens may be partially filled, and FlashInfer handles that correctly.

---

## What Layer 9 Represents in the SGLang Stack

The three changes in Layer 9 — `ReqToTokenPool`, the Triton `kv_indices` kernel, and variable `page_size` — are the direct equivalents of the structures in SGLang's `srt/mem_cache/memory_pool.py` and `srt/layers/attention/utils.py`. Production SGLang adds prefix caching, chunked prefill, tensor parallelism, and speculative decoding on top of exactly this foundation. The KV pool shape, the page-index table, and the on-GPU index build are unchanged between the educational Layer 9 and the production implementation. Understanding Layer 9 is understanding the core of how SGLang manages GPU memory for concurrent LLM inference.
