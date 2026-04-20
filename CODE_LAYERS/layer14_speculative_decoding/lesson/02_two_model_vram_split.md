# 02 — Two Models, Two Pools, One GPU

`SpecRunner` owns two `ModelRunner` instances — the target (large) model and the draft (small) model — each with its own independent KV pool and request tracking pool. Both models and both pools must fit on the same GPU simultaneously. This requires coordinating the VRAM allocation to leave headroom for both.

---

## The SpecRunner Constructor

```python
# spec_runner.py — SpecRunner.__init__
class SpecRunner:
    def __init__(self, cfg: SpecConfig, dtype=torch.bfloat16):
        self.cfg       = cfg
        self.target_mr = ModelRunner(
            model_path            = cfg.target_model_path,
            kv_memory_fraction    = cfg.target_kv_fraction,
            dtype                 = dtype,
            use_gptq              = cfg.target_use_gptq,
            enable_prefix_caching = cfg.enable_prefix_caching,
        )
        self.draft_mr  = ModelRunner(
            model_path            = cfg.draft_model_path,
            kv_memory_fraction    = cfg.draft_kv_fraction,
            dtype                 = dtype,
            use_gptq              = False,   # draft model is always bfloat16
            enable_prefix_caching = False,   # no prefix caching for draft
        )
        self._n_speculative_tokens = cfg.num_speculative_tokens
```

`target_kv_fraction + draft_kv_fraction` must be less than `1.0 - (model_weight_fraction)`. `model_weight_fraction` is approximately 0.08 for the target (GPTQ 1.7B, 1.25 GB on a 16 GB card) plus 0.075 for the draft (bfloat16 0.6B, 1.2 GB). Together, about 0.155 of total VRAM is consumed by weights. The remainder (approximately 0.845 = 13.5 GB) is available for the two KV pools.

A typical split: `target_kv_fraction=0.45`, `draft_kv_fraction=0.35`. Together 0.80, slightly under the 0.845 available, leaving approximately 0.045 (720 MB) for PyTorch's caching allocator, activations, and other runtime overhead.

---

## Independent ModelRunner Instances

Each `ModelRunner` is fully self-contained. It calls `torch.cuda.mem_get_info()` after loading its model weights and allocates `int(free_bytes * kv_memory_fraction / page_cost)` pages for its pool. If the target model loads first and consumes 1.25 GB, `free_bytes ≈ 14.75 GB`. With `target_kv_fraction=0.45`: `target_kv_pages ≈ int(14.75e9 × 0.45 / (16 × 114688)) ≈ 3630 pages`. After the target pool is allocated (3630 × 16 × 114688 ≈ 6.6 GB), `free_bytes ≈ 7.9 GB` remains. The draft model loads (consuming 1.2 GB), leaving `free_bytes ≈ 6.7 GB`. With `draft_kv_fraction=0.35`: `draft_kv_pages ≈ int(6.7e9 × 0.35 / (16 × draft_page_cost))`.

The draft model (Qwen3-0.6B) has different architecture parameters — 16 layers, 8 KV heads, 64 head dim (hypothetical). Its `bytes_per_token = 16 × 2 × 8 × 64 × 2 = 32768` bytes. `draft_kv_pages ≈ int(6.7e9 × 0.35 / (16 × 32768)) ≈ 4441 pages`.

These calculations happen automatically inside each `ModelRunner.__init__` — `SpecRunner` only needs to set the fractions, not compute page counts.

---

## ReqToTokenPool Symmetry

Both `ModelRunner` instances maintain a `ReqToTokenPool`. For each active request, the target model has `req.req_pool_idx` pointing to a row in the target `req_to_token_pool`, and the draft model has `d_req.req_pool_idx` pointing to a row in the draft `req_to_token_pool`. The two indices can be different values (the pools allocate independently from their own free lists).

The `SpecRunner` maintains a `_req_to_dreq` dict mapping each target `Req` to its corresponding `DraftReq`. `DraftReq` holds the draft pool index and draft slot indices (pages in the draft pool). The target's `req` and the draft's `d_req` represent the same conversation — but in separate physical pools on the same GPU.

---

## Why Two Separate Pools Instead of Shared

Sharing one pool between the target and draft model would require the target attention kernel and the draft attention kernel to read from the same `k_pool`/`v_pool` tensors. This is impossible because the two models have different KV head counts and head dimensions. The target model (1.7B) has 8 KV heads at 128 head dim; the draft (0.6B) might have 4 KV heads at 64 head dim. The pool shape `[total_pages, page_size, n_kv_heads, head_dim]` is model-specific and cannot be shared.

Separate pools also allow the eviction policies and page sizes to differ between the two models, and simplify cleanup on request completion — `SpecRunner.finish_req(req)` calls the target's `cache_finished_req` and the draft's separate free logic independently, with no cross-pool coordination.

Section 03 explains the draft KV mirroring mechanism — how the draft model's KV state is kept aligned with the target model's committed sequence.
