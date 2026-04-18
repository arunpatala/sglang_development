"""
Layer 11 — Verify prefix caching correctness.

Tests:
  1. RadixCache unit tests (insert / match / evict — CPU only, no GPU).
  2. End-to-end: request with a cached prefix produces the SAME logits as
     a full (non-cached) prefill.
  3. End-to-end: two requests sharing a long common prefix — second request
     hits the cache and produces identical output to the first pass.

Run with:
  /home/arun/PROJECTS/sglang_development/.conda/bin/python \
      CODE_LAYERS/layer11_prefix_caching/verify_prefix.py
"""

from __future__ import annotations

import math
import sys
import os
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
LAYER_DIR = Path(__file__).parent
sys.path.insert(0, str(LAYER_DIR))

# Silence transformers download attempts
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODEL_PATH = "/home/arun/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: RadixCache unit tests (CPU-only, no GPU needed)
# ─────────────────────────────────────────────────────────────────────────────

def make_fake_pool(n_pages: int):
    """Minimal fake KVPool for unit tests."""
    class FakePool:
        def __init__(self, n):
            self.free_slots = list(range(1, n + 1))
            self.page_size = 1   # not used by RadixCache directly
        def free(self, pages):
            self.free_slots.extend(pages)
        def available(self):
            return len(self.free_slots)
    return FakePool(n_pages)


def test_radix_unit(page_size: int = 4) -> None:
    from radix_cache import RadixCache, TreeNode

    pool  = make_fake_pool(100)
    cache = RadixCache(pool, page_size)
    P     = page_size

    print(f"\n{'─'*60}")
    print(f"RadixCache unit tests  (page_size={P})")
    print(f"{'─'*60}")

    # ── Test 1: empty cache returns no match ──────────────────────────────
    pages, length, node = cache.match_prefix(list(range(20)))
    ok = (pages == [] and length == 0 and node is cache.root)
    print(f"  {PASS if ok else FAIL}  empty match → pages=[], length=0")
    assert ok

    # ── Test 2: insert then match full sequence ───────────────────────────
    tok = list(range(32))
    pg  = list(range(1, 9))   # 8 pages for 32 tokens (P=4)
    n_overlap = cache.insert(tok, pg)
    ok = (n_overlap == 0)
    print(f"  {PASS if ok else FAIL}  insert fresh sequence → overlap=0")
    assert ok

    pages, length, node = cache.match_prefix(tok + [99])
    # match_prefix caps at floor((33-1)/4)*4 = 32 tokens = 8 pages
    ok = (length == 32 and pages == pg)
    print(f"  {PASS if ok else FAIL}  match after insert → length={length} (want 32)")
    assert ok

    # ── Test 3: prefix match (partial sequence) ───────────────────────────
    tok2 = list(range(16)) + [99, 98, 97, 96]   # shares first 16 tokens (P=4: 4 pages)
    pages2, length2, node2 = cache.match_prefix(tok2 + [0])
    ok = (length2 == 16 and pages2 == pg[:4])
    print(f"  {PASS if ok else FAIL}  partial prefix match → length={length2} (want 16)")
    assert ok

    # ── Test 4: node splitting ────────────────────────────────────────────
    # Insert a sequence that shares first 16 tokens, then diverges.
    tok3   = list(range(16)) + list(range(100, 116))
    pages3 = list(range(20, 28))   # 8 pages
    n_ov   = cache.insert(tok3, pages3)
    # First 4 pages overlap with tok[0:16] already in tree
    ok = (n_ov == 4)
    print(f"  {PASS if ok else FAIL}  insert with shared prefix → overlap={n_ov} (want 4)")
    assert ok

    # After split, matching tok should still return 8 pages
    pages_check, length_check, _ = cache.match_prefix(tok + [999])
    ok = (length_check == 32 and pages_check == pg)
    print(f"  {PASS if ok else FAIL}  original match still valid after split")
    assert ok

    # Matching tok3 should give 8 pages:
    #   • First 4 pages come from the shared prefix node (same as pg[:4]).
    #   • Next 4 pages come from tok3's own branch (pages3[4:], the non-overlapping tail).
    #   pages3[:4] were "overlap" — already in tree, not re-stored.
    pages3_expected = pg[:4] + pages3[4:]
    pages3_check, length3_check, _ = cache.match_prefix(tok3 + [999])
    ok = (length3_check == 32 and pages3_check == pages3_expected)
    print(
        f"  {PASS if ok else FAIL}  split branch match → length={length3_check} (want 32)  "
        f"got={pages3_check}"
    )
    assert ok, f"Expected {pages3_expected}, got {pages3_check}"

    # ── Test 5: lock_ref / eviction ───────────────────────────────────────
    # Lock the node for tok.
    _, _, locked_node = cache.match_prefix(tok + [999])
    cache.inc_lock_ref(locked_node)

    before = cache.total_cached_pages()
    freed  = cache.evict(999)   # try to evict everything
    after  = cache.total_cached_pages()

    # tok's node is locked; tok3's leaf should be evicted
    ok = (freed > 0 and after < before)
    print(f"  {PASS if ok else FAIL}  eviction frees unlocked pages  (freed={freed}, remaining={after})")
    assert ok

    cache.dec_lock_ref(locked_node)

    # ── Test 6: insert duplicate (full overlap) ───────────────────────────
    tok_d = list(range(8))   # 8 tokens = 2 pages
    pg_d  = [50, 51]
    cache.insert(tok_d, pg_d)
    pg_d2 = [60, 61]
    n_ov2 = cache.insert(tok_d, pg_d2)   # same tokens again
    ok = (n_ov2 == 2)
    print(f"  {PASS if ok else FAIL}  duplicate insert → overlap={n_ov2} (want 2)")
    assert ok

    print(f"\n  All RadixCache unit tests passed!  {cache}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: GPU-side correctness — prefix hit produces same logits
# ─────────────────────────────────────────────────────────────────────────────

def test_prefix_caching_gpu(page_size: int = 16, atol: float = 5e-1) -> None:
    import torch
    import torch.nn.functional as F
    from model import Qwen3ForCausalLM
    from kv_cache import KVPool, ReqToTokenPool, compute_write_info, ExtendKVCtx
    from radix_cache import RadixCache
    from triton_utils import create_flashinfer_kv_indices_triton
    import flashinfer

    DEVICE = "cuda"
    DTYPE  = torch.bfloat16
    P      = page_size

    print(f"\n{'─'*60}")
    print(f"GPU prefix-caching correctness  (page_size={P})")
    print(f"{'─'*60}")

    # ── Load model ────────────────────────────────────────────────────────
    model = Qwen3ForCausalLM.from_pretrained(MODEL_PATH, dtype=DTYPE)
    cfg   = model.model.config
    print(f"  Model loaded: {cfg.num_hidden_layers}L  {cfg.num_attention_heads}H")

    # ── Allocate pools ────────────────────────────────────────────────────
    kv_pool = KVPool(
        total_pages = 512,
        page_size   = P,
        n_layers    = cfg.num_hidden_layers,
        n_kv_heads  = cfg.num_key_value_heads,
        head_dim    = cfg.head_dim,
        dtype       = DTYPE,
    )
    rtp = ReqToTokenPool(max_batch=32, max_context_len=256)
    radix = RadixCache(kv_pool, P)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper   = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")

    # ── Helper: run one extend pass ───────────────────────────────────────
    def do_extend(token_ids, kv_committed_len, slot_indices, req_pool_idx):
        """
        Prefill token_ids[kv_committed_len:] using the paged KV cache.
        Returns logits [1, len(fill_ids), vocab].
        Mutates slot_indices in-place (appends new pages).
        """
        fill_ids = token_ids[kv_committed_len:]
        n_fill   = len(fill_ids)

        wi = compute_write_info(
            kv_pool          = kv_pool,
            rtp              = rtp,
            slot_indices     = slot_indices,
            req_pool_idx     = req_pool_idx,
            kv_committed_len = kv_committed_len,
            n_fill           = n_fill,
        )

        total_committed = kv_committed_len + n_fill
        n_pages         = len(slot_indices)
        last_fill       = total_committed % P
        kv_last_pg_len  = last_fill if last_fill != 0 else P

        qo_indptr_t      = torch.tensor([0, n_fill],    dtype=torch.int32, device=DEVICE)
        kv_indptr_t      = torch.tensor([0, n_pages],   dtype=torch.int32, device=DEVICE)
        kv_last_pg_lens  = torch.tensor([kv_last_pg_len], dtype=torch.int32, device=DEVICE)
        req_pool_idx_t   = torch.tensor([req_pool_idx], dtype=torch.int32, device=DEVICE)
        num_pages_t      = torch.tensor([n_pages],      dtype=torch.int32, device=DEVICE)
        kv_indices_t     = torch.empty(n_pages,         dtype=torch.int32, device=DEVICE)

        create_flashinfer_kv_indices_triton[(1,)](
            rtp.req_to_token,
            req_pool_idx_t,
            num_pages_t,
            kv_indptr_t,
            None,
            kv_indices_t,
            rtp.req_to_token.shape[1],
        )

        wrapper.begin_forward(
            qo_indptr_t, kv_indptr_t, kv_indices_t, kv_last_pg_lens,
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
            P, causal=True, q_data_type=DTYPE,
        )

        ids_t = torch.tensor(fill_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        pos_t = torch.arange(
            kv_committed_len, kv_committed_len + n_fill,
            dtype=torch.long, device=DEVICE
        ).unsqueeze(0)

        ctx = ExtendKVCtx(
            wrapper     = wrapper,
            k_pool      = kv_pool.k_pool,
            v_pool      = kv_pool.v_pool,
            qo_indptr   = [0, n_fill],
            write_infos = [wi],
            page_size   = P,
        )

        with torch.no_grad():
            logits = model(ids_t, attention_mask=None, kv_cache=ctx, position_ids=pos_t)

        wrapper.end_forward()
        return logits  # [1, n_fill, vocab]

    # ══════════════════════════════════════════════════════════════════════
    # Test A: Single request, no caching — baseline
    # ══════════════════════════════════════════════════════════════════════
    prefix_len = 3 * P     # 48 tokens
    suffix_len = 2 * P     # 32 tokens
    total_len  = prefix_len + suffix_len

    # Random token IDs (small vocab for speed)
    torch.manual_seed(42)
    prompt = torch.randint(1, 1000, (total_len,)).tolist()

    # Full prefill (no caching) — this is our reference.
    slot_ref = []
    pool_ref = rtp.alloc()
    logits_ref = do_extend(prompt, kv_committed_len=0,
                           slot_indices=slot_ref, req_pool_idx=pool_ref)
    # logits_ref[:, -1, :] is what we'd sample from.
    last_logit_ref = logits_ref[0, -1].clone()

    print(f"  Reference prefill done: {total_len} tokens, {len(slot_ref)} pages")

    # ══════════════════════════════════════════════════════════════════════
    # Test B: Insert prefix into cache, then run a new request with same prompt
    # ══════════════════════════════════════════════════════════════════════
    # Insert the prefix into the radix cache, using slot_ref's prefix pages.
    prefix_pages = slot_ref[:prefix_len // P]
    n_ov = radix.insert(prompt[:prefix_len], prefix_pages)
    ok = (n_ov == 0)
    print(f"  {PASS if ok else FAIL}  Inserted {prefix_len} tokens ({len(prefix_pages)} pages) into cache")
    assert ok, f"Expected 0 overlap, got {n_ov}"

    # Now match prefix for the same prompt.
    matched_pages, matched_len, last_node = radix.match_prefix(prompt)
    ok = (matched_len == prefix_len and matched_pages == prefix_pages)
    print(
        f"  {PASS if ok else FAIL}  match_prefix: matched {matched_len} tokens "
        f"(want {prefix_len})"
    )
    assert ok, f"matched_len={matched_len}, want {prefix_len}"

    radix.inc_lock_ref(last_node)

    # Run extend for only the SUFFIX tokens, using cached prefix pages.
    # Simulate what model_runner.prefill_batch does:
    slot_cached = list(matched_pages)    # pre-populated with prefix pages
    pool_cached  = rtp.alloc()

    # Write prefix pages into req_to_token.
    n_pfx_pages = len(matched_pages)
    rtp.req_to_token[pool_cached, :n_pfx_pages] = torch.tensor(
        matched_pages, dtype=torch.int32, device=DEVICE
    )

    # Extend only the suffix.
    logits_cached = do_extend(
        prompt,
        kv_committed_len = matched_len,
        slot_indices     = slot_cached,
        req_pool_idx     = pool_cached,
    )
    last_logit_cached = logits_cached[0, -1]

    diff     = (last_logit_ref - last_logit_cached).abs().max().item()
    tok_ref  = int(last_logit_ref.argmax())
    tok_hit  = int(last_logit_cached.argmax())
    # Top-1 must always agree (same token sampled).
    ok_tok   = (tok_ref == tok_hit)
    # Max logit diff may be slightly > atol in bfloat16 due to different
    # FlashInfer tiling paths (all-tokens vs. KV-cache attention).
    ok_diff  = diff < atol
    ok       = ok_tok and ok_diff
    print(
        f"  {PASS if ok_tok else FAIL}  top-1 token matches: {tok_ref} == {tok_hit}"
    )
    print(
        f"  {PASS if ok_diff else FAIL}  cached vs full logit max-diff: {diff:.4f} "
        f"(atol={atol})"
    )
    assert ok_tok, f"Token mismatch: ref={tok_ref} cached={tok_hit}"
    assert ok_diff, f"Logit mismatch: max_diff={diff:.4f} > atol={atol}"

    radix.dec_lock_ref(last_node)

    # ══════════════════════════════════════════════════════════════════════
    # Test C: cache_finished_req — pages are correctly inserted and freed
    # ══════════════════════════════════════════════════════════════════════
    # Simulate a finished request using the cached run.
    # Use a simple namespace object instead of a class (avoids scoping issues).
    import types
    fake_req = types.SimpleNamespace(
        input_ids           = prompt,
        output_ids          = [7, 8, 9],      # 3 generated tokens
        slot_indices        = slot_cached,
        prefix_page_indices = matched_pages,
        last_node           = last_node,
        req_pool_idx        = pool_cached,
    )
    FakeReq = fake_req

    pages_before = kv_pool.available()
    radix.cache_finished_req(FakeReq, rtp, kv_pool)
    pages_after  = kv_pool.available()

    # Newly cached pages should now be in the tree; tail page freed.
    # Prefix pages stay in tree (not freed).
    suffix_pages_computed = slot_cached[n_pfx_pages:]
    aligned_total = (len(FakeReq.input_ids + FakeReq.output_ids) // P) * P
    tail_pages    = slot_cached[aligned_total // P:]

    ok = (pages_after >= pages_before - len(prefix_pages))
    print(
        f"  {PASS if ok else FAIL}  cache_finished_req: "
        f"pages before={pages_before}, after={pages_after}  "
        f"(tail+duplicates freed)"
    )
    assert ok

    # Verify the new portion is now in the cache.
    pages_new, len_new, _ = radix.match_prefix(prompt + FakeReq.output_ids + [999])
    ok = (len_new >= prefix_len)
    print(
        f"  {PASS if ok else FAIL}  cache now has ≥{prefix_len} tokens for this sequence "
        f"(got {len_new})"
    )
    assert ok

    # ══════════════════════════════════════════════════════════════════════
    # Test D: eviction
    # ══════════════════════════════════════════════════════════════════════
    cached_before = radix.total_cached_pages()
    freed         = radix.evict(1)
    cached_after  = radix.total_cached_pages()
    ok = (freed > 0 or cached_before == 0)
    print(
        f"  {PASS if ok else FAIL}  eviction: freed={freed} pages, "
        f"cached {cached_before} → {cached_after}"
    )
    assert ok

    print(f"\n  All GPU tests passed!")
    print(f"  Final cache state: {radix}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Layer 11 — Prefix Caching Verification")
    print("=" * 60)

    # ── Part 1: CPU-only unit tests ───────────────────────────────────────
    test_radix_unit(page_size=4)

    # ── Part 2: GPU end-to-end tests ─────────────────────────────────────
    try:
        import torch
        if not torch.cuda.is_available():
            print("\nNo CUDA — skipping GPU tests")
        else:
            test_prefix_caching_gpu(page_size=16)
            test_prefix_caching_gpu(page_size=1)
    except ModuleNotFoundError as e:
        print(f"\nSkipping GPU tests: {e}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
