"""
triton_utils.py — GPU-side index utilities for the paged KV cache.

Ported from SGLang:
  sglang/srt/layers/attention/utils.py::create_flashinfer_kv_indices_triton

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Why this kernel exists
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FlashInfer's decode wrapper needs a flat int32 tensor `kv_indices` listing
every physical pool slot for every active request, laid out as:

    [slot_0_req0, slot_1_req0, ..., slot_N0_req0,
     slot_0_req1, slot_1_req1, ..., slot_N1_req1, ...]

In Layer 7 this was built in Python:

    for i, req in enumerate(reqs):
        kv_indices_list.extend(req.slot_indices)
        kv_indices_list.append(new_slots[i])
    kv_indices = torch.tensor(kv_indices_list, device='cuda')   # CPU → GPU copy

Problems:
  - Python iteration over per-request slot lists (O(Σ kv_lens) Python ops)
  - Host-to-device copy of the entire index array every step

Layer 8 replaces this with `create_flashinfer_kv_indices_triton`:
  - One GPU threadblock per request (all B requests in parallel)
  - Reads directly from `req_to_token[req_pool_idx, 0:seq_len]` on GPU
  - Writes the flat `kv_indices` output — no Python loop, no CPU→GPU copy

The only CPU→GPU data that moves each step is the small metadata:
    req_pool_indices  [B] int32   ← which req_to_token row per request
    seq_lens          [B] int32   ← how many tokens each request has
Both fit in a single cache line and are unavoidable.
"""

import triton
import triton.language as tl


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,       # [max_batch, max_context_len]  int32 GPU table
    req_pool_indices_ptr,   # [B]  which row per request
    page_kernel_lens_ptr,   # [B]  how many KV tokens per request (seq_len+1)
    kv_indptr,              # [B+1]  cumulative offsets into kv_indices
    kv_start_idx,           # [B] or None — offset within the row (for SWA, unused here)
    kv_indices_ptr,         # [Σ seq_lens]  output: flat pool slot indices
    req_to_token_ptr_stride: tl.constexpr,  # req_to_token.shape[1]
):
    """
    One Triton program per request (launched with grid=(B,)).

    For request pid:
      row  = req_to_token[req_pool_indices[pid]]
      out  = kv_indices[kv_indptr[pid] : kv_indptr[pid+1]]
      out  = row[kv_start : kv_start + kv_len]

    Processes the row in 512-element chunks (BLOCK_SIZE) to coalesce
    memory accesses across the warp.
    """
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    req_pool_index   = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end   = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end   = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask   = offset < kv_end - kv_start
        data   = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)
