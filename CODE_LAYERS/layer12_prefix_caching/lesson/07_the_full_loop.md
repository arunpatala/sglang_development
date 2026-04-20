# 07 — The Full Loop

The previous sections explained the radix tree, prefix matching, lock reference counting, insert, eviction, and integration with `prefill_batch` independently. This section traces two sequential requests through the complete system: the first request populates the tree; the second request benefits from it.

---

## Setup

`page_size = 16`. `radix_cache` is initialized (empty). System prompt: 512 tokens (32 pages). Each request has the same system prompt followed by a 64-token user query.

R1: `input_ids = system_prompt(512) + user1(64)` — 576 tokens.
R2: `input_ids = system_prompt(512) + user2(64)` — 576 tokens.

---

## R1: First Request (Tree Empty)

**match_prefix(R1.input_ids):**

Tree is empty. `max_match = (576-1)//16*16 = 575//16*16 = 35*16 = 560`. Key = `tuple(input_ids[:560])`. `_match_helper(root, key)` finds no child at root — returns `([], root)`. Result: `([], 0, root)`.

`_apply_prefix_match` sees `matched_len = 0`, sets:
- `R1.prefix_page_indices = []`
- `R1.kv_committed_len = 0`
- `R1.last_node = root`
- `inc_lock_ref(root)` — root is never locked (the loop stops at `node is self.root`), so effectively no-op.
- `R1.fill_ids = R1.input_ids` (all 576 tokens).

**prefill_batch([R1]):**

Step 2 (prefix injection): `prefix_page_indices = []`, no-op.

`compute_write_info(kv_pool, rtp, slot_indices=[], req_pool_idx=row1, kv_committed_len=0, n_fill=576)`:
- `n_leftover = 0`. `remaining = 576`. Allocate `ceil(576/16) = 36` pages: `[a0..a35]`.
- `R1.slot_indices = [a0..a35]`. `req_to_token[row1, 0:36] = [a0..a35]`.

All 576 token IDs packed. Position IDs `[0..575]`. `qo_indptr = [0, 576]`. `kv_indptr = [0, 36]`. `kv_last_page_lens = [576%16 = 0 → 16]`.

The extend kernel processes 576 tokens, writes K/V into pages `a0..a35` across all 28 layers. First token sampled.

`R1.kv_committed_len = 576`. R1 → `RUNNING`.

**R1 Decode (10 steps):**

`decode_step([R1])` runs 10 times. `R1.slot_indices` grows as needed. `R1.output_ids` accumulates 10 tokens.

**R1 Finishes (EOS at step 10):**

`cache_finished_req(R1, rtp, kv_pool)`:
- `token_ids = R1.input_ids + R1.output_ids` = 576 + 10 = 586 tokens.
- `aligned_len = (586//16)*16 = 36*16 = 576`. `aligned_page_count = 36`.
- `n_overlap = insert(token_ids, R1.slot_indices)`.
  - Tree is empty; no existing edges. New leaf created with `key = tuple(token_ids[:576])` and `value = [a0..a35]`. Returns `n_overlap = 0`.
- `n_overlap (0) <= n_prefix_pages (0)`: no overlap pages to free.
- `aligned_page_count (36) < len(R1.slot_indices) (say 37 with decode)`: free the 1 tail page.
- `dec_lock_ref(root)`: root is never locked; no-op.
- `rtp.free(row1)`.

After this call: The tree has one leaf node with 576 tokens and 36 pages (`a0..a35`). The tail decode page is freed. `rtp` row1 is free.

---

## R2: Second Request (Tree Has System Prompt)

**match_prefix(R2.input_ids):**

`max_match = (576-1)//16*16 = 560`. Key = `tuple(R2.input_ids[:560])`.

`_match_helper(root, key)`:
- `child_key = tuple(R2.input_ids[:16])`. This is the first page of the system prompt — identical to R1's first page.
- `root.children[child_key]` = the leaf node from R1's insert. Found.
- `_count_match_pages(leaf.key, key)`: compares page by page. The system prompt's 512 tokens (32 pages) match perfectly. The user1 query's 4 pages might or might not match (they probably don't for a different user query). Say 32 pages match: `n_match = 32`.
- Since `n_match < len(leaf.key) // P` (32 < 36): partial edge match. Return `(leaf.value[:32], leaf)`.

`page_indices = [a0..a31]`, `matched_len = 32 * 16 = 512`, `last_node = leaf`.

`_apply_prefix_match`:
- `R2.prefix_page_indices = [a0..a31]`.
- `R2.kv_committed_len = 512`.
- `R2.last_node = leaf`.
- `inc_lock_ref(leaf)`: walk from leaf to root: `leaf.lock_ref += 1`.
- `R2.fill_ids = R2.input_ids[512:]` (64 tokens of user2 query).

**prefill_batch([R2]):**

Step 2 (prefix injection):
- `n_pfx = 32`. Write `[a0..a31]` into `req_to_token[row2, 0:32]`.

`compute_write_info(kv_pool, rtp, slot_indices=[a0..a31], req_pool_idx=row2, kv_committed_len=512, n_fill=64)`:
- `n_leftover = 512 % 16 = 0`. No partial page.
- `remaining = 64`. Allocate `ceil(64/16) = 4` new pages: `[b0..b3]`.
- `R2.slot_indices = [a0..a31, b0..b3]`. `req_to_token[row2, 32:36] = [b0..b3]`.

Token IDs = R2's 64-token user query. Position IDs `[512..575]`. `qo_indptr = [0, 64]`. `kv_indptr = [0, 36]` (32 prefix + 4 new). `kv_last_page_lens = [(512+64)%16 = 576%16 = 0 → 16]`.

Triton kernel reads `req_to_token[row2, 0:36]` = `[a0..a31, b0..b3]`.

The extend kernel processes 64 query tokens, attends over 36 pages (576 KV positions). Pages `a0..a31` hold the system prompt K/V computed by R1. Pages `b0..b3` hold the user2 query K/V computed now. No recomputation of the 512-token system prompt.

First output token sampled. R2 → `RUNNING`.

**R2 Finishes:**

`cache_finished_req(R2, ...)`:
- `insert(R2.input_ids + R2.output_ids, R2.slot_indices)`.
  - The tree already has R1's entry covering `token_ids[:576]`. R2's sequence shares the 512-token system prompt (32 pages). The 4 pages of user2 query are new — they become a new branch off the system-prompt node (after `_split_node` at the 32-page mark, creating a shared prefix node for the system prompt).
- `dec_lock_ref(leaf)`: `leaf.lock_ref -= 1` (back to 0). The old leaf is now evictable.

After R2 finishes: the tree has a shared system-prompt prefix node (32 pages) with two children — one for user1's query and one for user2's query. Future requests with either conversation history can match up to 512 tokens (system prompt) or 576 tokens (full shared conversation).

---

## What the Trace Shows

R2 computed 64 tokens of forward pass instead of 576. The 512-token system prompt's K/V — 32 pages × 28 layers × 2 (K and V) of `[16, 8, 128]` float16 tensors — was read directly from the pool without recomputation. The `req_to_token_pool` injection placed the cached page indices in the correct row columns before the Triton kernel ran. The extend kernel attended R2's suffix queries over the full 36-page history seamlessly.

Section 08 discusses what prefix caching leaves on the table and how Layer 13's GPTQ quantization addresses a different bottleneck.
