# Layer 12 — Lesson Outline

## What This Lesson Covers

Layer 11 allowed long prompts to be split into chunks so that decode requests could run between prefill chunks. But every request still computed its own K/V from scratch: two requests sharing a 1024-token system prompt each spent a full 1024-position extend round writing identical K/V data into independent pool pages. The wasted GPU memory and bandwidth grew linearly with the number of requests sharing that prefix.

Layer 12 adds a `RadixCache` — a compressed radix tree keyed on page-aligned token sequences — that maps shared prompt prefixes to their already-computed KV page indices. When a new request arrives and its prompt shares a page-aligned prefix with a cached entry, `match_prefix` returns those page indices along with the number of tokens matched. `kv_committed_len` is set to `matched_len` before `prefill_batch` is called, so the extend kernel only processes the unique suffix tokens. The pool pages for the cached prefix are owned by the tree and write-protected with a `lock_ref` reference count that prevents eviction while any active request is using them.

The change touches three files: `radix_cache.py` (new — the tree, `match_prefix`, `insert`, `evict`, `cache_finished_req`, and `lock_ref` reference counting), `scheduler.py` (which adds `_apply_prefix_match` to `PrefillAdder` and calls `cache_finished_req` on request completion), and `model_runner.py` (which injects `prefix_page_indices` into `req_to_token` before `compute_write_info`, and calls `radix_cache.evict` when the pool is low). `decode_step`, `forward_batch.py`, and all model files are unchanged.

The sections follow the cache lifecycle from top to bottom: why duplicate KV writes are expensive, the radix tree structure, prefix matching and lock protection, `insert` and `_split_node`, `evict` and LRU ordering, integration with `prefill_batch`, request completion and `cache_finished_req`, the full call trace, and what static caching cannot do that Layer 13's quantization addresses.

---

## Sections

### 01 — From Repeated Writes to Shared Pages (`01_from_repeated_to_shared.md`)
- Layer 11's waste: two 1024-token prompts sharing a 512-token system prefix each call `compute_write_info` independently and write 512 identical KV positions into separate pool pages — the second request's prefix write is pure redundancy
- Layer 12's change: `PrefillAdder._apply_prefix_match` calls `radix_cache.match_prefix(req.input_ids)` before setting `fill_ids`; for a 512-token cache hit, `req.kv_committed_len = 512` and `req.fill_ids = req.input_ids[512:]` — only the unique suffix is processed
- What `prefix_page_indices` is: the list of pool page indices owned by the tree for the matched prefix, injected into `req_to_token_pool` by `prefill_batch` before the Triton `kv_indices` kernel runs
- What stays the same: `ExtendKVCtx`, `compute_write_info`, `decode_step`, `ForwardBatch`, and all model files — the extend kernel sees only a different `kv_committed_len`; everything downstream is unchanged

### 02 — The Radix Tree Structure (`02_the_radix_tree.md`)
- `TreeNode` fields: `children: Dict[Tuple[int,...], TreeNode]` keyed on the first-page token tuple; `parent`; `key: Tuple[int,...]` (token sequence for this edge, length always a multiple of `page_size`); `value: List[int]` (page indices for this edge's tokens); `lock_ref: int`; `last_access_time: float`
- Compressed edges: each edge represents one or more full pages of tokens — not one token at a time; this gives `O(P)` space per cached page instead of `O(1)` per token
- `page_size` alignment invariant: `match_prefix` never matches beyond `floor((len-1)/P)*P` tokens — at least one non-cached token must remain so the forward pass has a logit to sample from
- `_split_node(child, n_pages)`: when a new request shares only a partial match with an existing edge, the edge is split into a prefix node (up to the match point) and the remaining suffix node; `lock_ref` and `last_access_time` are copied to the new prefix node
- Root node as sentinel: always present, never evicted, never holds a `value`; all real prefix nodes are descendants of root

### 03 — match_prefix and Lock Reference Counting (`03_match_prefix_lock_ref.md`)
- `match_prefix(token_ids)`: traverses from root following `_child_key` lookups; at each node, `_count_match_pages` compares the edge key page-by-page; depth-first walk, updating `last_access_time` at each visited node
- Return value: `(page_indices, matched_len, last_node)` — `page_indices` is the flat list of pool page indices for the matched prefix; `matched_len = len(page_indices) * page_size`; `last_node` is the deepest node reached
- Why `inc_lock_ref(last_node)` is called immediately after `match_prefix`: increments `lock_ref` on every ancestor from `last_node` to root; prevents any of those nodes from being evicted while the request is active
- `dec_lock_ref(req.last_node)` called in `cache_finished_req`: walks the same ancestor chain and decrements; nodes with `lock_ref == 0` and no children become eviction candidates immediately
- The evictable predicate: `lock_ref == 0 and not self.children` — only leaf nodes with no active requests can be evicted

### 04 — insert and _split_node (`04_insert_split.md`)
- `insert(token_ids, page_indices)` is called by `cache_finished_req` after a request finishes; inserts the page-aligned prefix `token_ids[:aligned_len]` with pages `page_indices[:aligned_len // P]` into the tree
- Return value `n_overlap_pages`: the number of pages the tree already contained for this prefix before the insert; `cache_finished_req` calls `kv_pool.free(all_pages[n_prefix_pages:n_overlap])` to return the redundantly-computed pages
- Why overlap pages happen: two requests with identical prompts can compute the same KV pages independently and both try to insert them; the second insert finds the pages already cached and the returned overlap count tells it how many to free
- `_insert_helper` recursive logic: follow existing edges as far as they match; if an edge matches exactly, continue deeper; if an edge is a prefix of the new key, split the edge via `_split_node`, then continue; if no edge matches, create a new leaf
- `_split_node(child, n_pages)`: creates a new node containing the first `n_pages` pages of `child`'s edge; moves the remaining suffix back to `child`; attaches the new node between `parent` and `child`; preserves `lock_ref` and `last_access_time`

### 05 — evict and LRU Ordering (`05_evict_lru.md`)
- `evict(n_pages_needed)` is called by `prefill_batch` when `kv_pool.free_pages_count() < pages_needed`; frees LRU-eligible leaves until the pool has enough free pages
- Heap ordering: `TreeNode.__lt__` compares by `last_access_time`; `heapq.heapify(leaves)` gives a min-heap with the oldest node first
- Eviction loop: pop the oldest evictable leaf, call `kv_pool.free(node.value)`, delete the leaf via `_delete_leaf`; if the parent becomes an evictable leaf after deletion, push it onto the heap and continue
- Why `not node.evictable` is checked after popping: another eviction round could have decremented a sibling's reference or modified the tree between the heapify and the pop; stale heap entries must be discarded
- `_delete_leaf(node)`: removes `node` from `parent.children`; if parent now has no children and `lock_ref == 0`, it becomes evictable (the loop will push it to the heap)

### 06 — Integration with prefill_batch (`06_integration_prefill_batch.md`)
- `PrefillAdder._apply_prefix_match` sets `req.prefix_page_indices`, `req.prefix_len`, `req.last_node`, and `req.kv_committed_len` before setting `fill_ids`; guards against double-matching on continuation chunks via `req.last_node is not None`
- In `model_runner.prefill_batch`: prefix pages are written into `req_to_token_pool` at positions `0 : n_prefix_pages` before `compute_write_info` runs, so `slot_indices` and `req_to_token` are consistent before the Triton kernel builds `kv_indices`
- Eviction call site: before `compute_write_info`, check `kv_pool.free_pages_count()` against total pages needed; if low, `radix_cache.evict(pages_needed - available)` frees the oldest unlocked pages
- Position IDs: each request's tokens start at `req.kv_committed_len`, not at zero; this is identical to Layer 11 — the prefix cache hit simply increases `kv_committed_len` from zero to `prefix_len`
- `cache_finished_req` is called by the scheduler instead of the Layer 11 direct `kv_pool.free` + `req_to_token_pool.free`; it handles `insert`, overlap-page freeing, lock release, and pool/table cleanup atomically

### 07 — The Full Loop (`07_the_full_loop.md`)
- End-to-end trace: first request with a 512-token system prefix, then a second request with the same prefix
- Step 1 — First request: `match_prefix` returns `[]`, `matched_len=0`; `fill_ids = input_ids`; `prefill_batch` computes K/V for all tokens; on completion `cache_finished_req` inserts the page-aligned prefix into the tree; pool pages now owned by tree
- Step 2 — Second request: `match_prefix` returns the 512-token prefix pages, `matched_len=512`; `inc_lock_ref` protects the nodes; `req.kv_committed_len = 512`; `fill_ids = input_ids[512:]`; extend kernel processes only the suffix; prefix K/V is read from pool via `kv_indices` — no recomputation
- Step 3 — Eviction pressure: if the pool is close to full when a third request arrives, `evict` frees oldest unlocked nodes; locked nodes (belonging to active decode requests) are skipped
- Step 4 — Completion: `cache_finished_req` inserts new tokens, computes overlap (none if unique suffix), releases lock, frees req_pool_idx

### 08 — What Comes Next (`08_whats_next.md`)
- The remaining cost: model weights are stored in bfloat16; projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, FFN linear layers) dominate GPU memory at 2 bytes per parameter
- Layer 13 (GPTQ quantization): replaces `nn.Linear` with `GPTQLinear` — a 4-bit quantized weight store using `qweight` (int32-packed), `scales` (fp16), and `qzeros` (int32-packed); the fused `gptq_gemm` kernel dequantizes on-the-fly at inference time, reducing weight memory by ~4× without changing the K/V cache, scheduler, or attention kernel
- What changes in Layer 13: `model_gptq/gptq_linear.py` (new), `model_gptq/attention.py` / `decoder_layer.py` / `qwen3.py` (using `GPTQLinear`); `model_runner.py` gains a `use_gptq` flag; all caching and scheduling logic is unchanged

---

## Supporting Files

- `summary.md` — blog-post-style summary covering all sections
- `sglang_reference.md` — maps Layer 12 concepts to SGLang source: `RadixCache` → `RadixCache` in `srt/managers/tree_cache.py`; `lock_ref` → `lock_ref` on `TreeNode`; `cache_finished_req` → `cache_req` in `TreeCache`; `match_prefix` → `match_prefix_helper`

---

## Key Code Anchors

| Concept | Location |
|---|---|
| `TreeNode` dataclass | `radix_cache.py` line 73: `__slots__ = ("children", "parent", "key", "value", "lock_ref", "last_access_time")` |
| `evictable` predicate | `radix_cache.py` line 85: `return self.lock_ref == 0 and not self.children` |
| `RadixCache` class | `radix_cache.py` line 97: `class RadixCache:` |
| `match_prefix` entry | `radix_cache.py` line 107: `def match_prefix(` |
| Max-match cap (last token guard) | `radix_cache.py` line 124: `max_match = ((len(token_ids) - 1) // P) * P` |
| `insert` entry | `radix_cache.py` line 133: `def insert(` |
| `inc_lock_ref` | `radix_cache.py` line 153: `def inc_lock_ref(self, node: TreeNode) -> None:` |
| `dec_lock_ref` | `radix_cache.py` line 159: `def dec_lock_ref(self, node: TreeNode) -> None:` |
| `evict` entry | `radix_cache.py` line 166: `def evict(self, n_pages_needed: int) -> int:` |
| LRU heap build | `radix_cache.py` line 172: `heapq.heapify(leaves)` |
| `cache_finished_req` | `radix_cache.py` line 189: `def cache_finished_req(` |
| Overlap-page free | `radix_cache.py` line 215: `kv_pool.free(all_pages[n_prefix_pages:n_overlap])` |
| `dec_lock_ref` on finish | `radix_cache.py` line 223: `self.dec_lock_ref(req.last_node)` |
| `_split_node` | `radix_cache.py` line 249: `def _split_node(self, child: TreeNode, n_pages: int) -> TreeNode:` |
| `_apply_prefix_match` | `scheduler.py` line 119: `def _apply_prefix_match(self, req: Req) -> None:` |
| Prefix page injection | `model_runner.py` line 213: `n_pfx = len(req.prefix_page_indices)` |
| Eviction call site | `model_runner.py` line 225: `freed = self.radix_cache.evict(pages_needed - available)` |
| Position IDs from `kv_committed_len` | `model_runner.py` line 263: `pos_list.append(req.kv_committed_len + j)` |
| `kv_committed_len` update | `model_runner.py` line 343: `req.kv_committed_len += req.extend_input_len` |
