# SGLang Multi-LoRA Implementation

> Based on: **S-LoRA** (OSDI 2024) and **Punica** (MLSys 2024)  
> Source: `python/sglang/srt/lora/`

---

## 1. What is LoRA and Why Multi-LoRA?

LoRA (Low-Rank Adaptation) fine-tunes a model by adding a small rank-decomposed delta to each weight matrix:

```
output = x @ W.T  +  (x @ A.T) @ B.T * scaling

where:
  W:       [out_dim, in_dim]   — frozen base model weight
  A:       [rank, in_dim]      — LoRA "shrink" matrix (adapter-specific)
  B:       [out_dim, rank]     — LoRA "expand" matrix (adapter-specific)
  scaling: lora_alpha / r      — scalar, controls delta magnitude
```

At rank=16 on a 7B model, each adapter adds ~0.1% extra parameters (~10–50 MB). This makes it practical to serve many fine-tuned variants of the same base model on a single GPU — one base model in VRAM, many adapters swapped in and out.

**The challenge**: serving multiple adapters in a single batched forward pass where different requests use different adapters, without launching separate kernel calls per adapter.

---

## 2. File Map

```
python/sglang/srt/lora/
├── lora_config.py          — reads adapter_config.json (rank, alpha, target_modules)
├── lora.py                 — LoRAAdapter: loads weights to CPU, normalizes stacking
├── mem_pool.py             — LoRAMemoryPool: pre-allocated GPU A/B buffers, LRU eviction
├── lora_manager.py         — LoRAManager: orchestrates load/evict/prepare per batch
├── lora_registry.py        — LoRARegistry: tokenizer-manager-side name→ID mapping
├── layers.py               — LoRA-wrapped layer classes (QKV, MLP, embedding, lm_head)
├── eviction_policy.py      — LRU and FIFO eviction policies
├── utils.py                — LoRABatchInfo dataclass, helper functions
├── backend/
│   ├── base_backend.py     — BaseLoRABackend abstract interface
│   ├── triton_backend.py   — TritonLoRABackend: SGEMM kernels + CUDA graph support
│   ├── chunked_backend.py  — ChunkedSGMVBackend (csgmv, default)
│   └── torch_backend.py    — TorchLoRABackend: pure PyTorch fallback
└── triton_ops/
    ├── sgemm_lora_a.py     — Triton segmented GEMM for LoRA A (shrink)
    ├── sgemm_lora_b.py     — Triton segmented GEMM for LoRA B (expand + fused add)
    ├── qkv_lora_b.py       — specialized B kernel for fused QKV
    └── gate_up_lora_b.py   — specialized B kernel for fused gate/up
```

---

## 3. Configuration and CLI Flags

Defined in `server_args.py`:

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-lora` | `False` | Enable LoRA support (auto-set if `--lora-paths` given) |
| `--lora-paths` | `None` | Adapters to pre-load: `<PATH>`, `<NAME>=<PATH>`, or JSON |
| `--max-lora-rank` | auto | Max rank across all adapters; determines buffer sizes |
| `--lora-target-modules` | auto | Which linear layers get LoRA buffers (or `all`) |
| `--max-loras-per-batch` | `8` | Number of adapter slots in the GPU memory pool |
| `--max-loaded-loras` | `None` | Max adapters kept in CPU memory |
| `--lora-backend` | `csgmv` | Kernel backend: `triton`, `csgmv`, `torch_native`, `ascend` |
| `--lora-eviction-policy` | `lru` | Pool eviction policy: `lru` or `fifo` |
| `--enable-lora-overlap-loading` | `False` | Async CPU→GPU weight copy (overlap with compute) |

**Example launch:**
```bash
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --lora-paths finance=./finance-lora code=./code-lora \
  --max-loras-per-batch 4 \
  --max-lora-rank 64 \
  --lora-backend triton
```

---

## 4. Startup: LoRAManager Initialization

Entry point: `model_runner.py:init_lora_manager()` (line ~1818)

```python
# model_runner.py
if server_args.enable_lora:
    self.init_lora_manager()
    if not server_args.disable_cuda_graph:
        self._init_lora_cuda_graph_moe_buffers()  # Phase 1 of CUDA graph init

def init_lora_manager(self):
    self.lora_manager = LoRAManager(
        base_model=self.model,
        base_hf_config=self.model_config.hf_config,
        max_loras_per_batch=self.server_args.max_loras_per_batch,
        load_config=self.load_config,
        dtype=self.dtype,
        server_args=self.server_args,
        lora_backend=self.server_args.lora_backend,
        tp_size=self.tp_size,
        tp_rank=self.tp_rank,
        max_lora_rank=self.server_args.max_lora_rank,
        target_modules=self.server_args.lora_target_modules,
        lora_paths=self.server_args.lora_paths,
    )
```

`LoRAManager.__init__` calls `init_state()` which runs 5 phases in order:

```
init_lora_adapters()   — load configs + weights for pre-specified adapters
init_lora_shapes()     — infer target_modules and max_lora_rank
init_lora_modules()    — replace base Linear layers with LoRA-wrapped versions
init_memory_pool()     — allocate GPU A/B buffers
update_lora_info()     — point each LoRA layer at the right buffer slice
```

---

## 5. Adapter Loading: LoRAConfig and LoRAAdapter

### 5.1 LoRAConfig (`lora_config.py`)

Reads `adapter_config.json` from the checkpoint:

```python
class LoRAConfig:
    path: str
    r: int              # rank
    lora_alpha: float   # scaling numerator
    target_modules: List[str]  # e.g. ["q_proj", "v_proj", "gate_proj"]
```

### 5.2 LoRAAdapter (`lora.py`)

Loads the actual weight tensors from the checkpoint into **CPU memory**:

```python
class LoRAAdapter(nn.Module):
    uid: str
    config: LoRAConfig
    scaling: float = lora_alpha / r
    layers: List[LoRALayer]   # one per transformer layer
    embedding_layers: Dict    # embed_tokens weights if targeted
```

**Weight normalization** happens at load time (CPU, once):

```python
# Separate q/k/v → stacked qkv_proj
qkv_lora_A = torch.cat([q_lora_A, k_lora_A, v_lora_A], dim=0)
# shape: [3*rank, hidden_dim]

# Separate gate/up → stacked gate_up_proj
gate_up_lora_A = torch.cat([gate_lora_A, up_lora_A], dim=0)
# shape: [2*rank, hidden_dim]
```

This stacking means a single SGEMM call handles all 3 QKV projections simultaneously.

If `--enable-lora-overlap-loading` is set, weights are pinned in CPU memory (`pin_memory=True`) to enable async non-blocking CPU→GPU copies.

---

## 6. GPU Memory Pool (`mem_pool.py`)

### 6.1 Buffer Layout

`LoRAMemoryPool` pre-allocates fixed-size GPU tensors at startup:

```python
# Standard (non-MoE) modules — 3D buffers
A_buffer["qkv_proj"][layer_id]:   [max_loras_per_batch, 3*max_rank, hidden_dim]
B_buffer["qkv_proj"][layer_id]:   [max_loras_per_batch, q_dim+2*kv_dim, max_rank]

A_buffer["gate_up_proj"][layer_id]: [max_loras_per_batch, 2*max_rank, hidden_dim]
B_buffer["gate_up_proj"][layer_id]: [max_loras_per_batch, 2*inter_dim, max_rank]

A_buffer["o_proj"][layer_id]:     [max_loras_per_batch, max_rank, hidden_dim]
B_buffer["o_proj"][layer_id]:     [max_loras_per_batch, hidden_dim, max_rank]

# MoE modules — 4D buffers
A_buffer["gate_up_proj_moe"][layer_id]: [max_loras, num_experts, 2*max_rank, hidden]
B_buffer["down_proj_moe"][layer_id]:    [max_loras, num_experts, hidden, max_rank]
```

All buffers are allocated once at startup. Adapters are **slotted in** by copying weights into a specific row index (`buffer_id`).

### 6.2 Slot Management

```python
uid_to_buffer_id: Dict[str, int]   # adapter_id → slot index
buffer_id_to_uid: List[str]        # slot index → adapter_id (EMPTY_SLOT if free)
```

### 6.3 Eviction (`eviction_policy.py`)

When a new adapter is needed and all slots are occupied:

1. Find candidates: slots not needed by the current batch and not pinned
2. Apply eviction policy (LRU or FIFO) to select victim
3. Zero the victim's slot buffers (prevents contamination)
4. Copy new adapter weights into the freed slot

```python
# LRU policy
class LRUEvictionPolicy:
    access_order: OrderedDict  # uid → last_access_time

    def mark_used(self, uid):
        self.access_order.move_to_end(uid)  # most recent = end

    def select_victim(self, candidates):
        # iterate from oldest (front) to find first candidate
        for uid in self.access_order:
            if uid in candidates:
                return uid
```

### 6.4 Loading Weights into a Slot

`load_lora_weight_to_buffer(uid, buffer_id, lora_adapter, ...)`:

1. For each transformer layer, extract A and B weights from `lora_adapter.layers[layer_id].weights`
2. Apply TP sharding via `module.slice_lora_a_weights(A, tp_rank)` / `slice_lora_b_weights(B, tp_rank)`
3. Copy into `A_buffer[module][layer_id][buffer_id, :rank, :]` via `.copy_(weight, non_blocking=True)`

The `non_blocking=True` enables async H2D transfer when weights are pinned.

---

## 7. Layer Wrapping (`layers.py`)

`LoRAManager.init_lora_modules()` replaces base Linear layers with LoRA-wrapped versions using `replace_submodule()`:

| Base Layer | LoRA Wrapper |
|-----------|-------------|
| `QKVParallelLinear` | `QKVParallelLinearWithLoRA` |
| `MergedColumnParallelLinear` | `MergedColumnParallelLinearWithLoRA` |
| `ColumnParallelLinear` | `ColumnParallelLinearWithLoRA` |
| `RowParallelLinear` | `RowParallelLinearWithLoRA` |
| `ReplicatedLinear` | `ReplicatedLinearWithLoRA` |
| `VocabParallelEmbedding` | `VocabParallelEmbeddingWithLoRA` |
| `ParallelLMHead` | `ParallelLMHeadWithLoRA` |
| `FusedMoE` | `FusedMoEWithLoRA` |

Each wrapper's `forward()` follows the same pattern:

```python
def forward(self, x):
    base_output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
    if self.set_lora:
        base_output = self.apply_lora(base_output, x)
    return base_output

def apply_lora(self, base_output, x):
    lora_a_out = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
    lora_output = self.lora_backend.run_lora_b_sgemm(
        x=lora_a_out, weights=self.B_buffer, base_output=base_output
    )
    return lora_output
```

`set_lora_info(A_buffer, B_buffer)` is called by `LoRAManager.update_lora_info()` to point each layer at the right GPU buffer slice.

### TP Sharding Rules

| Module | LoRA A sharding | LoRA B sharding |
|--------|----------------|----------------|
| `QKVParallelLinear` | Unsharded (full input) | Sliced per Q/K/V shard |
| `MergedColumnParallelLinear` | Unsharded | Sliced along output dim |
| `RowParallelLinear` | Sliced along input dim | Unsharded (full output) |
| `ParallelLMHead` | Unsharded | Sliced along vocab dim |
| `VocabParallelEmbedding` | Unsharded | Unsharded |

---

## 8. Per-Batch Preparation

### 8.1 Request Flow

Each request carries a `lora_id` (or `None` for base model). The tokenizer manager resolves the human-readable name to a UUID via `LoRARegistry.acquire()`.

### 8.2 `prepare_lora_batch` (called every forward pass)

```python
# model_runner.py — called before every forward pass
if lora_ids is not None:
    self.lora_manager.prepare_lora_batch(forward_batch)
```

Inside `LoRAManager.prepare_lora_batch(forward_batch)`:

```python
# 1. Build weight_indices: which pool slot does each token use?
weight_indices = [0] * batch_size
lora_ranks = [0] * max_loras_per_batch
scalings = [0.0] * max_loras_per_batch

for i, uid in enumerate(forward_batch.lora_ids):
    weight_indices[i] = memory_pool.get_buffer_id(uid)
    if uid is not None:
        lora = self.loras[uid]
        lora_ranks[weight_indices[i]] = lora.config.r
        scalings[weight_indices[i]] = lora.scaling

# 2. Delegate to backend to build LoRABatchInfo
self.lora_backend.prepare_lora_batch(
    forward_batch, weight_indices, lora_ranks, scalings, use_cuda_graph
)
```

### 8.3 `LoRABatchInfo` (`utils.py`)

The central metadata struct passed to every Triton kernel:

```python
@dataclass
class LoRABatchInfo:
    bs: int                        # batch size
    seg_lens: torch.Tensor         # [B] tokens per request
    seg_indptr: torch.Tensor       # [B+1] cumulative token offsets
    weight_indices: torch.Tensor   # [B] pool slot per request
    lora_ranks: torch.Tensor       # [max_loras] rank per slot
    scalings: torch.Tensor         # [max_loras] scaling per slot
    max_len: int                   # max sequence length in batch
    permutation: Optional[Tensor]  # [total_tokens] sorted-by-adapter order
    use_cuda_graph: bool
```

### 8.4 Decode Optimization: Sort by Adapter

For decode (1 token per request), the Triton backend sorts tokens by adapter before running SGEMM:

```python
# triton_backend.py: compute_sgemm_routing()
perm = torch.argsort(weight_indices, stable=True)  # sort tokens by adapter
sorted_wi = weight_indices[perm]

# Build merged segments: all tokens for adapter 0, then adapter 1, etc.
seg_starts = torch.searchsorted(sorted_wi, adapter_ids)
seg_ends   = torch.searchsorted(sorted_wi, adapter_ids, right=True)
seg_lens   = seg_ends - seg_starts
```

This maximizes memory coalescing: the kernel reads contiguous rows of A/B for each adapter.

---

## 9. Triton SGEMM Kernels

### 9.1 Segmented GEMM Concept (Punica)

Instead of one matmul per adapter, a single kernel handles all adapters simultaneously. The batch is divided into **segments** — contiguous groups of tokens sharing the same adapter.

```
Batch (decode, sorted by adapter):
  tokens [0,1,2]   → adapter slot 0  (finance-lora)
  tokens [3,4]     → adapter slot 1  (code-lora)
  tokens [5,6,7,8] → adapter slot 2  (base model, rank=0 → no-op)

seg_lens    = [3, 2, 4]
seg_indptr  = [0, 3, 5, 9]
weight_indices = [0, 1, 2]

Grid: (ceil(max_len/BLOCK_S) * ceil(R/BLOCK_R), num_segments)
      One threadblock per (tile, segment) pair — all run in parallel.
```

### 9.2 `sgemm_lora_a_fwd` (`triton_ops/sgemm_lora_a.py`)

**Input**: `x [s, hidden]`, `weights [max_loras, stack_num*rank, hidden]`  
**Output**: `[s, stack_num*rank]`

```python
# Kernel: _sgemm_lora_a_kernel
batch_id = tl.program_id(axis=1)   # which segment
w_index  = tl.load(weight_indices + batch_id)
rank     = tl.load(lora_ranks + w_index)
if rank == 0: return               # base model → no-op

seg_start = tl.load(seg_indptr + batch_id)
seg_len   = tl.load(seg_lens + batch_id)

# Tiled matmul: x[seg_start:seg_start+seg_len] @ weights[w_index].T
# BLOCK_S=16, BLOCK_K=256, BLOCK_R=16
# Accumulates in fp32, stores in input dtype
```

`stack_num=3` for QKV (handles q/k/v in one call), `stack_num=2` for gate/up.

### 9.3 `sgemm_lora_b_fwd` (`triton_ops/sgemm_lora_b.py`)

**Input**: `x [s, rank]` (output of A), `weights [max_loras, out_dim, rank]`, `base_output [s, out_dim]`  
**Output**: `base_output` modified in-place: `base_output += x @ weights[w_index].T * scaling`

The scaling is **fused inside the kernel** — no separate multiply needed:

```python
# Inside kernel:
partial_sum *= scaling   # fused scaling
partial_sum += tl.load(output_ptr, ...)   # fused add to base output
tl.store(output_ptr, partial_sum, ...)
```

### 9.4 Specialized Kernels

**`qkv_lora_b_fwd`** (`triton_ops/qkv_lora_b.py`): handles the split output offsets for Q, K, V projections which have different output dimensions.

**`gate_up_lora_b_fwd`** (`triton_ops/gate_up_lora_b.py`): handles the split output for gate and up projections.

Both use `output_offset` tensors to scatter results into the correct positions of the fused output tensor.

---

## 10. Complete Forward Pass Walkthrough

For a decode batch with 4 requests using 2 different adapters:

```
Requests: [req0(finance), req1(code), req2(finance), req3(base)]
lora_ids: ["finance-lora", "code-lora", "finance-lora", None]

Step 1: fetch_new_loras({"finance-lora", "code-lora", None})
  → ensure all 3 are in GPU pool slots
  → finance-lora → slot 0, code-lora → slot 1, None → slot 2 (zeroed)

Step 2: prepare_lora_batch()
  weight_indices = [0, 1, 0, 2]   ← slot per request
  lora_ranks     = [16, 32, 0, ...]  ← rank per slot (slot 2 = 0 = no-op)
  scalings       = [1.0, 2.0, 0.0, ...]

Step 3: compute_sgemm_routing() (decode path)
  perm = argsort([0,1,0,2]) = [0,2,1,3]  ← sort by adapter
  sorted segments: slot0=[tok0,tok2], slot1=[tok1], slot2=[tok3]
  seg_lens=[2,1,1], seg_indptr=[0,2,3,4]

Step 4: For each transformer layer:
  a. base QKV projection: output = x @ W_qkv.T
  b. sgemm_lora_a(x, A_buffer["qkv_proj"][layer], batch_info)
     → [4, 3*rank] intermediate (sorted order)
  c. qkv_lora_b_fwd(intermediate, B_buffer["qkv_proj"][layer], base_output)
     → base_output += delta * scaling  (in-place, fused)
  d. Same for gate_up_proj, down_proj, o_proj

Step 5: Sample logits, return tokens
```

---

## 11. Dynamic Adapter Loading/Unloading

SGLang supports loading and unloading adapters at runtime via HTTP endpoints.

### Load

```
POST /lora/load
{"lora_name": "new-adapter", "lora_path": "/path/to/adapter"}
```

Flow:
1. `LoRARegistry.register(lora_ref)` — tokenizer manager registers name→ID
2. `model_runner.load_lora_adapter(lora_ref)` → `LoRAManager.load_lora_adapter()`
3. `LoRAConfig` reads `adapter_config.json`
4. `LoRAAdapter.initialize_weights()` loads weights to CPU
5. Adapter is available for requests immediately

### Unload

```
POST /lora/unload
{"lora_name": "new-adapter"}
```

Flow:
1. `LoRARegistry.unregister(lora_name)` — removes from registry
2. `LoRARegistry.wait_for_unload(lora_id)` — waits for in-flight requests to finish
3. `model_runner.unload_lora_adapter(lora_ref)` — removes from `lora_manager.loras` dict
4. GPU pool slot is freed lazily (on next eviction cycle)

---

## 12. CUDA Graph Support

LoRA is CUDA-graph-aware. Two initialization phases:

**Phase 1** (`_init_lora_cuda_graph_moe_buffers`): Pre-allocates MoE intermediate buffers before memory profiling, so they're accounted for in the KV pool budget.

**Phase 2** (`init_cuda_graph_batch_info`): Called during `CudaGraphRunner.__init__()`. Pre-allocates static `LoRABatchInfo` tensors:

```python
# triton_backend.py
self.cuda_graph_batch_info = LoRABatchInfo(
    bs=max_bs_in_cuda_graph,
    use_cuda_graph=True,
    seg_lens=torch.full((max_bs,), num_tokens_per_bs, dtype=torch.int32),
    seg_indptr=torch.zeros(max_bs + 1, dtype=torch.int32),
    weight_indices=torch.zeros(max_bs, dtype=torch.int32),
    lora_ranks=torch.zeros(max_loras_per_batch, dtype=torch.int32),
    scalings=torch.zeros(max_loras_per_batch, dtype=torch.float),
    ...
)
```

At replay time, `weight_indices`, `lora_ranks`, and `scalings` are updated in-place via `.copy_()` before `graph.replay()`. The tensor objects (and GPU memory addresses) never change.

**Note**: LoRA forces `disable_piecewise_cuda_graph = True` (from `server_args.py`), meaning the full decode graph is captured as a single graph rather than piecewise.

---

## 13. LoRARegistry: Tokenizer Manager Side

`LoRARegistry` lives in the tokenizer manager process and acts as the source of truth for name→ID mapping:

```python
class LoRARegistry:
    _registry: OrderedDict[str, LoRARef]   # name → LoRARef (LRU order)
    _counters: Dict[str, ConcurrentCounter] # lora_id → in-flight request count

    async def acquire(lora_name) → lora_id:
        # Lookup name → ID, increment counter, move to end (LRU)

    async def release(lora_id):
        # Decrement counter

    async def wait_for_unload(lora_id):
        # Block until counter reaches 0 (safe to unload)
```

The `ConcurrentCounter` ensures adapters are not evicted while requests are in-flight.

---

## 14. Key Design Decisions

### Why pre-allocate GPU buffers at startup?

Avoids dynamic GPU memory allocation during inference (which would cause CUDA graph invalidation and fragmentation). The pool size is fixed: `max_loras_per_batch × max_lora_rank × hidden_dim × num_layers × 2 (A+B) × dtype_bytes`.

### Why sort tokens by adapter for decode?

At decode time (1 token per request), tokens from different adapters are interleaved. Sorting groups them into contiguous segments, enabling coalesced memory access in the SGEMM kernel. The permutation is applied to the intermediate tensor, not the input — so the base model output is unaffected.

### Why fuse scaling into the B kernel?

Avoids a separate elementwise multiply pass. The B kernel reads `scalings[w_index]` and multiplies before the fused add to `base_output`. One kernel, one memory pass.

### Why stack q/k/v into qkv_proj?

A single SGEMM call for all 3 projections is more efficient than 3 separate calls. The stacking happens at CPU load time (once per adapter), not at inference time.

### Why `set_lora` flag on each layer?

When a batch has no LoRA requests (all `lora_id=None`), `has_active_lora=False` and the LoRA path is skipped entirely — zero overhead for base-model-only batches.

---

## 15. Code Anchors

| Concept | File | Line/Symbol |
|---------|------|-------------|
| CLI flags | `srt/server_args.py` | ~4897 |
| `LoRAManager` init | `srt/model_executor/model_runner.py` | `init_lora_manager()` ~1818 |
| `LoRAConfig` | `srt/lora/lora_config.py` | `LoRAConfig.__init__` |
| `LoRAAdapter` weight loading | `srt/lora/lora.py` | `initialize_weights()` |
| QKV stacking normalization | `srt/lora/lora.py` | `normalize_qkv_proj()` |
| gate/up stacking | `srt/lora/lora.py` | `normalize_gate_up_proj()` |
| GPU buffer allocation | `srt/lora/mem_pool.py` | `LoRAMemoryPool.init_buffers()` |
| Slot eviction | `srt/lora/mem_pool.py` | `prepare_lora_batch()` |
| Weight copy to slot | `srt/lora/mem_pool.py` | `load_lora_weight_to_buffer()` |
| Layer replacement | `srt/lora/lora_manager.py` | `init_lora_modules()` |
| Per-batch weight_indices | `srt/lora/lora_manager.py` | `prepare_lora_batch()` |
| Sort by adapter (decode) | `srt/lora/backend/triton_backend.py` | `compute_sgemm_routing()` |
| SGEMM LoRA A kernel | `srt/lora/triton_ops/sgemm_lora_a.py` | `_sgemm_lora_a_kernel` |
| SGEMM LoRA B kernel | `srt/lora/triton_ops/sgemm_lora_b.py` | `_sgemm_lora_b_kernel` |
| QKV forward with LoRA | `srt/lora/layers.py` | `QKVParallelLinearWithLoRA.apply_lora()` |
| LRU eviction policy | `srt/lora/eviction_policy.py` | `LRUEvictionPolicy` |
| LoRARegistry (tokenizer side) | `srt/lora/lora_registry.py` | `LoRARegistry` |
| CUDA graph batch info | `srt/lora/backend/triton_backend.py` | `init_cuda_graph_batch_info()` |
| Dynamic load endpoint | `srt/model_executor/model_runner.py` | `load_lora_adapter()` ~1870 |
