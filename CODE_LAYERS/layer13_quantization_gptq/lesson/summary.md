# Layer 13 — Summary

Layer 13 replaces every `nn.Linear` projection in the attention and FFN layers with `GPTQLinear` — a 4-bit quantized weight module that stores packed int4 weights and calls `gptq_gemm` for dequantize-fused matrix multiplication at inference time — reducing model weight memory from ~3.4 GB to ~0.85 GB for Qwen3-1.7B. The `RadixCache`, `KVPool`, `Scheduler`, `decode_step`, `FlashInfer` wrappers, and `ForwardBatch` are completely unchanged; only `model_runner.py` gains a `use_gptq` flag and the `model_gptq/` directory is added.

---

## From Layer 12 to Layer 13

In Layer 12, every projection matrix was stored in bfloat16:

```python
# Layer 12 — model/attention.py (standard nn.Linear projection)
self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=False)
```

For `hidden_size=2048`, `q_dim=2048`, `kv_dim=1024`: `q_proj` alone stores 2048 × 2048 × 2 = 8 MB. Across 28 layers and 7 projections per layer, this sums to roughly 3.4 GB — leaving little room for a large KV pool on a 16 GB GPU.

In Layer 13, the same four lines read:

```python
# Layer 13 — model_gptq/attention.py (GPTQLinear projection)
self.q_proj = GPTQLinear(config.hidden_size, q_dim, bits, group_size)
self.k_proj = GPTQLinear(config.hidden_size, kv_dim, bits, group_size)
self.v_proj = GPTQLinear(config.hidden_size, kv_dim, bits, group_size)
self.o_proj = GPTQLinear(q_dim, config.hidden_size, bits, group_size)
```

The `forward` signature of `Qwen3Attention` is unchanged — `q_proj(hidden_states)` calls `GPTQLinear.forward` and returns the same shape tensor. The rest of the attention and KV path sees no difference.

---

## The qweight Layout

`GPTQLinear` stores four buffers per layer:

```python
self.pack_factor = 32 // bits      # int4: 8 values per int32

self.register_buffer(
    "qweight",
    torch.empty(in_features // self.pack_factor, out_features, dtype=torch.int32),
)
self.register_buffer(
    "scales",
    torch.empty(in_features // group_size, out_features, dtype=torch.float16),
)
self.register_buffer(
    "qzeros",
    torch.empty(in_features // group_size, out_features // self.pack_factor, dtype=torch.int32),
)
seq = torch.arange(in_features, dtype=torch.int32) // group_size
self.register_buffer("_g_idx_seq", seq)
```

`qweight` packs 8 int4 values into each int32 element, so its shape is `[K // 8, N]` rather than the `[K, N]` of a bfloat16 weight. `register_buffer` makes these tensors part of the `state_dict` and moves them to GPU with the module, but excludes them from gradient computation. Crucially, `model.to(dtype)` applies only to floating-point tensors — `qweight`, `qzeros`, and `_g_idx_seq` (all int32) survive a `bfloat16` cast unchanged.

`scales` is registered as `torch.float16` explicitly. `gptq_gemm` reads scale values by interpreting their bits as IEEE 754 fp16 — if `scales` were cast to bfloat16, the bit patterns would be reinterpreted incorrectly and produce wrong dequantized weights. The loading code in `from_pretrained` skips `scales` when applying `model.to(bfloat16)`.

---

## Scales, Zeros, and Group Quantization

Group quantization divides each column's `K` input rows into groups of `group_size` rows, each with its own scale and zero-point. For `K=2048` and `group_size=128`, there are 16 groups per column. The dequantize formula applied by `gptq_gemm` is:

```python
# Conceptual; the actual operation is fused in the CUDA kernel
w_fp = (qweight_unpacked - qzeros) * scales
```

`qzeros` is packed the same way as `qweight` — 8 int4 values per int32 — so its shape is `[K // group_size, N // pack_factor]`. The zero-point centers each group's weight distribution before multiplying by the scale; for weights that are symmetric around zero, `qzeros` is all-zeros, but the field must still be present for the kernel interface.

`_g_idx_seq` is a pre-computed int32 tensor where element `i` equals `i // group_size` — a sequential group assignment. The `gptq_gemm` `use_shuffle=False` path uses this to map each input row to its scale group row at kernel time. Qwen3 GPTQ checkpoints use `desc_act=False`, meaning weights are not reordered by activation magnitude, so the sequential assignment is always correct.

---

## gptq_gemm and the use_shuffle=False Path

`GPTQLinear.forward` casts the input to fp16, calls `gptq_gemm`, and casts back:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    from sgl_kernel import gptq_gemm

    orig_dtype = x.dtype
    out_shape  = x.shape[:-1] + (self.out_features,)

    # gptq_gemm only produces correct results with fp16 activations.
    x_2d = x.reshape(-1, self.in_features).to(torch.float16)

    y = gptq_gemm(
        x_2d,
        self.qweight,
        self.qzeros,
        self.scales,
        self._g_idx_seq,   # sequential [0,0,...,1,1,...] — required for use_shuffle=False
        False,             # use_shuffle=False: raw qweight, no gptq_shuffle pre-processing
        self.bits,
    )
    return y.to(orig_dtype).reshape(out_shape)
```

`x.reshape(-1, in_features)` flattens any leading batch and sequence dimensions into one row axis — `gptq_gemm` accepts 2D input only. After the call, `reshape(out_shape)` restores the original leading dimensions, making `GPTQLinear` a drop-in for `nn.Linear` from the caller's perspective.

The `use_shuffle=True` path requires calling `gptq_shuffle` (a weight pre-permutation) to reorder `qweight` into a tile-friendly layout before the first forward, which can improve memory access patterns in the decode kernel. However, `sgl_kernel` version 0.4.1's `gptq_shuffle` produces incorrect results for the Qwen3 GPTQ checkpoint layout. Layer 13 therefore uses `use_shuffle=False` exclusively and `prepare()` is effectively a no-op (`_prepared = True` is set as a correctness sentinel).

---

## GPTQLinear Threading Through the Model

`bits` and `group_size` are read from `quantize_config.json` at load time and threaded top-down through the model hierarchy:

```python
# model_gptq/qwen3.py — Qwen3Model.__init__
self.layers = nn.ModuleList(
    [Qwen3DecoderLayer(config, i, bits, group_size)
     for i in range(config.num_hidden_layers)]
)
```

`Qwen3DecoderLayer` receives `bits` and `group_size` and passes them to `Qwen3Attention` and `Qwen3MLP`. `Qwen3MLP` replaces its three projections:

```python
# model_gptq/qwen3.py — Qwen3MLP (abbreviated)
self.gate_proj = GPTQLinear(config.hidden_size, config.intermediate_size, bits, group_size)
self.up_proj   = GPTQLinear(config.hidden_size, config.intermediate_size, bits, group_size)
self.down_proj = GPTQLinear(config.intermediate_size, config.hidden_size, bits, group_size)
```

The FFN computation `F.silu(gate_proj(x)) * up_proj(x)` followed by `down_proj` is structurally identical to the bfloat16 version. `embed_tokens` (`nn.Embedding`) and `lm_head` (`nn.Linear`) are not quantized — their parameter counts are small relative to the decoder projections, and quantizing the embedding table degrades generation quality noticeably.

---

## Model Loading with GPTQ

`from_pretrained` performs eight steps to construct a working quantized model:

```python
# model_gptq/qwen3.py — Qwen3ForCausalLM.from_pretrained (key steps)
model_dir = _resolve_model_path(model_path)
config    = Qwen3Config.from_pretrained(model_dir)           # Step 1
bits, group_size = _read_gptq_config(model_dir)              # Steps 2-3

model = cls(config, bits=bits, group_size=group_size)        # Step 4 — empty buffers

# Steps 5-6: load safetensors weights into each buffer by name
for shard in sorted(model_dir.glob("*.safetensors")):
    with safe_open(shard, framework="pt", device=str(device)) as f:
        for key in f.keys():
            load_into_model(model, key, f.get_tensor(key))

# Step 7: cast floating-point parameters only (skips int32 buffers and fp16 scales)
for name, param in model.named_parameters():
    param.data = param.data.to(dtype)

# Step 8: mark every GPTQLinear as prepared
n_prepared = 0
for m in model.modules():
    if isinstance(m, GPTQLinear):
        m.prepare()
        n_prepared += 1
```

Step 7 iterates `named_parameters()` rather than calling `model.to(dtype)`. `named_parameters()` yields only true `nn.Parameter` objects — the `register_buffer` tensors (`qweight`, `scales`, `qzeros`, `_g_idx_seq`) are not included and are therefore never cast. This preserves `scales` as fp16 and `qweight`/`qzeros` as int32.

---

## The Full Loop

`ModelRunner.__init__` with `use_gptq=True` imports `model_gptq.Qwen3ForCausalLM` at the two-line dispatch:

```python
if use_gptq:
    from model_gptq import Qwen3ForCausalLM as ModelClass
else:
    from model import Qwen3ForCausalLM as ModelClass

self.model = ModelClass.from_pretrained(model_path, dtype=DTYPE)
```

After `from_pretrained`, pool sizing queries `torch.cuda.mem_get_info()` — which now returns more free memory because the model weights occupy ~0.85 GB instead of ~3.4 GB. `KVPool` and `RadixCache` are constructed identically to Layer 12.

At prefill time, `prefill_batch` assembles `ids_t`, calls `begin_forward`, constructs `ExtendKVCtx`, and calls `self.model(...)`. Inside the model, each attention layer calls `self.q_proj(hidden_states)` which dispatches to `GPTQLinear.forward` → `gptq_gemm`. The output tensor shape `[B, q_len, n_heads, head_dim]` is the same as in the bfloat16 path. FlashInfer's extend kernel operates on `ExtendKVCtx` without any awareness of quantization.

At decode time, `decode_step` is identical to Layer 12. `GPTQLinear.forward` is called for 4 projections in each of 28 layers on every decode step — 112 `gptq_gemm` calls per step instead of 112 `F.linear` calls. The `RadixCache`, `begin_forward`/`end_forward`, slot allocation, and sampling are unchanged.

---

## What Comes Next

Layer 13 reduces weight memory but does not change the token throughput formula: each decode step commits exactly one token per request, regardless of how fast the model forward pass runs. The bottleneck for generation latency is the number of target model forward passes required per output token. Layer 14 addresses this with speculative decoding: a small draft model (Qwen3-0.6B) autoregressively generates N candidate tokens in N cheap draft decode steps; the target model (Qwen3-1.7B, optionally with GPTQ weights) verifies all N+1 positions — the N draft tokens plus the last confirmed token — in a single extend pass. For greedy decoding, each draft token is accepted if and only if the target model's argmax agrees. The accepted prefix length `k ∈ [0, N]` plus one bonus token is committed per target call, increasing effective throughput by `(k+1)` relative to a single non-speculative decode step. `spec_runner.py` in Layer 14 manages two independent `ModelRunner` instances, draft KV mirroring, KV rewind on rejection, and the accept/reject arithmetic.
