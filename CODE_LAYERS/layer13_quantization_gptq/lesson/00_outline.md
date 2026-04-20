# Layer 13 — Lesson Outline

## What This Lesson Covers

Layer 12 introduced `RadixCache` to eliminate redundant KV computation for shared prompt prefixes, but left model weights in bfloat16. For Qwen3-1.7B the 28 decoder layers' projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and FFN linear layers (`gate_proj`, `up_proj`, `down_proj`) dominate GPU memory — roughly 3.4 GB — which limits how many KV pages can coexist with the weights in VRAM. A quantized model frees that headroom.

Layer 13 adds `GPTQLinear`: a 4-bit quantized weight module that packs 8 int4 values per int32 element in `qweight`, stores per-group scales in fp16 (`scales`), and stores quantized zero-points packed the same way as `qweight` (`qzeros`). At inference time, the `gptq_gemm` fused CUDA kernel (from `sgl_kernel`) dequantizes and matrix-multiplies in a single pass — no separate dequantize step, no full fp16 weight materialization. The `model_gptq/` directory mirrors `model/` but replaces all `nn.Linear` projection layers with `GPTQLinear`. `model_runner.py` gains a `use_gptq` flag that selects the correct `Qwen3ForCausalLM` class at startup.

The change adds one new file (`model_gptq/gptq_linear.py`) and modifies four existing ones inside `model_gptq/` (which was already established for the attention refactoring in prior layers). The `RadixCache`, `KVPool`, `Scheduler`, `decode_step`, `ForwardBatch`, and all `forward_batch.py` logic are completely unchanged.

The sections follow the quantization pipeline: why weight memory is the binding constraint, the `qweight` bit-packing layout, the `scales` and `qzeros` group structure, the `gptq_gemm` kernel interface and the `use_shuffle=False` path, the `prepare()` no-op, how `GPTQLinear` is threaded through `Qwen3Attention` and `Qwen3MLP`, model loading order (`from_pretrained` → `quantize_config.json` → buffer registration → weight loading → `model.to(dtype)` skipping buffers → `prepare()`), the `use_gptq` dispatch in `model_runner`, the full call trace, and what quantization alone cannot address that Layer 14's speculative decoding targets.

---

## Sections

### 01 — From nn.Linear to GPTQLinear (`01_from_linear_to_gptq.md`)
- Layer 12's weight memory: `nn.Linear(hidden_size, q_dim)` stores `2048 × 2048 × 2` = 8 MB in bfloat16; 28 layers × 7 linear layers per layer = 196 matrices ≈ 3.4 GB total
- Layer 13's reduction: `GPTQLinear` stores the same matrix as `qweight [K//8, N]` int32 — 8 values packed per int32 — reducing weight memory to ~0.85 GB at 4 bits; the 4× headroom is available for a larger KV pool or a second model (speculative decoding)
- What stays the same: `Qwen3Model`, `Qwen3DecoderLayer`, `Qwen3Attention.forward` shapes (`[B, q_len, heads, head_dim]`), the entire KV path, scheduler, prefix caching
- Dispatch at startup: `model_runner.py` checks `use_gptq` and imports either `model.Qwen3ForCausalLM` or `model_gptq.Qwen3ForCausalLM`; after that, both paths share identical `prefill_batch`, `decode_step`, `FlashInfer` calls, and `RadixCache` interaction

### 02 — The qweight Layout (`02_qweight_layout.md`)
- int4-packed int32: 8 four-bit values are packed into one 32-bit integer, low-nibble first; `qweight` shape is `[K // pack_factor, N]` = `[K // 8, N]` where K = `in_features`, N = `out_features`
- Contrast with `nn.Linear.weight [N, K]`: `qweight` is transposed relative to the PyTorch convention because `gptq_gemm` expects column-major weight layout; `in_features` varies along axis 0 in the packed tensor
- `pack_factor = 32 // bits` — 8 for 4-bit; changing `bits` to 8 gives 4 values per int32 and halves the compression
- Buffer registration via `register_buffer`: PyTorch treats buffers as part of `state_dict` but does not apply gradient accumulation; `model.to(dtype)` skips non-floating-point buffers — `qweight` (int32) and `qzeros` (int32) survive `model.to(bfloat16)` intact
- `_g_idx_seq [K]` int32: a pre-computed sequential group index `[0,0,...,1,1,...]` with `group_size` repetitions; required by the `use_shuffle=False` kernel path to map each input row to its scale group

### 03 — Scales, Zeros, and Group Quantization (`03_scales_zeros_groups.md`)
- Why per-group quantization: a single scale per output column would force all 2048 input rows to share one scale value, losing accuracy in the tails of the weight distribution; group_size=128 gives 16 scale groups per column for K=2048
- `scales [K // group_size, N]` float16: one scale per group per output column; `gptq_gemm` reads scale bits as raw fp16 — any cast to bfloat16 would corrupt the values by reinterpreting the bit pattern; `scales` buffer is always registered as `torch.float16` and never included in `model.to(dtype)`
- `qzeros [K // group_size, N // pack_factor]` int32: zero-points packed the same way as `qweight`; the dequantize formula `w_fp = (qweight_unpacked - qzeros) * scales` centers each group's weight distribution around zero before multiplying by the scale
- Why `g_idx` from the checkpoint is ignored: Qwen3 GPTQ checkpoints use `desc_act=False` — weights are not reordered by activation magnitude; the sequential `_g_idx_seq` is always correct and avoids a permutation lookup
- Memory accounting: `scales` in fp16 adds `K/group_size × N × 2` bytes; for K=2048, N=2048, group_size=128: `16 × 2048 × 2 = 64 KB` per matrix — negligible relative to the 8 MB it replaces

### 04 — gptq_gemm and the use_shuffle=False Path (`04_gptq_gemm_kernel.md`)
- `gptq_gemm(x, qweight, qzeros, scales, g_idx, use_shuffle, bits)` from `sgl_kernel`: fused CUDA kernel that dequantizes and matrix-multiplies in one pass; signature matches ExLlama v2's kernel interface
- `use_shuffle=False`: the kernel reads `qweight` in the raw packed layout and uses `g_idx` to look up the correct scale/zero row for each input row; no pre-processing of `qweight` is required
- `use_shuffle=True`: a gptq_shuffle pre-permutation reorders `qweight`'s rows into a tile-friendly layout for faster cache access; this path requires calling `gptq_shuffle` before the first forward — SGLang 0.4.1's `gptq_shuffle` produces incorrect results for this checkpoint layout, so Layer 13 uses `use_shuffle=False` exclusively
- Activation dtype constraint: `gptq_gemm` reads activation bits as fp16; passing bfloat16 activations causes drastically wrong outputs (different bit-level representation for the same real number); `forward` casts `x` to `float16` before calling `gptq_gemm` and casts the result back to `orig_dtype`
- Input reshape: `x.reshape(-1, self.in_features)` flattens any leading batch/sequence dimensions into a single matrix row dimension; `gptq_gemm` operates on 2D inputs; the result is reshaped back to `out_shape = x.shape[:-1] + (out_features,)` after the call

### 05 — GPTQLinear Threading Through the Model (`05_gptq_threading.md`)
- `Qwen3DecoderLayer.__init__(config, layer_idx, bits, group_size)`: receives `bits` and `group_size` from `Qwen3Model` which reads them from `quantize_config.json`; passes them to `Qwen3Attention` and `Qwen3MLP`
- `Qwen3Attention`: replaces `self.q_proj = nn.Linear(...)` with `self.q_proj = GPTQLinear(config.hidden_size, q_dim, bits, group_size)`; same for `k_proj`, `v_proj`, `o_proj`; `forward` is otherwise identical — `q_proj(hidden_states)` calls `GPTQLinear.forward` transparently
- `Qwen3MLP`: replaces `gate_proj`, `up_proj`, `down_proj` with `GPTQLinear`; the SiLU gate + elementwise product logic is unchanged: `F.silu(self.gate_proj(x)) * self.up_proj(x)` and then `down_proj`
- `embed_tokens` and `lm_head` stay as `nn.Embedding` and `nn.Linear`: these layers are not quantized in standard GPTQ recipes; the embedding table's memory is proportional to `vocab_size × hidden_size`, small relative to the decoder layers
- `RMSNorm` also stays in bfloat16: normalization layers have negligible parameter count; quantizing them would reduce accuracy with no meaningful memory saving

### 06 — Model Loading with GPTQ (`06_model_loading.md`)
- `_read_gptq_config(model_dir)`: reads `quantize_config.json` from the checkpoint directory; extracts `bits` and `group_size`; falls back to `(4, 128)` if the file is absent
- `from_pretrained` loading sequence: (1) read `config.json` → `Qwen3Config`; (2) read `quantize_config.json` → `bits`, `group_size`; (3) build model with `GPTQLinear` layers (empty buffers, CPU); (4) `model.to(DEVICE)` — moves buffers to GPU, skips non-fp tensors implicitly; (5) load `safetensors` weights into each buffer by name; (6) `model.to(dtype=bfloat16)` selectively — only floating-point parameters; (7) call `GPTQLinear.prepare()` on every quantized layer
- Why `model.to(dtype)` must skip buffers: `model.to(bfloat16)` would cast `scales` from fp16 to bfloat16, corrupting the bit pattern that `gptq_gemm` reads as fp16; `scales` must remain fp16; the loading code skips buffer casting by iterating `named_parameters` instead of calling `module.to(dtype)` globally
- `prepare()` is a no-op in this implementation: `use_shuffle=False` does not require weight permutation; `_prepared = True` is set as a correctness marker to detect accidental double-prepare or missing prepare
- `use_gptq` dispatch in `model_runner.__init__`: two lines suffice — `from model_gptq import Qwen3ForCausalLM as ModelClass` or `from model import Qwen3ForCausalLM as ModelClass`; all subsequent code is identical

### 07 — The Full Loop (`07_the_full_loop.md`)
- Startup: `ModelRunner.__init__` with `use_gptq=True` imports `model_gptq.Qwen3ForCausalLM`; `from_pretrained` loads 28 decoder layers with `GPTQLinear` projections; pool sizing proceeds as in Layer 12 with more free memory available
- Prefill: `prefill_batch` runs identically to Layer 12; at each attention layer, `self.q_proj(hidden_states)` dispatches to `GPTQLinear.forward` → `gptq_gemm`; output shape `[B, q_len, n_heads, head_dim]` is unchanged; FlashInfer extend kernel operates on the same `ExtendKVCtx`
- Decode: `decode_step` identical to Layer 12; `GPTQLinear.forward` is called for every projection in every decoder layer on every step; no change to `DecodeKVCtx`, `begin_forward`/`end_forward`, or sampling
- Prefix caching: completely unchanged; `RadixCache.match_prefix`, `cache_finished_req`, `evict`, and `lock_ref` operate on pool pages without any knowledge of whether weights are quantized
- Metrics: throughput (tok/s) increases because model weight load time from VRAM decreases at each decode step; perplexity is measured against the fp16 reference to verify quantization accuracy

### 08 — What Comes Next (`08_whats_next.md`)
- The bottleneck that remains: even with 4-bit weights and prefix caching, each token requires a full 28-layer forward pass through the target model; at decode time only one new token is committed per step regardless of how cheap the forward pass becomes
- Layer 14 (speculative decoding): a small draft model (Qwen3-0.6B) autoregressively generates N candidate tokens in N sequential draft decode steps; the target model (Qwen3-1.7B) verifies all N+1 positions in a single EXTEND pass; the longest accepted prefix is committed; accepted tokens per target call is the `N × acceptance_rate` factor instead of 1
- What changes in Layer 14: `spec_runner.py` (new — `SpecRunner` manages two `ModelRunner` instances, draft KV mirroring, accept/reject logic, and KV rewind); `server.py` routes requests through `SpecRunner.prefill` and `SpecRunner.spec_decode_step` instead of `ModelRunner`; all per-layer files are unchanged

---

## Supporting Files

- `summary.md` — blog-post-style summary covering all sections
- `sglang_reference.md` — maps Layer 13 concepts to SGLang source: `GPTQLinear` → `torch_weight_quant.py` `GPTQMarlinMethod`; `gptq_gemm` → `sgl_kernel.gptq_gemm`; `quantize_config.json` → standard GPTQ checkpoint format; `use_gptq` flag → `ServerArgs.quantization`

---

## Key Code Anchors

| Concept | Location |
|---|---|
| `GPTQLinear` class | `model_gptq/gptq_linear.py` line 39: `class GPTQLinear(nn.Module):` |
| `pack_factor` | `model_gptq/gptq_linear.py` line 59: `self.pack_factor = 32 // bits` |
| `qweight` buffer | `model_gptq/gptq_linear.py` line 64: `torch.empty(in_features // self.pack_factor, out_features, dtype=torch.int32)` |
| `scales` buffer (fp16) | `model_gptq/gptq_linear.py` line 71: `torch.empty(in_features // group_size, out_features, dtype=torch.float16)` |
| `qzeros` buffer | `model_gptq/gptq_linear.py` line 75: `torch.empty(in_features // group_size, out_features // self.pack_factor, dtype=torch.int32)` |
| `_g_idx_seq` | `model_gptq/gptq_linear.py` line 87: `torch.arange(in_features, dtype=torch.int32) // group_size` |
| `prepare()` no-op | `model_gptq/gptq_linear.py` line 96: `def prepare(self) -> None:` |
| `forward` fp16 cast | `model_gptq/gptq_linear.py` line 130: `x_2d = x.reshape(-1, self.in_features).to(torch.float16)` |
| `gptq_gemm` call | `model_gptq/gptq_linear.py` line 132: `y = gptq_gemm(x_2d, self.qweight, self.qzeros, self.scales, self._g_idx_seq, False, self.bits)` |
| `Qwen3Attention` GPTQ projections | `model_gptq/attention.py` line 57: `self.q_proj = GPTQLinear(config.hidden_size, q_dim, bits, group_size)` |
| `bits`/`group_size` threading | `model_gptq/decoder_layer.py` line 37: `bits: int = 4` |
| `Qwen3Model` GPTQ init | `model_gptq/qwen3.py` line 98: `def __init__(self, config, bits=4, group_size=128)` |
| `_read_gptq_config` | `model_gptq/qwen3.py` line 75: `def _read_gptq_config(model_dir)` |
| `prepare()` loop after load | `model_gptq/qwen3.py` line 337: `if isinstance(m, GPTQLinear): m.prepare()` |
| `use_gptq` dispatch | `model_runner.py` line 94: `if use_gptq: from model_gptq import Qwen3ForCausalLM as ModelClass` |
