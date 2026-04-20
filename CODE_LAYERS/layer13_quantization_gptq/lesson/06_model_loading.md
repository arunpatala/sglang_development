# 06 — Model Loading with GPTQ

Loading a GPTQ model requires a specific sequence of steps that differs from loading a standard bfloat16 model. The key challenge is that the model has a mixed dtype structure: most floating-point parameters (RMSNorm weights, `lm_head`, `embed_tokens`) should be cast to the target activation dtype (bfloat16), while the quantization buffers (`scales` in float16, `qweight`/`qzeros`/`g_idx` in int32) must remain at their registered dtypes.

---

## Reading quantize_config.json

```python
# model_gptq/qwen3.py — _read_gptq_config
def _read_gptq_config(model_dir: Path) -> tuple[int, int]:
    cfg_path = model_dir / "quantize_config.json"
    if not cfg_path.exists():
        return 4, 128   # defaults

    with open(cfg_path) as f:
        cfg = json.load(f)
    bits       = cfg.get("bits", 4)
    group_size = cfg.get("group_size", 128)
    logger.info(f"GPTQ config: bits={bits}, group_size={group_size}")
    return bits, group_size
```

`quantize_config.json` is a standard file present in all GPTQ checkpoints from the AutoGPTQ and GPTQ-for-LLaMa ecosystems. It contains at minimum `bits` and `group_size`; it may also contain `desc_act` (whether weights are reordered by activation magnitude), `sym` (symmetric vs asymmetric quantization), and other fields. Layer 13 reads only `bits` and `group_size` — the other fields are either not applicable (Qwen3 uses `desc_act=False`) or not needed (`sym` is handled by whether `qzeros` is all-8s or variable).

---

## The Eight-Step Loading Sequence

```python
# model_gptq/qwen3.py — Qwen3ForCausalLM.from_pretrained (key steps)
@classmethod
def from_pretrained(cls, model_path: str, dtype=torch.bfloat16, device="cuda"):
    model_dir = _resolve_model_path(model_path)            # Step 1: resolve HF Hub or local path

    config    = Qwen3Config.from_pretrained(model_dir)     # Step 2: read config.json
    bits, group_size = _read_gptq_config(model_dir)        # Step 3: read quantize_config.json

    model = cls(config, bits=bits, group_size=group_size)  # Step 4: build with empty buffers

    model = model.to(device)                               # Step 5: move to GPU (empty buffers)

    # Step 6: load safetensors weights into each buffer by name
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device=str(device)) as f:
            for key in f.keys():
                # match key to parameter/buffer and copy into the pre-allocated tensor
                load_into_model(model, key, f.get_tensor(key))

    # Step 7: cast floating-point parameters only (skips int32 buffers and fp16 scales)
    for name, param in model.named_parameters():
        param.data = param.data.to(dtype)

    # Step 8: prepare every GPTQLinear
    n_prepared = 0
    for m in model.modules():
        if isinstance(m, GPTQLinear):
            m.prepare()
            n_prepared += 1

    return model
```

**Step 4** constructs the model with all `GPTQLinear` buffers empty. `torch.empty` allocates uninitialized GPU memory — no zeros are written, no CUDA memset is called. The buffer shapes are fixed by `bits`, `group_size`, `in_features`, and `out_features`.

**Step 5** calls `model.to(device)`. This moves the empty buffers to GPU. For int32 buffers, the move is a raw CUDA `cudaMemcpy`; for float16 `scales`, it is the same. No dtype conversion happens here.

**Step 6** loads the checkpoint tensors. A GPTQ checkpoint's safetensors file contains keys like `model.layers.0.self_attn.q_proj.qweight`, `model.layers.0.self_attn.q_proj.scales`, etc. The `load_into_model` helper matches each key to the corresponding buffer and copies the tensor in-place. Because `model.to(device)` already placed the buffers on GPU, the copy goes directly from the safetensors memory-mapped file to GPU — no intermediate CPU tensor.

**Step 7** casts floating-point parameters. `model.named_parameters()` yields:
- `model.embed_tokens.weight` (float16 from the checkpoint → cast to bfloat16)
- `model.layers.N.input_layernorm.weight` (float16 → bfloat16, per layer)
- `model.layers.N.post_attention_layernorm.weight` (float16 → bfloat16, per layer)
- `model.norm.weight` (float16 → bfloat16)
- `model.lm_head.weight` (float16 → bfloat16)

It does NOT yield `qweight`, `scales`, `qzeros`, `g_idx`, or `_g_idx_seq` — these are registered buffers, not parameters. They retain their dtypes: int32 for `qweight`/`qzeros`/`g_idx`, float16 for `scales`.

**Step 8** calls `prepare()` on every `GPTQLinear`. In the current implementation, `prepare()` sets `_prepared = True` and returns. This is a no-op from a computation perspective; the `use_shuffle=False` kernel path does not require weight pre-processing. The call exists to mark the layer as ready (and to provide a hook for future `use_shuffle=True` support where `gptq_shuffle` would be called here).

---

## use_gptq Dispatch in model_runner

```python
# model_runner.py — ModelRunner.__init__
if use_gptq:
    from model_gptq import Qwen3ForCausalLM as ModelClass
else:
    from model import Qwen3ForCausalLM as ModelClass

self.model = ModelClass.from_pretrained(model_path, dtype=DTYPE)
```

Two lines. After this, all subsequent code — pool sizing, `prefill_batch`, `decode_step`, `radix_cache` — is identical regardless of which model class was loaded. The `model(ids_t, ...)` call dispatches through `Qwen3ForCausalLM.forward` → `Qwen3Model.forward` → each `Qwen3DecoderLayer.forward` → `Qwen3Attention.forward` → `GPTQLinear.forward` for each projection. The output shapes and dtypes are the same as the bfloat16 model.

The `from_pretrained` call after the import measures `torch.cuda.mem_get_info()` after the model loads (inside `ModelRunner.__init__`, after `self.model = ModelClass.from_pretrained(...)`). With GPTQ weights, the model occupies approximately 1.25 GB instead of 5 GB, leaving approximately 3.75 GB more for the KV pool — directly increasing `max_pages` in `KVPool`.

Section 07 traces a full prefill and decode sequence to show the end-to-end flow with the GPTQ model.
