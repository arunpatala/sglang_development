# 03 — Weight Loading

## The `safetensors` Format

HuggingFace used to distribute model weights as `.bin` files — Python pickle archives produced by `torch.save`. Pickle is convenient but carries a serious risk: loading an untrusted `.bin` file executes arbitrary Python code, because pickle is a general serialisation protocol. It also has performance problems: `torch.load` deserialises the entire file into memory before the tensors are accessible, making peak memory during loading roughly twice the model size.

`safetensors` is a purpose-built binary format for storing tensors that addresses both problems. The file layout is:

```
[8 bytes: header length N]
[N bytes: JSON header]
[raw tensor data — flat binary, no padding between tensors]
```

The JSON header maps each tensor name to its dtype, shape, and byte offset within the file:

```json
{
  "model.embed_tokens.weight": {"dtype": "BF16", "shape": [151936, 1024], "data_offsets": [0, 311296000]},
  "model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [2048, 1024], "data_offsets": [311296000, 315490304]},
  "model.layers.0.self_attn.k_proj.weight": {"dtype": "BF16", "shape": [1024, 1024], "data_offsets": [315490304, 317587456]},
  ...
}
```

Because every tensor's location is encoded in the header, the file can be opened as a memory-mapped region. `safe_open` maps the file into the process's virtual address space with a single `mmap` syscall; `f.get_tensor(key)` reads the byte offset from the header and constructs a tensor backed by that mapped memory — no file I/O, no copy, no allocation. The tensor data is read from disk only when the bytes are actually accessed (by the CPU casting operation in `.to(dtype)`).

There is no code execution during loading — the JSON header is parsed, the byte offsets are followed, and raw bytes are interpreted as IEEE 754 floats. A malicious `safetensors` file can corrupt your model's weights, but it cannot execute shell commands.

---

## What Is in `model.safetensors`

For Qwen3-0.6B, `model.safetensors` contains 290 tensors. Their names match HuggingFace's module hierarchy exactly:

```
model.embed_tokens.weight                          [151936, 1024]  bf16  ~310 MB
model.norm.weight                                  [1024]          bf16  ~2 KB

model.layers.0.input_layernorm.weight              [1024]          bf16
model.layers.0.self_attn.q_proj.weight             [2048, 1024]    bf16
model.layers.0.self_attn.q_norm.weight             [128]           bf16
model.layers.0.self_attn.k_proj.weight             [1024, 1024]    bf16
model.layers.0.self_attn.k_norm.weight             [128]           bf16
model.layers.0.self_attn.v_proj.weight             [1024, 1024]    bf16
model.layers.0.self_attn.o_proj.weight             [1024, 1024]    bf16
model.layers.0.post_attention_layernorm.weight     [1024]          bf16
model.layers.0.mlp.gate_proj.weight                [3072, 1024]    bf16
model.layers.0.mlp.up_proj.weight                  [3072, 1024]    bf16
model.layers.0.mlp.down_proj.weight                [1024, 3072]    bf16
... × 28 layers
lm_head.weight                                     [151936, 1024]  bf16  ~310 MB
```

Notably, both `model.embed_tokens.weight` and `lm_head.weight` appear in the file — both are 310 MB. When `tie_word_embeddings` is true, they are identical copies of the same matrix. The checkpoint stores both because it was saved from a model where the tie was in effect at training time, but the file format has no mechanism to express shared storage — each tensor gets its own byte range. The weight loading code must handle this explicitly.

---

## Cast Before Copy

After the HF skeleton is constructed (section 02), `from_pretrained` casts the model to `bfloat16`:

```python
model = model.to(dtype)   # Step 4 — cast before copying weights
```

This step is deliberately placed between skeleton construction and weight loading. The reason is memory efficiency.

`AutoModelForCausalLM.from_config` initialises all parameters in `float32` by default — PyTorch's global default dtype. Qwen3-0.6B with 0.6B parameters at `float32` takes roughly 2.4 GB on CPU. Casting to `bfloat16` before loading halves this to ~1.2 GB.

More importantly, the cast changes the dtype of all parameter tensors in-place. When `copy_` runs later, it writes `bfloat16` source bytes directly into `bfloat16` parameter buffers — one memory operation. If instead `copy_` ran first (writing `bfloat16` checkpoint values into `float32` parameters) and casting happened after, PyTorch would perform two full passes over the parameter memory: one for the copy and one for the cast. Casting first eliminates the second pass.

---

## The Streaming Iterator

The weight source for `load_weights` is a Python generator that wraps `safe_open`:

```python
def _iter():
    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            yield key, f.get_tensor(key).to(dtype)
```

`safe_open` opens the file and parses the JSON header — this is fast, a few milliseconds. `f.keys()` returns the list of tensor names in the order they appear in the header. For each key, `f.get_tensor(key)` returns a CPU tensor backed by the memory-mapped file region (it does not copy data from disk unless the pages are not yet loaded). `.to(dtype)` casts the tensor to `bfloat16`, allocating a new CPU tensor with the converted values.

This generator yields one tensor at a time. At any given moment during loading, there is at most one freshly-cast tensor in memory — the one currently being processed by `copy_`. The moment the loop moves to the next key, the previous tensor goes out of scope and is freed. Peak additional memory during loading is the size of the single largest tensor: `model.embed_tokens.weight` at 310 MB.

The alternative — `torch.load` or `safetensors.torch.load_file` — reads all tensors into a `dict[str, Tensor]` before any copying begins. For Qwen3-0.6B, this materialises an additional ~1.2 GB on CPU before a single parameter is updated. For a 70B model at `bfloat16`, the equivalent is ~140 GB — more than most machines have.

---

## `load_weights` — One Copy_ Per Tensor

`load_weights` accepts the generator and processes each `(name, tensor)` pair:

```python
def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
    params = dict(self._model.named_parameters())
    loaded: set[str] = set()

    for name, tensor in weights:
        if name == "lm_head.weight" and self.config.tie_word_embeddings:
            continue                    # already tied by HF — skip
        if name in params:
            params[name].data.copy_(tensor)
            loaded.add(name)
        else:
            logger.debug(f"Skipping unknown weight: {name}")

    logger.info(f"Loaded {len(loaded)} weight tensors")
```

`dict(self._model.named_parameters())` traverses HF's model and builds a flat `{name: parameter}` mapping. The names match the safetensors keys exactly — both are derived from the same PyTorch module hierarchy — so a simple `name in params` lookup is enough. No regex matching, no name translation.

`params[name].data.copy_(tensor)` accesses the parameter's underlying `data` tensor (bypassing autograd bookkeeping) and copies the source tensor in-place. The parameter buffer already exists in memory at the right dtype and shape (from the skeleton construction and cast steps); `copy_` fills it with the checkpoint values. No new tensors are allocated.

---

## Tied Weights

The last subtlety is the `lm_head.weight` skip.

Qwen3 sets `tie_word_embeddings: true`. The input embedding matrix (`model.embed_tokens.weight`, shape `[151936, 1024]`) and the output projection matrix (`lm_head.weight`, same shape) are meant to be the same tensor in memory — one set of parameters that serves two roles. For Qwen3-0.6B, each copy at `bfloat16` is `151_936 × 1024 × 2 bytes ≈ 310 MB`. Tying avoids allocating that second copy on GPU.

HuggingFace handles the tie at construction time: `AutoModelForCausalLM.from_config` calls `tie_weights()` before returning the skeleton, which makes `lm_head.weight` a Python alias for `model.embed_tokens.weight`. The two attributes point to the same underlying `torch.Storage` object. After this point, writing to either one writes to both.

This is why the loading loop skips `"lm_head.weight"`:

```python
if name == "lm_head.weight" and self.config.tie_word_embeddings:
    continue   # copying embed_tokens.weight is sufficient
```

The loop will later encounter `"model.embed_tokens.weight"` and copy the embedding matrix into `params["model.embed_tokens.weight"].data`. Because the tie is already in place, this copy simultaneously updates the storage that `lm_head.weight` points to. The `lm_head` is correctly initialised without loading the 310 MB `lm_head.weight` tensor from the file at all.

If `load_weights` did not have this `continue`, two things would happen: it would load the `embed_tokens` matrix correctly, and then it would try to load `lm_head.weight` — overwriting the same storage again with the same values. This is not incorrect, but it wastes one redundant 310 MB CPU → GPU copy.

---

## After Loading

After `load_weights` returns, `from_pretrained` finishes with:

```python
model = model.to("cuda").eval()
```

`.to("cuda")` moves every parameter tensor — all ~0.6B values, ~1.2 GB at `bfloat16` — from CPU to GPU in one sweep. `.eval()` sets the model to inference mode, disabling dropout and batch-normalisation tracking (neither is present in Qwen3, but `eval()` is idiomatic and prevents accidental gradient accumulation).

At this point `self.model` in `model_runner.py` holds a fully initialised `Qwen3ForCausalLM`: our `Qwen3Config`, HF's forward computation, all 290 checkpoint tensors correctly loaded, `lm_head` and `embed_tokens` sharing a single 310 MB GPU allocation, everything in `bfloat16`. The generate loop in section 01 can now call `self.model(...)`.
