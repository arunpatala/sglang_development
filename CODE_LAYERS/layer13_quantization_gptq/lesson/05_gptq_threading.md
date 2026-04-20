# 05 — GPTQLinear Threading Through the Model

`GPTQLinear` replaces `nn.Linear` in seven places per decoder layer. These replacements are implemented in `model_gptq/`, which mirrors `model/` in every other respect. The `bits` and `group_size` parameters flow top-down from `Qwen3ForCausalLM` through `Qwen3Model` to each `Qwen3DecoderLayer`, and then separately to `Qwen3Attention` and `Qwen3MLP`.

---

## Qwen3Model and Qwen3DecoderLayer

```python
# model_gptq/qwen3.py — Qwen3Model.__init__
class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config, bits: int = 4, group_size: int = 128) -> None:
        super().__init__()
        self.config = config
        # embed_tokens stays as nn.Embedding — not quantized
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i, bits, group_size)
             for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

`bits` and `group_size` are read from `quantize_config.json` by `_read_gptq_config` in `from_pretrained` and passed at construction. Each of the 28 `Qwen3DecoderLayer` instances receives them.

```python
# model_gptq/decoder_layer.py — Qwen3DecoderLayer.__init__
def __init__(
    self,
    config: Qwen3Config,
    layer_idx: int,
    bits: int = 4,
    group_size: int = 128,
) -> None:
    super().__init__()
    self.self_attn = Qwen3Attention(config, layer_idx, bits, group_size)
    self.mlp       = Qwen3MLP(config, bits, group_size)
    self.input_layernorm      = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

`RMSNorm` layers are not quantized: they contain only a learnable scale vector of shape `[hidden_size]` — 2048 × 2 bytes = 4 KB per norm. Quantizing them would yield negligible memory savings while introducing potential accuracy degradation in the layer normalization that all other computations depend on.

---

## Qwen3Attention

```python
# model_gptq/attention.py — Qwen3Attention.__init__
def __init__(
    self,
    config: Qwen3Config,
    layer_idx: int,
    bits: int = 4,
    group_size: int = 128,
) -> None:
    super().__init__()
    q_dim  = config.num_attention_heads    * config.head_dim   # 2048
    kv_dim = config.num_key_value_heads   * config.head_dim   # 1024

    self.q_proj = GPTQLinear(config.hidden_size, q_dim,  bits, group_size)
    self.k_proj = GPTQLinear(config.hidden_size, kv_dim, bits, group_size)
    self.v_proj = GPTQLinear(config.hidden_size, kv_dim, bits, group_size)
    self.o_proj = GPTQLinear(q_dim, config.hidden_size,  bits, group_size)
    self.backend = PagedExtendBackend(config)
```

The `forward` method is identical to `model/attention.py`:

```python
def forward(self, hidden_states, position_ids, forward_batch):
    B, q_len, _ = hidden_states.shape
    q = self.q_proj(hidden_states).view(B, q_len, self.num_heads,    self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
    # ... RoPE ...
    attn_out = self.backend.forward(q, k, v, self.layer_idx, forward_batch)
    out = attn_out.transpose(1, 2).reshape(B, q_len, -1)
    return self.o_proj(out)   # GPTQLinear.forward
```

`self.q_proj(hidden_states)` dispatches to `GPTQLinear.forward` → `gptq_gemm`. The shape contract is unchanged: `hidden_states [B, q_len, hidden_size]` → `q [B, q_len, q_dim]` → split into heads. The `PagedExtendBackend` receives the same `[B, n_heads, q_len, head_dim]` tensors and calls FlashInfer identically to the bfloat16 path.

---

## Qwen3MLP

```python
# model_gptq/qwen3.py — Qwen3MLP (inside the decoder layer in qwen3.py)
class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config, bits: int = 4, group_size: int = 128) -> None:
        super().__init__()
        self.gate_proj = GPTQLinear(config.hidden_size, config.intermediate_size, bits, group_size)
        self.up_proj   = GPTQLinear(config.hidden_size, config.intermediate_size, bits, group_size)
        self.down_proj = GPTQLinear(config.intermediate_size, config.hidden_size, bits, group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

For Qwen3-1.7B, `intermediate_size = 11008`. The three FFN projections:
- `gate_proj [11008, 2048]` bfloat16: 45.1 MB → `[2048//8, 11008]` int32: 11.3 MB + minimal scales
- `up_proj [11008, 2048]`: same as `gate_proj`
- `down_proj [2048, 11008]`: same

Total FFN per layer in bfloat16: ~135 MB. After GPTQ: ~34 MB. Across 28 layers: savings of approximately 2.8 GB just from FFN quantization.

The SiLU gating formula `F.silu(gate_proj(x)) * up_proj(x)` is unchanged. `gate_proj(x)` returns bfloat16 activations (after the float16→bfloat16 cast inside `GPTQLinear.forward`). `F.silu` operates on bfloat16. The elementwise product is bfloat16. `down_proj` receives bfloat16 input, casts to float16 internally for `gptq_gemm`, and returns bfloat16. The entire FFN is numerically equivalent to the bfloat16 version modulo the quantization error in the weights.

---

## embed_tokens and lm_head

```python
# model_gptq/qwen3.py — Qwen3Model / Qwen3ForCausalLM
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # NOT quantized
# ...
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # NOT quantized
```

`embed_tokens` is an `nn.Embedding` — it is a lookup table, not a projection matrix, and embedding quantization requires separate approaches (not supported by `gptq_gemm`). `lm_head` is an `nn.Linear` with `hidden_size = 2048` and `vocab_size = 151936` for Qwen3; quantizing it would affect the output logits distribution for every token position. Standard GPTQ recipes leave these two layers in full precision. For Qwen3-1.7B, `lm_head.weight` is `151936 × 2048 × 2 = 623 MB` — significant but outside the GPTQ scope for this layer.

Section 06 explains the loading sequence that populates these buffers from a GPTQ checkpoint.
