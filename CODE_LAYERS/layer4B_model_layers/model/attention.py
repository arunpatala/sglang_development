"""
Qwen3Attention — multi-head self-attention with:
  • Grouped Query Attention (GQA): 16 Q heads share 8 KV heads
  • Per-head QK RMSNorm (Qwen3-specific, not in Llama/Qwen2)
  • Rotary Position Embedding applied after QK norm
  • KV cache via our own KVCache object

Mirror of HuggingFace Qwen3Attention (modeling_qwen3.py L222-291) but:
  - layer_idx stored on the object (SGLang style) for KV cache dispatch
  - kv_cache passed explicitly (not returned as output)
  - attention_mask is a prebuilt additive tensor from the model's forward

Extensibility surfaces (what SGLang swaps out):
  • q_proj / k_proj / v_proj → QKVParallelLinear (fused + tensor-parallel)
  • o_proj → RowParallelLinear
  • F.scaled_dot_product_attention → RadixAttention (paged KV, FlashAttn)
  • q_norm / k_norm → fused with QKV projection kernel on ROCm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3Config
from .norm import RMSNorm
from .rope import apply_rotary_pos_emb

# Imported at use-site to avoid circular imports
# from kv_cache import KVCache


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand KV heads to match Q head count for GQA.
    [B, n_kv_heads, S, D] → [B, n_kv_heads * n_rep, S, D]

    Uses expand + reshape (no data copy) like HuggingFace.
    """
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, H, n_rep, S, D)
        .reshape(B, H * n_rep, S, D)
    )


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx          # which KV cache slot to use
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads      # 16
        self.num_kv_heads = config.num_key_value_heads   # 8
        self.num_kv_groups = config.num_kv_groups        # 2
        self.head_dim = config.head_dim                  # 128
        self.scale = self.head_dim ** -0.5

        q_dim = self.num_heads * self.head_dim       # 16 * 128 = 2048
        kv_dim = self.num_kv_heads * self.head_dim   # 8  * 128 = 1024

        self.q_proj = nn.Linear(config.hidden_size, q_dim,  bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(q_dim, config.hidden_size,  bias=config.attention_bias)

        # Per-head QK normalisation — Qwen3-specific stabilisation.
        # Weight shape [head_dim=128], applied independently to each head's
        # query/key before RoPE.  Not present in Llama or Qwen2.
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, q_len, hidden]
        cos: torch.Tensor,                    # [B, q_len, head_dim]
        sin: torch.Tensor,                    # [B, q_len, head_dim]
        attention_mask: torch.Tensor | None,  # [B, 1, q_len, kv_len] additive
        kv_cache=None,                        # KVCache | None
    ) -> torch.Tensor:
        B, q_len, _ = hidden_states.shape

        # ── 1. Project Q / K / V ──────────────────────────────────────────
        # Reshape to [B, seq, n_heads, head_dim] then transpose to
        # [B, n_heads, seq, head_dim] for batch matmul.
        q = self.q_proj(hidden_states).view(B, q_len, self.num_heads,    self.head_dim)
        k = self.k_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)

        # ── 2. Per-head QK RMSNorm (Qwen3-specific) ──────────────────────
        # Applied per-head before RoPE so the scale is head_dim-dimensional.
        q = self.q_norm(q)  # normalises last dim (head_dim)
        k = self.k_norm(k)

        # Transpose to [B, n_heads, seq, head_dim] for attention.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ── 3. Rotary Position Embedding ──────────────────────────────────
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ── 4. KV Cache update ────────────────────────────────────────────
        # update() appends the new k/v and returns the full accumulated
        # [B, n_kv_heads, past+q_len, head_dim] tensors.
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)

        # ── 5. Grouped Query Attention: expand KV to match Q head count ──
        # [B, 8, kv_len, 128] → [B, 16, kv_len, 128]  (no data copy)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # ── 6. Scaled dot-product attention ──────────────────────────────
        # attention_mask is additive (0 = attend, -inf = mask).
        # SDPA handles flash-attention fusing automatically on CUDA.
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            scale=self.scale,
        )  # [B, n_heads, q_len, head_dim]

        # ── 7. Merge heads and output projection ─────────────────────────
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, q_len, -1)
        return self.o_proj(attn_out)
