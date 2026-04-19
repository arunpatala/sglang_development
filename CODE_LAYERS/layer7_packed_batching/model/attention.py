"""
Qwen3Attention — multi-head self-attention with:
  • Grouped Query Attention (GQA): 16 Q heads share 8 KV heads
  • Per-head QK RMSNorm (Qwen3-specific, not in Llama/Qwen2)
  • Rotary Position Embedding applied after QK norm
  •   Two KV cache / attention backends selected by kv_cache type:

    ┌──────────────────────────────────────────────────────────────────┐
    │ kv_cache type      │ Backend         │ When used                 │
    │────────────────────┼─────────────────┼───────────────────────────│
    │ None               │ F.sdpa          │ no-cache prefill           │
    │ PerReqKVCache      │ F.sdpa          │ B=1 prefill (in scheduler) │
    │ PackedKVCache      │ FlashInfer      │ B=N decode step            │
    └──────────────────────────────────────────────────────────────────┘

  Separation of concerns:
    kv_cache.update(layer, k, v) — data only: packs/accumulates KV tensors,
                                   returns (k_full, v_full) for this layer.
    attention.py                 — kernel only: decides F.sdpa vs FlashInfer
                                   and calls the chosen kernel with the data.

  The FlashInfer path (PackedKVCache) uses
  BatchPrefillWithRaggedKVCacheWrapper with q_len=1 per request.
  FlashInfer natively supports GQA, so we skip the repeat_kv() expand
  that the F.sdpa path needs.

Mirror of HuggingFace Qwen3Attention (modeling_qwen3.py L222-291) but:
  - layer_idx stored on the object (SGLang style) for KV cache dispatch
  - kv_cache passed explicitly (not returned as output)
  - attention_mask is a prebuilt additive tensor for the F.sdpa path;
    ignored entirely for the FlashInfer path (ragged attention handles masking)

Extensibility surfaces (what SGLang swaps out):
  • q_proj / k_proj / v_proj → QKVParallelLinear (fused + tensor-parallel)
  • o_proj → RowParallelLinear
  • F.sdpa → RadixAttention (paged KV, FlashAttn)   ← this file does the same swap
  • q_norm / k_norm → fused with QKV projection kernel on ROCm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3Config
from .norm import RMSNorm
from .rope import apply_rotary_pos_emb

# Imported at use-site to avoid circular imports
# from kv_cache import PerReqKVCache, PackedKVCache


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand KV heads to match Q head count for GQA.
    [B, n_kv_heads, S, D] → [B, n_kv_heads * n_rep, S, D]

    Uses expand + reshape (no data copy) like HuggingFace.
    Only used by the F.sdpa path; FlashInfer handles GQA natively.
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
        attention_mask: torch.Tensor | None,  # [B, 1, q_len, kv_len] additive (F.sdpa path)
        kv_cache=None,                        # PerReqKVCache | PackedKVCache | None
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

        # ── 4. Backend dispatch ───────────────────────────────────────────
        #
        # Both paths call kv_cache.update() to get the KV data for this layer.
        # This layer then decides which attention kernel to use.
        #
        #   PackedKVCache  → update() packs ragged KV → FlashInfer kernel
        #   PerReqKVCache  → update() returns accumulated rectangular KV → F.sdpa
        #   None (no cache) → skip update, use q/k/v as-is → F.sdpa
        #
        if kv_cache is not None and hasattr(kv_cache, "wrapper"):
            # ── FlashInfer path (PackedKVCache, decode B=N) ───────────────
            # Reshape q/k/v from [B, heads, 1, dim] → [B, heads, dim]
            # (FlashInfer NHD layout: tokens × heads × dim; q_len=1 per req)
            q_fi = q.squeeze(2)   # [B, n_q_heads,  head_dim]
            k_fi = k.squeeze(2)   # [B, n_kv_heads, head_dim]
            v_fi = v.squeeze(2)   # [B, n_kv_heads, head_dim]

            # update() packs historical KV + new token into a ragged tensor.
            # Returns (k_packed, v_packed): [total_kv_tokens, n_kv_heads, head_dim]
            k_packed, v_packed = kv_cache.update(self.layer_idx, k_fi, v_fi)

            # FlashInfer handles GQA natively (no repeat_kv needed).
            # kv_cache.wrapper already has the plan (begin_forward was called
            # once before the 28-layer forward pass in model_runner.decode_step).
            # Output: [B, n_q_heads, head_dim]
            attn_out = kv_cache.wrapper.forward(q_fi, k_packed, v_packed)

            # Re-add the seq dim so the merge-heads step below is unchanged.
            # [B, n_q_heads, head_dim] → [B, n_q_heads, 1, head_dim]
            attn_out = attn_out.unsqueeze(2)

        else:
            # ── F.sdpa path (PerReqKVCache or no cache, prefill B=1) ──────
            # update() appends new k/v and returns the full accumulated cache.
            if kv_cache is not None:
                k, v = kv_cache.update(self.layer_idx, k, v)

            # GQA: expand KV heads to match Q head count.
            # [B, 8, kv_len, 128] → [B, 16, kv_len, 128]  (no data copy)
            k = repeat_kv(k, self.num_kv_groups)
            v = repeat_kv(v, self.num_kv_groups)

            # attention_mask is additive (0 = attend, -inf = mask).
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                scale=self.scale,
            )  # [B, n_heads, q_len, head_dim]

        # ── 5. Merge heads and output projection ─────────────────────────
        # [B, n_heads, q_len, head_dim] → [B, q_len, hidden_size]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, q_len, -1)
        return self.o_proj(attn_out)
