"""
Rotary Position Embedding (RoPE).

Math mirrors HuggingFace's Qwen3RotaryEmbedding + apply_rotary_pos_emb
(modeling_qwen3.py L86-181).  Simplified to the standard "default" rope
type (no dynamic scaling, no yarn) since Qwen3-0.6B uses rope_theta only.

How RoPE works:
  1.  Precompute inv_freq = 1 / (theta ^ (2i / dim))  for i in 0..dim/2-1
  2.  At each forward pass:
        freqs = inv_freq ⊗ position_ids            [B, q_len, dim/2]
        emb   = cat(freqs, freqs)                  [B, q_len, dim]
        cos, sin = emb.cos(), emb.sin()
  3.  Rotate q and k:
        rotate_half splits each vector into two halves and swaps with sign:
            [-x2, x1]  where x = [x1, x2]
        q_rotated = q * cos + rotate_half(q) * sin

Why this encodes relative position:
  The dot product Q_i · K_j ends up depending only on (i - j), so the
  attention logit naturally encodes the relative distance.

Extensibility:
  To add YaRN / dynamic NTK scaling (as SGLang supports via get_rope()),
  override the forward() to recompute inv_freq at runtime.
"""

import torch
import torch.nn as nn

from .config import Qwen3Config


class RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        dim = config.head_dim
        theta = config.rope_theta

        # inv_freq[i] = 1 / (theta ^ (2i / dim))  shape [dim/2]
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,           # [B, q_len, hidden] — used for device/dtype only
        position_ids: torch.Tensor, # [B, q_len]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, q_len = position_ids.shape

        # Outer product: [B, dim/2, 1] × [B, 1, q_len] → [B, q_len, dim/2]
        inv = self.inv_freq[None, :, None].float().expand(B, -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)         # [B, q_len, dim/2]

        emb = torch.cat([freqs, freqs], dim=-1)     # [B, q_len, dim]
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin                             # each [B, q_len, head_dim]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[-x2, x1] where x = [x1 | x2] split at dim // 2."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,    # [B, n_heads,    q_len, head_dim]
    k: torch.Tensor,    # [B, n_kv_heads, q_len, head_dim]
    cos: torch.Tensor,  # [B, q_len, head_dim]
    sin: torch.Tensor,  # [B, q_len, head_dim]
) -> tuple[torch.Tensor, torch.Tensor]:
    # Unsqueeze to broadcast over the head dimension.
    cos = cos.unsqueeze(1)  # [B, 1, q_len, head_dim]
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
