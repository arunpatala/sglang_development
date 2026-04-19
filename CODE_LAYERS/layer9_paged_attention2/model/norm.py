"""
RMSNorm — Root Mean Square Layer Normalisation.

Identical math to HuggingFace's Qwen3RMSNorm (modeling_qwen3.py L50-67).
Used in three places:
  1. input_layernorm  (pre-attention)
  2. post_attention_layernorm (pre-MLP)
  3. model.norm  (final norm before lm_head)
  4. self_attn.q_norm / k_norm  (per Q/K head, Qwen3-specific)

Extensibility:
  To fuse with the preceding linear (as SGLang does for speed), swap this
  class for a CUDA kernel that reads the linear output and writes the normed
  result in one pass.  Everything else stays unchanged.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability, then cast back.
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * norm).to(dtype)

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"
