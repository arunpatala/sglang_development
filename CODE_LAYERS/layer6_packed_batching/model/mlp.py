"""
Qwen3MLP — SwiGLU feed-forward network.

Identical to HuggingFace's Qwen3MLP (modeling_qwen3.py L70-83):
    output = down_proj(silu(gate_proj(x)) * up_proj(x))

SwiGLU: instead of a single up-projection + activation, there are two
parallel projections.  gate_proj provides a learned gating signal passed
through SiLU.  up_proj provides the "content" vector.  Element-wise
multiplication acts as a soft gate before projecting back down.

Weight shapes for Qwen3-0.6B:
    gate_proj: [3072, 1024]   (intermediate × hidden)
    up_proj:   [3072, 1024]
    down_proj: [1024, 3072]

Extensibility:
    SGLang fuses gate_proj + up_proj into a single gate_up_proj matrix
    [6144, 1024] and slices it at runtime, reducing kernel-launch overhead.
    To replicate: merge the two Linear layers into one here and update
    load_weights() with the stacked_params_mapping table.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3Config


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
