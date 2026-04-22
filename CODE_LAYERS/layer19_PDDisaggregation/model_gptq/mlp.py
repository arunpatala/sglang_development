"""
Qwen3MLP — SwiGLU feed-forward network with 4-bit GPTQ weights.

Forward pass is identical to the fp16 version:
    output = down_proj(silu(gate_proj(x)) * up_proj(x))

The only change: gate_proj / up_proj / down_proj are GPTQLinear
instead of nn.Linear.  Their weight tensors are 4-bit packed int32
loaded from the GPTQ checkpoint.

Weight shapes for Qwen3-0.6B:
    gate_proj qweight: [1024 // 8, 3072] = [128, 3072]   int32
    up_proj   qweight: [128, 3072]                         int32
    down_proj qweight: [3072 // 8, 1024] = [384, 1024]   int32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3Config
from .gptq_linear import GPTQLinear


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config, bits: int = 4, group_size: int = 128) -> None:
        super().__init__()
        self.gate_proj = GPTQLinear(config.hidden_size, config.intermediate_size, bits, group_size)
        self.up_proj   = GPTQLinear(config.hidden_size, config.intermediate_size, bits, group_size)
        self.down_proj = GPTQLinear(config.intermediate_size, config.hidden_size, bits, group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
