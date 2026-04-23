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

Layer 20 changes vs Layer 19:
  • __init__ now takes layer_idx so the LoRA adapter can look up per-layer
    weights.
  • forward() accepts an optional forward_batch; when forward_batch.lora_adapter
    is set, masked LoRA deltas are added after gate_proj, up_proj, and down_proj.

Extensibility:
    SGLang fuses gate_proj + up_proj into a single gate_up_proj matrix
    [6144, 1024] and slices it at runtime, reducing kernel-launch overhead.
    To replicate: merge the two Linear layers into one here and update
    load_weights() with the stacked_params_mapping table.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3Config

if TYPE_CHECKING:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from forward_batch import ForwardBatch


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        forward_batch: Optional["ForwardBatch"] = None,
    ) -> torch.Tensor:
        gate = self.gate_proj(x)
        up   = self.up_proj(x)

        # ── LoRA deltas for gate_proj and up_proj ─────────────────────────
        if forward_batch is not None and forward_batch.lora_adapter is not None:
            ada  = forward_batch.lora_adapter
            mask = forward_batch.lora_mask   # [B, q_len, 1]
            delta_gate = ada.apply(x, self.layer_idx, "gate_proj")
            delta_up   = ada.apply(x, self.layer_idx, "up_proj")
            if delta_gate is not None:
                gate = gate + delta_gate * mask
            if delta_up is not None:
                up = up + delta_up * mask

        hidden = F.silu(gate) * up
        out    = self.down_proj(hidden)

        # ── LoRA delta for down_proj ──────────────────────────────────────
        if forward_batch is not None and forward_batch.lora_adapter is not None:
            delta_down = forward_batch.lora_adapter.apply(
                hidden, self.layer_idx, "down_proj"
            )
            if delta_down is not None:
                out = out + delta_down * forward_batch.lora_mask

        return out
