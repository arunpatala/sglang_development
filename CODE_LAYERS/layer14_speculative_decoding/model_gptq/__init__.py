"""
model_gptq — GPTQ-quantised Qwen3 implementation for Layer 12.

Public API:
    from model_gptq import Qwen3ForCausalLM, Qwen3Config
"""

from .config import Qwen3Config
from .gptq_linear import GPTQLinear
from .qwen3 import Qwen3ForCausalLM

__all__ = ["Qwen3Config", "Qwen3ForCausalLM", "GPTQLinear"]
