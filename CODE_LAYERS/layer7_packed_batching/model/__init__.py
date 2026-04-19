"""
layer4_custom_model.model — our hand-written Qwen3 implementation.

Public API (everything the server and model_runner need):
    from model import Qwen3ForCausalLM, Qwen3Config
"""

from .config import AttnBackend, Qwen3Config
from .qwen3 import Qwen3ForCausalLM

__all__ = ["AttnBackend", "Qwen3Config", "Qwen3ForCausalLM"]
