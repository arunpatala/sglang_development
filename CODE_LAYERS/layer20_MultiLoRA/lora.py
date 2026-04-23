"""
Minimal single-adapter LoRA for Layer 20.

Loads one LoRA adapter from disk at startup. Each forward pass receives a
lora_mask [B, q_len, 1] (1.0 = apply adapter, 0.0 = base model) so that a
mixed batch of LoRA and base-model requests is handled in one forward pass:

    output = base_output + (x @ A.T) @ B.T * scaling * mask

No pool management, no eviction, no segmented GEMM — those belong to the full
multi-LoRA implementation.

Adapter checkpoint format (HuggingFace PEFT):
    base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight  [rank, hidden]
    base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight  [out_dim, rank]
    ...
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Module names we support LoRA on.
_SUPPORTED_MODULES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


def _resolve_adapter_path(adapter_path: str) -> Path:
    path = Path(adapter_path)
    if path.is_dir() and (path / "adapter_config.json").exists():
        return path
    logger.info(f"Resolving HF Hub adapter: {adapter_path}")
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    from huggingface_hub import snapshot_download
    resolved = snapshot_download(adapter_path, local_files_only=offline)
    return Path(resolved)


class LoRAAdapter:
    """
    Holds A and B weight matrices for one LoRA adapter on GPU.

    A_weights[layer_idx][module_name] : Tensor [rank, in_dim]
    B_weights[layer_idx][module_name] : Tensor [out_dim, rank]
    scaling                           : float  (lora_alpha / r)
    """

    def __init__(
        self,
        adapter_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> None:
        path = _resolve_adapter_path(adapter_path)
        logger.info(f"Loading LoRA adapter from: {path}")

        # ── Read adapter_config.json ─────────────────────────────────────
        with open(path / "adapter_config.json") as f:
            cfg = json.load(f)

        self.r       = cfg["r"]
        self.alpha   = cfg.get("lora_alpha", self.r)
        self.scaling = self.alpha / self.r
        target_modules = set(cfg.get("target_modules", _SUPPORTED_MODULES))
        self.target_modules = target_modules & _SUPPORTED_MODULES

        logger.info(
            f"Adapter config: r={self.r}  alpha={self.alpha}  "
            f"scaling={self.scaling:.4f}  modules={sorted(self.target_modules)}"
        )

        # ── Load weights ──────────────────────────────────────────────────
        self.A_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        self.B_weights: Dict[int, Dict[str, torch.Tensor]] = {}

        self._load_weights(path, dtype, device)
        logger.info(
            f"LoRA adapter ready: {len(self.A_weights)} layers loaded  "
            f"device={device}"
        )

    def _load_weights(
        self, path: Path, dtype: torch.dtype, device: str
    ) -> None:
        """Stream weights from adapter_model.safetensors or adapter_model.bin."""
        weight_file = path / "adapter_model.safetensors"
        if weight_file.exists():
            from safetensors import safe_open
            def _iter():
                with safe_open(str(weight_file), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        yield key, f.get_tensor(key).to(dtype)
        else:
            weight_file = path / "adapter_model.bin"
            def _iter():
                ckpt = torch.load(str(weight_file), map_location="cpu")
                for key, tensor in ckpt.items():
                    yield key, tensor.to(dtype)

        for name, tensor in _iter():
            # Expected key format:
            # base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight
            # base_model.model.model.layers.{i}.mlp.gate_proj.lora_A.weight
            parts = name.split(".")
            if "lora_A" not in parts and "lora_B" not in parts:
                continue

            # Find layer index
            try:
                layers_idx = parts.index("layers")
                layer_idx  = int(parts[layers_idx + 1])
            except (ValueError, IndexError):
                continue

            # Find module name
            module_name = None
            for mod in self.target_modules:
                if mod in parts:
                    module_name = mod
                    break
            if module_name is None:
                continue

            ab = "A" if "lora_A" in parts else "B"
            t  = tensor.to(device)

            if ab == "A":
                self.A_weights.setdefault(layer_idx, {})[module_name] = t
            else:
                self.B_weights.setdefault(layer_idx, {})[module_name] = t

    def has_layer(self, layer_idx: int, module_name: str) -> bool:
        return (
            layer_idx in self.A_weights
            and module_name in self.A_weights[layer_idx]
            and layer_idx in self.B_weights
            and module_name in self.B_weights[layer_idx]
        )

    def apply(
        self,
        x: torch.Tensor,
        layer_idx: int,
        module_name: str,
    ) -> Optional[torch.Tensor]:
        """
        Compute the LoRA delta for one linear projection.

        x    : [..., in_dim]
        A    : [rank, in_dim]
        B    : [out_dim, rank]
        Returns [..., out_dim] = (x @ A.T) @ B.T * scaling
        Returns None if this layer/module is not targeted.
        """
        if not self.has_layer(layer_idx, module_name):
            return None
        A = self.A_weights[layer_idx][module_name]   # [rank, in_dim]
        B = self.B_weights[layer_idx][module_name]   # [out_dim, rank]
        return (x @ A.T) @ B.T * self.scaling
