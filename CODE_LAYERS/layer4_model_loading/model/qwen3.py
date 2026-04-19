"""
Qwen3ForCausalLM — Layer 4A: our config + weight loading, HF forward pass.

Layer 4A introduces two things compared to Layer 3:
  1. Qwen3Config   — a plain dataclass replacing PretrainedConfig.
  2. from_pretrained / load_weights — our own config-reading and weight-
     streaming pipeline, replacing AutoModelForCausalLM.from_pretrained.

The forward computation is still HuggingFace's Qwen3ForCausalLM under the
hood (self._model).  Layer 4B replaces self._model with our own Qwen3Model
(RMSNorm, RoPE, MLP, Attention, DecoderLayer) and switches the KV cache to
our in-place KVCache object.

Progression:
  Layer 3  → HF loads + HF forward + HF past_key_values
  Layer 4A → we load  + HF forward + HF past_key_values   ← this file
  Layer 4B → we load  + our forward + our in-place KVCache
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn

from .config import Qwen3Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: resolve "Qwen/Qwen3-0.6B" → local directory
# ---------------------------------------------------------------------------

def _resolve_model_path(model_path: str) -> Path:
    """Accept a local directory or a HuggingFace Hub model ID."""
    path = Path(model_path)
    if path.is_dir() and (path / "config.json").exists():
        return path

    logger.info(f"Resolving HF Hub model: {model_path}")
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_path, local_files_only=offline))


# ---------------------------------------------------------------------------
# Qwen3ForCausalLM
# ---------------------------------------------------------------------------

class Qwen3ForCausalLM(nn.Module):
    """
    Layer 4A: our from_pretrained and load_weights; HF model for forward.

    Call site in model_runner.py is identical to Layer 3 except:
      - model init:    Qwen3ForCausalLM.from_pretrained(path, dtype=...)
      - forward call:  logits, past_kv = model(ids, attention_mask=...,
                           past_key_values=past_kv, position_ids=...)

    In Layer 4B the call site stays the same but self._model is replaced
    with our hand-written Qwen3Model and past_key_values becomes kv_cache.
    """

    def __init__(self, config: Qwen3Config, hf_model: nn.Module) -> None:
        super().__init__()
        self.config  = config
        self._model  = hf_model   # HF's Qwen3ForCausalLM; replaced in 4B

    # ------------------------------------------------------------------
    # Forward — delegates to HF; returns (logits, past_key_values)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,                        # [B, q_len]
        attention_mask: torch.Tensor | None = None,     # [B, kv_len]
        past_key_values=None,                           # HF cache or None
        position_ids: torch.Tensor | None = None,       # [B, q_len]
    ):                                                  # → (logits, past_kv)
        out = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True,
        )
        return out.logits, out.past_key_values          # [B, q_len, vocab], kv

    # ------------------------------------------------------------------
    # load_weights — SGLang-style iterator
    # ------------------------------------------------------------------

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> None:
        """
        Stream (name, tensor) pairs into self._model's parameters.

        Keys in model.safetensors match HF's named_parameters() exactly
        (e.g. "model.layers.0.self_attn.q_proj.weight"), so a simple
        name-lookup + copy_ is sufficient.

        The lm_head.weight key is skipped when tie_word_embeddings=True
        because HF already tied it to embed_tokens.weight at construction
        time — copying embed_tokens.weight is enough to update both.
        """
        params = dict(self._model.named_parameters())
        loaded: set[str] = set()

        for name, tensor in weights:
            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                continue                         # already tied — skip
            if name in params:
                params[name].data.copy_(tensor)
                loaded.add(name)
            else:
                logger.debug(f"Skipping unknown weight: {name}")

        logger.info(f"Loaded {len(loaded)} weight tensors")

    # ------------------------------------------------------------------
    # from_pretrained — replaces AutoModelForCausalLM.from_pretrained
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "Qwen3ForCausalLM":
        """
        1. Resolve model_path → local directory (HF hub or local)
        2. Read config.json → Qwen3Config  (our dataclass, not PretrainedConfig)
        3. Build HF model skeleton from their config (empty / random weights)
        4. Cast to dtype BEFORE copying weights  (one copy_ op, not two)
        5. Stream weights from model.safetensors one tensor at a time
        6. Move to CUDA, set eval mode
        """
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForCausalLM

        model_dir = _resolve_model_path(model_path)
        logger.info(f"Loading from: {model_dir}")

        # Step 2 — our config
        config = Qwen3Config.from_json(model_dir / "config.json")
        logger.info(
            f"Config: {config.num_hidden_layers}L "
            f"h={config.hidden_size} "
            f"heads={config.num_attention_heads}Q/{config.num_key_value_heads}KV"
        )

        # Step 3 — HF model skeleton (architecture not ours yet)
        hf_config = AutoConfig.from_pretrained(model_dir)
        hf_model  = AutoModelForCausalLM.from_config(hf_config)
        model     = cls(config, hf_model)

        # Step 4 — cast before copy
        model = model.to(dtype)

        # Step 5 — stream weights
        weights_path = model_dir / "model.safetensors"
        logger.info(f"Reading weights: {weights_path}")

        def _iter():
            with safe_open(str(weights_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key).to(dtype)

        model.load_weights(_iter())

        # Step 6
        model = model.to("cuda").eval()
        logger.info(
            f"Model ready  "
            f"GPU={torch.cuda.memory_allocated() / 1024**2:.0f} MB"
        )
        return model
