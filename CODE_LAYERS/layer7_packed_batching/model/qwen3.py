"""
Qwen3Model + Qwen3ForCausalLM — the top-level model classes.

Qwen3Model:
    embed_tokens → 28 × Qwen3DecoderLayer → final RMSNorm
    Precomputes RoPE (cos, sin) once per forward pass and distributes to
    all layers (vs. recomputing per-layer as in some older implementations).

Qwen3ForCausalLM:
    Qwen3Model → lm_head (tied weights)
    Owns from_pretrained() and load_weights() — the two methods that replace
    AutoModelForCausalLM.from_pretrained().

load_weights() design (SGLang-style):
    Takes an Iterable[Tuple[str, Tensor]], mirrors the signature used in
    sglang/srt/models/qwen3.py L592.  This makes future extensions
    (e.g. adding stacked_params_mapping for fused QKV) a local change
    in this one method without touching the rest of the model.

Tied weights:
    Qwen3-0.6B sets tie_word_embeddings=True, meaning lm_head.weight and
    embed_tokens.weight are the same tensor.  After load_weights() we
    point lm_head.weight at the embedding matrix so they share memory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn

from .config import AttnBackend, Qwen3Config
from .decoder_layer import Qwen3DecoderLayer
from .norm import RMSNorm
from .rope import RotaryEmbedding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: resolve "Qwen/Qwen3-0.6B" → local directory
# ---------------------------------------------------------------------------

def _resolve_model_path(model_path: str) -> Path:
    """
    Accept either a local directory or a HuggingFace Hub model ID.
    Returns the path to a directory containing config.json + model.safetensors.
    """
    path = Path(model_path)
    if path.is_dir() and (path / "config.json").exists():
        return path

    logger.info(f"Resolving HF Hub model: {model_path}")
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    from huggingface_hub import snapshot_download
    resolved = snapshot_download(model_path, local_files_only=offline)
    return Path(resolved)


# ---------------------------------------------------------------------------
# Qwen3Model
# ---------------------------------------------------------------------------

class Qwen3Model(nn.Module):
    """
    The backbone: embedding + 28 transformer layers + final norm.
    Does NOT include the language model head.
    """

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(
        self,
        input_ids:    torch.Tensor,                # [B, q_len]
        forward_batch=None,                        # ForwardBatch | None
        position_ids: torch.Tensor | None = None,  # [B, q_len] explicit positions
    ) -> torch.Tensor:
        B, q_len = input_ids.shape
        kv_cache = forward_batch.kv_cache if forward_batch is not None else None
        past_len = kv_cache.get_seq_length() if kv_cache is not None else 0

        # ── Token embeddings ──────────────────────────────────────────────
        hidden = self.embed_tokens(input_ids)  # [B, q_len, hidden]

        # ── Position IDs and RoPE ─────────────────────────────────────────
        # Caller supplies position_ids for batched/padded inputs.
        # Fall back to sequential positions only for B=1 no-cache calls.
        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + q_len, device=input_ids.device
            ).unsqueeze(0).expand(B, -1)                 # [B, q_len]
        cos, sin = self.rotary_emb(hidden, position_ids)  # each [B, q_len, head_dim]

        # ── 28 × Decoder layers ───────────────────────────────────────────
        # Mask construction moved into the backend (backend.py:build_additive_mask)
        # so the FlashInfer path never builds a mask it does not need.
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, forward_batch)

        return self.norm(hidden)  # [B, q_len, hidden]


# ---------------------------------------------------------------------------
# Qwen3ForCausalLM
# ---------------------------------------------------------------------------

class Qwen3ForCausalLM(nn.Module):
    """
    Full language model: Qwen3Model + lm_head.

    Entry points:
        Qwen3ForCausalLM.from_pretrained(model_path)  — loads weights
        model(input_ids, attention_mask, kv_cache)    — returns logits
    """

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model  = Qwen3Model(config)
        # lm_head projects hidden → vocab.  Bias=False matches Qwen3.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids:    torch.Tensor,                # [B, q_len]
        forward_batch=None,                        # ForwardBatch | None
        position_ids: torch.Tensor | None = None,  # [B, q_len]
    ) -> torch.Tensor:                             # [B, q_len, vocab_size]
        hidden = self.model(input_ids, forward_batch, position_ids)
        return self.lm_head(hidden)

    # ------------------------------------------------------------------ #
    # Weight loading  (SGLang-style: takes Iterable[(name, tensor)])
    # ------------------------------------------------------------------ #

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> None:
        """
        Copy pre-trained tensors into our parameters.

        The weight keys in model.safetensors match our module hierarchy
        exactly (e.g. "model.layers.0.self_attn.q_proj.weight"), so a
        simple name-lookup + copy_ is all we need.

        Extensibility: to add fused QKV like SGLang, insert a remapping
        table here that routes "q_proj" → a slice of "qkv_proj".
        """
        params = dict(self.named_parameters())
        loaded: set[str] = set()

        for name, tensor in weights:
            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                # Will be tied to embed_tokens.weight after the loop.
                continue
            if name in params:
                params[name].data.copy_(tensor)
                loaded.add(name)
            else:
                logger.debug(f"Skipping unknown weight: {name}")

        # Tie lm_head to the embedding matrix (saves ~600MB for large models).
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        unloaded = set(params) - loaded - {"lm_head.weight"}
        if unloaded:
            logger.warning(f"Parameters not found in checkpoint: {unloaded}")

        logger.info(f"Loaded {len(loaded)} weight tensors")

    # ------------------------------------------------------------------ #
    # from_pretrained — replaces AutoModelForCausalLM.from_pretrained
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        attn_backend: AttnBackend = AttnBackend.SDPA,
    ) -> "Qwen3ForCausalLM":
        """
        1. Resolve model_path → local directory (HF hub or local)
        2. Read config.json → Qwen3Config
        3. Build architecture (empty weights)
        4. Stream weights from model.safetensors → load_weights()
        5. Cast to dtype, move to CUDA, set eval mode
        """
        from safetensors import safe_open

        model_dir = _resolve_model_path(model_path)
        logger.info(f"Loading from: {model_dir}")

        config = Qwen3Config.from_json(model_dir / "config.json")
        config.attn_backend = attn_backend
        logger.info(
            f"Config: {config.num_hidden_layers}L "
            f"h={config.hidden_size} "
            f"heads={config.num_attention_heads}Q/{config.num_key_value_heads}KV "
            f"backend={attn_backend.value}"
        )

        # Build model on CPU with float32 (default nn init dtype).
        model = cls(config)

        # Cast to target dtype BEFORE copying weights so copy_ is same-dtype.
        model = model.to(dtype)

        # Stream weights from safetensors (one tensor at a time, no full copy).
        weights_path = model_dir / "model.safetensors"
        logger.info(f"Reading weights: {weights_path}")

        def _iter():
            with safe_open(str(weights_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key).to(dtype)

        model.load_weights(_iter())

        model = model.to("cuda").eval()
        logger.info(
            f"Model ready  "
            f"GPU={torch.cuda.memory_allocated() / 1024**2:.0f} MB"
        )
        return model
