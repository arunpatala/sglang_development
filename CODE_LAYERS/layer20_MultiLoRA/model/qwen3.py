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
import sys
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from forward_batch import ForwardBatch, ForwardMode  # noqa: E402

from .config import Qwen3Config
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
        input_ids:      torch.Tensor,                       # [B, q_len]
        attention_mask: torch.Tensor | None,                # [B, kv_len] binary (1=real, 0=pad)
        kv_cache=None,                                      # ExtendKVCtx | DecodeKVCtx | None
        position_ids:   torch.Tensor | None = None,         # [B, q_len] explicit positions
        lora_mask:      torch.Tensor | None = None,         # [B, q_len, 1]  Layer 20
        lora_adapter=None,                                  # LoRAAdapter | None  Layer 20
    ) -> torch.Tensor:
        B, q_len = input_ids.shape

        # ── Token embeddings ──────────────────────────────────────────────
        hidden = self.embed_tokens(input_ids)  # [B, q_len, hidden]

        # ── Position IDs and RoPE ─────────────────────────────────────────
        # Model-runner always supplies explicit position_ids for EXTEND/DECODE.
        # Fallback to sequential [0..q_len-1] for the NOCACHE verify path.
        if position_ids is None:
            position_ids = torch.arange(q_len, device=input_ids.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_emb(hidden, position_ids)  # each [B, q_len, head_dim]

        # ── Build ForwardBatch ────────────────────────────────────────────
        # Detect which kernel path to use from the kv_cache type.
        # All three modes share the same external model API so model_runner
        # does not need to change.
        if kv_cache is not None and hasattr(kv_cache, "extend_wrapper"):
            # Paged prefill: FlashInfer handles causal masking internally.
            forward_batch = ForwardBatch(
                mode=ForwardMode.EXTEND,
                kv_cache=kv_cache,
                attention_mask=None,
                lora_mask=lora_mask,
                lora_adapter=lora_adapter,
            )
        elif kv_cache is not None:
            # Paged decode: FlashInfer handles causal masking internally.
            forward_batch = ForwardBatch(
                mode=ForwardMode.DECODE,
                kv_cache=kv_cache,
                attention_mask=None,
                lora_mask=lora_mask,
                lora_adapter=lora_adapter,
            )
        else:
            # No KV pool: plain F.sdpa (verify_batch.py baseline).
            # Build the additive mask here so it's precomputed once for all layers.
            additive_mask = _build_additive_mask(
                attention_mask=attention_mask,
                q_len=q_len,
                kv_len=q_len,   # no past tokens in no-cache mode
                dtype=hidden.dtype,
                device=hidden.device,
            )
            forward_batch = ForwardBatch(
                mode=ForwardMode.NOCACHE,
                kv_cache=None,
                attention_mask=additive_mask,
                lora_mask=lora_mask,
                lora_adapter=lora_adapter,
            )

        # ── 28 × Decoder layers ───────────────────────────────────────────
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, forward_batch)

        return self.norm(hidden)  # [B, q_len, hidden]


def _build_additive_mask(
    attention_mask: torch.Tensor | None,
    q_len: int,
    kv_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Build a [B, 1, q_len, kv_len] additive mask:
      0     → position is attended to
      -inf  → position is masked (future token or padding)

    For q_len == 1 (decode step): the causal part is all-zero because a
    single query token can attend to every key in the cache.
    """
    NEG_INF = torch.finfo(dtype).min

    # ── Causal mask ───────────────────────────────────────────────────────
    # Upper-triangular positions are in the future → -inf.
    # For q_len=1, triu(diagonal=kv_len) is empty, so causal is all zeros.
    causal = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
    if q_len > 1:
        # diagonal = kv_len - q_len + 1 so that position i can attend to
        # positions 0..past_len+i (causal within the new tokens).
        mask_upper = torch.ones(q_len, kv_len, dtype=torch.bool, device=device)
        mask_upper = torch.triu(mask_upper, diagonal=kv_len - q_len + 1)
        causal = causal.masked_fill(mask_upper, NEG_INF)

    # [1, 1, q_len, kv_len] — broadcasts over batch and heads
    causal = causal.unsqueeze(0).unsqueeze(0)

    if attention_mask is None:
        return causal

    # ── Padding mask ──────────────────────────────────────────────────────
    # attention_mask: [B, kv_len] binary (1=real, 0=pad)
    # We need it to cover exactly kv_len columns.  During prefill kv_len ==
    # prompt_len; during decode the caller extends it by 1 each step.
    pad = attention_mask.to(dtype)                # 1.0 or 0.0
    pad = (1.0 - pad) * NEG_INF                  # 0.0 or -inf
    pad = pad[:, None, None, :]                   # [B, 1, 1, kv_len]

    return causal + pad                           # [B, 1, q_len, kv_len]


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
        input_ids:      torch.Tensor,                      # [B, q_len]
        attention_mask: torch.Tensor | None = None,        # [B, kv_len]
        kv_cache=None,                                     # KVCache | None
        position_ids:   torch.Tensor | None = None,        # [B, q_len]
        lora_mask=None,                                    # [B, q_len, 1]  Layer 20
        lora_adapter=None,                                 # LoRAAdapter | None
    ) -> torch.Tensor:                                     # [B, q_len, vocab_size]
        hidden = self.model(
            input_ids, attention_mask, kv_cache, position_ids,
            lora_mask=lora_mask, lora_adapter=lora_adapter,
        )
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
        logger.info(
            f"Config: {config.num_hidden_layers}L "
            f"h={config.hidden_size} "
            f"heads={config.num_attention_heads}Q/{config.num_key_value_heads}KV"
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
