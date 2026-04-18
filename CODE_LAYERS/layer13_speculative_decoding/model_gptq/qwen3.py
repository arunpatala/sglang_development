"""
Qwen3Model + Qwen3ForCausalLM — GPTQ-quantised version.

Layer 12 changes vs Layer 11
─────────────────────────────
1. Constructor accepts bits / group_size (read from quantize_config.json by
   from_pretrained; defaults to 4 / 128 matching JunHowie/Qwen3-0.6B-GPTQ-Int4).
   These are threaded down through Qwen3Model → Qwen3DecoderLayer →
   Qwen3Attention / Qwen3MLP so they construct GPTQLinear instead of nn.Linear.

2. load_weights() searches BOTH named_parameters() AND named_buffers().
   GPTQLinear stores its four tensors (qweight, scales, qzeros, g_idx) as
   buffers, not parameters.  Without named_buffers() they would silently be
   skipped, leaving the model with garbage weights.

3. from_pretrained() has two fixes:
   a. Reads quantize_config.json to discover bits / group_size (falls back to
      4 / 128 if the file is absent).
   b. Does NOT cast int32 tensors to bfloat16 inside _iter().
      PyTorch's Module.to(dtype) already skips non-floating tensors so the
      empty int32 buffers survive the pre-load dtype cast.  The bug is in the
      safetensors iterator which did `.to(dtype)` unconditionally — calling
      `.to(bfloat16)` on an int32 tensor silently reinterprets the packed
      4-bit data as bf16 bit patterns, producing completely wrong outputs.

4. After load_weights() and before moving to CUDA, from_pretrained() calls
   GPTQLinear.prepare() on every quantised layer to run gptq_shuffle().
   gptq_shuffle permutes qweight in-place into the ExLlama v2 tile layout
   that gptq_gemm requires.  It must be called exactly once.

lm_head and embed_tokens are NOT quantised (quantize_config "lm_head": false).
They stay as nn.Linear / nn.Embedding in bfloat16.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn

from .config import Qwen3Config
from .decoder_layer import Qwen3DecoderLayer
from .gptq_linear import GPTQLinear
from .norm import RMSNorm
from .rope import RotaryEmbedding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: resolve "JunHowie/Qwen3-0.6B-GPTQ-Int4" → local directory
# ---------------------------------------------------------------------------

def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if path.is_dir() and (path / "config.json").exists():
        return path

    logger.info(f"Resolving HF Hub model: {model_path}")
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    from huggingface_hub import snapshot_download
    resolved = snapshot_download(model_path, local_files_only=offline)
    return Path(resolved)


def _read_gptq_config(model_dir: Path) -> tuple[int, int]:
    """
    Read bits and group_size from quantize_config.json.
    Falls back to (4, 128) if the file is absent (plain fp model).
    """
    cfg_path = model_dir / "quantize_config.json"
    if not cfg_path.exists():
        return 4, 128
    with open(cfg_path) as f:
        cfg = json.load(f)
    bits       = cfg.get("bits", 4)
    group_size = cfg.get("group_size", 128)
    logger.info(f"GPTQ config: bits={bits}, group_size={group_size}")
    return bits, group_size


# ---------------------------------------------------------------------------
# Qwen3Model
# ---------------------------------------------------------------------------

class Qwen3Model(nn.Module):
    """Embedding + 28 × Qwen3DecoderLayer + final RMSNorm."""

    def __init__(self, config: Qwen3Config, bits: int = 4, group_size: int = 128) -> None:
        super().__init__()
        self.config = config
        # embed_tokens is NOT quantised — it stays bf16
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i, bits, group_size) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        kv_cache=None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, q_len = input_ids.shape
        past_len  = kv_cache.get_seq_length() if kv_cache is not None else 0

        hidden = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + q_len, device=input_ids.device
            ).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_emb(hidden, position_ids)

        additive_mask = _build_additive_mask(
            attention_mask=attention_mask,
            q_len=q_len,
            kv_len=past_len + q_len,
            dtype=hidden.dtype,
            device=hidden.device,
        )

        for layer in self.layers:
            hidden = layer(hidden, cos, sin, additive_mask, kv_cache)

        return self.norm(hidden)


def _build_additive_mask(
    attention_mask: torch.Tensor | None,
    q_len: int,
    kv_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    NEG_INF = torch.finfo(dtype).min

    causal = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
    if q_len > 1:
        mask_upper = torch.ones(q_len, kv_len, dtype=torch.bool, device=device)
        mask_upper = torch.triu(mask_upper, diagonal=kv_len - q_len + 1)
        causal = causal.masked_fill(mask_upper, NEG_INF)
    causal = causal.unsqueeze(0).unsqueeze(0)

    if attention_mask is None:
        return causal

    pad = attention_mask.to(dtype)
    pad = (1.0 - pad) * NEG_INF
    pad = pad[:, None, None, :]
    return causal + pad


# ---------------------------------------------------------------------------
# Qwen3ForCausalLM
# ---------------------------------------------------------------------------

class Qwen3ForCausalLM(nn.Module):
    """
    Full GPTQ-quantised language model: Qwen3Model + lm_head.

    Entry points:
        Qwen3ForCausalLM.from_pretrained(model_path)  — loads GPTQ weights
        model(input_ids, attention_mask, kv_cache)    — returns logits [B, q, V]
    """

    def __init__(self, config: Qwen3Config, bits: int = 4, group_size: int = 128) -> None:
        super().__init__()
        self.config = config
        self.model  = Qwen3Model(config, bits, group_size)
        # lm_head is NOT quantised ("lm_head": false in quantize_config.json)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache=None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.model(input_ids, attention_mask, kv_cache, position_ids)
        return self.lm_head(hidden)

    # ------------------------------------------------------------------ #
    # load_weights
    # ------------------------------------------------------------------ #

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """
        Copy checkpoint tensors into our parameters AND buffers.

        Key difference from the fp16 version: GPTQLinear registers its four
        tensors (qweight, scales, qzeros, g_idx) as buffers, not parameters.
        named_parameters() does NOT enumerate buffers, so we must also call
        named_buffers() to build the lookup dict.

        Checkpoint key examples:
            model.layers.0.self_attn.q_proj.qweight   → buffer
            model.layers.0.self_attn.q_proj.scales    → buffer
            model.layers.0.self_attn.q_proj.qzeros    → buffer
            model.layers.0.self_attn.q_proj.g_idx     → buffer
            model.layers.0.input_layernorm.weight      → parameter
            model.embed_tokens.weight                  → parameter
        """
        params  = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        # Merge: buffers take precedence for GPTQ tensor names
        lookup  = {**params, **buffers}
        loaded: set[str] = set()

        for name, tensor in weights:
            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                continue
            if name in lookup:
                lookup[name].data.copy_(tensor)
                loaded.add(name)
            else:
                logger.debug(f"Skipping unknown weight: {name}")

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        unloaded = set(params) - loaded - {"lm_head.weight"}
        if unloaded:
            logger.warning(f"Parameters not found in checkpoint: {sorted(unloaded)}")

        logger.info(f"Loaded {len(loaded)} weight tensors")

    # ------------------------------------------------------------------ #
    # from_pretrained
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "Qwen3ForCausalLM":
        """
        Load a GPTQ-quantised Qwen3 model.

        Steps:
          1. Resolve model_path (HF hub or local dir).
          2. Read config.json → Qwen3Config.
          3. Read quantize_config.json → bits, group_size.
          4. Build model with GPTQLinear layers (empty buffers).
          5. model.to(dtype): casts floating-point tensors (embed_tokens,
             norms, lm_head) to bf16.  int32 buffers are unaffected.
          6. Stream weights from model.safetensors:
               - fp tensors (scales, norms, embed) → cast to dtype
               - int32 tensors (qweight, qzeros, g_idx) → keep as int32
          7. Call GPTQLinear.prepare() on every quantised layer (runs
             gptq_shuffle to permute qweight into ExLlama v2 tile layout).
          8. Move to CUDA and set eval mode.
        """
        from safetensors import safe_open

        model_dir  = _resolve_model_path(model_path)
        logger.info(f"Loading from: {model_dir}")

        config = Qwen3Config.from_json(model_dir / "config.json")
        logger.info(
            f"Config: {config.num_hidden_layers}L "
            f"h={config.hidden_size} "
            f"heads={config.num_attention_heads}Q/{config.num_key_value_heads}KV"
        )

        bits, group_size = _read_gptq_config(model_dir)

        # Step 4: build on CPU
        model = cls(config, bits=bits, group_size=group_size)

        # Step 5: stream weights exactly as stored in the checkpoint.
        # We do NOT do a global model.to(dtype) before loading because that
        # would cast the fp16 scale buffers to bf16, corrupting their bit
        # patterns (gptq_gemm reads scale bits as fp16 regardless of dtype).
        #
        # Checkpoint dtypes:
        #   qweight / qzeros / g_idx  → torch.int32   (must stay int32)
        #   scales                    → torch.float16  (must stay fp16!)
        #   embed_tokens / norms / lm_head → torch.bfloat16 (already correct)
        weights_path = model_dir / "model.safetensors"
        logger.info(f"Reading weights: {weights_path}")

        def _iter():
            with safe_open(str(weights_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)   # Keep every tensor's native dtype

        model.load_weights(_iter())

        # Step 6: cast floating PARAMETERS (embed_tokens, norms, lm_head) to the
        # target dtype.  We deliberately skip buffers so that GPTQLinear.scales
        # remains fp16 and GPTQLinear.qweight/qzeros/g_idx remain int32.
        for param in model.parameters():
            if param.is_floating_point():
                param.data = param.data.to(dtype)

        # Step 7: move to GPU — model.to("cuda") moves all tensors; buffers keep
        # their dtypes (fp16 scales, int32 qweight etc.).  gptq_shuffle is CUDA-only
        # so prepare() must be called after this.
        model = model.to("cuda").eval()

        # Step 8: prepare every GPTQLinear (gptq_shuffle permutes qweight in-place)
        n_prepared = 0
        for m in model.modules():
            if isinstance(m, GPTQLinear):
                m.prepare()
                n_prepared += 1
        logger.info(f"Called prepare() on {n_prepared} GPTQLinear layers")
        logger.info(
            f"GPTQ model ready  "
            f"GPU={torch.cuda.memory_allocated() / 1024**2:.0f} MB"
        )
        return model
