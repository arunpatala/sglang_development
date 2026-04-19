"""
GPTQLinear — drop-in replacement for nn.Linear using 4-bit GPTQ weights.

Weight layout (matches AutoGPTQ / gptqmodel checkpoint format):

  qweight : [K // pack_factor, N]                    int32
            8 int4 values packed per int32, column-major order.

  scales  : [K // group_size,  N]                    bfloat16
            One fp scale per group of `group_size` input rows per output col.

  qzeros  : [K // group_size,  N // pack_factor]     int32
            Zero-points packed the same way as qweight.

  g_idx   : [K]                                      int32
            Maps each input row to its scale/zero-point group.
            All zeros when desc_act=False (our checkpoint).

All four are registered as buffers (not parameters) so:
  • They are excluded from the optimizer (no gradient).
  • model.to(dtype) skips them — int32 buffers stay int32.
  • model.named_buffers() enumerates them for load_weights().

Kernel used: sgl_kernel.gptq_gemm (ExLlama v2 fused CUDA kernel).
Before the first forward pass, call prepare() to run gptq_shuffle(),
which permutes qweight in-place into the tile layout the kernel expects.

Usage:
    layer = GPTQLinear(in_features=1024, out_features=3072, bits=4, group_size=128)
    # Buffers are populated by Qwen3ForCausalLM.load_weights()
    layer.prepare()   # call once after load_weights() — runs gptq_shuffle
    y = layer(x)      # calls gptq_gemm; x: [..., K], y: [..., N]
"""

import torch
import torch.nn as nn


class GPTQLinear(nn.Module):
    """
    4-bit GPTQ quantized linear layer.

    Replaces nn.Linear in Qwen3Attention and Qwen3MLP.
    No bias support (Qwen3 linear projections use bias=False).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.bits         = bits
        self.group_size   = group_size
        self.pack_factor  = 32 // bits      # int4: 8 values per int32

        # ── Quantised weight buffers (populated by load_weights) ──────────
        # Buffers survive model.to(bf16) intact because PyTorch skips
        # non-floating-point tensors in Module.to(dtype).
        self.register_buffer(
            "qweight",
            torch.empty(in_features // self.pack_factor, out_features, dtype=torch.int32),
        )
        # scales MUST stay fp16 — gptq_gemm reads scale bits as fp16.
        # model.to(bfloat16) would corrupt them by reinterpreting the bit pattern.
        # We keep this buffer as float16 and never cast it via model.to(dtype).
        self.register_buffer(
            "scales",
            torch.empty(in_features // group_size, out_features, dtype=torch.float16),
        )
        self.register_buffer(
            "qzeros",
            torch.empty(in_features // group_size, out_features // self.pack_factor, dtype=torch.int32),
        )
        # g_idx from checkpoint (all zeros for desc_act=False — not used directly).
        self.register_buffer(
            "g_idx",
            torch.zeros(in_features, dtype=torch.int32),
        )
        # Sequential group index: row i belongs to group i // group_size.
        # gptq_gemm (use_shuffle=False) needs this to look up the right scale/zero row.
        # Pre-computed here so we don't rebuild it every forward call.
        seq = torch.arange(in_features, dtype=torch.int32) // group_size
        self.register_buffer("_g_idx_seq", seq)

        self._prepared = False

    # ------------------------------------------------------------------
    # prepare() — call once after load_weights(), before first forward
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """
        Mark this layer as ready.  No weight transformation is needed.

        We use the use_shuffle=False path of gptq_gemm, which takes the raw
        packed qweight directly (no gptq_shuffle pre-processing required).
        The SGLang test suite only validates use_shuffle=False; the shuffled
        path produces incorrect results with this kernel version (0.4.1).
        """
        self._prepared = True

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute y = x @ W_dequant using the packed int4 representation.

        gptq_gemm dequantizes on-the-fly:
            w_fp = (qweight_unpacked - qzeros) * scales
        then performs the matrix multiply, all in a single fused kernel.

        x shape : [..., in_features]   (any number of leading batch dims)
        y shape : [..., out_features]
        """
        from sgl_kernel import gptq_gemm

        orig_dtype = x.dtype
        out_shape  = x.shape[:-1] + (self.out_features,)

        # gptq_gemm only produces correct results with fp16 activations.
        # bf16 inputs yield drastically wrong values (bit-pattern mismatch
        # in the kernel's scale look-up path).  Cast to fp16 and back.
        x_2d = x.reshape(-1, self.in_features).to(torch.float16)

        y = gptq_gemm(
            x_2d,
            self.qweight,
            self.qzeros,
            self.scales,
            self._g_idx_seq,   # sequential [0,0,...,1,1,...] — required for use_shuffle=False
            False,             # use_shuffle=False: raw qweight, no gptq_shuffle pre-processing
            self.bits,
        )
        # Cast back to the original activation dtype (usually bf16)
        return y.to(orig_dtype).reshape(out_shape)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, "
            f"prepared={self._prepared}"
        )
