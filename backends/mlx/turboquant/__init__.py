"""
TurboQuant KV cache quantization for MLX.

Implements the TurboQuant algorithm (Zandieh et al., ICLR 2026) for
compressing the KV cache during inference. Two-stage approach:
  Stage 1: Random rotation + Lloyd-Max scalar quantizer (b-1 bits)
  Stage 2: 1-bit QJL (Quantized Johnson-Lindenstrauss) on the residual

Computes attention scores directly on compressed state via fused Metal
kernels — no full dequantization during decode.

Reference: arXiv:2504.19874
"""

from .cache import TurboQuantKVCache, make_turboquant_cache

__all__ = ["TurboQuantKVCache", "make_turboquant_cache"]
