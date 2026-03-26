# 010 — TurboQuant Rotation Overhead Analysis

**Date:** 2026-03-26
**Hardware:** Apple M1, 16 GB unified memory, 68.25 GB/s theoretical bandwidth
**Backend:** MLX
**Model:** `mlx-community/Llama-3.2-3B-Instruct-4bit`

---

## Motivation

All TurboQuant configurations showed decode slowdown vs normal KV cache (0.21–0.61x).
This experiment analyzes why the paper's "8x speedup" doesn't materialize on M1.

## Cost Breakdown

TurboQuant adds per-token overhead at two points:

### 1. Quantization (on write — each new KV token)

Per vector (head_dim=128):
- Rotation: x @ Pi^T — 128×128 matmul = 16,384 FLOPs
- Centroid lookup: argmin over codebook — 128 × 2^b comparisons
- QJL projection: r @ S^T — 128×128 matmul = 16,384 FLOPs
- **Total: ~33K FLOPs per vector**

For Llama 3.2 3B: 28 layers × 8 KV heads × 2 (K+V) = 448 vectors per token.
**Total quantization cost: ~15M FLOPs per new token.**

### 2. Dequantization (on read — compatibility path)

Same rotation cost in reverse, applied to all cached tokens:
- Centroid lookup from packed indices
- Inverse rotation: centroids @ Pi — 128×128 matmul
- QJL reconstruction: signs @ S — 128×128 matmul

This runs for ALL cached tokens every decode step because mlx_lm's SDPA
requires full fp16 K,V tensors. Cost scales as O(T × layers × heads × d²).

### The paper's solution

The paper avoids dequantization entirely by computing Q·K scores directly on
packed data via fused GPU kernels. The rotation is applied to the query instead:
```
score = <q @ Pi^T, codebook[indices]> + scale * <q @ S^T, signs>
```
This is O(T × d) per head — no d×d matmul on the KV side. Our Metal kernels
implement this but the current integration still uses the dequantization path
for compatibility with mlx_lm's standard attention.

## Speed vs Normal Decode

| KV Cache | Normal (tok/s) | TurboQuant (tok/s) | Overhead | Cause |
|----------|---------------|-------------------|----------|-------|
| 47 | 27.1 | 16.5 | 1.6x | Quantize + dequant 47 tokens |
| 223 | 26.4 | 9.6 | 2.8x | Dequant 223+ tokens per step |
| 465 | 26.4 | 5.5 | 4.8x | Dequant 465+ tokens per step |

Overhead grows linearly with cached tokens — confirming dequantization is the bottleneck.

## When would TurboQuant be faster?

The fused kernel path (no dequantization) would be faster when:
1. KV cache memory read dominates inference time (very long contexts, 32K+)
2. Fused kernels avoid the d×d matmul by pre-transforming queries instead
3. Batch size > 1 so memory savings translate to throughput

On H100 with 3.35 TB/s HBM bandwidth, reading a 70B model's full KV cache
at 128K context is ~50GB. At 3-bit TurboQuant that's ~9GB — a 5.5x bandwidth
reduction that directly translates to throughput improvement. On M1 with a 3B
model at 500 tokens, the KV cache is ~55MB — insignificant vs 1.81GB weights.

## Findings

1. **The rotation matrix multiply is the dominant overhead** — not codebook lookup,
   not bitpacking, not QJL. Two 128×128 matmuls per vector, 448 vectors per token.

2. **Dequantization scales linearly with context length.** This is fatal for the
   compatibility path. The fused kernel path (Metal kernels operating directly on
   packed bits) avoids this entirely.

3. **TurboQuant is a serving-scale optimization.** It trades compute for memory.
   This trade is profitable when memory bandwidth is the bottleneck (large models,
   long contexts, batched serving). For single-user local inference with short
   contexts, fp16 KV cache is strictly better.
