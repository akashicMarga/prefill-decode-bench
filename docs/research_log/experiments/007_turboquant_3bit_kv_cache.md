# 007 — TurboQuant 3-bit KV Cache Quantization

**Date:** 2026-03-26
**Hardware:** Apple M1, 16 GB unified memory, 68.25 GB/s theoretical bandwidth
**Backend:** MLX
**Model:** `mlx-community/Llama-3.2-3B-Instruct-4bit`
**Params:** 2.82B logical, 1.81 GB weights
**Method:** TurboQuant (arXiv:2504.19874) — 3-bit KV cache (2-bit Lloyd-Max MSE + 1-bit QJL residual)

---

## Setup

Implemented TurboQuant as described in the ICLR 2026 paper:
- **Keys:** ProdCodec — rotation + 2-bit Lloyd-Max + 1-bit QJL residual (unbiased inner products)
- **Values:** MSECodec — rotation + 3-bit Lloyd-Max (MSE-optimal reconstruction)
- **Codebook:** Lloyd-Max centroids computed on the exact Beta distribution (not Gaussian approximation)
- **Storage:** Bit-packed uint32 with Metal kernel fast path for pack/unpack

Prefill runs through normal fp16 cache, then KV state is quantized to TurboQuant for decode.

## Normal Decode Baseline

| KV Cache | tok/s | ms/tok | BW (GB/s) | BW Util | Peak Mem |
|----------|-------|--------|-----------|---------|----------|
| 47 | 27.1 | 36.9 | 49.2 | 72% | 1.85 GB |
| 223 | 26.4 | 37.9 | 48.4 | 71% | 1.89 GB |
| 465 | 26.4 | 37.9 | 49.2 | 72% | 1.99 GB |

## TurboQuant 3-bit Decode

| KV Cache | tok/s | ms/tok | vs Normal | Compression | Peak Mem |
|----------|-------|--------|-----------|-------------|----------|
| 47 | 16.5 | 60.6 | 0.61x | 5.3x | 1.89 GB |
| 223 | 9.6 | 104.4 | 0.36x | 5.3x | 1.98 GB |
| 465 | 5.5 | 180.8 | 0.21x | 5.3x | 2.17 GB |

## Quality Check

**Prompt:** "Explain what gravity is in one paragraph:"

**Normal KV:**
> Gravity is a fundamental force of nature that attracts two objects with mass towards each other. It is a universal force that affects everything with mass or energy, from the smallest subatomic particles to the largest galaxies. Gravity is what keeps planets in orbit around their stars...

**TurboQuant 3-bit:**
> Gravity is the force that attracts objects towards each other, and it is a universal force that affects everything in the universe. It is a fundamental force that is present in all matter and energy, and it is responsible for the curvature of space-time...

Both outputs are coherent, factually correct, and well-structured. Different phrasing but equivalent quality.

## Findings

1. **5.3x KV cache compression at 3 bits** with coherent output. The paper claims quality neutrality at 3.5 bits; 3 bits is on the boundary but works well for this model.

2. **Decode is slower, not faster.** 0.21–0.61x of normal speed. The paper's "8x speedup" requires custom CUDA kernels on H100 — not achievable via Python-level dequantization on M1. The dominant cost is the d×d rotation matrix multiply (16K FLOPs per vector at head_dim=128) that runs per layer, per head, every token.

3. **Speed degrades with KV size** because dequantization cost scales with the number of accumulated tokens being concatenated into the growing buffer. At KV=47 the overhead is manageable (0.61x); at KV=465 it's severe (0.21x).

4. **Memory savings are negligible at these context lengths.** KV=465 stores ~55MB of KV cache vs 1.81GB of model weights. Compressing 55MB by 5.3x saves 44MB — irrelevant when the model itself dominates memory. TurboQuant matters at 16K+ tokens where KV cache rivals model size.
