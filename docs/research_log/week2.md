# Week 2 — TurboQuant KV Cache Quantization

**Period:** March 22–28, 2026
**Hardware:** Apple M1 (16 GB unified, 68.25 GB/s theoretical bandwidth)
**Paper:** TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (Zandieh et al., ICLR 2026, arXiv:2504.19874)

---

## Experiments

| # | Date | Title | Backend | Model | Detail |
|---|------|-------|---------|-------|--------|
| 007 | 03-26 | TurboQuant 3-bit KV cache | MLX | Llama-3.2-3B-Instruct-4bit | [detail](experiments/007_turboquant_3bit_kv_cache.md) |
| 008 | 03-26 | TurboQuant 3.5-bit quality-neutral | MLX | Llama-3.2-3B-Instruct-4bit | [detail](experiments/008_turboquant_35bit_quality_neutral.md) |
| 009 | 03-26 | TurboQuant 2.5-bit failure mode | MLX | Llama-3.2-3B-Instruct-4bit | [detail](experiments/009_turboquant_25bit_failure.md) |
| 010 | 03-26 | Rotation overhead analysis | MLX | Llama-3.2-3B-Instruct-4bit | [detail](experiments/010_turboquant_rotation_bottleneck.md) |

---

## What We Did

Implemented TurboQuant from scratch based on the ICLR 2026 paper. Two iterations:

**v1 (proof of concept):** Single-file implementation. Random rotation with Gaussian
approximation for centroids. Dequantize-then-SDPA path. Validated the math works.

**v2 (refactored):** Modular package (`backends/mlx/turboquant/`) with proper
Lloyd-Max codebook on Beta PDF, bit-packed uint32 storage with Metal kernel fast
path, separate ProdCodec (keys) and MSECodec (values), fused Metal kernels for
direct scoring on compressed state, and outlier channel splitting for non-integer bits.

Also reviewed [mlx-vlm PR #858](https://github.com/Blaizzy/mlx-vlm/pull/858)
(Blaizzy's independent TurboQuant implementation, 3,077 lines) for comparison.

---

## Key Findings

1. **TurboQuant's accuracy claims hold up.** 3-bit produces coherent, correct text.
   3.5-bit is genuinely quality-neutral — output is near-identical to fp16. The
   core algorithm (random rotation → Lloyd-Max → QJL residual) works as described.

2. **The speedup claims do not hold on local hardware.** Every configuration was
   slower than normal decode (0.21–0.61x). The paper's "8x speedup" requires fused
   CUDA kernels on H100 with long contexts and batched serving — conditions that
   don't exist in single-user local inference.
   ([010](experiments/010_turboquant_rotation_bottleneck.md))

3. **The rotation matrix multiply is the bottleneck.** Two 128×128 matmuls per
   vector per head per layer, applied during both quantization (write) and
   dequantization (read). The paper sidesteps this by fusing the rotation into the
   query transform, but this requires replacing the SDPA implementation — not just
   wrapping the KV cache.

4. **2.5-bit fails on 3B models.** Output degenerates into repetitive loops.
   The 1-bit MSE stage (for 75% of channels) produces too much quantization noise
   for softmax to recover meaningful attention patterns. Minimum viable bit-width
   depends on model size — 3 bits for 3B, possibly 2.5 for 8B+.
   ([009](experiments/009_turboquant_25bit_failure.md))

5. **Non-integer bits add overhead without proportional benefit.** 3.5-bit is
   slower than 3-bit due to dual-codec overhead from outlier channel splitting.
   In practice, pick an integer bit-width.
   ([008](experiments/008_turboquant_35bit_quality_neutral.md))

6. **KV cache compression is irrelevant at short contexts.** At KV=500 tokens,
   the KV cache is ~55MB vs 1.81GB model weights. Compressing 55MB by 5.3x saves
   44MB. The optimization only matters at 16K+ tokens where KV cache approaches
   or exceeds model weight size.

7. **The paper vs implementation gap is where the hard work lives.** The algorithm
   is two pages. Integrating with mlx_lm's cache interface, handling GQA head
   repeats, bit-packing, Metal kernel codegen, online softmax chunking, and buffer
   management took 1,100+ lines across 6 modules. Blaizzy's PR is 3,077 lines —
   and that's still "far from optimal" per their own assessment.

---

## Implementation Summary

```
backends/mlx/turboquant/
├── __init__.py      — Public API
├── codebook.py      — Lloyd-Max on Beta PDF, rotation/projection matrices
├── bitpack.py       — Bit-packing with Metal kernel fast path
├── codec.py         — MSECodec (values), ProdCodec (keys), SplitCodec
├── kernels.py       — Metal kernels for fused attention scoring
├── cache.py         — TurboQuantKVCache (drop-in for mlx_lm)
```

**Key design decisions:**
- Keys use ProdCodec (unbiased inner products for Q·K scoring)
- Values use MSECodec (MSE-optimal for weighted sum reconstruction)
- Codebook computed on exact Beta distribution, not Gaussian approximation
- Quantized state stored as bit-packed uint32 NamedTuples
- Dequantized cache maintained in pre-allocated buffer for SDPA compatibility
- Metal kernels ready for fused attention bypass (not yet wired to SDPA)

---

## Results Summary

| Bit-width | Compression | Speed (KV=64) | Speed (KV=512) | Quality |
|-----------|------------|---------------|----------------|---------|
| 2.5 | 6.4x | 0.55x | 0.54x | FAIL (garbage) |
| **3** | **5.3x** | **0.61x** | **0.21x** | **PASS** |
| 3.5 | 4.6x | 0.40x | 0.30x | PASS (near-identical) |

---

## Paper Rating: 7/10

Strong theoretical contribution with clean algorithm and proven optimality bounds.
Genuinely useful for cloud serving at scale. Headline numbers (6x compression, 8x
speedup) are cherry-picked for best-case (large models, long contexts, batched
serving, custom kernels). For local inference the technique is a memory optimization
with a speed penalty. The real audience is infrastructure teams running 70B+ models
at 128K context, not end users running 3B locally.

---

## Known Issues / TODO

- [ ] Wire fused Metal kernels into SDPA path (bypass dequantization during decode)
- [ ] Benchmark at truly long contexts (8K, 16K, 32K) where KV cache pressure is real
- [ ] Test on larger model (7B+) where 2.5-bit may become viable
- [ ] Compare TurboQuant vs mlx_lm's built-in QuantizedKVCache (uniform scalar quantization)
- [ ] Profile rotation overhead in isolation to quantify the exact cost
- [ ] Test with batched inference to see if memory savings translate to throughput

---

## Hardware Tested

| Device | Chip | Memory | Bandwidth | Backends |
|--------|------|--------|-----------|----------|
| MacBook Pro | Apple M1 | 16 GB unified | 68.25 GB/s | MLX |

---

## Files Produced

```
results/
├── profile_mlx_Apple-M1_mlx-community__Llama-3.2-3B-Instruct-4bit.json
│   (includes turboquant_decode results at 3-bit, 3.5-bit)
└── *.png

backends/mlx/turboquant/
├── __init__.py
├── codebook.py
├── bitpack.py
├── codec.py
├── kernels.py
└── cache.py
```
