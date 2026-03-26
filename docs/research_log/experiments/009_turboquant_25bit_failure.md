# 009 — TurboQuant 2.5-bit: Failure Mode on Small Models

**Date:** 2026-03-26
**Hardware:** Apple M1, 16 GB unified memory, 68.25 GB/s theoretical bandwidth
**Backend:** MLX
**Model:** `mlx-community/Llama-3.2-3B-Instruct-4bit`
**Params:** 2.82B logical, 1.81 GB weights
**Method:** TurboQuant 2.5-bit — outlier splitting (outliers at 3-bit, regular at 2-bit)

---

## Setup

Most aggressive compression tested. At 2.5 bits, the MSE stage uses only 1-bit
(sign) quantization for regular channels — effectively the QJL residual is
operating on a very coarse MSE estimate.

## TurboQuant 2.5-bit Decode

| KV Cache | tok/s | ms/tok | vs Normal | Compression |
|----------|-------|--------|-----------|-------------|
| 47 | 14.4 | 69.6 | 0.55x | 6.4x |
| 223 | 14.1 | 70.7 | 0.54x | 6.4x |

## Quality Check — FAILED

**Prompt:** "Explain what gravity is in one paragraph:"

**TurboQuant 2.5-bit output:**
> 1-paragraph The paragraph you paragraph paragraph paragraph paragraph
> The paragraph is a paragraph, a paragraph paragraph paragraph a paragraph,
> paragraph paragraph paragraph. We need to do, the, the, the, the, the, the...

**Verdict:** Garbage. Repetitive, non-semantic output with degenerate token loops.

## Analysis

At 2.5 bits for a 3B model with head_dim=128:
- **Regular channels (96 of 128):** 2-bit = 1-bit MSE + 1-bit QJL. The 1-bit MSE
  stage has only 2 centroids (±√(2/πd)), which is barely better than keeping just
  the sign. The QJL residual can't compensate for this much quantization error.
- **MSE distortion at 1 bit:** D_mse ≈ 0.36 (per coordinate), vs 0.117 at 2 bits.
  That's 3x more error in the majority of channels.
- **Attention score corruption:** With 75% of channels at 1-bit MSE precision,
  the Q·K scores are too noisy for softmax to recover meaningful attention patterns.

The paper reports 2.5-bit working on Llama-3.1-8B-Instruct with "marginal quality
degradation." The difference is likely head_dim (128 both, but 8B has more layers
and attention heads providing redundancy) and the fact that their evaluation used
benchmarks (LongBench), not open-ended generation which is more sensitive to
attention score accuracy.

## Findings

1. **2.5-bit is too aggressive for 3B models.** The quantization error in Q·K
   scoring exceeds the model's ability to compensate through redundancy.

2. **Failure mode is degenerate repetition**, not random noise. The model locks
   onto high-probability tokens ("the", "paragraph") because noisy attention
   scores flatten the distribution, causing greedy decode to loop.

3. **Minimum viable bit-width depends on model size.** Larger models (8B+) have
   more attention heads providing redundancy against per-head quantization noise.
   For 3B models on M1, 3-bit is the practical floor.
