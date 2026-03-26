# 008 — TurboQuant 3.5-bit: Quality-Neutral KV Cache Compression

**Date:** 2026-03-26
**Hardware:** Apple M1, 16 GB unified memory, 68.25 GB/s theoretical bandwidth
**Backend:** MLX
**Model:** `mlx-community/Llama-3.2-3B-Instruct-4bit`
**Params:** 2.82B logical, 1.81 GB weights
**Method:** TurboQuant 3.5-bit — outlier channel splitting (outliers at 4-bit, regular at 3-bit)

---

## Setup

Non-integer bit-widths use the paper's outlier channel splitting strategy:
- Channels ranked by average magnitude across the prefill tokens
- Top-magnitude channels quantized at higher precision (4-bit = 3-bit MSE + 1-bit QJL)
- Remaining channels at lower precision (3-bit = 2-bit MSE + 1-bit QJL)
- Effective: (n_outlier × 4 + n_regular × 3) / dim = 3.5 bits/channel

Each group uses an independent ProdCodec (keys) or MSECodec (values) with
rotation matrices sized to the subgroup dimension.

## Normal Decode Baseline

| KV Cache | tok/s | ms/tok | BW (GB/s) | BW Util |
|----------|-------|--------|-----------|---------|
| 47 | 27.5 | 36.3 | 50.0 | 73% |
| 223 | 27.3 | 36.6 | 50.1 | 73% |

## TurboQuant 3.5-bit Decode

| KV Cache | tok/s | ms/tok | vs Normal | Compression |
|----------|-------|--------|-----------|-------------|
| 47 | 10.9 | 91.5 | 0.40x | 4.6x |
| 223 | 8.1 | 123.0 | 0.30x | 4.6x |

## Quality Check

**Prompt:** "Explain what gravity is in one paragraph:"

**Normal KV:**
> Gravity is a fundamental force of nature that attracts two objects with mass towards each other. It is a universal force that affects everything with mass or energy, from the smallest subatomic particles to the largest galaxies. Gravity is what keeps planets in orbit around their stars, what makes...

**TurboQuant 3.5-bit:**
> Gravity is a fundamental force of nature that attracts objects with mass towards each other. It is a universal force that affects everything with mass, from the smallest subatomic particles to the largest galaxies. Gravity is what keeps planets in orbit around their stars, holds objects on the surf...

Output is **near-identical** — same structure, same facts, same level of detail. This confirms the paper's claim that 3.5-bit is quality-neutral.

## Findings

1. **3.5-bit is genuinely quality-neutral.** Output is almost word-for-word identical to fp16 KV. The paper's claim is validated for this model/prompt pair.

2. **3.5-bit is slower than 3-bit** (0.30–0.40x vs 0.36–0.61x at comparable KV sizes). The dual-codec overhead from outlier splitting (two rotation matrices, two codebooks, channel gather/scatter) costs more than the extra bit saves. In practice, integer bits are simpler and faster.

3. **4.6x compression** — less than 3-bit's 5.3x, by design. The extra half-bit goes to outlier channels where quantization error hurts most.

4. **Outlier channel selection is data-dependent.** The splitting uses average magnitude across prefill tokens to identify outliers. This means the quality benefit is sensitive to the initial prompt distribution — a limitation not discussed in the paper.
