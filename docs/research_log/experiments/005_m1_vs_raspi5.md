# 005 — Apple M1 vs Raspberry Pi 5: Cross-Hardware Analysis

**Date:** 2026-03-16
**Purpose:** Direct hardware comparison — same model, same backend, same settings
**Model:** Qwen3.5-0.8B Q4_K_M (0.52 GB), llama.cpp, 3 runs median

---

## Prefill (compute-bound)

| Tokens | M1 (Metal) | Pi 5 (CPU) | M1 / Pi 5 |
|--------|------------|------------|-----------|
| 128 | 902.6 tok/s | 106.0 tok/s | **8.5x** |
| 256 | 1,009.7 tok/s | 93.8 tok/s | **10.8x** |
| 512 | 1,029.4 tok/s | 90.3 tok/s | **11.4x** |
| 1024 | 1,024.3 tok/s | 89.0 tok/s | **11.5x** |

The M1 GPU (8-core, ~2.6 TFLOPS FP32 + quantized kernels via Metal) dwarfs
the Pi 5's quad-core Cortex-A76 (~50 GFLOPS with NEON). The gap widens at
longer sequences because the GPU sustains throughput while the CPU cache
starts thrashing.

## Decode (bandwidth-bound)

| KV Cache | M1 (Metal) | Pi 5 (CPU) | M1 / Pi 5 |
|----------|------------|------------|-----------|
| 64 | 49.2 tok/s | 14.4 tok/s | **3.4x** |
| 256 | 48.9 tok/s | 14.3 tok/s | **3.4x** |
| 512 | 48.5 tok/s | 14.5 tok/s | **3.3x** |

## Analysis

| Metric | M1 | Pi 5 | Ratio |
|--------|-----|------|-------|
| Theoretical BW | 68.25 GB/s | 34.1 GB/s | 2.0x |
| Effective BW (decode) | ~37 GB/s | ~7.5 GB/s | 4.9x |
| BW utilization | ~54% | ~22% | 2.5x |
| Prefill tok/s (512) | 1,029 | 90.3 | 11.4x |
| Decode tok/s (KV=64) | 49.2 | 14.4 | 3.4x |

**Key insight: the compute vs bandwidth gap explains the ratios.**

- **Prefill (8.5–11.5x):** Compute-bound. The M1 GPU has ~50x the raw
  throughput of the Pi 5 CPU, but quantized matmul doesn't exploit all of it.
  Still, the compute advantage dominates.

- **Decode (3.3–3.4x):** Bandwidth-bound. M1 has 2x the theoretical bandwidth
  and ~2.5x the bandwidth utilization, yielding ~3.4x effective advantage.
  The decode bottleneck narrows the gap to a bandwidth ratio.

- **Pi 5 decode is perfectly flat** across KV sizes. No degradation at all.
  The model is small enough (0.52 GB) that even at KV=512 the total memory
  traffic is dominated by weight reads, not KV cache.

## Findings

1. Hardware with more compute (GPU) wins big on prefill, but the decode
   bottleneck narrows the hardware gap to a bandwidth ratio.
2. The M1 achieves 2.5x better bandwidth utilization than the Pi 5 — GPU
   memory controllers sustain higher throughput than CPU load instructions.
3. For edge deployment, the Pi 5's 14.4 tok/s is usable for short tasks.
   The 3.4x decode gap means an M1-class device is needed for conversational
   speed with larger models.
