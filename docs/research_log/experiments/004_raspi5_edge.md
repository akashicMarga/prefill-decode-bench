# 004 — Raspberry Pi 5: Edge Inference Baseline

**Date:** 2026-03-16
**Hardware:** Raspberry Pi 5 (Cortex-A76 quad-core, 8 GB LPDDR4X-4267, 34.1 GB/s theoretical)
**Backend:** llama.cpp, `--gpu-layers 0` (CPU, ARM NEON)
**Model:** `unsloth/Qwen3.5-0.8B-GGUF` Q4_K_M (0.52 GB)

---

## Prefill

| Tokens | tok/s | Time (ms) | ms/tok |
|--------|-------|-----------|--------|
| 128 | 106.0 | 1,208 | 9.43 |
| 256 | 93.8 | 2,730 | 10.67 |
| 512 | 90.3 | 5,671 | 11.08 |
| 1024 | 89.0 | 11,501 | 11.23 |

Prefill is compute-bound. The quad-core Cortex-A76 (~50 GFLOPS with NEON)
sustains ~90-106 tok/s, declining slightly at longer sequences as L2 cache
pressure increases.

## Decode

| KV Cache | tok/s | ms/tok |
|----------|-------|--------|
| 64 | 14.4 | 69.6 |
| 256 | 14.3 | 70.1 |
| 512 | 14.5 | 69.2 |

Decode degradation KV=64→512: **~0%** — essentially flat. The 0.52 GB model
is small enough that KV cache overhead is negligible at this context range.
Total memory traffic is dominated by weight reads.

## Bandwidth Analysis

Effective bandwidth: 0.522 GB × 14.4 tok/s ≈ **7.5 GB/s** (22% of theoretical 34.1 GB/s).

Much lower utilization than M1's 58–63%. CPU-based inference cannot saturate
DRAM bandwidth the way a GPU memory controller can — the CPU must issue
discrete load instructions through the cache hierarchy rather than streaming
from a wide memory bus.

## Findings

- **14.4 tok/s on Pi 5 ≈ 1 token every 70ms** — functional for edge deployment.
- Not conversational-speed for long outputs, but viable for short completions,
  classification, or summarization tasks on an $80 board.
- Zero decode degradation at this model size / context range.
- DRAM bandwidth utilization is the primary bottleneck at only 22%.
