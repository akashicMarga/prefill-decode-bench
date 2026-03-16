# 001 — Llama 3.2 3B Instruct MLX Baseline

**Date:** 2026-03-15
**Hardware:** Apple M1, 16 GB unified memory, 68.25 GB/s theoretical bandwidth
**Backend:** MLX
**Model:** `mlx-community/Llama-3.2-3B-Instruct-4bit`
**Params:** 3.21B logical, 1.74 GB weights

---

## Prefill (compute-bound)

| Tokens | tok/s | Time (ms) | TFLOPS | Peak Mem |
|--------|-------|-----------|--------|----------|
| 128 | 191.9 | 667 | 1.23 | 2.19 GB |
| 512 | 198.1 | 2586 | 1.27 | 3.51 GB |
| 1024 | 199.6 | 5128 | 1.28 | 4.04 GB |
| 2048 | 197.3 | 10385 | 1.27 | 6.11 GB |
| 4096 | 186.7 | 21939 | 1.20 | 10.49 GB |

Prefill throughput is stable ~195-200 tok/s from 512–2048, dipping at 4096 as
memory pressure from the large KV allocation starts to bite.

## Decode (memory-bandwidth-bound)

| KV Cache | tok/s | ms/tok | BW (GB/s) | BW Util |
|----------|-------|--------|-----------|---------|
| 64 | 24.6 | 40.7 | 43.1 | 63% |
| 512 | 23.1 | 43.3 | 41.9 | 61% |
| 1024 | 21.8 | 45.8 | 41.2 | 60% |
| 2048 | 19.3 | 51.7 | 39.5 | 58% |

Decode degradation KV=64→2048: **21.4%** — moderate bandwidth pressure as the
KV cache grows. At KV=2048 the combined weight + KV read is ~1.74 GB + 0.5 GB
per token, pushing effective bandwidth down.

## Findings

- Prefill saturates at ~200 tok/s (~1.28 TFLOPS) on M1 for this model size.
- Decode bandwidth utilization is 58–63% of theoretical 68.25 GB/s.
- 21% decode degradation across KV=64→2048 shows meaningful KV cache pressure
  at longer contexts with a 3B model on 16 GB unified memory.
