# Week 1 — Baseline Profiling & Cross-Hardware Analysis

**Period:** March 15 – 21, 2026
**Hardware:**
- Apple M1 (16 GB unified, 68.25 GB/s theoretical bandwidth)
- Raspberry Pi 5 (Cortex-A76, 8 GB LPDDR4X, 34.1 GB/s theoretical bandwidth)

---

## Goals

- Stand up the benchmarking framework with MLX, llama.cpp, and CUDA backends
- Run baseline profiling on Apple M1 with small quantized models
- Compare MLX vs llama.cpp on the same model to understand backend-level differences
- Run llama.cpp on Raspberry Pi 5 (CPU-only) for edge hardware comparison
- Validate measurement methodology before expanding to more hardware

---

## Experiments

### 1. MLX — Llama 3.2 3B Instruct (4-bit)

**Model:** `mlx-community/Llama-3.2-3B-Instruct-4bit`
**Params:** 3.21B logical, 1.74 GB weights

| Phase | Tokens | tok/s | Time (ms) | TFLOPS | Peak Mem |
|-------|--------|-------|-----------|--------|----------|
| Prefill | 128 | 191.9 | 667 | 1.23 | 2.19 GB |
| Prefill | 512 | 198.1 | 2586 | 1.27 | 3.51 GB |
| Prefill | 1024 | 199.6 | 5128 | 1.28 | 4.04 GB |
| Prefill | 2048 | 197.3 | 10385 | 1.27 | 6.11 GB |
| Prefill | 4096 | 186.7 | 21939 | 1.20 | 10.49 GB |

| Phase | KV Cache | tok/s | ms/tok | BW (GB/s) | BW Util |
|-------|----------|-------|--------|-----------|---------|
| Decode | 64 | 24.6 | 40.7 | 43.1 | 63% |
| Decode | 512 | 23.1 | 43.3 | 41.9 | 61% |
| Decode | 1024 | 21.8 | 45.8 | 41.2 | 60% |
| Decode | 2048 | 19.3 | 51.7 | 39.5 | 58% |

Decode degradation KV=64→2048: 21.4% — moderate bandwidth pressure.

### 2. MLX vs llama.cpp — Qwen2.5 1.5B Instruct (4-bit)

Fair comparison: text-only model, standard transformer architecture.

**MLX model:** `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (0.87 GB)
**llama.cpp model:** `Qwen/Qwen2.5-1.5B-Instruct-GGUF` Q4_K_M (1.11 GB)

#### Prefill

| Tokens | MLX tok/s | llama.cpp tok/s | Difference |
|--------|-----------|-----------------|------------|
| 111 | 436.7 | 451.2 | ~equal |
| 243 | 489.0 | 486.5 | ~equal |
| 485 | 490.1 | 489.6 | ~equal |
| 969 | 497.6 | 489.5 | ~equal |

**Finding:** Prefill performance is virtually identical across backends (within 3%).

#### Decode

| KV Cache | MLX tok/s | llama.cpp tok/s | Difference |
|----------|-----------|-----------------|------------|
| 45 | 46.5 | 41.8 | MLX 11% faster |
| 243 | 47.7 | 39.1 | MLX 22% faster |
| 485 | 47.8 | 39.3 | MLX 22% faster |

**Finding:** MLX achieves ~15-22% better decode throughput.
MLX bandwidth utilization: 59-62%. llama.cpp: 64-68%.
Despite llama.cpp hitting higher raw bandwidth numbers, the larger Q4_K_M
weight size (1.11 GB vs 0.87 GB) means more bytes per token, yielding
fewer tokens per second.

### 3. MLX (VLM) vs llama.cpp — Qwen3.5 0.8B (4-bit)

**MLX model:** `mlx-community/Qwen3.5-0.8B-MLX-4bit` (loaded via mlx_vlm, language_model only)
**llama.cpp model:** `unsloth/Qwen3.5-0.8B-GGUF` Q4_K_M

| Phase | MLX tok/s | llama.cpp tok/s | Ratio |
|-------|-----------|-----------------|-------|
| Prefill 512 | 527 | 1029 | llama.cpp 2.0x |
| Decode KV=64 | 82.0 | 49.2 | MLX 1.7x |

Large discrepancies — but **not a fair comparison** due to:

1. **Different quantization:** MLX uniform 4-bit (0.42 GB) vs GGUF Q4_K_M mixed precision (0.52 GB).
   Embedding stored at Q6_K in GGUF (209 MB) vs 4-bit in MLX (127 MB).
2. **Hybrid architecture:** Qwen3.5 mixes standard self-attention with linear attention layers.
   Each framework may optimize these differently.
3. **Massive vocabulary:** 248K vocab size — the embedding/lm_head is 34% of total model
   parameters. This disproportionately affects small models and varies in efficiency per backend.
4. **Param counting bug found:** our `_count_params_logical()` misses `QuantizedEmbedding` modules,
   reporting 0.50B instead of the true 0.75B. TFLOPS values were underestimated.

**Lesson:** VLM models loaded via mlx_vlm introduce confounding variables.
For cross-backend comparisons, use text-only models with standard architectures.

### 4. Raspberry Pi 5 — Qwen3.5 0.8B (Q4_K_M, CPU-only)

**Device:** Raspberry Pi 5 (Cortex-A76 quad-core, 8 GB LPDDR4X-4267)
**Model:** `unsloth/Qwen3.5-0.8B-GGUF` Q4_K_M (0.52 GB)
**Backend:** llama.cpp, `--gpu-layers 0` (CPU, ARM NEON)

#### Prefill

| Tokens | tok/s | Time (ms) | ms/tok |
|--------|-------|-----------|--------|
| 128 | 106.0 | 1,208 | 9.43 |
| 256 | 93.8 | 2,730 | 10.67 |
| 512 | 90.3 | 5,671 | 11.08 |
| 1024 | 89.0 | 11,501 | 11.23 |

#### Decode

| KV Cache | tok/s | ms/tok |
|----------|-------|--------|
| 64 | 14.4 | 69.6 |
| 256 | 14.3 | 70.1 |
| 512 | 14.5 | 69.2 |

Decode degradation KV=64→512: **~0%** — essentially flat. The 0.52 GB model
is small enough that KV cache overhead is negligible at this context range.

**Effective bandwidth:** 0.522 GB × 14.4 tok/s ≈ **7.5 GB/s** (22% of theoretical 34.1 GB/s).
Much lower utilization than M1 — CPU-based inference cannot saturate DRAM bandwidth
the way a GPU memory controller can.

### 5. Apple M1 vs Raspberry Pi 5 — same model, same backend (llama.cpp)

Direct comparison: Qwen3.5-0.8B Q4_K_M, llama.cpp, 3 runs median.

#### Prefill (compute-bound)

| Tokens | M1 (Metal) | Pi 5 (CPU) | M1 / Pi 5 |
|--------|------------|------------|-----------|
| 128 | 902.6 tok/s | 106.0 tok/s | **8.5x** |
| 256 | 1,009.7 tok/s | 93.8 tok/s | **10.8x** |
| 512 | 1,029.4 tok/s | 90.3 tok/s | **11.4x** |
| 1024 | 1,024.3 tok/s | 89.0 tok/s | **11.5x** |

#### Decode (bandwidth-bound)

| KV Cache | M1 (Metal) | Pi 5 (CPU) | M1 / Pi 5 |
|----------|------------|------------|-----------|
| 64 | 49.2 tok/s | 14.4 tok/s | **3.4x** |
| 256 | 48.9 tok/s | 14.3 tok/s | **3.4x** |
| 512 | 48.5 tok/s | 14.5 tok/s | **3.3x** |

**Analysis:**

- **Prefill: M1 is 8.5–11.5x faster.** Prefill is compute-bound. The M1 GPU
  (8-core, ~2.6 TFLOPS FP32 + quantized kernels via Metal) dwarfs the Pi 5's
  quad-core Cortex-A76 (~50 GFLOPS with NEON). The gap widens at longer sequences
  because the GPU sustains throughput while the CPU cache starts thrashing.

- **Decode: M1 is only 3.3–3.4x faster.** Decode is bandwidth-bound. M1 has
  68.25 GB/s vs Pi 5's 34.1 GB/s — a 2x bandwidth ratio. The remaining ~1.7x
  gap comes from GPU memory controllers achieving higher sustained bandwidth
  utilization (51% on M1 vs 22% on Pi 5).

- **Pi 5 decode is perfectly flat** across KV sizes. No degradation at all.
  The model is small enough (0.52 GB) that even at KV=512 the total memory
  traffic is dominated by weight reads, not KV cache.

- **14.4 tok/s on Pi 5 ≈ 1 token every 70ms** — functional for edge deployment.
  Not conversational-speed for long outputs, but viable for short completions,
  classification, or summarization tasks on a $80 board.

---

## Key Observations

1. **Prefill is compute-bound, decode is bandwidth-bound** — confirmed empirically
   on both M1 (GPU) and Pi 5 (CPU). Prefill tok/s stays flat as prompt length grows.
   Decode tok/s drops with KV cache size (on M1; flat on Pi 5 at this model size).

2. **On standard text-only models, MLX and llama.cpp perform similarly on M1.**
   Prefill within noise. Decode favors MLX by 10-22% due to smaller weight size
   from uniform 4-bit quantization.

3. **Quantization format matters more than framework.**
   MLX 4-bit uniform produces smaller files, fewer bytes per token, faster decode.
   GGUF Q4_K_M uses mixed precision (Q4/Q5/Q6) for better accuracy, at cost of larger size.
   This is a quality-vs-speed tradeoff, not a framework flaw.

4. **M1 effective bandwidth utilization: 45-65% of theoretical 68.25 GB/s.**
   Pi 5 bandwidth utilization: ~22% of theoretical 34.1 GB/s. CPU-based inference
   cannot saturate DRAM bandwidth the way a GPU memory controller can.

5. **The compute vs bandwidth gap explains the M1/Pi ratio.**
   M1 is 8-11x faster at prefill (compute advantage) but only 3.4x faster at
   decode (bandwidth advantage). This is the key insight: hardware with more
   compute (GPU) wins big on prefill, but the decode bottleneck narrows the gap
   to a bandwidth ratio.

6. **Pi 5 is viable for edge LLM inference** at ~14 tok/s with a 0.8B Q4 model.
   That's 1 token every 70ms — usable for short completions on an $80 board.

7. **VLM support added to MLX backend** via `_LMAdapter` wrapper that extracts
   `language_model` from mlx_vlm models. Works for any VLM — automatic fallback
   when `mlx_lm.load()` fails.

---

## Known Issues / TODO

- [ ] Fix `_count_params_logical()` to handle `QuantizedEmbedding` and `Embedding` modules
- [ ] Fix KV bytes/token calculation for VLM language models (config nested inside VLM)
- [x] Run same experiments on **Raspberry Pi 5** (llama.cpp CPU-only) for edge comparison
- [ ] Add Raspberry Pi DRAM bandwidth to hardware bandwidth lookup table (rename from `APPLE_BANDWIDTH_GBS`)
- [ ] Compare Q4_K_M vs Q8_0 on llama.cpp to isolate quantization quality vs speed tradeoff

---

## Hardware Tested

| Device | Chip | Memory | Bandwidth | Backends |
|--------|------|--------|-----------|----------|
| MacBook Pro | Apple M1 | 16 GB unified | 68.25 GB/s | MLX, llama.cpp (Metal) |
| Raspberry Pi 5 | Cortex-A76 (quad) | 8 GB LPDDR4X | 34.1 GB/s | llama.cpp (CPU, NEON) |

---

## Files Produced

```
results/
├── profile_mlx_Apple-M1_mlx-community__Llama-3.2-3B-Instruct-4bit.json
├── profile_mlx_Apple-M1_mlx-community__Qwen2.5-1.5B-Instruct-4bit.json
├── profile_mlx_Apple-M1_mlx-community__Qwen3.5-0.8B-MLX-4bit.json
├── profile_llamacpp_Apple-M1_Qwen__Qwen2.5-1.5B-Instruct-GGUF.json
├── profile_llamacpp_Apple-M1_unsloth__Qwen3.5-0.8B-GGUF.json
├── profile_llamacpp_Cortex-A76_models__Qwen3.5-0.8B-Q4_K_M.gguf.json  (Pi 5)
└── *.png  (corresponding charts)
```
