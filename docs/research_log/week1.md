# Week 1 — Baseline Profiling & Cross-Hardware Analysis

**Period:** March 15–21, 2026
**Hardware:**
- Apple M1 (16 GB unified, 68.25 GB/s theoretical bandwidth)
- Raspberry Pi 5 (Cortex-A76, 8 GB LPDDR4X, 34.1 GB/s theoretical bandwidth)

---

## Experiments

| # | Date | Title | Backend | Model | Detail |
|---|------|-------|---------|-------|--------|
| 001 | 03-15 | Llama 3.2 3B MLX baseline | MLX | Llama-3.2-3B-Instruct-4bit | [detail](experiments/001_llama3b_mlx_baseline.md) |
| 002 | 03-15 | Qwen2.5 1.5B cross-backend | MLX + llama.cpp | Qwen2.5-1.5B-Instruct 4-bit | [detail](experiments/002_qwen2.5_cross_backend.md) |
| 003 | 03-15 | Qwen3.5 0.8B VLM comparison | MLX + llama.cpp | Qwen3.5-0.8B 4-bit | [detail](experiments/003_qwen3.5_vlm_comparison.md) |
| 004 | 03-16 | Raspberry Pi 5 edge baseline | llama.cpp (CPU) | Qwen3.5-0.8B Q4_K_M | [detail](experiments/004_raspi5_edge.md) |
| 005 | 03-16 | M1 vs Pi 5 cross-hardware | llama.cpp | Qwen3.5-0.8B Q4_K_M | [detail](experiments/005_m1_vs_raspi5.md) |
| 006 | 03-16 | Speculative decoding Qwen3.5 | MLX + llama.cpp | Qwen3.5-4B + 0.8B draft | [detail](experiments/006_speculative_qwen3.5.md) |

---

## Key Findings

1. **Prefill is compute-bound, decode is bandwidth-bound** — confirmed on both
   M1 (GPU) and Pi 5 (CPU). Prefill tok/s stays flat as prompt grows; decode
   tok/s drops with KV cache size (on M1; flat on Pi 5 at this model size).

2. **MLX and llama.cpp perform similarly on standard text-only models.** Prefill
   within noise. Decode favors MLX by 10–22% due to smaller weights from uniform
   4-bit quantization. ([002](experiments/002_qwen2.5_cross_backend.md))

3. **Quantization format matters more than framework.** MLX 4-bit uniform =
   smaller files, faster decode. GGUF Q4_K_M = mixed precision, better accuracy,
   larger files. Quality-vs-speed tradeoff, not a framework flaw.

4. **M1 bandwidth utilization: 45–65%.** Pi 5: ~22%. GPU memory controllers
   sustain higher throughput than CPU load instructions through cache hierarchy.

5. **M1 is 8–11x faster at prefill (compute gap), only 3.3x at decode
   (bandwidth gap).** The decode bottleneck narrows the hardware advantage to a
   bandwidth ratio. ([005](experiments/005_m1_vs_raspi5.md))

6. **Pi 5 is viable for edge LLM inference** at ~14 tok/s with a 0.8B Q4 model.
   1 token every 70ms — usable for short completions on an $80 board.
   ([004](experiments/004_raspi5_edge.md))

7. **VLM models introduce confounding variables** for cross-backend comparison.
   Use text-only standard transformers for fair benchmarks.
   ([003](experiments/003_qwen3.5_vlm_comparison.md))

8. **Speculative decoding is incompatible with hybrid Mamba+Attention models.**
   Both llama.cpp (M-RoPE position errors) and MLX (silent no-op cache trim)
   produce invalid output on Qwen3.5. A custom snapshot-restore fix produces
   correct output but no speedup on this hardware/model pair.
   ([006](experiments/006_speculative_qwen3.5.md))

---

## Known Issues / TODO

- [ ] Fix `_count_params_logical()` to handle `QuantizedEmbedding` and `Embedding` modules
- [ ] Fix KV bytes/token calculation for VLM language models (config nested inside VLM)
- [x] Run same experiments on **Raspberry Pi 5** (llama.cpp CPU-only) for edge comparison
- [ ] Add Raspberry Pi DRAM bandwidth to hardware bandwidth lookup table (rename from `APPLE_BANDWIDTH_GBS`)
- [ ] Compare Q4_K_M vs Q8_0 on llama.cpp to isolate quantization quality vs speed tradeoff
- [x] Test speculative decoding on MLX (Qwen3.5-4B + 0.8B) — broken due to non-trimmable Mamba cache
- [x] Test speculative decoding on llama.cpp (Qwen3.5-4B + 0.8B) — broken due to M-RoPE position errors
- [ ] Retry speculative decoding with a pure-transformer model pair (e.g. Qwen2.5 family) to get valid numbers
- [ ] Implement llama.cpp speculative decoding in profiler once arch compatibility is resolved

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
├── profile_mlx_Apple-M1_mlx-community__Qwen3.5-4B-4bit.json           (incl. speculative_decode)
└── *.png  (corresponding charts)
```
