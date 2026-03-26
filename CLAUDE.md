# CLAUDE.md

## Project Overview

**prefill-decode-bench** — A benchmarking framework for studying transformer inference behavior (prefill vs decode phases) across hardware platforms and backends.

## Quick Reference

```bash
# Run profiler (auto-detects backend: MLX > CUDA > llama.cpp)
python profiler.py --model mlx-community/Llama-3.2-3B-Instruct-4bit

# Force a specific backend
python profiler.py --backend mlx --model <model-id>
python profiler.py --backend cuda --model <model-id>
python profiler.py --backend llamacpp --model ./models/<model>.gguf

# Plot results
python plot_results.py results/<file>.json

# Compare runs
python plot_results.py results/run1.json results/run2.json

# Speculative decoding (MLX only)
python profiler.py --model <main-model> --draft-model <draft-model> --num-draft-tokens 4

# Speculative decoding experiment
python -m experiments.speculative_decoding.run --main <main> --draft <draft> --prompt "..." --num-draft 4

# TurboQuant KV cache quantization (MLX only, 2/2.5/3/3.5/4 bits)
python profiler.py --model <model> --turboquant-bits 3

# Concurrency benchmark (MLX only)
python -m backends.mlx.benchmark_concurrent --model <model> --whisper-model <whisper> --audio <file>.wav
```

## Architecture

- **`profiler.py`** — Main entry point. Auto-detects backend and dispatches to backend-specific `run()`.
- **`backends/`** — Multi-backend abstraction layer:
  - `types.py` — Shared dataclasses (`SystemInfo`, `PrefillResult`, `DecodeResult`, `HardwareMetrics`, `ProfileRun`)
  - `utils.py` — Shared utilities (`build_prompt()`, `print_summary()`)
  - `mlx/profiler.py` — Apple Silicon via Metal
  - `cuda/profiler.py` — Nvidia GPUs via PyTorch/transformers
  - `llamacpp/profiler.py` — GGUF models via llama.cpp subprocess
- **`plot_results.py`** — Visualization: single-run charts, hardware metrics, multi-run comparisons
- **`backends/mlx/turboquant/`** — TurboQuant KV cache quantization (ICLR 2026, arXiv:2504.19874):
  - `codebook.py` — Lloyd-Max on Beta PDF, rotation/projection matrices
  - `bitpack.py` — Bit-packing with Metal kernel fast path
  - `codec.py` — MSECodec (values), ProdCodec (keys), SplitCodec (non-integer bits)
  - `kernels.py` — Metal kernels for fused quantized attention scoring
  - `cache.py` — TurboQuantKVCache drop-in for mlx_lm
- **`experiments/speculative_decoding/`** — Hybrid Mamba+Attention speculative decoding with snapshot-restore-replay
- **`docs/research_log/`** — Weekly experiment notes and findings

## Dependencies

Backend-specific — see `requirements.txt`:
- **MLX**: `mlx-lm>=0.19.0`
- **CUDA**: `torch>=2.1.0`, `transformers>=4.40.0`, `accelerate>=0.27.0`
- **llama.cpp**: Built from source via `./setup_llamacpp.sh`
- **All**: `matplotlib>=3.7.0`

## Key Conventions

- Each backend implements the same interface: load model, warmup, measure prefill sweep, measure decode sweep, save JSON, print summary, optionally plot.
- Results are saved as JSON (`ProfileRun` dataclass) to `results/`.
- `results/`, `models/`, and `vendor/` are git-ignored.
- Timing synchronization differs per backend: MLX uses `mx.eval()`, CUDA uses `torch.cuda.synchronize()`, llama.cpp parses subprocess output.
