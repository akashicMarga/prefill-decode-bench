"""
profiler.py
===========
prefill-decode-bench — entry point.

Measures prefill and decode speed and surfaces the memory-bandwidth-bound
nature of LLM inference. Works on Apple Silicon via MLX, Nvidia GPUs
via CUDA (torch + transformers), and any platform via llama.cpp (GGUF).

Usage:
    # Auto-detect backend
    python profiler.py --model meta-llama/Llama-3.2-3B-Instruct

    # Force a backend
    python profiler.py --backend mlx      --model mlx-community/Llama-3.2-3B-Instruct-4bit
    python profiler.py --backend cuda     --model meta-llama/Llama-3.2-3B-Instruct
    python profiler.py --backend llamacpp --model ./models/model.gguf
    python profiler.py --backend llamacpp --model unsloth/Qwen3.5-0.8B-GGUF --gguf-file "*Q4_K_M.gguf"

    # Save chart
    python profiler.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --plot

    # Custom sweep
    python profiler.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \\
        --prefill-lengths 128 512 1024 2048 4096 \\
        --decode-kv-sizes 64 256 512 1024 2048 \\
        --decode-tokens 150 --runs 5 --plot
"""

import argparse
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent


def _llamacpp_available() -> bool:
    """Check if llama-bench binary exists (vendor build or $PATH)."""
    vendor_bin = REPO_ROOT / "vendor" / "llama.cpp" / "build" / "bin" / "llama-bench"
    if vendor_bin.is_file():
        return True
    if shutil.which("llama-bench"):
        return True
    return False


def detect_backend() -> str:
    """Return best available backend: mlx > cuda > llamacpp."""
    try:
        import mlx.core as mx  # noqa: F401
        return "mlx"
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    if _llamacpp_available():
        return "llamacpp"
    print("No supported backend found.")
    print("  Apple Silicon: pip install mlx-lm")
    print("  Nvidia GPU:    pip install torch transformers accelerate")
    print("  llama.cpp:     ./setup_llamacpp.sh")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Profile LLM prefill and decode speed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--backend", choices=["mlx", "cuda", "llamacpp"], default=None,
        help="Backend to use. Auto-detected if not set.",
    )
    p.add_argument(
        "--model", required=True,
        help=(
            "Model ID or local path. "
            "MLX: mlx-community/* HuggingFace IDs. "
            "CUDA: any HuggingFace causal LM. "
            "llamacpp: HF GGUF repo or local .gguf file."
        ),
    )
    p.add_argument("--runs", type=int, default=3,
                   help="Timing runs per data point (median used).")
    p.add_argument("--decode-tokens", type=int, default=100,
                   help="Tokens to generate per decode measurement.")
    p.add_argument("--plot", action="store_true",
                   help="Save a chart to results/")
    p.add_argument("--output-dir", default="results",
                   help="Directory for JSON results and optional chart.")
    p.add_argument(
        "--prefill-lengths", nargs="+", type=int,
        default=[128, 256, 512, 1024, 2048, 4096],
        help="Prompt lengths (tokens) to sweep for prefill.",
    )
    p.add_argument(
        "--decode-kv-sizes", nargs="+", type=int,
        default=[64, 256, 512, 1024, 2048],
        help="KV cache sizes (tokens) to sweep for decode.",
    )

    # TurboQuant KV cache quantization
    p.add_argument(
        "--turboquant-bits", type=float, default=None,
        help="Enable TurboQuant KV cache quantization (MLX only). Bits per channel: 2, 2.5, 3, 3.5, or 4.",
    )

    # Speculative decoding options
    p.add_argument(
        "--draft-model", default=None,
        help="Draft model ID for speculative decoding (MLX only). Same tokenizer required.",
    )
    p.add_argument(
        "--num-draft-tokens", type=int, default=4,
        help="Draft tokens per speculation step.",
    )

    # llama.cpp-specific options
    p.add_argument(
        "--gguf-file", default="*Q4_K_M.gguf",
        help="GGUF filename pattern when loading from HuggingFace (llamacpp).",
    )
    p.add_argument(
        "--gpu-layers", type=int, default=-1,
        help="Layers to offload to GPU (llamacpp). -1 = all, 0 = CPU only.",
    )
    p.add_argument(
        "--llamacpp-bin", default=None,
        help="Path to llama-bench binary (llamacpp). Auto-detected if not set.",
    )

    return p.parse_args()


def main():
    args = parse_args()
    backend = args.backend or detect_backend()
    print(f"\nBackend: {backend.upper()}")

    common = dict(
        model_id=args.model,
        prefill_lengths=args.prefill_lengths,
        decode_kv_sizes=args.decode_kv_sizes,
        decode_tokens=args.decode_tokens,
        runs=args.runs,
        plot=args.plot,
        output_dir=args.output_dir,
    )

    if backend == "mlx":
        from backends.mlx.profiler import run
        run(**common,
            draft_model_id=args.draft_model,
            num_draft_tokens=args.num_draft_tokens,
            turboquant_bits=args.turboquant_bits)
    elif backend == "cuda":
        from backends.cuda.profiler import run
        run(**common)
    elif backend == "llamacpp":
        from backends.llamacpp.profiler import run
        run(**common,
            gguf_file=args.gguf_file,
            n_gpu_layers=args.gpu_layers,
            llamacpp_bin=args.llamacpp_bin)


if __name__ == "__main__":
    main()
