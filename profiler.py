"""
profiler.py
===========
prefill-decode-bench — entry point.

Measures prefill and decode speed and surfaces the memory-bandwidth-bound
nature of LLM inference. Works on Apple Silicon via MLX and Nvidia GPUs
via CUDA (torch + transformers).

Usage:
    # Auto-detect backend
    python profiler.py --model meta-llama/Llama-3.2-3B-Instruct

    # Force a backend
    python profiler.py --backend mlx  --model mlx-community/Llama-3.2-3B-Instruct-4bit
    python profiler.py --backend cuda --model meta-llama/Llama-3.2-3B-Instruct

    # Save chart
    python profiler.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --plot

    # Custom sweep
    python profiler.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \\
        --prefill-lengths 128 512 1024 2048 4096 \\
        --decode-kv-sizes 64 256 512 1024 2048 \\
        --decode-tokens 150 --runs 5 --plot
"""

import argparse
import sys


def detect_backend() -> str:
    """Return 'mlx' on Apple Silicon, 'cuda' if a CUDA GPU is available."""
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
    print("No supported backend found.")
    print("  Apple Silicon: pip install mlx-lm")
    print("  Nvidia GPU:    pip install torch transformers accelerate")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Profile LLM prefill and decode speed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--backend", choices=["mlx", "cuda"], default=None,
        help="Backend to use. Auto-detected if not set.",
    )
    p.add_argument(
        "--model", required=True,
        help=(
            "Model ID or local path. "
            "MLX: mlx-community/* HuggingFace IDs. "
            "CUDA: any HuggingFace causal LM."
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
    return p.parse_args()


def main():
    args = parse_args()
    backend = args.backend or detect_backend()
    print(f"\nBackend: {backend.upper()}")

    if backend == "mlx":
        from backends.mlx.profiler import run
    else:
        from backends.cuda.profiler import run

    run(
        model_id=args.model,
        prefill_lengths=args.prefill_lengths,
        decode_kv_sizes=args.decode_kv_sizes,
        decode_tokens=args.decode_tokens,
        runs=args.runs,
        plot=args.plot,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
