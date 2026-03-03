"""
backends/mlx/benchmark_concurrent.py
=====================================
Measures bandwidth contention when LLM decode and Whisper ASR run
simultaneously on Apple Silicon via MLX.

Each model runs in a separate process (required by Metal — one command
encoder per process). Contention happens at the hardware level: both
processes compete for the shared memory bus.

Usage:
    python -m backends.mlx.benchmark_concurrent \
        --model mlx-community/Llama-3.2-3B-Instruct-4bit \
        --whisper-model mlx-community/whisper-tiny

    # With a real audio file
    python -m backends.mlx.benchmark_concurrent \
        --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
        --whisper-model mlx-community/whisper-large-v3-turbo-q4 \
        --audio recording.wav
"""

import argparse
import multiprocessing as mp
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _chip() -> str:
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if "Chip" in line:
                return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Apple Silicon"


def _memory_gb() -> float:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return round(int(out.strip()) / (1024 ** 3), 1)
    except Exception:
        return 0.0


def _make_audio(audio_path, duration_s=10.0, sample_rate=16000):
    import numpy as np
    if audio_path is not None:
        try:
            import soundfile as sf
            audio_np, sr = sf.read(audio_path, dtype="float32")
            if sr != sample_rate:
                ratio = sample_rate / sr
                new_len = int(len(audio_np) * ratio)
                indices = np.linspace(0, len(audio_np) - 1, new_len).astype(int)
                audio_np = audio_np[indices]
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            return audio_np
        except ImportError:
            print("  soundfile not installed — using noise. pip install soundfile")
    num_samples = int(duration_s * sample_rate)
    return (np.random.randn(num_samples).astype(np.float32) * 0.001)


# ---------------------------------------------------------------------------
# Worker functions — each runs in its own process with its own Metal context
# ---------------------------------------------------------------------------

def _llm_decode_worker(result_queue, model_id, kv_size, decode_tokens, runs):
    """Run LLM decode in an isolated process."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from backends.utils import build_prompt

    model, tokenizer = load(model_id)

    # warmup
    warm = tokenizer.encode("Warmup pass to compile Metal kernels.")
    mx.eval(model(mx.array(warm)[None]))

    prompt = build_prompt(tokenizer, kv_size)
    tokens = tokenizer.encode(prompt)

    results = []
    for _ in range(runs):
        ids = mx.array(tokens)[None]
        cache = make_prompt_cache(model)
        logits = model(ids, cache=cache)
        mx.eval(logits)

        last = mx.array([[tokens[-1]]])
        t0 = time.perf_counter()
        for _ in range(decode_tokens):
            logits = model(last, cache=cache)
            mx.eval(logits)
            next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            mx.eval(next_tok)
            last = next_tok
        elapsed = time.perf_counter() - t0
        results.append(decode_tokens / elapsed)

    results.sort()
    result_queue.put(("llm", results[len(results) // 2]))


def _whisper_worker(result_queue, whisper_model, audio, runs):
    """Run Whisper transcription in an isolated process."""
    import mlx_whisper

    audio_duration = len(audio) / 16000

    # warmup
    short = audio[:16000] if len(audio) > 16000 else audio
    mlx_whisper.transcribe(short, path_or_hf_repo=whisper_model)

    results = []
    for _ in range(runs):
        t0 = time.perf_counter()
        mlx_whisper.transcribe(audio, path_or_hf_repo=whisper_model)
        elapsed = time.perf_counter() - t0
        rtf = audio_duration / elapsed
        results.append(rtf)

    results.sort()
    result_queue.put(("whisper", results[len(results) // 2]))


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

def run_isolated(model_id, whisper_model, audio, kv_size, decode_tokens, runs):
    """Run LLM and Whisper separately (sequential, no contention)."""
    q = mp.Queue()

    print("  LLM decode (isolated)...", end="", flush=True)
    p = mp.Process(target=_llm_decode_worker,
                   args=(q, model_id, kv_size, decode_tokens, runs))
    p.start()
    p.join()
    _, llm_tps = q.get()
    print(f" {llm_tps:.1f} tok/s")

    print("  Whisper    (isolated)...", end="", flush=True)
    p = mp.Process(target=_whisper_worker,
                   args=(q, whisper_model, audio, runs))
    p.start()
    p.join()
    _, whisper_rtf = q.get()
    print(f" {whisper_rtf:.2f}x real-time")

    return llm_tps, whisper_rtf


def run_concurrent(model_id, whisper_model, audio, kv_size, decode_tokens, runs):
    """Run LLM and Whisper simultaneously (contention)."""
    q = mp.Queue()

    print("  LLM + Whisper (concurrent)...", end="", flush=True)
    p_llm = mp.Process(target=_llm_decode_worker,
                       args=(q, model_id, kv_size, decode_tokens, runs))
    p_whisper = mp.Process(target=_whisper_worker,
                           args=(q, whisper_model, audio, runs))

    p_llm.start()
    p_whisper.start()
    p_llm.join()
    p_whisper.join()

    results = {}
    while not q.empty():
        key, val = q.get()
        results[key] = val

    llm_tps = results.get("llm", 0.0)
    whisper_rtf = results.get("whisper", 0.0)
    print(f" LLM={llm_tps:.1f} tok/s, Whisper={whisper_rtf:.2f}x")

    return llm_tps, whisper_rtf


def print_results(chip, memory_gb, model_id, whisper_id,
                  llm_isolated, whisper_isolated, llm_concurrent, whisper_concurrent):
    w = 66
    llm_delta = (llm_concurrent - llm_isolated) / llm_isolated * 100
    whisper_delta = (whisper_concurrent - whisper_isolated) / whisper_isolated * 100
    worst_delta = min(llm_delta, whisper_delta)

    print()
    print("=" * w)
    print(f"  Bandwidth Contention — MLX backend")
    print(f"  Chip    : {chip}")
    print(f"  Memory  : {memory_gb} GB unified")
    print(f"  LLM     : {model_id}")
    print(f"  Whisper : {whisper_id}")
    print("=" * w)
    print()
    print(f"  {'':40s} {'Isolated':>10s}  {'Concurrent':>10s}  {'Δ':>8s}")
    print("  " + "-" * w)
    print(f"  {'LLM decode (tok/s)':40s} {llm_isolated:>10.1f}  {llm_concurrent:>10.1f}  {llm_delta:>+7.1f}%")
    print(f"  {'Whisper (real-time factor)':40s} {whisper_isolated:>9.2f}x  {whisper_concurrent:>9.2f}x  {whisper_delta:>+7.1f}%")
    print()

    abs_delta = abs(worst_delta)
    if abs_delta < 5:
        verdict = "Minimal contention. Comfortable for concurrent use."
    elif abs_delta < 15:
        verdict = "Moderate contention. Noticeable under sustained load."
    else:
        verdict = (
            f"Significant contention ({abs_delta:.1f}%). Both models are competing for\n"
            f"    the same memory bus. This will produce audible jitter in a real\n"
            f"    speech pipeline."
        )
    print(f"  → {verdict}")
    print("=" * w)


def plot_contention(chip, memory_gb, model_id, whisper_id,
                    llm_isolated, whisper_isolated,
                    llm_concurrent, whisper_concurrent,
                    output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("pip install matplotlib to enable charts")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_delta = (llm_concurrent - llm_isolated) / llm_isolated * 100
    whisper_delta = (whisper_concurrent - whisper_isolated) / whisper_isolated * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Bandwidth Contention — LLM + Whisper\n"
        f"{chip}  ·  {memory_gb}GB unified  ·  MLX",
        fontsize=10, fontweight="bold",
    )

    bar_width = 0.35
    iso_color = "#2563eb"
    con_color = "#dc2626"

    # LLM decode
    bars1 = ax1.bar(
        [0 - bar_width / 2, 0 + bar_width / 2],
        [llm_isolated, llm_concurrent],
        width=bar_width,
        color=[iso_color, con_color],
        edgecolor="white", linewidth=1.5,
    )
    ax1.set_xticks([0 - bar_width / 2, 0 + bar_width / 2])
    ax1.set_xticklabels(["Isolated", "Concurrent"], fontsize=9)
    ax1.set_ylabel("Tokens / second", fontsize=9)
    ax1.set_title("LLM Decode", fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(bottom=0, top=max(llm_isolated, llm_concurrent) * 1.25)
    for bar in bars1:
        ax1.annotate(
            f"{bar.get_height():.1f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9,
        )
    ax1.annotate(
        f"{llm_delta:+.1f}%",
        xy=(0, max(llm_isolated, llm_concurrent) * 1.12),
        ha="center", fontsize=11, fontweight="bold",
        color=con_color,
    )

    # Whisper RTF
    bars2 = ax2.bar(
        [0 - bar_width / 2, 0 + bar_width / 2],
        [whisper_isolated, whisper_concurrent],
        width=bar_width,
        color=[iso_color, con_color],
        edgecolor="white", linewidth=1.5,
    )
    ax2.set_xticks([0 - bar_width / 2, 0 + bar_width / 2])
    ax2.set_xticklabels(["Isolated", "Concurrent"], fontsize=9)
    ax2.set_ylabel("Real-time factor (higher = faster)", fontsize=9)
    ax2.set_title("Whisper ASR", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(bottom=0, top=max(whisper_isolated, whisper_concurrent) * 1.25)
    for bar in bars2:
        ax2.annotate(
            f"{bar.get_height():.2f}x",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9,
        )
    ax2.annotate(
        f"{whisper_delta:+.1f}%",
        xy=(0, max(whisper_isolated, whisper_concurrent) * 1.12),
        ha="center", fontsize=11, fontweight="bold",
        color=con_color,
    )

    legend_elements = [
        Patch(facecolor=iso_color, label="Isolated (full bus)"),
        Patch(facecolor=con_color, label="Concurrent (shared bus)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    safe_model = model_id.replace("/", "__").replace(":", "-")
    safe_chip = chip.replace(" ", "-")
    out_path = output_dir / f"contention_{safe_chip}_{safe_model}.png"
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(
        description="Benchmark LLM + Whisper concurrent bandwidth contention (MLX).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="MLX LLM model ID (e.g. mlx-community/Llama-3.2-3B-Instruct-4bit)")
    p.add_argument("--whisper-model", default="mlx-community/whisper-tiny",
                   help="MLX Whisper model ID")
    p.add_argument("--audio", default=None,
                   help="Path to audio file (WAV). Uses noise if not provided.")
    p.add_argument("--kv-size", type=int, default=512,
                   help="KV cache size (tokens) for LLM decode.")
    p.add_argument("--decode-tokens", type=int, default=100,
                   help="Tokens to generate per LLM decode measurement.")
    p.add_argument("--runs", type=int, default=3,
                   help="Timing runs per measurement (median used).")
    p.add_argument("--plot", action="store_true",
                   help="Save a contention chart to results/")
    p.add_argument("--output-dir", default="results",
                   help="Directory for optional chart output.")
    args = p.parse_args()

    try:
        import mlx.core  # noqa: F401
    except ImportError:
        print("MLX not available. Install with: pip install mlx-lm")
        sys.exit(1)

    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        print("mlx-whisper not available. Install with: pip install mlx-whisper")
        sys.exit(1)

    chip = _chip()
    memory_gb = _memory_gb()
    audio = _make_audio(args.audio)
    audio_sec = len(audio) / 16000

    print(f"\nBandwidth Contention Benchmark")
    print(f"Chip    : {chip}")
    print(f"Memory  : {memory_gb} GB unified")
    print(f"LLM     : {args.model}")
    print(f"Whisper : {args.whisper_model}")
    print(f"Audio   : {audio_sec:.1f}s {'(noise)' if args.audio is None else args.audio}")
    print()

    print("Isolated benchmarks:")
    llm_isolated, whisper_isolated = run_isolated(
        args.model, args.whisper_model, audio,
        args.kv_size, args.decode_tokens, args.runs,
    )

    print("\nConcurrent benchmark:")
    llm_concurrent, whisper_concurrent = run_concurrent(
        args.model, args.whisper_model, audio,
        args.kv_size, args.decode_tokens, args.runs,
    )

    print_results(
        chip, memory_gb, args.model, args.whisper_model,
        llm_isolated, whisper_isolated,
        llm_concurrent, whisper_concurrent,
    )

    if args.plot:
        chart = plot_contention(
            chip, memory_gb, args.model, args.whisper_model,
            llm_isolated, whisper_isolated,
            llm_concurrent, whisper_concurrent,
            args.output_dir,
        )
        if chart:
            print(f"\nChart saved → {chart}")


if __name__ == "__main__":
    main()
