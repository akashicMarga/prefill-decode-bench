"""
backends/mlx/profiler.py
========================
MLX backend for prefill-decode-bench.

Sync point: mx.eval() — MLX is lazy-evaluated. Timing always wraps eval()
to measure actual Metal execution, not graph construction.
"""

import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backends import SystemInfo, PrefillResult, DecodeResult, ProfileRun
from backends import build_prompt, print_summary


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _warmup(model, tokenizer, mx):
    print("  Warming up Metal kernels...", end="", flush=True)
    tokens = tokenizer.encode("Warmup pass to compile Metal kernels.")
    ids = mx.array(tokens)[None]
    logits = model(ids)
    mx.eval(logits)
    print(" done")


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def _measure_prefill(model, tokenizer, mx, prompt_lengths, runs) -> list[PrefillResult]:
    results = []
    for target in prompt_lengths:
        prompt = build_prompt(tokenizer, target)
        tokens = tokenizer.encode(prompt)
        actual = len(tokens)
        ids = mx.array(tokens)[None]

        times = []
        for _ in range(runs):
            mx.eval(ids)
            t0 = time.perf_counter()
            logits = model(ids)
            mx.eval(logits)
            times.append(time.perf_counter() - t0)

        times.sort()
        elapsed = times[len(times) // 2]
        tps = actual / elapsed
        results.append(PrefillResult(target, actual, round(elapsed, 4), round(tps, 1)))
        print(f"  Prefill  {actual:>5} tok  →  {tps:6.1f} tok/s  ({elapsed*1000:.0f} ms)")

    return results


def _measure_decode(model, tokenizer, mx, make_cache, kv_sizes, decode_tokens, runs) -> list[DecodeResult]:
    results = []
    for kv_len in kv_sizes:
        prompt = build_prompt(tokenizer, kv_len)
        tokens = tokenizer.encode(prompt)
        actual_kv = len(tokens)

        times = []
        for _ in range(runs):
            ids = mx.array(tokens)[None]
            cache = make_cache(model)
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
            times.append(time.perf_counter() - t0)

        times.sort()
        elapsed = times[len(times) // 2]
        tps = decode_tokens / elapsed
        ms_tok = elapsed / decode_tokens * 1000
        results.append(DecodeResult(actual_kv, decode_tokens, round(elapsed, 4),
                                    round(tps, 1), round(ms_tok, 1)))
        print(f"  Decode   KV={actual_kv:>5} tok  →  {tps:6.1f} tok/s  ({ms_tok:.1f} ms/tok)")

    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(model_id, prefill_lengths, decode_kv_sizes, decode_tokens, runs, plot, output_dir):
    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
    except ImportError:
        print("MLX not available. Install with: pip install mlx-lm")
        print("Requires Apple Silicon (M1 or later).")
        sys.exit(1)

    chip = _chip()
    memory_gb = _memory_gb()

    print(f"Chip   : {chip}")
    print(f"Memory : {memory_gb} GB unified")
    print(f"Model  : {model_id}\n")

    print("Loading model...")
    model, tokenizer = load(model_id)
    print("Model loaded.\n")

    _warmup(model, tokenizer, mx)
    print()

    print(f"Prefill sweep ({len(prefill_lengths)} lengths, {runs} runs each)...")
    prefill = _measure_prefill(model, tokenizer, mx, prefill_lengths, runs)

    print(f"\nDecode sweep ({len(decode_kv_sizes)} KV sizes, {runs} runs, "
          f"{decode_tokens} tokens each)...")
    decode = _measure_decode(model, tokenizer, mx, make_prompt_cache, decode_kv_sizes, decode_tokens, runs)

    system = SystemInfo("mlx", chip, memory_gb, "unified", model_id)
    profile = ProfileRun(system, prefill, decode)

    print_summary(profile)

    out = Path(output_dir)
    json_path = profile.save(out)
    print(f"\nResults saved → {json_path}")

    if plot:
        try:
            from plot_results import plot_single
            chart = plot_single(profile, out)
            if chart:
                print(f"Chart saved   → {chart}")
        except ImportError:
            print("pip install matplotlib to enable charts")
