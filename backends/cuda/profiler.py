"""
backends/cuda/profiler.py
=========================
CUDA backend for prefill-decode-bench.

Uses HuggingFace transformers with PyTorch. Works on any Nvidia GPU
with CUDA support. Also runs on CPU (slowly) for testing.

Sync point: torch.cuda.synchronize() — CUDA operations are async by default.
All timing wraps synchronize() to measure actual GPU execution time.

Model notes:
  - Loads in float16 by default (use --dtype bfloat16 for Ampere+)
  - Quantized models: pass any GPTQ/AWQ model ID — transformers handles loading
  - For 4-bit: pip install bitsandbytes and pass --load-in-4bit (via HF AutoModelForCausalLM)
"""

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

def _gpu_info() -> tuple[str, float]:
    """Return (gpu_name, vram_gb). Falls back gracefully if torch not available."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return name, round(vram, 1)
    except Exception:
        pass
    return "CPU", 0.0


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _warmup(model, tokenizer, device):
    import torch
    print("  Warming up CUDA kernels...", end="", flush=True)
    ids = tokenizer("Warmup pass.", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        model(ids)
    if device != "cpu":
        torch.cuda.synchronize()
    print(" done")


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def _measure_prefill(model, tokenizer, device, prompt_lengths, runs) -> list[PrefillResult]:
    import torch
    use_cuda = device != "cpu"
    results = []

    for target in prompt_lengths:
        prompt = build_prompt(tokenizer, target)
        tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        actual = tokens.shape[1]

        times = []
        for _ in range(runs):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(tokens, use_cache=True)
            if use_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        times.sort()
        elapsed = times[len(times) // 2]
        tps = actual / elapsed
        results.append(PrefillResult(target, actual, round(elapsed, 4), round(tps, 1)))
        print(f"  Prefill  {actual:>5} tok  →  {tps:6.1f} tok/s  ({elapsed*1000:.0f} ms)")

    return results


def _measure_decode(model, tokenizer, device, kv_sizes, decode_tokens, runs) -> list[DecodeResult]:
    import torch
    use_cuda = device != "cpu"
    results = []

    for kv_len in kv_sizes:
        prompt = build_prompt(tokenizer, kv_len)
        tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        actual_kv = tokens.shape[1]

        times = []
        for _ in range(runs):
            # Build KV cache via prefill
            with torch.no_grad():
                prefill_out = model(tokens, use_cache=True)
            past_kv = prefill_out.past_key_values

            # Last token as the starting point for decode
            last = tokens[:, -1:]

            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(decode_tokens):
                with torch.no_grad():
                    out = model(last, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                # Greedy: take the argmax — we don't care about output quality
                last = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if use_cuda:
                torch.cuda.synchronize()
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
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("CUDA backend requires: pip install torch transformers accelerate")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: no CUDA GPU found, running on CPU. Results will be slow.")

    gpu_name, vram_gb = _gpu_info()

    print(f"Device : {device.upper()}")
    print(f"GPU    : {gpu_name}")
    print(f"VRAM   : {vram_gb} GB")
    print(f"Model  : {model_id}\n")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.\n")

    _warmup(model, tokenizer, device)
    print()

    print(f"Prefill sweep ({len(prefill_lengths)} lengths, {runs} runs each)...")
    prefill = _measure_prefill(model, tokenizer, device, prefill_lengths, runs)

    print(f"\nDecode sweep ({len(decode_kv_sizes)} KV sizes, {runs} runs, "
          f"{decode_tokens} tokens each)...")
    decode = _measure_decode(model, tokenizer, device, decode_kv_sizes, decode_tokens, runs)

    memory_type = "vram" if device == "cuda" else "ram"
    system = SystemInfo("cuda", gpu_name, vram_gb, memory_type, model_id)
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
