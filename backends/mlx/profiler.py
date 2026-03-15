"""
backends/mlx/profiler.py
========================
MLX backend for prefill-decode-bench.

Sync point: mx.eval() — MLX is lazy-evaluated. Timing always wraps eval()
to measure actual Metal execution, not graph construction.

Hardware metrics: peak Metal memory per phase, effective bandwidth,
arithmetic intensity, and TFLOPS via mx.metal memory APIs and model
parameter counting.
"""

import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backends import SystemInfo, PrefillResult, DecodeResult, HardwareMetrics, ProfileRun
from backends import build_prompt, print_summary
from backends.types import lookup_bandwidth


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
# Model introspection
# ---------------------------------------------------------------------------

def _count_params_logical(model) -> int:
    """True parameter count, accounting for quantized weight packing.

    MLX QuantizedLinear stores N weights packed into fewer uint32 elements.
    We recover the logical count from (group_size, bits) stored on each
    quantized layer. For non-quantized params we just count elements.
    """
    import mlx.nn as nn
    total = 0
    for module in model.modules():
        if isinstance(module, nn.QuantizedLinear):
            out_features, packed = module.weight.shape
            bits = module.bits
            in_features = packed * 32 // bits
            total += out_features * in_features
            if hasattr(module, "bias") and module.bias is not None:
                total += module.bias.size
        elif isinstance(module, nn.Linear):
            total += module.weight.size
            if hasattr(module, "bias") and module.bias is not None:
                total += module.bias.size
    if total == 0:
        import mlx.utils
        leaves = mlx.utils.tree_flatten(model.parameters())
        total = sum(v.size for _, v in leaves)
    return total


def _count_params_storage(model) -> int:
    """Storage element count (what actually sits in memory)."""
    import mlx.utils
    leaves = mlx.utils.tree_flatten(model.parameters())
    return sum(v.size for _, v in leaves)


def _model_weight_bytes(model) -> int:
    import mlx.utils
    leaves = mlx.utils.tree_flatten(model.parameters())
    return sum(v.size * v.dtype.size for _, v in leaves)


def _kv_bytes_per_token(model) -> int:
    """Bytes of KV cache read per decode token.

    2 (K+V) × num_layers × kv_heads × head_dim × dtype_bytes.
    KV cache is stored in the cache dtype (usually float16).
    """
    cfg = getattr(model, "args", getattr(model, "config", None))
    if cfg is None:
        return 0
    num_layers = getattr(cfg, "num_hidden_layers", 0)
    num_kv_heads = getattr(cfg, "num_key_value_heads",
                           getattr(cfg, "num_attention_heads", 0))
    head_dim = getattr(cfg, "head_dim", 0)
    if head_dim == 0:
        hidden = getattr(cfg, "hidden_size", 0)
        num_heads = getattr(cfg, "num_attention_heads", 0)
        head_dim = hidden // num_heads if num_heads > 0 else 0
    kv_dtype_bytes = 2  # float16
    return 2 * num_layers * num_kv_heads * head_dim * kv_dtype_bytes


def _reset_peak(mx):
    """Use the non-deprecated API when available."""
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:
        mx.metal.reset_peak_memory()


def _get_peak(mx) -> int:
    if hasattr(mx, "get_peak_memory"):
        return mx.get_peak_memory()
    return mx.metal.get_peak_memory()


# ---------------------------------------------------------------------------
# VLM language-model wrapper (adapts LanguageModelOutput → raw logits)
# ---------------------------------------------------------------------------

class _LMAdapter:
    """Wraps a VLM's language_model so it quacks like an mlx_lm model.

    mlx_lm models return a raw logits tensor from __call__.
    mlx_vlm language models return LanguageModelOutput(logits=...).
    This adapter unwraps that so the rest of the profiler works unchanged.
    """

    def __init__(self, vlm_model):
        self._lm = vlm_model.language_model
        self._vlm = vlm_model

    def __call__(self, *args, **kwargs):
        out = self._lm(*args, **kwargs)
        return out.logits if hasattr(out, "logits") else out

    def modules(self):
        return self._lm.modules()

    def parameters(self):
        return self._lm.parameters()

    def __getattr__(self, name):
        if name in ("_lm", "_vlm"):
            raise AttributeError(name)
        return getattr(self._lm, name)


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

def _measure_prefill(model, tokenizer, mx, prompt_lengths, runs, model_params) -> list[PrefillResult]:
    results = []
    for target in prompt_lengths:
        prompt = build_prompt(tokenizer, target)
        tokens = tokenizer.encode(prompt)
        actual = len(tokens)
        ids = mx.array(tokens)[None]

        times = []
        peak_mems = []
        for _ in range(runs):
            mx.eval(ids)
            _reset_peak(mx)
            t0 = time.perf_counter()
            logits = model(ids)
            mx.eval(logits)
            elapsed = time.perf_counter() - t0
            peak_mems.append(_get_peak(mx))
            times.append(elapsed)

        times.sort()
        peak_mems.sort()
        elapsed = times[len(times) // 2]
        peak_mem_gb = round(peak_mems[len(peak_mems) // 2] / 1e9, 3)
        tps = actual / elapsed

        flops = 2.0 * model_params * actual
        tflops = round(flops / elapsed / 1e12, 3)

        results.append(PrefillResult(
            target, actual, round(elapsed, 4), round(tps, 1),
            peak_memory_gb=peak_mem_gb, tflops=tflops,
        ))
        print(f"  Prefill  {actual:>5} tok  →  {tps:6.1f} tok/s  "
              f"({elapsed*1000:.0f} ms)  mem={peak_mem_gb:.2f}GB  {tflops:.2f} TFLOPS")

    return results


def _measure_decode(model, tokenizer, mx, make_cache, kv_sizes, decode_tokens, runs,
                    logical_params, model_bytes, kv_bytes_tok, theoretical_bw) -> list[DecodeResult]:
    results = []
    for kv_len in kv_sizes:
        prompt = build_prompt(tokenizer, kv_len)
        tokens = tokenizer.encode(prompt)
        actual_kv = len(tokens)

        times = []
        peak_mems = []
        for _ in range(runs):
            ids = mx.array(tokens)[None]
            cache = make_cache(model)
            logits = model(ids, cache=cache)
            mx.eval(logits)

            last = mx.array([[tokens[-1]]])
            _reset_peak(mx)
            t0 = time.perf_counter()
            for _ in range(decode_tokens):
                logits = model(last, cache=cache)
                mx.eval(logits)
                next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                mx.eval(next_tok)
                last = next_tok
            elapsed = time.perf_counter() - t0
            peak_mems.append(_get_peak(mx))
            times.append(elapsed)

        times.sort()
        peak_mems.sort()
        elapsed = times[len(times) // 2]
        peak_mem_gb = round(peak_mems[len(peak_mems) // 2] / 1e9, 3)
        tps = decode_tokens / elapsed
        ms_tok = elapsed / decode_tokens * 1000

        avg_kv_during_decode = actual_kv + decode_tokens // 2
        kv_cache_bytes = kv_bytes_tok * avg_kv_during_decode if kv_bytes_tok > 0 else 0
        bytes_per_token = model_bytes + kv_cache_bytes
        eff_bw = round(bytes_per_token * tps / 1e9, 2)
        bw_util = round(eff_bw / theoretical_bw * 100, 1) if theoretical_bw > 0 else 0.0

        flops_per_token = 2.0 * logical_params
        arith_intensity = round(flops_per_token / bytes_per_token, 3) if bytes_per_token > 0 else 0.0

        results.append(DecodeResult(
            actual_kv, decode_tokens, round(elapsed, 4), round(tps, 1), round(ms_tok, 1),
            peak_memory_gb=peak_mem_gb,
            effective_bandwidth_gbs=eff_bw,
            bandwidth_utilization_pct=bw_util,
            arithmetic_intensity=arith_intensity,
        ))
        kv_mb = kv_cache_bytes / 1e6
        print(f"  Decode   KV={actual_kv:>5} tok  →  {tps:6.1f} tok/s  ({ms_tok:.1f} ms/tok)  "
              f"mem={peak_mem_gb:.2f}GB  BW={eff_bw:.1f}/{theoretical_bw:.0f} GB/s ({bw_util:.0f}%)  "
              f"KV read={kv_mb:.0f}MB")

    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _load_model(model_id, mx):
    """Load an MLX model, falling back to mlx_vlm for vision-language models."""
    # Try mlx_lm (text-only) first
    try:
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        model, tokenizer = load(model_id)
        return model, tokenizer, make_prompt_cache
    except (ImportError, ValueError, Exception) as e:
        mlx_lm_err = e

    # Fall back to mlx_vlm for VLMs
    try:
        from mlx_vlm.utils import load_model
        from mlx_lm.models.cache import make_prompt_cache
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download
        from pathlib import Path as _P

        model_path = _P(snapshot_download(model_id))
        vlm = load_model(model_path)
        model = _LMAdapter(vlm)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"  (loaded as VLM via mlx_vlm, using language_model for benchmarking)")
        return model, tokenizer, make_prompt_cache
    except Exception as vlm_err:
        print(f"Failed to load model with mlx_lm: {mlx_lm_err}")
        print(f"Failed to load model with mlx_vlm: {vlm_err}")
        print("Install with: pip install mlx-lm  or  pip install mlx-vlm")
        sys.exit(1)


def run(model_id, prefill_lengths, decode_kv_sizes, decode_tokens, runs, plot, output_dir):
    try:
        import mlx.core as mx
    except ImportError:
        print("MLX not available. Install with: pip install mlx-lm")
        print("Requires Apple Silicon (M1 or later).")
        sys.exit(1)

    chip = _chip()
    memory_gb = _memory_gb()
    theoretical_bw = lookup_bandwidth(chip) or 0.0

    print(f"Chip   : {chip}")
    print(f"Memory : {memory_gb} GB unified")
    print(f"Model  : {model_id}")
    if theoretical_bw > 0:
        print(f"Peak BW: {theoretical_bw} GB/s (theoretical)")
    print()

    print("Loading model...")
    _reset_peak(mx)
    model, tokenizer, make_prompt_cache = _load_model(model_id, mx)
    mx.eval(model.parameters())
    model_load_mem = _get_peak(mx) / 1e9

    logical_params = _count_params_logical(model)
    storage_params = _count_params_storage(model)
    model_bytes = _model_weight_bytes(model)
    kv_bytes_tok = _kv_bytes_per_token(model)
    print(f"  Parameters : {logical_params / 1e9:.2f}B (logical)  "
          f"{storage_params / 1e9:.2f}B (storage elements)")
    print(f"  Weight size: {model_bytes / 1e9:.2f} GB")
    print(f"  KV / token : {kv_bytes_tok / 1024:.1f} KB")
    print(f"  Metal mem  : {model_load_mem:.2f} GB")
    print()

    _warmup(model, tokenizer, mx)
    print()

    print(f"Prefill sweep ({len(prefill_lengths)} lengths, {runs} runs each)...")
    prefill = _measure_prefill(model, tokenizer, mx, prefill_lengths, runs, logical_params)

    print(f"\nDecode sweep ({len(decode_kv_sizes)} KV sizes, {runs} runs, "
          f"{decode_tokens} tokens each)...")
    decode = _measure_decode(model, tokenizer, mx, make_prompt_cache, decode_kv_sizes, decode_tokens, runs,
                             logical_params, model_bytes, kv_bytes_tok, theoretical_bw)

    hw = HardwareMetrics(
        model_size_gb=round(model_bytes / 1e9, 3),
        model_params_b=round(logical_params / 1e9, 3),
        theoretical_bandwidth_gbs=theoretical_bw,
        model_load_memory_gb=round(model_load_mem, 3),
    )

    system = SystemInfo("mlx", chip, memory_gb, "unified", model_id)
    profile = ProfileRun(system, prefill, decode, hardware=hw)

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
