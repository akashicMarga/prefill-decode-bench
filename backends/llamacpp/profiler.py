"""
backends/llamacpp/profiler.py
=============================
llama.cpp backend for prefill-decode-bench — native CLI approach.

Calls the llama-bench binary (built from source) as a subprocess and
parses its JSON output into ProfileRun. This always uses the latest
llama.cpp, so new model architectures work immediately after a rebuild.

Setup:
    ./setup_llamacpp.sh          # one-time clone + build
    ./setup_llamacpp.sh --update # pull latest + rebuild

The binary is found automatically at vendor/llama.cpp/build/bin/llama-bench,
or can be overridden with --llamacpp-bin.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backends import SystemInfo, PrefillResult, DecodeResult, HardwareMetrics, ProfileRun
from backends import print_summary
from backends.types import lookup_bandwidth

REPO_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Locate the llama-bench binary
# ---------------------------------------------------------------------------

def find_llama_bench(override: str | None = None) -> Path:
    """Find llama-bench binary: explicit path > vendor build > $PATH."""
    if override:
        p = Path(override)
        if p.is_file():
            return p
        raise FileNotFoundError(f"--llamacpp-bin not found: {override}")

    vendor_bin = REPO_ROOT / "vendor" / "llama.cpp" / "build" / "bin" / "llama-bench"
    if vendor_bin.is_file():
        return vendor_bin

    which = shutil.which("llama-bench")
    if which:
        return Path(which)

    print("llama-bench binary not found.")
    print("  Run: ./setup_llamacpp.sh")
    print("  Or:  --llamacpp-bin /path/to/llama-bench")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Resolve GGUF model path (local file or HuggingFace download)
# ---------------------------------------------------------------------------

def resolve_model(model_id: str, gguf_file: str) -> Path:
    """Return local path to a .gguf file, downloading from HF if needed."""
    p = Path(model_id)
    if p.suffix == ".gguf" or p.is_file():
        if not p.is_file():
            print(f"Model file not found: {model_id}")
            sys.exit(1)
        return p

    try:
        from huggingface_hub import hf_hub_download, HfApi
    except ImportError:
        print("Loading from HuggingFace requires: pip install huggingface-hub")
        sys.exit(1)

    import fnmatch

    filename = gguf_file
    if "*" in gguf_file or "?" in gguf_file:
        print(f"  Resolving {gguf_file} in {model_id}...")
        api = HfApi()
        files = [
            f.rfilename for f in api.model_info(model_id).siblings
            if f.rfilename.endswith(".gguf")
        ]
        matches = fnmatch.filter(files, gguf_file)
        if not matches:
            print(f"No files matching '{gguf_file}' in {model_id}")
            print(f"Available GGUF files: {files}")
            sys.exit(1)
        filename = matches[0]

    print(f"  Downloading {model_id} / {filename}...")
    local = hf_hub_download(repo_id=model_id, filename=filename)
    return Path(local)


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def _chip() -> str:
    if platform.system() == "Darwin":
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
    try:
        out = subprocess.check_output(
            ["lscpu"], text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if "Model name" in line:
                return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def _memory_gb() -> float:
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True, stderr=subprocess.DEVNULL,
            )
            return round(int(out.strip()) / (1024 ** 3), 1)
        except Exception:
            pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2), 1)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Run llama-bench and parse JSON
# ---------------------------------------------------------------------------

def _run_bench(bench_bin: Path, model_path: Path, args: list[str]) -> list[dict]:
    """Run llama-bench with given args, return parsed JSON array."""
    cmd = [str(bench_bin), "-m", str(model_path), "-o", "json"] + args

    print(f"  $ {' '.join(cmd[-8:])}")  # show tail of command
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        print(f"llama-bench failed (exit {result.returncode}):")
        if stderr:
            for line in stderr.splitlines()[-10:]:
                print(f"  {line}")
        sys.exit(1)

    stdout = result.stdout.strip()
    if not stdout:
        print("llama-bench produced no output.")
        sys.exit(1)

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        lines = stdout.splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith("[") or line.startswith("{"):
                try:
                    data = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        else:
            print("Could not parse llama-bench JSON output.")
            print("Raw output (last 5 lines):")
            for l in lines[-5:]:
                print(f"  {l}")
            sys.exit(1)

    if isinstance(data, dict):
        data = [data]
    return data


def _extract_model_info(entries: list[dict]) -> dict:
    """Extract model metadata from llama-bench JSON entries."""
    if not entries:
        return {}
    e = entries[0]
    return {
        "model_filename": e.get("model_filename", ""),
        "model_type": e.get("model_type", ""),
        "model_size": e.get("model_size", 0),
        "model_n_params": e.get("model_n_params", 0),
        "n_gpu_layers": e.get("n_gpu_layers", 0),
        "n_threads": e.get("n_threads", 0),
        "cpu_info": e.get("cpu_info", ""),
        "gpu_info": e.get("gpu_info", ""),
    }


# ---------------------------------------------------------------------------
# Map llama-bench JSON to our dataclasses
# ---------------------------------------------------------------------------

def _parse_prefill(entries: list[dict]) -> list[PrefillResult]:
    results = []
    for e in entries:
        n_prompt = e.get("n_prompt", 0)
        if n_prompt <= 0:
            continue
        avg_ts = e.get("avg_ts", 0.0)
        avg_ns = e.get("avg_ns", 0.0)
        time_s = avg_ns / 1e9 if avg_ns > 0 else (n_prompt / avg_ts if avg_ts > 0 else 0)

        results.append(PrefillResult(
            prompt_tokens=n_prompt,
            actual_tokens=n_prompt,
            time_seconds=round(time_s, 4),
            tokens_per_second=round(avg_ts, 1),
        ))
    return results


def _parse_decode(entries: list[dict], decode_tokens: int) -> list[DecodeResult]:
    results = []
    for e in entries:
        n_gen = e.get("n_gen", 0)
        if n_gen <= 0:
            continue
        avg_ts = e.get("avg_ts", 0.0)
        avg_ns = e.get("avg_ns", 0.0)
        time_s = avg_ns / 1e9 if avg_ns > 0 else (n_gen / avg_ts if avg_ts > 0 else 0)
        ms_tok = (time_s / n_gen * 1000) if n_gen > 0 else 0

        n_depth = e.get("n_depth", 0)
        prefill_tokens = n_depth if n_depth > 0 else 0

        results.append(DecodeResult(
            prefill_tokens=prefill_tokens,
            decode_tokens=n_gen,
            time_seconds=round(time_s, 4),
            tokens_per_second=round(avg_ts, 1),
            ms_per_token=round(ms_tok, 1),
        ))
    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(model_id, prefill_lengths, decode_kv_sizes, decode_tokens, runs,
        plot, output_dir, gguf_file="*Q4_K_M.gguf", n_gpu_layers=-1,
        llamacpp_bin=None):

    bench_bin = find_llama_bench(llamacpp_bin)

    chip = _chip()
    memory_gb = _memory_gb()
    theoretical_bw = lookup_bandwidth(chip) or 0.0
    is_mac = platform.system() == "Darwin"

    print(f"Chip     : {chip}")
    print(f"Memory   : {memory_gb} GB {'unified' if is_mac else 'system'}")
    if theoretical_bw > 0:
        print(f"Peak BW  : {theoretical_bw} GB/s (theoretical)")
    print(f"Model    : {model_id}")
    print(f"Binary   : {bench_bin}")
    print()

    model_path = resolve_model(model_id, gguf_file)
    print(f"  GGUF file: {model_path}")
    print()

    ngl_str = str(n_gpu_layers) if n_gpu_layers >= 0 else "99"
    repetitions = str(runs)

    # --- Prefill sweep ---
    pp_values = ",".join(str(n) for n in prefill_lengths)
    print(f"Prefill sweep ({len(prefill_lengths)} lengths, {runs} reps each)...")

    prefill_entries = _run_bench(bench_bin, model_path, [
        "-p", pp_values,
        "-n", "0",
        "-ngl", ngl_str,
        "-r", repetitions,
    ])

    info = _extract_model_info(prefill_entries)
    prefill = _parse_prefill(prefill_entries)

    for r in prefill:
        print(f"  Prefill  {r.actual_tokens:>5} tok  →  {r.tokens_per_second:6.1f} tok/s  "
              f"({r.time_seconds*1000:.0f} ms)")

    # --- Decode sweep ---
    print(f"\nDecode sweep ({len(decode_kv_sizes)} KV sizes, {runs} reps, "
          f"{decode_tokens} tokens each)...")

    decode = []
    for kv_size in decode_kv_sizes:
        entries = _run_bench(bench_bin, model_path, [
            "-p", "0",
            "-n", str(decode_tokens),
            "-d", str(kv_size),
            "-ngl", ngl_str,
            "-r", repetitions,
        ])
        batch = _parse_decode(entries, decode_tokens)
        for r in batch:
            r.prefill_tokens = kv_size
        decode.extend(batch)

    for r in decode:
        print(f"  Decode   KV={r.prefill_tokens:>5} tok  →  {r.tokens_per_second:6.1f} tok/s  "
              f"({r.ms_per_token:.1f} ms/tok)")

    # --- Build ProfileRun ---
    model_size_gb = round(info.get("model_size", 0) / 1e9, 3) if info.get("model_size") else 0
    model_params_b = round(info.get("model_n_params", 0) / 1e9, 3) if info.get("model_n_params") else 0

    hw = HardwareMetrics(
        model_size_gb=model_size_gb,
        model_params_b=model_params_b,
        theoretical_bandwidth_gbs=theoretical_bw,
    )

    memory_type = "unified" if is_mac else "ram"
    system = SystemInfo("llamacpp", chip, memory_gb, memory_type, model_id)
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
