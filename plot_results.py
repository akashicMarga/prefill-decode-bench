"""
plot_results.py
===============
Visualize profiler results. Works with output from either backend.

Usage:
    # Single run
    python plot_results.py results/profile_mlx_Apple-M3-Max_....json

    # Compare two runs (different models or chips — overlaid on same axes)
    python plot_results.py results/profile_mlx_...json results/profile_cuda_...json

    # All JSON files in results/
    python plot_results.py results/

Requirements:
    pip install matplotlib
"""

import argparse
import sys
from pathlib import Path

from backends.types import ProfileRun


COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"]


def short_label(run: ProfileRun) -> str:
    model = run.system.model.split("/")[-1]
    for suffix in ["-Instruct", "-instruct", "-chat", "-hf"]:
        model = model.replace(suffix, "")
    chip = run.system.chip.split()[-1]   # "M3" from "Apple M3 Max"
    return f"{model[:22]} ({chip})"


def _has_hw_metrics(run: ProfileRun) -> bool:
    return (any(r.tflops > 0 for r in run.prefill) or
            any(r.effective_bandwidth_gbs > 0 for r in run.decode))


def plot_single(run: ProfileRun, output_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Prefill vs Decode — {run.system.model}\n"
        f"{run.system.chip}  ·  {run.system.memory_gb}GB {run.system.memory_type}  ·  {run.system.backend.upper()}",
        fontsize=10, fontweight="bold",
    )

    # Prefill
    x1 = [r.actual_tokens for r in run.prefill]
    y1 = [r.tokens_per_second for r in run.prefill]
    ax1.plot(x1, y1, marker="o", lw=2, color="#2563eb")
    ax1.fill_between(x1, y1, alpha=0.07, color="#2563eb")
    ax1.set_title("Prefill (compute-bound)", fontsize=10)
    ax1.set_xlabel("Prompt length (tokens)", fontsize=9)
    ax1.set_ylabel("Tokens / second", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    for xi, yi in zip(x1, y1):
        ax1.annotate(f"{yi:.0f}", (xi, yi), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=8)

    # Decode
    x2 = [r.prefill_tokens for r in run.decode]
    y2 = [r.tokens_per_second for r in run.decode]
    ax2.plot(x2, y2, marker="o", lw=2, color="#dc2626")
    ax2.fill_between(x2, y2, alpha=0.07, color="#dc2626")
    ax2.set_title("Decode (memory-bandwidth-bound)", fontsize=10)
    ax2.set_xlabel("KV cache size / conversation context (tokens)", fontsize=9)
    ax2.set_ylabel("Tokens / second", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    for xi, yi in zip(x2, y2):
        ax2.annotate(f"{yi:.0f}", (xi, yi), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=8)
    if len(x2) > 1:
        ax2.annotate(
            "Slows as KV cache grows\n(more bytes read per token)",
            xy=(x2[-1], y2[-1]),
            xytext=(x2[len(x2) // 2], max(y2) * 0.35),
            fontsize=8, color="#dc2626",
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2),
        )

    safe = run.system.model.replace("/", "__").replace(":", "-")
    out = output_dir / f"profile_{run.system.backend}_{safe}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if _has_hw_metrics(run):
        hw_out = plot_hardware(run, output_dir)
        if hw_out:
            print(f"HW chart  → {hw_out}")

    return out


def plot_hardware(run: ProfileRun, output_dir: Path) -> Path | None:
    """Generate hardware metrics chart: memory, bandwidth utilization, TFLOPS."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    has_bw = any(r.effective_bandwidth_gbs > 0 for r in run.decode)
    has_tflops = any(r.tflops > 0 for r in run.prefill)
    has_mem = any(r.peak_memory_gb > 0 for r in run.prefill) or any(r.peak_memory_gb > 0 for r in run.decode)

    ncols = sum([has_mem, has_bw, has_tflops])
    if ncols == 0:
        return None

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    fig.suptitle(
        f"Hardware Metrics — {run.system.model}\n"
        f"{run.system.chip}  ·  {run.system.memory_gb}GB {run.system.memory_type}  ·  {run.system.backend.upper()}",
        fontsize=10, fontweight="bold",
    )

    ax_idx = 0

    # Panel 1: Peak memory per phase
    if has_mem:
        ax = axes[ax_idx]; ax_idx += 1
        prefill_mems = [(r.actual_tokens, r.peak_memory_gb) for r in run.prefill if r.peak_memory_gb > 0]
        decode_mems = [(r.prefill_tokens, r.peak_memory_gb) for r in run.decode if r.peak_memory_gb > 0]

        if prefill_mems:
            px, py = zip(*prefill_mems)
            ax.plot(px, py, marker="o", lw=2, color="#2563eb", label="Prefill")
        if decode_mems:
            dx, dy = zip(*decode_mems)
            ax.plot(dx, dy, marker="s", lw=2, color="#dc2626", label="Decode")

        if run.hardware and run.hardware.model_load_memory_gb > 0:
            ax.axhline(run.hardware.model_load_memory_gb, ls="--", color="#666", alpha=0.6, lw=1)
            ax.annotate(f"Model weights: {run.hardware.model_load_memory_gb:.2f} GB",
                        xy=(0.02, run.hardware.model_load_memory_gb),
                        xycoords=("axes fraction", "data"),
                        fontsize=7, color="#666", va="bottom")

        ax.set_title("Peak Metal Memory", fontsize=10)
        ax.set_xlabel("Tokens (prompt / KV cache)", fontsize=9)
        ax.set_ylabel("Peak memory (GB)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    # Panel 2: Effective bandwidth + utilization
    if has_bw:
        ax = axes[ax_idx]; ax_idx += 1
        x = [r.prefill_tokens for r in run.decode]
        y_bw = [r.effective_bandwidth_gbs for r in run.decode]
        y_util = [r.bandwidth_utilization_pct for r in run.decode]

        ax.plot(x, y_bw, marker="o", lw=2, color="#dc2626", label="Effective BW")

        if run.hardware and run.hardware.theoretical_bandwidth_gbs > 0:
            peak = run.hardware.theoretical_bandwidth_gbs
            ax.axhline(peak, ls="--", color="#16a34a", alpha=0.7, lw=1.5,
                       label=f"Theoretical peak ({peak:.0f} GB/s)")
            ax.fill_between(x, y_bw, alpha=0.07, color="#dc2626")

        for xi, yi, ui in zip(x, y_bw, y_util):
            ax.annotate(f"{yi:.1f}\n({ui:.0f}%)", (xi, yi), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=7)

        ax.set_title("Effective Memory Bandwidth (Decode)", fontsize=10)
        ax.set_xlabel("KV cache size (tokens)", fontsize=9)
        ax.set_ylabel("GB/s", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    # Panel 3: Prefill TFLOPS
    if has_tflops:
        ax = axes[ax_idx]; ax_idx += 1
        x = [r.actual_tokens for r in run.prefill]
        y = [r.tflops for r in run.prefill]

        ax.bar(x, y, width=[t * 0.3 for t in x], color="#7c3aed", alpha=0.8)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=8)

        ax.set_title("Prefill Compute (TFLOPS)", fontsize=10)
        ax.set_xlabel("Prompt length (tokens)", fontsize=9)
        ax.set_ylabel("TFLOPS", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    safe = run.system.model.replace("/", "__").replace(":", "-")
    safe_chip = run.system.chip.replace(" ", "-")
    out = output_dir / f"hardware_{run.system.backend}_{safe_chip}_{safe}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_compare(runs: list[ProfileRun], output_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("prefill-decode-bench — comparison", fontsize=11, fontweight="bold")

    for i, run in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        label = short_label(run)

        x1 = [r.actual_tokens for r in run.prefill]
        y1 = [r.tokens_per_second for r in run.prefill]
        ax1.plot(x1, y1, marker="o", lw=2, color=c, label=label)

        x2 = [r.prefill_tokens for r in run.decode]
        y2 = [r.tokens_per_second for r in run.decode]
        ax2.plot(x2, y2, marker="o", lw=2, color=c, label=label)

    for ax, title, xlabel in [
        (ax1, "Prefill (compute-bound)", "Prompt length (tokens)"),
        (ax2, "Decode (memory-bandwidth-bound)", "KV cache size (tokens)"),
    ]:
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Tokens / second", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    out = output_dir / "comparison.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(description="Plot profiler results.")
    p.add_argument("inputs", nargs="+", help="JSON file(s) or a results/ directory.")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    paths = []
    for inp in args.inputs:
        pt = Path(inp)
        if pt.is_dir():
            paths.extend(sorted(pt.glob("profile_*.json")))
        elif pt.is_file():
            paths.append(pt)

    if not paths:
        print("No JSON files found.")
        sys.exit(1)

    runs = []
    for path in paths:
        try:
            runs.append(ProfileRun.load(path))
            print(f"Loaded: {path.name}")
        except Exception as e:
            print(f"Skipping {path.name}: {e}")

    if not runs:
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else paths[0].parent

    if len(runs) == 1:
        chart = plot_single(runs[0], out_dir)
    else:
        chart = plot_compare(runs, out_dir)
        # Also save individual charts
        for run in runs:
            plot_single(run, out_dir)

    if chart:
        print(f"Chart → {chart}")


if __name__ == "__main__":
    main()
