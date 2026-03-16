"""
Shared utilities used by both backends.
No MLX or PyTorch imports — purely Python.
"""

from .types import ProfileRun

BENCH_SENTENCE = (
    "The memory bandwidth of a processor determines how quickly "
    "data can be moved from memory to the compute units during inference. "
)


def build_prompt(tokenizer, target_tokens: int) -> str:
    """
    Repeat a technical sentence until we hit approximately target_tokens.
    Works with any tokenizer that has .encode() and .decode().
    """
    sample = tokenizer.encode(BENCH_SENTENCE)
    per_sentence = max(1, len(sample))
    repeats = max(1, target_tokens // per_sentence)
    text = BENCH_SENTENCE * repeats
    tokens = tokenizer.encode(text)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
    return tokenizer.decode(tokens)


def print_summary(run: ProfileRun):
    s = run.system
    has_hw = any(r.tflops > 0 for r in run.prefill) or any(r.effective_bandwidth_gbs > 0 for r in run.decode)
    w = 80 if has_hw else 64
    print()
    print("=" * w)
    print(f"  prefill-decode-bench")
    print(f"  Backend : {s.backend.upper()}")
    print(f"  Chip    : {s.chip}")
    print(f"  Memory  : {s.memory_gb} GB ({s.memory_type})")
    print(f"  Model   : {s.model}")
    if run.hardware:
        h = run.hardware
        print(f"  Params  : {h.model_params_b:.2f}B  ({h.model_size_gb:.2f} GB weights)")
        if h.theoretical_bandwidth_gbs > 0:
            print(f"  Peak BW : {h.theoretical_bandwidth_gbs:.0f} GB/s (theoretical)")
    print("=" * w)

    print()
    print("PREFILL  (compute-bound — all input tokens processed in parallel)")
    if has_hw:
        print(f"  {'Tokens':>8}  {'tok/s':>10}  {'ms':>8}  {'TFLOPS':>8}  {'Peak mem':>10}")
        print("  " + "-" * 54)
        for r in run.prefill:
            ms = r.time_seconds * 1000
            print(
                f"  {r.actual_tokens:>8}  {r.tokens_per_second:>10.1f}"
                f"  {ms:>8.0f}  {r.tflops:>8.2f}  {r.peak_memory_gb:>9.2f}G"
            )
    else:
        print(f"  {'Tokens':>8}  {'tok/s':>10}  {'ms':>8}  {'ms/tok':>8}")
        print("  " + "-" * 42)
        for r in run.prefill:
            ms = r.time_seconds * 1000
            print(
                f"  {r.actual_tokens:>8}  {r.tokens_per_second:>10.1f}"
                f"  {ms:>8.0f}  {ms/max(1,r.actual_tokens):>8.2f}"
            )

    print()
    print("DECODE   (memory-bandwidth-bound — sequential, reads all weights + KV cache per token)")
    if has_hw:
        print(f"  {'KV cache':>10}  {'tok/s':>10}  {'ms/tok':>8}  {'BW GB/s':>9}  {'BW util':>8}  {'Peak mem':>10}")
        print("  " + "-" * 66)
        for r in run.decode:
            print(
                f"  {r.prefill_tokens:>10}  {r.tokens_per_second:>10.1f}"
                f"  {r.ms_per_token:>8.1f}  {r.effective_bandwidth_gbs:>9.1f}"
                f"  {r.bandwidth_utilization_pct:>7.0f}%  {r.peak_memory_gb:>9.2f}G"
            )
    else:
        print(f"  {'KV cache':>10}  {'tok/s':>10}  {'ms/tok':>8}")
        print("  " + "-" * 34)
        for r in run.decode:
            print(
                f"  {r.prefill_tokens:>10}  {r.tokens_per_second:>10.1f}"
                f"  {r.ms_per_token:>8.1f}"
            )

    print()
    if run.decode:
        fastest = max(run.decode, key=lambda r: r.tokens_per_second)
        slowest = min(run.decode, key=lambda r: r.tokens_per_second)
        pct = (fastest.tokens_per_second - slowest.tokens_per_second) / fastest.tokens_per_second * 100
        print(f"  Decode degradation  KV={fastest.prefill_tokens}→{slowest.prefill_tokens}: {pct:.1f}%")
        if pct < 10:
            note = "Minimal KV cache pressure at this context range."
        elif pct < 25:
            note = "Moderate — long sessions will feel slower than short ones."
        else:
            note = "Significant — bandwidth is the bottleneck for long conversations."
        print(f"  {note}")

    if has_hw and run.decode:
        avg_bw = sum(r.effective_bandwidth_gbs for r in run.decode) / len(run.decode)
        avg_util = sum(r.bandwidth_utilization_pct for r in run.decode) / len(run.decode)
        print(f"\n  Avg effective bandwidth: {avg_bw:.1f} GB/s ({avg_util:.0f}% of theoretical)")
        if run.decode[0].arithmetic_intensity > 0:
            ai = run.decode[0].arithmetic_intensity
            print(f"  Arithmetic intensity: {ai:.2f} FLOPs/byte — {'bandwidth-bound' if ai < 10 else 'compute-bound'}")

    if run.speculative_decode:
        spec = run.speculative_decode
        draft_name = spec[0].draft_model or "?"
        ndraft = spec[0].num_draft_tokens
        print()
        print(f"SPECULATIVE DECODE  (draft: {draft_name}, {ndraft} tokens/step)")
        print(f"  {'KV cache':>10}  {'tok/s':>10}  {'ms/tok':>8}  {'accept':>8}  {'vs vanilla':>10}")
        print("  " + "-" * 56)

        vanilla_by_kv = {r.prefill_tokens: r for r in run.decode}
        for r in spec:
            vanilla = vanilla_by_kv.get(r.prefill_tokens)
            speedup = r.tokens_per_second / vanilla.tokens_per_second if vanilla and vanilla.tokens_per_second > 0 else 0
            speedup_str = f"{speedup:.2f}x" if speedup > 0 else "—"
            print(
                f"  {r.prefill_tokens:>10}  {r.tokens_per_second:>10.1f}"
                f"  {r.ms_per_token:>8.1f}  {r.acceptance_rate:>7.0%}"
                f"  {speedup_str:>10}"
            )

        if len(spec) > 0:
            avg_acc = sum(r.acceptance_rate for r in spec) / len(spec)
            avg_speedup_parts = []
            for r in spec:
                v = vanilla_by_kv.get(r.prefill_tokens)
                if v and v.tokens_per_second > 0:
                    avg_speedup_parts.append(r.tokens_per_second / v.tokens_per_second)
            avg_speedup = sum(avg_speedup_parts) / len(avg_speedup_parts) if avg_speedup_parts else 0
            print(f"\n  Avg acceptance rate: {avg_acc:.0%}")
            if avg_speedup > 0:
                print(f"  Avg speedup vs vanilla decode: {avg_speedup:.2f}x")

    print("=" * w)
