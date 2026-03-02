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
    w = 64
    print()
    print("=" * w)
    print(f"  prefill-decode-bench")
    print(f"  Backend : {s.backend.upper()}")
    print(f"  Chip    : {s.chip}")
    print(f"  Memory  : {s.memory_gb} GB ({s.memory_type})")
    print(f"  Model   : {s.model}")
    print("=" * w)

    print()
    print("PREFILL  (compute-bound — all input tokens processed in parallel)")
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
    print("=" * w)
