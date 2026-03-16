# 006 — Speculative Decoding: Qwen3.5-4B + 0.8B Draft

**Date:** 2026-03-16
**Hardware:** Apple M1, 16 GB unified memory, 68.25 GB/s theoretical bandwidth
**Purpose:** Test speculative decoding on a hybrid Mamba+Attention architecture

**Main model:** Qwen3.5-4B Q4_K_M (2.54 GiB GGUF / 2.37 GB MLX 4-bit)
**Draft model:** Qwen3.5-0.8B Q4_K_M (497 MiB GGUF / 0.44 GB MLX 4-bit)
**Draft tokens per step:** 4

---

## MLX Results (Metal GPU)

| Mode | KV=45 tok/s | KV=243 tok/s | KV=485 tok/s | Accept % |
|------|-------------|--------------|--------------|----------|
| Vanilla decode | 19.6 | 18.9 | 16.0 | — |
| Speculative | 18.1 | 20.6 | 17.1 | 79.6% |
| Speedup | **0.92x** | **1.09x** | **1.07x** | |

## llama.cpp Results (Metal GPU, ctx=2048)

| Mode | tok/s | Accept % | Notes |
|------|-------|----------|-------|
| Vanilla (`llama-bench`) | 14.1 | — | Clean baseline |
| `llama-speculative-simple` | 17.6 | 73.7% | M-RoPE errors throughout |
| `llama-speculative` (full) | 19.3 | 1.6% | M-RoPE errors, aggressive re-drafting |

## Finding: Speculative decoding is broken on Qwen3.5

**Both backends produce incorrect output.** The throughput numbers above are
misleading because the generated text is garbage or degraded.

### llama.cpp failure

M-RoPE (Multi-dimensional Rotary Position Embedding) position tracking breaks
when speculative tokens are rejected. The error `for M-RoPE, it is required
that the position satisfies: X < Y` repeats hundreds of times.
`speculative-simple` produced "Imagine" repeated 150 times. The throughput
numbers reflect decode speed of corrupted state, not valid inference.

### MLX failure

The failure is more subtle. Qwen3.5's hybrid architecture creates a cache with
24 Mamba/SSM layers (`ArraysCache`, trimmable=False) and 8 attention layers
(`KVCache`, trimmable=True). The speculative decode algorithm calls
`trim_prompt_cache()` to roll back rejected draft tokens, but
`can_trim_prompt_cache()` returns `False` because Mamba layers aren't
trimmable. **The cache rewind is silently a no-op.** Rejected tokens pollute
the SSM recurrent state, causing output degradation (loops, repetition) rather
than obvious crashes.

### Root cause summary

| | llama.cpp | MLX |
|---|---|---|
| Root cause | M-RoPE position tracking fails on rollback | Mamba state can't be trimmed |
| Symptom | Explicit crashes (`X < Y` errors) | Silent no-op cache trim |
| Output quality | Obvious garbage | Subtle degradation (loops) |
| Numbers valid? | No | No |

### Hybrid snapshot-restore fix (MLX)

We implemented a custom `speculative_generate_hybrid()` function that snapshots
all cache states (KV offsets + Mamba arrays) before each draft+verify cycle.
After acceptance, it restores from the snapshot and replays only the accepted
tokens through both models as a batch. This produces **correct, coherent output
identical to vanilla generation**.

However, performance is still a net slowdown:
- The 0.8B draft model is only ~1.1x faster than the 4B main model (both are
  bandwidth-bound on M1 due to Qwen's massive 248K vocabulary).
- The replay pass adds overhead per iteration.
- Speculative decoding needs a 3–5x draft:main speed ratio to break even.

The implementation is correct but the hardware/model pair doesn't yield speedup.
Code is in `experiments/speculative_decoding/`.

## Vanilla 4B Baselines (for reference)

**llama-bench**, Qwen3.5-4B Q4_K_M, M1 Metal GPU:

| Test | tok/s |
|------|-------|
| Prefill 128 | 180.8 |
| Decode 50 | 14.1 |

**MLX**, Qwen3.5-4B-4bit, M1 Metal GPU:

| Test | tok/s |
|------|-------|
| Prefill 128 | 92.6 |
| Prefill 512 | 102.7 |
| Prefill 1024 | 105.3 |
| Decode KV=45 | 19.6 |
| Decode KV=243 | 18.9 |
| Decode KV=485 | 16.0 |

Note: llama.cpp prefill is ~1.7x faster on this model. Decode favors MLX by
1.4x (same pattern as Qwen2.5-1.5B — MLX's smaller 4-bit weights require
less bandwidth).

## Lessons

1. **Speculative decoding is architecture-dependent.** Always verify output
   quality, not just throughput numbers.
2. **Hybrid Mamba+Attention models** (Qwen3.5, Jamba, etc.) are not compatible
   with current speculative decoding implementations in either MLX or llama.cpp
   as of March 2026.
3. A snapshot-restore approach can fix correctness for Mamba caches, but the
   replay overhead and insufficient draft:main speed ratio negate the benefit
   on M1 hardware with this model pair.
4. **Next step:** Retry with a pure-transformer model pair (e.g. Qwen2.5
   family) where standard KV cache trimming works correctly.
