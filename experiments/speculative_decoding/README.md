# Speculative Decoding on Hybrid Mamba+Attention Models

## Goal

Test whether speculative decoding can accelerate inference on Qwen3.5 models,
which use a hybrid architecture mixing Mamba (SSM) layers with standard
self-attention layers in a 3:1 ratio.

**Model pair:**
- Main (verifier): `Qwen/Qwen3.5-4B` — 4-bit quantized, 2.37 GB
- Draft (proposer): `Qwen/Qwen3.5-0.8B` — 4-bit quantized, 0.44 GB

**Hardware:** Apple M1, 16 GB unified memory, 68.25 GB/s bandwidth

## Background

Speculative decoding speeds up autoregressive generation by using a small
"draft" model to propose N candidate tokens, then verifying them all at once
with the larger "main" model. Accepted tokens skip individual decode steps.
The expected speedup depends on:

1. **Draft:main speed ratio** — the draft model needs to be 3–5x faster
2. **Acceptance rate** — how often the main model agrees with the draft
3. **Rollback cost** — undoing rejected tokens from the model's cached state

For pure transformers, rollback is cheap: just truncate the KV cache. This
experiment investigates what happens when the model has state that *can't*
be truncated.

## What We Found

### Both existing implementations are broken on Qwen3.5

**llama.cpp** crashes with M-RoPE position tracking errors:

```
for M-RoPE, it is required that the position satisfies: X < Y
```

Output: "Imagine" repeated 150 times. The position embedding system can't
handle the rollback of rejected tokens in Qwen3.5's multi-dimensional
rotary encoding.

**MLX** fails silently. Qwen3.5 creates a cache with:
- 24 Mamba/SSM layers → `ArraysCache` (`trimmable=False`)
- 8 attention layers → `KVCache` (`trimmable=True`)

The standard `trim_prompt_cache()` checks `can_trim_prompt_cache()`, which
returns `False` because the Mamba layers aren't trimmable. **The rollback
is silently a no-op.** Rejected draft tokens permanently pollute the Mamba
recurrent state, causing progressive output degradation — loops, repetition,
and eventually incoherent text.

### Root cause: Mamba state is cumulative

Attention KV caches are append-only logs — you can truncate them to any
prior length. Mamba/SSM layers maintain a fixed-size recurrent state that
gets updated with each token. There is no "undo" operation. Processing
token `t` irreversibly modifies the state, so if `t` is later rejected
by the verifier, the state is corrupted.

```
Attention cache:  [k1, k2, k3, k4_draft]  →  trim to [k1, k2, k3]  ✓
Mamba state:      f(f(f(f(s0, t1), t2), t3), t4_draft)  →  ???      ✗
```

## Our Fix: Snapshot-Restore-Replay

We implemented `speculative_generate_hybrid()` in `hybrid_generate.py` with
a checkpoint-based approach:

```
for each speculative step:
    1. SNAPSHOT  — deep-copy all cache states
                   (KV offsets for attention, full array copies for Mamba)

    2. DRAFT    — run draft model for N tokens (serial, each depends on prev)

    3. VERIFY   — run main model on [last_accepted + draft_tokens] as one batch

    4. COUNT    — find how many draft tokens match main model's greedy output

    5. RESTORE  — reset ALL caches to the snapshot (both models)

    6. REPLAY   — feed [last_accepted + accepted_draft_tokens] as a batch
                   through both models, correctly advancing KV and Mamba state

    7. YIELD    — emit accepted tokens + correction token
```

The key insight is step 5+6: instead of trying to "undo" rejected tokens
(impossible for Mamba), we restore to a known-good state and replay only the
tokens we want to keep. This is equivalent to autoregressive generation of
the accepted sequence but done as a single batch forward pass.

**Cost:** One extra forward pass of (k+1) tokens per iteration for the replay.
At k=3-4 this is effectively a small prefill operation.

### Correctness: Verified

Output from hybrid speculative decoding is **identical** to vanilla
autoregressive generation (greedy/argmax). We verified this by:

1. Running both methods on the same prompt
2. Comparing token-by-token output — exact match
3. Inspecting Mamba state tensors after batch vs autoregressive processing —
   argmax agreement despite small floating-point differences

### Performance: No speedup on this hardware/model pair

| Mode | tok/s | Accept % |
|------|-------|----------|
| Vanilla decode (4B) | 10.6 | — |
| Hybrid speculative (4B + 0.8B) | 4.5 | ~74% |

**Why:** The 0.8B draft model is only ~1.1x faster than the 4B main model.
Both are bandwidth-bound on M1, and Qwen's massive 248K vocabulary means the
`lm_head` projection (vocab_size × hidden_dim) dominates inference time
regardless of model depth. Speculative decoding needs a 3–5x draft:main ratio
to overcome its overhead.

Additionally, the replay step (absent in standard speculative decoding) adds
a forward pass per iteration that wouldn't exist with a pure transformer.

## How to Run

```bash
# From the repo root
python -m experiments.speculative_decoding.run \
    --main mlx-community/Qwen3.5-4B-4bit \
    --draft mlx-community/Qwen3.5-0.8B-MLX-4bit \
    --prompt "Explain general relativity in 3 sentences." \
    --max-tokens 100 \
    --num-draft 4
```

Add `--skip-vanilla` to skip the comparison vanilla generation run.

The runner automatically detects hybrid architectures (via `has_mamba_cache()`)
and uses the snapshot-restore path.

## Files

| File | Purpose |
|------|---------|
| `hybrid_generate.py` | `speculative_generate_hybrid()` — the snapshot-restore generator, plus `has_mamba_cache()` detection helper |
| `run.py` | Standalone experiment runner — loads models, runs speculative + vanilla, compares output and throughput |

## What Would Make This Work

Speculative decoding *would* show speedup with:

1. **Pure-transformer models** (e.g. Qwen2.5, Llama) — no Mamba state, no
   replay overhead, standard KV trim works. This is the obvious next test.

2. **Higher draft:main speed ratio** — a draft model that's 3–5x faster.
   The 0.8B/4B Qwen3.5 pair is only ~1.1x because both are bandwidth-bound
   with the same vocabulary size.

3. **Smaller vocabulary models** — with a typical 32K vocab instead of 248K,
   the lm_head wouldn't dominate, and the 5x parameter difference between
   0.8B and 4B would translate to a real speed difference.

4. **Hardware with heterogeneous compute** — e.g. draft model on CPU while
   main model runs on GPU, so draft generation is "free" (parallel).

## Related

- [Experiment 006](../../docs/research_log/experiments/006_speculative_qwen3.5.md) — full benchmark numbers
- [Leviathan et al., 2023](https://arxiv.org/abs/2211.17192) — original speculative decoding paper
- [MLX speculative_generate_step](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py) — upstream MLX implementation (works for pure transformers)
