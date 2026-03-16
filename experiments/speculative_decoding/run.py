#!/usr/bin/env python3
"""
Standalone runner for speculative decoding experiments.

Usage:
    python -m experiments.speculative_decoding.run \
        --main mlx-community/Qwen3.5-4B-4bit \
        --draft mlx-community/Qwen3.5-0.8B-MLX-4bit \
        --prompt "Explain general relativity in 3 sentences." \
        --max-tokens 100 --num-draft 4

Runs speculative decoding (with hybrid snapshot-restore for Mamba models)
alongside vanilla generation and compares output quality + throughput.
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Speculative decoding experiment (MLX)",
    )
    parser.add_argument("--main", required=True, help="Main (verifier) model ID")
    parser.add_argument("--draft", required=True, help="Draft (proposer) model ID")
    parser.add_argument("--prompt", default="Explain general relativity in 3 sentences.")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--num-draft", type=int, default=4)
    parser.add_argument("--skip-vanilla", action="store_true",
                        help="Skip vanilla comparison run")
    args = parser.parse_args()

    try:
        import mlx.core as mx
    except ImportError:
        print("MLX not available. Install with: pip install mlx-lm")
        sys.exit(1)

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from experiments.speculative_decoding.hybrid_generate import (
        has_mamba_cache,
        speculative_generate_hybrid,
    )

    print(f"Main model : {args.main}")
    print(f"Draft model: {args.draft}")
    print(f"Prompt     : {args.prompt!r}")
    print(f"Max tokens : {args.max_tokens}")
    print(f"Draft/step : {args.num_draft}")
    print()

    print("Loading main model...")
    model, tok = load(args.main)
    print("Loading draft model...")
    draft, _ = load(args.draft)

    test_cache = make_prompt_cache(model)
    hybrid = has_mamba_cache(test_cache)
    del test_cache
    print(f"Architecture: {'hybrid Mamba+Attention (snapshot-restore)' if hybrid else 'pure transformer (standard trim)'}")
    print()

    tokens = tok.encode(args.prompt)
    prompt_ids = mx.array(tokens, dtype=mx.uint32)

    # ---- speculative generation ----
    print("=" * 60)
    print("SPECULATIVE DECODING")
    print("=" * 60)

    gen = speculative_generate_hybrid(
        prompt_ids, model, draft, mx,
        num_draft_tokens=args.num_draft,
        max_tokens=args.max_tokens,
    )

    first_tok, first_draft = next(gen)
    output_tokens = [first_tok]
    n_accepted = 1 if first_draft else 0

    t0 = time.perf_counter()
    for tok_id, from_draft in gen:
        output_tokens.append(tok_id)
        if from_draft:
            n_accepted += 1
    elapsed = time.perf_counter() - t0

    n_total = len(output_tokens)
    decode_tokens = n_total - 1
    tps = decode_tokens / elapsed if elapsed > 0 else 0
    acc = n_accepted / max(1, n_total)

    text = tok.decode(output_tokens)
    print(f"\n{n_total} tokens in {elapsed:.2f}s  ({tps:.1f} tok/s)")
    print(f"Acceptance: {n_accepted}/{n_total} = {acc:.0%}")
    print(f"\n{text}\n")

    # ---- vanilla comparison ----
    if not args.skip_vanilla:
        print("=" * 60)
        print("VANILLA DECODE (for comparison)")
        print("=" * 60)
        from mlx_lm import generate
        vanilla = generate(model, tok, prompt=args.prompt,
                           max_tokens=args.max_tokens, verbose=True)
        print(f"\n{vanilla}\n")

        print("=" * 60)
        print("MATCH?" if text.strip() == vanilla.strip() else "OUTPUT DIFFERS (may be sampling-dependent)")
        print("=" * 60)


if __name__ == "__main__":
    main()
