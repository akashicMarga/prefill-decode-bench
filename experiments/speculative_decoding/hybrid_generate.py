"""
experiments/speculative_decoding/hybrid_generate.py
====================================================
Speculative decoding generator for hybrid Mamba+Attention models.

Standard speculative decoding trims KV caches to roll back rejected draft
tokens.  Mamba/SSM layers (ArraysCache) are NOT trimmable — recurrent state
is cumulative, so ``trim_prompt_cache`` is silently a no-op for these layers.

This module provides a snapshot-restore approach: all cache states (KV offsets
+ Mamba arrays) are checkpointed before each draft+verify cycle.  After
acceptance, the snapshot is restored and only the accepted tokens are replayed
through both models as a batch, keeping KV and Mamba states consistent.

Cost: one extra batch forward pass of (k+1) tokens per iteration for the
replay step.  At k=3-4 on GPU this is effectively a small prefill.
"""


def has_mamba_cache(cache_list) -> bool:
    """Check if any cache layer is a non-trimmable ArraysCache (Mamba/SSM)."""
    from mlx_lm.models.cache import ArraysCache
    return any(isinstance(c, ArraysCache) for c in cache_list)


def speculative_generate_hybrid(
    prompt, model, draft_model, mx, *,
    num_draft_tokens=4, max_tokens=256,
):
    """Speculative decoding for hybrid Mamba+Attention models.

    Snapshots all cache states (KV offsets + Mamba arrays) before each
    draft+verify cycle.  After acceptance, restores from the snapshot and
    replays only the accepted tokens through both models as a batch.

    Args:
        prompt: 1-D ``mx.array`` of prompt token IDs.
        model: Main (verifier) model.
        draft_model: Draft (proposer) model.
        mx: The ``mlx.core`` module (passed to avoid import at module level).
        num_draft_tokens: Candidate tokens per speculative step.
        max_tokens: Maximum tokens to generate.

    Yields:
        ``(token_id: int, from_draft: bool)`` per accepted token.
    """
    from mlx_lm.models.cache import make_prompt_cache, ArraysCache, KVCache

    model_cache = make_prompt_cache(model)
    draft_cache = make_prompt_cache(draft_model)

    def _snapshot(cache_list):
        snaps = []
        for c in cache_list:
            if isinstance(c, ArraysCache) and not c.empty():
                snaps.append([mx.array(a) for a in c.cache])
            elif isinstance(c, KVCache):
                snaps.append(c.offset)
            else:
                snaps.append(None)
        return snaps

    def _restore(cache_list, snaps):
        for c, snap in zip(cache_list, snaps):
            if snap is None:
                continue
            if isinstance(c, ArraysCache) and isinstance(snap, list):
                c.cache = snap
            elif isinstance(c, KVCache) and isinstance(snap, int):
                c.offset = snap

    # ---- prefill both models ----
    prompt_arr = prompt.astype(mx.uint32)
    logits = model(prompt_arr[None], cache=model_cache)
    draft_model(prompt_arr[None], cache=draft_cache)
    mx.eval(logits)
    last_tok = mx.argmax(logits[0, -1, :]).item()

    ntoks = 0
    while ntoks < max_tokens:
        m_snap = _snapshot(model_cache)
        d_snap = _snapshot(draft_cache)

        n_draft = min(max_tokens - ntoks, num_draft_tokens)

        # ---- draft N tokens (serial, each depends on previous) ----
        draft_tokens = []
        d_in = mx.array([last_tok], mx.uint32)
        for _ in range(n_draft):
            d_logits = draft_model(d_in[None], cache=draft_cache)
            d_tok = mx.argmax(d_logits[0, -1, :]).item()
            draft_tokens.append(d_tok)
            d_in = mx.array([d_tok], mx.uint32)

        # ---- verify with main model (single batched forward) ----
        verify_input = mx.array([last_tok] + draft_tokens, mx.uint32)
        v_logits = model(verify_input[None], cache=model_cache)
        main_tokens = mx.argmax(v_logits[0], axis=-1)
        mx.eval(main_tokens)
        main_tokens = main_tokens.tolist()

        # ---- count accepted ----
        n_accept = 0
        for i in range(n_draft):
            if main_tokens[i] == draft_tokens[i]:
                n_accept += 1
            else:
                break
        correction = main_tokens[n_accept]

        # ---- restore caches to pre-draft state ----
        _restore(model_cache, m_snap)
        _restore(draft_cache, d_snap)

        # ---- replay accepted tokens through both models ----
        replay = mx.array([last_tok] + draft_tokens[:n_accept], mx.uint32)
        model(replay[None], cache=model_cache)
        draft_model(replay[None], cache=draft_cache)
        to_eval = []
        for c in model_cache:
            if isinstance(c, ArraysCache) and not c.empty():
                to_eval.extend(a for a in c.cache if a is not None)
        for c in draft_cache:
            if isinstance(c, ArraysCache) and not c.empty():
                to_eval.extend(a for a in c.cache if a is not None)
        if to_eval:
            mx.eval(*to_eval)

        # ---- yield results ----
        for i in range(n_accept):
            yield draft_tokens[i], True
            ntoks += 1
            if ntoks >= max_tokens:
                return
        yield correction, False
        ntoks += 1
        last_tok = correction
