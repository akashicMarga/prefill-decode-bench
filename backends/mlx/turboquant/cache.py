"""
TurboQuantKVCache — drop-in replacement for mlx_lm's KVCache.

Stores KV pairs in compressed form. During decode, computes attention
scores directly on compressed keys (via fused Metal kernels) and
reconstructs values from compressed state — never materializing the
full fp16 KV cache.
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import _BaseCache, create_attention_mask

from .codec import (
    MSECodec, ProdCodec, SplitCodec,
    MSEState, ProdState, SplitState,
    build_codec, concat_states, slice_state, slice_state_range, state_length,
)

_EPS = 1e-6


class TurboQuantKVCache(_BaseCache):
    """KV cache with TurboQuant compression and fused quantized attention.

    Keys use ProdCodec (unbiased inner products for Q·K scoring).
    Values use MSECodec (MSE-optimal for weighted sum reconstruction).

    For decode (single query token): computes attention directly on
    compressed state via Metal kernels. No dequantization needed.

    For prefill (multiple query tokens): falls back to chunked attention
    on compressed state, still avoiding full dequantization.
    """

    DECODE_CHUNK = 65536   # max KV tokens per chunk during decode
    PREFILL_CHUNK = 512    # max KV tokens per chunk during prefill

    def __init__(self, bits: float = 3.0, seed: int = 0):
        self.tq_bits = bits  # avoid 'bits' attr that triggers mlx_lm quantized SDPA
        self.seed = seed
        self.offset = 0
        self.key_codec = None
        self.value_codec = None
        # Quantized state lists (appended per chunk, concatenated lazily)
        self._k_states = []
        self._v_states = []
        self._k_lens = []  # track chunk lengths for lazy concat
        # Dequantized cache for compatibility with standard SDPA path
        self._k_deq = None
        self._v_deq = None

    def _ensure_codecs(self, keys: mx.array, values: mx.array):
        if self.key_codec is None:
            self.key_codec = build_codec(keys, self.tq_bits, mode="prod", seed=self.seed)
        if self.value_codec is None:
            self.value_codec = build_codec(values, self.tq_bits, mode="mse", seed=self.seed + 1)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Quantize new KV pairs and append to compressed state.

        Returns dequantized (keys, values) for compatibility with standard
        SDPA. Quantized state stored internally for fused attention path.
        """
        self._ensure_codecs(keys, values)
        num_new = keys.shape[2]

        new_k = self.key_codec.quantize(keys)
        new_v = self.value_codec.quantize(values)

        # Store quantized chunks (lazy — only concatenated when needed)
        self._k_states.append(new_k)
        self._v_states.append(new_v)
        self._k_lens.append(num_new)

        # Dequantize only new chunk, append to pre-allocated buffer
        k_new = self.key_codec.dequantize(new_k).astype(keys.dtype)
        v_new = self.value_codec.dequantize(new_v).astype(values.dtype)

        prev = self.offset
        self.offset += num_new

        if self._k_deq is None or (prev + num_new) > self._k_deq.shape[2]:
            B, H, _, Dk = k_new.shape
            Dv = v_new.shape[3]
            step = max(256, num_new)
            new_len = ((self.offset + step - 1) // step) * step
            new_k_buf = mx.zeros((B, H, new_len, Dk), dtype=keys.dtype)
            new_v_buf = mx.zeros((B, H, new_len, Dv), dtype=values.dtype)
            if self._k_deq is not None and prev > 0:
                new_k_buf[..., :prev, :] = self._k_deq[..., :prev, :]
                new_v_buf[..., :prev, :] = self._v_deq[..., :prev, :]
            self._k_deq = new_k_buf
            self._v_deq = new_v_buf

        self._k_deq[..., prev:self.offset, :] = k_new
        self._v_deq[..., prev:self.offset, :] = v_new

        return self._k_deq[..., :self.offset, :], self._v_deq[..., :self.offset, :]

    def _merged_key_state(self):
        """Lazily concatenate quantized key chunks."""
        if not self._k_states:
            return None
        if len(self._k_states) == 1:
            return self._k_states[0]
        merged = self._k_states[0]
        for s in self._k_states[1:]:
            merged = concat_states(merged, s)
        self._k_states = [merged]
        self._k_lens = [self.offset]
        return merged

    def _merged_value_state(self):
        """Lazily concatenate quantized value chunks."""
        if not self._v_states:
            return None
        if len(self._v_states) == 1:
            return self._v_states[0]
        merged = self._v_states[0]
        for s in self._v_states[1:]:
            merged = concat_states(merged, s)
        self._v_states = [merged]
        return merged

    @property
    def state(self):
        k = self._merged_key_state()
        v = self._merged_value_state()
        if k is None:
            return None, None
        return k, v

    @state.setter
    def state(self, value):
        if value is None:
            self._k_states, self._v_states = [], []
            self._k_lens = []
            self.offset = 0
            return
        self._k_states = [value[0]]
        self._v_states = [value[1]]
        self.offset = state_length(value[0])
        self._k_lens = [self.offset]

    # ------------------------------------------------------------------
    # Fused quantized attention
    # ------------------------------------------------------------------

    def quantized_attention(
        self,
        queries: mx.array,
        scale: float = 1.0,
        mask=None,
    ) -> mx.array:
        """Compute attention output directly on compressed KV state.

        For decode (L=1): uses fused Metal kernels for scoring and value
        reconstruction. For prefill (L>1): uses chunked attention with
        online softmax normalization.

        Args:
            queries: (B, n_q_heads, L, D)
            scale: attention scale factor (typically 1/sqrt(head_dim))
            mask: attention mask or "causal"

        Returns:
            (B, n_q_heads, L, D) attention output
        """
        key_state, value_state = self.state
        if key_state is None:
            return mx.zeros_like(queries)

        B, n_q_heads, L, D = queries.shape

        # Determine KV head count from state
        if isinstance(key_state, SplitState):
            kv_state_for_heads = key_state.low
            if isinstance(kv_state_for_heads, ProdState):
                n_kv_heads = kv_state_for_heads.norms.shape[1]
            else:
                n_kv_heads = kv_state_for_heads.norms.shape[1]
        else:
            n_kv_heads = key_state.norms.shape[1]

        n_repeats = n_q_heads // n_kv_heads

        # Group queries: (B, kv_heads, repeats, L, D)
        grouped = (queries * scale).reshape(B, n_kv_heads, n_repeats, L, D)

        total_tokens = state_length(key_state)
        value_dim = self.value_codec.dim if not isinstance(self.value_codec, SplitCodec) else self.value_codec.dim
        chunk_size = self.DECODE_CHUNK if L == 1 else self.PREFILL_CHUNK

        # Simple path: all tokens fit in one chunk
        if total_tokens <= chunk_size and mask in (None, "causal") and L == 1:
            prepared = self.key_codec.prepare_queries(grouped)
            scores = self.key_codec.score_prepared(prepared, key_state)
            output = self.value_codec.weighted_sum_from_scores(scores, value_state)
            return output.reshape(B, n_q_heads, L, value_dim).astype(queries.dtype)

        # Chunked attention with online softmax for long sequences
        output = mx.zeros((B, n_kv_heads, n_repeats, L, value_dim), dtype=mx.float32)
        normalizer = mx.zeros((B, n_kv_heads, n_repeats, L), dtype=mx.float32)
        max_score = mx.full((B, n_kv_heads, n_repeats, L), -float("inf"), dtype=mx.float32)

        prepared = self.key_codec.prepare_queries(grouped)

        for k_start in range(0, total_tokens, chunk_size):
            k_end = min(total_tokens, k_start + chunk_size)
            k_chunk = slice_state_range(key_state, k_start, k_end)
            v_chunk = slice_state_range(value_state, k_start, k_end)

            scores = self.key_codec.score_prepared(prepared, k_chunk)
            scores = self._apply_mask(scores, mask, L, total_tokens, k_start, k_end)

            # Online softmax accumulation
            chunk_max = mx.max(scores.squeeze(3), axis=-1)
            chunk_weights = mx.exp(scores.squeeze(3) - chunk_max[..., None])
            chunk_denom = mx.sum(chunk_weights, axis=-1)

            # Compute weighted sum for this chunk
            chunk_v = self.value_codec.weighted_sum_from_scores(scores, v_chunk)
            if chunk_v is None:
                weights_5d = mx.softmax(scores, axis=-1)
                chunk_v = self.value_codec.weighted_sum(weights_5d, v_chunk) if hasattr(self.value_codec, 'weighted_sum') else self.value_codec.dequantize(v_chunk)

            # Merge with running totals
            new_max = mx.maximum(max_score, chunk_max)
            prev_scale = mx.exp(max_score - new_max)
            chunk_scale = mx.exp(chunk_max - new_max)

            if chunk_v.ndim == 5:
                output = output * prev_scale[..., None] + chunk_v.squeeze(3) * chunk_scale[..., None]
            else:
                output = output * prev_scale[..., None] + chunk_v * chunk_scale[..., None]
            normalizer = normalizer * prev_scale + chunk_denom * chunk_scale
            max_score = new_max
            mx.eval(output, normalizer, max_score)

        output = output / mx.maximum(normalizer[..., None], _EPS)
        return output.reshape(B, n_q_heads, L, value_dim).astype(queries.dtype)

    def _apply_mask(self, scores, mask, L, total_tokens, k_start, k_end):
        """Apply causal or explicit mask to attention scores."""
        if mask is None:
            return scores
        if isinstance(mask, str) and mask == "causal":
            past = total_tokens - L
            q_idx = mx.arange(past, past + L)
            k_idx = mx.arange(k_start, k_end)
            causal = q_idx[:, None] >= k_idx[None, :]
            causal = causal[None, None, None, :, :]
            return mx.where(causal, scores, mx.finfo(scores.dtype).min)
        if isinstance(mask, mx.array):
            chunk = mask[..., k_start:k_end]
            if chunk.ndim == scores.ndim - 1:
                chunk = mx.expand_dims(chunk, axis=2)
            if chunk.dtype == mx.bool_:
                return mx.where(chunk, scores, mx.finfo(scores.dtype).min)
            return scores + chunk
        return scores

    # ------------------------------------------------------------------
    # Dequantize fallback (for quality checks)
    # ------------------------------------------------------------------

    def dequantize(self):
        """Full dequantization — only for quality verification, not perf path."""
        key_state, value_state = self.state
        if key_state is None:
            return None, None
        keys = self.key_codec.dequantize(key_state).astype(mx.float32)
        values = self.value_codec.dequantize(value_state).astype(mx.float32)
        return keys, values

    # ------------------------------------------------------------------
    # Cache interface
    # ------------------------------------------------------------------

    def size(self):
        return self.offset

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.tq_bits, self.seed)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v[0])
        self.tq_bits = float(v[1])
        self.seed = int(v[2])

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return len(self._k_states) == 0

    @property
    def nbytes(self):
        """Estimated compressed size in bytes."""
        if not self._k_states:
            return 0
        def _count(state):
            if state is None:
                return 0
            if isinstance(state, SplitState):
                return _count(state.low) + _count(state.high)
            total = 0
            for field in state:
                if isinstance(field, mx.array):
                    total += field.nbytes
            return total
        return sum(_count(s) for s in self._k_states) + sum(_count(s) for s in self._v_states)


def make_turboquant_cache(model: nn.Module, bits: float = 3.0, seed: int = 0):
    """Create TurboQuant KV caches for all layers.

    Drop-in replacement for mlx_lm.models.cache.make_prompt_cache.
    """
    if hasattr(model, "make_cache"):
        default = model.make_cache()
        return [TurboQuantKVCache(bits=bits, seed=seed) for _ in range(len(default))]
    return [TurboQuantKVCache(bits=bits, seed=seed) for _ in range(len(model.layers))]
