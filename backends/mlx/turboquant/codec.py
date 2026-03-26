"""
TurboQuant codecs: quantize/dequantize/score/weighted_sum.

Three codec types:
  - MSECodec: rotation + Lloyd-Max scalar quantization (for values)
  - ProdCodec: MSE + 1-bit QJL residual (for keys — unbiased inner products)
  - SplitCodec: outlier channel splitting for non-integer bits (2.5, 3.5)
"""

import math
from typing import NamedTuple, Optional

import mlx.core as mx
import numpy as np

from .codebook import codebook, rotation_matrix, projection_matrix
from .bitpack import pack_lowbit, unpack_lowbit, _metal_available
from .kernels import (
    metal_prod_score,
    metal_mse_weighted_sum_from_scores,
)

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Quantized state types
# ---------------------------------------------------------------------------

class MSEState(NamedTuple):
    """Quantized state for MSE codec."""
    norms: mx.array           # (B, H, T) — L2 norms
    indices: mx.array         # (B, H, T, packed_width) — bit-packed centroid indices


class ProdState(NamedTuple):
    """Quantized state for inner-product codec."""
    norms: mx.array           # (B, H, T)
    mse_indices: mx.array     # (B, H, T, packed_width)
    residual_norms: mx.array  # (B, H, T)
    qjl_signs: mx.array       # (B, H, T, sign_packed_width) — 1-bit packed


# ---------------------------------------------------------------------------
# State utilities
# ---------------------------------------------------------------------------

def concat_states(lhs, rhs):
    """Concatenate two quantized states along the token dimension."""
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    if isinstance(lhs, MSEState):
        return MSEState(
            mx.concatenate([lhs.norms, rhs.norms], axis=2),
            mx.concatenate([lhs.indices, rhs.indices], axis=2),
        )
    if isinstance(lhs, ProdState):
        return ProdState(
            mx.concatenate([lhs.norms, rhs.norms], axis=2),
            mx.concatenate([lhs.mse_indices, rhs.mse_indices], axis=2),
            mx.concatenate([lhs.residual_norms, rhs.residual_norms], axis=2),
            mx.concatenate([lhs.qjl_signs, rhs.qjl_signs], axis=2),
        )
    if isinstance(lhs, SplitState):
        return SplitState(
            concat_states(lhs.low, rhs.low),
            concat_states(lhs.high, rhs.high),
        )
    raise TypeError(f"Unknown state type: {type(lhs)}")


def slice_state(state, end: int):
    """Slice state to first `end` tokens."""
    if state is None:
        return None
    if isinstance(state, MSEState):
        return MSEState(state.norms[..., :end], state.indices[..., :end, :])
    if isinstance(state, ProdState):
        return ProdState(
            state.norms[..., :end],
            state.mse_indices[..., :end, :],
            state.residual_norms[..., :end],
            state.qjl_signs[..., :end, :],
        )
    if isinstance(state, SplitState):
        return SplitState(slice_state(state.low, end), slice_state(state.high, end))
    raise TypeError(f"Unknown state type: {type(state)}")


def slice_state_range(state, start: int, end: int):
    """Slice state to tokens [start, end)."""
    if state is None:
        return None
    if isinstance(state, MSEState):
        return MSEState(state.norms[..., start:end], state.indices[..., start:end, :])
    if isinstance(state, ProdState):
        return ProdState(
            state.norms[..., start:end],
            state.mse_indices[..., start:end, :],
            state.residual_norms[..., start:end],
            state.qjl_signs[..., start:end, :],
        )
    if isinstance(state, SplitState):
        return SplitState(
            slice_state_range(state.low, start, end),
            slice_state_range(state.high, start, end),
        )
    raise TypeError(f"Unknown state type: {type(state)}")


def state_length(state) -> int:
    """Number of tokens stored in the state."""
    if state is None:
        return 0
    if isinstance(state, MSEState):
        return state.norms.shape[2]
    if isinstance(state, ProdState):
        return state.norms.shape[2]
    if isinstance(state, SplitState):
        return state_length(state.low)
    raise TypeError(f"Unknown state type: {type(state)}")


class SplitState(NamedTuple):
    """Quantized state for split (non-integer bit) codec."""
    low: object   # MSEState or ProdState for regular channels
    high: object  # MSEState or ProdState for outlier channels


# ---------------------------------------------------------------------------
# MSE Codec (Algorithm 1 from paper)
# ---------------------------------------------------------------------------

class MSECodec:
    """Rotation + Lloyd-Max scalar quantizer. Optimal for MSE distortion.

    Used for value cache (V) where MSE matters for weighted sum quality.
    """

    def __init__(self, dim: int, bits: int, seed: int = 42):
        self.dim = dim
        self.bits = bits
        self.rotation = rotation_matrix(dim, seed)
        self.rotation_t = self.rotation.T if dim > 0 else self.rotation
        self.cb = codebook(dim, bits)

    def _quantize_unit(self, unit: mx.array) -> tuple[mx.array, mx.array]:
        """Quantize unit-norm vectors. Returns (packed_indices, mse_estimate)."""
        if self.bits == 0:
            zeros_idx = mx.zeros((*unit.shape[:-1], 0), dtype=mx.uint32)
            zeros_est = mx.zeros(unit.shape, dtype=mx.float32)
            return zeros_idx, zeros_est

        rotated = mx.matmul(unit, self.rotation_t)
        # Find nearest centroid for each coordinate
        distances = mx.abs(rotated[..., None] - self.cb)
        indices = mx.argmin(distances, axis=-1).astype(mx.uint32)
        packed = pack_lowbit(indices, self.bits)
        # Reconstruct estimate
        estimated_rot = mx.take(self.cb, indices, axis=0)
        estimated = mx.matmul(estimated_rot, self.rotation)
        return packed, estimated

    def _dequantize_unit(self, packed_indices: mx.array) -> mx.array:
        """Dequantize packed indices to unit-norm vectors."""
        if self.bits == 0:
            return mx.zeros((*packed_indices.shape[:-1], self.dim), dtype=mx.float32)
        indices = unpack_lowbit(packed_indices, self.bits, self.dim).astype(mx.int32)
        rotated = mx.take(self.cb, indices, axis=0)
        return mx.matmul(rotated, self.rotation)

    def quantize(self, vectors: mx.array) -> MSEState:
        vectors_f32 = vectors.astype(mx.float32)
        norms = mx.linalg.norm(vectors_f32, axis=-1)
        safe_norms = mx.maximum(norms[..., None], _EPS)
        unit = vectors_f32 / safe_norms
        packed, _ = self._quantize_unit(unit)
        return MSEState(norms.astype(vectors.dtype), packed)

    def dequantize(self, state: MSEState) -> mx.array:
        unit = self._dequantize_unit(state.indices)
        return state.norms[..., None].astype(unit.dtype) * unit

    def prepare_queries(self, queries: mx.array) -> mx.array:
        """Pre-rotate queries for scoring."""
        return mx.matmul(queries, self.rotation_t)

    def score_prepared(self, prep_q: mx.array, state: MSEState) -> mx.array:
        """Compute Q·K scores from prepared queries and quantized keys."""
        indices = unpack_lowbit(state.indices, self.bits, self.dim).astype(mx.int32)
        rotated = mx.take(self.cb, indices, axis=0)
        dots = mx.einsum("bhmld,bhtd->bhmlt", prep_q, rotated)
        return dots * state.norms.astype(mx.float32)[:, :, None, None, :]

    def weighted_sum(self, weights: mx.array, state: MSEState) -> mx.array:
        """Compute weighted sum of dequantized values."""
        indices = unpack_lowbit(state.indices, self.bits, self.dim).astype(mx.int32)
        rotated = mx.take(self.cb, indices, axis=0)
        weighted_rot = mx.einsum(
            "bhmlt,bht,bhtd->bhmld",
            weights,
            state.norms.astype(mx.float32),
            rotated,
        )
        return mx.matmul(weighted_rot, self.rotation)

    def weighted_sum_from_scores(self, scores: mx.array, state: MSEState) -> mx.array:
        """Compute softmax(scores) @ V from scores and quantized V state.

        Tries Metal fast path first, falls back to Python.
        """
        fast = metal_mse_weighted_sum_from_scores(
            scores, state, self.bits, self.cb, self.rotation,
        )
        if fast is not None:
            return fast
        return self.weighted_sum(mx.softmax(scores, axis=-1), state)


# ---------------------------------------------------------------------------
# Prod Codec (Algorithm 2 from paper)
# ---------------------------------------------------------------------------

class ProdCodec:
    """MSE quantizer + 1-bit QJL on residual. Unbiased for inner products.

    Used for key cache (K) where unbiased Q·K scores are critical.
    Total bit-width b = (b-1) bits MSE + 1 bit QJL.
    """

    def __init__(self, dim: int, bits: int, seed: int = 42):
        self.dim = dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 0)
        self.mse = MSECodec(dim, self.mse_bits, seed)
        self.proj = projection_matrix(dim, seed + 1)
        self.proj_t = self.proj.T if dim > 0 else self.proj
        # Pre-concatenated transform: [rotation^T | projection^T]
        self.query_transform_t = (
            mx.concatenate([self.mse.rotation_t, self.proj_t], axis=-1)
            if dim > 0 else mx.zeros((0, 0), dtype=mx.float32)
        )
        self.scale = math.sqrt(math.pi / 2) / dim if dim > 0 else 0.0
        self.scale_array = mx.array([self.scale], dtype=mx.float32)

    def quantize(self, vectors: mx.array) -> ProdState:
        vectors_f32 = vectors.astype(mx.float32)
        norms = mx.linalg.norm(vectors_f32, axis=-1)
        safe_norms = mx.maximum(norms[..., None], _EPS)
        unit = vectors_f32 / safe_norms

        mse_packed, mse_estimate = self.mse._quantize_unit(unit)

        residual = unit - mse_estimate
        residual_norms = mx.linalg.norm(residual, axis=-1)

        projected = mx.matmul(residual, self.proj_t)
        signs = mx.where(projected >= 0, 1, 0).astype(mx.uint32)
        signs_packed = pack_lowbit(signs, 1)

        return ProdState(
            norms.astype(vectors.dtype),
            mse_packed,
            residual_norms.astype(vectors.dtype),
            signs_packed,
        )

    def dequantize(self, state: ProdState) -> mx.array:
        mse_unit = self.mse._dequantize_unit(state.mse_indices)
        sign_bits = unpack_lowbit(state.qjl_signs, 1, self.dim).astype(mx.float32)
        signs = sign_bits * 2.0 - 1.0
        qjl_unit = (
            self.scale
            * state.residual_norms[..., None].astype(mx.float32)
            * mx.matmul(signs, self.proj)
        )
        return state.norms[..., None].astype(mx.float32) * (mse_unit + qjl_unit)

    def prepare_queries(self, queries: mx.array) -> tuple[mx.array, mx.array]:
        """Transform queries: returns (rotated_q, projected_q)."""
        transformed = mx.matmul(queries, self.query_transform_t)
        return transformed[..., :self.dim], transformed[..., self.dim:]

    def score_prepared(
        self,
        prepared: tuple[mx.array, mx.array],
        state: ProdState,
    ) -> mx.array:
        """Compute Q·K scores from prepared queries and quantized K state.

        Tries fused Metal kernel first, falls back to Python.
        """
        mse_q, proj_q = prepared

        # Try fused Metal path
        if proj_q.shape[-2] == 1:  # decode (single query token)
            fast = metal_prod_score(
                mse_q.reshape(mse_q.shape[0], mse_q.shape[1], mse_q.shape[2], mse_q.shape[-1]),
                proj_q.reshape(proj_q.shape[0], proj_q.shape[1], proj_q.shape[2], proj_q.shape[-1]),
                state, self.mse_bits, self.mse.cb, self.scale_array,
            )
            if fast is not None:
                return fast

        # Python fallback: MSE score
        if self.mse_bits > 0:
            mse_score = self.mse.score_prepared(
                mse_q, MSEState(state.norms, state.mse_indices)
            )
        else:
            shape = (*proj_q.shape[:4], state.norms.shape[2])
            mse_score = mx.zeros(shape, dtype=mx.float32)

        # QJL score
        sign_bits = unpack_lowbit(state.qjl_signs, 1, self.dim).astype(mx.float32)
        signs = sign_bits * 2.0 - 1.0
        qjl_score = self.scale * state.residual_norms.astype(mx.float32)[
            :, :, None, None, :
        ] * mx.einsum("bhmld,bhtd->bhmlt", proj_q, signs)

        norms = state.norms.astype(mx.float32)[:, :, None, None, :]
        return mse_score + norms * qjl_score

    def score(self, queries: mx.array, state: ProdState) -> mx.array:
        return self.score_prepared(self.prepare_queries(queries), state)


# ---------------------------------------------------------------------------
# Split Codec (non-integer bits via outlier channel splitting)
# ---------------------------------------------------------------------------

def _select_outlier_indices(tensor: mx.array, avg_bits: float) -> tuple[np.ndarray, np.ndarray]:
    """Select outlier channels by average magnitude."""
    lower_bits = math.floor(avg_bits)
    upper_bits = math.ceil(avg_bits)
    dim = tensor.shape[-1]
    high_count = int(round((avg_bits - lower_bits) * dim / (upper_bits - lower_bits)))
    high_count = max(1, min(dim - 1, high_count))

    scores = mx.mean(mx.abs(tensor.astype(mx.float32)), axis=(0, 1, 2))
    order = np.argsort(np.asarray(scores))
    high_idx = np.sort(order[-high_count:].astype(np.int32))
    low_mask = np.ones(dim, dtype=bool)
    low_mask[high_idx] = False
    low_idx = np.nonzero(low_mask)[0].astype(np.int32)
    return low_idx, high_idx


class SplitCodec:
    """Outlier channel splitting for non-integer bits.

    Splits channels into low-bit and high-bit groups based on magnitude.
    E.g., 2.5-bit: some channels at 2 bits, outliers at 3 bits.
    """

    def __init__(self, tensor: mx.array, bits: float, mode: str, seed: int = 42):
        self.bits = bits
        self.dim = tensor.shape[-1]
        self.lower_bits = math.floor(bits)
        self.upper_bits = math.ceil(bits)

        low_idx, high_idx = _select_outlier_indices(tensor, bits)
        self.low_idx = mx.array(low_idx, dtype=mx.int32)
        self.high_idx = mx.array(high_idx, dtype=mx.int32)
        concat_order = np.concatenate([low_idx, high_idx])
        self.restore_order = mx.array(np.argsort(concat_order), dtype=mx.int32)

        Codec = ProdCodec if mode == "prod" else MSECodec
        self.low_codec = Codec(len(low_idx), self.lower_bits, seed)
        self.high_codec = Codec(len(high_idx), self.upper_bits, seed + 97)

    def quantize(self, tensor: mx.array) -> SplitState:
        low_t = mx.take(tensor, self.low_idx, axis=-1)
        high_t = mx.take(tensor, self.high_idx, axis=-1)
        return SplitState(self.low_codec.quantize(low_t), self.high_codec.quantize(high_t))

    def dequantize(self, state: SplitState) -> mx.array:
        low_v = self.low_codec.dequantize(state.low)
        high_v = self.high_codec.dequantize(state.high)
        merged = mx.concatenate([low_v, high_v], axis=-1)
        return mx.take(merged, self.restore_order, axis=-1)

    def prepare_queries(self, queries: mx.array):
        low_q = mx.take(queries, self.low_idx, axis=-1)
        high_q = mx.take(queries, self.high_idx, axis=-1)
        return (self.low_codec.prepare_queries(low_q),
                self.high_codec.prepare_queries(high_q))

    def score_prepared(self, prepared, state: SplitState) -> mx.array:
        low_p, high_p = prepared
        return (self.low_codec.score_prepared(low_p, state.low)
                + self.high_codec.score_prepared(high_p, state.high))

    def score(self, queries: mx.array, state: SplitState) -> mx.array:
        return self.score_prepared(self.prepare_queries(queries), state)

    def weighted_sum_from_scores(self, scores: mx.array, state: SplitState) -> mx.array:
        low_v = self.low_codec.weighted_sum_from_scores(scores, state.low)
        high_v = self.high_codec.weighted_sum_from_scores(scores, state.high)
        merged = mx.concatenate([low_v, high_v], axis=-1)
        return mx.take(merged, self.restore_order, axis=-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_codec(tensor: mx.array, bits: float, mode: str, seed: int = 42):
    """Build the appropriate codec for the given bit-width.

    Args:
        tensor: Sample tensor to determine dimension and outlier channels.
        bits: Target bits per coordinate (2, 2.5, 3, 3.5, 4).
        mode: "prod" for keys (unbiased inner products), "mse" for values.
        seed: RNG seed for rotation/projection matrices.
    """
    int_bits = round(bits)
    if math.isclose(bits, int_bits, abs_tol=1e-6):
        Codec = ProdCodec if mode == "prod" else MSECodec
        return Codec(tensor.shape[-1], int_bits, seed)
    return SplitCodec(tensor, bits, mode, seed)
