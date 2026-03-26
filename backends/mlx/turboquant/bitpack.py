"""
Bit-packing utilities for low-bit quantized indices.

Packs b-bit integers into uint32 words for memory-efficient storage.
Provides Metal kernel fast path on Apple Silicon, Python fallback otherwise.
"""

from functools import lru_cache
from typing import Optional

import mlx.core as mx


def _metal_available() -> bool:
    return hasattr(mx, "metal") and mx.metal.is_available()


def packed_width(length: int, bits: int) -> int:
    """Number of uint32 words needed to store `length` values at `bits` each."""
    if length == 0 or bits == 0:
        return 0
    return (length * bits + 31) // 32


# ---------------------------------------------------------------------------
# Metal kernels
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _pack_kernel():
    if not _metal_available():
        return None
    source = r"""
        auto word = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= values_shape[0] || word >= PackedWidth) return;

        auto values_ptr = values + row * Length;
        uint packed_word = 0u;
        int start = max(0, (int(word) * 32 - (Bits - 1)) / Bits);
        int end = min(Length, ((int(word) + 1) * 32 + (Bits - 1)) / Bits);

        for (int idx = start; idx < end; ++idx) {
            int bit_offset = idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = values_ptr[idx] & ((1u << Bits) - 1u);
            if (word_idx == word) packed_word |= value << offset;
            if (word_idx + 1 == word) {
                int spill = offset + Bits - 32;
                if (spill > 0) packed_word |= value >> (Bits - spill);
            }
        }
        out[row * PackedWidth + word] = packed_word;
    """
    return mx.fast.metal_kernel(
        name="tq_pack_lowbit",
        input_names=["values"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _unpack_kernel():
    if not _metal_available():
        return None
    source = r"""
        auto idx = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= packed_shape[0] || idx >= Length) return;

        auto packed_ptr = packed + row * PackedWidth;
        int bit_offset = idx * Bits;
        int word_idx = bit_offset / 32;
        int offset = bit_offset % 32;
        uint value = packed_ptr[word_idx] >> offset;
        int spill = offset + Bits - 32;
        if (spill > 0) value |= packed_ptr[word_idx + 1] << (Bits - spill);
        out[row * Length + idx] = value & ((1u << Bits) - 1u);
    """
    return mx.fast.metal_kernel(
        name="tq_unpack_lowbit",
        input_names=["packed"],
        output_names=["out"],
        source=source,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pack_lowbit(values: mx.array, bits: int) -> mx.array:
    """Pack b-bit integer values into uint32 words.

    Args:
        values: (..., length) uint or int array with values in [0, 2^bits).
        bits: Bits per value.

    Returns:
        (..., packed_width) uint32 array.
    """
    if bits == 0:
        return mx.zeros((*values.shape[:-1], 0), dtype=mx.uint32)

    values = values.astype(mx.uint32)
    length = values.shape[-1]
    pw = packed_width(length, bits)
    flat = values.reshape((-1, length))

    kernel = _pack_kernel()
    if kernel is not None:
        packed = kernel(
            inputs=[flat],
            template=[("Bits", bits), ("Length", length), ("PackedWidth", pw)],
            grid=(pw, flat.shape[0], 1),
            threadgroup=(min(32, pw), 1, 1),
            output_shapes=[(flat.shape[0], pw)],
            output_dtypes=[mx.uint32],
        )[0]
        return packed.reshape((*values.shape[:-1], pw))

    # Python fallback
    packed = mx.zeros((flat.shape[0], pw), dtype=mx.uint32)
    for idx in range(length):
        bit_offset = idx * bits
        word_idx = bit_offset // 32
        offset = bit_offset % 32
        packed[:, word_idx] = packed[:, word_idx] | (flat[:, idx] << offset)
        spill = offset + bits - 32
        if spill > 0:
            packed[:, word_idx + 1] = packed[:, word_idx + 1] | (flat[:, idx] >> (bits - spill))
    return packed.reshape((*values.shape[:-1], pw))


def unpack_lowbit(packed: mx.array, bits: int, length: int) -> mx.array:
    """Unpack uint32 words back to b-bit integer values.

    Args:
        packed: (..., packed_width) uint32 array.
        bits: Bits per value.
        length: Original number of values per row.

    Returns:
        (..., length) uint32 array.
    """
    if bits == 0:
        return mx.zeros((*packed.shape[:-1], 0), dtype=mx.uint32)

    packed = packed.astype(mx.uint32)
    flat = packed.reshape((-1, packed.shape[-1]))

    kernel = _unpack_kernel()
    if kernel is not None:
        unpacked = kernel(
            inputs=[flat],
            template=[("Bits", bits), ("Length", length), ("PackedWidth", flat.shape[-1])],
            grid=(length, flat.shape[0], 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(flat.shape[0], length)],
            output_dtypes=[mx.uint32],
        )[0]
        return unpacked.reshape((*packed.shape[:-1], length))

    # Python fallback
    mask = (1 << bits) - 1
    unpacked = mx.zeros((flat.shape[0], length), dtype=mx.uint32)
    for idx in range(length):
        bit_offset = idx * bits
        word_idx = bit_offset // 32
        offset = bit_offset % 32
        value = flat[:, word_idx] >> offset
        spill = offset + bits - 32
        if spill > 0:
            value = value | (flat[:, word_idx + 1] << (bits - spill))
        unpacked[:, idx] = value & mask
    return unpacked.reshape((*packed.shape[:-1], length))
