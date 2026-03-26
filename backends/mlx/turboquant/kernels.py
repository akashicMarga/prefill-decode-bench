"""
Metal kernels for fused TurboQuant attention scoring.

Key insight: compute Q·K attention scores DIRECTLY on compressed state
without dequantizing K. Similarly, compute softmax(scores)·V directly
on compressed V state.

Each kernel operates on bit-packed data using SIMD warp-level reductions.
"""

from functools import lru_cache
from typing import Optional

import mlx.core as mx

from .bitpack import _metal_available


# ---------------------------------------------------------------------------
# Fused prod score: MSE component + QJL component in one kernel
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _prod_score_kernel():
    """Fused kernel: computes Q_rot · codebook[packed] + scale * Q_proj · signs."""
    if not _metal_available():
        return None

    source = r"""
        auto lane = thread_position_in_grid.x;
        auto repeat_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = q_rot_shape[2];
        if (repeat_idx >= repeat_count) return;

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_rot_ptr = q_rot + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto q_proj_ptr = q_proj + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto mse_ptr = mse_packed + ((b * kv_heads + h) * token_count + t) * MsePackedWidth;
        auto sign_ptr = signs + ((b * kv_heads + h) * token_count + t) * SignPackedWidth;

        float mse_acc = 0.0f;
        float qjl_acc = 0.0f;
        for (int d = lane; d < Dim; d += 32) {
            // Unpack MSE index inline
            int bit_offset = d * MseBits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = mse_ptr[word_idx] >> offset;
            int spill = offset + MseBits - 32;
            if (spill > 0) value |= mse_ptr[word_idx + 1] << (MseBits - spill);
            value &= ((1u << MseBits) - 1u);
            mse_acc += static_cast<float>(q_rot_ptr[d]) * codebook[value];

            // Unpack QJL sign inline
            int sign_word = d / 32;
            int sign_offset = d % 32;
            uint bit = (sign_ptr[sign_word] >> sign_offset) & 1u;
            float sign = bit ? 1.0f : -1.0f;
            qjl_acc += static_cast<float>(q_proj_ptr[d]) * sign;
        }

        mse_acc = simd_sum(mse_acc);
        qjl_acc = simd_sum(qjl_acc);
        if (thread_index_in_simdgroup == 0) {
            auto idx = (b * kv_heads + h) * token_count + t;
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                static_cast<float>(norms[idx]) * (
                    mse_acc
                    + scale[0] * static_cast<float>(residual_norms[idx]) * qjl_acc
                );
        }
    """
    return mx.fast.metal_kernel(
        name="tq_prod_score",
        input_names=[
            "q_rot", "q_proj",
            "norms", "residual_norms",
            "mse_packed", "signs",
            "codebook", "scale",
        ],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _prod_score_repeat_kernel(repeat_count: int):
    """Unrolled version for GQA: avoids dynamic repeat loop."""
    if not _metal_available() or repeat_count <= 1:
        return None

    # Generate unrolled accumulators
    acc_decls = "\n".join(
        f"        float mse_acc_{r} = 0.0f; float qjl_acc_{r} = 0.0f;"
        for r in range(repeat_count)
    )
    inner_loop = "\n".join(
        f"            mse_acc_{r} += static_cast<float>(q_rot_base[{r} * Dim + d]) * code;\n"
        f"            qjl_acc_{r} += static_cast<float>(q_proj_base[{r} * Dim + d]) * sign;"
        for r in range(repeat_count)
    )
    reductions = "\n".join(
        f"        mse_acc_{r} = simd_sum(mse_acc_{r}); qjl_acc_{r} = simd_sum(qjl_acc_{r});"
        for r in range(repeat_count)
    )
    outputs = "\n".join(
        f"            out[((b * kv_heads + h) * repeat_count + {r}) * token_count + t] =\n"
        f"                norm * (mse_acc_{r} + scale[0] * residual_norm * qjl_acc_{r});"
        for r in range(repeat_count)
    )

    source = f"""
        auto lane = thread_position_in_grid.x;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = q_rot_shape[2];

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_rot_base = q_rot + ((b * kv_heads + h) * repeat_count) * Dim;
        auto q_proj_base = q_proj + ((b * kv_heads + h) * repeat_count) * Dim;
        auto mse_ptr = mse_packed + ((b * kv_heads + h) * token_count + t) * MsePackedWidth;
        auto sign_ptr = signs + ((b * kv_heads + h) * token_count + t) * SignPackedWidth;

        auto idx = (b * kv_heads + h) * token_count + t;
        float norm = static_cast<float>(norms[idx]);
        float residual_norm = static_cast<float>(residual_norms[idx]);

{acc_decls}

        for (int d = lane; d < Dim; d += 32) {{
            int bit_offset = d * MseBits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = mse_ptr[word_idx] >> offset;
            int spill = offset + MseBits - 32;
            if (spill > 0) value |= mse_ptr[word_idx + 1] << (MseBits - spill);
            value &= ((1u << MseBits) - 1u);
            float code = codebook[value];

            int sign_word = d / 32;
            int sign_offset = d % 32;
            uint bit = (sign_ptr[sign_word] >> sign_offset) & 1u;
            float sign = bit ? 1.0f : -1.0f;

{inner_loop}
        }}

{reductions}

        if (thread_index_in_simdgroup == 0) {{
{outputs}
        }}
    """
    return mx.fast.metal_kernel(
        name=f"tq_prod_score_repeat_{repeat_count}",
        input_names=[
            "q_rot", "q_proj",
            "norms", "residual_norms",
            "mse_packed", "signs",
            "codebook", "scale",
        ],
        output_names=["out"],
        source=source,
    )


# ---------------------------------------------------------------------------
# MSE weighted sum: softmax-weighted reconstruction from packed values
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _mse_weighted_rot_kernel(repeat_count: int):
    """Compute softmax(scores) @ V in rotated space, then rotate back."""
    if not _metal_available():
        return None

    acc_decls = "\n".join(f"        float acc_{r} = 0.0f;" for r in range(repeat_count))
    inner = "\n".join(
        f"            acc_{r} += static_cast<float>(scores_base[{r} * token_count + t]) * norm * code;"
        for r in range(repeat_count)
    )
    reductions = "\n".join(f"        acc_{r} = simd_sum(acc_{r});" for r in range(repeat_count))
    outputs = "\n".join(
        f"            out[((b * kv_heads + h) * repeat_count + {r}) * Dim + dim_idx] = acc_{r};"
        for r in range(repeat_count)
    )

    source = f"""
        auto lane = thread_position_in_grid.x;
        auto dim_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        if (dim_idx >= Dim) return;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = scores_shape[2];
        auto b = n / kv_heads;
        auto h = n % kv_heads;

        auto scores_base = scores + ((b * kv_heads + h) * repeat_count) * token_count;
        auto norms_ptr = norms + (b * kv_heads + h) * token_count;
        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;

        int bit_offset = dim_idx * Bits;
        int word_idx = bit_offset / 32;
        int offset = bit_offset % 32;

        // Find max for numerical stability
        float max_scores[{repeat_count}];
{chr(10).join(f"        max_scores[{r}] = -INFINITY;" for r in range(repeat_count))}
        for (int t = lane; t < token_count; t += 32) {{
{chr(10).join(f"            max_scores[{r}] = max(max_scores[{r}], static_cast<float>(scores_base[{r} * token_count + t]));" for r in range(repeat_count))}
        }}
{chr(10).join(f"        max_scores[{r}] = simd_max(max_scores[{r}]);" for r in range(repeat_count))}

{acc_decls}

        for (int t = lane; t < token_count; t += 32) {{
            auto token_ptr = packed_ptr + t * PackedWidth;
            uint value = token_ptr[word_idx] >> offset;
            int spill = offset + Bits - 32;
            if (spill > 0) value |= token_ptr[word_idx + 1] << (Bits - spill);
            value &= ((1u << Bits) - 1u);
            float code = codebook[value];
            float norm = static_cast<float>(norms_ptr[t]);
{chr(10).join(f"            float w_{r} = exp(static_cast<float>(scores_base[{r} * token_count + t]) - max_scores[{r}]);" for r in range(repeat_count))}
{chr(10).join(f"            acc_{r} += w_{r} * norm * code;" for r in range(repeat_count))}
        }}

{reductions}

        if (thread_index_in_simdgroup == 0) {{
{outputs}
        }}
    """
    return mx.fast.metal_kernel(
        name=f"tq_mse_weighted_rot_{repeat_count}",
        input_names=["scores", "norms", "packed", "codebook"],
        output_names=["out"],
        source=source,
    )


# ---------------------------------------------------------------------------
# Public dispatch functions
# ---------------------------------------------------------------------------

def metal_prod_score(
    q_rot: mx.array,
    q_proj: mx.array,
    state,  # ProdState
    mse_bits: int,
    cb: mx.array,
    scale: mx.array,
) -> Optional[mx.array]:
    """Try to compute prod scores via Metal kernel. Returns None if unavailable."""
    if (
        mse_bits <= 0
        or not _metal_available()
        or q_rot.ndim != 4
        or state.norms.shape[2] == 0
    ):
        return None

    B, H, R, D = q_rot.shape
    T = state.norms.shape[2]

    # Try unrolled repeat kernel for GQA
    if R > 1:
        kernel = _prod_score_repeat_kernel(R)
        if kernel is not None:
            scores = kernel(
                inputs=[
                    q_rot, q_proj,
                    state.norms, state.residual_norms,
                    state.mse_indices.astype(mx.uint32),
                    state.qjl_signs.astype(mx.uint32),
                    cb, scale,
                ],
                template=[
                    ("Dim", D), ("MseBits", mse_bits),
                    ("MsePackedWidth", state.mse_indices.shape[-1]),
                    ("SignPackedWidth", state.qjl_signs.shape[-1]),
                ],
                grid=(32, 1, B * H * T),
                threadgroup=(32, 1, 1),
                output_shapes=[(B, H, R, T)],
                output_dtypes=[mx.float32],
            )[0]
            return mx.expand_dims(scores, axis=3)

    # Single-repeat fallback
    kernel = _prod_score_kernel()
    if kernel is None:
        return None

    scores = kernel(
        inputs=[
            q_rot, q_proj,
            state.norms, state.residual_norms,
            state.mse_indices.astype(mx.uint32),
            state.qjl_signs.astype(mx.uint32),
            cb, scale,
        ],
        template=[
            ("Dim", D), ("MseBits", mse_bits),
            ("MsePackedWidth", state.mse_indices.shape[-1]),
            ("SignPackedWidth", state.qjl_signs.shape[-1]),
        ],
        grid=(32, R, B * H * T),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, T)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.expand_dims(scores, axis=3)


def metal_mse_weighted_sum_from_scores(
    scores: mx.array,
    state,  # MSEState
    bits: int,
    cb: mx.array,
    rotation: mx.array,
) -> Optional[mx.array]:
    """Try to compute softmax(scores) @ V via Metal kernel. Returns None if unavailable."""
    if (
        bits <= 0
        or not _metal_available()
        or scores.ndim != 5
        or scores.shape[-2] != 1
        or state.norms.shape[2] == 0
    ):
        return None

    scores_2d = scores.reshape(scores.shape[0], scores.shape[1], scores.shape[2], scores.shape[-1])
    B, H, R, T = scores_2d.shape
    D = rotation.shape[0]

    if R <= 1:
        return None

    kernel = _mse_weighted_rot_kernel(R)
    if kernel is None:
        return None

    weighted_rot = kernel(
        inputs=[
            scores_2d,
            state.norms,
            state.indices.astype(mx.uint32),
            cb,
        ],
        template=[
            ("Dim", D), ("Bits", bits),
            ("PackedWidth", state.indices.shape[-1]),
        ],
        grid=(32, D, B * H),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, D)],
        output_dtypes=[mx.float32],
    )[0]
    output = mx.matmul(weighted_rot, rotation)
    return mx.expand_dims(output, axis=3)
