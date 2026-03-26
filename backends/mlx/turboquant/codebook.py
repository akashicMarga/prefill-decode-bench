"""
Codebook computation and random matrix generation.

Lloyd-Max optimal scalar quantizers for the Beta distribution that arises
from random rotation of unit-norm vectors (Lemma 1 of the paper).
"""

import math
from functools import lru_cache

import mlx.core as mx
import numpy as np

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Random matrices (cached, deterministic via seed)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def rotation_matrix(dim: int, seed: int = 42) -> mx.array:
    """Random orthogonal matrix via QR decomposition of Gaussian matrix."""
    if dim <= 0:
        return mx.zeros((0, 0), dtype=mx.float32)
    if dim == 1:
        return mx.ones((1, 1), dtype=mx.float32)
    rng = np.random.default_rng(seed + dim * 7919)
    G = rng.standard_normal((dim, dim), dtype=np.float32)
    Q, R = np.linalg.qr(G)
    Q *= np.sign(np.diag(R))
    return mx.array(Q)


@lru_cache(maxsize=None)
def projection_matrix(dim: int, seed: int = 137) -> mx.array:
    """Random Gaussian matrix for QJL."""
    if dim <= 0:
        return mx.zeros((0, 0), dtype=mx.float32)
    rng = np.random.default_rng(seed + dim * 2971 + 17)
    S = rng.standard_normal((dim, dim), dtype=np.float32)
    return mx.array(S)


# ---------------------------------------------------------------------------
# Beta PDF for coordinate distribution on unit hypersphere
# ---------------------------------------------------------------------------

def _beta_pdf(grid: np.ndarray, dim: int) -> np.ndarray:
    """PDF of a single coordinate of a uniform point on S^{d-1}.

    fX(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
    """
    if dim <= 1:
        return np.ones_like(grid)
    coeff = math.gamma(dim / 2) / (math.sqrt(math.pi) * math.gamma((dim - 1) / 2))
    pdf = coeff * np.power(np.clip(1.0 - grid ** 2, 0.0, None), (dim - 3) / 2)
    total = pdf.sum()
    if total == 0:
        return np.full_like(grid, 1.0 / len(grid))
    return pdf / total


# ---------------------------------------------------------------------------
# Lloyd-Max codebook via iterative k-means on 1-D distribution
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def codebook(dim: int, bits: int) -> mx.array:
    """Compute optimal Lloyd-Max centroids for the Beta distribution.

    For dimension `dim` and `bits` bits, returns 2^bits centroids in [-1, 1].
    """
    if bits <= 0:
        return mx.zeros((0,), dtype=mx.float32)

    levels = 1 << bits

    if dim <= 1:
        return mx.array(np.linspace(-1.0, 1.0, levels, dtype=np.float32))

    # Dense grid for numerical integration
    grid = np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, 32768, dtype=np.float32)
    weights = _beta_pdf(grid, dim)

    # Initialize centroids via quantile placement
    cdf = np.cumsum(weights)
    quantiles = (np.arange(levels, dtype=np.float32) + 0.5) / levels
    centroids = np.interp(quantiles, cdf, grid).astype(np.float32)

    # Lloyd-Max iteration
    for _ in range(100):
        boundaries = np.empty(levels + 1, dtype=np.float32)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

        new_centroids = centroids.copy()
        for i in range(levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            mask = (grid >= lo) & (grid < hi) if i < levels - 1 else (grid >= lo) & (grid <= hi)
            bucket_w = weights[mask]
            if bucket_w.size > 0 and bucket_w.sum() > 0:
                new_centroids[i] = np.sum(bucket_w * grid[mask]) / bucket_w.sum()

        if np.max(np.abs(new_centroids - centroids)) < 1e-6:
            centroids = new_centroids
            break
        centroids = new_centroids

    return mx.array(centroids.astype(np.float32))


@lru_cache(maxsize=None)
def boundaries(dim: int, bits: int) -> mx.array:
    """Voronoi boundaries = midpoints between consecutive centroids."""
    c = codebook(dim, bits)
    return (c[:-1] + c[1:]) / 2.0
