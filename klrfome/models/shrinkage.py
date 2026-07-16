"""Effective-sample-size utilities for finite-bag embedding shrinkage."""

import numpy as np
from numpy.typing import NDArray


def spatial_effective_sample_size(
    coordinates: NDArray[np.floating], correlation_range: float
) -> float:
    """Return the independent-sample equivalent for an unweighted spatial mean.

    For the exponential correlation model ``R[i, j] = exp(-d[i, j] / range)``,
    the variance of an equally weighted mean is proportional to
    ``1.T @ R @ 1 / n**2``. The returned value is therefore
    ``n**2 / (1.T @ R @ 1)``, clipped to ``[1, n]`` for the supported
    nonnegative correlation model.

    Duplicate coordinates are counted once because repeated cells do not add
    spatial information. A zero range represents independent unique cells.
    """
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("coordinates must have shape (n, 2)")
    if len(values) == 0:
        raise ValueError("coordinates must be nonempty")
    if not np.isfinite(values).all():
        raise ValueError("coordinates must be finite")
    if not np.isfinite(correlation_range) or correlation_range < 0:
        raise ValueError("correlation_range must be finite and nonnegative")

    unique = np.unique(values, axis=0)
    count = len(unique)
    if count == 1 or correlation_range == 0:
        return float(count)

    distances = np.linalg.norm(unique[:, None, :] - unique[None, :, :], axis=2)
    correlation = np.exp(-distances / correlation_range)
    denominator = float(np.sum(correlation))
    effective_size = count**2 / denominator
    return float(np.clip(effective_size, 1.0, float(count)))
