"""Tests for spatial effective sample size used by embedding shrinkage."""

import numpy as np
import pytest

from klrfome.models.shrinkage import spatial_effective_sample_size


def test_spatial_effective_size_matches_equal_weight_mean_variance():
    coordinates = np.column_stack([np.arange(6, dtype=float), np.zeros(6)])
    correlation_range = 2.5
    distances = np.abs(coordinates[:, None, 0] - coordinates[None, :, 0])
    correlation = np.exp(-distances / correlation_range)
    expected = len(coordinates) ** 2 / correlation.sum()
    assert spatial_effective_sample_size(coordinates, correlation_range) == pytest.approx(expected)


def test_spatial_effective_size_decreases_with_range_and_ignores_duplicates():
    coordinates = np.column_stack([np.arange(12, dtype=float), np.zeros(12)])
    independent = spatial_effective_sample_size(coordinates, 0.0)
    short_range = spatial_effective_sample_size(coordinates, 1.0)
    long_range = spatial_effective_sample_size(coordinates, 5.0)
    duplicated = spatial_effective_sample_size(np.repeat(coordinates, 3, axis=0), 5.0)
    assert independent == 12.0
    assert 1.0 < long_range < short_range < independent
    assert duplicated == pytest.approx(long_range)


@pytest.mark.parametrize(
    ("coordinates", "correlation_range", "message"),
    [
        (np.ones((3, 1)), 1.0, "shape"),
        (np.ones((3, 2)), -1.0, "nonnegative"),
        (np.array([[0.0, np.nan]]), 1.0, "finite"),
    ],
)
def test_spatial_effective_size_validates_inputs(coordinates, correlation_range, message):
    with pytest.raises(ValueError, match=message):
        spatial_effective_sample_size(coordinates, correlation_range)
