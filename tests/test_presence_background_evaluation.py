"""Tests for availability-based presence-background evaluation."""

import numpy as np
import pytest

from klrfome.utils.evaluation import (
    availability_capture_metrics,
    availability_percentile_ranks,
    continuous_boyce_from_availability,
)


def test_availability_percentiles_put_fold_scores_on_a_common_rank_scale():
    availability = np.asarray([10.0, 20.0, 30.0, 40.0])
    targets = np.asarray([5.0, 20.0, 35.0, 50.0])
    np.testing.assert_allclose(
        availability_percentile_ranks(targets, availability),
        [0.0, 0.375, 0.75, 1.0],
    )


def test_capture_metrics_use_mapped_area_not_sampled_class_prevalence():
    availability = np.linspace(0.0, 1.0, 1001)
    sites = np.asarray([0.91, 0.96, 0.99, 1.0])
    rows = availability_capture_metrics(sites, availability, (0.05, 0.10, 0.20))
    assert [row["capture"] for row in rows] == [0.75, 1.0, 1.0]
    assert rows[0]["lift"] == pytest.approx(rows[0]["capture"] / rows[0]["achieved_area_fraction"])
    assert rows[1]["gain"] == pytest.approx(
        1.0 - rows[1]["achieved_area_fraction"] / rows[1]["capture"]
    )
    assert rows[0]["threshold"] > rows[1]["threshold"] > rows[2]["threshold"]


def test_capture_metrics_report_tie_expansion_and_use_achieved_area_for_lift():
    availability = np.asarray([0.0] * 90 + [1.0] * 10)
    sites = np.asarray([1.0, 1.0, 0.0, 0.0])
    row = availability_capture_metrics(sites, availability, (0.05,))[0]
    assert row["achieved_area_fraction"] == pytest.approx(0.10)
    assert row["capture"] == 0.5
    assert row["lift"] == pytest.approx(5.0)


def test_continuous_boyce_is_positive_for_high_rank_sites_and_honest_when_flat():
    availability = np.linspace(0.0, 1.0, 2001)
    concentrated_sites = np.linspace(0.75, 1.0, 80)
    diagnostic = continuous_boyce_from_availability(concentrated_sites, availability)
    assert diagnostic["boyce"] is not None
    assert diagnostic["boyce"] > 0.75
    assert min(diagnostic["site_percentiles"]) >= 0.749

    flat = continuous_boyce_from_availability(
        np.ones(20),
        np.ones(100),
    )
    assert flat["boyce"] is None


@pytest.mark.parametrize("bad_fraction", [0.0, -0.1, 1.1])
def test_availability_metrics_validate_area_fractions(bad_fraction):
    with pytest.raises(ValueError, match="area_fractions"):
        availability_capture_metrics(
            np.asarray([0.5]),
            np.asarray([0.1, 0.9]),
            (bad_fraction,),
        )
