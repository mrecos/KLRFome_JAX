"""Tests for controlled synthetic bag scenarios and invariance transforms."""

import numpy as np
import pytest

from klrfome.data.synthetic import (
    SyntheticScenarioConfig,
    duplicate_all_cells,
    duplicate_selected_cells,
    generate_synthetic_bags,
    permute_bag_cells,
)
from klrfome.models.distribution import DistributionClassifier
from klrfome.models.spec import ModelSpec
from klrfome.utils.reproducibility import dataset_fingerprint


def test_synthetic_generator_is_deterministic_and_seed_sensitive():
    configuration = SyntheticScenarioConfig("mean_shift", n_bags_per_class=4, bag_size=7, seed=9)
    first = generate_synthetic_bags(configuration)
    second = generate_synthetic_bags(configuration)
    changed = generate_synthetic_bags(
        SyntheticScenarioConfig("mean_shift", n_bags_per_class=4, bag_size=7, seed=10)
    )
    assert dataset_fingerprint(first) == dataset_fingerprint(second)
    assert dataset_fingerprint(first) != dataset_fingerprint(changed)


def test_multimodal_scenario_matches_low_order_moments_approximately():
    dataset = generate_synthetic_bags(
        SyntheticScenarioConfig(
            "multimodal",
            n_bags_per_class=30,
            n_features=1,
            bag_size=100,
            effect_size=0.9,
            seed=3,
        )
    )
    background = np.concatenate(
        [np.asarray(bag.samples) for bag in dataset.collections if bag.label == 0]
    )[:, 0]
    presence = np.concatenate(
        [np.asarray(bag.samples) for bag in dataset.collections if bag.label == 1]
    )[:, 0]
    assert abs(background.mean() - presence.mean()) < 0.08
    assert abs(background.var() - presence.var()) < 0.12
    assert abs(np.mean(background**4) - np.mean(presence**4)) > 0.25


def test_correlation_scenario_changes_dependence_not_marginal_scale():
    dataset = generate_synthetic_bags(
        SyntheticScenarioConfig(
            "correlation_shift",
            n_bags_per_class=25,
            n_features=2,
            bag_size=80,
            effect_size=0.8,
            seed=7,
        )
    )
    by_label = {
        label: np.concatenate(
            [np.asarray(bag.samples) for bag in dataset.collections if bag.label == label]
        )
        for label in (0, 1)
    }
    assert abs(np.corrcoef(by_label[0].T)[0, 1]) < 0.1
    assert np.corrcoef(by_label[1].T)[0, 1] > 0.7
    np.testing.assert_allclose(by_label[0].std(axis=0), by_label[1].std(axis=0), atol=0.08)


def test_unequal_bags_respect_limits_and_spatial_coordinates():
    dataset = generate_synthetic_bags(
        SyntheticScenarioConfig(
            "variance_shift",
            n_bags_per_class=10,
            unequal_bag_sizes=True,
            min_bag_size=3,
            max_bag_size=15,
            spatial_range=2.0,
            seed=2,
        )
    )
    sizes = [bag.n_samples for bag in dataset.collections]
    assert min(sizes) >= 3 and max(sizes) <= 15
    assert len(set(sizes)) > 1
    assert sizes[:10] == sizes[10:]
    assert all(bag.coordinates is not None for bag in dataset.collections)


def test_spatial_dependence_reduces_effective_information():
    independent = generate_synthetic_bags(
        SyntheticScenarioConfig(
            "null",
            n_bags_per_class=30,
            n_features=1,
            bag_size=49,
            spatial_range=0.0,
            seed=22,
        )
    )
    dependent = generate_synthetic_bags(
        SyntheticScenarioConfig(
            "null",
            n_bags_per_class=30,
            n_features=1,
            bag_size=49,
            spatial_range=5.0,
            seed=22,
        )
    )
    independent_means = np.asarray(
        [np.asarray(bag.samples).mean() for bag in independent.collections]
    )
    dependent_means = np.asarray([np.asarray(bag.samples).mean() for bag in dependent.collections])
    assert dependent_means.var() > independent_means.var() * 5


def test_moment_matched_xor_hides_signal_from_mean_and_scale_summaries():
    dataset = generate_synthetic_bags(
        SyntheticScenarioConfig(
            "moment_matched_xor",
            n_bags_per_class=30,
            n_features=2,
            bag_size=300,
            effect_size=0.9,
            seed=3,
        )
    )
    summaries = {}
    for label in (0, 1):
        bags = [np.asarray(bag.samples) for bag in dataset.collections if bag.label == label]
        summaries[label] = np.asarray(
            [np.concatenate([bag.mean(axis=0), bag.std(axis=0)]) for bag in bags]
        ).mean(axis=0)
    np.testing.assert_allclose(summaries[0], summaries[1], atol=0.08)


def test_transformations_distinguish_invariance_from_reweighting():
    dataset = generate_synthetic_bags(
        SyntheticScenarioConfig("mean_shift", n_bags_per_class=4, bag_size=9, seed=4)
    )
    permuted = permute_bag_cells(dataset, seed=5)
    duplicated = duplicate_all_cells(dataset, repeats=3)
    selective = duplicate_selected_cells(dataset, fraction=0.3, seed=6)
    for original, candidate in zip(dataset.collections, duplicated.collections):
        np.testing.assert_allclose(
            np.asarray(original.samples).mean(axis=0),
            np.asarray(candidate.samples).mean(axis=0),
            atol=1e-7,
        )
    for original, candidate in zip(dataset.collections, permuted.collections):
        np.testing.assert_allclose(
            np.sort(np.asarray(original.samples), axis=0),
            np.sort(np.asarray(candidate.samples), axis=0),
        )
    assert any(
        not np.allclose(
            np.asarray(original.samples).mean(axis=0), np.asarray(candidate.samples).mean(axis=0)
        )
        for original, candidate in zip(dataset.collections, selective.collections)
    )


@pytest.mark.parametrize(
    "spec",
    [ModelSpec.m0(), ModelSpec.m1(12), ModelSpec.m2(12), ModelSpec.m3(8, 8)],
)
def test_m0_m3_predictions_are_permutation_and_uniform_duplication_invariant(spec):
    dataset = generate_synthetic_bags(
        SyntheticScenarioConfig("mean_shift", n_bags_per_class=5, n_features=2, bag_size=8, seed=11)
    )
    model = DistributionClassifier(spec, seed=11, round_exact_kernel=False).fit(dataset)
    reference = np.asarray(model.predict_bags(dataset))
    permuted = np.asarray(model.predict_bags(permute_bag_cells(dataset, seed=12)))
    duplicated = np.asarray(model.predict_bags(duplicate_all_cells(dataset, repeats=2)))
    np.testing.assert_allclose(permuted, reference, atol=2e-5)
    np.testing.assert_allclose(duplicated, reference, atol=2e-5)
