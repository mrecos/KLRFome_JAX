"""Correctness tests for the M0--M4 model foundation."""

import jax.numpy as jnp
import numpy as np
import pytest

from klrfome.data.formats import Bag, BagDataset, SampleCollection, TrainingData
from klrfome.data.synthetic import duplicate_all_cells
from klrfome.models.distribution import DistributionClassifier
from klrfome.models.klr import KernelLogisticRegression
from klrfome.models.primal import PrimalLogisticRegression
from klrfome.models.spec import ModelSpec
from klrfome.utils.validation import cross_validate, make_fold_plan
from klrfome.utils.serialization import save_model
from klrfome.api import KLRfome


def _dataset(seed: int = 2) -> BagDataset:
    rng = np.random.default_rng(seed)
    bags = []
    for label, offset in ((0, -0.7), (1, 0.7)):
        for index in range(5):
            samples = rng.normal(offset, 0.8, size=(8 + index, 3))
            coordinates = np.column_stack(
                [np.arange(len(samples)) + 100 * index, np.full(len(samples), label)]
            )
            bags.append(
                Bag(
                    samples=jnp.asarray(samples),
                    label=label,
                    id=f"{label}-{index}",
                    coordinates=jnp.asarray(coordinates),
                    group_id=f"g-{label}-{index}",
                    stratum_id="test",
                )
            )
    return BagDataset(bags, ["a", "b", "c"], crs="EPSG:26918")


def test_compatibility_aliases_are_canonical_types():
    assert SampleCollection is Bag
    assert TrainingData is BagDataset


def test_bag_dataset_validation_rejects_contract_violations():
    good = Bag(jnp.ones((3, 2)), 1, "a", coordinates=jnp.ones((3, 2)))
    with pytest.raises(ValueError, match="unique"):
        BagDataset([good, Bag(jnp.zeros((3, 2)), 0, "a")], ["x", "y"])
    with pytest.raises(ValueError, match="x/y"):
        Bag(jnp.ones((3, 2)), 1, "bad", coordinates=jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        Bag(jnp.array([[1.0, jnp.nan]]), 1, "bad")


def test_extreme_logits_and_near_singular_gram_remain_finite():
    gram = jnp.ones((8, 8), dtype=jnp.float32)
    gram = gram.at[0, 0].set(1.000001)
    labels = jnp.asarray([0, 1] * 4, dtype=jnp.float32)
    result = KernelLogisticRegression(lambda_reg=1e-6, tol=1e-5, max_iter=200).fit(
        gram * 1e5, labels
    )
    probabilities = KernelLogisticRegression().predict_proba(gram * 1e5, result.alpha)
    assert np.isfinite(np.asarray(result.alpha)).all()
    assert np.isfinite(result.final_loss)
    assert np.isfinite(np.asarray(probabilities)).all()


def test_primal_matches_equivalent_small_dual_problem():
    rng = np.random.default_rng(4)
    features = jnp.asarray(rng.normal(size=(12, 5)), dtype=jnp.float32)
    labels = jnp.asarray([0, 1] * 6, dtype=jnp.float32)
    gram = features @ features.T
    dual = KernelLogisticRegression(lambda_reg=0.3, tol=1e-6).fit(gram, labels)
    primal = PrimalLogisticRegression(lambda_reg=0.3, tol=1e-6).fit(features, labels)
    dual_probabilities = KernelLogisticRegression().predict_proba(gram, dual.alpha)
    primal_probabilities = PrimalLogisticRegression.predict_proba(features, primal.coefficients)
    np.testing.assert_allclose(primal_probabilities, dual_probabilities, atol=2e-5)


def test_high_level_spec_rejects_wasserstein_one():
    with pytest.raises(ValueError, match="Wasserstein-2"):
        ModelSpec(
            "sliced_wasserstein",
            "rbf",
            "dual_klr",
            wasserstein_p=1,
        )


@pytest.mark.parametrize(
    "spec",
    [
        ModelSpec.m0(),
        ModelSpec.m1(32),
        ModelSpec.m2(32),
        ModelSpec.m3(12, 16),
        ModelSpec.m4(0.5, rff_features=32, n_projections=12, n_quantiles=16),
    ],
)
def test_all_supported_methods_fit_and_predict_finitely(spec):
    dataset = _dataset()
    original = np.asarray(dataset.collections[0].samples).copy()
    model = DistributionClassifier(spec, seed=3, round_exact_kernel=False).fit(dataset)
    probabilities = model.predict_bags(dataset)
    assert probabilities.shape == (dataset.n_locations,)
    assert np.isfinite(np.asarray(probabilities)).all()
    assert np.all((np.asarray(probabilities) >= 0) & (np.asarray(probabilities) <= 1))
    np.testing.assert_array_equal(np.asarray(dataset.collections[0].samples), original)
    if spec.method_id == "M1":
        assert model.gram_matrix_ is None
        assert model.diagnostics_["constructed_gram_matrix"] is False
    else:
        gram = np.asarray(model.gram_matrix_)
        np.testing.assert_allclose(gram, gram.T, atol=1e-5)
        assert np.linalg.eigvalsh(gram).min() >= -2e-4


def test_fold_plan_has_complete_repeats_and_no_group_leakage():
    dataset = _dataset()
    # Pair adjacent bags into spatial blocks to make leakage testable.
    for index, bag in enumerate(dataset.collections):
        bag.group_id = f"block-{index // 2}"
    plan = make_fold_plan(dataset, n_splits=2, n_repeats=3, seed=11)
    for repeat in range(3):
        tested = []
        for assignment in plan.assignments:
            if assignment.repeat != repeat:
                continue
            tested.extend(assignment.test_indices)
            train_groups = {dataset.collections[i].group_id for i in assignment.train_indices}
            test_groups = {dataset.collections[i].group_id for i in assignment.test_indices}
            assert not train_groups & test_groups
        assert sorted(tested) == list(range(dataset.n_locations))


def test_methods_share_exact_fold_plan_and_preprocessing_is_train_only():
    dataset = _dataset()
    plan = make_fold_plan(
        dataset, n_splits=2, seed=5, group_ids=[bag.id for bag in dataset.collections]
    )
    results = []
    for spec in (ModelSpec.m0(), ModelSpec.m1(24)):
        model = DistributionClassifier(spec, seed=9, round_exact_kernel=False)
        results.append(cross_validate(model, dataset, fold_plan=plan))
    assert [fold["test_ids"] for fold in results[0]["folds"]] == [
        fold["test_ids"] for fold in results[1]["folds"]
    ]

    first = plan.assignments[0]
    train = dataset.subset(first.train_indices)
    fitted = DistributionClassifier(ModelSpec.m1(16), seed=9).fit(train)
    train_cells = np.concatenate([np.asarray(bag.samples) for bag in train.collections])
    global_cells = np.concatenate([np.asarray(bag.samples) for bag in dataset.collections])
    np.testing.assert_allclose(fitted.preprocessor_.means, train_cells.mean(axis=0), atol=1e-6)
    if not np.allclose(train_cells.mean(axis=0), global_cells.mean(axis=0)):
        assert not np.allclose(fitted.preprocessor_.means, global_cells.mean(axis=0))


def test_one_class_auc_is_reported_undefined():
    from klrfome.utils.validation import compute_roc_auc

    assert np.isnan(compute_roc_auc([0.1, 0.2, 0.3], [1, 1, 1]))


def test_legacy_api_maps_to_m1_and_serializes_primal_coefficients(tmp_path):
    model = KLRfome(n_rff_features=16, seed=4).fit(_dataset())
    assert model._resolved_spec.method_id == "M1"
    path = tmp_path / "model.klrfome"
    save_model(model, str(path))
    from klrfome.utils.serialization import load_model

    restored = load_model(str(path))
    assert isinstance(restored, KLRfome)
    assert restored._core_model is not None
    assert restored._core_model.spec.method_id == "M1"
    assert np.isfinite(np.asarray(restored._core_model.fit_result_.coefficients)).all()


def test_rff_shrinkage_is_finite_bounded_and_duplicate_invariant():
    dataset = _dataset()
    model = DistributionClassifier(
        ModelSpec.m1(32, rff_scheme="orthogonal", embedding_estimator="shrinkage"),
        seed=12,
    ).fit(dataset)
    factors = np.asarray(model.training_shrinkage_factors_)
    assert factors.shape == (dataset.n_locations,)
    assert np.isfinite(factors).all()
    assert np.all((factors >= 0) & (factors <= 1))

    original = np.asarray(model.predict_bags(dataset))
    duplicated = np.asarray(model.predict_bags(duplicate_all_cells(dataset, repeats=3)))
    np.testing.assert_allclose(duplicated, original, atol=2e-6)

    without_coordinates = _dataset()
    for bag in without_coordinates.collections:
        bag.coordinates = None
    coordinate_free_model = DistributionClassifier(
        ModelSpec.m1(32, rff_scheme="orthogonal", embedding_estimator="shrinkage"),
        seed=12,
    ).fit(without_coordinates)
    original = np.asarray(coordinate_free_model.predict_bags(without_coordinates))
    duplicated = np.asarray(
        coordinate_free_model.predict_bags(duplicate_all_cells(without_coordinates, repeats=3))
    )
    np.testing.assert_allclose(duplicated, original, atol=2e-6)


def test_metadata_effective_size_is_explicit_and_validated():
    dataset = _dataset()
    spec = ModelSpec.m1(
        24,
        embedding_estimator="shrinkage",
        shrinkage_effective_size="metadata",
    )
    with pytest.raises(ValueError, match="effective_sample_size"):
        DistributionClassifier(spec).fit(dataset)

    for bag in dataset.collections:
        bag.metadata = {**(bag.metadata or {}), "effective_sample_size": 2.0}
    fitted = DistributionClassifier(spec).fit(dataset)
    assert np.isfinite(np.asarray(fitted.training_shrinkage_factors_)).all()


def test_spatial_effective_size_uses_coordinates_and_explicit_range():
    dataset = _dataset()
    missing_range = ModelSpec.m1(
        24,
        embedding_estimator="shrinkage",
        shrinkage_effective_size="spatial",
    )
    with pytest.raises(ValueError, match="spatial correlation range"):
        DistributionClassifier(missing_range).fit(dataset)

    short_range = DistributionClassifier(
        ModelSpec.m1(
            24,
            embedding_estimator="shrinkage",
            shrinkage_effective_size="spatial",
            shrinkage_spatial_range=0.5,
        )
    ).fit(dataset)
    long_range = DistributionClassifier(
        ModelSpec.m1(
            24,
            embedding_estimator="shrinkage",
            shrinkage_effective_size="spatial",
            shrinkage_spatial_range=5.0,
        )
    ).fit(dataset)
    assert np.mean(long_range.training_effective_sizes_) < np.mean(
        short_range.training_effective_sizes_
    )
    assert np.mean(long_range.training_shrinkage_factors_) < np.mean(
        short_range.training_shrinkage_factors_
    )

    duplicated = duplicate_all_cells(dataset, repeats=3)
    np.testing.assert_allclose(
        long_range.predict_bags(duplicated),
        long_range.predict_bags(dataset),
        atol=2e-6,
    )


def test_hybrid_raw_endpoints_reproduce_component_gram_matrices():
    dataset = _dataset()
    common = {"seed": 15, "round_exact_kernel": False}

    m0 = DistributionClassifier(ModelSpec.m0(), **common).fit(dataset)
    mean_endpoint = DistributionClassifier(
        ModelSpec.m4(
            1.0,
            hybrid_mean_representation="exact_kme",
            n_projections=10,
            n_quantiles=12,
            hybrid_normalize=False,
        ),
        **common,
    ).fit(dataset)
    np.testing.assert_allclose(mean_endpoint.gram_matrix_, m0.gram_matrix_, atol=2e-6)
    np.testing.assert_allclose(
        mean_endpoint.predict_bags(dataset), m0.predict_bags(dataset), atol=2e-5
    )

    m3 = DistributionClassifier(ModelSpec.m3(10, 12), **common).fit(dataset)
    transport_endpoint = DistributionClassifier(
        ModelSpec.m4(
            0.0,
            hybrid_mean_representation="rff_kme",
            rff_features=16,
            n_projections=10,
            n_quantiles=12,
            hybrid_normalize=False,
        ),
        **common,
    ).fit(dataset)
    np.testing.assert_allclose(transport_endpoint.gram_matrix_, m3.gram_matrix_, atol=2e-6)
    np.testing.assert_allclose(
        transport_endpoint.predict_bags(dataset), m3.predict_bags(dataset), atol=2e-5
    )
