"""Round-trip and validation tests for versioned fitted-model archives."""

import io
import json
import zipfile

import numpy as np
import pytest
import jax.numpy as jnp

from klrfome.api import KLRfome
from klrfome.data.synthetic import SyntheticScenarioConfig, generate_synthetic_bags
from klrfome.data.formats import RasterStack
from klrfome.models.distribution import DistributionClassifier
from klrfome.models.spec import ModelSpec
from klrfome.utils.serialization import load_model, save_model


@pytest.mark.parametrize(
    "spec",
    [
        ModelSpec.m0(),
        ModelSpec.m1(12),
        ModelSpec.m1(12, rff_scheme="orthogonal", embedding_estimator="shrinkage"),
        ModelSpec.m2(12),
        ModelSpec.m3(8, 8),
        ModelSpec.m4(
            0.4,
            rff_features=12,
            rff_scheme="orthogonal",
            embedding_estimator="shrinkage",
            n_projections=8,
            n_quantiles=8,
        ),
        ModelSpec.m4(
            0.6,
            hybrid_mean_representation="exact_kme",
            n_projections=8,
            n_quantiles=8,
        ),
    ],
)
def test_distribution_classifier_round_trip_predictions(spec, tmp_path):
    dataset = generate_synthetic_bags(
        SyntheticScenarioConfig(
            "variance_shift", n_bags_per_class=5, n_features=2, bag_size=8, seed=14
        )
    )
    model = DistributionClassifier(spec, seed=14, round_exact_kernel=False).fit(dataset)
    expected = np.asarray(model.predict_bags(dataset))
    path = tmp_path / f"{spec.method_id}.klrfome"
    save_model(model, str(path))
    restored = load_model(str(path))
    assert isinstance(restored, DistributionClassifier)
    actual = np.asarray(restored.predict_bags(dataset))
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    if spec.method_id in ("M1", "M2"):
        assert restored.training_data_ is None


def test_public_facade_round_trip_restores_fitted_core(tmp_path):
    dataset = generate_synthetic_bags(
        SyntheticScenarioConfig("mean_shift", n_bags_per_class=5, bag_size=7, seed=8)
    )
    model = KLRfome(n_rff_features=12, seed=8).fit(dataset)
    expected = np.asarray(model._core_model.predict_bags(dataset))
    path = tmp_path / "facade.klrfome"
    save_model(model, str(path))
    restored = load_model(str(path))
    assert isinstance(restored, KLRfome)
    assert restored._core_model is not None
    np.testing.assert_allclose(restored._core_model.predict_bags(dataset), expected, atol=2e-6)
    raster = RasterStack(
        data=jnp.asarray(np.arange(3 * 4 * 4, dtype=float).reshape(3, 4, 4)),
        transform=None,
        crs=dataset.crs,
        band_names=list(dataset.feature_names),
    )
    np.testing.assert_allclose(
        restored.predict(raster, show_progress=False),
        model.predict(raster, show_progress=False),
        atol=2e-6,
    )


def test_archive_rejects_unknown_version(tmp_path):
    dataset = generate_synthetic_bags(
        SyntheticScenarioConfig("mean_shift", n_bags_per_class=4, bag_size=6, seed=2)
    )
    path = tmp_path / "model.klrfome"
    save_model(DistributionClassifier(ModelSpec.m1(8), seed=2).fit(dataset), str(path))
    with zipfile.ZipFile(path, "r") as source:
        manifest = json.loads(source.read("manifest.json"))
        arrays = source.read("arrays.npz")
    manifest["archive_version"] = "999"
    changed = tmp_path / "changed.klrfome"
    with zipfile.ZipFile(changed, "w") as destination:
        destination.writestr("manifest.json", json.dumps(manifest))
        destination.writestr("arrays.npz", io.BytesIO(arrays).getvalue())
    with pytest.raises(ValueError, match="Unsupported model archive version"):
        load_model(str(changed))


def test_loaded_model_rejects_feature_reordering(tmp_path):
    dataset = generate_synthetic_bags(
        SyntheticScenarioConfig("mean_shift", n_bags_per_class=4, n_features=2, bag_size=6)
    )
    path = tmp_path / "model.klrfome"
    save_model(DistributionClassifier(ModelSpec.m1(8)).fit(dataset), str(path))
    restored = load_model(str(path))
    reordered = type(dataset)(
        dataset.collections,
        list(reversed(dataset.feature_names)),
        crs=dataset.crs,
        study_design=dataset.study_design,
    )
    with pytest.raises(ValueError, match="feature order"):
        restored.predict_bags(reordered)
