"""Tests for the support-controlled Section 6 evaluation contract."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import rasterio
from jsonschema import validate
from rasterio.transform import from_origin

from benchmarks.run_section6_evaluation import (
    _annotate_mask_boundary_distances,
    build_focal_availability_datasets,
    evaluate_design,
    full_window_only_dataset,
    matched_cell_count_dataset,
)
from klrfome.data.formats import Bag, BagDataset
from klrfome.io.raster_source import RasterSource
from klrfome.utils.reproducibility import configuration_fingerprint
from klrfome.utils.validation import make_fold_plan


def _write(path: Path, values: np.ndarray) -> str:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:26918",
        transform=from_origin(0, values.shape[0], 1, 1),
        nodata=-9999.0,
    ) as destination:
        destination.write(values.astype("float32"), 1)
    return str(path)


def _bag_dataset(seed: int = 4) -> BagDataset:
    rng = np.random.default_rng(seed)
    bags = []
    for label, offset in ((0, -0.4), (1, 0.4)):
        for index in range(6):
            samples = rng.normal(offset, 0.8, size=(9, 2))
            coordinates = np.column_stack(
                [np.arange(9, dtype=float) + index * 20, np.full(9, label * 20)]
            )
            bags.append(
                Bag(
                    jnp.asarray(samples),
                    label,
                    f"{label}-{index}",
                    coordinates=jnp.asarray(coordinates),
                    group_id=f"{label}-{index}",
                    stratum_id="test",
                )
            )
    return BagDataset(bags, ["a", "b"], crs="EPSG:26918")


def _configuration():
    return {
        "seed": 8,
        "lambda_reg": 0.1,
        "rff_features": 16,
        "wasserstein_projections": 8,
        "wasserstein_quantiles": 8,
        "rf_estimators": 10,
        "area_fractions": [0.05, 0.10, 0.20],
        "boyce_windows": 10,
        "boyce_window_fraction": 0.2,
        "spatial_neighbors": 2,
        "spatial_permutations": 19,
        "spatial_fdr_alpha": 0.05,
    }


def test_tracked_configuration_and_schema_define_the_primary_metric_contract():
    root = Path(__file__).parents[1]
    configuration = json.loads((root / "benchmarks/section6_evaluation_config.json").read_text())
    assert configuration["primary_window"] == 7
    assert configuration["support_windows"] == [7, 9, 11]
    assert configuration["area_fractions"] == [0.05, 0.10, 0.20]
    schema = json.loads((root / "benchmarks/section6_evaluation_result_schema.json").read_text())
    validate(
        {
            "schema_version": "2.0",
            "configuration": configuration,
            "configuration_sha256": configuration_fingerprint(configuration),
            "mode": "prepare",
            "metric_hierarchy": {},
            "settings": {
                "test": {
                    "data_audit": {},
                    "designs": {
                        "focal_7": {
                            "support_note": "test",
                            "dataset_fingerprint": "data",
                            "availability_fingerprint": "availability",
                            "fold_plan": {},
                            "bag_index": [],
                        }
                    },
                }
            },
        },
        schema,
    )


def test_availability_builder_reuses_ordered_uniform_anchors(tmp_path):
    values = np.arange(400, dtype=float).reshape(20, 20)
    source = RasterSource(
        [_write(tmp_path / "a.tif", values), _write(tmp_path / "b.tif", values * 2)],
        band_names=["a", "b"],
    )
    datasets, audit = build_focal_availability_datasets(
        source,
        [3, 5],
        n_anchors=12,
        min_cells=3,
        seed=7,
        stratum_id="test",
        excluded_cells=[(10, 10)],
    )
    assert audit["n_retained"] == 12
    assert [bag.id for bag in datasets[3].collections] == [
        bag.id for bag in datasets[5].collections
    ]
    assert audit["dataset_fingerprints"]["3"] != audit["dataset_fingerprints"]["5"]
    assert all(
        (bag.metadata or {})["evaluation_role"] == "mapped_availability"
        for bag in datasets[3].collections
    )


def test_evaluation_pools_each_bag_once_and_uses_availability_percentiles():
    dataset = _bag_dataset()
    availability = BagDataset(
        [
            Bag(
                jnp.asarray(np.full((9, 2), value)),
                0,
                f"available-{index}",
                coordinates=jnp.asarray(
                    np.column_stack([np.arange(9, dtype=float), np.full(9, index)])
                ),
                metadata={"anchor_xy": [float(index), float(index)]},
            )
            for index, value in enumerate(np.linspace(-1.5, 1.5, 50))
        ],
        ["a", "b"],
        crs="EPSG:26918",
    )
    plan = make_fold_plan(
        dataset,
        n_splits=3,
        n_repeats=2,
        seed=8,
        group_ids=[bag.id for bag in dataset.collections],
    )
    result = evaluate_design(
        dataset,
        availability,
        plan,
        _configuration(),
        selected_methods=["M0", "M1", "LR-mean"],
    )
    assert len(result["fold_results"]) == 3 * len(plan.assignments)
    assert len(result["pooled_repeat_results"]) == 6
    assert len(result["availability_disagreement_vs_M0"]) == 4
    for row in result["pooled_repeat_results"]:
        assert row["n_observations"] == dataset.n_locations
        assert len(set(row["bag_ids"])) == dataset.n_locations
        assert 0 <= min(row["availability_percentiles"]) <= 1
        assert 0 <= max(row["availability_percentiles"]) <= 1
        assert row["capture_5_percent"] <= row["capture_10_percent"]
        assert row["capture_10_percent"] <= row["capture_20_percent"]
        assert row["capture_surplus_10_percent"] == pytest.approx(
            row["capture_10_percent"] - row["achieved_area_10_percent"]
        )


def _variable_size_dataset() -> BagDataset:
    bags = []
    for label, sizes in ((1, [9, 9, 8, 6]), (0, [9, 9, 9, 7])):
        for index, size in enumerate(sizes):
            coordinates = np.column_stack([np.arange(size, dtype=float), np.full(size, index)])
            bags.append(
                Bag(
                    jnp.ones((size, 2)) * (label + index),
                    label,
                    f"{label}-{index}",
                    coordinates=jnp.asarray(coordinates),
                    metadata={
                        "window_size": 3,
                        "cell_indices": np.column_stack(
                            [np.arange(size), np.arange(size)]
                        ).tolist(),
                    },
                )
            )
    return BagDataset(bags, ["a", "b"], crs="EPSG:26918")


def test_geometry_sensitivities_require_complete_windows_and_match_counts():
    dataset = _variable_size_dataset()
    complete, complete_audit = full_window_only_dataset(dataset, 3, seed=4)
    assert complete.n_sites == complete.n_background == 2
    assert {bag.n_samples for bag in complete.collections} == {9}
    assert complete_audit["required_cells"] == 9

    matched, matched_audit = matched_cell_count_dataset(dataset, seed=4)
    site_sizes = sorted(bag.n_samples for bag in matched.collections if bag.label == 1)
    background_sizes = sorted(bag.n_samples for bag in matched.collections if bag.label == 0)
    assert site_sizes == background_sizes
    assert matched_audit["site_cell_sizes"] == matched_audit["background_cell_sizes"]


def test_mask_boundary_distance_is_attached_to_anchor_bags(tmp_path):
    values = np.ones((9, 9), dtype=float)
    values[:, 0] = -9999.0
    source = RasterSource(
        [_write(tmp_path / "mask-a.tif", values), _write(tmp_path / "mask-b.tif", values)],
        band_names=["a", "b"],
    )
    bags = []
    for identifier, row, column in (("near", 4, 1), ("far", 4, 6)):
        x_coordinate, y_coordinate = rasterio.transform.xy(
            source.transform, row, column, offset="center"
        )
        bags.append(
            Bag(
                jnp.ones((1, 2)),
                0,
                identifier,
                coordinates=jnp.asarray([[x_coordinate, y_coordinate]]),
                metadata={"anchor_cell": [row, column], "window_size": 1},
            )
        )
    dataset = BagDataset(bags, ["a", "b"], crs=source.crs)
    audit = _annotate_mask_boundary_distances(source, [dataset])
    distances = {
        bag.id: (bag.metadata or {})["distance_to_mask_boundary"] for bag in dataset.collections
    }
    assert distances["far"] > distances["near"]
    assert audit["n_annotated_bags"] == 2
