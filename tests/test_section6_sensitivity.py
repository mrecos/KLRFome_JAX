"""Tests for Section 6 classical baselines and support sensitivities."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from benchmarks.run_section6_sensitivity import (
    bag_summary_matrix,
    build_common_focal_datasets,
    run_baseline_comparison,
)
from klrfome.data.formats import Bag, BagDataset
from klrfome.io.raster_source import RasterSource
from klrfome.utils.validation import make_fold_plan


def _dataset(seed: int = 5) -> BagDataset:
    rng = np.random.default_rng(seed)
    bags = []
    for label, offset in ((0, -0.6), (1, 0.6)):
        for index in range(6):
            samples = rng.normal(offset, 0.7, size=(6 + index, 2))
            coordinates = np.column_stack(
                [np.arange(len(samples), dtype=float) + index * 20, np.full(len(samples), label)]
            )
            bags.append(
                Bag(
                    jnp.asarray(samples),
                    label,
                    f"{label}-{index}",
                    coordinates=jnp.asarray(coordinates),
                    group_id=f"group-{label}-{index}",
                    stratum_id="test",
                )
            )
    return BagDataset(bags, ["a", "b"], crs="EPSG:26918")


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


def test_bag_summary_baselines_share_fold_plan_and_remain_finite():
    dataset = _dataset()
    plan = make_fold_plan(
        dataset,
        n_splits=3,
        n_repeats=2,
        seed=9,
        group_ids=[bag.id for bag in dataset.collections],
    )
    rows = run_baseline_comparison(
        dataset,
        plan,
        seed=9,
        include_diagnostics=False,
        rf_estimators=20,
    )
    assert len(rows) == 2 * len(plan.assignments)
    assert {row["method"] for row in rows} == {"LR-mean", "RF-mean"}
    assert {(row["repeat"], row["fold"]) for row in rows} == {
        (assignment.repeat + 1, assignment.fold + 1) for assignment in plan.assignments
    }
    assert np.isfinite([[row["auc"], row["pr_auc"]] for row in rows]).all()
    assert all(row["fit_seconds"] >= 0 for row in rows)
    assert all(row["predict_seconds"] >= 0 for row in rows)
    assert all(row["peak_python_memory_mb"] >= 0 for row in rows)
    assert bag_summary_matrix(dataset, "mean").shape == (dataset.n_locations, 2)
    assert bag_summary_matrix(dataset, "mean_std").shape == (dataset.n_locations, 4)


def test_common_focal_sensitivity_uses_shared_ids_and_support(tmp_path):
    values = np.arange(144, dtype=float).reshape(12, 12)
    source = RasterSource(
        [_write(tmp_path / "a.tif", values), _write(tmp_path / "b.tif", values * 2)],
        band_names=["a", "b"],
    )
    references = [
        source.extract_geometry(box(1, 9, 3, 11), "site-a", 1),
        source.extract_geometry(box(9, 1, 11, 3), "site-b", 1),
    ]
    datasets, audit = build_common_focal_datasets(
        source,
        references,
        window_sizes=[3, 5],
        min_cells=3,
        seed=4,
        stratum_id="test",
    )
    assert audit["n_site_bags"] == audit["n_background_bags"] == 2
    assert [bag.id for bag in datasets[3].collections] == [
        bag.id for bag in datasets[5].collections
    ]
    for window_size, dataset in datasets.items():
        assert dataset.n_sites == 2
        assert dataset.n_background == 2
        assert all(bag.n_samples <= window_size**2 for bag in dataset.collections)
        site_cells = {
            tuple(cell)
            for bag in dataset.collections
            if bag.label == 1
            for cell in bag.metadata["cell_indices"]
        }
        background_cells = {
            tuple(cell)
            for bag in dataset.collections
            if bag.label == 0
            for cell in bag.metadata["cell_indices"]
        }
        assert not site_cells & background_cells
        background_cell_sets = [
            {tuple(cell) for cell in bag.metadata["cell_indices"]}
            for bag in dataset.collections
            if bag.label == 0
        ]
        assert not background_cell_sets[0] & background_cell_sets[1]
