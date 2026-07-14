#!/usr/bin/env python3
"""Classical baselines and support-scale sensitivities for Section 6.

This runner leaves the established M0--M3 comparison unchanged.  It adds:

* logistic-regression and random-forest baselines on one mean vector per bag;
* optional mean-plus-standard-deviation and geometry-only diagnostics;
* an exact background bag-size sensitivity; and
* common focal-support datasets where sites and backgrounds are extracted with
  identical square windows.

All comparisons use immutable spatial grouped fold plans.  Scores remain
presence-background relative suitability, not occurrence probabilities.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
import tracemalloc
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from rasterio.transform import xy
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from klrfome.data.formats import Bag, BagDataset
from klrfome.io.raster_source import RasterSource, build_spatial_background_bags
from klrfome.utils.validation import FoldPlan

if __package__:
    from benchmarks.run_section6_comparison import (
        _bag_diameter,
        _valid_spatial_plan,
        fold_metrics,
        prepare_setting,
        run_comparison,
    )
else:  # Direct execution from the benchmarks directory.
    from run_section6_comparison import (
        _bag_diameter,
        _valid_spatial_plan,
        fold_metrics,
        prepare_setting,
        run_comparison,
    )


BASELINE_CONFIGURATION = {
    "LR-mean": {
        "summary": "mean",
        "estimator": "standardized L2 logistic regression",
        "C": 1.0,
    },
    "RF-mean": {
        "summary": "mean",
        "estimator": "random forest",
        "n_estimators": 500,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    },
}


def bag_summary_matrix(dataset: BagDataset, summary: str = "mean") -> np.ndarray:
    """Return one fixed-length summary vector per bag."""
    means = np.asarray(
        [np.asarray(bag.samples, dtype=float).mean(axis=0) for bag in dataset.collections]
    )
    if summary == "mean":
        return means
    if summary == "mean_std":
        stds = np.asarray(
            [np.asarray(bag.samples, dtype=float).std(axis=0) for bag in dataset.collections]
        )
        return np.column_stack([means, stds])
    if summary == "geometry":
        sizes = np.asarray([bag.n_samples for bag in dataset.collections], dtype=float)
        diameters = np.asarray([_bag_diameter(bag) for bag in dataset.collections])
        return np.column_stack([np.log1p(sizes), np.log1p(diameters)])
    raise ValueError("summary must be 'mean', 'mean_std', or 'geometry'")


@dataclass
class FittedBagBaseline:
    """A fitted classical estimator that accepts canonical bag datasets."""

    name: str
    summary: str
    estimator: Any
    fit_seconds: float
    peak_python_memory_mb: float

    def predict_bags(self, dataset: BagDataset) -> np.ndarray:
        return np.asarray(
            self.estimator.predict_proba(bag_summary_matrix(dataset, self.summary))[:, 1]
        )


def fit_baseline_models(
    dataset: BagDataset,
    seed: int = 42,
    include_diagnostics: bool = False,
    rf_estimators: int = 500,
) -> Dict[str, FittedBagBaseline]:
    """Fit mean-only LR/RF baselines and optional diagnostic controls."""
    specifications: List[Tuple[str, str, Any]] = [
        (
            "LR-mean",
            "mean",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=5000, random_state=seed),
            ),
        ),
        (
            "RF-mean",
            "mean",
            RandomForestClassifier(
                n_estimators=rf_estimators,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    ]
    if include_diagnostics:
        specifications.extend(
            [
                (
                    "LR-mean-std",
                    "mean_std",
                    make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=1.0, max_iter=5000, random_state=seed),
                    ),
                ),
                (
                    "NEG-geometry",
                    "geometry",
                    make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=1.0, max_iter=5000, random_state=seed),
                    ),
                ),
            ]
        )

    labels = np.asarray(dataset.labels, dtype=int)
    fitted = {}
    for name, summary, estimator in specifications:
        tracemalloc.start()
        started = time.perf_counter()
        estimator.fit(bag_summary_matrix(dataset, summary), labels)
        fit_seconds = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        fitted[name] = FittedBagBaseline(
            name,
            summary,
            estimator,
            fit_seconds=fit_seconds,
            peak_python_memory_mb=peak_bytes / (1024**2),
        )
    return fitted


def run_baseline_comparison(
    dataset: BagDataset,
    plan: FoldPlan,
    seed: int = 42,
    include_diagnostics: bool = False,
    rf_estimators: int = 500,
) -> List[Dict[str, Any]]:
    """Evaluate classical baselines on an existing immutable fold plan."""
    rows: List[Dict[str, Any]] = []
    for assignment in plan.assignments:
        train = dataset.subset(assignment.train_indices)
        test = dataset.subset(assignment.test_indices)
        models = fit_baseline_models(
            train,
            seed=seed,
            include_diagnostics=include_diagnostics,
            rf_estimators=rf_estimators,
        )
        labels = np.asarray(test.labels, dtype=int)
        for name, model in models.items():
            tracemalloc.start()
            started = time.perf_counter()
            scores = model.predict_bags(test)
            predict_seconds = time.perf_counter() - started
            _, prediction_peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append(
                {
                    "method": name,
                    "repeat": assignment.repeat + 1,
                    "fold": assignment.fold + 1,
                    "n_train": train.n_locations,
                    "n_test": test.n_locations,
                    **fold_metrics(labels, scores),
                    "fit_seconds": model.fit_seconds,
                    "predict_seconds": predict_seconds,
                    "peak_python_memory_mb": max(
                        model.peak_python_memory_mb,
                        prediction_peak_bytes / (1024**2),
                    ),
                    "summary": model.summary,
                }
            )
    return rows


def raster_source_for_setting(root: Path, setting: Mapping[str, str]) -> RasterSource:
    """Create the same feature-ordered raster source used by the main comparison."""
    csv_path = root / setting["csv"]
    raster_directory = root / setting["raster_directory"]
    columns = pd.read_csv(csv_path, nrows=0).columns
    excluded = {"Unnamed: 0", "x", "y", "SITENO", "presence"}
    feature_names = [column for column in columns if column not in excluded]
    paths = [str(raster_directory / f"{feature}.tif") for feature in feature_names]
    return RasterSource(paths, band_names=feature_names)


def _extract_focal_bag(
    source: RasterSource,
    x_coordinate: float,
    y_coordinate: float,
    window_size: int,
    bag_id: str,
    label: int,
    group_id: str,
    stratum_id: str,
) -> Bag:
    bag = source.extract_geometry(
        Point(x_coordinate, y_coordinate),
        bag_id,
        label,
        window_size=window_size,
        group_id=group_id,
        stratum_id=stratum_id,
    )
    metadata = dict(bag.metadata or {})
    metadata.update({"sensitivity_design": "common_focal_support", "window_size": window_size})
    bag.metadata = metadata
    return bag


def build_common_focal_datasets(
    source: RasterSource,
    reference_sites: Sequence[Bag],
    window_sizes: Sequence[int] = (7, 9, 11),
    min_cells: int = 3,
    seed: int = 42,
    stratum_id: str = "riverine",
) -> Tuple[Dict[int, BagDataset], Dict[str, Any]]:
    """Build datasets with identical focal support for sites and backgrounds.

    Site anchors are the retained site centroids.  One shared set of uniformly
    sampled background anchors is accepted using the largest requested window,
    then reused for every smaller window.  Consequently every returned dataset
    has identical ordered bag IDs and can share one fold plan.
    """
    windows = sorted(set(int(size) for size in window_sizes))
    if not windows or any(size < 1 or size % 2 == 0 for size in windows):
        raise ValueError("window_sizes must contain positive odd integers")
    if not reference_sites:
        raise ValueError("reference_sites must be nonempty")

    site_bags_by_window: Dict[int, List[Bag]] = {size: [] for size in windows}
    excluded_site_ids = []
    for reference in reference_sites:
        center = np.asarray(reference.coordinates, dtype=float).mean(axis=0)
        extracted = {}
        try:
            for size in windows:
                extracted[size] = _extract_focal_bag(
                    source,
                    float(center[0]),
                    float(center[1]),
                    size,
                    reference.id,
                    1,
                    reference.group_id or reference.id,
                    stratum_id,
                )
        except ValueError:
            excluded_site_ids.append(reference.id)
            continue
        if any(bag.n_samples < min_cells for bag in extracted.values()):
            excluded_site_ids.append(reference.id)
            continue
        for size in windows:
            site_bags_by_window[size].append(extracted[size])

    retained_sites = len(site_bags_by_window[windows[-1]])
    if retained_sites < 2:
        raise RuntimeError("Too few site anchors remain for focal-support sensitivity")
    occupied = {
        tuple(map(int, cell))
        for bag in site_bags_by_window[windows[-1]]
        for cell in (bag.metadata or {}).get("cell_indices", [])
    }

    background_bags_by_window: Dict[int, List[Bag]] = {size: [] for size in windows}
    candidate_count = max(retained_sites * 25, retained_sites)
    candidates = source.sample_valid_anchors(candidate_count, seed=seed, candidate_multiplier=1)
    for anchor_row, anchor_col in candidates:
        if len(background_bags_by_window[windows[-1]]) >= retained_sites:
            break
        x_coordinate, y_coordinate = xy(source.transform, anchor_row, anchor_col, offset="center")
        index = len(background_bags_by_window[windows[-1]])
        bag_id = f"background-focal-{index:05d}"
        extracted = {}
        try:
            for size in windows:
                extracted[size] = _extract_focal_bag(
                    source,
                    float(x_coordinate),
                    float(y_coordinate),
                    size,
                    bag_id,
                    0,
                    bag_id,
                    stratum_id,
                )
        except ValueError:
            continue
        if any(bag.n_samples < min_cells for bag in extracted.values()):
            continue
        largest_cells = {
            tuple(map(int, cell))
            for cell in (extracted[windows[-1]].metadata or {}).get("cell_indices", [])
        }
        if occupied & largest_cells:
            continue
        for size in windows:
            background_bags_by_window[size].append(extracted[size])
        occupied.update(largest_cells)

    if len(background_bags_by_window[windows[-1]]) != retained_sites:
        raise RuntimeError(
            f"Could construct only {len(background_bags_by_window[windows[-1]])} "
            f"of {retained_sites} shared focal background bags"
        )

    datasets = {}
    design_audit: Dict[str, Any] = {
        "design": "common_focal_support",
        "window_sizes": windows,
        "excluded_site_ids": excluded_site_ids,
        "n_site_bags": retained_sites,
        "n_background_bags": retained_sites,
        "windows": {},
    }
    feature_names = list(source.band_names or [])
    reference_ids = None
    for size in windows:
        dataset = BagDataset(
            [*site_bags_by_window[size], *background_bags_by_window[size]],
            feature_names,
            crs=source.crs,
            study_design="presence_background",
            metadata={"design": "common_focal_support", "window_size": size},
        )
        ids = [bag.id for bag in dataset.collections]
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise RuntimeError("Focal-support datasets do not share ordered bag IDs")
        datasets[size] = dataset
        design_audit["windows"][str(size)] = {
            "site_cell_sizes": [bag.n_samples for bag in site_bags_by_window[size]],
            "background_cell_sizes": [bag.n_samples for bag in background_bags_by_window[size]],
            "site_diameters": [_bag_diameter(bag) for bag in site_bags_by_window[size]],
            "background_diameters": [_bag_diameter(bag) for bag in background_bags_by_window[size]],
        }
    return datasets, design_audit


def exact_size_background_dataset(
    source: RasterSource,
    reference_dataset: BagDataset,
    cap_cells: int,
    seed: int,
    stratum_id: str,
) -> BagDataset:
    """Regenerate the background class with an exact permutation of site sizes."""
    sites = [bag for bag in reference_dataset.collections if bag.label == 1]
    backgrounds = build_spatial_background_bags(
        source,
        sites,
        n_background=len(sites),
        cap_cells=cap_cells,
        seed=seed,
        stratum_id=stratum_id,
        match_sizes_exactly=True,
    )
    dataset = BagDataset(
        [*sites, *backgrounds],
        list(reference_dataset.feature_names),
        crs=reference_dataset.crs,
        study_design=reference_dataset.study_design,
        metadata={"design": "exact_background_size_permutation", "setting": stratum_id},
    )
    if [bag.id for bag in dataset.collections] != [bag.id for bag in reference_dataset.collections]:
        raise RuntimeError("Exact-size sensitivity changed the ordered bag IDs")
    return dataset


def _existing_model_rows(path: Path, setting_name: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return list(payload.get("settings", {}).get(setting_name, {}).get("fold_results", []))


def _design_result(
    dataset: BagDataset,
    plan: FoldPlan,
    config: Dict[str, Any],
    run_models: bool,
    existing_rows: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    model_rows = list(existing_rows)
    if run_models and not model_rows:
        model_rows = run_comparison(dataset, plan, config)
    baseline_rows = run_baseline_comparison(
        dataset,
        plan,
        seed=config["seed"],
        include_diagnostics=True,
    )
    return {"model_results": model_rows, "baseline_results": baseline_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/section6_comparison_config.json")
    parser.add_argument("--setting", default="riverine")
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[7, 9, 11])
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--output",
        default="site_data/r91_section_6_data/section6_sensitivity/results.json",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    root = Path(config["data_root"])
    setting = config["settings"][args.setting]
    original, original_plan, _, original_audit = prepare_setting(
        args.setting, setting, config, root
    )
    source = raster_source_for_setting(root, setting)
    existing_path = Path(config["output"])
    run_models = not args.baseline_only

    result: Dict[str, Any] = {
        "schema_version": "1.0",
        "setting": args.setting,
        "interpretation": "presence-background relative suitability, not occurrence probability",
        "baseline_configuration": BASELINE_CONFIGURATION,
        "designs": {},
    }
    result["designs"]["original"] = {
        "audit": original_audit,
        **_design_result(
            original,
            original_plan,
            config,
            run_models,
            existing_rows=_existing_model_rows(existing_path, args.setting),
        ),
    }

    exact = exact_size_background_dataset(
        source,
        original,
        cap_cells=config["cell_cap"],
        seed=config["seed"],
        stratum_id=args.setting,
    )
    result["designs"]["exact_background_sizes"] = {
        "audit": {
            "design": "exact_background_size_permutation",
            "site_cell_sizes": [bag.n_samples for bag in exact.collections if bag.label == 1],
            "background_cell_sizes": [bag.n_samples for bag in exact.collections if bag.label == 0],
        },
        **_design_result(exact, original_plan, config, run_models),
    }

    focal_datasets, focal_audit = build_common_focal_datasets(
        source,
        [bag for bag in original.collections if bag.label == 1],
        window_sizes=args.window_sizes,
        min_cells=config["min_cells"],
        seed=config["seed"],
        stratum_id=args.setting,
    )
    largest_window = max(focal_datasets)
    diameters = np.asarray(
        [_bag_diameter(bag) for bag in focal_datasets[largest_window].collections if bag.label == 1]
    )
    block_width = max(2.0 * float(np.percentile(diameters, 95)), max(source.resolution))
    focal_plan, _ = _valid_spatial_plan(
        focal_datasets[largest_window],
        config["n_splits"],
        config["n_repeats"],
        config["seed"],
        block_width,
    )
    for size, dataset in focal_datasets.items():
        result["designs"][f"focal_{size}"] = {
            "audit": {
                **focal_audit["windows"][str(size)],
                "design": "common_focal_support",
                "window_size": size,
                "shared_block_width": block_width,
                "excluded_site_ids": focal_audit["excluded_site_ids"],
            },
            **_design_result(dataset, focal_plan, config, run_models),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    print(output)


if __name__ == "__main__":
    main()
