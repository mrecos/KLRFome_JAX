#!/usr/bin/env python3
"""Support-controlled, availability-based evaluation for Section 6.

The primary design represents both sites and training backgrounds with the
same focal window.  Every fitted fold also scores one fixed, uniform sample of
valid raster availability.  Held-out scores are converted to percentiles of
that fold-specific availability distribution before folds are pooled within a
repeat.  This makes mapped-area capture, lift, and Continuous Boyce the primary
metrics while retaining AUC and PR AUC as secondary diagnostics.
"""

import argparse
import json
from pathlib import Path
import time
import tracemalloc
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from rasterio.transform import xy
from rasterio.windows import Window
from sklearn.metrics import average_precision_score, roc_auc_score

from klrfome.data.formats import Bag, BagDataset
from klrfome.io.raster_source import RasterSource
from klrfome.models.baselines import BagSummaryClassifier, baseline_models
from klrfome.models.distribution import DistributionClassifier
from klrfome.models.spec import ModelSpec
from klrfome.utils.evaluation import (
    availability_capture_metrics,
    availability_percentile_ranks,
    continuous_boyce_from_availability,
    paired_method_differences,
)
from klrfome.utils.reproducibility import (
    configuration_fingerprint,
    dataset_fingerprint,
    environment_manifest,
    serialize_fold_plan,
    write_strict_json,
)
from klrfome.utils.validation import FoldPlan

if __package__:
    from benchmarks.run_section6_comparison import (
        _bag_diameter,
        _valid_spatial_plan,
        prepare_setting,
    )
    from benchmarks.run_section6_sensitivity import (
        build_common_focal_datasets,
        raster_source_for_setting,
    )
else:
    from run_section6_comparison import _bag_diameter, _valid_spatial_plan, prepare_setting
    from run_section6_sensitivity import (
        build_common_focal_datasets,
        raster_source_for_setting,
    )

FittedModel = Union[DistributionClassifier, BagSummaryClassifier]


def method_models(configuration: Mapping[str, Any]) -> Dict[str, FittedModel]:
    """Return the configured unfitted reference models and diagnostic baselines."""
    seed = int(configuration["seed"])
    models: Dict[str, FittedModel] = {
        "M0": DistributionClassifier(
            ModelSpec.m0(),
            lambda_reg=float(configuration["lambda_reg"]),
            seed=seed,
            round_exact_kernel=True,
        ),
        "M1": DistributionClassifier(
            ModelSpec.m1(int(configuration["rff_features"])),
            lambda_reg=float(configuration["lambda_reg"]),
            seed=seed,
            round_exact_kernel=True,
        ),
        "M2": DistributionClassifier(
            ModelSpec.m2(int(configuration["rff_features"])),
            lambda_reg=float(configuration["lambda_reg"]),
            seed=seed,
            round_exact_kernel=True,
        ),
        "M3": DistributionClassifier(
            ModelSpec.m3(
                int(configuration["wasserstein_projections"]),
                int(configuration["wasserstein_quantiles"]),
            ),
            lambda_reg=float(configuration["lambda_reg"]),
            seed=seed,
            round_exact_kernel=True,
        ),
    }
    models.update(
        baseline_models(
            seed=seed,
            rf_estimators=int(configuration.get("rf_estimators", 500)),
            include_geometry=True,
        )
    )
    return models


def build_focal_availability_datasets(
    source: RasterSource,
    window_sizes: Sequence[int],
    n_anchors: int,
    min_cells: int,
    seed: int,
    stratum_id: str,
    excluded_cells: Sequence[Tuple[int, int]] = (),
) -> Tuple[Dict[int, BagDataset], Dict[str, Any]]:
    """Extract one fixed set of uniform valid anchors at every support size."""
    windows = sorted(set(int(value) for value in window_sizes))
    if not windows or any(value < 1 or value % 2 == 0 for value in windows):
        raise ValueError("window_sizes must contain positive odd integers")
    if n_anchors < 1:
        raise ValueError("n_anchors must be positive")
    excluded = {tuple(map(int, cell)) for cell in excluded_cells}
    accepted: Dict[int, List[Bag]] = {window: [] for window in windows}
    candidates = source.sample_valid_anchors(
        max(n_anchors * 8, n_anchors), seed=seed, candidate_multiplier=1
    )
    requests = []
    largest = windows[-1]
    largest_half = largest // 2
    for anchor_row, anchor_col in candidates:
        if (anchor_row, anchor_col) in excluded:
            continue
        x_coordinate, y_coordinate = xy(source.transform, anchor_row, anchor_col, offset="center")
        window = Window(
            max(0, anchor_col - largest_half),
            max(0, anchor_row - largest_half),
            min(source.width, anchor_col + largest_half + 1) - max(0, anchor_col - largest_half),
            min(source.height, anchor_row + largest_half + 1) - max(0, anchor_row - largest_half),
        )
        requests.append((anchor_row, anchor_col, float(x_coordinate), float(y_coordinate), window))

    rejected = len(candidates) - len(requests)
    for request, (stack, valid) in zip(
        requests, source.read_windows(item[-1] for item in requests)
    ):
        if len(accepted[windows[-1]]) >= n_anchors:
            break
        anchor_row, anchor_col, x_coordinate, y_coordinate, window = request
        identifier = f"availability-{len(accepted[windows[-1]]):05d}"
        local_rows, local_cols = np.nonzero(valid)
        if len(local_rows) < min_cells:
            rejected += 1
            continue
        global_rows: NDArray[np.int_] = local_rows + int(window.row_off)
        global_cols: NDArray[np.int_] = local_cols + int(window.col_off)
        samples = stack[:, local_rows, local_cols].T
        cell_x, cell_y = xy(source.transform, global_rows, global_cols, offset="center")
        coordinates = np.column_stack([cell_x, cell_y])
        extracted: Dict[int, Bag] = {}
        for support in windows:
            half = support // 2
            selected = (np.abs(global_rows - anchor_row) <= half) & (
                np.abs(global_cols - anchor_col) <= half
            )
            if int(selected.sum()) < min_cells:
                extracted = {}
                break
            extracted[support] = Bag(
                samples[selected],
                0,
                identifier,
                metadata={
                    "adapter": "raster",
                    "feature_names": list(cast(Sequence[str], source.band_names)),
                    "crs": source.crs,
                    "cell_indices": np.column_stack(
                        [global_rows[selected], global_cols[selected]]
                    ).tolist(),
                    "evaluation_role": "mapped_availability",
                    "anchor_cell": [anchor_row, anchor_col],
                    "anchor_xy": [x_coordinate, y_coordinate],
                    "window_size": support,
                },
                coordinates=coordinates[selected],
                group_id=identifier,
                stratum_id=stratum_id,
            )
        if not extracted:
            rejected += 1
            continue
        for support, bag in extracted.items():
            accepted[support].append(bag)
    if len(accepted[windows[-1]]) != n_anchors:
        raise RuntimeError(
            f"Could construct only {len(accepted[windows[-1]])} of {n_anchors} "
            "availability focal bags"
        )

    datasets: Dict[int, BagDataset] = {}
    ordered_ids: Optional[List[str]] = None
    for window in windows:
        dataset = BagDataset(
            accepted[window],
            list(cast(Sequence[str], source.band_names)),
            crs=source.crs,
            study_design="presence_background",
            metadata={
                "evaluation_role": "mapped_availability",
                "window_size": window,
                "seed": seed,
            },
        )
        ids = [bag.id for bag in dataset.collections]
        if ordered_ids is None:
            ordered_ids = ids
        elif ids != ordered_ids:
            raise RuntimeError("Availability datasets do not share ordered anchors")
        datasets[window] = dataset
    audit = {
        "n_requested": n_anchors,
        "n_retained": len(accepted[windows[-1]]),
        "n_rejected": rejected,
        "seed": seed,
        "window_sizes": windows,
        "anchor_ids": ordered_ids,
        "anchor_xy": [(bag.metadata or {})["anchor_xy"] for bag in accepted[windows[-1]]],
        "dataset_fingerprints": {
            str(window): dataset_fingerprint(datasets[window]) for window in windows
        },
    }
    return datasets, audit


def _fresh_model(model: FittedModel) -> FittedModel:
    if isinstance(model, BagSummaryClassifier):
        return model.clone()
    return model.clone()


def _predict(model: FittedModel, dataset: BagDataset) -> np.ndarray:
    return np.asarray(model.predict_bags(dataset), dtype=float)


def evaluate_design(
    dataset: BagDataset,
    availability: BagDataset,
    plan: FoldPlan,
    configuration: Mapping[str, Any],
    selected_methods: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Fit all methods on one immutable plan and pool OOF rows by repeat."""
    prototypes = method_models(configuration)
    if selected_methods is not None:
        missing = sorted(set(selected_methods) - set(prototypes))
        if missing:
            raise ValueError(f"Unknown methods: {missing}")
        prototypes = {name: prototypes[name] for name in selected_methods}
    fold_results: List[Dict[str, Any]] = []
    for method_name, prototype in prototypes.items():
        for assignment in plan.assignments:
            print(
                f"starting {method_name} repeat={assignment.repeat + 1} "
                f"fold={assignment.fold + 1}",
                flush=True,
            )
            train = dataset.subset(assignment.train_indices)
            test = dataset.subset(assignment.test_indices)
            model = _fresh_model(prototype)
            tracemalloc.start()
            fit_started = time.perf_counter()
            model.fit(train)
            fit_seconds = time.perf_counter() - fit_started
            prediction_started = time.perf_counter()
            test_scores = _predict(model, test)
            availability_scores = _predict(model, availability)
            predict_seconds = time.perf_counter() - prediction_started
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            if not np.isfinite(test_scores).all() or not np.isfinite(availability_scores).all():
                raise RuntimeError(
                    f"{method_name} produced nonfinite predictions for repeat "
                    f"{assignment.repeat + 1}, fold {assignment.fold + 1}"
                )
            test_percentiles = availability_percentile_ranks(test_scores, availability_scores)
            availability_percentiles = availability_percentile_ranks(
                availability_scores, availability_scores
            )
            labels = np.asarray(test.labels, dtype=int)
            has_both = np.unique(labels).size == 2
            test_predictions = []
            for bag, score, percentile in zip(test.collections, test_scores, test_percentiles):
                center = np.asarray(bag.coordinates, dtype=float).mean(axis=0)
                test_predictions.append(
                    {
                        "bag_id": bag.id,
                        "label": bag.label,
                        "x": float(center[0]),
                        "y": float(center[1]),
                        "raw_score": float(score),
                        "availability_percentile": float(percentile),
                    }
                )
            availability_predictions = []
            for bag, score, percentile in zip(
                availability.collections, availability_scores, availability_percentiles
            ):
                anchor = (bag.metadata or {})["anchor_xy"]
                availability_predictions.append(
                    {
                        "bag_id": bag.id,
                        "x": float(anchor[0]),
                        "y": float(anchor[1]),
                        "raw_score": float(score),
                        "availability_percentile": float(percentile),
                    }
                )
            diagnostics = (
                dict(model.diagnostics_)
                if isinstance(model, DistributionClassifier)
                else {"summary": model.summary, "estimator_kind": model.estimator_kind}
            )
            fold_results.append(
                {
                    "method": method_name,
                    "repeat": assignment.repeat + 1,
                    "fold": assignment.fold + 1,
                    "n_train": train.n_locations,
                    "n_test": test.n_locations,
                    "auc_secondary": (
                        float(roc_auc_score(labels, test_percentiles)) if has_both else None
                    ),
                    "pr_auc_secondary": (
                        float(average_precision_score(labels, test_percentiles))
                        if has_both
                        else None
                    ),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "peak_python_memory_mb": peak_bytes / (1024**2),
                    "test_predictions": test_predictions,
                    "availability_predictions": availability_predictions,
                    "diagnostics": diagnostics,
                }
            )
            print(
                f"completed {method_name} repeat={assignment.repeat + 1} "
                f"fold={assignment.fold + 1} fit={fit_seconds:.2f}s "
                f"predict={predict_seconds:.2f}s",
                flush=True,
            )
    pooled = _pool_repeat_results(fold_results, plan, configuration)
    metrics = (
        "capture_5_percent",
        "capture_10_percent",
        "capture_20_percent",
        "lift_5_percent",
        "lift_10_percent",
        "lift_20_percent",
        "boyce",
        "site_percentile_median",
        "auc_secondary",
        "pr_auc_secondary",
    )
    return {
        "fold_results": fold_results,
        "pooled_repeat_results": pooled,
        "paired_differences_vs_M0": paired_method_differences(
            pooled,
            baseline="M0",
            metrics=metrics,
            pairing_keys=("repeat",),
        ),
    }


def _pool_repeat_results(
    fold_results: Sequence[Mapping[str, Any]],
    plan: FoldPlan,
    configuration: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    pooled = []
    fractions = tuple(float(value) for value in configuration["area_fractions"])
    methods = sorted({str(row["method"]) for row in fold_results})
    for method in methods:
        for repeat in range(1, plan.n_repeats + 1):
            rows = [
                row
                for row in fold_results
                if row["method"] == method and int(row["repeat"]) == repeat
            ]
            predictions = [
                prediction
                for row in rows
                for prediction in cast(Sequence[Mapping[str, Any]], row["test_predictions"])
            ]
            if len(predictions) != len(plan.bag_ids):
                raise RuntimeError(
                    f"{method} repeat {repeat} does not contain each bag exactly once"
                )
            bag_ids = [str(row["bag_id"]) for row in predictions]
            if len(set(bag_ids)) != len(bag_ids) or set(bag_ids) != set(plan.bag_ids):
                raise RuntimeError(f"{method} repeat {repeat} has invalid OOF bag coverage")
            labels = np.asarray([row["label"] for row in predictions], dtype=int)
            percentiles = np.asarray(
                [row["availability_percentile"] for row in predictions], dtype=float
            )
            sites = percentiles[labels == 1]
            availability_reference = np.asarray(
                [
                    prediction["availability_percentile"]
                    for row in rows
                    for prediction in cast(
                        Sequence[Mapping[str, Any]], row["availability_predictions"]
                    )
                ],
                dtype=float,
            )
            capture_rows = availability_capture_metrics(sites, availability_reference, fractions)
            capture_lookup = {
                int(round(100 * cast(float, row["area_fraction"]))): row for row in capture_rows
            }
            boyce = continuous_boyce_from_availability(
                sites,
                availability_reference,
                n_windows=int(configuration.get("boyce_windows", 20)),
                window_fraction=float(configuration.get("boyce_window_fraction", 0.10)),
            )
            has_both = np.unique(labels).size == 2
            output: Dict[str, Any] = {
                "method": method,
                "repeat": repeat,
                "n_observations": len(predictions),
                "n_sites": int(np.sum(labels == 1)),
                "n_background": int(np.sum(labels == 0)),
                "bag_ids": bag_ids,
                "labels": labels.tolist(),
                "availability_percentiles": percentiles.tolist(),
                "site_percentiles": sites.tolist(),
                "capture_curve": capture_rows,
                "boyce": boyce["boyce"],
                "boyce_curve": {
                    "window_midpoints": boyce["window_midpoints"],
                    "predicted_expected_ratios": boyce["predicted_expected_ratios"],
                },
                "site_percentile_median": float(np.median(sites)),
                "site_percentile_q25": float(np.quantile(sites, 0.25)),
                "site_percentile_q75": float(np.quantile(sites, 0.75)),
                "auc_secondary": (float(roc_auc_score(labels, percentiles)) if has_both else None),
                "pr_auc_secondary": (
                    float(average_precision_score(labels, percentiles)) if has_both else None
                ),
                "fit_seconds": float(sum(float(row["fit_seconds"]) for row in rows)),
                "predict_seconds": float(sum(float(row["predict_seconds"]) for row in rows)),
                "peak_python_memory_mb": float(
                    max(float(row["peak_python_memory_mb"]) for row in rows)
                ),
            }
            for percentage in (5, 10, 20):
                if percentage not in capture_lookup:
                    continue
                capture = capture_lookup[percentage]
                output[f"capture_{percentage}_percent"] = capture["capture"]
                output[f"achieved_area_{percentage}_percent"] = capture["achieved_area_fraction"]
                output[f"lift_{percentage}_percent"] = capture["lift"]
                output[f"gain_{percentage}_percent"] = capture["gain"]
            pooled.append(output)
    return pooled


def _occupied_cells(bags: Sequence[Bag]) -> List[Tuple[int, int]]:
    return [
        (int(row), int(column))
        for bag in bags
        for row, column in (bag.metadata or {}).get("cell_indices", [])
    ]


def prepare_evaluation_setting(
    name: str,
    setting: Mapping[str, str],
    configuration: Dict[str, Any],
    root: Path,
) -> Dict[str, Any]:
    """Build original, common-support, and fixed-availability datasets."""
    original, original_plan, original_groups, original_audit = prepare_setting(
        name, dict(setting), configuration, root
    )
    source = raster_source_for_setting(root, setting)
    windows = [int(value) for value in configuration["support_windows"]]
    focal, focal_audit = build_common_focal_datasets(
        source,
        [bag for bag in original.collections if bag.label == 1],
        window_sizes=windows,
        min_cells=int(configuration["min_cells"]),
        seed=int(configuration["seed"]),
        stratum_id=name,
    )
    largest = max(windows)
    largest_sites = [bag for bag in focal[largest].collections if bag.label == 1]
    availability, availability_audit = build_focal_availability_datasets(
        source,
        windows,
        n_anchors=int(configuration["availability_sample_size"]),
        min_cells=int(configuration["min_cells"]),
        seed=int(configuration["availability_seed"]),
        stratum_id=name,
        excluded_cells=_occupied_cells(largest_sites),
    )
    diameters = np.asarray([_bag_diameter(bag) for bag in largest_sites])
    block_width = max(2.0 * float(np.percentile(diameters, 95)), max(source.resolution))
    focal_plan, focal_groups = _valid_spatial_plan(
        focal[largest],
        int(configuration["n_splits"]),
        int(configuration["n_repeats"]),
        int(configuration["seed"]),
        block_width,
    )
    return {
        "source": source,
        "original": original,
        "original_plan": original_plan,
        "original_groups": original_groups,
        "original_audit": original_audit,
        "focal": focal,
        "focal_plan": focal_plan,
        "focal_groups": focal_groups,
        "focal_audit": focal_audit,
        "availability": availability,
        "availability_audit": availability_audit,
        "shared_block_width": block_width,
    }


def run_setting(
    name: str,
    prepared: Mapping[str, Any],
    configuration: Dict[str, Any],
    mode: str,
    selected_methods: Optional[Sequence[str]],
) -> Dict[str, Any]:
    primary = int(configuration["primary_window"])
    windows = [int(value) for value in configuration["support_windows"]]
    designs = [f"focal_{primary}"]
    if mode == "full":
        designs.extend(f"focal_{window}" for window in windows if window != primary)
        designs.append("original_irregular")
    result: Dict[str, Any] = {
        "data_audit": {
            "original": prepared["original_audit"],
            "common_focal": prepared["focal_audit"],
            "availability": prepared["availability_audit"],
            "shared_block_width": prepared["shared_block_width"],
        },
        "designs": {},
    }
    for design in designs:
        if design == "original_irregular":
            dataset = cast(BagDataset, prepared["original"])
            plan = cast(FoldPlan, prepared["original_plan"])
            groups = cast(Sequence[str], prepared["original_groups"])
            availability = cast(Mapping[int, BagDataset], prepared["availability"])[primary]
            support_note = "original site/background bags; focal mapped availability"
        else:
            window = int(design.split("_")[1])
            dataset = cast(Mapping[int, BagDataset], prepared["focal"])[window]
            plan = cast(FoldPlan, prepared["focal_plan"])
            groups = cast(Sequence[str], prepared["focal_groups"])
            availability = cast(Mapping[int, BagDataset], prepared["availability"])[window]
            support_note = "identical focal support for sites, backgrounds, and availability"
        design_result: Dict[str, Any] = {
            "support_note": support_note,
            "dataset_fingerprint": dataset_fingerprint(dataset),
            "availability_fingerprint": dataset_fingerprint(availability),
            "fold_plan": serialize_fold_plan(plan, dataset, groups),
            "bag_index": [
                {
                    "bag_id": bag.id,
                    "label": bag.label,
                    "x": float(np.asarray(bag.coordinates, dtype=float).mean(axis=0)[0]),
                    "y": float(np.asarray(bag.coordinates, dtype=float).mean(axis=0)[1]),
                }
                for bag in dataset.collections
            ],
        }
        if mode != "prepare":
            design_result.update(
                evaluate_design(
                    dataset,
                    availability,
                    plan,
                    configuration,
                    selected_methods=selected_methods,
                )
            )
        result["designs"][design] = design_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/section6_evaluation_config.json")
    parser.add_argument("--mode", choices=("prepare", "primary", "full"), default="primary")
    parser.add_argument("--setting", choices=("all", "riverine", "upland"), default="all")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    config_path = Path(arguments.config)
    configuration = json.loads(config_path.read_text())
    root = Path(configuration["data_root"])
    names = list(configuration["settings"]) if arguments.setting == "all" else [arguments.setting]
    payload: Dict[str, Any] = {
        "schema_version": "2.0",
        "configuration": configuration,
        "configuration_sha256": configuration_fingerprint(configuration),
        "mode": arguments.mode,
        "interpretation": "presence-background relative suitability, not occurrence probability",
        "metric_hierarchy": {
            "primary": [
                "mapped availability capture and lift at 5%, 10%, and 20%",
                "continuous Boyce index",
                "held-out site availability percentiles",
            ],
            "secondary": ["ROC AUC", "PR AUC"],
            "diagnostic": [
                "geometry-only control",
                "fit and prediction time",
                "peak Python memory",
            ],
        },
        "environment": environment_manifest(Path.cwd()),
        "settings": {},
    }
    for name in names:
        prepared = prepare_evaluation_setting(
            name,
            configuration["settings"][name],
            configuration,
            root,
        )
        payload["settings"][name] = run_setting(
            name,
            prepared,
            configuration,
            arguments.mode,
            arguments.methods,
        )
        print(f"completed {name}", flush=True)
    output = Path(arguments.output or configuration["output"])
    if arguments.mode == "prepare":
        output = output.with_name("prepared_evaluation_audit.json")
    write_strict_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()
