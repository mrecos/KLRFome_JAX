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
from scipy.ndimage import distance_transform_edt
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
    spatial_autocorrelation_diagnostics,
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


def _bag_geometry_fields(bag: Bag) -> Dict[str, Any]:
    metadata = bag.metadata or {}
    window_size = metadata.get("window_size")
    capacity = int(window_size) ** 2 if window_size is not None else None
    return {
        "valid_cell_count": bag.n_samples,
        "window_capacity": capacity,
        "valid_cell_fraction": bag.n_samples / capacity if capacity else None,
        "distance_to_mask_boundary": metadata.get("distance_to_mask_boundary"),
    }


def _annotate_mask_boundary_distances(
    source: RasterSource, datasets: Sequence[BagDataset]
) -> Dict[str, Any]:
    """Attach anchor-to-mask-boundary distances without loading raster values into JAX."""
    valid = source.valid_mask()
    y_resolution = abs(float(source.resolution[1]))
    x_resolution = abs(float(source.resolution[0]))
    padded = np.pad(valid, 1, mode="constant", constant_values=False)
    center_distance = distance_transform_edt(padded, sampling=(y_resolution, x_resolution))[
        1:-1, 1:-1
    ]
    boundary_distance = np.maximum(center_distance - 0.5 * min(x_resolution, y_resolution), 0.0)
    annotated = 0
    for dataset in datasets:
        for bag in dataset.collections:
            metadata = dict(bag.metadata or {})
            anchor = metadata.get("anchor_cell")
            if anchor is None:
                continue
            row, column = map(int, anchor)
            if 0 <= row < source.height and 0 <= column < source.width:
                metadata["distance_to_mask_boundary"] = float(boundary_distance[row, column])
                metadata["anchor_is_all_band_valid"] = bool(valid[row, column])
                bag.metadata = metadata
                annotated += 1
    return {
        "definition": "anchor-center distance to nearest all-band-invalid cell or raster edge",
        "units": "CRS units",
        "n_annotated_bags": annotated,
        "valid_cell_fraction": float(np.mean(valid)),
    }


def full_window_only_dataset(
    dataset: BagDataset, window_size: int, seed: int
) -> Tuple[BagDataset, Dict[str, Any]]:
    """Retain balanced classes whose focal windows contain every expected cell."""
    expected = int(window_size) ** 2
    sites = [bag for bag in dataset.collections if bag.label == 1 and bag.n_samples == expected]
    backgrounds = [
        bag for bag in dataset.collections if bag.label == 0 and bag.n_samples == expected
    ]
    retained = min(len(sites), len(backgrounds))
    if retained < 2:
        raise RuntimeError("Too few complete windows remain for a balanced sensitivity")
    rng = np.random.default_rng(seed)
    selected_sites = [
        sites[index] for index in sorted(rng.choice(len(sites), retained, replace=False))
    ]
    selected_backgrounds = [
        backgrounds[index]
        for index in sorted(rng.choice(len(backgrounds), retained, replace=False))
    ]
    complete = BagDataset(
        [*selected_sites, *selected_backgrounds],
        list(dataset.feature_names),
        crs=dataset.crs,
        study_design=dataset.study_design,
        metadata={"design": "full_window_only", "window_size": window_size},
    )
    return complete, {
        "design": "full_window_only",
        "window_size": window_size,
        "required_cells": expected,
        "n_site_bags": retained,
        "n_background_bags": retained,
        "excluded_site_ids": sorted(
            bag.id for bag in dataset.collections if bag.label == 1 and bag not in selected_sites
        ),
        "excluded_background_ids": sorted(
            bag.id
            for bag in dataset.collections
            if bag.label == 0 and bag not in selected_backgrounds
        ),
    }


def matched_cell_count_dataset(dataset: BagDataset, seed: int) -> Tuple[BagDataset, Dict[str, Any]]:
    """Match the background cell-count multiset exactly to retained site bags.

    Background focal windows must contain at least the paired site count. Extra
    valid cells are deterministically subsampled. This removes cell count as a
    class shortcut while leaving the separate complete-window sensitivity to
    test raster-mask boundary effects.
    """
    rng = np.random.default_rng(seed)
    sites = [bag for bag in dataset.collections if bag.label == 1]
    available = [bag for bag in dataset.collections if bag.label == 0]
    matches: Dict[str, Bag] = {}
    excluded_sites = []
    for site in sorted(sites, key=lambda bag: (-bag.n_samples, bag.id)):
        eligible = [bag for bag in available if bag.n_samples >= site.n_samples]
        if not eligible:
            excluded_sites.append(site.id)
            continue
        background = min(eligible, key=lambda bag: (bag.n_samples, bag.id))
        available.remove(background)
        selected = np.arange(background.n_samples)
        if background.n_samples > site.n_samples:
            selected = np.sort(rng.choice(background.n_samples, site.n_samples, replace=False))
        metadata = dict(background.metadata or {})
        metadata.update(
            {
                "cell_count_matched": True,
                "matched_to_site_id": site.id,
                "pre_match_valid_cell_count": background.n_samples,
            }
        )
        if "cell_indices" in metadata:
            metadata["cell_indices"] = np.asarray(metadata["cell_indices"])[selected].tolist()
        matches[site.id] = Bag(
            background.samples[selected],
            0,
            background.id,
            metadata=metadata,
            coordinates=(
                background.coordinates[selected] if background.coordinates is not None else None
            ),
            group_id=background.group_id,
            stratum_id=background.stratum_id,
        )
    retained_sites = [bag for bag in sites if bag.id in matches]
    if len(retained_sites) < 2:
        raise RuntimeError("Too few sites can be matched to background focal windows")
    matched_backgrounds = [matches[bag.id] for bag in retained_sites]
    matched = BagDataset(
        [*retained_sites, *matched_backgrounds],
        list(dataset.feature_names),
        crs=dataset.crs,
        study_design=dataset.study_design,
        metadata={"design": "matched_valid_cell_counts"},
    )
    site_counts = sorted(bag.n_samples for bag in retained_sites)
    background_counts = sorted(bag.n_samples for bag in matched_backgrounds)
    if site_counts != background_counts:
        raise RuntimeError("Matched background cell counts do not reproduce the site distribution")
    return matched, {
        "design": "matched_valid_cell_counts",
        "n_site_bags": len(retained_sites),
        "n_background_bags": len(matched_backgrounds),
        "excluded_site_ids": sorted(excluded_sites),
        "unused_background_ids": sorted(bag.id for bag in available),
        "site_cell_sizes": site_counts,
        "background_cell_sizes": background_counts,
    }


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
                        **_bag_geometry_fields(bag),
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
                        **_bag_geometry_fields(bag),
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
    disagreement = _availability_disagreement_results(fold_results, plan, configuration)
    metrics = (
        "capture_5_percent",
        "capture_10_percent",
        "capture_20_percent",
        "lift_5_percent",
        "lift_10_percent",
        "lift_20_percent",
        "capture_surplus_5_percent",
        "capture_surplus_10_percent",
        "capture_surplus_20_percent",
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
        "availability_disagreement_vs_M0": disagreement,
    }


def _spatial_seed(configuration: Mapping[str, Any], method: str, repeat: int) -> int:
    return int(configuration["seed"]) + repeat * 1009 + sum(ord(character) for character in method)


def _spatial_diagnostics(
    values: np.ndarray,
    coordinates: np.ndarray,
    identifiers: Sequence[str],
    configuration: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    neighbors = min(int(configuration.get("spatial_neighbors", 8)), len(values) - 1)
    return spatial_autocorrelation_diagnostics(
        values,
        coordinates,
        identifiers=identifiers,
        n_neighbors=neighbors,
        permutations=int(configuration.get("spatial_permutations", 999)),
        seed=seed,
        alpha=float(configuration.get("spatial_fdr_alpha", 0.05)),
    )


def _availability_disagreement_results(
    fold_results: Sequence[Mapping[str, Any]],
    plan: FoldPlan,
    configuration: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Pool fold-specific availability ranks and diagnose spatial disagreement with M0."""
    methods = sorted({str(row["method"]) for row in fold_results})
    if "M0" not in methods:
        return []
    output = []
    for repeat in range(1, plan.n_repeats + 1):
        pooled: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {}
        for method in methods:
            by_anchor: Dict[str, List[Mapping[str, Any]]] = {}
            rows = [
                row
                for row in fold_results
                if row["method"] == method and int(row["repeat"]) == repeat
            ]
            for row in rows:
                for prediction in cast(
                    Sequence[Mapping[str, Any]], row["availability_predictions"]
                ):
                    by_anchor.setdefault(str(prediction["bag_id"]), []).append(prediction)
            pooled[method] = by_anchor
        reference_ids = sorted(pooled["M0"])
        reference = np.asarray(
            [
                np.mean([float(row["availability_percentile"]) for row in pooled["M0"][identifier]])
                for identifier in reference_ids
            ]
        )
        coordinates = np.asarray(
            [
                [pooled["M0"][identifier][0]["x"], pooled["M0"][identifier][0]["y"]]
                for identifier in reference_ids
            ],
            dtype=float,
        )
        for method in methods:
            if method == "M0":
                continue
            if sorted(pooled[method]) != reference_ids:
                raise RuntimeError(f"{method} does not share M0 availability anchors")
            candidate = np.asarray(
                [
                    np.mean(
                        [
                            float(row["availability_percentile"])
                            for row in pooled[method][identifier]
                        ]
                    )
                    for identifier in reference_ids
                ]
            )
            diagnostic = _spatial_diagnostics(
                candidate - reference,
                coordinates,
                reference_ids,
                configuration,
                _spatial_seed(configuration, method, repeat),
            )
            output.append({"method": method, "baseline": "M0", "repeat": repeat, **diagnostic})
    return output


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
            site_predictions = [row for row in predictions if int(row["label"]) == 1]
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
            site_shortfall = 1.0 - sites
            site_coordinates = np.asarray(
                [[row["x"], row["y"]] for row in site_predictions], dtype=float
            )
            site_ids = [str(row["bag_id"]) for row in site_predictions]
            spatial_failure = _spatial_diagnostics(
                site_shortfall,
                site_coordinates,
                site_ids,
                configuration,
                _spatial_seed(configuration, method, repeat),
            )
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
                "site_shortfall_spatial_diagnostic": spatial_failure,
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
                output[f"capture_surplus_{percentage}_percent"] = capture["capture_surplus"]
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
    primary = int(configuration["primary_window"])
    primary_focal = focal[primary]
    primary_availability = availability[primary]
    mask_audit = _annotate_mask_boundary_distances(
        source,
        [*focal.values(), *availability.values()],
    )

    complete, complete_audit = full_window_only_dataset(
        primary_focal, primary, int(configuration["seed"])
    )
    complete_availability_bags = [
        bag for bag in primary_availability.collections if bag.n_samples == primary**2
    ]
    if len(complete_availability_bags) < 2:
        raise RuntimeError("Too few complete availability windows remain")
    complete_availability = BagDataset(
        complete_availability_bags,
        list(primary_availability.feature_names),
        crs=primary_availability.crs,
        study_design=primary_availability.study_design,
        metadata={"design": "full_window_only", "window_size": primary},
    )
    complete_plan, complete_groups = _valid_spatial_plan(
        complete,
        int(configuration["n_splits"]),
        int(configuration["n_repeats"]),
        int(configuration["seed"]),
        block_width,
    )

    matched, matched_audit = matched_cell_count_dataset(primary_focal, int(configuration["seed"]))
    matched_plan, matched_groups = _valid_spatial_plan(
        matched,
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
        "mask_audit": mask_audit,
        "geometry_designs": {
            f"focal_{primary}_full_window": {
                "dataset": complete,
                "availability": complete_availability,
                "plan": complete_plan,
                "groups": complete_groups,
                "audit": {
                    **complete_audit,
                    "n_availability_bags": complete_availability.n_locations,
                },
                "support_note": "all classes and availability require every focal-window cell",
            },
            f"focal_{primary}_matched_counts": {
                "dataset": matched,
                "availability": primary_availability,
                "plan": matched_plan,
                "groups": matched_groups,
                "audit": matched_audit,
                "support_note": "background cell-count multiset exactly matches retained sites",
            },
        },
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
    if mode in ("geometry", "full"):
        designs.extend([f"focal_{primary}_full_window", f"focal_{primary}_matched_counts"])
    if mode == "full":
        designs.extend(f"focal_{window}" for window in windows if window != primary)
        designs.append("original_irregular")
    result: Dict[str, Any] = {
        "data_audit": {
            "original": prepared["original_audit"],
            "common_focal": prepared["focal_audit"],
            "availability": prepared["availability_audit"],
            "mask_boundary": prepared["mask_audit"],
            "geometry_sensitivities": {
                design: payload["audit"]
                for design, payload in cast(
                    Mapping[str, Mapping[str, Any]], prepared["geometry_designs"]
                ).items()
            },
            "shared_block_width": prepared["shared_block_width"],
        },
        "designs": {},
    }
    for design in designs:
        geometry_designs = cast(Mapping[str, Mapping[str, Any]], prepared["geometry_designs"])
        if design in geometry_designs:
            geometry = geometry_designs[design]
            dataset = cast(BagDataset, geometry["dataset"])
            plan = cast(FoldPlan, geometry["plan"])
            groups = cast(Sequence[str], geometry["groups"])
            availability = cast(BagDataset, geometry["availability"])
            support_note = str(geometry["support_note"])
        elif design == "original_irregular":
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
                    **_bag_geometry_fields(bag),
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
    parser.add_argument(
        "--mode", choices=("prepare", "primary", "geometry", "full"), default="geometry"
    )
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
                "mapped availability capture, lift, and capture surplus at 5%, 10%, and 20%",
                "continuous Boyce index",
                "held-out site availability percentiles",
            ],
            "secondary": ["ROC AUC", "PR AUC", "Kvamme Gain (redundant with lift)"],
            "diagnostic": [
                "geometry-only control",
                "full-window and matched-valid-cell-count sensitivities",
                "global and local Moran diagnostics of site shortfall and model disagreement",
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
