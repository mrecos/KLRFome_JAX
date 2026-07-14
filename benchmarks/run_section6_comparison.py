#!/usr/bin/env python3
"""Deterministic, presence-background Section 6 M0--M3 comparison."""

import argparse
import hashlib
import json
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from klrfome.data.formats import Bag, BagDataset
from klrfome.io.raster_source import (
    RasterSource,
    align_bags_to_raster,
    build_spatial_background_bags,
)
from klrfome.io.tabular import TabularBagConfig, load_tabular_bags
from klrfome.models.distribution import DistributionClassifier
from klrfome.models.spec import ModelSpec
from klrfome.utils.validation import FoldPlan, make_fold_plan


def _read_site_rows(path: Path, feature_names: Sequence[str]) -> pd.DataFrame:
    columns = [*feature_names, "x", "y", "SITENO", "presence"]
    chunks = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=100_000):
        sites = chunk[chunk["presence"] == 1]
        if not sites.empty:
            chunks.append(sites)
    if not chunks:
        raise ValueError(f"No presence rows found in {path}")
    return pd.concat(chunks, ignore_index=True)


def _bag_diameter(bag: Bag) -> float:
    if bag.coordinates is None:
        raise ValueError(f"Bag {bag.id!r} lacks coordinates")
    coordinates = np.asarray(bag.coordinates)
    return float(np.hypot(np.ptp(coordinates[:, 0]), np.ptp(coordinates[:, 1])))


def _spatial_groups(dataset: BagDataset, block_width: float) -> List[str]:
    centroids = np.asarray(
        [np.asarray(bag.coordinates).mean(axis=0) for bag in dataset.collections]
    )
    origin = centroids.min(axis=0)
    cells = np.floor((centroids - origin) / block_width).astype(int)
    return [f"{row}:{col}" for row, col in cells]


def _valid_spatial_plan(
    dataset: BagDataset, requested_splits: int, repeats: int, seed: int, block_width: float
) -> Tuple[FoldPlan, List[str]]:
    groups = _spatial_groups(dataset, block_width)
    for n_splits in range(requested_splits, 1, -1):
        try:
            plan = make_fold_plan(
                dataset,
                n_splits=n_splits,
                n_repeats=repeats,
                seed=seed,
                stratified=True,
                group_ids=groups,
            )
        except ValueError:
            continue
        valid = True
        for assignment in plan.assignments:
            train_labels = np.asarray(dataset.labels)[list(assignment.train_indices)]
            test_labels = np.asarray(dataset.labels)[list(assignment.test_indices)]
            if np.unique(train_labels).size < 2 or np.unique(test_labels).size < 2:
                valid = False
                break
        if valid:
            return plan, groups
    raise RuntimeError("No spatial grouped fold count from 5 to 2 contains both classes")


def prepare_setting(name: str, setting: Dict, config: Dict, root: Path):
    csv_path = root / setting["csv"]
    raster_directory = root / setting["raster_directory"]
    header = pd.read_csv(csv_path, nrows=0)
    excluded = {"Unnamed: 0", "x", "y", "SITENO", "presence"}
    feature_names = [column for column in header.columns if column not in excluded]
    raster_paths = [str(raster_directory / f"{feature}.tif") for feature in feature_names]
    missing = [path for path in raster_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing rasters for {name}: {missing}")
    source = RasterSource(raster_paths, band_names=feature_names)
    site_rows = _read_site_rows(csv_path, feature_names)
    tabular = load_tabular_bags(
        site_rows,
        TabularBagConfig(
            feature_columns=feature_names,
            label_column="presence",
            id_column="SITENO",
            x_column="x",
            y_column="y",
            crs=source.crs,
            study_design="presence_background",
            min_unique_cells=config["min_cells"],
        ),
        labels=[1],
    )
    sites = align_bags_to_raster(
        tabular,
        source,
        min_cells=config["min_cells"],
        cap_cells=config["cell_cap"],
        seed=config["seed"],
    )
    for bag in sites.collections:
        bag.stratum_id = name

    backgrounds = build_spatial_background_bags(
        source,
        sites.collections,
        n_background=int(round(sites.n_sites * config["background_ratio"])),
        cap_cells=config["cell_cap"],
        seed=config["seed"],
        stratum_id=name,
    )
    dataset = BagDataset(
        [*sites.collections, *backgrounds],
        feature_names,
        crs=source.crs,
        study_design="presence_background",
        metadata={"setting": name},
    )
    diameters = np.asarray([_bag_diameter(bag) for bag in sites.collections])
    block_width = max(2.0 * float(np.percentile(diameters, 95)), max(source.resolution))
    plan, groups = _valid_spatial_plan(
        dataset,
        config["n_splits"],
        config["n_repeats"],
        config["seed"],
        block_width,
    )
    audit = {
        "relative_suitability_only": True,
        "n_site_bags": sites.n_sites,
        "n_background_bags": len(backgrounds),
        "site_cell_sizes": [bag.n_samples for bag in sites.collections],
        "background_cell_sizes": [bag.n_samples for bag in backgrounds],
        "tabular_exclusions": (tabular.metadata or {}).get("exclusions", []),
        "raster_alignment_exclusions": (sites.metadata or {}).get(
            "raster_alignment_exclusions", []
        ),
        "block_width": block_width,
        "site_diameter_95th_percentile": float(np.percentile(diameters, 95)),
        "feature_names": feature_names,
        "crs": source.crs,
    }
    return dataset, plan, groups, audit


def boyce_index(labels: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    quantiles = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(quantiles) < 4:
        return float("nan")
    bins = np.digitize(scores, quantiles[1:-1], right=True)
    ratios = []
    centers = []
    for bin_index in range(len(quantiles) - 1):
        member = bins == bin_index
        presence_total = max(int((labels == 1).sum()), 1)
        background_total = max(int((labels == 0).sum()), 1)
        expected = (labels[member] == 0).sum() / background_total
        if expected <= 0:
            continue
        ratios.append(((labels[member] == 1).sum() / presence_total) / expected)
        centers.append(float(np.mean(scores[member])))
    if len(ratios) < 3:
        return float("nan")
    return float(spearmanr(centers, ratios).statistic)


def fold_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    threshold = np.quantile(scores, 0.95)
    top = scores >= threshold
    baseline = labels.mean()
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "boyce": boyce_index(labels, scores),
        "top_5_percent_lift": float(labels[top].mean() / baseline),
    }


def model_specs(config: Dict) -> Dict[str, ModelSpec]:
    return {
        "M0": ModelSpec.m0(),
        "M1": ModelSpec.m1(config["rff_features"]),
        "M2": ModelSpec.m2(config["rff_features"]),
        "M3": ModelSpec.m3(config["wasserstein_projections"], config["wasserstein_quantiles"]),
    }


def run_comparison(dataset: BagDataset, plan: FoldPlan, config: Dict) -> List[Dict]:
    rows = []
    for method_id, spec in model_specs(config).items():
        for assignment in plan.assignments:
            print(
                f"starting {method_id} repeat={assignment.repeat + 1} "
                f"fold={assignment.fold + 1}",
                flush=True,
            )
            train = dataset.subset(assignment.train_indices)
            test = dataset.subset(assignment.test_indices)
            model = DistributionClassifier(
                spec,
                lambda_reg=config["lambda_reg"],
                seed=config["seed"],
                round_exact_kernel=True,
            )
            tracemalloc.start()
            fit_start = time.perf_counter()
            model.fit(train)
            fit_seconds = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            scores = np.asarray(model.predict_bags(test))
            predict_seconds = time.perf_counter() - predict_start
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            if not np.isfinite(scores).all():
                raise RuntimeError(
                    f"{method_id} produced nonfinite scores for repeat {assignment.repeat}, "
                    f"fold {assignment.fold}"
                )
            labels = np.asarray(test.labels, dtype=int)
            rows.append(
                {
                    "method": method_id,
                    "repeat": assignment.repeat + 1,
                    "fold": assignment.fold + 1,
                    "n_train": train.n_locations,
                    "n_test": test.n_locations,
                    **fold_metrics(labels, scores),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "peak_python_memory_mb": peak_bytes / (1024**2),
                    "diagnostics": model.diagnostics_,
                }
            )
            print(
                f"completed {method_id} repeat={assignment.repeat + 1} "
                f"fold={assignment.fold + 1} fit={fit_seconds:.3f}s "
                f"predict={predict_seconds:.3f}s",
                flush=True,
            )
    return rows


def paired_differences(rows: List[Dict]) -> Dict:
    metrics = ["auc", "pr_auc", "boyce", "top_5_percent_lift"]
    lookup = {(row["method"], row["repeat"], row["fold"]): row for row in rows}
    output = {}
    for method in ("M1", "M2", "M3"):
        output[method] = {}
        for metric in metrics:
            differences = []
            for row in rows:
                if row["method"] != method:
                    continue
                baseline = lookup[("M0", row["repeat"], row["fold"])][metric]
                difference = row[metric] - baseline
                if np.isfinite(difference):
                    differences.append(float(difference))
            values = np.asarray(differences)
            standard_error = (
                float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            )
            mean = float(values.mean())
            output[method][metric] = {
                "n_pairs": len(values),
                "mean_difference_vs_M0": mean,
                "standard_error": standard_error,
                "ci_95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
            }
    return output


def serialize_plan(plan: FoldPlan, dataset: BagDataset, groups: Sequence[str]) -> Dict:
    return {
        "n_splits": plan.n_splits,
        "n_repeats": plan.n_repeats,
        "seed": plan.seed,
        "assignments": [
            {
                "repeat": assignment.repeat + 1,
                "fold": assignment.fold + 1,
                "train_ids": [dataset.collections[i].id for i in assignment.train_indices],
                "test_ids": [dataset.collections[i].id for i in assignment.test_indices],
                "train_groups": sorted({groups[i] for i in assignment.train_indices}),
                "test_groups": sorted({groups[i] for i in assignment.test_indices}),
            }
            for assignment in plan.assignments
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/section6_comparison_config.json")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Validate data and folds without fitting"
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    root = Path(config["data_root"])
    result = {
        "schema_version": "1.0",
        "configuration": config,
        "configuration_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "interpretation": "presence-background relative suitability, not occurrence probability",
        "settings": {},
    }
    for name, setting in config["settings"].items():
        dataset, plan, groups, audit = prepare_setting(name, setting, config, root)
        print(
            f"prepared {name}: {audit['n_site_bags']} sites, "
            f"{audit['n_background_bags']} backgrounds, "
            f"{plan.n_splits} folds x {plan.n_repeats} repeats",
            flush=True,
        )
        rows = [] if args.prepare_only else run_comparison(dataset, plan, config)
        result["settings"][name] = {
            "data_audit": audit,
            "fold_plan": serialize_plan(plan, dataset, groups),
            "fold_results": rows,
            "paired_differences": {} if args.prepare_only else paired_differences(rows),
        }
    output = Path(config["output"])
    if args.prepare_only:
        output = output.with_name("prepared_data_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    print(output)


if __name__ == "__main__":
    main()
