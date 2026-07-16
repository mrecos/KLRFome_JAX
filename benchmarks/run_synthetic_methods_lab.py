#!/usr/bin/env python3
"""Run deterministic synthetic M0--M4 distribution-regression experiments."""

import argparse
import json
import time
import tracemalloc
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from klrfome.data.synthetic import (
    SyntheticScenarioConfig,
    duplicate_all_cells,
    duplicate_selected_cells,
    generate_reference_bags,
    generate_synthetic_bags,
    permute_bag_cells,
)
from klrfome.kernels.rff import clear_rff_frequency_cache, rff_frequency_cache_info
from klrfome.models.baselines import baseline_models
from klrfome.models.distribution import DistributionClassifier
from klrfome.models.spec import ModelSpec
from klrfome.utils.evaluation import (
    kernel_approximation_diagnostics,
    paired_method_differences,
    presence_background_metrics,
    score_agreement,
)
from klrfome.utils.reproducibility import (
    configuration_fingerprint,
    dataset_fingerprint,
    environment_manifest,
    serialize_fold_plan,
    write_strict_json,
)
from klrfome.utils.validation import make_fold_plan


def expand_cases(configuration: Mapping[str, Any]) -> List[SyntheticScenarioConfig]:
    """Expand explicit scenario axes into deterministic generator configurations."""
    cases = []
    master_seed = int(configuration["seed"])
    default_bags = int(configuration.get("n_bags_per_class", 24))
    default_features = int(configuration.get("n_features", 3))
    default_bag_size = int(configuration.get("bag_size", 30))
    for scenario_index, scenario in enumerate(configuration["scenarios"]):
        effects = scenario.get("effect_sizes", [scenario.get("effect_size", 0.75)])
        bag_sizes = scenario.get("bag_sizes", [scenario.get("bag_size", default_bag_size)])
        spatial_ranges = scenario.get("spatial_ranges", [scenario.get("spatial_range", 0.0)])
        replicates = int(scenario.get("replicates", configuration.get("replicates", 1)))
        combination_index = 0
        for effect in effects:
            for bag_size in bag_sizes:
                for spatial_range in spatial_ranges:
                    for replicate in range(replicates):
                        seed = int(
                            np.random.SeedSequence(
                                [master_seed, scenario_index, combination_index, replicate]
                            ).generate_state(1)[0]
                        )
                        cases.append(
                            SyntheticScenarioConfig(
                                scenario=scenario["scenario"],
                                n_bags_per_class=int(
                                    scenario.get("n_bags_per_class", default_bags)
                                ),
                                n_features=int(scenario.get("n_features", default_features)),
                                n_signal_features=int(scenario.get("n_signal_features", 1)),
                                bag_size=int(bag_size),
                                unequal_bag_sizes=bool(scenario.get("unequal_bag_sizes", False)),
                                min_bag_size=int(scenario.get("min_bag_size", 3)),
                                max_bag_size=int(scenario.get("max_bag_size", 120)),
                                effect_size=float(effect),
                                spatial_range=float(spatial_range),
                                bags_per_group=int(scenario.get("bags_per_group", 1)),
                                seed=seed,
                            )
                        )
                    combination_index += 1
    return cases


def method_specifications(configuration: Mapping[str, Any]) -> Dict[str, ModelSpec]:
    """Build named model variants from configuration."""
    output = {}
    for item in configuration["methods"]:
        method = str(item["method"])
        identifier = str(item.get("id", method))
        if identifier in output:
            raise ValueError(f"Duplicate method id: {identifier}")
        rff_features = int(item.get("rff_features", 128))
        rff_scheme = str(item.get("rff_scheme", "iid"))
        embedding_estimator = str(item.get("embedding_estimator", "empirical"))
        shrinkage_effective_size = str(item.get("shrinkage_effective_size", "nominal"))
        configured_spatial_range = item.get("shrinkage_spatial_range")
        shrinkage_spatial_range = (
            float(configured_spatial_range) if configured_spatial_range is not None else None
        )
        if method == "M0":
            spec = ModelSpec.m0()
        elif method == "M1":
            spec = ModelSpec.m1(
                rff_features,
                rff_scheme=rff_scheme,
                embedding_estimator=embedding_estimator,
                shrinkage_effective_size=shrinkage_effective_size,
                shrinkage_spatial_range=shrinkage_spatial_range,
            )
        elif method == "M2":
            spec = ModelSpec.m2(
                rff_features,
                rff_scheme=rff_scheme,
                embedding_estimator=embedding_estimator,
                shrinkage_effective_size=shrinkage_effective_size,
                shrinkage_spatial_range=shrinkage_spatial_range,
            )
        elif method == "M3":
            spec = ModelSpec.m3(
                int(item.get("n_projections", 64)), int(item.get("n_quantiles", 64))
            )
        elif method == "M4":
            weights = item.get("hybrid_weights")
            initial_weight = float(weights[0]) if weights else float(item.get("hybrid_weight", 0.5))
            spec = ModelSpec.m4(
                hybrid_weight=initial_weight,
                hybrid_mean_representation=str(item.get("hybrid_mean_representation", "rff_kme")),
                rff_features=rff_features,
                rff_scheme=rff_scheme,
                embedding_estimator=embedding_estimator,
                shrinkage_effective_size=shrinkage_effective_size,
                shrinkage_spatial_range=shrinkage_spatial_range,
                n_projections=int(item.get("n_projections", 64)),
                n_quantiles=int(item.get("n_quantiles", 64)),
                hybrid_normalize=bool(item.get("hybrid_normalize", True)),
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
        output[identifier] = spec
    return output


def run_case(
    case: SyntheticScenarioConfig, configuration: Mapping[str, Any], case_index: int
) -> Dict[str, Any]:
    """Run all configured methods on one generated dataset and shared folds."""
    dataset = generate_synthetic_bags(case)
    reference_bag_size = int(configuration.get("reference_bag_size", 0))
    reference_dataset = (
        generate_reference_bags(case, reference_bag_size) if reference_bag_size > 0 else None
    )
    groups = [bag.group_id or bag.id for bag in dataset.collections]
    plan = make_fold_plan(
        dataset,
        n_splits=int(configuration["n_splits"]),
        n_repeats=int(configuration.get("n_repeats", 1)),
        seed=case.seed,
        stratified=True,
        group_ids=groups,
    )
    specifications = method_specifications(configuration)
    model_seed = int(configuration.get("model_seed", case.seed))
    method_items = {str(item.get("id", item["method"])): item for item in configuration["methods"]}
    rows: List[Dict[str, Any]] = []
    oof_scores: Dict[Tuple[str, int], Dict[int, float]] = {}
    method_architectures = {
        method_id: specification.method_id for method_id, specification in specifications.items()
    }
    for assignment in plan.assignments:
        train = dataset.subset(assignment.train_indices)
        test = dataset.subset(assignment.test_indices)
        reference_test = (
            reference_dataset.subset(assignment.test_indices)
            if reference_dataset is not None
            else None
        )
        fold_models: Dict[str, DistributionClassifier] = {}
        test_scores: Dict[str, np.ndarray] = {}
        for method_id, spec in specifications.items():
            fitted_spec = _resolve_fold_specification(
                spec,
                method_items[method_id],
                train,
                configuration,
                case.seed + assignment.repeat * 1009 + assignment.fold,
                model_seed,
            )
            model = DistributionClassifier(
                fitted_spec,
                lambda_reg=float(configuration.get("lambda_reg", 0.1)),
                seed=model_seed,
                round_exact_kernel=False,
            )
            tracemalloc.start()
            fit_started = time.perf_counter()
            model.fit(train)
            fit_seconds = time.perf_counter() - fit_started
            predict_started = time.perf_counter()
            scores = np.asarray(model.predict_bags(test), dtype=float)
            predict_seconds = time.perf_counter() - predict_started
            train_scores = np.asarray(model.predict_bags(train), dtype=float)
            embedding_mse = (
                _reference_embedding_mse(model, test, reference_test)
                if reference_test is not None and model._rff is not None
                else None
            )
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            test_metrics = presence_background_metrics(np.asarray(test.labels), scores)
            train_metrics = presence_background_metrics(np.asarray(train.labels), train_scores)
            row = {
                "method": method_id,
                "architecture": spec.method_id,
                "repeat": assignment.repeat + 1,
                "fold": assignment.fold + 1,
                "n_train": train.n_locations,
                "n_test": test.n_locations,
                **test_metrics,
                "train_auc": train_metrics["auc"],
                "auc_generalization_gap": _difference(train_metrics["auc"], test_metrics["auc"]),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "peak_python_memory_mb": peak_bytes / (1024**2),
                "embedding_mse": embedding_mse,
                "diagnostics": dict(model.diagnostics_),
            }
            rows.append(row)
            _record_oof_predictions(
                oof_scores,
                method_id,
                assignment.repeat + 1,
                assignment.test_indices,
                scores,
            )
            fold_models[method_id] = model
            test_scores[method_id] = scores

        _attach_m1_diagnostics(
            rows, assignment.repeat + 1, assignment.fold + 1, fold_models, test_scores
        )
        if bool(configuration.get("include_baselines", True)):
            for method_id, baseline in baseline_models(
                seed=model_seed,
                rf_estimators=int(configuration.get("rf_estimators", 200)),
                include_mean_std=bool(configuration.get("include_mean_std_baseline", True)),
            ).items():
                method_architectures[method_id] = "bag_summary_baseline"
                tracemalloc.start()
                fit_started = time.perf_counter()
                baseline.fit(train)
                fit_seconds = time.perf_counter() - fit_started
                predict_started = time.perf_counter()
                scores = baseline.predict_bags(test)
                predict_seconds = time.perf_counter() - predict_started
                train_scores = baseline.predict_bags(train)
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                test_metrics = presence_background_metrics(np.asarray(test.labels), scores)
                train_metrics = presence_background_metrics(np.asarray(train.labels), train_scores)
                rows.append(
                    {
                        "method": method_id,
                        "architecture": "bag_summary_baseline",
                        "repeat": assignment.repeat + 1,
                        "fold": assignment.fold + 1,
                        "n_train": train.n_locations,
                        "n_test": test.n_locations,
                        **test_metrics,
                        "train_auc": train_metrics["auc"],
                        "auc_generalization_gap": _difference(
                            train_metrics["auc"], test_metrics["auc"]
                        ),
                        "fit_seconds": fit_seconds,
                        "predict_seconds": predict_seconds,
                        "peak_python_memory_mb": peak_bytes / (1024**2),
                        "embedding_mse": None,
                        "diagnostics": {"summary": baseline.summary},
                    }
                )
                _record_oof_predictions(
                    oof_scores,
                    method_id,
                    assignment.repeat + 1,
                    assignment.test_indices,
                    scores,
                )

    case_id = f"case-{case_index:04d}-{case.scenario}"
    out_of_fold_results = _finalize_oof_results(
        dataset,
        oof_scores,
        method_architectures,
        int(configuration.get("n_repeats", 1)),
    )
    return {
        "case_id": case_id,
        "scenario": asdict(case),
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "fold_plan": serialize_fold_plan(plan, dataset, groups),
        "fold_results": rows,
        "paired_differences": paired_method_differences(rows, baseline="M0"),
        "out_of_fold_results": out_of_fold_results,
        "paired_oof_differences": paired_method_differences(
            out_of_fold_results,
            baseline="M0",
            pairing_keys=("repeat",),
        ),
        "invariance": (
            run_invariance_checks(dataset, specifications, configuration, model_seed)
            if bool(configuration.get("run_invariance_checks", False))
            else None
        ),
    }


def _resolve_fold_specification(
    specification: ModelSpec,
    method_item: Mapping[str, Any],
    training_data: Any,
    configuration: Mapping[str, Any],
    selection_seed: int,
    model_seed: int,
) -> ModelSpec:
    """Select an M4 mixture weight using only the current outer training fold."""
    weights = method_item.get("hybrid_weights")
    if specification.method_id != "M4" or not weights:
        return specification
    candidates = sorted({float(value) for value in weights})
    if not candidates:
        raise ValueError("hybrid_weights must contain at least one value")
    groups = [bag.group_id or bag.id for bag in training_data.collections]
    inner_splits = min(int(configuration.get("inner_splits", 3)), len(set(groups)))
    if inner_splits < 2:
        raise ValueError("M4 weight selection requires at least two inner groups")
    plan = make_fold_plan(
        training_data,
        n_splits=inner_splits,
        n_repeats=1,
        seed=selection_seed,
        stratified=True,
        group_ids=groups,
    )
    evaluations = []
    for weight in candidates:
        candidate = replace(specification, hybrid_weight=weight)
        oof_scores = np.full(training_data.n_locations, np.nan, dtype=float)
        for assignment in plan.assignments:
            inner_train = training_data.subset(assignment.train_indices)
            inner_test = training_data.subset(assignment.test_indices)
            model = DistributionClassifier(
                candidate,
                lambda_reg=float(configuration.get("lambda_reg", 0.1)),
                seed=model_seed,
                round_exact_kernel=False,
            ).fit(inner_train)
            oof_scores[np.asarray(assignment.test_indices, dtype=int)] = np.asarray(
                model.predict_bags(inner_test), dtype=float
            )
        if not np.isfinite(oof_scores).all():
            raise ValueError("Inner M4 selection did not predict every training bag")
        auc = presence_background_metrics(np.asarray(training_data.labels), oof_scores)["auc"]
        evaluations.append((float(auc) if auc is not None else -np.inf, weight))
    _, selected = min(evaluations, key=lambda item: (-item[0], abs(item[1] - 0.5), item[1]))
    return replace(specification, hybrid_weight=selected)


def _reference_embedding_mse(
    model: DistributionClassifier,
    observed: Any,
    reference: Any,
) -> float:
    """Compare finite-bag RFF embeddings with independent large reference bags."""
    observed_embeddings = np.asarray(model.embedding_vectors(observed), dtype=float)
    reference_embeddings = np.asarray(
        model.embedding_vectors(reference, apply_shrinkage=False), dtype=float
    )
    if observed_embeddings.shape != reference_embeddings.shape:
        raise ValueError("Observed and reference embeddings must align")
    squared_errors = np.sum((observed_embeddings - reference_embeddings) ** 2, axis=1)
    return float(np.mean(squared_errors))


def _record_oof_predictions(
    storage: Dict[Tuple[str, int], Dict[int, float]],
    method: str,
    repeat: int,
    test_indices: Sequence[int],
    scores: np.ndarray,
) -> None:
    """Record one method's held-out scores and reject fold overlap."""
    if len(test_indices) != len(scores):
        raise ValueError("Held-out indices and prediction scores must align")
    destination = storage.setdefault((method, repeat), {})
    for index, score in zip(test_indices, scores):
        position = int(index)
        if position in destination:
            raise ValueError(
                f"Bag index {position} appears more than once for {method} repeat {repeat}"
            )
        destination[position] = float(score)


def _finalize_oof_results(
    dataset: Any,
    storage: Mapping[Tuple[str, int], Mapping[int, float]],
    architectures: Mapping[str, str],
    n_repeats: int,
) -> List[Dict[str, Any]]:
    """Pool complete held-out predictions before computing ranking metrics."""
    expected_indices = set(range(dataset.n_locations))
    expected_keys = {
        (method, repeat) for method in architectures for repeat in range(1, n_repeats + 1)
    }
    if set(storage) != expected_keys:
        raise ValueError("Out-of-fold prediction coverage differs across methods or repeats")
    labels = np.asarray(dataset.labels, dtype=int)
    bag_ids = [bag.id for bag in dataset.collections]
    output = []
    for method, repeat in sorted(storage, key=lambda key: (key[1], key[0])):
        predictions = storage[(method, repeat)]
        if set(predictions) != expected_indices:
            raise ValueError(
                f"Out-of-fold predictions for {method} repeat {repeat} do not cover every bag"
            )
        scores = np.asarray(
            [predictions[index] for index in range(dataset.n_locations)],
            dtype=float,
        )
        threshold = float(np.quantile(scores, 0.95))
        output.append(
            {
                "method": method,
                "architecture": architectures[method],
                "repeat": repeat,
                "n_observations": dataset.n_locations,
                "top_5_percent_selected": int((scores >= threshold).sum()),
                "bag_ids": bag_ids,
                "labels": labels.tolist(),
                "scores": scores.tolist(),
                **presence_background_metrics(labels, scores),
            }
        )
    return output


def run_invariance_checks(
    dataset: Any,
    specifications: Mapping[str, ModelSpec],
    configuration: Mapping[str, Any],
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Compare predictions after order-preserving and mass-changing transformations."""
    transformed = {
        "permuted": permute_bag_cells(dataset, seed + 1),
        "uniformly_duplicated": duplicate_all_cells(dataset, repeats=2),
        "selectively_duplicated": duplicate_selected_cells(dataset, fraction=0.25, seed=seed + 2),
    }
    output = {}
    for method_id, spec in specifications.items():
        model = DistributionClassifier(
            spec,
            lambda_reg=float(configuration.get("lambda_reg", 0.1)),
            seed=seed,
            round_exact_kernel=False,
        ).fit(dataset)
        reference = np.asarray(model.predict_bags(dataset), dtype=float)
        output[method_id] = {}
        for name, candidate in transformed.items():
            scores = np.asarray(model.predict_bags(candidate), dtype=float)
            output[method_id][f"{name}_maximum_absolute_score_change"] = float(
                np.max(np.abs(scores - reference))
            )
    return output


def run_lab(
    configuration: Mapping[str, Any],
    repository: Path,
    case_indices: Optional[Sequence[int]] = None,
    progress: bool = False,
) -> Dict[str, Any]:
    """Run all expanded cases and return a strict-JSON-compatible result."""
    clear_rff_frequency_cache()
    cases = expand_cases(configuration)
    selected = list(range(len(cases))) if case_indices is None else list(case_indices)
    if any(index < 0 or index >= len(cases) for index in selected):
        raise ValueError("case_indices contains an out-of-range case")
    results = []
    for position, index in enumerate(selected, start=1):
        if progress:
            print(
                f"starting synthetic case {position}/{len(selected)}: "
                f"{cases[index].scenario} seed={cases[index].seed}",
                flush=True,
            )
        results.append(run_case(cases[index], configuration, index))
    return {
        "schema_version": "1.2",
        "configuration": dict(configuration),
        "configuration_sha256": configuration_fingerprint(configuration),
        "interpretation": "synthetic presence-background relative ranking",
        "environment": environment_manifest(repository),
        "rff_frequency_cache": rff_frequency_cache_info(),
        "cases": results,
    }


def _attach_m1_diagnostics(
    rows: List[Dict[str, Any]],
    repeat: int,
    fold: int,
    models: Mapping[str, DistributionClassifier],
    scores: Mapping[str, np.ndarray],
) -> None:
    m0_ids = [identifier for identifier, model in models.items() if model.spec.method_id == "M0"]
    if not m0_ids:
        return
    reference_id = m0_ids[0]
    reference = models[reference_id]
    if reference.gram_matrix_ is None:
        return
    for method_id, model in models.items():
        if model.spec.method_id != "M1" or model.training_embeddings_ is None:
            continue
        embeddings = np.asarray(model.training_embeddings_)
        approximation = embeddings @ embeddings.T
        diagnostics = {
            "kernel_approximation": kernel_approximation_diagnostics(
                np.asarray(reference.gram_matrix_), approximation
            ),
            "score_agreement_with_M0": score_agreement(scores[reference_id], scores[method_id]),
        }
        for row in rows:
            if row["method"] == method_id and row["repeat"] == repeat and row["fold"] == fold:
                row["diagnostics"].update(diagnostics)
                break


def _difference(left: Any, right: Any) -> Any:
    return None if left is None or right is None else float(left) - float(right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/synthetic_lab_config.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--case-index",
        type=int,
        action="append",
        default=None,
        help="Run only this zero-based expanded case index; may be repeated",
    )
    args = parser.parse_args()
    configuration_path = Path(args.config)
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    cases = expand_cases(configuration)
    if args.list_cases:
        for index, case in enumerate(cases):
            print(index, json.dumps(asdict(case), sort_keys=True))
        return
    repository = Path(__file__).resolve().parents[1]
    result = run_lab(configuration, repository, case_indices=args.case_index, progress=True)
    output = Path(args.output or configuration["output"])
    write_strict_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
