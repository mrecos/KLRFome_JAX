"""Tests for experiment metrics, fingerprints, and strict result payloads."""

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import validate

from benchmarks.run_synthetic_methods_lab import (
    expand_cases,
    method_specifications,
    run_lab,
)
from klrfome.data.synthetic import (
    SyntheticScenarioConfig,
    generate_reference_bags,
    generate_synthetic_bags,
)
from klrfome.utils.evaluation import (
    kernel_approximation_diagnostics,
    paired_method_differences,
    presence_background_metrics,
    replicate_summary,
    score_agreement,
)
from klrfome.utils.reproducibility import (
    configuration_fingerprint,
    dataset_fingerprint,
    write_strict_json,
)


def _smoke_configuration():
    return {
        "schema_version": "1.0",
        "suite": "test",
        "seed": 5,
        "n_bags_per_class": 4,
        "n_features": 2,
        "bag_size": 6,
        "replicates": 1,
        "n_splits": 2,
        "n_repeats": 1,
        "inner_splits": 2,
        "reference_bag_size": 30,
        "lambda_reg": 0.1,
        "rf_estimators": 5,
        "include_baselines": False,
        "run_invariance_checks": False,
        "methods": [
            {"id": "M0", "method": "M0"},
            {"id": "M1", "method": "M1", "rff_features": 8},
            {
                "id": "M1-orf-shrink",
                "method": "M1",
                "rff_features": 8,
                "rff_scheme": "orthogonal",
                "embedding_estimator": "shrinkage",
            },
            {"id": "M2", "method": "M2", "rff_features": 8},
            {"id": "M3", "method": "M3", "n_projections": 4, "n_quantiles": 6},
            {
                "id": "M4",
                "method": "M4",
                "rff_features": 8,
                "n_projections": 4,
                "n_quantiles": 6,
                "hybrid_weights": [0.0, 1.0],
            },
        ],
        "scenarios": [{"scenario": "mean_shift", "effect_sizes": [0.8]}],
        "output": "unused.json",
    }


def test_metrics_and_approximation_diagnostics_are_well_defined():
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    metrics = presence_background_metrics(labels, scores)
    assert metrics["auc"] == 1.0
    agreement = score_agreement(scores, scores + 0.01)
    assert agreement["spearman"] == 1.0
    exact = np.asarray([[1.0, 0.5], [0.5, 1.0]])
    diagnostics = kernel_approximation_diagnostics(exact, exact * 0.99)
    assert diagnostics["relative_frobenius_error"] < 0.02
    replicate = replicate_summary([0.1, 0.2, 0.3])
    assert replicate["n"] == 3
    assert replicate["mean"] == pytest.approx(0.2)
    assert replicate["confidence_interval"][0] < 0.1
    assert replicate["confidence_interval"][1] > 0.3


def test_pairing_keys_support_pooled_repeat_comparisons():
    rows = [
        {"method": "M0", "repeat": 1, "auc": 0.6},
        {"method": "M1", "repeat": 1, "auc": 0.7},
        {"method": "M0", "repeat": 2, "auc": 0.5},
        {"method": "M1", "repeat": 2, "auc": 0.55},
    ]
    paired = paired_method_differences(
        rows,
        metrics=("auc",),
        pairing_keys=("repeat",),
    )
    assert paired["M1"]["auc"]["n_pairs"] == 2
    assert paired["M1"]["auc"]["mean_difference"] == pytest.approx(0.075)


def test_fingerprints_capture_ordered_scientific_content():
    dataset = generate_synthetic_bags(
        SyntheticScenarioConfig("mean_shift", n_bags_per_class=3, bag_size=5, seed=3)
    )
    fingerprint = dataset_fingerprint(dataset)
    reordered = dataset.subset(tuple(reversed(range(dataset.n_locations))))
    assert fingerprint != dataset_fingerprint(reordered)
    assert configuration_fingerprint({"a": 1, "b": [2, 3]}) == configuration_fingerprint(
        {"b": [2, 3], "a": 1}
    )


def test_case_expansion_and_runner_are_deterministic(tmp_path):
    configuration = _smoke_configuration()
    assert [case.seed for case in expand_cases(configuration)] == [
        case.seed for case in expand_cases(configuration)
    ]
    first = run_lab(configuration, tmp_path)
    second = run_lab(configuration, tmp_path)
    first_case = first["cases"][0]
    second_case = second["cases"][0]
    assert first_case["dataset_fingerprint"] == second_case["dataset_fingerprint"]
    assert first_case["fold_plan"] == second_case["fold_plan"]
    deterministic_fields = ["method", "repeat", "fold", "auc", "pr_auc", "boyce"]
    assert [
        {field: row[field] for field in deterministic_fields} for row in first_case["fold_results"]
    ] == [
        {field: row[field] for field in deterministic_fields} for row in second_case["fold_results"]
    ]
    assert first_case["out_of_fold_results"] == second_case["out_of_fold_results"]
    assert first_case["paired_oof_differences"] == second_case["paired_oof_differences"]
    for row in first_case["out_of_fold_results"]:
        assert row["n_observations"] == 8
        assert len(row["bag_ids"]) == len(row["labels"]) == len(row["scores"]) == 8
        assert row["top_5_percent_selected"] >= 1
    embedding_rows = [row for row in first_case["fold_results"] if row["embedding_mse"] is not None]
    assert embedding_rows
    assert all(row["embedding_mse"] >= 0 for row in embedding_rows)
    selected_weights = {
        row["diagnostics"]["hybrid_weight"]
        for row in first_case["fold_results"]
        if row["method"] == "M4"
    }
    assert selected_weights <= {0.0, 1.0}
    output = tmp_path / "result.json"
    write_strict_json(output, {"finite": 1.0, "undefined": float("nan")})
    assert json.loads(output.read_text()) == {"finite": 1.0, "undefined": None}


def test_targeted_v2_configuration_expands_to_documented_cases():
    configuration_path = (
        Path(__file__).parents[1] / "benchmarks/synthetic_lab_targeted_v2_config.json"
    )
    configuration = json.loads(configuration_path.read_text())
    cases = expand_cases(configuration)
    assert len(cases) == 68
    assert sum(case.scenario == "moment_matched_xor" for case in cases) == 12
    assert sum(case.unequal_bag_sizes for case in cases) == 5


def test_representation_extension_configuration_expands_to_documented_cases():
    configuration_path = (
        Path(__file__).parents[1] / "benchmarks/synthetic_lab_extensions_config.json"
    )
    configuration = json.loads(configuration_path.read_text())
    cases = expand_cases(configuration)
    assert len(cases) == 90
    assert sum(case.scenario == "null" for case in cases) == 20
    assert sum(case.scenario == "moment_matched_xor" for case in cases) == 12
    assert sum(case.unequal_bag_sizes for case in cases) == 5


def test_spatial_shrinkage_configuration_is_focused_and_uses_fixed_model_seed():
    configuration_path = (
        Path(__file__).parents[1] / "benchmarks/synthetic_lab_spatial_shrinkage_config.json"
    )
    configuration = json.loads(configuration_path.read_text())
    cases = expand_cases(configuration)
    specifications = method_specifications(configuration)
    assert len(cases) == 40
    assert configuration["model_seed"] == 42
    assert len(specifications) == 5
    spatial = specifications["M1-orf128-shrink-spatial"]
    assert spatial.shrinkage_effective_size == "spatial"
    assert spatial.shrinkage_spatial_range is None


def test_reference_bags_preserve_case_states_with_larger_independent_samples():
    configuration = SyntheticScenarioConfig(
        "moment_matched_xor",
        n_bags_per_class=4,
        n_features=2,
        bag_size=5,
        effect_size=0.9,
        spatial_range=3.0,
        seed=23,
    )
    observed = generate_synthetic_bags(configuration)
    reference = generate_reference_bags(configuration, reference_bag_size=80)
    assert [bag.id for bag in reference.collections] == [bag.id for bag in observed.collections]
    assert all(bag.n_samples == 80 for bag in reference.collections)
    assert reference.metadata["configuration"]["spatial_range"] == 0.0


def test_smoke_result_satisfies_tracked_schema(tmp_path):
    configuration = _smoke_configuration()
    result = run_lab(configuration, tmp_path)
    schema_path = Path(__file__).parents[1] / "benchmarks/synthetic_lab_result_schema.json"
    validate(
        instance=json.loads(json.dumps(result, default=str)),
        schema=json.loads(schema_path.read_text()),
    )
