"""Tests for experiment metrics, fingerprints, and strict result payloads."""

import json
from pathlib import Path

import numpy as np
from jsonschema import validate

from benchmarks.run_synthetic_methods_lab import expand_cases, run_lab
from klrfome.data.synthetic import SyntheticScenarioConfig, generate_synthetic_bags
from klrfome.utils.evaluation import (
    kernel_approximation_diagnostics,
    presence_background_metrics,
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
        "lambda_reg": 0.1,
        "rf_estimators": 5,
        "include_baselines": False,
        "run_invariance_checks": False,
        "methods": [
            {"id": "M0", "method": "M0"},
            {"id": "M1", "method": "M1", "rff_features": 8},
            {"id": "M2", "method": "M2", "rff_features": 8},
            {"id": "M3", "method": "M3", "n_projections": 4, "n_quantiles": 6},
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
    output = tmp_path / "result.json"
    write_strict_json(output, {"finite": 1.0, "undefined": float("nan")})
    assert json.loads(output.read_text()) == {"finite": 1.0, "undefined": None}


def test_smoke_result_satisfies_tracked_schema(tmp_path):
    configuration = _smoke_configuration()
    result = run_lab(configuration, tmp_path)
    schema_path = Path(__file__).parents[1] / "benchmarks/synthetic_lab_result_schema.json"
    validate(
        instance=json.loads(json.dumps(result, default=str)),
        schema=json.loads(schema_path.read_text()),
    )
