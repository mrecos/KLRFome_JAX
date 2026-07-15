"""Evaluation and method-comparison diagnostics for presence-background studies."""

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr, t
from sklearn.metrics import average_precision_score, roc_auc_score


def boyce_index(labels: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> Optional[float]:
    """Return a continuous-style Boyce rank index, or ``None`` when undefined."""
    labels, scores = _validated_scores(labels, scores)
    quantiles = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(quantiles) < 4:
        return None
    bins = np.digitize(scores, quantiles[1:-1], right=True)
    ratios = []
    centers = []
    presence_total = int((labels == 1).sum())
    background_total = int((labels == 0).sum())
    if presence_total == 0 or background_total == 0:
        return None
    for bin_index in range(len(quantiles) - 1):
        member = bins == bin_index
        if not member.any():
            continue
        expected = (labels[member] == 0).sum() / background_total
        if expected <= 0:
            continue
        ratios.append(((labels[member] == 1).sum() / presence_total) / expected)
        centers.append(float(np.mean(scores[member])))
    if len(ratios) < 3 or np.unique(ratios).size < 2:
        return None
    value = float(spearmanr(centers, ratios).statistic)
    return value if np.isfinite(value) else None


def presence_background_metrics(
    labels: np.ndarray, scores: np.ndarray, top_fraction: float = 0.05
) -> Dict[str, Optional[float]]:
    """Compute ranking metrics without interpreting scores as occurrence probability."""
    labels, scores = _validated_scores(labels, scores)
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    has_both_classes = np.unique(labels).size == 2
    auc = float(roc_auc_score(labels, scores)) if has_both_classes else None
    pr_auc = float(average_precision_score(labels, scores)) if has_both_classes else None
    threshold = float(np.quantile(scores, 1.0 - top_fraction))
    selected = scores >= threshold
    prevalence = float(labels.mean())
    lift = (
        float(labels[selected].mean() / prevalence) if selected.any() and prevalence > 0 else None
    )
    score_separation = (
        float(scores[labels == 1].mean() - scores[labels == 0].mean()) if has_both_classes else None
    )
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "boyce": boyce_index(labels, scores),
        "top_5_percent_lift": lift,
        "score_separation": score_separation,
    }


def paired_method_differences(
    rows: Sequence[Mapping[str, Any]],
    baseline: str = "M0",
    metrics: Iterable[str] = ("auc", "pr_auc", "boyce", "top_5_percent_lift"),
    pairing_keys: Sequence[str] = ("repeat", "fold"),
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Summarize paired differences against a baseline using explicit pairing keys."""
    if not pairing_keys:
        raise ValueError("pairing_keys must contain at least one field")
    lookup = {(str(row["method"]), *(int(row[key]) for key in pairing_keys)): row for row in rows}
    methods = sorted({str(row["method"]) for row in rows if row["method"] != baseline})
    output: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for method in methods:
        output[method] = {}
        for metric in metrics:
            differences = []
            for key, row in lookup.items():
                if key[0] != method:
                    continue
                reference = lookup.get((baseline, *key[1:]))
                if reference is None:
                    continue
                value = row.get(metric)
                baseline_value = reference.get(metric)
                if value is None or baseline_value is None:
                    continue
                difference = float(value) - float(baseline_value)
                if np.isfinite(difference):
                    differences.append(difference)
            values: np.ndarray = np.asarray(differences, dtype=float)
            if len(values) == 0:
                output[method][metric] = {
                    "n_pairs": 0,
                    "mean_difference": None,
                    "standard_error": None,
                    "ci_95": None,
                }
                continue
            mean = float(values.mean())
            standard_error = (
                float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
            )
            output[method][metric] = {
                "n_pairs": len(values),
                "mean_difference": mean,
                "standard_error": standard_error,
                "ci_95": (
                    [mean - 1.96 * standard_error, mean + 1.96 * standard_error]
                    if standard_error is not None
                    else None
                ),
            }
    return output


def replicate_summary(values: Sequence[float], confidence: float = 0.95) -> Dict[str, Any]:
    """Summarize independent replicate values with a small-sample t interval.

    Cross-validation folds are not independent replicates. Callers should first
    aggregate folds or pooled out-of-fold predictions within each generated
    dataset, then pass the resulting case-level values here.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    array: np.ndarray = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {
            "n": 0,
            "values": [],
            "mean": None,
            "standard_deviation": None,
            "standard_error": None,
            "confidence": confidence,
            "confidence_interval": None,
        }
    mean = float(array.mean())
    if len(array) == 1:
        return {
            "n": 1,
            "values": array.tolist(),
            "mean": mean,
            "standard_deviation": None,
            "standard_error": None,
            "confidence": confidence,
            "confidence_interval": None,
        }
    standard_deviation = float(array.std(ddof=1))
    standard_error = standard_deviation / float(np.sqrt(len(array)))
    critical_value = float(t.ppf((1.0 + confidence) / 2.0, df=len(array) - 1))
    return {
        "n": len(array),
        "values": array.tolist(),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "confidence": confidence,
        "confidence_interval": [
            mean - critical_value * standard_error,
            mean + critical_value * standard_error,
        ],
    }


def score_agreement(reference: np.ndarray, approximation: np.ndarray) -> Dict[str, float]:
    """Compare two score vectors without imposing a classification threshold."""
    reference = np.asarray(reference, dtype=float)
    approximation = np.asarray(approximation, dtype=float)
    if reference.shape != approximation.shape or reference.ndim != 1:
        raise ValueError("score vectors must be one-dimensional with identical shape")
    if not np.isfinite(reference).all() or not np.isfinite(approximation).all():
        raise ValueError("score vectors must be finite")
    pearson = _safe_correlation(reference, approximation, rank=False)
    rank = _safe_correlation(reference, approximation, rank=True)
    count = max(1, int(np.ceil(0.05 * len(reference))))
    reference_top = set(np.argsort(reference)[-count:])
    approximation_top = set(np.argsort(approximation)[-count:])
    return {
        "mae": float(np.mean(np.abs(reference - approximation))),
        "pearson": pearson,
        "spearman": rank,
        "top_5_percent_overlap": len(reference_top & approximation_top) / count,
    }


def kernel_approximation_diagnostics(
    exact: np.ndarray, approximation: np.ndarray
) -> Dict[str, float]:
    """Measure RFF Gram approximation fidelity against an exact reference."""
    exact = np.asarray(exact, dtype=float)
    approximation = np.asarray(approximation, dtype=float)
    if exact.shape != approximation.shape or exact.ndim != 2 or exact.shape[0] != exact.shape[1]:
        raise ValueError("kernel matrices must be square with identical shape")
    difference = exact - approximation
    denominator: float = max(float(np.linalg.norm(exact)), float(np.finfo(np.float64).eps))
    upper = np.triu_indices(exact.shape[0], k=1)
    correlation = _safe_correlation(exact[upper], approximation[upper], rank=False)
    return {
        "relative_frobenius_error": float(np.linalg.norm(difference) / denominator),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "upper_triangle_correlation": correlation,
    }


def _validated_scores(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("labels and scores must be one-dimensional with identical shape")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return labels, scores


def _safe_correlation(left: np.ndarray, right: np.ndarray, rank: bool) -> float:
    if len(left) < 2:
        return 1.0 if np.allclose(left, right) else 0.0
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return 1.0 if np.allclose(left, right) else 0.0
    value = (
        float(spearmanr(left, right).statistic) if rank else float(np.corrcoef(left, right)[0, 1])
    )
    return value if np.isfinite(value) else 0.0
