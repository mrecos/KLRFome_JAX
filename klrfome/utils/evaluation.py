"""Evaluation and method-comparison diagnostics for presence-background studies."""

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, t
from sklearn.metrics import average_precision_score, roc_auc_score


def availability_percentile_ranks(
    target_scores: np.ndarray, availability_scores: np.ndarray
) -> np.ndarray:
    """Rank target scores against a fixed sample of mapped availability.

    The returned value is the midpoint empirical rank: the fraction strictly
    below plus half the fraction tied.  This makes scores from independently
    fitted cross-validation folds comparable without treating their raw
    logistic values as calibrated occurrence probabilities.
    """
    targets = _validated_finite_vector(target_scores, "target_scores")
    availability = _validated_finite_vector(availability_scores, "availability_scores")
    if len(availability) == 0:
        raise ValueError("availability_scores must be nonempty")
    ordered = np.sort(availability)
    lower = np.searchsorted(ordered, targets, side="left")
    upper = np.searchsorted(ordered, targets, side="right")
    ranks: np.ndarray = np.asarray((lower + upper) / (2.0 * len(ordered)), dtype=float)
    return ranks


def availability_capture_metrics(
    site_scores: np.ndarray,
    availability_scores: np.ndarray,
    area_fractions: Sequence[float] = (0.05, 0.10, 0.20),
) -> list[Dict[str, Optional[float]]]:
    """Report held-out site capture and area-efficiency at mapped-area budgets.

    Thresholds are defined only by the availability distribution.  ``capture``
    is the fraction of held-out sites above the threshold, ``lift`` is capture
    divided by mapped area, and ``capture_surplus`` is capture minus mapped
    area (the vertical distance above random allocation). ``gain`` retains
    Kvamme's gain ``1-area/capture`` for archaeological comparison; it is a
    monotone transform of lift and is therefore not independent evidence.
    """
    sites = _validated_finite_vector(site_scores, "site_scores")
    availability = _validated_finite_vector(availability_scores, "availability_scores")
    if len(sites) == 0 or len(availability) == 0:
        raise ValueError("site_scores and availability_scores must be nonempty")
    rows: list[Dict[str, Optional[float]]] = []
    for fraction in area_fractions:
        area = float(fraction)
        if not 0 < area <= 1:
            raise ValueError("area_fractions must be in (0, 1]")
        threshold = float(np.quantile(availability, 1.0 - area, method="lower"))
        achieved_area = float(np.mean(availability >= threshold))
        capture = float(np.mean(sites >= threshold))
        rows.append(
            {
                "area_fraction": area,
                "achieved_area_fraction": achieved_area,
                "threshold": threshold,
                "capture": capture,
                "lift": capture / achieved_area,
                "capture_surplus": capture - achieved_area,
                "gain": 1.0 - achieved_area / capture if capture > 0 else None,
            }
        )
    return rows


def spatial_autocorrelation_diagnostics(
    values: np.ndarray,
    coordinates: np.ndarray,
    identifiers: Optional[Sequence[str]] = None,
    n_neighbors: int = 8,
    permutations: int = 999,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute exploratory global and local Moran diagnostics on a kNN graph.

    The graph is the symmetric union of directed k-nearest-neighbor links and
    is row standardized. Permutation p-values use unconditional randomization;
    local values are adjusted with Benjamini-Hochberg before cluster flags are
    reported. These diagnostics describe residual or disagreement structure;
    they are not predictive-performance scores.
    """
    array = _validated_finite_vector(values, "values")
    points = np.asarray(coordinates, dtype=float)
    if points.shape != (len(array), 2) or not np.isfinite(points).all():
        raise ValueError("coordinates must be a finite array with shape (n, 2)")
    if len(array) < 3:
        raise ValueError("spatial diagnostics require at least three observations")
    if not 1 <= n_neighbors < len(array):
        raise ValueError("n_neighbors must be in [1, n_observations)")
    if permutations < 0:
        raise ValueError("permutations must be nonnegative")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    ids = (
        list(identifiers)
        if identifiers is not None
        else [str(index) for index in range(len(array))]
    )
    if len(ids) != len(array) or len(set(ids)) != len(ids):
        raise ValueError("identifiers must be unique and aligned with values")

    centered = array - float(np.mean(array))
    variance = float(np.mean(centered**2))
    if variance <= np.finfo(float).eps:
        return {
            "n_observations": len(array),
            "n_neighbors": n_neighbors,
            "permutations": permutations,
            "global_moran_i": None,
            "global_p_value": None,
            "local": [],
            "note": "undefined because values are constant",
        }

    tree = cKDTree(points)
    directed = np.asarray(tree.query(points, k=n_neighbors + 1)[1][:, 1:], dtype=int)
    rows: np.ndarray = np.repeat(np.arange(len(array)), n_neighbors)
    adjacency = csr_matrix(
        (np.ones(len(rows), dtype=float), (rows, directed.reshape(-1))),
        shape=(len(array), len(array)),
    )
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    row_sums = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    if np.any(row_sums == 0):
        raise RuntimeError("kNN symmetrization produced an observation without neighbors")
    weights = adjacency.multiply((1.0 / row_sums)[:, None]).tocsr()
    lag = weights @ centered
    global_i = float(np.sum(centered * lag) / np.sum(centered**2))
    local_i = centered * lag / variance

    global_p: Optional[float] = None
    local_raw: np.ndarray = np.ones(len(array), dtype=float)
    if permutations:
        rng = np.random.default_rng(seed)
        simulated_global: np.ndarray = np.empty(permutations, dtype=float)
        simulated_local: np.ndarray = np.empty((permutations, len(array)), dtype=float)
        for index in range(permutations):
            permuted = rng.permutation(centered)
            permuted_lag = weights @ permuted
            simulated_global[index] = np.sum(permuted * permuted_lag) / np.sum(permuted**2)
            simulated_local[index] = permuted * permuted_lag / variance
        expected_global = -1.0 / (len(array) - 1)
        global_p = float(
            (
                1
                + np.sum(
                    np.abs(simulated_global - expected_global) >= abs(global_i - expected_global)
                )
            )
            / (permutations + 1)
        )
        local_raw = (1 + np.sum(np.abs(simulated_local) >= np.abs(local_i)[None, :], axis=0)) / (
            permutations + 1
        )
    local_fdr = _benjamini_hochberg(local_raw)

    local_rows = []
    for index, identifier in enumerate(ids):
        if centered[index] >= 0 and lag[index] >= 0:
            cluster = "high-high"
        elif centered[index] < 0 and lag[index] < 0:
            cluster = "low-low"
        elif centered[index] >= 0:
            cluster = "high-low"
        else:
            cluster = "low-high"
        local_rows.append(
            {
                "id": identifier,
                "x": float(points[index, 0]),
                "y": float(points[index, 1]),
                "value": float(array[index]),
                "spatial_lag": float(lag[index]),
                "local_moran_i": float(local_i[index]),
                "cluster": cluster,
                "p_value": float(local_raw[index]) if permutations else None,
                "p_value_fdr": float(local_fdr[index]) if permutations else None,
                "significant_fdr": bool(local_fdr[index] <= alpha) if permutations else False,
            }
        )
    return {
        "n_observations": len(array),
        "n_neighbors": n_neighbors,
        "permutations": permutations,
        "global_moran_i": global_i,
        "global_p_value": global_p,
        "local": local_rows,
        "note": "exploratory kNN Moran diagnostics; not a performance metric",
    }


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return monotone Benjamini-Hochberg adjusted p-values."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def continuous_boyce_from_availability(
    site_scores: np.ndarray,
    availability_scores: np.ndarray,
    n_windows: int = 20,
    window_fraction: float = 0.10,
) -> Dict[str, Any]:
    """Compute continuous Boyce diagnostics from sites and mapped availability.

    Scores are first converted to availability percentiles, so pooled
    out-of-fold predictions from different fitted folds share a common scale.
    The moving-window predicted-to-expected ratio is then correlated with the
    window midpoint.  Undefined cases are returned honestly as ``None``.
    """
    if n_windows < 3:
        raise ValueError("n_windows must be at least 3")
    if not 0 < window_fraction <= 1:
        raise ValueError("window_fraction must be in (0, 1]")
    site_percentiles = availability_percentile_ranks(site_scores, availability_scores)
    availability_percentiles = availability_percentile_ranks(
        availability_scores, availability_scores
    )
    starts = np.linspace(0.0, 1.0 - window_fraction, n_windows)
    midpoints = []
    ratios = []
    for start in starts:
        end = start + window_fraction
        include_end = np.isclose(end, 1.0)
        site_member = (site_percentiles >= start) & (
            site_percentiles <= end if include_end else site_percentiles < end
        )
        available_member = (availability_percentiles >= start) & (
            availability_percentiles <= end if include_end else availability_percentiles < end
        )
        expected = float(np.mean(available_member))
        if expected <= 0:
            continue
        midpoints.append(float((start + end) / 2.0))
        ratios.append(float(np.mean(site_member) / expected))
    value: Optional[float] = None
    if len(ratios) >= 3 and np.unique(ratios).size >= 2:
        correlation = float(spearmanr(midpoints, ratios).statistic)
        if np.isfinite(correlation):
            value = correlation
    return {
        "boyce": value,
        "window_midpoints": midpoints,
        "predicted_expected_ratios": ratios,
        "site_percentiles": site_percentiles.tolist(),
    }


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


def _validated_finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    array: np.ndarray = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _safe_correlation(left: np.ndarray, right: np.ndarray, rank: bool) -> float:
    if len(left) < 2:
        return 1.0 if np.allclose(left, right) else 0.0
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return 1.0 if np.allclose(left, right) else 0.0
    value = (
        float(spearmanr(left, right).statistic) if rank else float(np.corrcoef(left, right)[0, 1])
    )
    return value if np.isfinite(value) else 0.0
