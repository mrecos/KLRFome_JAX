"""Validation and metrics utilities for KLRfome."""

from dataclasses import dataclass
from typing import List, Dict, Union, Optional, Sequence, Tuple, Any
import copy
import numpy as np
import pandas as pd
from jaxtyping import Array, Float

from ..data.formats import TrainingData


def CM_quads(
    pred: Union[Float[Array, "n"], np.ndarray, List[float]],
    obs: Union[Float[Array, "n"], np.ndarray, List[int]],
    threshold: Union[float, List[float]] = 0.5,
) -> pd.DataFrame:
    """
    Compute Confusion Matrix quadrants at one or more thresholds.

    Parameters:
        pred: Predicted probabilities
        obs: Observed presence/absence (1/0)
        threshold: Single threshold or list of thresholds

    Returns:
        DataFrame with TP, FP, TN, FN for each threshold
    """
    pred = np.array(pred)
    obs = np.array(obs)

    if isinstance(threshold, (int, float)):
        threshold = [threshold]

    results = []

    for thresh in threshold:
        pred_cat = (pred >= thresh).astype(int)

        TP = np.sum((pred_cat == 1) & (obs == 1))
        FP = np.sum((pred_cat == 1) & (obs == 0))
        TN = np.sum((pred_cat == 0) & (obs == 0))
        FN = np.sum((pred_cat == 0) & (obs == 1))

        results.append({"Threshold": thresh, "TP": TP, "FP": FP, "TN": TN, "FN": FN})

    return pd.DataFrame(results)


def cohens_kappa(TP: int, TN: int, FP: int, FN: int) -> float:
    """
    Compute Cohen's Kappa statistic.

    Parameters:
        TP: True Positives
        TN: True Negatives
        FP: False Positives
        FN: False Negatives

    Returns:
        Cohen's Kappa value
    """
    A, B, C, D = TP, FP, FN, TN
    n = A + B + C + D

    if n == 0:
        return 0.0

    Po = (A + D) / n
    Pe_a = ((A + B) * (A + C)) / n
    Pe_b = ((C + D) * (B + D)) / n
    Pe = (Pe_a + Pe_b) / n

    if Pe == 1:
        return 1.0 if Po == 1 else 0.0

    k = (Po - Pe) / (1 - Pe)
    return float(k)


def metrics(TP: int, TN: int, FP: int, FN: int) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics.

    Returns 50+ metrics based on TP, TN, FP, FN.
    Similar to the R version.

    Parameters:
        TP: True Positives
        TN: True Negatives
        FP: False Positives
        FN: False Negatives

    Returns:
        Dictionary of metric names and values
    """
    # Basic calculations
    n = TP + TN + FP + FN
    if n == 0:
        return {}

    Sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    Specificity = TN / (FP + TN) if (FP + TN) > 0 else 0.0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    Recall = Sensitivity

    # Create binary vectors for some metrics
    pred = np.array([1] * TP + [1] * FP + [0] * FN + [0] * TN)
    obs = np.array([1] * TP + [0] * FP + [1] * FN + [0] * TN)

    # Helper function for logit
    def logit(p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.log(p / (1 - p))

    # Helper function to safely divide
    def safe_divide(numerator, denominator, default=0.0):
        """Safely divide, returning default if denominator is zero."""
        try:
            # Check for zero or very close to zero
            if denominator == 0 or (
                isinstance(denominator, (int, float)) and abs(denominator) < 1e-10
            ):
                return default
            # Also check if it's a numpy array/scalar that's close to zero
            if hasattr(denominator, "__array__"):
                if np.abs(denominator) < 1e-10:
                    return default
            result = numerator / denominator
            # Check for inf or nan
            if np.isinf(result) or np.isnan(result):
                return default
            return result
        except (ZeroDivisionError, TypeError, ValueError):
            return default

    # Compute all metrics with safe division
    try:
        result = {
            "Sensitivity": Sensitivity,
            "Specificity": Specificity,
            "Prevalence": (TP + FN) / n,
            "Accuracy": (TP + TN) / n,
            "Err_Rate": (FP + FN) / n,
            "other": (TN + FP) / n,
            "Pm": (TP + FP) / n,
            "Pm_prime": (FN + TN) / n,
            "Precision": Precision,
            "Recall": Recall,
            "F_Measure": (
                (2 * Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0.0
            ),
            "Geo_Mean": np.sqrt(TP * TN) if TP > 0 and TN > 0 else 0.0,
            "MAE": float(np.mean(np.abs(pred - obs))),
            "RMSE": float(np.sqrt(np.mean((pred - obs) ** 2))),
            "FPR": 1 - Specificity,
            "FNR": 1 - Sensitivity,
            "TPR": Sensitivity,
            "TNR": Specificity,
            "FOR": FN / (FN + TN) if (FN + TN) > 0 else 0.0,
            "FDR": FP / (TP + FP) if (TP + FP) > 0 else 0.0,
            "Power": 1 - (1 - Sensitivity),
            "LRP": (
                safe_divide(Sensitivity, (1 - Specificity), default=float("inf"))
                if Specificity < 1
                else float("inf")
            ),
            "log_LRP": (
                np.log10(safe_divide(Sensitivity, (1 - Specificity), default=float("inf")))
                if Specificity < 1
                and not np.isinf(safe_divide(Sensitivity, (1 - Specificity), default=float("inf")))
                else float("inf")
            ),
            "LRN": (
                safe_divide((1 - Sensitivity), Specificity, default=float("inf"))
                if Specificity > 0
                else float("inf")
            ),
            "PPV": Precision,
            "NPV": TN / (FN + TN) if (FN + TN) > 0 else 0.0,
            "KG": 1 - ((1 - Specificity) / Sensitivity) if Sensitivity > 0 else 0.0,
            "KG2": 1 - (((TP + FP) / n) / Sensitivity) if Sensitivity > 0 else 0.0,
            "DOR": (
                safe_divide(
                    safe_divide(Sensitivity, (1 - Specificity), default=float("inf")),
                    safe_divide((1 - Sensitivity), Specificity, default=float("inf")),
                    default=float("inf"),
                )
                if Specificity < 1 and Specificity > 0
                else float("inf")
            ),
            "log_DOR": (
                np.log10(
                    safe_divide(
                        safe_divide(Sensitivity, (1 - Specificity), default=float("inf")),
                        safe_divide((1 - Sensitivity), Specificity, default=float("inf")),
                        default=float("inf"),
                    )
                )
                if Specificity < 1
                and Specificity > 0
                and not np.isinf(
                    safe_divide(
                        safe_divide(Sensitivity, (1 - Specificity), default=float("inf")),
                        safe_divide((1 - Sensitivity), Specificity, default=float("inf")),
                        default=float("inf"),
                    )
                )
                else float("inf")
            ),
            "D": logit(Sensitivity) - logit(1 - Specificity),
            "S": logit(Sensitivity) + logit(1 - Specificity),
            "Kappa": cohens_kappa(TP, TN, FP, FN),
            "Opp_Precision": (
                ((TP + TN) / n) - (np.abs(Specificity - Sensitivity) / (Specificity + Sensitivity))
                if (Specificity + Sensitivity) > 0
                else 0.0
            ),
            "MCC": (
                (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
                if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) > 0
                else 0.0
            ),
            "Informedness": Sensitivity + Specificity - 1,  # TSS, Youden's J
            "Markedness": (TP / (TP + FP) if (TP + FP) > 0 else 0.0)
            + (TN / (FN + TN) if (FN + TN) > 0 else 0.0)
            - 1,
            "AFK": (
                np.sqrt(
                    Sensitivity
                    * safe_divide((Sensitivity - (1 - Specificity)), ((TN + FP) / n), default=0.0)
                )
                if Sensitivity > 0 and (TN + FP) > 0 and n > 0
                else 0.0
            ),
            "Indicative": (
                safe_divide(Sensitivity, (1 - Specificity), default=float("inf"))
                if Specificity < 1
                else float("inf")
            ),
            "Indicative2": (
                safe_divide(Sensitivity, ((TP + FP) / n), default=float("inf"))
                if (TP + FP) > 0 and n > 0
                else float("inf")
            ),
            "Indicative_norm": (
                safe_divide(
                    safe_divide(Sensitivity, (1 - Specificity), default=float("inf")),
                    ((TN + FP) / n),
                    default=float("inf"),
                )
                if Specificity < 1 and (TN + FP) > 0 and n > 0
                else float("inf")
            ),
            "Indicative_norm2": (
                safe_divide(
                    safe_divide(Sensitivity, ((TP + FP) / n), default=float("inf")),
                    ((TN + FP) / n),
                    default=float("inf"),
                )
                if (TP + FP) > 0 and (TN + FP) > 0 and n > 0
                else float("inf")
            ),
            "Brier": float(np.mean((obs - pred) ** 2)),
            "X1": (
                safe_divide(safe_divide(TP, (TP + FP), default=0.0), ((TP + FN) / n), default=0.0)
                if (TP + FP) > 0 and (TP + FN) > 0 and n > 0
                else 0.0
            ),
            "X2": (
                safe_divide(safe_divide(FN, (FN + TN), default=0.0), ((TP + FN) / n), default=0.0)
                if (FN + TN) > 0 and (TP + FN) > 0 and n > 0
                else 0.0
            ),
            "X3": (
                safe_divide(safe_divide(FP, (TP + FP), default=0.0), ((TN + FP) / n), default=0.0)
                if (TP + FP) > 0 and (TN + FP) > 0 and n > 0
                else 0.0
            ),
            "X4": (
                safe_divide(safe_divide(TN, (FN + TN), default=0.0), ((TN + FP) / n), default=0.0)
                if (FN + TN) > 0 and (TN + FP) > 0 and n > 0
                else 0.0
            ),
            "PPG": (
                safe_divide(safe_divide(TP, (TP + FP), default=0.0), ((TP + FN) / n), default=0.0)
                if (TP + FP) > 0 and (TP + FN) > 0 and n > 0
                else 0.0
            ),
            "NPG": (
                safe_divide(safe_divide(FN, (FN + TN), default=0.0), ((TP + FN) / n), default=0.0)
                if (FN + TN) > 0 and (TP + FN) > 0 and n > 0
                else 0.0
            ),
            "Balance": (
                (1 - ((1 - Specificity) / Sensitivity) if Sensitivity > 0 else 0.0)
                + (1 - ((1 - Sensitivity) / Specificity) if Specificity > 0 else 0.0)
            )
            / 2,
            "Balance2": (
                (
                    1 - safe_divide(((TP + FP) / n), Sensitivity, default=0.0)
                    if Sensitivity > 0 and n > 0
                    else 0.0
                )
                + (
                    1 - safe_divide((1 - Sensitivity), ((FN + TN) / n), default=0.0)
                    if (FN + TN) > 0 and n > 0
                    else 0.0
                )
            )
            / 2,
        }
    except (ZeroDivisionError, ValueError, TypeError):
        # If any calculation fails, return empty dict or minimal metrics
        # This should rarely happen with all the guards, but just in case
        result = {
            "Sensitivity": Sensitivity,
            "Specificity": Specificity,
            "Precision": Precision,
            "Recall": Recall,
            "Accuracy": (TP + TN) / n if n > 0 else 0.0,
        }

    # Handle inf and nan values
    for key, value in result.items():
        if np.isinf(value) or np.isnan(value):
            result[key] = 0.0

    return result


def compute_roc_auc(
    pred: Union[Float[Array, "n"], np.ndarray, List[float]],
    obs: Union[Float[Array, "n"], np.ndarray, List[int]],
) -> float:
    """
    Compute ROC AUC using trapezoidal rule.

    Parameters:
        pred: Predicted probabilities
        obs: Observed labels (1/0)

    Returns:
        AUC value
    """
    from sklearn.metrics import roc_auc_score

    pred = np.array(pred)
    obs = np.array(obs)

    if np.unique(obs).size < 2:
        return float("nan")
    return float(roc_auc_score(obs, pred))


@dataclass(frozen=True)
class FoldAssignment:
    """One immutable train/test split."""

    repeat: int
    fold: int
    train_indices: Tuple[int, ...]
    test_indices: Tuple[int, ...]


@dataclass(frozen=True)
class FoldPlan:
    """Reusable folds shared by every method in a comparison."""

    bag_ids: Tuple[str, ...]
    assignments: Tuple[FoldAssignment, ...]
    n_splits: int
    n_repeats: int
    seed: int

    def validate_for(self, dataset: TrainingData) -> None:
        if tuple(bag.id for bag in dataset.collections) != self.bag_ids:
            raise ValueError("FoldPlan bag ids/order differ from the supplied dataset")
        expected = set(range(len(self.bag_ids)))
        for repeat_index in range(self.n_repeats):
            observed = []
            for assignment in self.assignments:
                if assignment.repeat == repeat_index:
                    observed.extend(assignment.test_indices)
                    if set(assignment.train_indices) & set(assignment.test_indices):
                        raise ValueError("FoldPlan contains train/test overlap")
            if len(observed) != len(expected) or set(observed) != expected:
                raise ValueError("Every bag must occur exactly once per repeat")


def make_fold_plan(
    dataset: TrainingData,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 42,
    stratified: bool = True,
    group_ids: Optional[Sequence[str]] = None,
) -> FoldPlan:
    """Create deterministic complete folds, optionally keeping groups together."""
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedGroupKFold,
        StratifiedKFold,
    )

    labels = np.asarray(dataset.labels, dtype=int)
    indices = np.arange(dataset.n_locations)
    if n_splits < 2 or n_splits > dataset.n_locations:
        raise ValueError("n_splits must be between 2 and the number of bags")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    if group_ids is None and any(bag.group_id is not None for bag in dataset.collections):
        group_ids = [bag.group_id or bag.id for bag in dataset.collections]
    groups = np.asarray(group_ids) if group_ids is not None else None
    if groups is not None and groups.shape != (dataset.n_locations,):
        raise ValueError("group_ids must have one value per bag")

    assignments = []
    for repeat_index in range(n_repeats):
        repeat_seed = seed + repeat_index
        if groups is not None and stratified:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=repeat_seed
            )
            splits = splitter.split(indices, labels, groups)
        elif groups is not None:
            # GroupKFold has no random state; deterministically relabel groups per repeat.
            rng = np.random.default_rng(repeat_seed)
            unique = np.unique(groups)
            relabel = {value: rank for rank, value in enumerate(rng.permutation(unique))}
            shuffled_groups = np.asarray([relabel[value] for value in groups])
            splitter = GroupKFold(n_splits=n_splits)
            splits = splitter.split(indices, labels, shuffled_groups)
        elif stratified:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat_seed)
            splits = splitter.split(indices, labels)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=repeat_seed)
            splits = splitter.split(indices)

        for fold_index, (train_indices, test_indices) in enumerate(splits):
            if groups is not None:
                train_groups = set(groups[train_indices])
                test_groups = set(groups[test_indices])
                if train_groups & test_groups:
                    raise RuntimeError("Generated fold leaks group members")
            assignments.append(
                FoldAssignment(
                    repeat_index,
                    fold_index,
                    tuple(int(value) for value in train_indices),
                    tuple(int(value) for value in test_indices),
                )
            )

    plan = FoldPlan(
        tuple(bag.id for bag in dataset.collections),
        tuple(assignments),
        n_splits,
        n_repeats,
        seed,
    )
    plan.validate_for(dataset)
    return plan


def cross_validate(
    model: Any,
    training_data: TrainingData,
    n_folds: int = 5,
    stratified: bool = True,
    seed: int = 42,
    fold_plan: Optional[FoldPlan] = None,
) -> Dict:
    """
    Perform k-fold cross-validation.

    Parameters:
        model: KLRfome model instance
        training_data: Training data
        n_folds: Number of folds
        stratified: Whether to stratify by label
        seed: Random seed

    Returns:
        Dictionary with metrics per fold and aggregated statistics
    """
    plan = fold_plan or make_fold_plan(
        training_data, n_splits=n_folds, seed=seed, stratified=stratified
    )
    plan.validate_for(training_data)
    fold_results = []
    for assignment in plan.assignments:
        train_data = training_data.subset(assignment.train_indices)
        test_data = training_data.subset(assignment.test_indices)
        fitted_model = model.clone() if hasattr(model, "clone") else copy.deepcopy(model)
        fitted_model.fit(train_data)
        if hasattr(fitted_model, "predict_bags"):
            test_probs_np = np.asarray(fitted_model.predict_bags(test_data))
        elif getattr(fitted_model, "_core_model", None) is not None:
            test_probs_np = np.asarray(fitted_model._core_model.predict_bags(test_data))
        else:
            raise TypeError("Model must provide predict_bags or a fitted core model")
        test_labels_np = np.asarray(test_data.labels)

        # Compute confusion matrix at threshold 0.5
        cm_df = CM_quads(test_probs_np, test_labels_np, threshold=0.5)
        if len(cm_df) > 0:
            TP = int(cm_df.iloc[0]["TP"])
            FP = int(cm_df.iloc[0]["FP"])
            TN = int(cm_df.iloc[0]["TN"])
            FN = int(cm_df.iloc[0]["FN"])
        else:
            TP = FP = TN = FN = 0

        # Compute metrics
        fold_metrics = metrics(TP, TN, FP, FN)

        # Compute AUC
        auc = compute_roc_auc(test_probs_np, test_labels_np)
        fold_metrics["AUC"] = auc
        fold_results.append(
            {
                "repeat": assignment.repeat + 1,
                "fold": assignment.fold + 1,
                "n_train": len(assignment.train_indices),
                "n_test": len(assignment.test_indices),
                "train_ids": [training_data.collections[i].id for i in assignment.train_indices],
                "test_ids": [training_data.collections[i].id for i in assignment.test_indices],
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN,
                "metrics": fold_metrics,
            }
        )

    # Compute aggregated statistics
    metric_names = list(fold_results[0]["metrics"].keys())
    aggregated = {}

    for metric_name in metric_names:
        values = [r["metrics"][metric_name] for r in fold_results]
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        aggregated[f"{metric_name}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
        aggregated[f"{metric_name}_std"] = float(np.std(finite)) if finite.size else float("nan")

    auc_values = np.asarray([row["metrics"]["AUC"] for row in fold_results], dtype=float)
    best_fold = None
    if np.isfinite(auc_values).any():
        best_index = int(np.nanargmax(auc_values))
        best_fold = fold_results[best_index]["fold"]

    return {
        "folds": fold_results,
        "n_folds": plan.n_splits,
        "n_repeats": plan.n_repeats,
        "mean_train_size": float(np.mean([r["n_train"] for r in fold_results])),
        "mean_test_size": float(np.mean([r["n_test"] for r in fold_results])),
        "aggregated_metrics": aggregated,
        "best_fold": best_fold,
        "fold_plan": plan,
    }
