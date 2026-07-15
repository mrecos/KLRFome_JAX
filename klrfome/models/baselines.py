"""Classical fixed-length baselines for canonical bag datasets."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data.formats import BagDataset


def bag_summary_matrix(dataset: BagDataset, summary: str = "mean") -> np.ndarray:
    """Return one fixed-length summary vector per bag."""
    means = np.asarray(
        [np.asarray(bag.samples, dtype=float).mean(axis=0) for bag in dataset.collections]
    )
    if summary == "mean":
        return means
    if summary == "mean_std":
        standard_deviations = np.asarray(
            [np.asarray(bag.samples, dtype=float).std(axis=0) for bag in dataset.collections]
        )
        return np.column_stack([means, standard_deviations])
    if summary == "geometry":
        sizes: np.ndarray = np.asarray([bag.n_samples for bag in dataset.collections], dtype=float)
        diameters = []
        for bag in dataset.collections:
            if bag.coordinates is None:
                raise ValueError(f"Bag {bag.id!r} requires coordinates for geometry summaries")
            coordinates = np.asarray(bag.coordinates)
            diameters.append(float(np.hypot(np.ptp(coordinates[:, 0]), np.ptp(coordinates[:, 1]))))
        return np.column_stack([np.log1p(sizes), np.log1p(diameters)])
    raise ValueError("summary must be 'mean', 'mean_std', or 'geometry'")


@dataclass
class BagSummaryClassifier:
    """A cloneable LR or RF estimator operating on fixed bag summaries."""

    name: str
    estimator_kind: str
    summary: str = "mean"
    seed: int = 42
    rf_estimators: int = 500
    estimator_: Any = field(default=None, init=False, repr=False)

    def clone(self) -> "BagSummaryClassifier":
        return BagSummaryClassifier(
            self.name,
            self.estimator_kind,
            self.summary,
            self.seed,
            self.rf_estimators,
        )

    def fit(self, dataset: BagDataset) -> "BagSummaryClassifier":
        if self.estimator_kind == "logistic":
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=5000, random_state=self.seed),
            )
        elif self.estimator_kind == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=self.rf_estimators,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=1,
            )
        else:
            raise ValueError("estimator_kind must be 'logistic' or 'random_forest'")
        estimator.fit(
            bag_summary_matrix(dataset, self.summary), np.asarray(dataset.labels, dtype=int)
        )
        self.estimator_ = estimator
        return self

    def predict_bags(self, dataset: BagDataset) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError("Baseline must be fit before prediction")
        return np.asarray(
            self.estimator_.predict_proba(bag_summary_matrix(dataset, self.summary))[:, 1]
        )


def baseline_models(
    seed: int = 42,
    rf_estimators: int = 500,
    include_mean_std: bool = False,
    include_geometry: bool = False,
) -> Dict[str, BagSummaryClassifier]:
    """Return unfitted baseline models with deterministic configurations."""
    models: List[BagSummaryClassifier] = [
        BagSummaryClassifier("LR-mean", "logistic", "mean", seed, rf_estimators),
        BagSummaryClassifier("RF-mean", "random_forest", "mean", seed, rf_estimators),
    ]
    if include_mean_std:
        models.append(
            BagSummaryClassifier("LR-mean-std", "logistic", "mean_std", seed, rf_estimators)
        )
    if include_geometry:
        models.append(
            BagSummaryClassifier("NEG-geometry", "logistic", "geometry", seed, rf_estimators)
        )
    return {model.name: model for model in models}
