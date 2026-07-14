"""Fitted, reusable preprocessing for bag datasets."""

from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np

from .formats import Bag, BagDataset


@dataclass(frozen=True)
class BagStandardizer:
    """Z-score cell features using statistics learned from training bags only."""

    means: Tuple[float, ...]
    scales: Tuple[float, ...]
    feature_names: Tuple[str, ...]
    crs: Optional[str] = None

    @classmethod
    def fit(cls, dataset: BagDataset) -> "BagStandardizer":
        pooled = np.concatenate([np.asarray(bag.samples) for bag in dataset.collections], axis=0)
        means = pooled.mean(axis=0)
        scales = pooled.std(axis=0)
        scales = np.where(scales < 1e-12, 1.0, scales)
        return cls(
            tuple(float(value) for value in means),
            tuple(float(value) for value in scales),
            tuple(dataset.feature_names),
            dataset.crs,
        )

    def transform(self, dataset: BagDataset) -> BagDataset:
        if tuple(dataset.feature_names) != self.feature_names:
            raise ValueError("Dataset feature order differs from the fitted preprocessor")
        if self.crs is not None and dataset.crs is not None and str(dataset.crs) != str(self.crs):
            raise ValueError("Dataset CRS differs from the fitted preprocessor")
        means = np.asarray(self.means)
        scales = np.asarray(self.scales)
        bags = []
        for bag in dataset.collections:
            bags.append(
                Bag(
                    samples=jnp.asarray((np.asarray(bag.samples) - means) / scales),
                    label=bag.label,
                    id=bag.id,
                    metadata=dict(bag.metadata) if bag.metadata is not None else None,
                    coordinates=bag.coordinates,
                    group_id=bag.group_id,
                    stratum_id=bag.stratum_id,
                )
            )
        return BagDataset(
            collections=bags,
            feature_names=list(dataset.feature_names),
            crs=dataset.crs,
            study_design=dataset.study_design,
            metadata=dict(dataset.metadata) if dataset.metadata is not None else None,
        )
