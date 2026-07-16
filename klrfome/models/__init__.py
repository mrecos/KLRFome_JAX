"""Model implementations for KLRfome."""

from .klr import KernelLogisticRegression, KLRFitResult
from .distribution import DistributionClassifier
from .primal import PrimalFitResult, PrimalLogisticRegression
from .spec import ModelSpec
from .shrinkage import spatial_effective_sample_size
from .baselines import BagSummaryClassifier, bag_summary_matrix, baseline_models

__all__ = [
    "KernelLogisticRegression",
    "KLRFitResult",
    "PrimalLogisticRegression",
    "PrimalFitResult",
    "ModelSpec",
    "spatial_effective_sample_size",
    "DistributionClassifier",
    "BagSummaryClassifier",
    "bag_summary_matrix",
    "baseline_models",
]
