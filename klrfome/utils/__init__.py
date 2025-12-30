"""Utility functions for validation, scaling, GPU, etc."""

from .gpu import check_gpu_available, get_device_info, test_gpu_kernel_computation
from .scaling import (
    compute_scaling_stats,
    scale_data,
    scale_prediction_rasters,
    get_scaling_from_training_data
)
from .validation import (
    CM_quads,
    cohens_kappa,
    metrics,
    compute_roc_auc,
    cross_validate
)

__all__ = [
    "check_gpu_available",
    "get_device_info",
    "test_gpu_kernel_computation",
    "compute_scaling_stats",
    "scale_data",
    "scale_prediction_rasters",
    "get_scaling_from_training_data",
    "CM_quads",
    "cohens_kappa",
    "metrics",
    "compute_roc_auc",
    "cross_validate",
]
