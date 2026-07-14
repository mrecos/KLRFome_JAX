"""Utility functions for validation, scaling, GPU, etc."""

from .gpu import check_gpu_available, get_device_info, test_gpu_kernel_computation
from .scaling import (
    compute_scaling_stats,
    scale_data,
    scale_prediction_rasters,
    get_scaling_from_training_data,
)
from .validation import CM_quads, cohens_kappa, metrics, compute_roc_auc, cross_validate
from .evaluation import (
    boyce_index,
    kernel_approximation_diagnostics,
    paired_method_differences,
    presence_background_metrics,
    score_agreement,
)
from .reproducibility import (
    configuration_fingerprint,
    dataset_fingerprint,
    environment_manifest,
    serialize_fold_plan,
    write_strict_json,
)
from .serialization import load_model, save_model

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
    "boyce_index",
    "presence_background_metrics",
    "paired_method_differences",
    "score_agreement",
    "kernel_approximation_diagnostics",
    "configuration_fingerprint",
    "dataset_fingerprint",
    "environment_manifest",
    "serialize_fold_plan",
    "write_strict_json",
    "save_model",
    "load_model",
]
