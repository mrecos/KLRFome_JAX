"""
KLRfome - Kernel Logistic Regression on Focal Mean Embeddings

A Python/JAX implementation for geographic distribution regression using kernel methods.
"""

__version__ = "0.1.0"

# Core data structures
from .data.formats import (
    SampleCollection,
    TrainingData,
    RasterStack,
)

# Kernels
from .kernels.rbf import RBFKernel
from .kernels.rff import RandomFourierFeatures
from .kernels.distribution import MeanEmbeddingKernel
from .kernels.wasserstein import (
    SlicedWassersteinDistance,
    WassersteinKernel,
    estimate_sigma_from_distances,
)

# Models
from .models.klr import (
    KernelLogisticRegression,
    KLRFitResult,
)

# Prediction
from .prediction.focal import FocalPredictor, WassersteinFocalPredictor

# High-level API
from .api import KLRfome

__all__ = [
    # Version
    "__version__",
    # Data structures
    "SampleCollection",
    "TrainingData",
    "RasterStack",
    # Kernels
    "RBFKernel",
    "RandomFourierFeatures",
    "MeanEmbeddingKernel",
    "SlicedWassersteinDistance",
    "WassersteinKernel",
    "estimate_sigma_from_distances",
    # Models
    "KernelLogisticRegression",
    "KLRFitResult",
    # Prediction
    "FocalPredictor",
    "WassersteinFocalPredictor",
    # High-level API
    "KLRfome",
]

