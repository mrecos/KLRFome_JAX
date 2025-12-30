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

# Models
from .models.klr import (
    KernelLogisticRegression,
    KLRFitResult,
)

# Prediction
from .prediction.focal import FocalPredictor

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
    # Models
    "KernelLogisticRegression",
    "KLRFitResult",
    # Prediction
    "FocalPredictor",
    # High-level API
    "KLRfome",
]

