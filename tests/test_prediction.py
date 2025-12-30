"""Tests for prediction functionality."""

import pytest
import jax.numpy as jnp
import jax.random as random
import numpy as np
from rasterio.transform import from_bounds

from klrfome.prediction.focal import FocalPredictor
from klrfome.kernels.rbf import RBFKernel
from klrfome.kernels.rff import RandomFourierFeatures
from klrfome.kernels.distribution import MeanEmbeddingKernel
from klrfome.data.formats import SampleCollection, TrainingData, RasterStack


@pytest.fixture
def simple_training_data():
    """Simple training data for prediction tests."""
    key = random.PRNGKey(42)
    
    collections = []
    for i in range(3):
        samples = random.normal(key, (10, 2))
        key, _ = random.split(key)
        coll = SampleCollection(
            samples=samples,
            label=1 if i < 2 else 0,
            id=f"loc_{i}"
        )
        collections.append(coll)
    
    return TrainingData(
        collections=collections,
        feature_names=["var1", "var2"]
    )


@pytest.fixture
def simple_raster_stack():
    """Simple raster stack for prediction tests."""
    n_bands = 2
    height = 20
    width = 20
    data = jnp.array(np.random.rand(n_bands, height, width))
    
    transform = from_bounds(0, 0, 1, 1, width, height)
    crs = "EPSG:4326"
    band_names = ["var1", "var2"]
    
    return RasterStack(
        data=data,
        transform=transform,
        crs=crs,
        band_names=band_names
    )


def test_focal_predictor_single_window(simple_training_data):
    """Test single window prediction."""
    # Create kernel and fit dummy alpha
    kernel = MeanEmbeddingKernel(RBFKernel(sigma=1.0))
    alpha = jnp.array([0.1, 0.2, 0.15])
    
    predictor = FocalPredictor(
        distribution_kernel=kernel,
        klr_alpha=alpha,
        training_data=simple_training_data,
        window_size=3
    )
    
    # Create dummy window samples
    key = random.PRNGKey(123)
    window_samples = random.normal(key, (9, 2))  # 3x3 window, 2 features
    
    prob = predictor.predict_window(window_samples)
    
    assert 0 <= prob <= 1
    assert isinstance(prob, (float, np.floating))


def test_focal_predictor_raster_shape(simple_training_data, simple_raster_stack):
    """Test that prediction raster has correct shape."""
    # Use RFF for faster computation
    rff = RandomFourierFeatures(sigma=1.0, n_features=64, seed=42)
    kernel = MeanEmbeddingKernel(rff)
    alpha = jnp.array([0.1, 0.2, 0.15])
    
    predictor = FocalPredictor(
        distribution_kernel=kernel,
        klr_alpha=alpha,
        training_data=simple_training_data,
        window_size=3
    )
    
    # Predict on small raster
    predictions = predictor.predict_raster(
        simple_raster_stack,
        batch_size=100,
        show_progress=False
    )
    
    assert predictions.shape == (simple_raster_stack.height, simple_raster_stack.width)
    assert jnp.all((predictions >= 0) & (predictions <= 1))

