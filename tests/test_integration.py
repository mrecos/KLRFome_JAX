"""Integration tests for full pipeline."""

import pytest
import jax.numpy as jnp
import jax.random as random
import numpy as np
from rasterio.transform import from_bounds

from klrfome.api import KLRfome
from klrfome.data.formats import SampleCollection, TrainingData, RasterStack
from klrfome.kernels.rbf import RBFKernel
from klrfome.kernels.rff import RandomFourierFeatures
from klrfome.kernels.distribution import MeanEmbeddingKernel


def test_end_to_end_simulated():
    """Full pipeline should work on simulated data."""
    # Generate simulated training data
    key = random.PRNGKey(42)
    collections = []
    
    # Create site collections (label=1)
    for i in range(5):
        samples = random.normal(key, (15, 3)) + 1.0  # Offset for sites
        key, _ = random.split(key)
        coll = SampleCollection(
            samples=samples,
            label=1,
            id=f"site_{i}"
        )
        collections.append(coll)
    
    # Create background collections (label=0)
    for i in range(5):
        samples = random.normal(key, (15, 3)) - 1.0  # Offset for background
        key, _ = random.split(key)
        coll = SampleCollection(
            samples=samples,
            label=0,
            id=f"background_{i}"
        )
        collections.append(coll)
    
    training_data = TrainingData(
        collections=collections,
        feature_names=["var1", "var2", "var3"]
    )
    
    # Create simple raster stack
    n_bands = 3
    height = 30
    width = 30
    data = jnp.array(np.random.rand(n_bands, height, width))
    
    transform = from_bounds(0, 0, 1, 1, width, height)
    raster_stack = RasterStack(
        data=data,
        transform=transform,
        crs="EPSG:4326",
        band_names=["var1", "var2", "var3"]
    )
    
    # Initialize and fit model
    model = KLRfome(
        sigma=1.0,
        lambda_reg=0.1,
        n_rff_features=128,  # Use RFF for speed
        window_size=3,
        seed=42
    )
    
    model.fit(training_data)
    
    # Predict
    predictions = model.predict(
        raster_stack,
        batch_size=100,
        show_progress=False
    )
    
    # Check outputs
    assert predictions.shape == (height, width)
    assert jnp.all((predictions >= 0) & (predictions <= 1))
    assert model._fit_result is not None
    assert model._fit_result.converged


def test_klrfome_model_initialization():
    """Test KLRfome model initialization."""
    model = KLRfome(
        sigma=0.5,
        lambda_reg=0.2,
        n_rff_features=256,
        window_size=5,
        seed=123
    )
    
    assert model.sigma == 0.5
    assert model.lambda_reg == 0.2
    assert model.n_rff_features == 256
    assert model.window_size == 5
    assert model._distribution_kernel is not None
    assert model._klr is not None


def test_klrfome_fit_and_predict_workflow():
    """Test that fit and predict workflow works."""
    # Create minimal training data
    key = random.PRNGKey(42)
    collections = []
    
    # One site, one background
    site_samples = random.normal(key, (10, 2)) + 1.0
    key, _ = random.split(key)
    bg_samples = random.normal(key, (10, 2)) - 1.0
    
    collections.append(SampleCollection(samples=site_samples, label=1, id="site_1"))
    collections.append(SampleCollection(samples=bg_samples, label=0, id="bg_1"))
    
    training_data = TrainingData(
        collections=collections,
        feature_names=["var1", "var2"]
    )
    
    # Create minimal raster
    data = jnp.array(np.random.rand(2, 10, 10))
    transform = from_bounds(0, 0, 1, 1, 10, 10)
    raster_stack = RasterStack(
        data=data,
        transform=transform,
        crs="EPSG:4326",
        band_names=["var1", "var2"]
    )
    
    # Fit and predict
    model = KLRfome(sigma=1.0, lambda_reg=0.1, n_rff_features=64, window_size=3)
    model.fit(training_data)
    
    predictions = model.predict(raster_stack, show_progress=False)
    
    assert predictions.shape == (10, 10)

