"""Tests for data structures."""

import pytest
import jax.numpy as jnp
import jax.random as random

from klrfome.data.formats import SampleCollection, TrainingData, RasterStack


def test_sample_collection():
    """Test SampleCollection properties."""
    key = random.PRNGKey(42)
    samples = random.normal(key, (20, 5))
    
    coll = SampleCollection(
        samples=samples,
        label=1,
        id="test_site"
    )
    
    assert coll.n_samples == 20
    assert coll.n_features == 5
    assert coll.label == 1
    assert coll.id == "test_site"
    
    # Test mean embedding
    mean = coll.mean_embedding()
    assert mean.shape == (5,)
    assert jnp.allclose(mean, jnp.mean(samples, axis=0))


def test_training_data_split(training_data):
    """Test train/test split with stratification."""
    train_data, test_data = training_data.train_test_split(
        test_fraction=0.4,
        stratify=True,
        seed=42
    )
    
    # Check sizes
    assert train_data.n_locations + test_data.n_locations == training_data.n_locations
    
    # Check stratification (roughly)
    train_site_ratio = train_data.n_sites / train_data.n_locations
    test_site_ratio = test_data.n_sites / test_data.n_locations
    
    # Should be roughly similar
    assert abs(train_site_ratio - test_site_ratio) < 0.2


def test_training_data_properties(training_data):
    """Test TrainingData properties."""
    assert training_data.n_locations == len(training_data.collections)
    assert training_data.n_sites + training_data.n_background == training_data.n_locations
    
    labels = training_data.labels
    assert len(labels) == training_data.n_locations
    assert jnp.all((labels == 0) | (labels == 1))


def test_raster_stack_properties():
    """Test RasterStack basic properties."""
    import numpy as np
    from rasterio.transform import from_bounds
    
    # Create dummy data
    n_bands = 3
    height = 100
    width = 200
    data = jnp.array(np.random.rand(n_bands, height, width))
    
    transform = from_bounds(0, 0, 1, 1, width, height)
    crs = "EPSG:4326"
    band_names = ["band1", "band2", "band3"]
    
    stack = RasterStack(
        data=data,
        transform=transform,
        crs=crs,
        band_names=band_names
    )
    
    assert stack.n_bands == n_bands
    assert stack.height == height
    assert stack.width == width
    assert len(stack.band_names) == n_bands


def test_raster_stack_extract_window():
    """Test window extraction from RasterStack."""
    import numpy as np
    from rasterio.transform import from_bounds
    
    n_bands = 2
    height = 50
    width = 50
    data = jnp.array(np.random.rand(n_bands, height, width))
    
    transform = from_bounds(0, 0, 1, 1, width, height)
    crs = "EPSG:4326"
    band_names = ["band1", "band2"]
    
    stack = RasterStack(
        data=data,
        transform=transform,
        crs=crs,
        band_names=band_names
    )
    
    # Extract 5x5 window at center
    window = stack.extract_window(25, 25, 5)
    
    assert window.shape == (5, 5, n_bands)

