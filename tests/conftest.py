"""Shared fixtures for tests."""

import pytest
import jax.numpy as jnp
import jax.random as random
import numpy as np

from klrfome.data.formats import SampleCollection, TrainingData
from klrfome.kernels.rbf import RBFKernel
from klrfome.kernels.rff import RandomFourierFeatures


@pytest.fixture
def rng_key():
    """Random number generator key."""
    return random.PRNGKey(42)


@pytest.fixture
def sample_data_2d(rng_key):
    """Sample 2D data for testing."""
    key1, key2 = random.split(rng_key)
    X = random.normal(key1, (10, 5))
    Y = random.normal(key2, (8, 5))
    return X, Y


@pytest.fixture
def sample_collections():
    """Sample collections for testing."""
    key = random.PRNGKey(123)
    collections = []
    
    # Create 3 site collections
    for i in range(3):
        samples = random.normal(key, (20, 5))
        key, _ = random.split(key)
        coll = SampleCollection(
            samples=samples,
            label=1,
            id=f"site_{i}"
        )
        collections.append(coll)
    
    # Create 2 background collections
    for i in range(2):
        samples = random.normal(key, (15, 5))
        key, _ = random.split(key)
        coll = SampleCollection(
            samples=samples,
            label=0,
            id=f"background_{i}"
        )
        collections.append(coll)
    
    return collections


@pytest.fixture
def training_data(sample_collections):
    """Training data for testing."""
    return TrainingData(
        collections=sample_collections,
        feature_names=[f"var_{i}" for i in range(5)],
        crs="EPSG:4326"
    )


@pytest.fixture
def rbf_kernel():
    """RBF kernel for testing."""
    return RBFKernel(sigma=1.0)


@pytest.fixture
def rff_kernel():
    """Random Fourier Features kernel for testing."""
    return RandomFourierFeatures(sigma=1.0, n_features=128, seed=42)

