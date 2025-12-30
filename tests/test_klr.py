"""Tests for Kernel Logistic Regression."""

import pytest
import jax.numpy as jnp
import jax.random as random

from klrfome.models.klr import KernelLogisticRegression, KLRFitResult
from klrfome.kernels.rbf import RBFKernel
from klrfome.kernels.distribution import MeanEmbeddingKernel
from klrfome.data.formats import SampleCollection


@pytest.fixture
def separable_data():
    """Generate clearly separable data."""
    key = random.PRNGKey(42)
    
    # Site samples: mean [2, 2]
    key1, key = random.split(key)
    site_samples = random.normal(key1, (20, 2)) + 2.0
    
    # Background samples: mean [-2, -2]
    key2, key = random.split(key)
    background_samples = random.normal(key2, (20, 2)) - 2.0
    
    site_coll = SampleCollection(samples=site_samples, label=1, id="site_1")
    bg_coll = SampleCollection(samples=background_samples, label=0, id="background_1")
    
    return [site_coll, bg_coll]


def test_klr_fits_separable_data(separable_data):
    """KLR should achieve high accuracy on separable data."""
    from klrfome.data.formats import TrainingData
    
    training_data = TrainingData(
        collections=separable_data,
        feature_names=["var1", "var2"]
    )
    
    # Build similarity matrix
    kernel = MeanEmbeddingKernel(RBFKernel(sigma=1.0))
    K = kernel.build_similarity_matrix(separable_data)
    y = training_data.labels
    
    # Fit KLR
    klr = KernelLogisticRegression(lambda_reg=0.1, max_iter=100, tol=1e-6)
    result = klr.fit(K, y)
    
    # Predict on training data
    probs = klr.predict_proba(K, result.alpha)
    predictions = (probs > 0.5).astype(int)
    accuracy = jnp.mean(predictions == y)
    
    assert accuracy > 0.8  # Should be high for separable data
    assert result.converged


def test_klr_regularization_effect():
    """Higher lambda should produce smaller alpha magnitudes."""
    key = random.PRNGKey(123)
    
    # Create simple data
    X = random.normal(key, (10, 5))
    K = RBFKernel(sigma=1.0)(X, X)
    y = jnp.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    
    klr_low = KernelLogisticRegression(lambda_reg=0.01)
    klr_high = KernelLogisticRegression(lambda_reg=10.0)
    
    result_low = klr_low.fit(K, y)
    result_high = klr_high.fit(K, y)
    
    norm_low = jnp.linalg.norm(result_low.alpha)
    norm_high = jnp.linalg.norm(result_high.alpha)
    
    assert norm_high < norm_low


def test_klr_convergence():
    """KLR should converge within max_iter."""
    key = random.PRNGKey(456)
    
    X = random.normal(key, (15, 5))
    K = RBFKernel(sigma=1.0)(X, X)
    y = jnp.array([1] * 7 + [0] * 8)
    
    klr = KernelLogisticRegression(lambda_reg=0.1, max_iter=50, tol=1e-6)
    result = klr.fit(K, y)
    
    assert result.n_iterations <= 50
    assert isinstance(result.converged, bool)


def test_klr_predict_proba_range():
    """Predicted probabilities should be in [0, 1]."""
    key = random.PRNGKey(789)
    
    X = random.normal(key, (10, 5))
    K = RBFKernel(sigma=1.0)(X, X)
    y = jnp.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    
    klr = KernelLogisticRegression(lambda_reg=0.1)
    result = klr.fit(K, y)
    
    probs = klr.predict_proba(K, result.alpha)
    
    assert jnp.all(probs >= 0)
    assert jnp.all(probs <= 1)


def test_klr_predict_binary():
    """Binary predictions should be 0 or 1."""
    key = random.PRNGKey(321)
    
    X = random.normal(key, (10, 5))
    K = RBFKernel(sigma=1.0)(X, X)
    y = jnp.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    
    klr = KernelLogisticRegression(lambda_reg=0.1)
    result = klr.fit(K, y)
    
    predictions = klr.predict(K, result.alpha, threshold=0.5)
    
    assert jnp.all((predictions == 0) | (predictions == 1))

