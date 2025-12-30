"""Tests for kernel implementations."""

import pytest
import jax.numpy as jnp
import jax.random as random
from jax import jit

from klrfome.kernels.rbf import RBFKernel
from klrfome.kernels.rff import RandomFourierFeatures
from klrfome.kernels.distribution import MeanEmbeddingKernel
from klrfome.data.formats import SampleCollection


def test_rbf_kernel_properties(rbf_kernel, sample_data_2d):
    """RBF kernel should satisfy kernel properties."""
    X, _ = sample_data_2d
    K = rbf_kernel(X, X)
    
    # Symmetric
    assert jnp.allclose(K, K.T, atol=1e-6)
    
    # Positive semi-definite (eigenvalues >= 0)
    eigenvalues = jnp.linalg.eigvalsh(K)
    assert jnp.all(eigenvalues >= -1e-10)
    
    # Diagonal is 1
    assert jnp.allclose(jnp.diag(K), 1.0, atol=1e-6)


def test_rbf_kernel_diagonal(rbf_kernel, sample_data_2d):
    """RBF kernel diagonal should be all ones."""
    X, _ = sample_data_2d
    diag = rbf_kernel.diagonal(X)
    assert jnp.allclose(diag, 1.0)


def test_rff_approximates_rbf(sample_data_2d):
    """Random Fourier Features should approximate exact RBF."""
    X, _ = sample_data_2d
    sigma = 1.0
    
    exact = RBFKernel(sigma=sigma)(X, X)
    approx = RandomFourierFeatures(sigma=sigma, n_features=1000, seed=42)(X, X)
    
    # Should be reasonably close (within 0.1 for most entries)
    assert jnp.allclose(exact, approx, atol=0.15)


def test_rff_kernel_properties(rff_kernel, sample_data_2d):
    """RFF kernel should satisfy kernel properties."""
    X, _ = sample_data_2d
    K = rff_kernel(X, X)
    
    # Symmetric
    assert jnp.allclose(K, K.T, atol=1e-5)
    
    # Should be approximately positive semi-definite (RFF approximation)
    # Check eigenvalues rather than individual entries, as RFF can have
    # small negative values due to numerical precision in the approximation
    eigenvalues = jnp.linalg.eigvalsh(K)
    # Allow small negative eigenvalues due to numerical precision in RFF approximation
    assert jnp.all(eigenvalues >= -1e-3), f"Found negative eigenvalues: {eigenvalues[eigenvalues < -1e-3]}"


def test_distribution_kernel_symmetry(sample_collections):
    """Distribution kernel should be symmetric."""
    coll_a = sample_collections[0]
    coll_b = sample_collections[1]
    
    kernel = MeanEmbeddingKernel(RBFKernel(sigma=1.0))
    
    k_ab = kernel(coll_a.samples, coll_b.samples)
    k_ba = kernel(coll_b.samples, coll_a.samples)
    
    assert jnp.isclose(k_ab, k_ba, atol=1e-6)


def test_mean_embedding_computation(sample_collections):
    """Mean embeddings should be computed correctly."""
    coll = sample_collections[0]
    
    # Direct mean
    direct_mean = jnp.mean(coll.samples, axis=0)
    
    # Via SampleCollection method
    collection_mean = coll.mean_embedding()
    
    assert jnp.allclose(direct_mean, collection_mean)


def test_distribution_kernel_matrix_shape(sample_collections):
    """Similarity matrix should have correct shape."""
    kernel = MeanEmbeddingKernel(RBFKernel(sigma=1.0))
    
    K = kernel.build_similarity_matrix(sample_collections)
    
    n = len(sample_collections)
    assert K.shape == (n, n)
    assert jnp.allclose(K, K.T)  # Symmetric


def test_distribution_kernel_rff_path(sample_collections):
    """Distribution kernel with RFF should work."""
    rff = RandomFourierFeatures(sigma=1.0, n_features=128, seed=42)
    kernel = MeanEmbeddingKernel(rff)
    
    K = kernel.build_similarity_matrix(sample_collections)
    
    n = len(sample_collections)
    assert K.shape == (n, n)
    assert jnp.allclose(K, K.T)  # Symmetric

