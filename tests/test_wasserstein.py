"""Tests for Sliced Wasserstein distance and kernel implementations."""

import jax.numpy as jnp
import jax.random as random
import pytest

from klrfome.kernels.wasserstein import (
    SlicedWassersteinDistance,
    WassersteinKernel,
    sample_unit_sphere,
    wasserstein_1d_p1,
    wasserstein_1d_p2,
    estimate_sigma_from_distances,
)
from klrfome.data.formats import SampleCollection


class TestUnitSphereSampling:
    """Tests for random unit vector generation."""
    
    def test_vectors_are_unit_length(self):
        """All sampled vectors should have norm 1.0."""
        key = random.PRNGKey(42)
        vectors = sample_unit_sphere(key, n_projections=100, dimension=10)
        norms = jnp.linalg.norm(vectors, axis=1)
        assert jnp.allclose(norms, 1.0, atol=1e-6)
    
    def test_correct_shape(self):
        """Output should have shape (n_projections, dimension)."""
        key = random.PRNGKey(42)
        vectors = sample_unit_sphere(key, n_projections=50, dimension=5)
        assert vectors.shape == (50, 5)
    
    def test_reproducible_with_same_seed(self):
        """Same key should produce same vectors."""
        key1 = random.PRNGKey(42)
        key2 = random.PRNGKey(42)
        v1 = sample_unit_sphere(key1, n_projections=20, dimension=5)
        v2 = sample_unit_sphere(key2, n_projections=20, dimension=5)
        assert jnp.allclose(v1, v2)
    
    def test_different_seeds_produce_different_vectors(self):
        """Different keys should produce different vectors."""
        key1 = random.PRNGKey(42)
        key2 = random.PRNGKey(123)
        v1 = sample_unit_sphere(key1, n_projections=20, dimension=5)
        v2 = sample_unit_sphere(key2, n_projections=20, dimension=5)
        assert not jnp.allclose(v1, v2)


class TestWasserstein1D:
    """Tests for 1D Wasserstein distance functions."""
    
    def test_identical_distributions_p1(self):
        """W_1 of identical distributions should be 0."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert jnp.isclose(wasserstein_1d_p1(x, x), 0.0, atol=1e-6)
    
    def test_identical_distributions_p2(self):
        """W_2 of identical distributions should be 0."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert jnp.isclose(wasserstein_1d_p2(x, x), 0.0, atol=1e-6)
    
    def test_shifted_distribution_p1(self):
        """W_1 of distributions shifted by constant c should be c."""
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([1.0, 2.0, 3.0])  # Shifted by 1
        assert jnp.isclose(wasserstein_1d_p1(x, y), 1.0, atol=1e-6)
    
    def test_shifted_distribution_p2(self):
        """W_2 of distributions shifted by constant c should be c."""
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([1.0, 2.0, 3.0])  # Shifted by 1
        assert jnp.isclose(wasserstein_1d_p2(x, y), 1.0, atol=1e-6)
    
    def test_symmetric_p1(self):
        """W_1(x, y) should equal W_1(y, x)."""
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        y = jnp.array([0.5, 1.5, 2.5])
        assert jnp.isclose(wasserstein_1d_p1(x, y), wasserstein_1d_p1(y, x), atol=1e-6)
    
    def test_symmetric_p2(self):
        """W_2(x, y) should equal W_2(y, x)."""
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        y = jnp.array([0.5, 1.5, 2.5])
        assert jnp.isclose(wasserstein_1d_p2(x, y), wasserstein_1d_p2(y, x), atol=1e-6)
    
    def test_non_negative(self):
        """Wasserstein distance should be non-negative."""
        x = jnp.array([0.0, 1.0, 5.0])
        y = jnp.array([2.0, 3.0, 4.0])
        assert wasserstein_1d_p1(x, y) >= 0
        assert wasserstein_1d_p2(x, y) >= 0


class TestSlicedWasserstein:
    """Tests for the SlicedWassersteinDistance class."""
    
    def test_identical_samples_zero_distance(self):
        """SW distance to itself should be 0."""
        sw = SlicedWassersteinDistance(n_projections=100, seed=42)
        X = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        assert jnp.isclose(sw(X, X), 0.0, atol=1e-5)
    
    def test_symmetric(self):
        """SW(X, Y) should equal SW(Y, X)."""
        sw = SlicedWassersteinDistance(n_projections=100, seed=42)
        X = random.normal(random.PRNGKey(0), (20, 5))
        Y = random.normal(random.PRNGKey(1), (15, 5))
        assert jnp.isclose(sw(X, Y), sw(Y, X), atol=1e-5)
    
    def test_non_negative(self):
        """SW distance should be non-negative."""
        sw = SlicedWassersteinDistance(n_projections=100, seed=42)
        X = random.normal(random.PRNGKey(0), (20, 5))
        Y = random.normal(random.PRNGKey(1), (20, 5)) + 3.0
        assert sw(X, Y) >= 0
    
    def test_triangle_inequality(self):
        """SW should satisfy triangle inequality: d(X,Z) <= d(X,Y) + d(Y,Z)."""
        sw = SlicedWassersteinDistance(n_projections=200, seed=42)
        X = random.normal(random.PRNGKey(0), (20, 5))
        Y = random.normal(random.PRNGKey(1), (20, 5))
        Z = random.normal(random.PRNGKey(2), (20, 5))
        
        d_xy = sw(X, Y)
        d_yz = sw(Y, Z)
        d_xz = sw(X, Z)
        
        assert d_xz <= d_xy + d_yz + 1e-5  # Small tolerance for numerical error
    
    def test_detects_distributional_difference(self):
        """SW should detect bimodal vs unimodal even with similar means."""
        sw = SlicedWassersteinDistance(n_projections=100, seed=42)
        
        # Bimodal: two clusters
        bimodal = jnp.concatenate([
            random.normal(random.PRNGKey(0), (50, 3)) - 2,
            random.normal(random.PRNGKey(1), (50, 3)) + 2
        ])
        
        # Unimodal: single cluster at origin (similar mean as bimodal)
        unimodal = random.normal(random.PRNGKey(2), (100, 3)) * 2
        
        # Another unimodal sample
        unimodal2 = random.normal(random.PRNGKey(3), (100, 3)) * 2
        
        # Distance between unimodals should be less than bimodal-unimodal
        d_bimodal_unimodal = sw(bimodal, unimodal)
        d_unimodal_unimodal = sw(unimodal, unimodal2)
        
        assert d_bimodal_unimodal > d_unimodal_unimodal
    
    def test_pairwise_distances_shape(self):
        """pairwise_distances should return NxN matrix."""
        sw = SlicedWassersteinDistance(n_projections=50, seed=42)
        
        collections = [
            SampleCollection(
                samples=random.normal(random.PRNGKey(i), (10, 4)),
                label=i % 2,
                id=f"coll_{i}"
            )
            for i in range(5)
        ]
        
        D = sw.pairwise_distances(collections)
        assert D.shape == (5, 5)
    
    def test_pairwise_distances_symmetric(self):
        """Pairwise distance matrix should be symmetric."""
        sw = SlicedWassersteinDistance(n_projections=50, seed=42)
        
        collections = [
            SampleCollection(
                samples=random.normal(random.PRNGKey(i), (10, 4)),
                label=i % 2,
                id=f"coll_{i}"
            )
            for i in range(5)
        ]
        
        D = sw.pairwise_distances(collections)
        assert jnp.allclose(D, D.T, atol=1e-5)
    
    def test_pairwise_distances_diagonal_zero(self):
        """Diagonal of pairwise distance matrix should be 0."""
        sw = SlicedWassersteinDistance(n_projections=50, seed=42)
        
        collections = [
            SampleCollection(
                samples=random.normal(random.PRNGKey(i), (10, 4)),
                label=i % 2,
                id=f"coll_{i}"
            )
            for i in range(5)
        ]
        
        D = sw.pairwise_distances(collections)
        assert jnp.allclose(jnp.diag(D), 0.0, atol=1e-5)
    
    def test_p1_vs_p2(self):
        """p=1 and p=2 should give different but related distances."""
        sw1 = SlicedWassersteinDistance(n_projections=100, p=1, seed=42)
        sw2 = SlicedWassersteinDistance(n_projections=100, p=2, seed=42)
        
        X = random.normal(random.PRNGKey(0), (30, 5))
        Y = random.normal(random.PRNGKey(1), (30, 5)) + 1.0
        
        d1 = sw1(X, Y)
        d2 = sw2(X, Y)
        
        # Both should be positive
        assert d1 > 0
        assert d2 > 0
        # They should be different (p=2 weights larger differences more)
        assert not jnp.isclose(d1, d2, atol=0.01)


class TestWassersteinKernel:
    """Tests for the WassersteinKernel class."""
    
    def test_self_similarity_is_one(self):
        """K(X, X) should be 1.0."""
        kernel = WassersteinKernel(sigma=1.0, n_projections=100, seed=42)
        X = random.normal(random.PRNGKey(0), (20, 5))
        assert jnp.isclose(kernel(X, X), 1.0, atol=1e-4)
    
    def test_similarity_in_zero_one(self):
        """Kernel similarity should be in [0, 1]."""
        kernel = WassersteinKernel(sigma=1.0, n_projections=100, seed=42)
        X = random.normal(random.PRNGKey(0), (20, 5))
        Y = random.normal(random.PRNGKey(1), (20, 5)) + 5  # Shifted
        sim = kernel(X, Y)
        assert 0 <= sim <= 1
    
    def test_smaller_sigma_lower_similarity(self):
        """Smaller sigma should give lower similarity for distant distributions."""
        X = random.normal(random.PRNGKey(0), (20, 5))
        Y = random.normal(random.PRNGKey(1), (20, 5)) + 2
        
        kernel_small = WassersteinKernel(sigma=0.5, n_projections=100, seed=42)
        kernel_large = WassersteinKernel(sigma=2.0, n_projections=100, seed=42)
        
        sim_small = kernel_small(X, Y)
        sim_large = kernel_large(X, Y)
        
        assert sim_small < sim_large
    
    def test_similarity_matrix_shape(self):
        """build_similarity_matrix should return NxN matrix."""
        kernel = WassersteinKernel(sigma=1.0, n_projections=50, seed=42)
        
        collections = [
            SampleCollection(
                samples=random.normal(random.PRNGKey(i), (15, 4)),
                label=i % 2,
                id=f"coll_{i}"
            )
            for i in range(10)
        ]
        
        K = kernel.build_similarity_matrix(collections)
        assert K.shape == (10, 10)
    
    def test_similarity_matrix_symmetric(self):
        """Similarity matrix should be symmetric."""
        kernel = WassersteinKernel(sigma=1.0, n_projections=50, seed=42)
        
        collections = [
            SampleCollection(
                samples=random.normal(random.PRNGKey(i), (15, 4)),
                label=i % 2,
                id=f"coll_{i}"
            )
            for i in range(10)
        ]
        
        K = kernel.build_similarity_matrix(collections)
        assert jnp.allclose(K, K.T, atol=1e-5)
    
    def test_similarity_matrix_diagonal_ones(self):
        """Diagonal of similarity matrix should be 1.0."""
        kernel = WassersteinKernel(sigma=1.0, n_projections=50, seed=42)
        
        collections = [
            SampleCollection(
                samples=random.normal(random.PRNGKey(i), (15, 4)),
                label=i % 2,
                id=f"coll_{i}"
            )
            for i in range(10)
        ]
        
        K = kernel.build_similarity_matrix(collections)
        assert jnp.allclose(jnp.diag(K), 1.0, atol=1e-4)
    
    def test_similarity_matrix_values_in_range(self):
        """All similarity matrix values should be in [0, 1]."""
        kernel = WassersteinKernel(sigma=1.0, n_projections=50, seed=42)
        
        collections = [
            SampleCollection(
                samples=random.normal(random.PRNGKey(i), (15, 4)),
                label=i % 2,
                id=f"coll_{i}"
            )
            for i in range(10)
        ]
        
        K = kernel.build_similarity_matrix(collections)
        assert jnp.all(K >= 0)
        assert jnp.all(K <= 1)
    
    def test_compute_new_to_training(self):
        """compute_new_to_training should return correct shape."""
        kernel = WassersteinKernel(sigma=1.0, n_projections=50, seed=42)
        
        training = [
            SampleCollection(
                samples=random.normal(random.PRNGKey(i), (15, 4)),
                label=i % 2,
                id=f"train_{i}"
            )
            for i in range(5)
        ]
        
        new_samples = random.normal(random.PRNGKey(100), (10, 4))
        
        similarities = kernel.compute_new_to_training(new_samples, training)
        assert similarities.shape == (5,)
        assert jnp.all(similarities >= 0)
        assert jnp.all(similarities <= 1)


class TestEstimateSigma:
    """Tests for sigma estimation utility."""
    
    def test_returns_positive_value(self):
        """Estimated sigma should be positive."""
        D = jnp.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [2.0, 1.5, 0.0]
        ])
        sigma = estimate_sigma_from_distances(D)
        assert sigma > 0
    
    def test_median_percentile(self):
        """With percentile=50, should return median distance."""
        D = jnp.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0]
        ])
        # Upper triangle: [1.0, 2.0, 3.0], median = 2.0
        sigma = estimate_sigma_from_distances(D, percentile=50.0)
        assert jnp.isclose(sigma, 2.0, atol=0.1)


class TestIntegration:
    """Integration tests for full Wasserstein workflow."""
    
    def test_full_workflow(self):
        """Test complete workflow: collections -> similarity matrix."""
        # Create synthetic collections
        n_sites = 5
        n_background = 10
        samples_per_collection = 20
        n_features = 3
        
        collections = []
        
        # Sites: samples from one distribution
        for i in range(n_sites):
            samples = random.normal(random.PRNGKey(i), (samples_per_collection, n_features))
            collections.append(SampleCollection(
                samples=samples,
                label=1,
                id=f"site_{i}"
            ))
        
        # Background: samples from shifted distribution
        for i in range(n_background):
            samples = random.normal(
                random.PRNGKey(100 + i), 
                (samples_per_collection, n_features)
            ) + 2.0
            collections.append(SampleCollection(
                samples=samples,
                label=0,
                id=f"bg_{i}"
            ))
        
        # Build kernel matrix
        kernel = WassersteinKernel(sigma=1.0, n_projections=100, seed=42)
        K = kernel.build_similarity_matrix(collections)
        
        # Verify properties
        assert K.shape == (n_sites + n_background, n_sites + n_background)
        assert jnp.allclose(K, K.T, atol=1e-5)
        assert jnp.allclose(jnp.diag(K), 1.0, atol=1e-4)
        assert jnp.all(K >= 0)
        assert jnp.all(K <= 1)
        
        # Sites should be more similar to each other than to background
        site_site_sim = K[:n_sites, :n_sites]
        site_bg_sim = K[:n_sites, n_sites:]
        
        mean_site_site = jnp.mean(site_site_sim[jnp.triu_indices(n_sites, k=1)])
        mean_site_bg = jnp.mean(site_bg_sim)

        assert mean_site_site > mean_site_bg


class TestBucketing:
    """Tests for the new bucketing functionality."""

    def test_bucket_creation_with_width(self):
        """Test _create_buckets() with bucket_width parameter."""
        from klrfome.kernels.wasserstein import _create_buckets

        sizes = [10, 15, 23, 50, 73, 75, 100]
        buckets = _create_buckets(sizes, bucket_width=25)

        # Should create buckets: [0-24], [25-49], [50-74], [75-99], [100-124]
        assert 0 in buckets  # 10, 15, 23
        assert len(buckets[0]) == 3
        assert 2 in buckets  # 50, 73
        assert len(buckets[2]) == 2
        assert 3 in buckets  # 75
        assert 4 in buckets  # 100

    def test_bucket_creation_exact_mode(self):
        """Test _create_buckets() in exact mode (no bucketing)."""
        from klrfome.kernels.wasserstein import _create_buckets

        sizes = [10, 15, 50, 73]
        buckets = _create_buckets(sizes, bucket_width=None, bucket_tolerance=0)

        # Each unique size should be its own bucket
        assert 10 in buckets
        assert 15 in buckets
        assert 50 in buckets
        assert 73 in buckets

    def test_bucket_creation_legacy_tolerance(self):
        """Test _create_buckets() with legacy bucket_tolerance parameter."""
        from klrfome.kernels.wasserstein import _create_buckets

        sizes = [10, 15, 20, 25]
        buckets = _create_buckets(sizes, bucket_width=None, bucket_tolerance=5)

        # bucket_tolerance=5 uses integer division
        # 10//5=2, 15//5=3, 20//5=4, 25//5=5
        assert 2 in buckets
        assert 3 in buckets
        assert 4 in buckets
        assert 5 in buckets

    def test_bucketing_ceiling_vs_median(self):
        """Test that bucket_ceil parameter affects target size selection."""
        # Create test collections with sizes [25, 30, 35, 40] (all in same bucket)
        key = random.PRNGKey(42)
        collections = []
        for size in [25, 30, 35, 40]:
            samples = random.normal(key, shape=(size, 3))
            collections.append(SampleCollection(
                samples=samples,
                label=1,
                id=f"coll_{size}"
            ))
            key, _ = random.split(key)

        distance_fn = SlicedWassersteinDistance(n_projections=10, seed=42)

        # Test ceiling mode (should target 40)
        K_ceil = distance_fn.pairwise_distances(
            collections, bucket_width=50, bucket_ceil=True
        )

        # Test median mode (should target 32 or 33)
        K_median = distance_fn.pairwise_distances(
            collections, bucket_width=50, bucket_ceil=False
        )

        # Matrices should differ slightly due to different target sizes
        assert not jnp.allclose(K_ceil, K_median, atol=1e-6)

    def test_global_bucket_cap(self):
        """Test that bucket_cap limits maximum sample size."""
        # Create collections with sizes [100, 5000]
        key = random.PRNGKey(42)
        collections = [
            SampleCollection(
                samples=random.normal(key, shape=(100, 3)),
                label=1,
                id="small"
            ),
            SampleCollection(
                samples=random.normal(random.PRNGKey(123), shape=(5000, 3)),
                label=1,
                id="large"
            )
        ]

        distance_fn = SlicedWassersteinDistance(n_projections=10, seed=42)

        # With cap=2500, the 5000-sample collection should be downsampled
        K = distance_fn.pairwise_distances(
            collections, bucket_width=25, bucket_cap=2500
        )

        # Should complete without error and produce valid similarity matrix
        assert K.shape == (2, 2)
        assert jnp.all(jnp.isfinite(K))
        # Distance matrix should be symmetric
        assert jnp.allclose(K, K.T, atol=1e-5)

    def test_exact_mode_vs_bucketing(self):
        """Test that exact mode (no bucketing) differs from bucketing mode."""
        key = random.PRNGKey(42)
        collections = []
        for size in [48, 50, 52]:
            samples = random.normal(key, shape=(size, 3))
            collections.append(SampleCollection(
                samples=samples,
                label=1,
                id=f"coll_{size}"
            ))
            key, _ = random.split(key)

        distance_fn = SlicedWassersteinDistance(n_projections=100, seed=42)

        # Exact mode
        K_exact = distance_fn.pairwise_distances(collections, bucket_width=None)

        # Bucketing mode (all -> 52 samples via bucket_width=25)
        K_bucketed = distance_fn.pairwise_distances(
            collections, bucket_width=25, bucket_ceil=True
        )

        # Should differ slightly
        assert not jnp.allclose(K_exact, K_bucketed, atol=1e-4)
        # But should be reasonably close (quantile interpolation is nearly lossless)
        assert jnp.allclose(K_exact, K_bucketed, atol=0.15)

    def test_backward_compatibility_bucket_tolerance(self):
        """Test that legacy bucket_tolerance parameter still works."""
        key = random.PRNGKey(42)
        collections = []
        for size in [48, 50, 52, 98, 100, 102]:
            samples = random.normal(key, shape=(size, 3))
            collections.append(SampleCollection(
                samples=samples,
                label=1,
                id=f"coll_{size}"
            ))
            key, _ = random.split(key)

        distance_fn = SlicedWassersteinDistance(n_projections=50, seed=42)

        # Use legacy bucket_tolerance (should still work)
        K = distance_fn.pairwise_distances(
            collections, bucket_tolerance=10
        )

        # Should complete without error
        assert K.shape == (6, 6)
        assert jnp.all(jnp.isfinite(K))
        assert jnp.allclose(K, K.T, atol=1e-5)

    def test_wasserstein_kernel_with_bucketing(self):
        """Test WassersteinKernel.build_similarity_matrix() with bucketing parameters."""
        key = random.PRNGKey(42)
        collections = []
        for i, size in enumerate([25, 30, 50, 75, 100]):
            samples = random.normal(key, shape=(size, 3))
            collections.append(SampleCollection(
                samples=samples,
                label=i % 2,  # Alternate labels
                id=f"coll_{i}"
            ))
            key, _ = random.split(key)

        kernel = WassersteinKernel(sigma=1.0, n_projections=100, seed=42)
        K = kernel.build_similarity_matrix(
            collections,
            bucket_width=25,
            bucket_ceil=True,
            bucket_cap=200
        )

        # Verify properties
        assert K.shape == (5, 5)
        assert jnp.allclose(K, K.T, atol=1e-5)
        # Diagonal should be close to 1 (self-similarity)
        assert jnp.allclose(jnp.diag(K), 1.0, atol=1e-3)
        assert jnp.all(K >= 0)
        assert jnp.all(K <= 1)

