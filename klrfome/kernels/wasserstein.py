"""
Sliced Wasserstein distance and Wasserstein-based distribution kernels.

This module provides:
- SlicedWassersteinDistance: Compute SW distance between sample sets
- WassersteinKernel: RBF kernel on Sliced Wasserstein distance
- Utilities for generating random projections
"""

import jax.numpy as jnp
import jax.random as random
from jax import vmap
from functools import partial
from typing import Optional, List, Literal
from dataclasses import dataclass, field
from jaxtyping import Array, Float, PRNGKeyArray

from ..data.formats import SampleCollection


def sample_unit_sphere(
    key: PRNGKeyArray,
    n_projections: int,
    dimension: int
) -> Float[Array, "n_projections dimension"]:
    """
    Sample unit vectors uniformly from the d-dimensional unit sphere.
    
    Method: Sample from standard normal, then normalize.
    This works because the standard normal is spherically symmetric.
    
    Parameters:
        key: JAX random key
        n_projections: Number of random directions to sample (L)
        dimension: Dimensionality of the space (d)
    
    Returns:
        Array of shape (n_projections, dimension) with unit vectors
    """
    # Sample from standard normal
    raw = random.normal(key, shape=(n_projections, dimension))
    
    # Normalize to unit length
    norms = jnp.linalg.norm(raw, axis=1, keepdims=True)
    
    # Handle edge case of zero vector (astronomically unlikely but be safe)
    norms = jnp.maximum(norms, 1e-10)
    
    return raw / norms


def wasserstein_1d_p1(
    x_sorted: Float[Array, "n"],
    y_sorted: Float[Array, "m"]
) -> float:
    """
    Compute 1-Wasserstein distance between 1D empirical distributions.
    
    Assumes inputs are already sorted. Handles unequal sample sizes
    via linear interpolation of quantile functions.
    
    Parameters:
        x_sorted: Sorted samples from distribution P
        y_sorted: Sorted samples from distribution Q
    
    Returns:
        W_1(P, Q) - the 1-Wasserstein distance
    """
    n = x_sorted.shape[0]
    m = y_sorted.shape[0]
    
    # Use common grid approach for both equal and unequal sizes
    n_points = int(max(n, m))
    
    # Quantile positions (0 to 1)
    quantiles = jnp.linspace(0, 1, n_points)
    
    # Interpolate both distributions to these quantiles
    x_indices = quantiles * (n - 1)
    y_indices = quantiles * (m - 1)
    
    # Linear interpolation
    x_interp = jnp.interp(x_indices, jnp.arange(n), x_sorted)
    y_interp = jnp.interp(y_indices, jnp.arange(m), y_sorted)
    
    return jnp.mean(jnp.abs(x_interp - y_interp))


def wasserstein_1d_p2(
    x_sorted: Float[Array, "n"],
    y_sorted: Float[Array, "m"]
) -> float:
    """
    Compute 2-Wasserstein distance between 1D empirical distributions.
    
    Same as wasserstein_1d_p1 but with squared differences.
    Returns W_2 (not W_2²).
    
    Parameters:
        x_sorted: Sorted samples from distribution P
        y_sorted: Sorted samples from distribution Q
    
    Returns:
        W_2(P, Q) - the 2-Wasserstein distance
    """
    n = x_sorted.shape[0]
    m = y_sorted.shape[0]
    
    n_points = int(max(n, m))
    quantiles = jnp.linspace(0, 1, n_points)
    
    x_indices = quantiles * (n - 1)
    y_indices = quantiles * (m - 1)
    
    x_interp = jnp.interp(x_indices, jnp.arange(n), x_sorted)
    y_interp = jnp.interp(y_indices, jnp.arange(m), y_sorted)
    
    return jnp.sqrt(jnp.mean((x_interp - y_interp) ** 2))


@dataclass
class SlicedWassersteinDistance:
    """
    Sliced Wasserstein distance between empirical distributions.
    
    Approximates the Wasserstein distance by averaging 1D Wasserstein
    distances over random projections.
    
    Parameters:
        n_projections: Number of random projection directions (L).
                      More projections = better approximation but slower.
                      Typical values: 50-500. Default 100 is usually sufficient.
        p: Order of Wasserstein distance (1 or 2). Default 2.
        seed: Random seed for reproducible projections.
    
    Example:
        >>> sw = SlicedWassersteinDistance(n_projections=100, p=2, seed=42)
        >>> X = jnp.array([[0, 0], [1, 0], [0, 1]])  # 3 samples, 2D
        >>> Y = jnp.array([[2, 2], [3, 2], [2, 3], [3, 3]])  # 4 samples, 2D
        >>> distance = sw(X, Y)
    """
    n_projections: int = 100
    p: Literal[1, 2] = 2
    seed: int = 42
    
    # Lazily initialized
    _projections: Optional[Float[Array, "L d"]] = field(default=None, init=False, repr=False)
    _dimension: Optional[int] = field(default=None, init=False, repr=False)
    
    def _ensure_projections(self, dimension: int):
        """Initialize projections if needed, or reinitialize if dimension changed."""
        if self._projections is None or self._dimension != dimension:
            key = random.PRNGKey(self.seed)
            self._projections = sample_unit_sphere(key, self.n_projections, dimension)
            self._dimension = dimension
    
    def __call__(
        self,
        X: Float[Array, "n d"],
        Y: Float[Array, "m d"]
    ) -> float:
        """
        Compute Sliced Wasserstein distance between sample sets X and Y.
        
        Parameters:
            X: Samples from distribution P, shape (n_samples, n_features)
            Y: Samples from distribution Q, shape (m_samples, n_features)
        
        Returns:
            SW_p(P, Q) - the sliced Wasserstein distance
        """
        assert X.shape[1] == Y.shape[1], "X and Y must have same dimensionality"
        dimension = X.shape[1]
        
        self._ensure_projections(dimension)
        
        return self._compute_sw(X, Y, self._projections)
    
    def _compute_sw(
        self,
        X: Float[Array, "n d"],
        Y: Float[Array, "m d"],
        projections: Float[Array, "L d"]
    ) -> float:
        """Core computation of Sliced Wasserstein distance."""
        
        # Project samples onto each direction
        # X @ projections.T gives shape (n, L) - each column is X projected onto one direction
        X_projected = X @ projections.T  # (n, L)
        Y_projected = Y @ projections.T  # (m, L)
        
        # Sort along sample axis for each projection
        X_sorted = jnp.sort(X_projected, axis=0)  # (n, L)
        Y_sorted = jnp.sort(Y_projected, axis=0)  # (m, L)
        
        # Compute 1D Wasserstein for each projection
        if self.p == 1:
            # W_1: use vmap over projections
            distances = vmap(wasserstein_1d_p1)(X_sorted.T, Y_sorted.T)  # (L,)
            return float(jnp.mean(distances))
        else:  # p == 2
            # W_2²: compute for each slice, average, then sqrt
            def w2_squared(x_sort, y_sort):
                w2 = wasserstein_1d_p2(x_sort, y_sort)
                return w2 ** 2
            squared_distances = vmap(w2_squared)(X_sorted.T, Y_sorted.T)
            # Average W_2², then sqrt to get SW_2
            return float(jnp.sqrt(jnp.mean(squared_distances)))
    
    def pairwise_distances(
        self,
        collections: List[SampleCollection]
    ) -> Float[Array, "N N"]:
        """
        Compute pairwise SW distances between all collections.
        
        Parameters:
            collections: List of SampleCollection objects
        
        Returns:
            Distance matrix of shape (N, N)
        """
        n = len(collections)
        
        if n == 0:
            return jnp.array([]).reshape(0, 0)
        
        # Ensure projections are initialized
        dim = collections[0].samples.shape[1]
        self._ensure_projections(dim)
        
        # Pre-project and sort all collections for efficiency
        projected_sorted = []
        for coll in collections:
            proj = jnp.array(coll.samples) @ self._projections.T  # (n_samples, L)
            sorted_proj = jnp.sort(proj, axis=0)
            projected_sorted.append(sorted_proj)
        
        # Compute pairwise distances
        distances = jnp.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = self._compute_sw_from_sorted(
                    projected_sorted[i],
                    projected_sorted[j]
                )
                distances = distances.at[i, j].set(d)
                distances = distances.at[j, i].set(d)
        
        return distances
    
    def _compute_sw_from_sorted(
        self,
        X_sorted: Float[Array, "n L"],
        Y_sorted: Float[Array, "m L"]
    ) -> float:
        """Compute SW from pre-projected, pre-sorted data."""
        if self.p == 1:
            distances = vmap(wasserstein_1d_p1)(X_sorted.T, Y_sorted.T)
            return float(jnp.mean(distances))
        else:
            def w2_squared(x, y):
                return wasserstein_1d_p2(x, y) ** 2
            sq_dist = vmap(w2_squared)(X_sorted.T, Y_sorted.T)
            return float(jnp.sqrt(jnp.mean(sq_dist)))


@dataclass
class WassersteinKernel:
    """
    RBF kernel on Sliced Wasserstein distance.
    
    K(P, Q) = exp(-SW(P, Q)² / (2σ²))
    
    This kernel measures similarity between distributions based on their
    shape, not just their mean. Two distributions with different shapes
    will have low similarity even if their means are identical.
    
    Parameters:
        sigma: Bandwidth parameter. Controls how quickly similarity
               decays with distance. Larger sigma = slower decay.
               Should be calibrated to typical SW distances in your data.
        n_projections: Number of random projections for SW approximation.
        p: Order of Wasserstein distance (1 or 2).
        seed: Random seed for projections.
    
    Example:
        >>> kernel = WassersteinKernel(sigma=1.0, n_projections=100)
        >>> similarity = kernel(site_samples, background_samples)
    """
    sigma: float = 1.0
    n_projections: int = 100
    p: Literal[1, 2] = 2
    seed: int = 42
    
    # Internal sliced wasserstein distance calculator
    _sw: SlicedWassersteinDistance = field(init=False, repr=False)
    
    def __post_init__(self):
        self._sw = SlicedWassersteinDistance(
            n_projections=self.n_projections,
            p=self.p,
            seed=self.seed
        )
    
    def __call__(
        self,
        X: Float[Array, "n d"],
        Y: Float[Array, "m d"]
    ) -> float:
        """
        Compute kernel similarity between sample sets X and Y.
        
        Parameters:
            X: Samples from distribution P
            Y: Samples from distribution Q
        
        Returns:
            K(P, Q) in [0, 1], with 1 meaning identical distributions
        """
        sw_distance = self._sw(X, Y)
        return float(jnp.exp(-sw_distance ** 2 / (2 * self.sigma ** 2)))
    
    def build_similarity_matrix(
        self,
        collections: List[SampleCollection],
        round_kernel: bool = False,
        kernel_decimals: int = 3
    ) -> Float[Array, "N N"]:
        """
        Build the N×N similarity matrix between all collections.
        
        This is the kernel matrix used for KLR fitting.
        Matches the interface of MeanEmbeddingKernel.build_similarity_matrix.
        
        Parameters:
            collections: List of SampleCollection objects
            round_kernel: Whether to round kernel values (for R compatibility)
            kernel_decimals: Number of decimals for rounding
        
        Returns:
            Kernel matrix K where K[i,j] = K(collections[i], collections[j])
        """
        # First compute distance matrix
        distances = self._sw.pairwise_distances(collections)
        
        # Convert to similarities via RBF
        K = jnp.exp(-distances ** 2 / (2 * self.sigma ** 2))
        
        if round_kernel:
            K = jnp.round(K, kernel_decimals)
        
        return K
    
    def compute_new_to_training(
        self,
        new_samples: Float[Array, "m d"],
        training_collections: List[SampleCollection]
    ) -> Float[Array, "n_train"]:
        """
        Compute kernel values between a new sample set and all training collections.
        
        Used during prediction.
        
        Parameters:
            new_samples: Samples from the new distribution (e.g., focal window)
            training_collections: The training SampleCollections
        
        Returns:
            Array of length n_train with K(new, train_i) for each training collection
        """
        similarities = []
        for coll in training_collections:
            sim = self(new_samples, jnp.array(coll.samples))
            similarities.append(sim)
        
        return jnp.array(similarities)


# =============================================================================
# Utilities
# =============================================================================

def estimate_sigma_from_distances(
    distance_matrix: Float[Array, "N N"],
    percentile: float = 50.0
) -> float:
    """
    Estimate a reasonable sigma based on the distribution of distances.
    
    A common heuristic is to use the median distance, which ensures
    roughly half of pairwise similarities are above/below exp(-0.5) ≈ 0.6.
    
    Parameters:
        distance_matrix: Pairwise SW distances
        percentile: Which percentile of distances to use (default: median)
    
    Returns:
        Suggested sigma value
    """
    # Extract upper triangle (excluding diagonal)
    n = distance_matrix.shape[0]
    upper_tri_indices = jnp.triu_indices(n, k=1)
    upper_tri = distance_matrix[upper_tri_indices]
    return float(jnp.percentile(upper_tri, percentile))

