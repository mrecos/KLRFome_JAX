# KLRfome Sliced Wasserstein Integration Specification

## Overview

This specification extends the KLRfome Python/JAX implementation to use **Sliced Wasserstein distance** as an alternative distribution similarity measure, replacing (or supplementing) mean embeddings.

**Goal**: Compare distributions by their shape, not just their mean in feature space, capturing multimodality and distributional structure that mean embeddings discard.

**Integration Principle**: The Wasserstein-based kernel slots into the existing architecture. All downstream components (KLR fitting, focal prediction) remain unchanged—we're only swapping the similarity measure.

---

## Mathematical Background

### The Wasserstein Distance

The p-Wasserstein distance between two probability distributions P and Q on ℝᵈ is defined as:

```
W_p(P, Q) = (inf_{γ ∈ Γ(P,Q)} ∫ ||x - y||^p dγ(x, y))^(1/p)
```

Where Γ(P, Q) is the set of all couplings (joint distributions) with marginals P and Q. Intuitively, it's the minimum "cost" to transport mass from P to Q.

**The Problem**: Computing this in high dimensions is O(n³) via linear programming, which is prohibitive.

### The 1D Wasserstein Distance (Closed Form)

In one dimension, Wasserstein has a beautiful closed-form solution. For empirical distributions with samples {x₁, ..., xₙ} and {y₁, ..., yₘ}:

```
W_p(P, Q) = (∫₀¹ |F_P⁻¹(t) - F_Q⁻¹(t)|^p dt)^(1/p)
```

Where F⁻¹ is the quantile function. For empirical distributions with equal sample sizes n:

```
W_p(P, Q) = ((1/n) Σᵢ |x_{(i)} - y_{(i)}|^p)^(1/p)
```

Where x_{(i)} and y_{(i)} are the sorted samples. **This is just: sort both, pair them up, compute distances.**

For unequal sample sizes, we interpolate the quantile functions (detailed below).

### Sliced Wasserstein Distance

The key insight: we can approximate high-dimensional Wasserstein by averaging 1D Wasserstein distances over random projections.

```
SW_p(P, Q) = (∫_{S^{d-1}} W_p^p(θ#P, θ#Q) dθ)^(1/p)
```

Where:
- S^{d-1} is the unit sphere in ℝᵈ
- θ#P is the pushforward of P onto the 1D subspace defined by θ (i.e., project all samples onto θ)
- We integrate over all possible projection directions

**In practice**, we approximate this integral with Monte Carlo sampling:

```
SW_p(P, Q) ≈ ((1/L) Σₗ W_p^p(θₗ#P, θₗ#Q))^(1/p)
```

Where θ₁, ..., θₗ are random unit vectors sampled uniformly from S^{d-1}.

### Properties of Sliced Wasserstein

1. **Valid metric**: SW satisfies all metric properties (non-negativity, identity of indiscernibles, symmetry, triangle inequality)

2. **Equivalent topology**: SW metrizes the same topology as true Wasserstein (weak convergence)

3. **Computational complexity**: O(L × n log n) where L is the number of slices and n is the number of samples (dominated by sorting)

4. **Differentiable**: Gradients flow through the sorting operation (important for future extensions)

5. **No curse of dimensionality**: Unlike kernel density estimation, SW doesn't degrade in high dimensions

---

## Implementation Specification

### New Module: `klrfome/kernels/wasserstein.py`

```python
"""
Sliced Wasserstein distance and Wasserstein-based distribution kernels.

This module provides:
- SlicedWassersteinDistance: Compute SW distance between sample sets
- WassersteinKernel: RBF kernel on Sliced Wasserstein distance
- ProjectionSampler: Utilities for generating random projections
"""

import jax.numpy as jnp
import jax.random as random
from jax import jit, vmap
from jax.lax import sort
from functools import partial
from typing import Optional, Tuple, Literal
from dataclasses import dataclass
from jaxtyping import Array, Float, PRNGKeyArray


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


@jit
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
    n, m = x_sorted.shape[0], y_sorted.shape[0]
    
    if n == m:
        # Equal sizes: direct pairing
        return jnp.mean(jnp.abs(x_sorted - y_sorted))
    
    # Unequal sizes: interpolate to common grid
    # Use the finer grid (max of n, m points)
    n_points = jnp.maximum(n, m)
    
    # Quantile positions (0 to 1)
    quantiles = jnp.linspace(0, 1, n_points)
    
    # Interpolate both distributions to these quantiles
    x_indices = quantiles * (n - 1)
    y_indices = quantiles * (m - 1)
    
    # Linear interpolation
    x_interp = jnp.interp(x_indices, jnp.arange(n), x_sorted)
    y_interp = jnp.interp(y_indices, jnp.arange(m), y_sorted)
    
    return jnp.mean(jnp.abs(x_interp - y_interp))


@jit
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
    n, m = x_sorted.shape[0], y_sorted.shape[0]
    
    if n == m:
        return jnp.sqrt(jnp.mean((x_sorted - y_sorted) ** 2))
    
    n_points = jnp.maximum(n, m)
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
        projections: Optional pre-computed projections. If None, will be
                    generated on first use based on input dimension.
    
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
    _projections: Optional[Float[Array, "L d"]] = None
    _dimension: Optional[int] = None
    
    def _ensure_projections(self, dimension: int):
        """Initialize projections if needed, or reinitialize if dimension changed."""
        if self._projections is None or self._dimension != dimension:
            key = random.PRNGKey(self.seed)
            # Use object.__setattr__ to modify frozen dataclass
            object.__setattr__(
                self, '_projections',
                sample_unit_sphere(key, self.n_projections, dimension)
            )
            object.__setattr__(self, '_dimension', dimension)
    
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
    
    @partial(jit, static_argnums=(0,))
    def _compute_sw(
        self,
        X: Float[Array, "n d"],
        Y: Float[Array, "m d"],
        projections: Float[Array, "L d"]
    ) -> float:
        """JIT-compiled core computation."""
        
        # Project samples onto each direction
        # X @ projections.T gives shape (n, L) - each column is X projected onto one direction
        X_projected = X @ projections.T  # (n, L)
        Y_projected = Y @ projections.T  # (m, L)
        
        # Sort along sample axis for each projection
        X_sorted = sort(X_projected, axis=0)  # (n, L)
        Y_sorted = sort(Y_projected, axis=0)  # (m, L)
        
        # Compute 1D Wasserstein for each projection
        if self.p == 1:
            # W_1: mean absolute difference
            if X.shape[0] == Y.shape[0]:
                # Equal sizes: vectorized computation
                distances = jnp.mean(jnp.abs(X_sorted - Y_sorted), axis=0)  # (L,)
            else:
                # Unequal sizes: use vmap over projections
                distances = vmap(wasserstein_1d_p1)(X_sorted.T, Y_sorted.T)  # (L,)
            
            return jnp.mean(distances)
        
        else:  # p == 2
            # W_2²: mean squared difference, then sqrt at the end
            if X.shape[0] == Y.shape[0]:
                squared_distances = jnp.mean((X_sorted - Y_sorted) ** 2, axis=0)  # (L,)
            else:
                # For unequal sizes, compute W_2² for each slice
                def w2_squared(x_sort, y_sort):
                    w2 = wasserstein_1d_p2(x_sort, y_sort)
                    return w2 ** 2
                squared_distances = vmap(w2_squared)(X_sorted.T, Y_sorted.T)
            
            # Average W_2², then sqrt to get SW_2
            return jnp.sqrt(jnp.mean(squared_distances))
    
    def pairwise_distances(
        self,
        collections: list  # List[SampleCollection]
    ) -> Float[Array, "N N"]:
        """
        Compute pairwise SW distances between all collections.
        
        Parameters:
            collections: List of SampleCollection objects
        
        Returns:
            Distance matrix of shape (N, N)
        """
        n = len(collections)
        
        # Ensure projections are initialized
        dim = collections[0].samples.shape[1]
        self._ensure_projections(dim)
        
        # Pre-project and sort all collections for efficiency
        projected_sorted = []
        for coll in collections:
            proj = coll.samples @ self._projections.T  # (n_samples, L)
            sorted_proj = sort(proj, axis=0)
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
            if X_sorted.shape[0] == Y_sorted.shape[0]:
                return jnp.mean(jnp.abs(X_sorted - Y_sorted))
            else:
                distances = vmap(wasserstein_1d_p1)(X_sorted.T, Y_sorted.T)
                return jnp.mean(distances)
        else:
            if X_sorted.shape[0] == Y_sorted.shape[0]:
                return jnp.sqrt(jnp.mean((X_sorted - Y_sorted) ** 2))
            else:
                def w2_squared(x, y):
                    return wasserstein_1d_p2(x, y) ** 2
                sq_dist = vmap(w2_squared)(X_sorted.T, Y_sorted.T)
                return jnp.sqrt(jnp.mean(sq_dist))


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
        return jnp.exp(-sw_distance ** 2 / (2 * self.sigma ** 2))
    
    def build_similarity_matrix(
        self,
        collections: list  # List[SampleCollection]
    ) -> Float[Array, "N N"]:
        """
        Build the N×N similarity matrix between all collections.
        
        This is the kernel matrix used for KLR fitting.
        
        Parameters:
            collections: List of SampleCollection objects
        
        Returns:
            Kernel matrix K where K[i,j] = K(collections[i], collections[j])
        """
        # First compute distance matrix
        distances = self._sw.pairwise_distances(collections)
        
        # Convert to similarities via RBF
        return jnp.exp(-distances ** 2 / (2 * self.sigma ** 2))
    
    def compute_new_to_training(
        self,
        new_samples: Float[Array, "m d"],
        training_collections: list  # List[SampleCollection]
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
            sim = self(new_samples, coll.samples)
            similarities.append(sim)
        
        return jnp.array(similarities)


# =============================================================================
# Optimized Batch Operations for Focal Prediction
# =============================================================================

@dataclass
class WassersteinFocalPredictor:
    """
    Optimized focal window predictor using Wasserstein kernel.
    
    Pre-computes projected and sorted training data for efficient
    repeated comparisons during focal window prediction.
    
    Parameters:
        wasserstein_kernel: Configured WassersteinKernel instance
        training_collections: Training data SampleCollections
        klr_alpha: Fitted KLR coefficients
    """
    wasserstein_kernel: WassersteinKernel
    training_collections: list  # List[SampleCollection]
    klr_alpha: Float[Array, "n_train"]
    
    def __post_init__(self):
        """Pre-compute projected and sorted training data."""
        sw = self.wasserstein_kernel._sw
        dim = self.training_collections[0].samples.shape[1]
        sw._ensure_projections(dim)
        
        # Store projections for use during prediction
        self._projections = sw._projections
        self._p = sw.p
        self._sigma = self.wasserstein_kernel.sigma
        
        # Pre-project and sort all training collections
        self._training_projected_sorted = []
        for coll in self.training_collections:
            proj = coll.samples @ self._projections.T
            sorted_proj = sort(proj, axis=0)
            self._training_projected_sorted.append(sorted_proj)
        
        # Stack for vectorized operations where possible
        # Note: This only works if all collections have same n_samples
        # For variable sizes, we keep the list
        sample_sizes = [coll.samples.shape[0] for coll in self.training_collections]
        if len(set(sample_sizes)) == 1:
            self._training_stacked = jnp.stack(self._training_projected_sorted)  # (n_train, n_samples, L)
            self._uniform_samples = True
        else:
            self._training_stacked = None
            self._uniform_samples = False
    
    def predict_window(
        self,
        window_samples: Float[Array, "m d"]
    ) -> float:
        """
        Predict probability for a single focal window.
        
        Parameters:
            window_samples: Feature vectors from the focal window
        
        Returns:
            Predicted probability of site presence
        """
        # Project and sort the window samples
        window_proj = window_samples @ self._projections.T  # (m, L)
        window_sorted = sort(window_proj, axis=0)
        
        # Compute similarity to each training collection
        similarities = self._compute_similarities(window_sorted)
        
        # Apply KLR prediction
        eta = jnp.dot(similarities, self.klr_alpha)
        return 1.0 / (1.0 + jnp.exp(-eta))
    
    def _compute_similarities(
        self,
        window_sorted: Float[Array, "m L"]
    ) -> Float[Array, "n_train"]:
        """Compute similarities between window and all training collections."""
        
        if self._uniform_samples and window_sorted.shape[0] == self._training_stacked.shape[1]:
            # Fully vectorized path when all sizes match
            return self._compute_similarities_vectorized(window_sorted)
        else:
            # Loop path for variable sizes
            return self._compute_similarities_loop(window_sorted)
    
    @partial(jit, static_argnums=(0,))
    def _compute_similarities_vectorized(
        self,
        window_sorted: Float[Array, "m L"]
    ) -> Float[Array, "n_train"]:
        """Vectorized similarity computation when sample sizes are uniform."""
        # training_stacked: (n_train, m, L)
        # window_sorted: (m, L)
        
        if self._p == 2:
            # Compute SW_2 distance to each training collection
            # Squared differences: (n_train, m, L)
            sq_diff = (self._training_stacked - window_sorted[None, :, :]) ** 2
            # Mean over samples, then over projections
            sw_squared = jnp.mean(jnp.mean(sq_diff, axis=1), axis=1)  # (n_train,)
            sw_distances = jnp.sqrt(sw_squared)
        else:
            # SW_1
            abs_diff = jnp.abs(self._training_stacked - window_sorted[None, :, :])
            sw_distances = jnp.mean(jnp.mean(abs_diff, axis=1), axis=1)
        
        # RBF kernel
        return jnp.exp(-sw_distances ** 2 / (2 * self._sigma ** 2))
    
    def _compute_similarities_loop(
        self,
        window_sorted: Float[Array, "m L"]
    ) -> Float[Array, "n_train"]:
        """Loop-based similarity for variable sample sizes."""
        similarities = []
        
        for train_sorted in self._training_projected_sorted:
            if self._p == 2:
                if window_sorted.shape[0] == train_sorted.shape[0]:
                    sw_sq = jnp.mean((window_sorted - train_sorted) ** 2)
                    sw_dist = jnp.sqrt(sw_sq)
                else:
                    # Interpolation needed
                    def w2_sq(w, t):
                        return wasserstein_1d_p2(w, t) ** 2
                    sw_sq = jnp.mean(vmap(w2_sq)(window_sorted.T, train_sorted.T))
                    sw_dist = jnp.sqrt(sw_sq)
            else:
                if window_sorted.shape[0] == train_sorted.shape[0]:
                    sw_dist = jnp.mean(jnp.abs(window_sorted - train_sorted))
                else:
                    sw_dist = jnp.mean(vmap(wasserstein_1d_p1)(window_sorted.T, train_sorted.T))
            
            sim = jnp.exp(-sw_dist ** 2 / (2 * self._sigma ** 2))
            similarities.append(sim)
        
        return jnp.array(similarities)
    
    @partial(jit, static_argnums=(0,))
    def predict_batch(
        self,
        batch_samples: Float[Array, "batch m d"]
    ) -> Float[Array, "batch"]:
        """
        Predict probabilities for a batch of focal windows.
        
        Parameters:
            batch_samples: Batch of window samples, shape (batch_size, window_samples, n_features)
        
        Returns:
            Predicted probabilities for each window
        """
        # Project all windows
        batch_proj = jnp.einsum('bmd,ld->bml', batch_samples, self._projections)  # (batch, m, L)
        
        # Sort each window
        batch_sorted = sort(batch_proj, axis=1)  # (batch, m, L)
        
        # Compute predictions for each window
        return vmap(self._predict_single_sorted)(batch_sorted)
    
    def _predict_single_sorted(
        self,
        window_sorted: Float[Array, "m L"]
    ) -> float:
        """Predict from pre-sorted window projection."""
        similarities = self._compute_similarities(window_sorted)
        eta = jnp.dot(similarities, self.klr_alpha)
        return 1.0 / (1.0 + jnp.exp(-eta))


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
    upper_tri = distance_matrix[jnp.triu_indices_from(distance_matrix, k=1)]
    return jnp.percentile(upper_tri, percentile)


def compare_embeddings_vs_wasserstein(
    collections: list,  # List[SampleCollection]
    sigma_me: float,
    sigma_sw: float,
    n_projections: int = 100,
    seed: int = 42
) -> dict:
    """
    Compare similarity matrices from mean embeddings vs Wasserstein.
    
    Useful for understanding how the two approaches differ on your data.
    
    Parameters:
        collections: List of SampleCollection objects
        sigma_me: Sigma for mean embedding RBF kernel
        sigma_sw: Sigma for Wasserstein RBF kernel
        n_projections: Number of SW projections
        seed: Random seed
    
    Returns:
        Dictionary with both similarity matrices and comparison statistics
    """
    from .rff import RandomFourierFeatures
    from .distribution import MeanEmbeddingKernel
    
    # Mean embedding similarity
    rff = RandomFourierFeatures(sigma=sigma_me, n_features=256, seed=seed)
    me_kernel = MeanEmbeddingKernel(rff)
    K_me = me_kernel.build_similarity_matrix(collections)
    
    # Wasserstein similarity
    sw_kernel = WassersteinKernel(sigma=sigma_sw, n_projections=n_projections, seed=seed)
    K_sw = sw_kernel.build_similarity_matrix(collections)
    
    # Comparison metrics
    correlation = jnp.corrcoef(K_me.ravel(), K_sw.ravel())[0, 1]
    max_diff = jnp.max(jnp.abs(K_me - K_sw))
    mean_diff = jnp.mean(jnp.abs(K_me - K_sw))
    
    return {
        'K_mean_embedding': K_me,
        'K_wasserstein': K_sw,
        'correlation': float(correlation),
        'max_absolute_difference': float(max_diff),
        'mean_absolute_difference': float(mean_diff)
    }
```

---

## Integration with Existing KLRfome API

### Update `klrfome/__init__.py`

Add imports and update the main class to support both kernel types:

```python
from .kernels.wasserstein import (
    SlicedWassersteinDistance,
    WassersteinKernel,
    WassersteinFocalPredictor,
    estimate_sigma_from_distances,
    compare_embeddings_vs_wasserstein
)

@dataclass
class KLRfome:
    """
    High-level interface for KLRfome modeling.
    
    Parameters:
        sigma: Kernel bandwidth
        lambda_reg: KLR regularization strength
        kernel_type: 'mean_embedding' or 'wasserstein'
        n_rff_features: Random Fourier features for mean embedding kernel
        n_projections: Random projections for Wasserstein kernel
        wasserstein_p: Order of Wasserstein distance (1 or 2)
        window_size: Focal window size for prediction
        seed: Random seed for reproducibility
    """
    sigma: float = 1.0
    lambda_reg: float = 0.1
    kernel_type: Literal['mean_embedding', 'wasserstein'] = 'wasserstein'
    n_rff_features: int = 256
    n_projections: int = 100
    wasserstein_p: Literal[1, 2] = 2
    window_size: int = 3
    seed: int = 42
    
    # ... existing fitted attributes ...
    
    def __post_init__(self):
        # Initialize kernel based on type
        if self.kernel_type == 'wasserstein':
            self._distribution_kernel = WassersteinKernel(
                sigma=self.sigma,
                n_projections=self.n_projections,
                p=self.wasserstein_p,
                seed=self.seed
            )
        else:
            if self.n_rff_features > 0:
                base_kernel = RandomFourierFeatures(
                    sigma=self.sigma,
                    n_features=self.n_rff_features,
                    seed=self.seed
                )
            else:
                base_kernel = RBFKernel(sigma=self.sigma)
            self._distribution_kernel = MeanEmbeddingKernel(base_kernel)
        
        self._klr = KernelLogisticRegression(lambda_reg=self.lambda_reg)
    
    def fit(self, training_data: TrainingData) -> 'KLRfome':
        """Fit the KLRfome model."""
        self._training_data = training_data
        
        # Build similarity matrix (method is the same regardless of kernel type)
        self._similarity_matrix = self._distribution_kernel.build_similarity_matrix(
            training_data.collections
        )
        
        # Fit KLR
        self._fit_result = self._klr.fit(
            self._similarity_matrix,
            training_data.labels
        )
        
        return self
    
    def predict(
        self,
        raster_stack: Union[RasterStack, List[str]],
        batch_size: int = 1000,
        show_progress: bool = True
    ) -> jnp.ndarray:
        """Generate predictions across a raster extent."""
        if self._fit_result is None:
            raise RuntimeError("Model must be fit before prediction")
        
        if isinstance(raster_stack, list):
            raster_stack = RasterStack.from_files(raster_stack)
        
        # Use appropriate predictor based on kernel type
        if self.kernel_type == 'wasserstein':
            predictor = WassersteinFocalPredictor(
                wasserstein_kernel=self._distribution_kernel,
                training_collections=self._training_data.collections,
                klr_alpha=self._fit_result.alpha
            )
        else:
            predictor = FocalPredictor(
                distribution_kernel=self._distribution_kernel,
                klr_alpha=self._fit_result.alpha,
                training_data=self._training_data,
                window_size=self.window_size
            )
        
        return predictor.predict_raster(
            raster_stack,
            batch_size=batch_size,
            show_progress=show_progress
        )
    
    def estimate_sigma(
        self,
        training_data: Optional[TrainingData] = None,
        percentile: float = 50.0
    ) -> float:
        """
        Estimate a reasonable sigma based on pairwise distances.
        
        Should be called before fit() to help choose sigma.
        
        Parameters:
            training_data: Data to estimate from (uses fitted data if None)
            percentile: Which percentile of distances to use
        
        Returns:
            Suggested sigma value
        """
        if training_data is None:
            training_data = self._training_data
        if training_data is None:
            raise ValueError("No training data available")
        
        if self.kernel_type == 'wasserstein':
            sw = SlicedWassersteinDistance(
                n_projections=self.n_projections,
                p=self.wasserstein_p,
                seed=self.seed
            )
            distances = sw.pairwise_distances(training_data.collections)
            return estimate_sigma_from_distances(distances, percentile)
        else:
            # For mean embeddings, estimate based on feature-space distances
            # This is less principled but can still be useful
            embeddings = []
            rff = RandomFourierFeatures(sigma=1.0, n_features=self.n_rff_features, seed=self.seed)
            for coll in training_data.collections:
                phi = rff.feature_map(coll.samples)
                embeddings.append(jnp.mean(phi, axis=0))
            embeddings = jnp.stack(embeddings)
            
            # Pairwise Euclidean distances in embedding space
            sq_dists = jnp.sum(embeddings**2, axis=1, keepdims=True) + \
                       jnp.sum(embeddings**2, axis=1) - \
                       2 * embeddings @ embeddings.T
            distances = jnp.sqrt(jnp.maximum(sq_dists, 0))
            return estimate_sigma_from_distances(distances, percentile)
```

---

## Benchmarking and Validation

### Create `benchmarks/compare_kernels.py`

```python
"""
Benchmark comparison between mean embedding and Wasserstein kernels.

Compares:
1. Computational performance
2. Predictive accuracy
3. Sensitivity to distributional shape
"""

import jax.numpy as jnp
import jax.random as random
import time
from typing import Dict, List, Tuple
import numpy as np

from klrfome import KLRfome, TrainingData, SampleCollection
from klrfome.kernels.wasserstein import compare_embeddings_vs_wasserstein
from klrfome.utils.validation import cross_validate


def generate_bimodal_vs_unimodal_test(
    key: random.PRNGKey,
    n_sites: int = 50,
    n_background: int = 100,
    samples_per_location: int = 20,
    n_features: int = 5
) -> Tuple[TrainingData, Dict]:
    """
    Generate synthetic data where sites have bimodal feature distributions
    and background has unimodal distributions.
    
    This is a scenario where Wasserstein should significantly outperform
    mean embeddings, since the distributions have similar means but different shapes.
    
    Returns:
        training_data: TrainingData object
        metadata: Dict with generation parameters
    """
    keys = random.split(key, n_sites + n_background)
    collections = []
    
    # Sites: bimodal (mixture of two Gaussians)
    for i in range(n_sites):
        k1, k2 = random.split(keys[i])
        # Half samples from mode 1, half from mode 2
        mode1 = random.normal(k1, (samples_per_location // 2, n_features)) * 0.5 + jnp.array([-2.0] * n_features)
        mode2 = random.normal(k2, (samples_per_location - samples_per_location // 2, n_features)) * 0.5 + jnp.array([2.0] * n_features)
        samples = jnp.concatenate([mode1, mode2], axis=0)
        
        collections.append(SampleCollection(
            samples=samples,
            label=1,
            id=f"site_{i}"
        ))
    
    # Background: unimodal (single Gaussian at origin - same mean as bimodal!)
    for i in range(n_background):
        samples = random.normal(keys[n_sites + i], (samples_per_location, n_features)) * 1.5
        
        collections.append(SampleCollection(
            samples=samples,
            label=0,
            id=f"background_{i}"
        ))
    
    training_data = TrainingData(
        collections=collections,
        feature_names=[f"var_{i}" for i in range(n_features)]
    )
    
    metadata = {
        'n_sites': n_sites,
        'n_background': n_background,
        'samples_per_location': samples_per_location,
        'n_features': n_features,
        'site_distribution': 'bimodal',
        'background_distribution': 'unimodal',
        'mean_difference': 'none (both centered at origin)'
    }
    
    return training_data, metadata


def benchmark_kernel_comparison(
    training_data: TrainingData,
    sigma_values: List[float] = [0.5, 1.0, 2.0, 5.0],
    n_projections_values: List[int] = [50, 100, 200],
    n_folds: int = 5,
    seed: int = 42
) -> Dict:
    """
    Comprehensive benchmark comparing mean embedding vs Wasserstein kernels.
    
    Returns:
        Dict with results for all configurations
    """
    results = {
        'mean_embedding': [],
        'wasserstein': []
    }
    
    # Mean embedding experiments
    print("Benchmarking Mean Embedding kernel...")
    for sigma in sigma_values:
        model = KLRfome(
            sigma=sigma,
            kernel_type='mean_embedding',
            n_rff_features=256,
            seed=seed
        )
        
        start_time = time.time()
        cv_results = cross_validate(model, training_data, n_folds=n_folds)
        elapsed = time.time() - start_time
        
        results['mean_embedding'].append({
            'sigma': sigma,
            'auc_mean': cv_results['auc_mean'],
            'auc_std': cv_results['auc_std'],
            'accuracy_mean': cv_results['accuracy_mean'],
            'time_seconds': elapsed
        })
        print(f"  sigma={sigma}: AUC={cv_results['auc_mean']:.3f}±{cv_results['auc_std']:.3f}")
    
    # Wasserstein experiments
    print("\nBenchmarking Wasserstein kernel...")
    for sigma in sigma_values:
        for n_proj in n_projections_values:
            model = KLRfome(
                sigma=sigma,
                kernel_type='wasserstein',
                n_projections=n_proj,
                wasserstein_p=2,
                seed=seed
            )
            
            start_time = time.time()
            cv_results = cross_validate(model, training_data, n_folds=n_folds)
            elapsed = time.time() - start_time
            
            results['wasserstein'].append({
                'sigma': sigma,
                'n_projections': n_proj,
                'auc_mean': cv_results['auc_mean'],
                'auc_std': cv_results['auc_std'],
                'accuracy_mean': cv_results['accuracy_mean'],
                'time_seconds': elapsed
            })
            print(f"  sigma={sigma}, n_proj={n_proj}: AUC={cv_results['auc_mean']:.3f}±{cv_results['auc_std']:.3f}")
    
    return results


def analyze_similarity_differences(
    training_data: TrainingData,
    sigma: float = 1.0
) -> Dict:
    """
    Analyze where Wasserstein and mean embedding similarities differ most.
    
    Returns cases where the two approaches disagree, which helps understand
    what distributional features Wasserstein captures that mean embeddings miss.
    """
    comparison = compare_embeddings_vs_wasserstein(
        training_data.collections,
        sigma_me=sigma,
        sigma_sw=sigma,
        n_projections=100
    )
    
    K_me = comparison['K_mean_embedding']
    K_sw = comparison['K_wasserstein']
    diff = K_me - K_sw
    
    # Find pairs with largest disagreement
    n = len(training_data.collections)
    disagreements = []
    
    for i in range(n):
        for j in range(i + 1, n):
            disagreements.append({
                'i': i,
                'j': j,
                'id_i': training_data.collections[i].id,
                'id_j': training_data.collections[j].id,
                'label_i': training_data.collections[i].label,
                'label_j': training_data.collections[j].label,
                'K_me': float(K_me[i, j]),
                'K_sw': float(K_sw[i, j]),
                'difference': float(diff[i, j])
            })
    
    # Sort by absolute difference
    disagreements.sort(key=lambda x: abs(x['difference']), reverse=True)
    
    return {
        'overall_correlation': comparison['correlation'],
        'mean_absolute_difference': comparison['mean_absolute_difference'],
        'top_disagreements': disagreements[:20],
        'K_mean_embedding': K_me,
        'K_wasserstein': K_sw
    }


if __name__ == "__main__":
    # Run benchmark on synthetic data designed to favor Wasserstein
    print("=" * 60)
    print("KLRfome Kernel Comparison Benchmark")
    print("=" * 60)
    
    key = random.PRNGKey(42)
    
    print("\n1. Generating bimodal vs unimodal test data...")
    training_data, metadata = generate_bimodal_vs_unimodal_test(key)
    print(f"   Sites: {metadata['n_sites']} ({metadata['site_distribution']})")
    print(f"   Background: {metadata['n_background']} ({metadata['background_distribution']})")
    
    print("\n2. Running cross-validation benchmarks...")
    results = benchmark_kernel_comparison(
        training_data,
        sigma_values=[0.5, 1.0, 2.0],
        n_projections_values=[50, 100],
        n_folds=5
    )
    
    print("\n3. Analyzing similarity differences...")
    analysis = analyze_similarity_differences(training_data)
    print(f"   Correlation between K_me and K_sw: {analysis['overall_correlation']:.3f}")
    print(f"   Mean absolute difference: {analysis['mean_absolute_difference']:.3f}")
    
    print("\n4. Top disagreements (ME thinks similar, SW thinks different or vice versa):")
    for d in analysis['top_disagreements'][:5]:
        print(f"   {d['id_i']} vs {d['id_j']}: K_me={d['K_me']:.3f}, K_sw={d['K_sw']:.3f}, diff={d['difference']:.3f}")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
```

---

## Hyperparameter Guidance

### Number of Projections (`n_projections`)

| Value | Use Case | Notes |
|-------|----------|-------|
| 50 | Quick experiments, hyperparameter search | May have noticeable variance |
| 100 | Default, good balance | Recommended starting point |
| 200 | Publication-quality results | Diminishing returns beyond this |
| 500 | Maximum precision needed | Rarely necessary |

**Rule of thumb**: Start with 100. If results are noisy across runs with different seeds, increase to 200.

### Wasserstein Order (`p`)

| Value | Behavior | When to Use |
|-------|----------|-------------|
| p=1 | More robust to outliers | Noisy data, heavy-tailed distributions |
| p=2 | Emphasizes large differences | Clean data, Gaussian-like distributions |

**Default**: p=2 is standard and connects to Euclidean geometry. Use p=1 if you have outliers or suspect heavy tails.

### Sigma Selection

The bandwidth σ controls how quickly similarity decays with Wasserstein distance:

- **Too small**: Only nearly-identical distributions get high similarity; model may underfit
- **Too large**: Everything looks similar; model loses discriminative power

**Recommended approach**:
1. Call `model.estimate_sigma(training_data)` to get median-based estimate
2. Try [0.5×, 1×, 2×] this estimate in cross-validation
3. Select based on AUC or your preferred metric

---

## Testing Requirements

Add to `tests/test_wasserstein.py`:

```python
import jax.numpy as jnp
import jax.random as random
import pytest

from klrfome.kernels.wasserstein import (
    SlicedWassersteinDistance,
    WassersteinKernel,
    sample_unit_sphere,
    wasserstein_1d_p1,
    wasserstein_1d_p2
)


class TestUnitSphereSampling:
    def test_vectors_are_unit_length(self):
        key = random.PRNGKey(42)
        vectors = sample_unit_sphere(key, n_projections=100, dimension=10)
        norms = jnp.linalg.norm(vectors, axis=1)
        assert jnp.allclose(norms, 1.0, atol=1e-6)
    
    def test_correct_shape(self):
        key = random.PRNGKey(42)
        vectors = sample_unit_sphere(key, n_projections=50, dimension=5)
        assert vectors.shape == (50, 5)


class TestWasserstein1D:
    def test_identical_distributions(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert wasserstein_1d_p1(x, x) == 0.0
        assert wasserstein_1d_p2(x, x) == 0.0
    
    def test_shifted_distribution(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([1.0, 2.0, 3.0])  # Shifted by 1
        assert jnp.isclose(wasserstein_1d_p1(x, y), 1.0)
        assert jnp.isclose(wasserstein_1d_p2(x, y), 1.0)
    
    def test_symmetric(self):
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        y = jnp.array([0.5, 1.5, 2.5])
        assert jnp.isclose(wasserstein_1d_p1(x, y), wasserstein_1d_p1(y, x))
        assert jnp.isclose(wasserstein_1d_p2(x, y), wasserstein_1d_p2(y, x))


class TestSlicedWasserstein:
    def test_identical_samples_zero_distance(self):
        sw = SlicedWassersteinDistance(n_projections=100)
        X = jnp.array([[0, 0], [1, 1], [2, 2]])
        assert jnp.isclose(sw(X, X), 0.0, atol=1e-6)
    
    def test_symmetric(self):
        sw = SlicedWassersteinDistance(n_projections=100, seed=42)
        key = random.PRNGKey(0)
        X = random.normal(key, (20, 5))
        Y = random.normal(random.PRNGKey(1), (15, 5))
        assert jnp.isclose(sw(X, Y), sw(Y, X), atol=1e-6)
    
    def test_triangle_inequality(self):
        sw = SlicedWassersteinDistance(n_projections=200, seed=42)
        X = random.normal(random.PRNGKey(0), (20, 5))
        Y = random.normal(random.PRNGKey(1), (20, 5))
        Z = random.normal(random.PRNGKey(2), (20, 5))
        
        d_xy = sw(X, Y)
        d_yz = sw(Y, Z)
        d_xz = sw(X, Z)
        
        assert d_xz <= d_xy + d_yz + 1e-6  # Small tolerance for numerical error
    
    def test_detects_distributional_difference(self):
        """Wasserstein should detect bimodal vs unimodal even with same mean."""
        sw = SlicedWassersteinDistance(n_projections=100, seed=42)
        
        # Bimodal: two clusters
        bimodal = jnp.concatenate([
            random.normal(random.PRNGKey(0), (50, 3)) - 2,
            random.normal(random.PRNGKey(1), (50, 3)) + 2
        ])
        
        # Unimodal: single cluster at origin (same mean as bimodal)
        unimodal = random.normal(random.PRNGKey(2), (100, 3)) * 2
        
        # Another unimodal sample
        unimodal2 = random.normal(random.PRNGKey(3), (100, 3)) * 2
        
        # Distance between unimodals should be less than bimodal-unimodal
        d_bimodal_unimodal = sw(bimodal, unimodal)
        d_unimodal_unimodal = sw(unimodal, unimodal2)
        
        assert d_bimodal_unimodal > d_unimodal_unimodal


class TestWassersteinKernel:
    def test_self_similarity_is_one(self):
        kernel = WassersteinKernel(sigma=1.0, n_projections=100)
        X = random.normal(random.PRNGKey(0), (20, 5))
        assert jnp.isclose(kernel(X, X), 1.0, atol=1e-5)
    
    def test_similarity_in_zero_one(self):
        kernel = WassersteinKernel(sigma=1.0, n_projections=100)
        X = random.normal(random.PRNGKey(0), (20, 5))
        Y = random.normal(random.PRNGKey(1), (20, 5)) + 5  # Shifted
        sim = kernel(X, Y)
        assert 0 <= sim <= 1
    
    def test_smaller_sigma_lower_similarity(self):
        X = random.normal(random.PRNGKey(0), (20, 5))
        Y = random.normal(random.PRNGKey(1), (20, 5)) + 2
        
        kernel_small = WassersteinKernel(sigma=0.5, n_projections=100)
        kernel_large = WassersteinKernel(sigma=2.0, n_projections=100)
        
        sim_small = kernel_small(X, Y)
        sim_large = kernel_large(X, Y)
        
        assert sim_small < sim_large
    
    def test_similarity_matrix_symmetric(self):
        from klrfome.data.formats import SampleCollection
        
        collections = [
            SampleCollection(
                samples=random.normal(random.PRNGKey(i), (15, 4)),
                label=i % 2,
                id=f"coll_{i}"
            )
            for i in range(10)
        ]
        
        kernel = WassersteinKernel(sigma=1.0, n_projections=100)
        K = kernel.build_similarity_matrix(collections)
        
        assert jnp.allclose(K, K.T)
    
    def test_similarity_matrix_diagonal_ones(self):
        from klrfome.data.formats import SampleCollection
        
        collections = [
            SampleCollection(
                samples=random.normal(random.PRNGKey(i), (15, 4)),
                label=i % 2,
                id=f"coll_{i}"
            )
            for i in range(10)
        ]
        
        kernel = WassersteinKernel(sigma=1.0, n_projections=100)
        K = kernel.build_similarity_matrix(collections)
        
        assert jnp.allclose(jnp.diag(K), 1.0, atol=1e-5)
```

---

## Implementation Priorities

### Phase 1: Core Wasserstein (This Spec)
1. `wasserstein_1d_p1` and `wasserstein_1d_p2` functions
2. `sample_unit_sphere` for random projections
3. `SlicedWassersteinDistance` class
4. `WassersteinKernel` class
5. Basic tests

### Phase 2: Integration
1. `WassersteinFocalPredictor` with optimized batch prediction
2. Update `KLRfome` class to support `kernel_type='wasserstein'`
3. Integration tests

### Phase 3: Benchmarking
1. `compare_embeddings_vs_wasserstein` utility
2. Synthetic data generators for controlled experiments
3. Benchmark script
4. Documentation of when to use which kernel

### Phase 4: Optimization
1. Profile and optimize hot paths
2. Consider max-sliced Wasserstein variant (uses optimization to find worst-case projection)
3. Investigate tree-sliced Wasserstein if needed

---

## Notes for AI Assistant

1. **JAX sorting**: Use `jax.lax.sort` which is JIT-compatible. Avoid `jnp.sort` in hot paths as it may not fuse optimally.

2. **Numerical stability**: Wasserstein distances can be very small for similar distributions. Ensure no division by zero or log of zero in downstream computations.

3. **Memory**: Unlike RFF which produces a fixed-size embedding, SW requires storing all samples. For very large sample counts, consider subsampling.

4. **Reproducibility**: The random projections should be seeded and deterministic. Store the seed and/or the projections themselves for reproducibility.

5. **Unequal sample sizes**: The interpolation approach for unequal sample sizes is correct but adds overhead. If performance is critical and sample sizes vary wildly, consider resampling to uniform size.

6. **Batching for focal prediction**: The key optimization is pre-projecting and pre-sorting training data, then only projecting/sorting new windows at prediction time.

7. **Testing against known implementations**: Consider validating against `POT` (Python Optimal Transport) library for reference, though we don't want it as a dependency.
