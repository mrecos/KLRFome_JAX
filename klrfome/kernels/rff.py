"""Random Fourier Features (RFF) approximation for RBF kernel."""

import jax.numpy as jnp
import jax.random as random
from jax import jit
from functools import partial
from typing import Optional
from jaxtyping import Array, Float


class RandomFourierFeatures:
    """
    Random Fourier Features approximation to RBF kernel.
    
    Approximates: k(x, y) ≈ φ(x)ᵀφ(y)
    
    Where φ(x) = sqrt(2/D) * cos(Wx + b)
    
    This transforms the kernel computation from O(n²) to O(nD) where
    D is the number of random features.
    
    Reference:
        Rahimi & Recht (2007). "Random Features for Large-Scale Kernel Machines"
    
    Parameters:
        sigma: Bandwidth of the RBF kernel to approximate
        n_features: Number of random features (D). Higher = better approximation.
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self, 
        sigma: float = 1.0, 
        n_features: int = 256,
        seed: int = 42
    ):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        
        self._sigma = sigma
        self._n_features = n_features
        self._seed = seed
        self._W: Optional[Float[Array, "d D"]] = None  # Lazily initialized
        self._b: Optional[Float[Array, "D"]] = None
        self._input_dim: Optional[int] = None
    
    @property
    def sigma(self) -> float:
        """Kernel bandwidth parameter."""
        return self._sigma
    
    @property
    def n_features(self) -> int:
        """Number of random features."""
        return self._n_features
    
    def _initialize_weights(self, input_dim: int):
        """Initialize random weights for the feature map."""
        if self._W is not None and self._input_dim == input_dim:
            return
        
        key = random.PRNGKey(self._seed)
        key_W, key_b = random.split(key)
        
        # W ~ N(0, 1/σ²) for RBF kernel
        # For RBF, we need W ~ N(0, I/σ²) where I is identity
        self._W = random.normal(key_W, (input_dim, self._n_features)) / self._sigma
        # b ~ Uniform(0, 2π)
        self._b = random.uniform(
            key_b, 
            (self._n_features,), 
            minval=0, 
            maxval=2 * jnp.pi
        )
        self._input_dim = input_dim
    
    @partial(jit, static_argnums=(0,))
    def feature_map(
        self, 
        X: Float[Array, "n d"]
    ) -> Float[Array, "n D"]:
        """
        Compute random Fourier features.
        
        φ(x) = sqrt(2/D) * cos(Wx + b)
        
        Parameters:
            X: Input points, shape (n, d)
        
        Returns:
            Feature map of shape (n, D)
        """
        if self._W is None or self._input_dim != X.shape[1]:
            # This will fail in JIT, so we initialize before JIT
            # In practice, call _initialize_weights first
            raise RuntimeError(
                "Weights not initialized. Call _initialize_weights() first, "
                "or use __call__ which handles initialization."
            )
        
        # Project: Wx + b
        projection = jnp.dot(X, self._W) + self._b  # (n, D)
        
        # Apply cosine and scale
        return jnp.sqrt(2.0 / self._n_features) * jnp.cos(projection)
    
    def __call__(
        self, 
        X: Float[Array, "n d"], 
        Y: Float[Array, "m d"]
    ) -> Float[Array, "n m"]:
        """
        Approximate kernel matrix via random features.
        
        K(X, Y) ≈ φ(X) @ φ(Y).T
        
        Parameters:
            X: First set of points, shape (n, d)
            Y: Second set of points, shape (m, d)
        
        Returns:
            Approximate kernel matrix of shape (n, m)
        """
        # Initialize weights if needed
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have same number of features")
        
        self._initialize_weights(X.shape[1])
        
        # Compute feature maps
        phi_X = self.feature_map(X)
        phi_Y = self.feature_map(Y)
        
        # Kernel matrix is dot product of feature maps
        return jnp.dot(phi_X, phi_Y.T)
    
    def self_similarity(
        self, 
        X: Float[Array, "n d"]
    ) -> Float[Array, "n n"]:
        """
        Compute K(X, X) efficiently.
        
        Parameters:
            X: Input points, shape (n, d)
        
        Returns:
            Self-similarity matrix of shape (n, n)
        """
        self._initialize_weights(X.shape[1])
        phi_X = self.feature_map(X)
        return jnp.dot(phi_X, phi_X.T)

