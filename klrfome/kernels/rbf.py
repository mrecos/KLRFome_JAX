"""Radial Basis Function (RBF) kernel implementation."""

import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import TYPE_CHECKING
from jaxtyping import Array, Float

if TYPE_CHECKING:
    from .base import Kernel


class RBFKernel:
    """
    Radial Basis Function (Gaussian) kernel.
    
    k(x, y) = exp(-||x - y||² / (2σ²))
    
    Parameters:
        sigma: Bandwidth parameter (length scale)
    """
    
    def __init__(self, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self._sigma = sigma
    
    @property
    def sigma(self) -> float:
        """Kernel bandwidth parameter."""
        return self._sigma
    
    @partial(jit, static_argnums=(0,))
    def __call__(
        self, 
        X: Float[Array, "n d"], 
        Y: Float[Array, "m d"]
    ) -> Float[Array, "n m"]:
        """
        Compute RBF kernel matrix.
        
        Uses the identity:
        ||x - y||² = ||x||² + ||y||² - 2<x, y>
        
        Parameters:
            X: First set of points, shape (n, d)
            Y: Second set of points, shape (m, d)
        
        Returns:
            Kernel matrix of shape (n, m)
        """
        # Compute squared norms
        X_sqnorm = jnp.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
        Y_sqnorm = jnp.sum(Y ** 2, axis=1, keepdims=True)  # (m, 1)
        
        # Compute squared distances using identity
        sq_distances = X_sqnorm + Y_sqnorm.T - 2 * jnp.dot(X, Y.T)  # (n, m)
        
        # Apply RBF kernel
        return jnp.exp(-sq_distances / (2 * self._sigma ** 2))
    
    @partial(jit, static_argnums=(0,))
    def diagonal(self, X: Float[Array, "n d"]) -> Float[Array, "n"]:
        """
        Diagonal of K(X, X) - always 1 for RBF.
        
        Parameters:
            X: Input points, shape (n, d)
        
        Returns:
            Diagonal values, shape (n,)
        """
        return jnp.ones(X.shape[0])

