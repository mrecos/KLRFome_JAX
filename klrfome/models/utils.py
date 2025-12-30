"""Utility functions for KLR models."""

import jax.numpy as jnp
from jaxtyping import Array, Float


def check_convergence(
    alpha_old: Float[Array, "n"],
    alpha_new: Float[Array, "n"],
    tol: float = 1e-6
) -> bool:
    """
    Check if IRLS has converged.
    
    Parameters:
        alpha_old: Previous alpha values
        alpha_new: New alpha values
        tol: Convergence tolerance
    
    Returns:
        True if converged
    """
    delta = jnp.max(jnp.abs(alpha_new - alpha_old))
    return delta < tol


def compute_irls_weights(
    prob: Float[Array, "n"]
) -> Float[Array, "n"]:
    """
    Compute IRLS weights from probabilities.
    
    Parameters:
        prob: Predicted probabilities
    
    Returns:
        Weights (diagonal of W matrix)
    """
    return prob * (1 - prob)
