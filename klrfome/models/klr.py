"""Kernel Logistic Regression implementation with IRLS solver."""

import jax.numpy as jnp
from jax.scipy.linalg import solve
from typing import Optional, NamedTuple
from dataclasses import dataclass
import warnings
from jaxtyping import Array, Float


class KLRFitResult(NamedTuple):
    """Result of KLR fitting."""
    alpha: Float[Array, "n"]  # Dual coefficients
    converged: bool
    n_iterations: int
    final_loss: float


@dataclass
class KernelLogisticRegression:
    """
    Kernel Logistic Regression with IRLS solver.
    
    Fits the model:
        P(y=1 | x) = σ(Σ_j α_j K(x, x_j))
    
    where σ is the sigmoid function and K is the kernel.
    
    Uses IRLS formulation: (K + λ·diag(1/W)) α = z
    where W = diag(p * (1 - p)) is the diagonal weight matrix.
    This matches the R implementation.
    
    Parameters:
        lambda_reg: L2 regularization strength
        max_iter: Maximum IRLS iterations
        tol: Convergence tolerance for alpha (default: 0.01 to match R)
        min_prob: Minimum probability to avoid numerical issues (clipping)
    """
    lambda_reg: float = 1.0
    max_iter: int = 100
    tol: float = 0.01  # Match R default tolerance
    min_prob: float = 1e-7
    
    def fit(
        self,
        K: Float[Array, "n n"],
        y: Float[Array, "n"],
        alpha_init: Optional[Float[Array, "n"]] = None
    ) -> KLRFitResult:
        """
        Fit KLR model using IRLS.
        
        Uses the R formulation: (K + λ·diag(1/W)) α = z
        where W = diag(p * (1 - p)) is the diagonal weight matrix.
        
        Parameters:
            K: Precomputed kernel/similarity matrix
            y: Binary labels (0 or 1)
            alpha_init: Initial alpha values (default: uniform 1/N)
        
        Returns:
            KLRFitResult with fitted coefficients and diagnostics
        """
        n = K.shape[0]
        if K.shape[1] != n:
            raise ValueError("K must be square")
        if len(y) != n:
            raise ValueError("y must have same length as K dimensions")
        
        # Match R initialization: rep(1/N, N) instead of zeros
        alpha = alpha_init if alpha_init is not None else jnp.ones(n) / n
        
        for iteration in range(self.max_iter):
            # Compute probabilities - match R exactly: pi <- 1 / (1 + exp(-Kalpha))
            # R uses simple sigmoid WITHOUT any clipping
            eta = K @ alpha
            # Match R exactly: spec <- 1 + exp(-Kalpha); pi <- 1 / spec
            # R does NOT clip probabilities, so we don't either
            prob = 1 / (1 + jnp.exp(-eta))
            # NOTE: R doesn't clip probabilities. We only clip to avoid division by zero
            # when computing W = prob * (1 - prob), not to change the actual prob values
            # Use a much smaller epsilon that won't affect results
            eps = 1e-15  # Machine epsilon level, won't change results
            prob_safe = jnp.clip(prob, eps, 1 - eps)  # Only for W computation
            
            # IRLS weights (diagonal of W) - use prob_safe to avoid division by zero
            # R: diagW <- pi * (1 - pi)
            W = prob_safe * (1 - prob_safe)
            
            # Working response - use original prob for numerator to match R exactly
            # R: z <- Kalpha + ((presence - pi) / diagW)
            z = eta + (y - prob) / W
            
            # Weighted least squares update with regularization
            # R formulation: (K + λ·diag(1/W)) α = z
            # This matches the R implementation which uses weighted ridge regression
            diagW_inv = 1.0 / W  # 1/diagW (inverse of diagonal weights)
            lhs = K + self.lambda_reg * jnp.diag(diagW_inv)
            rhs = z
            
            try:
                alpha_new = solve(lhs, rhs, assume_a='pos')
            except Exception as e:
                warnings.warn(f"Error solving linear system at iteration {iteration}: {e}")
                loss = self._compute_loss(K, y, alpha)
                return KLRFitResult(alpha, False, iteration, loss)
            
            # Check for NaN (matching R's error handling)
            if jnp.any(jnp.isnan(alpha_new)):
                warnings.warn(f"NaN detected in alpha at iteration {iteration}")
                loss = self._compute_loss(K, y, alpha)
                return KLRFitResult(alpha, False, iteration, loss)
            
            # Check convergence - match R: all(abs(alphan - alpha) <= tol)
            # R uses: all(abs(alphan - alpha) <= tol), not max
            converged = jnp.all(jnp.abs(alpha_new - alpha) <= self.tol)
            alpha = alpha_new
            
            if converged:
                loss = self._compute_loss(K, y, alpha)
                return KLRFitResult(alpha, True, iteration + 1, loss)
        
        warnings.warn(f"KLR did not converge in {self.max_iter} iterations")
        loss = self._compute_loss(K, y, alpha)
        return KLRFitResult(alpha, False, self.max_iter, loss)
    
    def predict_proba(
        self,
        K_new: Float[Array, "m n"],
        alpha: Float[Array, "n"]
    ) -> Float[Array, "m"]:
        """
        Predict probabilities for new data.
        
        Uses simple sigmoid to match R: pred <- 1 / (1 + exp(-eta))
        
        Parameters:
            K_new: Kernel matrix between new points and training points
                   Shape: (n_new, n_train)
            alpha: Fitted dual coefficients
        
        Returns:
            Predicted probabilities of class 1
        """
        eta = K_new @ alpha
        # Match R exactly: 1 / (1 + exp(-eta))
        # R uses simple sigmoid, not numerically stable version
        return 1 / (1 + jnp.exp(-eta))
    
    def predict(
        self,
        K_new: Float[Array, "m n"],
        alpha: Float[Array, "n"],
        threshold: float = 0.5
    ) -> Float[Array, "m"]:
        """
        Predict binary labels.
        
        Parameters:
            K_new: Kernel matrix between new points and training points
            alpha: Fitted dual coefficients
            threshold: Probability threshold for classification
        
        Returns:
            Binary predictions (0 or 1)
        """
        proba = self.predict_proba(K_new, alpha)
        return (proba >= threshold).astype(jnp.int32)
    
    @staticmethod
    def _sigmoid(x: Float[Array, "..."]) -> Float[Array, "..."]:
        """
        Numerically stable sigmoid.
        
        Parameters:
            x: Input values
        
        Returns:
            Sigmoid values
        """
        return jnp.where(
            x >= 0,
            1 / (1 + jnp.exp(-x)),
            jnp.exp(x) / (1 + jnp.exp(x))
        )
    
    def _compute_loss(
        self,
        K: Float[Array, "n n"],
        y: Float[Array, "n"],
        alpha: Float[Array, "n"]
    ) -> float:
        """
        Compute regularized negative log-likelihood.
        
        Parameters:
            K: Kernel matrix
            y: Labels
            alpha: Coefficients
        
        Returns:
            Loss value
        """
        prob = self.predict_proba(K, alpha)
        prob = jnp.clip(prob, self.min_prob, 1 - self.min_prob)
        
        # Negative log-likelihood
        nll = -jnp.mean(y * jnp.log(prob) + (1 - y) * jnp.log(1 - prob))
        # Regularization
        reg = 0.5 * self.lambda_reg * alpha @ K @ alpha
        
        return float(nll + reg)

