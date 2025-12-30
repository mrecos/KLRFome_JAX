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
    
    Parameters:
        lambda_reg: L2 regularization strength
        max_iter: Maximum IRLS iterations
        tol: Convergence tolerance for alpha
        min_prob: Minimum probability to avoid numerical issues (clipping)
    """
    lambda_reg: float = 1.0
    max_iter: int = 100
    tol: float = 1e-6
    min_prob: float = 1e-7
    
    def fit(
        self,
        K: Float[Array, "n n"],
        y: Float[Array, "n"],
        alpha_init: Optional[Float[Array, "n"]] = None
    ) -> KLRFitResult:
        """
        Fit KLR model using IRLS.
        
        Parameters:
            K: Precomputed kernel/similarity matrix
            y: Binary labels (0 or 1)
            alpha_init: Initial alpha values (default: zeros)
        
        Returns:
            KLRFitResult with fitted coefficients and diagnostics
        """
        n = K.shape[0]
        if K.shape[1] != n:
            raise ValueError("K must be square")
        if len(y) != n:
            raise ValueError("y must have same length as K dimensions")
        
        alpha = alpha_init if alpha_init is not None else jnp.zeros(n)
        
        for iteration in range(self.max_iter):
            # Compute probabilities
            eta = K @ alpha
            prob = self._sigmoid(eta)
            prob = jnp.clip(prob, self.min_prob, 1 - self.min_prob)
            
            # IRLS weights (diagonal of W)
            W = prob * (1 - prob)
            
            # Working response
            # z = η + (y - p) / (p * (1 - p))
            z = eta + (y - prob) / W
            
            # Weighted least squares update with regularization
            # Solve: (K W K + λI) α = K W z
            # Note: W is diagonal, so K W = K * W[None, :] (broadcasting)
            KW = K * W[None, :]  # Broadcasting for diagonal W
            lhs = KW @ K + self.lambda_reg * jnp.eye(n)
            rhs = KW @ z
            
            try:
                alpha_new = solve(lhs, rhs, assume_a='pos')
            except Exception as e:
                warnings.warn(f"Error solving linear system at iteration {iteration}: {e}")
                loss = self._compute_loss(K, y, alpha)
                return KLRFitResult(alpha, False, iteration, loss)
            
            # Check convergence
            delta = jnp.max(jnp.abs(alpha_new - alpha))
            alpha = alpha_new
            
            if delta < self.tol:
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
        
        Parameters:
            K_new: Kernel matrix between new points and training points
                   Shape: (n_new, n_train)
            alpha: Fitted dual coefficients
        
        Returns:
            Predicted probabilities of class 1
        """
        eta = K_new @ alpha
        return self._sigmoid(eta)
    
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

