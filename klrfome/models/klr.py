"""Numerically stable dual kernel logistic regression."""

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence, Tuple
import warnings

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve
from jaxtyping import Array, Float


class KLRFitResult(NamedTuple):
    """Fitted dual coefficients and explicit solver diagnostics."""

    alpha: Float[Array, "n"]
    converged: bool
    n_iterations: int
    final_loss: float
    failure_reason: Optional[str] = None
    jitter_used: float = 0.0


@dataclass
class KernelLogisticRegression:
    """Dual KLR fitted with the IRLS formulation used by the original R model.

    The update solves ``(K + lambda * diag(1 / W)) alpha = z``.  Stable
    sigmoid/cross-entropy calculations and bounded diagonal jitter make failure
    observable without silently returning NaNs.
    """

    lambda_reg: float = 1.0
    max_iter: int = 100
    tol: float = 0.01
    min_prob: float = 1e-7
    jitter_schedule: Sequence[float] = (0.0, 1e-8, 1e-6, 1e-4)

    def __post_init__(self) -> None:
        if self.lambda_reg <= 0:
            raise ValueError("lambda_reg must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if not 0 < self.min_prob < 0.5:
            raise ValueError("min_prob must be between 0 and 0.5")
        if not self.jitter_schedule or any(value < 0 for value in self.jitter_schedule):
            raise ValueError("jitter_schedule must contain nonnegative values")

    def fit(
        self,
        K: Float[Array, "n n"],
        y: Float[Array, "n"],
        alpha_init: Optional[Float[Array, "n"]] = None,
    ) -> KLRFitResult:
        """Fit a precomputed, symmetric bag-level Gram matrix."""
        K = jnp.asarray(K)
        y = jnp.asarray(y, dtype=K.dtype)
        n = K.shape[0]
        if K.ndim != 2 or K.shape[1] != n:
            raise ValueError("K must be square")
        if y.ndim != 1 or len(y) != n:
            raise ValueError("y must have the same length as K")
        if n == 0:
            raise ValueError("K and y must be nonempty")
        if not np.isfinite(np.asarray(K)).all() or not np.isfinite(np.asarray(y)).all():
            raise ValueError("K and y must be finite")
        if not np.isin(np.asarray(y), (0, 1)).all():
            raise ValueError("y must contain only 0 and 1")

        alpha = (
            jnp.asarray(alpha_init, dtype=K.dtype)
            if alpha_init is not None
            else jnp.ones(n, dtype=K.dtype) / n
        )
        if alpha.shape != (n,) or not np.isfinite(np.asarray(alpha)).all():
            raise ValueError("alpha_init must be a finite vector of length n")

        probability_floor = self._effective_min_prob(K.dtype)
        maximum_jitter = 0.0

        for iteration in range(self.max_iter):
            eta = K @ alpha
            probability = self._sigmoid(eta)
            probability_safe = jnp.clip(probability, probability_floor, 1.0 - probability_floor)
            weights = probability_safe * (1.0 - probability_safe)
            z = eta + (y - probability) / weights

            if not self._all_finite(eta, probability, weights, z):
                return self._failure(
                    K,
                    y,
                    alpha,
                    iteration,
                    "nonfinite_irls_state",
                    maximum_jitter,
                )

            lhs = K + self.lambda_reg * jnp.diag(1.0 / weights)
            alpha_new, jitter_used = self._solve_with_jitter(lhs, z)
            maximum_jitter = max(maximum_jitter, jitter_used)
            if alpha_new is None:
                return self._failure(
                    K,
                    y,
                    alpha,
                    iteration,
                    "linear_solve_failed",
                    maximum_jitter,
                )

            converged = bool(jnp.all(jnp.abs(alpha_new - alpha) <= self.tol))
            alpha = alpha_new
            if converged:
                loss = self._compute_loss(K, y, alpha)
                if not np.isfinite(loss):
                    return self._failure(
                        K,
                        y,
                        alpha,
                        iteration + 1,
                        "nonfinite_loss",
                        maximum_jitter,
                    )
                return KLRFitResult(alpha, True, iteration + 1, loss, None, maximum_jitter)

        warnings.warn(f"KLR did not converge in {self.max_iter} iterations")
        return KLRFitResult(
            alpha,
            False,
            self.max_iter,
            self._compute_loss(K, y, alpha),
            "maximum_iterations",
            maximum_jitter,
        )

    def _solve_with_jitter(
        self, lhs: Float[Array, "n n"], rhs: Float[Array, "n"]
    ) -> Tuple[Optional[Float[Array, "n"]], float]:
        scale = max(float(jnp.mean(jnp.abs(jnp.diag(lhs)))), 1.0)
        identity = jnp.eye(lhs.shape[0], dtype=lhs.dtype)
        for relative_jitter in self.jitter_schedule:
            absolute_jitter = float(relative_jitter) * scale
            try:
                candidate = solve(
                    lhs + absolute_jitter * identity,
                    rhs,
                    assume_a="pos",
                )
            except Exception:
                continue
            if not np.isfinite(np.asarray(candidate)).all():
                continue
            residual = (lhs + absolute_jitter * identity) @ candidate - rhs
            denominator = max(float(jnp.linalg.norm(rhs)), 1.0)
            relative_residual = float(jnp.linalg.norm(residual)) / denominator
            if np.isfinite(relative_residual) and relative_residual <= 5e-3:
                return candidate, absolute_jitter
        return None, float(self.jitter_schedule[-1]) * scale

    def predict_proba(
        self, K_new: Float[Array, "m n"], alpha: Float[Array, "n"]
    ) -> Float[Array, "m"]:
        """Predict class-one scores using a stable sigmoid."""
        eta = jnp.asarray(K_new) @ jnp.asarray(alpha)
        return self._sigmoid(eta)

    def predict(
        self,
        K_new: Float[Array, "m n"],
        alpha: Float[Array, "n"],
        threshold: float = 0.5,
    ) -> Float[Array, "m"]:
        return (self.predict_proba(K_new, alpha) >= threshold).astype(jnp.int32)

    @staticmethod
    def _sigmoid(x: Float[Array, "..."]) -> Float[Array, "..."]:
        return jnp.asarray(jnp.where(x >= 0, 1 / (1 + jnp.exp(-x)), jnp.exp(x) / (1 + jnp.exp(x))))

    def _effective_min_prob(self, dtype: jnp.dtype) -> float:
        return max(float(self.min_prob), float(jnp.finfo(dtype).eps))

    @staticmethod
    def _all_finite(*arrays: Array) -> bool:
        return all(np.isfinite(np.asarray(array)).all() for array in arrays)

    def _failure(
        self,
        K: Float[Array, "n n"],
        y: Float[Array, "n"],
        alpha: Float[Array, "n"],
        iteration: int,
        reason: str,
        jitter_used: float,
    ) -> KLRFitResult:
        warnings.warn(f"KLR stopped at iteration {iteration}: {reason}")
        return KLRFitResult(
            alpha,
            False,
            iteration,
            self._compute_loss(K, y, alpha),
            reason,
            jitter_used,
        )

    def _compute_loss(
        self,
        K: Float[Array, "n n"],
        y: Float[Array, "n"],
        alpha: Float[Array, "n"],
    ) -> float:
        eta = K @ alpha
        nll = jnp.mean(jnp.logaddexp(0.0, eta) - y * eta)
        regularization = 0.5 * self.lambda_reg * alpha @ K @ alpha
        return float(nll + regularization)
