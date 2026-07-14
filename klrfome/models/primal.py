"""Primal regularized logistic regression for explicit bag embeddings."""

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence, Tuple
import warnings

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve
from jaxtyping import Array, Float


class PrimalFitResult(NamedTuple):
    coefficients: Float[Array, "d"]
    converged: bool
    n_iterations: int
    final_loss: float
    failure_reason: Optional[str] = None
    jitter_used: float = 0.0


@dataclass
class PrimalLogisticRegression:
    """L2-regularized logistic regression without constructing an N-by-N Gram."""

    lambda_reg: float = 1.0
    max_iter: int = 100
    tol: float = 1e-6
    min_prob: float = 1e-7
    jitter_schedule: Sequence[float] = (0.0, 1e-8, 1e-6, 1e-4)

    def fit(
        self,
        features: Float[Array, "n d"],
        y: Float[Array, "n"],
        coefficients_init: Optional[Float[Array, "d"]] = None,
    ) -> PrimalFitResult:
        features = jnp.asarray(features)
        y = jnp.asarray(y, dtype=features.dtype)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("features must be a nonempty 2D matrix")
        if y.shape != (features.shape[0],):
            raise ValueError("y must have one value per feature row")
        if not np.isfinite(np.asarray(features)).all() or not np.isfinite(np.asarray(y)).all():
            raise ValueError("features and y must be finite")
        if not np.isin(np.asarray(y), (0, 1)).all():
            raise ValueError("y must contain only 0 and 1")
        if self.lambda_reg <= 0:
            raise ValueError("lambda_reg must be positive")

        d = features.shape[1]
        beta = (
            jnp.asarray(coefficients_init, dtype=features.dtype)
            if coefficients_init is not None
            else jnp.zeros(d, dtype=features.dtype)
        )
        if beta.shape != (d,):
            raise ValueError("coefficients_init must match the feature dimension")

        floor = max(float(self.min_prob), float(jnp.finfo(features.dtype).eps))
        identity = jnp.eye(d, dtype=features.dtype)
        maximum_jitter = 0.0

        for iteration in range(self.max_iter):
            eta = features @ beta
            probability = jnp.where(
                eta >= 0, 1 / (1 + jnp.exp(-eta)), jnp.exp(eta) / (1 + jnp.exp(eta))
            )
            safe_probability = jnp.clip(probability, floor, 1.0 - floor)
            weights = safe_probability * (1.0 - safe_probability)
            gradient = features.T @ (probability - y) + self.lambda_reg * beta
            hessian = (features.T * weights) @ features + self.lambda_reg * identity

            step, jitter = self._solve(hessian, gradient)
            maximum_jitter = max(maximum_jitter, jitter)
            if step is None:
                return self._result(
                    beta, False, iteration, "linear_solve_failed", maximum_jitter, features, y
                )
            beta_new = beta - step
            if not np.isfinite(np.asarray(beta_new)).all():
                return self._result(
                    beta, False, iteration, "nonfinite_iteration", maximum_jitter, features, y
                )
            converged = float(jnp.max(jnp.abs(beta_new - beta))) <= self.tol
            beta = beta_new
            if converged:
                return self._result(beta, True, iteration + 1, None, maximum_jitter, features, y)

        warnings.warn(f"Primal logistic regression did not converge in {self.max_iter} iterations")
        return self._result(
            beta, False, self.max_iter, "maximum_iterations", maximum_jitter, features, y
        )

    def _solve(
        self, matrix: Float[Array, "d d"], rhs: Float[Array, "d"]
    ) -> Tuple[Optional[Float[Array, "d"]], float]:
        scale = max(float(jnp.mean(jnp.abs(jnp.diag(matrix)))), 1.0)
        identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
        for relative_jitter in self.jitter_schedule:
            jitter = float(relative_jitter) * scale
            try:
                candidate = solve(matrix + jitter * identity, rhs, assume_a="pos")
            except Exception:
                continue
            if np.isfinite(np.asarray(candidate)).all():
                return candidate, jitter
        return None, float(self.jitter_schedule[-1]) * scale

    def _result(
        self,
        beta: Float[Array, "d"],
        converged: bool,
        iterations: int,
        failure: Optional[str],
        jitter: float,
        features: Float[Array, "n d"],
        y: Float[Array, "n"],
    ) -> PrimalFitResult:
        eta = features @ beta
        loss = jnp.sum(jnp.logaddexp(0.0, eta) - y * eta)
        loss += 0.5 * self.lambda_reg * jnp.dot(beta, beta)
        return PrimalFitResult(beta, converged, iterations, float(loss), failure, jitter)

    @staticmethod
    def predict_proba(
        features: Float[Array, "n d"], coefficients: Float[Array, "d"]
    ) -> Float[Array, "n"]:
        eta = jnp.asarray(features) @ jnp.asarray(coefficients)
        return jnp.where(eta >= 0, 1 / (1 + jnp.exp(-eta)), jnp.exp(eta) / (1 + jnp.exp(eta)))
