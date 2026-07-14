"""Unified M0--M3 estimators operating on canonical bag datasets."""

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np
from jax import jit
from jaxtyping import Array, Float
from numpy.typing import NDArray

from ..data.formats import Bag, BagDataset
from ..data.preprocessing import BagStandardizer
from ..kernels.distribution import MeanEmbeddingKernel
from ..kernels.rbf import RBFKernel
from ..kernels.rff import RandomFourierFeatures
from ..kernels.wasserstein import SlicedWassersteinDistance
from .klr import KLRFitResult, KernelLogisticRegression
from .primal import PrimalFitResult, PrimalLogisticRegression
from .spec import ModelSpec

FitResult = Union[KLRFitResult, PrimalFitResult]


@partial(jit, static_argnames=("round_kernel",))
def _exact_kme_block(
    left: Float[Array, "batch max_left d"],
    left_mask: Float[Array, "batch max_left"],
    right: Float[Array, "n_right max_right d"],
    right_mask: Float[Array, "n_right max_right"],
    sigma: float,
    round_kernel: bool,
) -> Float[Array, "batch n_right"]:
    """Fixed-shape exact KME block; one compilation replaces per-bag-size compiles."""
    left_norm = jnp.sum(left**2, axis=2)[:, None, :, None]
    right_norm = jnp.sum(right**2, axis=2)[None, :, None, :]
    cross = jnp.einsum("bmd,rnd->brmn", left, right)
    squared_distance = jnp.maximum(left_norm + right_norm - 2.0 * cross, 0.0)
    cell_kernel = jnp.exp(-squared_distance / (2.0 * sigma**2))
    pair_mask = left_mask[:, None, :, None] * right_mask[None, :, None, :]
    numerator = jnp.sum(cell_kernel * pair_mask, axis=(2, 3))
    denominator = jnp.sum(left_mask, axis=1)[:, None] * jnp.sum(right_mask, axis=1)[None, :]
    values = numerator / jnp.maximum(denominator, 1.0)
    return jnp.round(values, 3) if round_kernel else values


@dataclass
class DistributionClassifier:
    """Fit one validated distribution-regression architecture.

    ``sigma`` is the point-level RBF bandwidth for M0--M2. ``decision_sigma``
    is the bag-level RBF bandwidth for M2--M3. Each can be estimated using only
    the data passed to :meth:`fit`.
    """

    spec: ModelSpec
    sigma: float = 1.0
    decision_sigma: Optional[float] = None
    lambda_reg: float = 0.1
    scale_features: bool = True
    auto_sigma: bool = True
    seed: int = 42
    round_exact_kernel: bool = True
    max_iter: int = 100
    tol: float = 1e-3
    exact_batch_size: int = 2

    preprocessor_: Optional[BagStandardizer] = field(default=None, init=False)
    fit_result_: Optional[FitResult] = field(default=None, init=False)
    training_data_: Optional[BagDataset] = field(default=None, init=False)
    gram_matrix_: Optional[Float[Array, "n n"]] = field(default=None, init=False)
    training_embeddings_: Optional[Float[Array, "n d"]] = field(default=None, init=False)
    point_sigma_: Optional[float] = field(default=None, init=False)
    decision_sigma_: Optional[float] = field(default=None, init=False)
    diagnostics_: Dict[str, object] = field(default_factory=dict, init=False)
    _rff: Optional[RandomFourierFeatures] = field(default=None, init=False, repr=False)
    _mean_kernel: Optional[MeanEmbeddingKernel] = field(default=None, init=False, repr=False)
    _sw: Optional[SlicedWassersteinDistance] = field(default=None, init=False, repr=False)

    def clone(self) -> "DistributionClassifier":
        """Return an unfitted estimator with the same configuration."""
        return replace(self)

    def fit(self, dataset: BagDataset) -> "DistributionClassifier":
        self._reset_fit_state()
        if self.scale_features:
            self.preprocessor_ = BagStandardizer.fit(dataset)
            fitted_data = self.preprocessor_.transform(dataset)
        else:
            fitted_data = dataset.subset(range(dataset.n_locations))
        self.training_data_ = fitted_data
        labels = fitted_data.labels

        if self.spec.representation in ("exact_kme", "rff_kme"):
            self.point_sigma_ = (
                self._median_cell_distance(fitted_data) if self.auto_sigma else float(self.sigma)
            )

        if self.spec.representation == "exact_kme":
            assert self.point_sigma_ is not None
            self._mean_kernel = MeanEmbeddingKernel(RBFKernel(self.point_sigma_))
            gram = self._exact_kme_matrix(
                fitted_data.collections,
                fitted_data.collections,
            )
            gram = (gram + gram.T) / 2.0
            self._fit_dual(gram, labels)
        elif self.spec.representation == "rff_kme":
            assert self.point_sigma_ is not None
            self._rff = RandomFourierFeatures(
                sigma=self.point_sigma_,
                n_features=self.spec.rff_features,
                seed=self.seed,
            )
            embeddings = self._embed_rff(fitted_data.collections)
            self.training_embeddings_ = embeddings
            if self.spec.solver == "primal_logistic":
                solver = PrimalLogisticRegression(
                    lambda_reg=self.lambda_reg, max_iter=self.max_iter, tol=self.tol
                )
                self.fit_result_ = solver.fit(embeddings, labels)
                self.gram_matrix_ = None
            else:
                self.decision_sigma_ = self._resolve_decision_sigma(embeddings)
                gram = self._rbf_matrix(embeddings, embeddings, self.decision_sigma_)
                self._fit_dual(gram, labels)
        else:
            self._sw = SlicedWassersteinDistance(
                n_projections=self.spec.n_projections,
                p=2,
                seed=self.seed,
            )
            distances = self._sw.pairwise_distances_quantile(
                fitted_data.collections, self.spec.n_quantiles
            )
            self.decision_sigma_ = self._resolve_distance_sigma(distances)
            gram = jnp.exp(-(distances**2) / (2.0 * self.decision_sigma_**2))
            self._fit_dual(gram, labels)

        if self.fit_result_ is None:
            raise RuntimeError("Internal error: fit did not produce a result")
        self.diagnostics_ = {
            "method_id": self.spec.method_id,
            "converged": self.fit_result_.converged,
            "iterations": self.fit_result_.n_iterations,
            "failure_reason": self.fit_result_.failure_reason,
            "jitter_used": self.fit_result_.jitter_used,
            "point_sigma": self.point_sigma_,
            "decision_sigma": self.decision_sigma_,
            "constructed_gram_matrix": self.gram_matrix_ is not None,
        }
        return self

    def predict_bags(self, dataset: BagDataset) -> Float[Array, "n"]:
        if self.training_data_ is None or self.fit_result_ is None:
            raise RuntimeError("Model must be fit before prediction")
        prediction_data = self.preprocessor_.transform(dataset) if self.preprocessor_ else dataset

        if self.spec.representation == "exact_kme":
            cross = self._exact_kme_matrix(
                prediction_data.collections,
                self.training_data_.collections,
            )
            assert isinstance(self.fit_result_, KLRFitResult)
            return KernelLogisticRegression.predict_proba(
                KernelLogisticRegression(), cross, self.fit_result_.alpha
            )

        if self.spec.representation == "rff_kme":
            embeddings = self._embed_rff(prediction_data.collections)
            if self.spec.solver == "primal_logistic":
                assert isinstance(self.fit_result_, PrimalFitResult)
                return PrimalLogisticRegression.predict_proba(
                    embeddings, self.fit_result_.coefficients
                )
            assert self.training_embeddings_ is not None and self.decision_sigma_ is not None
            cross = self._rbf_matrix(embeddings, self.training_embeddings_, self.decision_sigma_)
        else:
            assert self._sw is not None and self.decision_sigma_ is not None
            distances = self._sw.cross_distances_quantile(
                prediction_data.collections,
                self.training_data_.collections,
                self.spec.n_quantiles,
            )
            cross = jnp.exp(-(distances**2) / (2.0 * self.decision_sigma_**2))

        assert isinstance(self.fit_result_, KLRFitResult)
        return KernelLogisticRegression.predict_proba(
            KernelLogisticRegression(), cross, self.fit_result_.alpha
        )

    def _fit_dual(self, gram: Float[Array, "n n"], labels: Float[Array, "n"]) -> None:
        self.gram_matrix_ = gram
        solver = KernelLogisticRegression(
            lambda_reg=self.lambda_reg, max_iter=self.max_iter, tol=self.tol
        )
        self.fit_result_ = solver.fit(gram, labels)

    def _embed_rff(self, bags: List[Bag]) -> Float[Array, "n d"]:
        if self._rff is None:
            raise RuntimeError("RFF representation is not initialized")
        self._rff._initialize_weights(bags[0].n_features)
        return jnp.stack([jnp.mean(self._rff.feature_map(bag.samples), axis=0) for bag in bags])

    def _exact_kme_matrix(self, left_bags: List[Bag], right_bags: List[Bag]) -> Float[Array, "n m"]:
        if self.point_sigma_ is None:
            raise RuntimeError("Point-level bandwidth is not initialized")
        if self.exact_batch_size < 1:
            raise ValueError("exact_batch_size must be positive")
        dimension = left_bags[0].n_features
        if any(bag.n_features != dimension for bag in [*left_bags, *right_bags]):
            raise ValueError("All bags must have the same feature dimension")
        max_left = max(bag.n_samples for bag in left_bags)
        max_right = max(bag.n_samples for bag in right_bags)
        right: NDArray[np.float32] = np.zeros(
            (len(right_bags), max_right, dimension), dtype=np.float32
        )
        right_mask: NDArray[np.float32] = np.zeros((len(right_bags), max_right), dtype=np.float32)
        for index, bag in enumerate(right_bags):
            count = bag.n_samples
            right[index, :count] = np.asarray(bag.samples)
            right_mask[index, :count] = 1.0

        blocks = []
        for start in range(0, len(left_bags), self.exact_batch_size):
            chunk = left_bags[start : start + self.exact_batch_size]
            left: NDArray[np.float32] = np.zeros(
                (self.exact_batch_size, max_left, dimension), dtype=np.float32
            )
            left_mask: NDArray[np.float32] = np.zeros(
                (self.exact_batch_size, max_left), dtype=np.float32
            )
            for local_index, bag in enumerate(chunk):
                count = bag.n_samples
                left[local_index, :count] = np.asarray(bag.samples)
                left_mask[local_index, :count] = 1.0
            block = _exact_kme_block(
                jnp.asarray(left),
                jnp.asarray(left_mask),
                jnp.asarray(right),
                jnp.asarray(right_mask),
                self.point_sigma_,
                self.round_exact_kernel,
            )
            blocks.append(block[: len(chunk)])
        return jnp.concatenate(blocks, axis=0)

    def _resolve_decision_sigma(self, embeddings: Float[Array, "n d"]) -> float:
        if not self.auto_sigma and self.decision_sigma is not None:
            return float(self.decision_sigma)
        distances = np.sqrt(
            np.asarray(self._squared_distances(embeddings, embeddings))[
                np.triu_indices(embeddings.shape[0], 1)
            ]
        )
        positive = distances[np.isfinite(distances) & (distances > 0)]
        return float(np.median(positive)) if positive.size else 1.0

    def _resolve_distance_sigma(self, distances: Float[Array, "n n"]) -> float:
        if not self.auto_sigma and self.decision_sigma is not None:
            return float(self.decision_sigma)
        values = np.asarray(distances)[np.triu_indices(distances.shape[0], 1)]
        positive = values[np.isfinite(values) & (values > 0)]
        return float(np.median(positive)) if positive.size else 1.0

    def _median_cell_distance(self, dataset: BagDataset) -> float:
        cells = np.concatenate([np.asarray(bag.samples) for bag in dataset.collections], axis=0)
        rng = np.random.default_rng(self.seed)
        if len(cells) > 2048:
            cells = cells[rng.choice(len(cells), 2048, replace=False)]
        left = cells[rng.integers(0, len(cells), size=min(8192, len(cells) * 4))]
        right = cells[rng.integers(0, len(cells), size=len(left))]
        distances = np.linalg.norm(left - right, axis=1)
        positive = distances[np.isfinite(distances) & (distances > 0)]
        return float(np.median(positive)) if positive.size else 1.0

    @staticmethod
    def _squared_distances(
        left: Float[Array, "n d"], right: Float[Array, "m d"]
    ) -> Float[Array, "n m"]:
        values = (
            jnp.sum(left**2, axis=1, keepdims=True)
            + jnp.sum(right**2, axis=1)[None, :]
            - 2.0 * left @ right.T
        )
        return jnp.maximum(values, 0.0)

    @classmethod
    def _rbf_matrix(
        cls,
        left: Float[Array, "n d"],
        right: Float[Array, "m d"],
        sigma: float,
    ) -> Float[Array, "n m"]:
        return jnp.exp(-cls._squared_distances(left, right) / (2.0 * sigma**2))

    def _reset_fit_state(self) -> None:
        self.preprocessor_ = None
        self.fit_result_ = None
        self.training_data_ = None
        self.gram_matrix_ = None
        self.training_embeddings_ = None
        self.point_sigma_ = None
        self.decision_sigma_ = None
        self.diagnostics_ = {}
        self._rff = None
        self._mean_kernel = None
        self._sw = None
