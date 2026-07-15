"""Validated model configurations for distribution-regression architectures."""

from dataclasses import dataclass
from typing import Literal, Optional

Representation = Literal["exact_kme", "rff_kme", "sliced_wasserstein", "hybrid"]
DecisionKernel = Literal["linear", "rbf", "hybrid"]
Solver = Literal["dual_klr", "primal_logistic"]
RFFScheme = Literal["iid", "orthogonal"]
EmbeddingEstimator = Literal["empirical", "shrinkage"]
ShrinkageEffectiveSize = Literal["nominal", "metadata"]
HybridMeanRepresentation = Literal["exact_kme", "rff_kme"]


@dataclass(frozen=True)
class ModelSpec:
    """Separate distribution representation, decision kernel, and solver."""

    representation: Representation
    decision_kernel: DecisionKernel
    solver: Solver
    rff_features: int = 256
    n_projections: int = 100
    n_quantiles: int = 128
    wasserstein_p: int = 2
    rff_scheme: RFFScheme = "iid"
    embedding_estimator: EmbeddingEstimator = "empirical"
    shrinkage_effective_size: ShrinkageEffectiveSize = "nominal"
    hybrid_weight: float = 0.5
    hybrid_mean_representation: HybridMeanRepresentation = "rff_kme"
    hybrid_normalize: bool = True

    def __post_init__(self) -> None:
        supported = {
            ("exact_kme", "linear", "dual_klr"): "M0",
            ("rff_kme", "linear", "primal_logistic"): "M1",
            ("rff_kme", "rbf", "dual_klr"): "M2",
            ("sliced_wasserstein", "rbf", "dual_klr"): "M3",
            ("hybrid", "hybrid", "dual_klr"): "M4",
        }
        if (self.representation, self.decision_kernel, self.solver) not in supported:
            raise ValueError(
                "Unsupported model architecture. Use ModelSpec.m0(), m1(), m2(), m3(), or m4()."
            )
        if self.rff_features <= 0:
            raise ValueError("rff_features must be positive")
        if self.n_projections <= 0 or self.n_quantiles <= 1:
            raise ValueError("n_projections must be positive and n_quantiles must exceed 1")
        if self.rff_scheme not in ("iid", "orthogonal"):
            raise ValueError("rff_scheme must be 'iid' or 'orthogonal'")
        if self.embedding_estimator not in ("empirical", "shrinkage"):
            raise ValueError("embedding_estimator must be 'empirical' or 'shrinkage'")
        if self.shrinkage_effective_size not in ("nominal", "metadata"):
            raise ValueError("shrinkage_effective_size must be 'nominal' or 'metadata'")
        if not 0.0 <= self.hybrid_weight <= 1.0:
            raise ValueError("hybrid_weight must be in [0, 1]")
        if self.hybrid_mean_representation not in ("exact_kme", "rff_kme"):
            raise ValueError("hybrid_mean_representation must be exact_kme or rff_kme")
        if self.representation == "sliced_wasserstein" and self.wasserstein_p != 2:
            raise ValueError(
                "The high-level model API supports sliced Wasserstein-2 only; "
                "Wasserstein-1 remains available as a research distance utility."
            )
        uses_rff = self.representation == "rff_kme" or (
            self.representation == "hybrid" and self.hybrid_mean_representation == "rff_kme"
        )
        if not uses_rff and self.rff_scheme != "iid":
            raise ValueError("rff_scheme is only valid for an RFF mean representation")
        if not uses_rff and self.embedding_estimator != "empirical":
            raise ValueError("shrinkage currently requires an RFF mean representation")
        if self.embedding_estimator == "empirical" and self.shrinkage_effective_size != "nominal":
            raise ValueError("metadata effective size is only valid with shrinkage")

    @property
    def method_id(self) -> str:
        mapping = {
            ("exact_kme", "linear", "dual_klr"): "M0",
            ("rff_kme", "linear", "primal_logistic"): "M1",
            ("rff_kme", "rbf", "dual_klr"): "M2",
            ("sliced_wasserstein", "rbf", "dual_klr"): "M3",
            ("hybrid", "hybrid", "dual_klr"): "M4",
        }
        return mapping[(self.representation, self.decision_kernel, self.solver)]

    @classmethod
    def m0(cls) -> "ModelSpec":
        return cls("exact_kme", "linear", "dual_klr")

    @classmethod
    def m1(
        cls,
        rff_features: int = 256,
        rff_scheme: RFFScheme = "iid",
        embedding_estimator: EmbeddingEstimator = "empirical",
        shrinkage_effective_size: ShrinkageEffectiveSize = "nominal",
    ) -> "ModelSpec":
        return cls(
            "rff_kme",
            "linear",
            "primal_logistic",
            rff_features=rff_features,
            rff_scheme=rff_scheme,
            embedding_estimator=embedding_estimator,
            shrinkage_effective_size=shrinkage_effective_size,
        )

    @classmethod
    def m2(
        cls,
        rff_features: int = 256,
        rff_scheme: RFFScheme = "iid",
        embedding_estimator: EmbeddingEstimator = "empirical",
        shrinkage_effective_size: ShrinkageEffectiveSize = "nominal",
    ) -> "ModelSpec":
        return cls(
            "rff_kme",
            "rbf",
            "dual_klr",
            rff_features=rff_features,
            rff_scheme=rff_scheme,
            embedding_estimator=embedding_estimator,
            shrinkage_effective_size=shrinkage_effective_size,
        )

    @classmethod
    def m3(cls, n_projections: int = 100, n_quantiles: int = 128) -> "ModelSpec":
        return cls(
            "sliced_wasserstein",
            "rbf",
            "dual_klr",
            n_projections=n_projections,
            n_quantiles=n_quantiles,
            wasserstein_p=2,
        )

    @classmethod
    def m4(
        cls,
        hybrid_weight: float = 0.5,
        hybrid_mean_representation: HybridMeanRepresentation = "rff_kme",
        rff_features: int = 256,
        rff_scheme: RFFScheme = "iid",
        embedding_estimator: EmbeddingEstimator = "empirical",
        shrinkage_effective_size: ShrinkageEffectiveSize = "nominal",
        n_projections: int = 100,
        n_quantiles: int = 128,
        hybrid_normalize: bool = True,
    ) -> "ModelSpec":
        """Return the experimental normalized mean/transport hybrid."""
        return cls(
            "hybrid",
            "hybrid",
            "dual_klr",
            rff_features=rff_features,
            n_projections=n_projections,
            n_quantiles=n_quantiles,
            wasserstein_p=2,
            rff_scheme=rff_scheme,
            embedding_estimator=embedding_estimator,
            shrinkage_effective_size=shrinkage_effective_size,
            hybrid_weight=hybrid_weight,
            hybrid_mean_representation=hybrid_mean_representation,
            hybrid_normalize=hybrid_normalize,
        )

    @classmethod
    def from_legacy(
        cls,
        kernel_type: str,
        n_rff_features: int,
        wasserstein_p: int = 2,
        n_projections: int = 100,
        n_quantiles: int = 128,
        embedding_kernel: Optional[str] = None,
    ) -> "ModelSpec":
        """Map the original ``KLRfome`` arguments to an explicit architecture.

        ``mean_embedding`` with no RFF is M0; RFF plus the historical/default
        linear embedding kernel is M1; callers can select ``embedding_kernel='rbf'``
        for M2. ``wasserstein`` maps to M3 and therefore rejects p=1.
        """
        if kernel_type == "wasserstein":
            if wasserstein_p != 2:
                raise ValueError(
                    "The high-level model API supports sliced Wasserstein-2 only; "
                    "Wasserstein-1 remains available as a research distance utility."
                )
            return cls.m3(n_projections=n_projections, n_quantiles=n_quantiles)
        if kernel_type != "mean_embedding":
            raise ValueError("kernel_type must be 'mean_embedding' or 'wasserstein'")
        if n_rff_features <= 0:
            return cls.m0()
        if embedding_kernel == "rbf":
            return cls.m2(n_rff_features)
        return cls.m1(n_rff_features)
