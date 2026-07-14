"""Validated model configurations for the four comparison methods."""

from dataclasses import dataclass
from typing import Literal, Optional

Representation = Literal["exact_kme", "rff_kme", "sliced_wasserstein"]
DecisionKernel = Literal["linear", "rbf"]
Solver = Literal["dual_klr", "primal_logistic"]


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

    def __post_init__(self) -> None:
        supported = {
            ("exact_kme", "linear", "dual_klr"): "M0",
            ("rff_kme", "linear", "primal_logistic"): "M1",
            ("rff_kme", "rbf", "dual_klr"): "M2",
            ("sliced_wasserstein", "rbf", "dual_klr"): "M3",
        }
        if (self.representation, self.decision_kernel, self.solver) not in supported:
            raise ValueError(
                "Unsupported model architecture. Use ModelSpec.m0(), m1(), m2(), or m3()."
            )
        if self.rff_features <= 0:
            raise ValueError("rff_features must be positive")
        if self.n_projections <= 0 or self.n_quantiles <= 1:
            raise ValueError("n_projections must be positive and n_quantiles must exceed 1")
        if self.representation == "sliced_wasserstein" and self.wasserstein_p != 2:
            raise ValueError(
                "The high-level model API supports sliced Wasserstein-2 only; "
                "Wasserstein-1 remains available as a research distance utility."
            )

    @property
    def method_id(self) -> str:
        mapping = {
            ("exact_kme", "linear", "dual_klr"): "M0",
            ("rff_kme", "linear", "primal_logistic"): "M1",
            ("rff_kme", "rbf", "dual_klr"): "M2",
            ("sliced_wasserstein", "rbf", "dual_klr"): "M3",
        }
        return mapping[(self.representation, self.decision_kernel, self.solver)]

    @classmethod
    def m0(cls) -> "ModelSpec":
        return cls("exact_kme", "linear", "dual_klr")

    @classmethod
    def m1(cls, rff_features: int = 256) -> "ModelSpec":
        return cls("rff_kme", "linear", "primal_logistic", rff_features=rff_features)

    @classmethod
    def m2(cls, rff_features: int = 256) -> "ModelSpec":
        return cls("rff_kme", "rbf", "dual_klr", rff_features=rff_features)

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
