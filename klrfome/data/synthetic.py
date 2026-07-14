"""Deterministic synthetic bag datasets for distribution-regression experiments."""

from dataclasses import asdict, dataclass
from typing import List, Literal, Optional

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .formats import Bag, BagDataset

ScenarioKind = Literal[
    "null",
    "mean_shift",
    "variance_shift",
    "heavy_tail",
    "multimodal",
    "correlation_shift",
    "sparse_signal",
    "nonlinear_mixture",
]


@dataclass(frozen=True)
class SyntheticScenarioConfig:
    """Configuration for one controlled bag-level classification problem.

    ``effect_size`` has a scenario-specific interpretation but always increases
    class separation from zero. Spatial dependence is introduced with a Gaussian
    copula, preserving each generated feature's empirical marginal distribution.
    """

    scenario: ScenarioKind
    n_bags_per_class: int = 24
    n_features: int = 3
    n_signal_features: int = 1
    bag_size: int = 30
    unequal_bag_sizes: bool = False
    min_bag_size: int = 3
    max_bag_size: int = 120
    effect_size: float = 0.75
    spatial_range: float = 0.0
    bags_per_group: int = 1
    seed: int = 42
    feature_prefix: str = "x"

    def __post_init__(self) -> None:
        if self.n_bags_per_class < 2:
            raise ValueError("n_bags_per_class must be at least 2")
        if self.n_features < 1:
            raise ValueError("n_features must be positive")
        if not 1 <= self.n_signal_features <= self.n_features:
            raise ValueError("n_signal_features must be between 1 and n_features")
        if self.scenario == "correlation_shift" and self.n_features < 2:
            raise ValueError("correlation_shift requires at least two features")
        if self.bag_size < 1 or self.min_bag_size < 1 or self.max_bag_size < 1:
            raise ValueError("bag sizes must be positive")
        if self.min_bag_size > self.max_bag_size:
            raise ValueError("min_bag_size cannot exceed max_bag_size")
        if self.effect_size < 0:
            raise ValueError("effect_size must be nonnegative")
        if self.spatial_range < 0:
            raise ValueError("spatial_range must be nonnegative")
        if self.bags_per_group < 1:
            raise ValueError("bags_per_group must be positive")


def generate_synthetic_bags(config: SyntheticScenarioConfig) -> BagDataset:
    """Generate a canonical, deterministic dataset with known class differences."""
    rng = np.random.default_rng(config.seed)
    bags: List[Bag] = []
    for label in (0, 1):
        for bag_index in range(config.n_bags_per_class):
            n_samples = _bag_size(config, rng)
            samples = _draw_samples(config, label, bag_index, n_samples, rng)
            coordinates = _grid_coordinates(n_samples, bag_index, label)
            if config.spatial_range > 0:
                samples = _impose_spatial_copula(samples, coordinates, config.spatial_range, rng)
            bag_id = f"{config.scenario}-class{label}-bag{bag_index:04d}"
            group_id = f"synthetic-group-{bag_index // config.bags_per_group:04d}"
            bags.append(
                Bag(
                    samples=jnp.asarray(samples, dtype=jnp.float32),
                    label=label,
                    id=bag_id,
                    coordinates=jnp.asarray(coordinates, dtype=jnp.float32),
                    group_id=group_id,
                    stratum_id=config.scenario,
                    metadata={
                        "adapter": "synthetic_distribution_regression",
                        "scenario": config.scenario,
                        "effect_size": config.effect_size,
                        "spatial_range": config.spatial_range,
                    },
                )
            )
    feature_names = [f"{config.feature_prefix}{index + 1}" for index in range(config.n_features)]
    return BagDataset(
        collections=bags,
        feature_names=feature_names,
        crs="LOCAL:SYNTHETIC",
        study_design="presence_background",
        metadata={
            "generator": "klrfome.data.synthetic.generate_synthetic_bags",
            "configuration": asdict(config),
            "interpretation": "synthetic presence-background relative ranking",
        },
    )


def permute_bag_cells(dataset: BagDataset, seed: int = 42) -> BagDataset:
    """Return an equivalent dataset with independently permuted cell order."""
    rng = np.random.default_rng(seed)
    bags = []
    for bag in dataset.collections:
        order = rng.permutation(bag.n_samples)
        bags.append(_copy_bag_with_cells(bag, np.asarray(bag.samples)[order], order=order))
    return _copy_dataset(dataset, bags, transformation="permuted_cells")


def duplicate_all_cells(dataset: BagDataset, repeats: int = 2) -> BagDataset:
    """Uniformly duplicate every cell, preserving each empirical distribution."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    bags = []
    for bag in dataset.collections:
        indices: NDArray[np.int_] = np.repeat(np.arange(bag.n_samples), repeats)
        bags.append(_copy_bag_with_cells(bag, np.asarray(bag.samples)[indices], order=indices))
    return _copy_dataset(dataset, bags, transformation=f"uniform_duplication_{repeats}")


def duplicate_selected_cells(
    dataset: BagDataset, fraction: float = 0.25, seed: int = 42
) -> BagDataset:
    """Duplicate a subset of cells, intentionally reweighting the distribution."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    bags = []
    for bag in dataset.collections:
        count = max(1, int(np.ceil(bag.n_samples * fraction)))
        selected = rng.choice(bag.n_samples, size=count, replace=False)
        indices = np.concatenate([np.arange(bag.n_samples), selected])
        bags.append(_copy_bag_with_cells(bag, np.asarray(bag.samples)[indices], order=indices))
    return _copy_dataset(dataset, bags, transformation=f"selective_duplication_{fraction:g}")


def _bag_size(config: SyntheticScenarioConfig, rng: np.random.Generator) -> int:
    if config.unequal_bag_sizes:
        return int(rng.integers(config.min_bag_size, config.max_bag_size + 1))
    return config.bag_size


def _draw_samples(
    config: SyntheticScenarioConfig,
    label: int,
    bag_index: int,
    n_samples: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    samples = rng.normal(size=(n_samples, config.n_features))
    signal = slice(0, config.n_signal_features)
    effect = config.effect_size

    if config.scenario == "null":
        return samples
    if config.scenario in ("mean_shift", "sparse_signal"):
        if label == 1:
            samples[:, signal] += effect
        return samples
    if config.scenario == "variance_shift":
        if label == 1:
            samples[:, signal] *= 1.0 + effect
        return samples
    if config.scenario == "heavy_tail":
        if label == 1 and effect > 0:
            probability = min(effect, 1.0)
            replace = rng.random((n_samples, config.n_signal_features)) < probability
            student = rng.standard_t(df=3.0, size=replace.shape) / np.sqrt(3.0)
            samples[:, signal] = np.where(replace, student, samples[:, signal])
        return samples
    if config.scenario == "multimodal":
        if label == 1 and effect > 0:
            separation = min(effect, 0.95)
            signs = rng.choice((-1.0, 1.0), size=(n_samples, config.n_signal_features))
            within_scale = np.sqrt(max(1.0 - separation**2, 0.05))
            samples[:, signal] = (
                signs * separation
                + rng.normal(size=(n_samples, config.n_signal_features)) * within_scale
            )
        return samples
    if config.scenario == "correlation_shift":
        if label == 1:
            correlation = min(effect, 0.95)
            covariance = np.array([[1.0, correlation], [correlation, 1.0]])
            samples[:, :2] = rng.multivariate_normal(np.zeros(2), covariance, size=n_samples)
        return samples
    if config.scenario == "nonlinear_mixture":
        separation = max(effect, 1e-8)
        within_scale = 0.35
        if label == 1:
            component = -1.0 if bag_index % 2 == 0 else 1.0
            samples[:, 0] = rng.normal(component * separation, within_scale, size=n_samples)
        else:
            components = rng.choice((-1.0, 1.0), size=n_samples)
            samples[:, 0] = rng.normal(components * separation, within_scale, size=n_samples)
        return samples
    raise ValueError(f"Unsupported scenario: {config.scenario}")


def _grid_coordinates(n_samples: int, bag_index: int, label: int) -> NDArray[np.float64]:
    side = int(np.ceil(np.sqrt(n_samples)))
    rows, cols = np.divmod(np.arange(n_samples), side)
    offset_x = float((bag_index % 12) * (side + 3))
    offset_y = float((bag_index // 12) * (side + 3) + label * 1000)
    return np.column_stack([cols + offset_x, rows + offset_y]).astype(float)


def _impose_spatial_copula(
    samples: NDArray[np.float64],
    coordinates: NDArray[np.float64],
    spatial_range: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    covariance = np.exp(-distances / spatial_range)
    covariance.flat[:: len(coordinates) + 1] += 1e-6
    output = samples.copy()
    for feature in range(samples.shape[1]):
        latent = rng.multivariate_normal(np.zeros(len(samples)), covariance)
        rank_order = np.argsort(latent)
        output[rank_order, feature] = np.sort(samples[:, feature])
    return output


def _copy_bag_with_cells(bag: Bag, samples: NDArray[np.floating], order: np.ndarray) -> Bag:
    coordinates: Optional[NDArray[np.floating]] = None
    if bag.coordinates is not None:
        coordinates = np.asarray(bag.coordinates)[list(order)]
    metadata = dict(bag.metadata or {})
    return Bag(
        samples=jnp.asarray(samples),
        label=bag.label,
        id=bag.id,
        metadata=metadata,
        coordinates=jnp.asarray(coordinates) if coordinates is not None else None,
        group_id=bag.group_id,
        stratum_id=bag.stratum_id,
    )


def _copy_dataset(dataset: BagDataset, bags: List[Bag], transformation: str) -> BagDataset:
    metadata = dict(dataset.metadata or {})
    metadata["transformation"] = transformation
    return BagDataset(
        collections=bags,
        feature_names=list(dataset.feature_names),
        crs=dataset.crs,
        study_design=dataset.study_design,
        metadata=metadata,
    )
