"""Versioned, non-executable fitted-model archives for KLRfome."""

import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Union, cast

import jax.numpy as jnp
import numpy as np

from ..api import KLRfome
from ..data.formats import Bag, BagDataset
from ..data.preprocessing import BagStandardizer
from ..kernels.distribution import MeanEmbeddingKernel
from ..kernels.rbf import RBFKernel
from ..kernels.rff import RandomFourierFeatures
from ..kernels.wasserstein import SlicedWassersteinDistance
from ..models.distribution import DistributionClassifier
from ..models.klr import KLRFitResult
from ..models.primal import PrimalFitResult
from ..models.spec import ModelSpec
from .reproducibility import dataset_fingerprint, environment_manifest, json_safe

ARCHIVE_VERSION = "1.0"
MANIFEST_NAME = "manifest.json"
ARRAYS_NAME = "arrays.npz"
FittedModel = Union[KLRfome, DistributionClassifier]


def save_model(model: FittedModel, file_path: str) -> None:
    """Save a fitted facade or core model without arbitrary-code pickle state."""
    core = _core_model(model)
    if core.fit_result_ is None or core.feature_names_ is None:
        raise RuntimeError("Model must be fit before serialization")
    manifest, arrays = _encode_core(core)
    manifest["model_class"] = "KLRfome" if isinstance(model, KLRfome) else "DistributionClassifier"
    if isinstance(model, KLRfome):
        manifest["facade"] = _facade_configuration(model)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)  # type: ignore[arg-type]
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            MANIFEST_NAME,
            json.dumps(json_safe(manifest), sort_keys=True, indent=2, allow_nan=False) + "\n",
        )
        archive.writestr(ARRAYS_NAME, buffer.getvalue())


def load_model(file_path: str) -> FittedModel:
    """Load and validate a fitted model archive."""
    path = Path(file_path)
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        if names != {MANIFEST_NAME, ARRAYS_NAME}:
            raise ValueError("Model archive must contain only manifest.json and arrays.npz")
        manifest = json.loads(archive.read(MANIFEST_NAME))
        with np.load(io.BytesIO(archive.read(ARRAYS_NAME)), allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]) for name in stored.files}
    if manifest.get("archive_version") != ARCHIVE_VERSION:
        raise ValueError(f"Unsupported model archive version: {manifest.get('archive_version')!r}")
    core = _decode_core(manifest, arrays)
    model_class = manifest.get("model_class")
    if model_class == "DistributionClassifier":
        return core
    if model_class != "KLRfome":
        raise ValueError(f"Unsupported archived model class: {model_class!r}")
    facade_configuration = dict(manifest.get("facade") or {})
    facade_configuration["spec"] = core.spec
    facade = KLRfome(**facade_configuration)
    facade._core_model = core
    facade._resolved_spec = core.spec
    facade._training_data = core.training_data_
    facade._similarity_matrix = core.gram_matrix_
    facade._fit_result = core.fit_result_  # type: ignore[assignment]
    if core.preprocessor_ is not None:
        facade._feature_means = jnp.asarray(core.preprocessor_.means)
        facade._feature_stds = jnp.asarray(core.preprocessor_.scales)
    return facade


def _core_model(model: FittedModel) -> DistributionClassifier:
    if isinstance(model, DistributionClassifier):
        return model
    if isinstance(model, KLRfome) and model._core_model is not None:
        return model._core_model
    raise RuntimeError("KLRfome model must be fit before serialization")


def _encode_core(core: DistributionClassifier) -> tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    fit = core.fit_result_
    assert fit is not None and core.feature_names_ is not None
    coefficient_kind = "primal" if isinstance(fit, PrimalFitResult) else "dual"
    coefficients = fit.coefficients if isinstance(fit, PrimalFitResult) else fit.alpha
    arrays: Dict[str, np.ndarray] = {"coefficients": _finite_array(coefficients)}
    if core._rff is not None and core._rff._W is not None:
        arrays["rff_weights"] = _finite_array(core._rff._W)
    if core.training_embeddings_ is not None:
        arrays["training_embeddings"] = _finite_array(core.training_embeddings_)
    if core.training_shrinkage_factors_ is not None:
        arrays["training_shrinkage_factors"] = _finite_array(core.training_shrinkage_factors_)
    if core.training_effective_sizes_ is not None:
        arrays["training_effective_sizes"] = _finite_array(core.training_effective_sizes_)
    if core._sw is not None and core._sw._projections is not None:
        arrays["wasserstein_projections"] = _finite_array(core._sw._projections)
    needs_reference_bags = core.spec.representation in (
        "exact_kme",
        "sliced_wasserstein",
        "hybrid",
    )
    if needs_reference_bags:
        if core.training_data_ is None:
            raise RuntimeError("This model architecture requires reference training bags")
        arrays.update(_encode_training_bags(core.training_data_))
    manifest: Dict[str, Any] = {
        "archive_version": ARCHIVE_VERSION,
        "environment": environment_manifest(),
        "training_data_fingerprint": (
            dataset_fingerprint(core.training_data_) if core.training_data_ is not None else None
        ),
        "spec": asdict(core.spec),
        "configuration": {
            "sigma": core.sigma,
            "decision_sigma": core.decision_sigma,
            "lambda_reg": core.lambda_reg,
            "scale_features": core.scale_features,
            "auto_sigma": core.auto_sigma,
            "seed": core.seed,
            "round_exact_kernel": core.round_exact_kernel,
            "max_iter": core.max_iter,
            "tol": core.tol,
            "exact_batch_size": core.exact_batch_size,
        },
        "fitted": {
            "feature_names": list(core.feature_names_),
            "crs": core.crs_,
            "study_design": core.study_design_,
            "point_sigma": core.point_sigma_,
            "decision_sigma": core.decision_sigma_,
            "hybrid_mean_scale": core.hybrid_mean_scale_,
            "hybrid_transport_scale": core.hybrid_transport_scale_,
            "coefficient_kind": coefficient_kind,
            "converged": fit.converged,
            "n_iterations": fit.n_iterations,
            "final_loss": fit.final_loss,
            "failure_reason": fit.failure_reason,
            "jitter_used": fit.jitter_used,
            "diagnostics": core.diagnostics_,
        },
        "preprocessor": (
            {
                "means": list(core.preprocessor_.means),
                "scales": list(core.preprocessor_.scales),
                "feature_names": list(core.preprocessor_.feature_names),
                "crs": core.preprocessor_.crs,
            }
            if core.preprocessor_ is not None
            else None
        ),
    }
    return manifest, arrays


def _decode_core(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> DistributionClassifier:
    spec = ModelSpec(**manifest["spec"])
    configuration = dict(manifest["configuration"])
    configuration["spec"] = spec
    core = DistributionClassifier(**configuration)
    fitted = manifest["fitted"]
    core.feature_names_ = tuple(fitted["feature_names"])
    core.crs_ = fitted.get("crs")
    core.study_design_ = fitted.get("study_design")
    core.point_sigma_ = fitted.get("point_sigma")
    core.decision_sigma_ = fitted.get("decision_sigma")
    core.hybrid_mean_scale_ = fitted.get("hybrid_mean_scale")
    core.hybrid_transport_scale_ = fitted.get("hybrid_transport_scale")
    core.diagnostics_ = dict(fitted.get("diagnostics") or {})
    preprocessor = manifest.get("preprocessor")
    if preprocessor is not None:
        core.preprocessor_ = BagStandardizer(
            tuple(float(value) for value in preprocessor["means"]),
            tuple(float(value) for value in preprocessor["scales"]),
            tuple(preprocessor["feature_names"]),
            preprocessor.get("crs"),
        )
    coefficients = _required_array(arrays, "coefficients")
    result_arguments = (
        bool(fitted["converged"]),
        int(fitted["n_iterations"]),
        float(fitted["final_loss"]),
        fitted.get("failure_reason"),
        float(fitted["jitter_used"]),
    )
    if fitted["coefficient_kind"] == "primal":
        core.fit_result_ = PrimalFitResult(jnp.asarray(coefficients), *result_arguments)
    elif fitted["coefficient_kind"] == "dual":
        core.fit_result_ = KLRFitResult(jnp.asarray(coefficients), *result_arguments)
    else:
        raise ValueError("Unknown coefficient kind in model archive")
    if spec.representation == "rff_kme":
        weights = _required_array(arrays, "rff_weights")
        if weights.ndim != 2 or weights.shape[1] != spec.rff_features:
            raise ValueError("Archived RFF weights have incompatible shape")
        core._rff = RandomFourierFeatures(
            sigma=float(core.point_sigma_),
            n_features=spec.rff_features,
            seed=core.seed,
            scheme=spec.rff_scheme,
        )
        core._rff._W = jnp.asarray(weights)
        core._rff._input_dim = weights.shape[0]
        core._rff._b = None
        if "training_shrinkage_factors" in arrays:
            core.training_shrinkage_factors_ = jnp.asarray(
                _required_array(arrays, "training_shrinkage_factors")
            )
        if "training_effective_sizes" in arrays:
            core.training_effective_sizes_ = jnp.asarray(
                _required_array(arrays, "training_effective_sizes")
            )
        if spec.solver == "dual_klr":
            core.training_embeddings_ = jnp.asarray(_required_array(arrays, "training_embeddings"))
    elif spec.representation == "hybrid":
        core.training_data_ = _decode_training_bags(arrays, fitted)
        projections = _required_array(arrays, "wasserstein_projections")
        core._sw = SlicedWassersteinDistance(spec.n_projections, 2, core.seed)
        core._sw._projections = jnp.asarray(projections)
        core._sw._dimension = projections.shape[1]
        if spec.hybrid_mean_representation == "rff_kme":
            weights = _required_array(arrays, "rff_weights")
            if weights.ndim != 2 or weights.shape[1] != spec.rff_features:
                raise ValueError("Archived RFF weights have incompatible shape")
            core._rff = RandomFourierFeatures(
                sigma=float(core.point_sigma_),
                n_features=spec.rff_features,
                seed=core.seed,
                scheme=spec.rff_scheme,
            )
            core._rff._W = jnp.asarray(weights)
            core._rff._input_dim = weights.shape[0]
            core._rff._b = None
            core.training_embeddings_ = jnp.asarray(_required_array(arrays, "training_embeddings"))
            if "training_shrinkage_factors" in arrays:
                core.training_shrinkage_factors_ = jnp.asarray(
                    _required_array(arrays, "training_shrinkage_factors")
                )
            if "training_effective_sizes" in arrays:
                core.training_effective_sizes_ = jnp.asarray(
                    _required_array(arrays, "training_effective_sizes")
                )
        else:
            core._mean_kernel = MeanEmbeddingKernel(RBFKernel(float(core.point_sigma_)))
    elif spec.representation == "sliced_wasserstein":
        projections = _required_array(arrays, "wasserstein_projections")
        core._sw = SlicedWassersteinDistance(spec.n_projections, 2, core.seed)
        core._sw._projections = jnp.asarray(projections)
        core._sw._dimension = projections.shape[1]
        core.training_data_ = _decode_training_bags(arrays, fitted)
    else:
        core.training_data_ = _decode_training_bags(arrays, fitted)
    return core


def _encode_training_bags(dataset: BagDataset) -> Dict[str, np.ndarray]:
    offsets = [0]
    values = []
    for bag in dataset.collections:
        values.append(np.asarray(bag.samples, dtype=np.float32))
        offsets.append(offsets[-1] + bag.n_samples)
    return {
        "training_samples": _finite_array(np.concatenate(values, axis=0)),
        "training_offsets": np.asarray(offsets, dtype=np.int64),
        "training_labels": np.asarray([bag.label for bag in dataset.collections], dtype=np.int8),
        "training_ids": np.asarray([bag.id for bag in dataset.collections], dtype=np.str_),
    }


def _decode_training_bags(
    arrays: Mapping[str, np.ndarray], fitted: Mapping[str, Any]
) -> BagDataset:
    samples = _required_array(arrays, "training_samples")
    offsets: np.ndarray = _required_array(arrays, "training_offsets").astype(int)
    labels: np.ndarray = _required_array(arrays, "training_labels").astype(int)
    identifiers: np.ndarray = _required_array(arrays, "training_ids").astype(str)
    if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != len(samples):
        raise ValueError("Archived training-bag offsets are invalid")
    if len(offsets) != len(labels) + 1 or len(labels) != len(identifiers):
        raise ValueError("Archived training-bag metadata lengths differ")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Archived training-bag labels must be binary")
    bags = [
        Bag(
            jnp.asarray(samples[offsets[index] : offsets[index + 1]]),
            cast(Literal[0, 1], int(labels[index])),
            str(identifiers[index]),
        )
        for index in range(len(labels))
    ]
    return BagDataset(
        bags,
        list(fitted["feature_names"]),
        crs=fitted.get("crs"),
        study_design=fitted.get("study_design") or "presence_background",
    )


def _facade_configuration(model: KLRfome) -> Dict[str, Any]:
    return {
        "sigma": model.sigma,
        "lambda_reg": model.lambda_reg,
        "kernel_type": model.kernel_type,
        "n_rff_features": model.n_rff_features,
        "n_projections": model.n_projections,
        "wasserstein_p": model.wasserstein_p,
        "n_quantiles": model.n_quantiles,
        "bucket_width": model.bucket_width,
        "bucket_ceil": model.bucket_ceil,
        "bucket_cap": model.bucket_cap,
        "window_size": model.window_size,
        "seed": model.seed,
        "scale_features": model.scale_features,
        "auto_sigma": model.auto_sigma,
        "embedding_kernel": model.embedding_kernel,
    }


def _finite_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError("Model archive arrays must be finite")
    return array


def _required_array(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"Model archive is missing required array {name!r}")
    return _finite_array(arrays[name]) if arrays[name].dtype.kind not in "US" else arrays[name]
