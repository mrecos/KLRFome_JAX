"""Reproducibility manifests and deterministic scientific-data fingerprints."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import jax
import numpy as np

from ..data.formats import BagDataset
from .validation import FoldPlan


def configuration_fingerprint(configuration: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible configuration using canonical ordering."""
    encoded = json.dumps(
        json_safe(configuration), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_fingerprint(dataset: BagDataset) -> str:
    """Hash ordered scientific content while excluding incidental metadata."""
    digest = hashlib.sha256()
    _update_text(digest, "klrfome-bag-dataset-v1")
    _update_text(digest, dataset.study_design)
    _update_text(digest, dataset.crs or "")
    for feature_name in dataset.feature_names:
        _update_text(digest, feature_name)
    for bag in dataset.collections:
        _update_text(digest, bag.id)
        _update_text(digest, str(bag.label))
        _update_text(digest, bag.group_id or "")
        _update_text(digest, bag.stratum_id or "")
        _update_array(digest, np.asarray(bag.samples))
        if bag.coordinates is None:
            _update_text(digest, "no-coordinates")
        else:
            _update_array(digest, np.asarray(bag.coordinates))
    return digest.hexdigest()


def serialize_fold_plan(
    plan: FoldPlan, dataset: BagDataset, group_ids: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Represent a fold plan with stable bag IDs rather than positional indices alone."""
    plan.validate_for(dataset)
    groups = list(group_ids) if group_ids is not None else None
    return {
        "n_splits": plan.n_splits,
        "n_repeats": plan.n_repeats,
        "seed": plan.seed,
        "bag_ids": list(plan.bag_ids),
        "assignments": [
            {
                "repeat": assignment.repeat + 1,
                "fold": assignment.fold + 1,
                "train_ids": [dataset.collections[index].id for index in assignment.train_indices],
                "test_ids": [dataset.collections[index].id for index in assignment.test_indices],
                "train_groups": (
                    sorted({groups[index] for index in assignment.train_indices})
                    if groups is not None
                    else None
                ),
                "test_groups": (
                    sorted({groups[index] for index in assignment.test_indices})
                    if groups is not None
                    else None
                ),
            }
            for assignment in plan.assignments
        ],
    }


def environment_manifest(repository: Optional[Path] = None) -> Dict[str, Any]:
    """Capture runtime versions and optional Git state without requiring a repository."""
    packages: Dict[str, Optional[str]] = {}
    for package in ("klrfome", "jax", "jaxlib", "numpy", "scipy", "scikit-learn"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    manifest: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }
    git = _git_manifest(repository or Path.cwd())
    if git is not None:
        manifest["git"] = git
    return manifest


def json_safe(value: Any) -> Any:
    """Convert NumPy values and nonfinite floats to strict JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def write_strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic, standards-compliant JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _update_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _update_array(digest: Any, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value, dtype="<f8")
    _update_text(digest, repr(array.shape))
    digest.update(array.tobytes(order="C"))


def _git_manifest(repository: Path) -> Optional[Dict[str, Any]]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True, stderr=subprocess.DEVNULL
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"revision": revision, "branch": branch, "tracked_worktree_dirty": bool(status.strip())}
