#!/usr/bin/env python3
"""
Unified validation script: Compare Python implementation against R benchmark.

This is the PRIMARY validation script for ensuring Python matches R.

Usage:
    1. Generate benchmark data:     python benchmarks/generate_benchmark_data.py
    2. Run R export:                Rscript benchmarks/validate_r_export.R
    3. Run this script:             python benchmarks/validate_against_r.py

The script compares:
    - Kernel matrices (should match exactly to 3 decimals)
    - Alpha coefficients from IRLS (should match closely)
    - Training predictions (should match closely)
"""

import json
import numpy as np
import pandas as pd
import jax.numpy as jnp
from pathlib import Path
from sklearn.metrics import roc_auc_score
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from klrfome.data.formats import SampleCollection, TrainingData
from klrfome.kernels.distribution import MeanEmbeddingKernel
from klrfome.kernels.rbf import RBFKernel
from klrfome.models.klr import KernelLogisticRegression


def load_r_collections(output_dir: Path):
    """Load collections exported by R."""
    manifest = pd.read_csv(output_dir / "r_collections_manifest.csv")
    
    collections = []
    for _, row in manifest.iterrows():
        idx = int(row['index'])
        coll_id = row['id']
        label = int(row['label'])
        
        coll_df = pd.read_csv(output_dir / f"r_collection_{idx:02d}.csv")
        var_cols = [c for c in coll_df.columns if c not in ['collection_id', 'collection_index', 'label']]
        samples = coll_df[var_cols].values
        
        collections.append(SampleCollection(
            samples=jnp.array(samples),
            label=label,
            id=coll_id
        ))
    
    return collections, manifest


def compare_arrays(py_arr, r_arr, name, threshold=0.001):
    """Compare two arrays and return match status."""
    diff = np.abs(np.array(py_arr) - np.array(r_arr))
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    match = max_diff < threshold
    status = "✓ MATCH" if match else "✗ DIFFER"
    
    return {
        'name': name,
        'match': match,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'status': status
    }


def main():
    output_dir = Path("diagnostic_output")
    
    print("=" * 70)
    print("PYTHON vs R VALIDATION")
    print("=" * 70)
    
    # Check for required files
    manifest_file = output_dir / "r_collections_manifest.csv"
    if not manifest_file.exists():
        print(f"\nERROR: {manifest_file} not found!")
        print("\nPlease run the R export first:")
        print("  Rscript benchmarks/validate_r_export.R")
        return False
    
    # Load R's exported collections
    print("\n[1/4] Loading R's formatted collections...")
    collections, manifest = load_r_collections(output_dir)
    labels = jnp.array([c.label for c in collections])
    print(f"  Loaded {len(collections)} collections")
    print(f"  Labels: {list(labels)}")
    
    # Build Python kernel
    print("\n[2/4] Building kernel matrix (Python)...")
    sigma = 0.5
    rbf_kernel = RBFKernel(sigma=sigma)
    me_kernel = MeanEmbeddingKernel(base_kernel=rbf_kernel)
    
    py_K = me_kernel.build_similarity_matrix(
        collections,
        round_kernel=True,
        kernel_decimals=3
    )
    print(f"  Shape: {py_K.shape}")
    print(f"  Mean: {float(jnp.mean(py_K)):.6f}, Diag mean: {float(jnp.mean(jnp.diag(py_K))):.6f}")
    
    # Load R kernel
    r_K = pd.read_csv(output_dir / "r_rdata_kernel.csv").values
    print(f"\n  R kernel: mean={r_K.mean():.6f}, diag_mean={np.diag(r_K).mean():.6f}")
    
    # Compare kernels
    kernel_result = compare_arrays(py_K, r_K, "Kernel Matrix", threshold=0.001)
    print(f"\n  {kernel_result['status']} (max diff: {kernel_result['max_diff']:.6f})")
    
    # Fit Python model
    print("\n[3/4] Fitting KLR model (Python)...")
    klr = KernelLogisticRegression(lambda_reg=0.1, max_iter=100, tol=0.001)
    result = klr.fit(py_K, labels)
    py_alpha = result.alpha
    print(f"  Converged: {result.converged} in {result.n_iterations} iterations")
    print(f"  Alpha: mean={float(jnp.mean(py_alpha)):.6f}")
    
    # Load R alpha
    r_alpha = pd.read_csv(output_dir / "r_rdata_alpha.csv").values.flatten()
    print(f"  R alpha: mean={r_alpha.mean():.6f}")
    
    # Compare alphas
    alpha_result = compare_arrays(py_alpha, r_alpha, "Alpha", threshold=0.01)
    print(f"\n  {alpha_result['status']} (max diff: {alpha_result['max_diff']:.6f})")
    
    # Compute predictions
    print("\n[4/4] Computing predictions...")
    py_pred = klr.predict_proba(py_K, py_alpha)
    py_auc = roc_auc_score(np.array(labels), np.array(py_pred))
    print(f"  Python: mean={float(jnp.mean(py_pred)):.4f}, AUC={py_auc:.4f}")
    
    # Load R predictions
    with open(output_dir / "r_rdata_predictions.json") as f:
        r_pred = np.array(json.load(f))
    r_auc = roc_auc_score(np.array(labels), r_pred)
    print(f"  R:      mean={r_pred.mean():.4f}, AUC={r_auc:.4f}")
    
    # Compare predictions
    pred_result = compare_arrays(py_pred, r_pred, "Predictions", threshold=0.01)
    print(f"\n  {pred_result['status']} (max diff: {pred_result['max_diff']:.6f})")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    all_match = kernel_result['match'] and alpha_result['match'] and pred_result['match']
    
    print(f"\n  Kernel Matrix:  {kernel_result['status']}")
    print(f"  Alpha Values:   {alpha_result['status']}")
    print(f"  Predictions:    {pred_result['status']}")
    print(f"\n  Python AUC: {py_auc:.4f}")
    print(f"  R AUC:      {r_auc:.4f}")
    
    if all_match:
        print("\n  ✓ ALL COMPONENTS MATCH - Python implementation is validated!")
    else:
        print("\n  ✗ VALIDATION FAILED - Check the mismatched components")
    
    print("=" * 70)
    
    return all_match


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

