#!/usr/bin/env python3
"""
Run Python model on R's EXACT formatted data.
This eliminates ALL data differences - we use R's actual collections.
"""

import json
import numpy as np
import pandas as pd
import jax.numpy as jnp
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from klrfome.data.formats import TrainingData, SampleCollection
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

        # Load collection data
        coll_df = pd.read_csv(output_dir / f"r_collection_{idx:02d}.csv")
        # Get variable columns (exclude metadata)
        var_cols = [c for c in coll_df.columns if c not in ['collection_id', 'collection_index', 'label']]
        samples = coll_df[var_cols].values

        collections.append(SampleCollection(
            samples=jnp.array(samples),
            label=label,
            id=coll_id
        ))

    return collections, manifest


def main():
    output_dir = Path("diagnostic_output")

    print("=" * 80)
    print("PYTHON USING R's EXACT FORMATTED DATA")
    print("=" * 80)

    # Check for manifest
    manifest_file = output_dir / "r_collections_manifest.csv"
    if not manifest_file.exists():
        print(f"ERROR: {manifest_file} not found!")
        print("Run 'Rscript benchmarks/export_formatted_data.R' first.")
        return

    # Load R's collections
    print("\n[1/4] Loading R's formatted collections...")
    collections, manifest = load_r_collections(output_dir)
    labels = jnp.array([c.label for c in collections])

    print(f"  ✓ Loaded {len(collections)} collections")
    print(f"  Labels: {list(labels)}")

    # Show first 3 collections
    print("\n  First 3 collections:")
    for i, coll in enumerate(collections[:3]):
        print(f"    {i}: id={coll.id}, label={coll.label}, n_samples={len(coll.samples)}")
        print(f"       mean={coll.samples.mean(axis=0)}")

    # Build kernel
    print("\n[2/4] Building kernel matrix...")
    sigma = 0.5
    rbf_kernel = RBFKernel(sigma=sigma)
    me_kernel = MeanEmbeddingKernel(base_kernel=rbf_kernel)

    K = me_kernel.build_similarity_matrix(
        collections,
        round_kernel=True,
        kernel_decimals=3
    )

    np.save(output_dir / "python_rdata_kernel.npy", np.array(K))

    print(f"  ✓ Kernel: {K.shape}, mean={jnp.mean(K):.6f}, diag_mean={jnp.mean(jnp.diag(K)):.6f}")

    # Save kernel stats
    kernel_stats = {
        'shape': list(K.shape),
        'mean': float(jnp.mean(K)),
        'diag_mean': float(jnp.mean(jnp.diag(K))),
        'off_diag_mean': float(jnp.mean(K - jnp.diag(jnp.diag(K))))
    }
    with open(output_dir / "python_rdata_kernel_stats.json", 'w') as f:
        json.dump(kernel_stats, f, indent=2)

    # Fit model
    print("\n[3/4] Fitting KLR model...")
    np.save(output_dir / "python_rdata_labels.npy", np.array(labels))

    klr = KernelLogisticRegression(lambda_reg=0.1, max_iter=100, tol=0.001)
    result = klr.fit(K, labels)

    np.save(output_dir / "python_rdata_alpha.npy", np.array(result.alpha))

    print(f"  ✓ Converged: {result.converged} in {result.n_iterations} iterations")
    print(f"  ✓ Alpha: mean={jnp.mean(result.alpha):.6f}, std={jnp.std(result.alpha):.6f}")

    # Training predictions
    print("\n[4/4] Computing predictions...")
    train_pred = klr.predict_proba(K, result.alpha)
    np.save(output_dir / "python_rdata_predictions.npy", np.array(train_pred))

    print(f"  ✓ Predictions: mean={jnp.mean(train_pred):.6f}, range=[{jnp.min(train_pred):.4f}, {jnp.max(train_pred):.4f}]")

    # Compute AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(np.array(labels), np.array(train_pred))
    print(f"  ✓ Training AUC: {auc:.4f}")

    print("\n" + "=" * 80)
    print("Done! Now compare with R's kernel and model outputs.")
    print("=" * 80)


if __name__ == "__main__":
    main()
