#!/usr/bin/env python3
"""
Quick diagnostic using SHARED data extracted by R.
This ensures Python and R use identical raw data.
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


def load_shared_data(csv_path: str) -> pd.DataFrame:
    """Load shared data CSV from R."""
    return pd.read_csv(csv_path)


def format_site_data_from_csv(df: pd.DataFrame, N_sites: int = 10,
                               train_test_split: float = 0.8,
                               sample_fraction: float = 0.9,
                               background_site_balance: int = 1,
                               seed: int = 42) -> tuple:
    """
    Format data EXACTLY as R's format_site_data does.
    This mirrors the R code line-by-line.
    """
    np.random.seed(seed)

    # Get variable columns (exclude presence, SITENO)
    variables = [c for c in df.columns if c not in ['presence', 'SITENO']]

    # Scale data (R's scale function: (x - mean) / sd)
    means = df[variables].mean()
    sds = df[variables].std()

    df_scaled = df.copy()
    for var in variables:
        df_scaled[var] = (df[var] - means[var]) / sds[var]

    print(f"  Scaling parameters:")
    print(f"    Means: {means.values}")
    print(f"    SDs:   {sds.values}")

    N_back_bags = N_sites * background_site_balance

    # Get unique site IDs (not background)
    site_ids = df_scaled[df_scaled['presence'] == 1]['SITENO'].unique()

    # Reduce to N_sites
    if len(site_ids) > N_sites:
        selected_sites = np.random.choice(site_ids, size=N_sites, replace=False)
    else:
        selected_sites = site_ids

    # Split into train/test
    n_train = int(len(selected_sites) * train_test_split)
    train_sites = np.random.choice(selected_sites, size=n_train, replace=False)
    test_sites = [s for s in selected_sites if s not in train_sites]

    print(f"  Sites: {len(selected_sites)} total, {len(train_sites)} train, {len(test_sites)} test")

    # Get train site data
    train_site_data = df_scaled[(df_scaled['presence'] == 1) &
                                 (df_scaled['SITENO'].isin(train_sites))]
    test_site_data = df_scaled[(df_scaled['presence'] == 1) &
                                (df_scaled['SITENO'].isin(test_sites))]

    # Get background data
    bg_data = df_scaled[df_scaled['presence'] == 0]

    # Apply sample_fraction to sites
    def reduce_df(df, fraction):
        n_keep = max(1, int(len(df) * fraction))
        if n_keep < len(df):
            return df.sample(n=n_keep, random_state=42)
        return df

    train_site_data = reduce_df(train_site_data, sample_fraction)
    test_site_data = reduce_df(test_site_data, sample_fraction)

    # Background: sample N_back_bags bags
    # R groups background samples into N_back_bags bags randomly
    bg_data = bg_data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    bg_data['bag_id'] = [i % N_back_bags for i in range(len(bg_data))]

    # Apply sample_fraction to each bag
    train_bg_bags = []
    for bag_id in range(N_back_bags):
        bag = bg_data[bg_data['bag_id'] == bag_id]
        bag = reduce_df(bag, sample_fraction)
        train_bg_bags.append(bag)

    # Convert to collections
    def df_to_collections(df, label, variables):
        collections = []
        for site_id in df['SITENO'].unique():
            site_df = df[df['SITENO'] == site_id]
            samples = site_df[variables].values
            collections.append(SampleCollection(
                samples=jnp.array(samples),
                label=label,
                id=site_id
            ))
        return collections

    train_site_colls = df_to_collections(train_site_data, 1, variables)
    test_site_colls = df_to_collections(test_site_data, 1, variables)

    # Background bags
    train_bg_colls = []
    for i, bag in enumerate(train_bg_bags):
        if len(bag) > 0:
            samples = bag[variables].values
            train_bg_colls.append(SampleCollection(
                samples=jnp.array(samples),
                label=0,
                id=f"bg_{i}"
            ))

    # Combine
    train_collections = train_site_colls + train_bg_colls
    train_data = TrainingData(collections=train_collections, feature_names=variables)

    test_collections = test_site_colls
    test_data = TrainingData(collections=test_collections, feature_names=variables)

    scaling_params = {'means': means.to_dict(), 'sds': sds.to_dict()}

    return train_data, test_data, scaling_params


def main():
    output_dir = Path("diagnostic_output")

    print("=" * 80)
    print("PYTHON DIAGNOSTIC (Using SHARED data from R)")
    print("=" * 80)

    # Load shared data
    shared_csv = output_dir / "shared_raw_data.csv"
    if not shared_csv.exists():
        print(f"ERROR: {shared_csv} not found!")
        print("Run 'Rscript benchmarks/extract_shared_data.R' first.")
        return

    print("\n[1/4] Loading shared data...")
    df = load_shared_data(shared_csv)
    print(f"  ✓ Loaded {len(df)} rows")
    print(f"    Sites: {sum(df['presence'] == 1)}, Background: {sum(df['presence'] == 0)}")

    # Format data
    print("\n[2/4] Formatting data...")
    train_data, test_data, scaling_params = format_site_data_from_csv(
        df, N_sites=10, train_test_split=0.8, sample_fraction=0.9,
        background_site_balance=1, seed=42
    )
    print(f"  ✓ Train: {train_data.n_locations} locations, Test: {test_data.n_locations} locations")

    # Save training info
    train_info = {
        'n_locations': train_data.n_locations,
        'n_sites': train_data.n_sites,
        'n_background': train_data.n_background,
        'n_samples_per_location': float(np.mean([len(c.samples) for c in train_data.collections])),
        'n_features': 3
    }
    with open(output_dir / "python_shared_training_info.json", 'w') as f:
        json.dump(train_info, f, indent=2)

    # Save sample data
    sample_data = []
    for i, coll in enumerate(train_data.collections[:3]):
        sample_data.append({
            'id': i,
            'n_samples': len(coll.samples),
            'mean': coll.samples.mean(axis=0).tolist(),
            'first_sample': coll.samples[0].tolist()
        })
    with open(output_dir / "python_shared_sample_data.json", 'w') as f:
        json.dump(sample_data, f, indent=2)

    # Build kernel
    print("\n[3/4] Building kernel matrix...")
    sigma = 0.5
    rbf_kernel = RBFKernel(sigma=sigma)
    me_kernel = MeanEmbeddingKernel(base_kernel=rbf_kernel)

    K = me_kernel.build_similarity_matrix(
        train_data.collections,
        round_kernel=True,
        kernel_decimals=3
    )

    np.save(output_dir / "python_shared_kernel.npy", np.array(K))

    kernel_stats = {
        'shape': list(K.shape),
        'mean': float(jnp.mean(K)),
        'std': float(jnp.std(K)),
        'min': float(jnp.min(K)),
        'max': float(jnp.max(K)),
        'diagonal_mean': float(jnp.mean(jnp.diag(K))),
        'off_diagonal_mean': float(jnp.mean(K - jnp.diag(jnp.diag(K))))
    }
    with open(output_dir / "python_shared_kernel_stats.json", 'w') as f:
        json.dump(kernel_stats, f, indent=2)

    print(f"  ✓ Kernel: {K.shape}, mean={jnp.mean(K):.6f}")

    # Fit model
    print("\n[4/4] Fitting KLR model...")
    labels = jnp.array([c.label for c in train_data.collections])
    np.save(output_dir / "python_shared_labels.npy", np.array(labels))

    klr = KernelLogisticRegression(lambda_reg=0.1, max_iter=100, tol=0.001)
    result = klr.fit(K, labels)

    np.save(output_dir / "python_shared_alpha.npy", np.array(result.alpha))

    # Training predictions
    train_pred = klr.predict_proba(K, result.alpha)
    np.save(output_dir / "python_shared_predictions.npy", np.array(train_pred))

    print(f"  ✓ Converged: {result.converged} in {result.n_iterations} iterations")
    print(f"  ✓ Alpha: mean={jnp.mean(result.alpha):.6f}, std={jnp.std(result.alpha):.6f}")
    print(f"  ✓ Predictions: mean={jnp.mean(train_pred):.6f}")

    print("\n" + "=" * 80)
    print("Done! Compare with R outputs.")
    print("=" * 80)


if __name__ == "__main__":
    main()
