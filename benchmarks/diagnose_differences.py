#!/usr/bin/env python3
"""
Diagnostic script to compare Python and R implementations.
Saves intermediate results for comparison.
"""

import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klrfome.api import KLRfome
from klrfome.data.formats import RasterStack
import geopandas as gpd


def save_kernel_matrix(K, filename):
    """Save kernel matrix to file."""
    K_np = np.array(K)
    np.save(filename, K_np)
    print(f"  Saved kernel matrix: {filename} (shape: {K_np.shape}, mean: {K_np.mean():.6f})")
    return K_np


def save_array(arr, filename):
    """Save array to file."""
    arr_np = np.array(arr)
    np.save(filename, arr_np)
    print(f"  Saved array: {filename} (shape: {arr_np.shape})")
    return arr_np


def diagnose_python_workflow(data_dir="benchmark_data", output_dir="diagnostic_output"):
    """Run Python workflow with detailed diagnostics."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("=" * 80)
    print("PYTHON DIAGNOSTIC WORKFLOW")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    data_path = Path(data_dir)
    metadata_file = data_path / "metadata.json"
    with open(metadata_file) as f:
        metadata = json.load(f)

    raster_files = [data_path / f"{name}.tif" for name in metadata['band_names']]
    raster_stack = RasterStack.from_files(raster_files)
    raster_stack.band_names = metadata['band_names']

    sites_file = data_path / "sites.shp"
    if not sites_file.exists():
        sites_file = data_path / "sites.geojson"
    sites_gdf = gpd.read_file(sites_file)

    print(f"  Loaded {len(metadata['band_names'])} rasters")
    print(f"  Loaded {len(sites_gdf)} sites")

    # Initialize model
    print("\n[2] Initializing model...")
    model = KLRfome(
        sigma=0.5,
        lambda_reg=0.1,
        n_rff_features=256,
        window_size=5,
        seed=42
    )

    # Prepare data
    print("\n[3] Preparing training data...")
    training_data = model.prepare_data(
        raster_stack=raster_stack,
        sites=sites_gdf,
        n_background=50,
        samples_per_location=20,
        site_buffer=0.01,
        background_exclusion_buffer=0.02
    )

    # Save training data info
    train_info = {
        'n_locations': training_data.n_locations,
        'n_sites': training_data.n_sites,
        'n_background': training_data.n_background,
        'n_samples_per_location': training_data.samples_per_location,
        'n_features': training_data.n_features
    }
    with open(output_path / "python_training_info.json", 'w') as f:
        json.dump(train_info, f, indent=2)
    print(f"  Training data: {train_info}")

    # Save sample data from first few collections
    sample_data = []
    for i, coll in enumerate(training_data.collections[:5]):
        sample_data.append({
            'id': i,
            'n_samples': len(coll.samples),
            'mean': coll.samples.mean(axis=0).tolist(),
            'std': coll.samples.std(axis=0).tolist(),
            'first_sample': coll.samples[0].tolist() if len(coll.samples) > 0 else []
        })
    with open(output_path / "python_sample_data.json", 'w') as f:
        json.dump(sample_data, f, indent=2)

    # Build kernel matrix with diagnostics
    print("\n[4] Building kernel matrix...")
    kernel = model._kernel
    K = kernel.build_similarity_matrix(training_data.collections)
    K_np = save_kernel_matrix(K, output_path / "python_kernel_matrix.npy")

    # Save kernel statistics
    kernel_stats = {
        'shape': list(K_np.shape),
        'mean': float(K_np.mean()),
        'std': float(K_np.std()),
        'min': float(K_np.min()),
        'max': float(K_np.max()),
        'diagonal_mean': float(np.diag(K_np).mean()),
        'off_diagonal_mean': float(K_np[~np.eye(K_np.shape[0], dtype=bool)].mean()),
        'is_symmetric': bool(np.allclose(K_np, K_np.T)),
        'is_psd': bool(np.all(np.linalg.eigvals(K_np) >= -1e-10))
    }
    with open(output_path / "python_kernel_stats.json", 'w') as f:
        json.dump(kernel_stats, f, indent=2)
    print(f"  Kernel stats: {kernel_stats}")

    # Fit model with iteration tracking
    print("\n[5] Fitting model (tracking iterations)...")

    # Get labels
    y = training_data.labels
    save_array(y, output_path / "python_labels.npy")

    # Track IRLS iterations manually
    alpha_history = []
    iteration_details = []

    n = K.shape[0]
    lambda_reg = model._klr.lambda_reg
    max_iter = model._klr.max_iter
    tol = model._klr.tol
    min_prob = model._klr.min_prob

    alpha = jnp.zeros(n)
    alpha_history.append(np.array(alpha))

    for iteration in range(max_iter):
        # Compute probabilities
        eta = K @ alpha
        prob = 1 / (1 + jnp.exp(-jnp.clip(eta, -500, 500)))
        prob = jnp.clip(prob, min_prob, 1 - min_prob)

        # IRLS weights
        W = prob * (1 - prob)

        # Working response
        z = eta + (y - prob) / W

        # Weighted least squares update (Python formulation)
        KW = K * W[None, :]
        lhs = KW @ K + lambda_reg * jnp.eye(n)
        rhs = KW @ z

        try:
            alpha_new = jnp.linalg.solve(lhs, rhs)
        except Exception as e:
            print(f"  Error at iteration {iteration}: {e}")
            break

        # Track iteration
        alpha_history.append(np.array(alpha_new))

        # Save iteration details
        iter_detail = {
            'iteration': iteration,
            'alpha_mean': float(jnp.mean(alpha_new)),
            'alpha_std': float(jnp.std(alpha_new)),
            'alpha_min': float(jnp.min(alpha_new)),
            'alpha_max': float(jnp.max(alpha_new)),
            'eta_mean': float(jnp.mean(eta)),
            'prob_mean': float(jnp.mean(prob)),
            'delta_max': float(jnp.max(jnp.abs(alpha_new - alpha)))
        }
        iteration_details.append(iter_detail)

        # Check convergence
        delta = jnp.max(jnp.abs(alpha_new - alpha))
        alpha = alpha_new

        if delta < tol:
            print(f"  Converged at iteration {iteration + 1} (delta={delta:.6e})")
            break

    # Save alpha history
    alpha_array = np.array(alpha_history)
    save_array(alpha_array, output_path / "python_alpha_history.npy")

    with open(output_path / "python_iteration_details.json", 'w') as f:
        json.dump(iteration_details, f, indent=2)

    print(f"  Final alpha: mean={np.mean(alpha):.6f}, std={np.std(alpha):.6f}")
    print(f"  Total iterations: {len(alpha_history) - 1}")

    # Compute final predictions on training data
    eta_final = K @ alpha
    prob_final = 1 / (1 + jnp.exp(-eta_final))
    save_array(prob_final, output_path / "python_training_predictions.npy")

    print(f"  Training predictions: mean={np.mean(prob_final):.6f}, range=[{np.min(prob_final):.6f}, {np.max(prob_final):.6f}]")

    # Predict on raster (sample)
    print("\n[6] Predicting on raster (sample)...")
    predictions = model.predict(
        raster_stack=raster_stack,
        batch_size=500,
        show_progress=False
    )

    pred_stats = {
        'shape': list(predictions.shape),
        'mean': float(np.mean(predictions)),
        'std': float(np.std(predictions)),
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'n_valid': int(np.sum(~np.isnan(predictions)))
    }
    with open(output_path / "python_prediction_stats.json", 'w') as f:
        json.dump(pred_stats, f, indent=2)

    # Save a sample of predictions
    pred_sample = predictions[::10, ::10]  # Every 10th pixel
    save_array(pred_sample, output_path / "python_prediction_sample.npy")

    print(f"  Prediction stats: {pred_stats}")

    print("\n" + "=" * 80)
    print("Python diagnostic complete!")
    print(f"Results saved to: {output_path}")
    print("=" * 80)

    return {
        'kernel_matrix': K_np,
        'alpha_final': np.array(alpha),
        'predictions': predictions,
        'output_dir': str(output_path)
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose Python workflow')
    parser.add_argument('--data-dir', default='benchmark_data', help='Data directory')
    parser.add_argument('--output-dir', default='diagnostic_output', help='Output directory')
    args = parser.parse_args()

    diagnose_python_workflow(args.data_dir, args.output_dir)
