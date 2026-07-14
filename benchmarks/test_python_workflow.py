#!/usr/bin/env python3
"""
Python/JAX workflow test script for benchmarking.

This script loads the benchmark data and runs the full KLRfome workflow:
1. Load rasters and site points
2. Prepare training data
3. Fit KLR model
4. Predict on rasters
5. Report timing and performance metrics

Usage:
    python benchmarks/test_python_workflow.py [--data-dir DATA_DIR] [--sigma SIGMA] [--lambda LAMBDA]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klrfome import KLRfome, RasterStack
from klrfome.utils.validation import compute_roc_auc


def load_benchmark_data(data_dir: str):
    """Load benchmark data from directory."""
    data_path = Path(data_dir)
    metadata_file = data_path / "metadata.json"

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Load rasters using from_files method
    raster_files = [str(data_path / f"{band_name}.tif") for band_name in metadata['band_names']]

    # Check all files exist
    for rf in raster_files:
        if not Path(rf).exists():
            raise FileNotFoundError(f"Raster file not found: {rf}")

    # Load all rasters into a single RasterStack
    raster_stack = RasterStack.from_files(raster_files)

    # Override band names from metadata
    raster_stack.band_names = metadata['band_names']

    # Load sites
    sites_file = data_path / metadata['sites_file'].split('/')[-1]  # Get filename
    if not sites_file.exists():
        # Try GeoJSON
        sites_file = data_path / "sites.geojson"

    sites_gdf = gpd.read_file(sites_file)

    return raster_stack, sites_gdf, metadata


def run_workflow(
    data_dir: str = "benchmark_data",
    sigma: float = 0.5,
    lambda_reg: float = 0.1,
    n_rff_features: int = 256,
    window_size: int = 5,
    n_background: int = 50,
    samples_per_location: int = 20,
    batch_size: int = 500,
    seed: int = 42
):
    """Run the full KLRfome workflow and report timing."""

    print("=" * 80)
    print("Python/JAX KLRfome Workflow Benchmark")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading benchmark data...")
    start_time = time.time()
    raster_stack, sites_gdf, metadata = load_benchmark_data(data_dir)
    load_time = time.time() - start_time
    print(f"  ✓ Loaded {len(metadata['band_names'])} rasters ({raster_stack.data.shape[1]}x{raster_stack.data.shape[2]})")
    print(f"  ✓ Loaded {len(sites_gdf)} site locations")
    print(f"  Time: {load_time:.3f}s")

    # Initialize model
    print(f"\n[2/5] Initializing model (sigma={sigma}, lambda={lambda_reg})...")
    start_time = time.time()
    model = KLRfome(
        sigma=sigma,
        lambda_reg=lambda_reg,
        n_rff_features=n_rff_features,
        window_size=window_size,
        seed=seed
    )
    init_time = time.time() - start_time
    print(f"  ✓ Model initialized")
    print(f"  Time: {init_time:.3f}s")

    # Prepare training data
    print(f"\n[3/5] Preparing training data...")
    start_time = time.time()
    training_data = model.prepare_data(
        raster_stack=raster_stack,
        sites=sites_gdf,
        n_background=n_background,
        samples_per_location=samples_per_location,
        site_buffer=0.01,
        background_exclusion_buffer=0.02
    )
    prep_time = time.time() - start_time
    print(f"  ✓ Prepared {training_data.n_locations} locations")
    print(f"    Sites: {training_data.n_sites}, Background: {training_data.n_background}")
    print(f"  Time: {prep_time:.3f}s")

    # Fit model
    print(f"\n[4/5] Fitting KLR model...")
    start_time = time.time()
    model.fit(training_data)
    fit_time = time.time() - start_time
    print(f"  ✓ Model fitted")
    if model._fit_result.converged:
        print(f"    Converged in {model._fit_result.n_iterations} iterations")
        print(f"    Final loss: {model._fit_result.final_loss:.6f}")
    else:
        print(f"    ⚠ Did not converge after {model._fit_result.n_iterations} iterations")
    print(f"  Time: {fit_time:.3f}s")

    # Predict
    print(f"\n[5/5] Predicting on raster...")
    start_time = time.time()
    predictions = model.predict(
        raster_stack=raster_stack,
        batch_size=batch_size,
        show_progress=False
    )
    predict_time = time.time() - start_time
    print(f"  ✓ Predictions generated: {predictions.shape}")
    print(f"    Range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
    print(f"    Mean: {np.mean(predictions):.3f}")
    print(f"  Time: {predict_time:.3f}s")

    # Summary
    total_time = load_time + init_time + prep_time + fit_time + predict_time
    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    print(f"  Load data:        {load_time:8.3f}s ({load_time/total_time*100:5.1f}%)")
    print(f"  Initialize model: {init_time:8.3f}s ({init_time/total_time*100:5.1f}%)")
    print(f"  Prepare data:     {prep_time:8.3f}s ({prep_time/total_time*100:5.1f}%)")
    print(f"  Fit model:        {fit_time:8.3f}s ({fit_time/total_time*100:5.1f}%)")
    print(f"  Predict:          {predict_time:8.3f}s ({predict_time/total_time*100:5.1f}%)")
    print(f"  {'-' * 60}")
    print(f"  TOTAL:            {total_time:8.3f}s")
    print("=" * 80)

    return {
        'load_time': load_time,
        'init_time': init_time,
        'prep_time': prep_time,
        'fit_time': fit_time,
        'predict_time': predict_time,
        'total_time': total_time,
        'n_iterations': model._fit_result.n_iterations if model._fit_result.converged else None,
        'converged': model._fit_result.converged,
        'predictions_shape': predictions.shape,
        'predictions_range': [float(np.min(predictions)), float(np.max(predictions))],
        'predictions_mean': float(np.mean(predictions))
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Python/JAX KLRfome workflow benchmark"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='benchmark_data',
        help='Directory containing benchmark data (default: benchmark_data)'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=0.5,
        help='Kernel bandwidth sigma (default: 0.5)'
    )
    parser.add_argument(
        '--lambda',
        type=float,
        dest='lambda_reg',
        default=0.1,
        help='Regularization strength lambda (default: 0.1)'
    )
    parser.add_argument(
        '--n-rff-features',
        type=int,
        default=256,
        help='Number of RFF features (default: 256, 0 for exact kernel)'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=5,
        help='Focal window size (default: 5)'
    )
    parser.add_argument(
        '--n-background',
        type=int,
        default=50,
        help='Number of background locations (default: 50)'
    )
    parser.add_argument(
        '--samples-per-location',
        type=int,
        default=20,
        help='Samples per location (default: 20)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=500,
        help='Batch size for prediction (default: 500)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for results (optional)'
    )

    args = parser.parse_args()

    results = run_workflow(
        data_dir=args.data_dir,
        sigma=args.sigma,
        lambda_reg=args.lambda_reg,
        n_rff_features=args.n_rff_features,
        window_size=args.window_size,
        n_background=args.n_background,
        samples_per_location=args.samples_per_location,
        batch_size=args.batch_size,
        seed=args.seed
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")


if __name__ == '__main__':
    main()
