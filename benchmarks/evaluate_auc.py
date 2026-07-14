#!/usr/bin/env python3
"""
Evaluate both Python and R implementations on the same test data.
Compute AUC to determine which implementation is correct.
"""

import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klrfome.api import KLRfome
from klrfome.data.formats import RasterStack
from klrfome.utils.validation import compute_roc_auc
import geopandas as gpd


def evaluate_python_auc(data_dir="benchmark_data", output_dir="diagnostic_output"):
    """Evaluate Python implementation and compute AUC on test data."""

    print("=" * 80)
    print("PYTHON AUC EVALUATION")
    print("=" * 80)

    # Load data
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

    # Initialize model
    model = KLRfome(
        sigma=0.5,
        lambda_reg=0.1,
        n_rff_features=0,  # Use exact kernel to match R implementation
        window_size=5,
        seed=42
    )

    # Prepare raw data
    raw_training_data = model.prepare_data(
        raster_stack=raster_stack,
        sites=sites_gdf,
        n_background=50,
        samples_per_location=20,
        site_buffer=0.01,
        background_exclusion_buffer=0.02
    )

    # Format data to match R (same as diagnostic)
    from benchmarks.format_site_data_python import format_site_data_python
    import importlib.util
    format_module_path = Path(__file__).parent / "format_site_data_python.py"
    spec = importlib.util.spec_from_file_location("format_site_data_python", format_module_path)
    format_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(format_module)
    format_site_data_python = format_module.format_site_data_python

    training_data, test_data, scaling_params = format_site_data_python(
        raw_training_data,
        N_sites=10,
        train_test_split=0.8,
        sample_fraction=0.9,
        background_site_balance=1,
        seed=42
    )

    print(f"\nTraining data: {training_data.n_locations} locations")
    print(f"Test data: {test_data.n_locations} locations")

    # Fit model
    print("\nFitting model...")
    # Set tolerance to match R evaluation scripts (tol=0.001)
    model._klr.tol = 0.001
    model.fit(training_data)
    print(f"  Converged: {model._fit_result.converged}")
    print(f"  Iterations: {model._fit_result.n_iterations}")

    # Diagnostic: Check alpha values
    alpha = model._fit_result.alpha
    print(f"\nAlpha diagnostics:")
    print(f"  Alpha shape: {alpha.shape}")
    print(f"  Alpha range: [{jnp.min(alpha):.6f}, {jnp.max(alpha):.6f}]")
    print(f"  Alpha mean: {jnp.mean(alpha):.6f}")
    print(f"  Alpha std: {jnp.std(alpha):.6f}")
    print(f"  Alpha sum: {jnp.sum(alpha):.6f}")
    print(f"  Alpha abs max: {jnp.max(jnp.abs(alpha)):.6f}")

    # Check if alpha values are reasonable (R typically has values in range [-10, 10])
    if jnp.max(jnp.abs(alpha)) < 1.0:
        print(f"  ⚠ WARNING: Alpha values seem small (max abs={jnp.max(jnp.abs(alpha)):.6f})")
        print(f"    This might indicate the model isn't learning strongly enough")

    # Diagnostic: Check similarity matrix
    K_train = model._similarity_matrix
    print(f"\nTraining kernel matrix diagnostics:")
    print(f"  K shape: {K_train.shape}")
    print(f"  K range: [{jnp.min(K_train):.6f}, {jnp.max(K_train):.6f}]")
    print(f"  K mean: {jnp.mean(K_train):.6f}")
    print(f"  K diagonal mean: {jnp.mean(jnp.diag(K_train)):.6f}")

    # Predict on test data
    print("\nPredicting on test data...")
    # Build kernel matrix between test and training data
    kernel = model._distribution_kernel

    # CRITICAL: Ensure test data order matches R exactly
    # R's test_data is a list, and test_presence is computed from names
    # We need to ensure our collections are in the same order as R's test_data
    print(f"\nTest data order check:")
    print(f"  Test collections: {len(test_data.collections)}")
    print(f"  Test labels distribution: {np.bincount([c.label for c in test_data.collections])}")

    # For each test collection, compute similarity to all training collections
    test_predictions = []
    test_labels = []
    K_test_all = []  # Store all K_test rows for diagnostics

    # IMPORTANT: Iterate in the same order as R's test_data list
    for idx, test_coll in enumerate(test_data.collections):
        # Compute kernel between this test collection and all training collections
        K_test = []
        for train_coll in training_data.collections:
            k_val = kernel(test_coll.samples, train_coll.samples)
            # Round to 3 decimals to match R's KLR_predict (line 132)
            k_val = round(float(k_val), 3)
            K_test.append(k_val)

        K_test = jnp.array(K_test)
        K_test_all.append(K_test)

        # Predict probability - match R's simple sigmoid formula exactly
        # R uses: pred <- 1 / (1 + exp(-as.vector(kstark %*% alphas_pred)))
        eta = K_test @ model._fit_result.alpha
        # Use simple sigmoid to match R (not numerically stable version)
        # R doesn't use numerical stability, so we match it exactly
        prob = 1 / (1 + jnp.exp(-eta))
        test_predictions.append(float(prob))
        test_labels.append(test_coll.label)

    # Diagnostic: Check test kernel values
    K_test_all = jnp.array(K_test_all)
    print(f"\nTest kernel matrix diagnostics:")
    print(f"  K_test shape: {K_test_all.shape}")
    print(f"  K_test range: [{jnp.min(K_test_all):.6f}, {jnp.max(K_test_all):.6f}]")
    print(f"  K_test mean: {jnp.mean(K_test_all):.6f}")
    print(f"  K_test std: {jnp.std(K_test_all):.6f}")

    # Diagnostic: Check unique values (to see if rounding made everything the same)
    unique_k_vals = len(jnp.unique(K_test_all))
    print(f"  K_test unique values: {unique_k_vals} out of {K_test_all.size}")

    # Check if kernel values are too similar (might indicate sigma issue)
    k_std = jnp.std(K_test_all)
    if k_std < 0.1:
        print(f"  ⚠ WARNING: Kernel values have low std ({k_std:.6f})")
        print(f"    This might indicate sigma is too large, making all kernels similar")

    # Show distribution of kernel values
    k_min, k_max = jnp.min(K_test_all), jnp.max(K_test_all)
    k_range = k_max - k_min
    print(f"  K_test value range: {k_range:.6f} (from {k_min:.6f} to {k_max:.6f})")
    if k_range < 0.5:
        print(f"  ⚠ WARNING: Kernel value range is narrow ({k_range:.6f})")
        print(f"    This will constrain eta values and limit prediction range")

    # Diagnostic: Check eta and prob values (using same formula as predictions)
    etas = jnp.array([K_test_all[i] @ alpha for i in range(len(test_labels))])
    probs = 1 / (1 + jnp.exp(-etas))  # Match R's simple sigmoid
    print(f"\nPrediction diagnostics:")
    print(f"  Eta range: [{jnp.min(etas):.6f}, {jnp.max(etas):.6f}]")
    print(f"  Eta mean: {jnp.mean(etas):.6f}")
    print(f"  Prob range: [{jnp.min(probs):.6f}, {jnp.max(probs):.6f}]")
    print(f"  Prob mean: {jnp.mean(probs):.6f}")
    print(f"  Unique prob values: {len(jnp.unique(probs))} out of {len(probs)}")

    # Additional diagnostic: Compare actual predictions vs diagnostic probs
    actual_preds = np.array(test_predictions)
    print(f"  Actual pred range: [{actual_preds.min():.6f}, {actual_preds.max():.6f}]")
    print(f"  Diagnostic prob range: [{probs.min():.6f}, {probs.max():.6f}]")
    if not np.allclose(actual_preds, np.array(probs), atol=1e-6):
        print(f"  ⚠ WARNING: Actual predictions differ from diagnostic probs!")
        print(f"    Max difference: {np.max(np.abs(actual_preds - np.array(probs))):.6f}")

    test_predictions = np.array(test_predictions)
    test_labels = np.array(test_labels)

    # CRITICAL DIAGNOSTIC: Check if predictions are inverted
    # If AUC < 0.5, predictions might be inverted
    pos_mean_pred = test_predictions[test_labels == 1].mean() if np.sum(test_labels == 1) > 0 else 0
    neg_mean_pred = test_predictions[test_labels == 0].mean() if np.sum(test_labels == 0) > 0 else 0
    print(f"\nPrediction-label correlation check:")
    print(f"  Mean prediction for positive (label=1): {pos_mean_pred:.6f}")
    print(f"  Mean prediction for negative (label=0): {neg_mean_pred:.6f}")
    if pos_mean_pred < neg_mean_pred:
        print(f"  ⚠ CRITICAL: Positive samples have LOWER predictions than negative!")
        print(f"    This indicates predictions are inverted or labels are wrong!")
        print(f"    Expected: pos_mean > neg_mean for good model")

    # Check for degenerate predictions (all same value)
    unique_preds = len(np.unique(test_predictions))
    if unique_preds == 1:
        print(f"\n⚠ WARNING: All predictions are identical ({test_predictions[0]:.6f})!")
        print(f"  This indicates the model is not learning or kernel matrix is degenerate.")
        print(f"  Possible causes:")
        print(f"    - Kernel values all rounded to same value (check K_test unique values above)")
        print(f"    - Alpha values all zero or constant (check Alpha diagnostics above)")
        print(f"    - Sigma too small/large for data scale")
        print(f"    - Numerical issues in kernel computation")

    # Compute AUC
    # Note: compute_roc_auc expects (pred, obs), not (obs, pred)
    try:
        auc = compute_roc_auc(test_predictions, test_labels)
    except Exception as e:
        print(f"\n⚠ Error computing AUC: {e}")
        print(f"  This often happens when all predictions are the same.")
        auc = 0.5  # Default to random

    print(f"\nResults:")
    print(f"  Test samples: {len(test_labels)}")
    print(f"  Positive samples: {np.sum(test_labels == 1)}")
    print(f"  Negative samples: {np.sum(test_labels == 0)}")
    print(f"  Prediction range: [{test_predictions.min():.6f}, {test_predictions.max():.6f}]")
    print(f"  Prediction mean: {test_predictions.mean():.6f}")
    print(f"  Unique predictions: {unique_preds} out of {len(test_predictions)}")
    print(f"  AUC: {auc:.6f}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = {
        'auc': float(auc),
        'n_test': len(test_labels),
        'n_positive': int(np.sum(test_labels == 1)),
        'n_negative': int(np.sum(test_labels == 0)),
        'prediction_range': [float(test_predictions.min()), float(test_predictions.max())],
        'prediction_mean': float(test_predictions.mean()),
        'test_predictions': test_predictions.tolist(),
        'test_labels': test_labels.tolist()
    }

    with open(output_path / "python_auc_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path / 'python_auc_results.json'}")

    # Try to load R results for comparison
    r_results_file = Path(output_dir) / "r_auc_results.json"
    if r_results_file.exists():
        with open(r_results_file) as f:
            r_results = json.load(f)

        print(f"\n" + "=" * 80)
        print("COMPARISON WITH R RESULTS")
        print("=" * 80)
        print(f"  R AUC: {r_results.get('auc', 'N/A'):.6f}")
        print(f"  Python AUC: {auc:.6f}")
        print(f"  Difference: {abs(r_results.get('auc', 0) - auc):.6f}")

        # Compare prediction ranges
        r_pred_range = r_results.get('prediction_range', [])
        if len(r_pred_range) == 2:
            print(f"\n  Prediction ranges:")
            print(f"    R:      [{r_pred_range[0]:.6f}, {r_pred_range[1]:.6f}]")
            print(f"    Python: [{test_predictions.min():.6f}, {test_predictions.max():.6f}]")

        # Compare label distributions
        r_n_pos = r_results.get('n_positive', 0)
        r_n_neg = r_results.get('n_negative', 0)
        print(f"\n  Test data:")
        print(f"    R:      {r_n_pos} pos, {r_n_neg} neg")
        print(f"    Python: {np.sum(test_labels == 1)} pos, {np.sum(test_labels == 0)} neg")

        if r_n_pos != np.sum(test_labels == 1) or r_n_neg != np.sum(test_labels == 0):
            print(f"  ⚠ WARNING: Test data split differs from R!")
            print(f"    This will cause different AUC even if predictions are correct")

    return results


def evaluate_r_auc(data_dir="benchmark_data", output_dir="diagnostic_output"):
    """Evaluate R implementation and compute AUC on test data."""

    import subprocess

    print("\n" + "=" * 80)
    print("R AUC EVALUATION")
    print("=" * 80)

    # Create R script for AUC evaluation
    r_script = Path(__file__).parent / "evaluate_r_auc.R"

    if not r_script.exists():
        print("  Creating R AUC evaluation script...")
        create_r_auc_script(r_script)

    print(f"\nRunning R AUC evaluation...")
    result = subprocess.run(
        ["Rscript", str(r_script), data_dir, output_dir],
        text=True
    )

    if result.returncode != 0:
        print(f"  ❌ R script failed with exit code {result.returncode}")
        return None

    # Load results
    results_file = Path(output_dir) / "r_auc_results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        print(f"\n✓ Results loaded from: {results_file}")
        return results
    else:
        print(f"  ⚠ Results file not found: {results_file}")
        return None


def create_r_auc_script(r_script_path):
    """Create R script for AUC evaluation."""
    r_script_content = '''#!/usr/bin/env Rscript
# R AUC evaluation script

source("benchmarks/klrfome_r_functions.R")

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if(length(args) > 0) args[1] else "benchmark_data"
output_dir <- if(length(args) > 1) args[2] else "diagnostic_output"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("================================================================================\\n")
cat("R AUC EVALUATION\\n")
cat("================================================================================\\n")

# Load and extract data (same as diagnostic)
set.seed(42)
metadata <- fromJSON(file.path(data_dir, "metadata.json"))
rast_stack <- terra::rast(file.path(data_dir, paste0(metadata$band_names, ".tif")))
names(rast_stack) <- metadata$band_names

sites_file <- file.path(data_dir, "sites.shp")
if (!file.exists(sites_file)) sites_file <- file.path(data_dir, "sites.geojson")
sites_sf <- st_read(sites_file, quiet = TRUE)

# Extract data
site_values <- terra::extract(rast_stack, sites_sf)
n_samples_per_site <- 20
site_data_list <- list()

for (i in 1:nrow(sites_sf)) {
  site_vals <- as.numeric(site_values[i, -1])
  for (j in 1:n_samples_per_site) {
    noise <- rnorm(length(metadata$band_names), mean = 0, sd = 0.1)
    sample_vals <- site_vals + noise * abs(site_vals)
    df_row <- data.frame(presence = 1, SITENO = paste0("Site", i), stringsAsFactors = FALSE)
    for(v in 1:length(metadata$band_names)){
      df_row[[metadata$band_names[v]]] <- sample_vals[v]
    }
    site_data_list[[length(site_data_list) + 1]] <- df_row
  }
}

n_background <- 50
n_samples_per_bg <- 20
bg_coords <- terra::spatSample(rast_stack, n_background * n_samples_per_bg, as.points = TRUE)
bg_values <- terra::extract(rast_stack, bg_coords)

for (i in 1:nrow(bg_values)) {
  bg_vals <- as.numeric(bg_values[i, -1])
  df_row <- data.frame(presence = 0, SITENO = "background", stringsAsFactors = FALSE)
  for(v in 1:length(metadata$band_names)){
    df_row[[metadata$band_names[v]]] <- bg_vals[v]
  }
  site_data_list[[length(site_data_list) + 1]] <- df_row
}

sim_data <- do.call(rbind, site_data_list)
sim_data <- sim_data[complete.cases(sim_data), ]

# Format data
formatted_data <- format_site_data(
  sim_data, N_sites = 10, train_test_split = 0.8,
  sample_fraction = 0.9, background_site_balance = 1
)

train_data <- formatted_data[["train_data"]]
train_presence <- formatted_data[["train_presence"]]
test_data <- formatted_data[["test_data"]]
test_presence <- formatted_data[["test_presence"]]

cat(sprintf("\\nTraining data: %d locations\\n", length(train_data)))
cat(sprintf("Test data: %d locations\\n", length(test_data)))

# Build kernel and fit
cat("\\nFitting model...\\n")
sigma <- 0.5
lambda <- 0.1
K <- build_K(train_data, sigma = sigma, progress = FALSE, dist_metric = "euclidean")
klr_result <- KLR(K, train_presence, lambda = lambda, maxiter = 100, tol = 0.001, verbose = 0)
cat(sprintf("  Converged: %s\\n", klr_result$converged))
cat(sprintf("  Iterations: %d\\n", klr_result$iterations))

# Predict on test data
cat("\\nPredicting on test data...\\n")
test_predictions <- KLR_predict(test_data, train_data, klr_result$alphas, sigma, progress = FALSE)

# Compute AUC using pROC package
if (!require("pROC", quietly = TRUE)) {
  # Fallback: manual AUC calculation
  cat("  Computing AUC manually (pROC not available)...\\n")

  # Sort by predictions
  ord <- order(test_predictions, decreasing = TRUE)
  sorted_pred <- test_predictions[ord]
  sorted_labels <- test_presence[ord]

  # Count positives and negatives
  n_pos <- sum(sorted_labels == 1)
  n_neg <- sum(sorted_labels == 0)

  if (n_pos == 0 || n_neg == 0) {
    auc <- 0.5
  } else {
    # Calculate AUC using trapezoidal rule
    tpr <- cumsum(sorted_labels == 1) / n_pos
    fpr <- cumsum(sorted_labels == 0) / n_neg

    # AUC = area under ROC curve
    auc <- sum(diff(fpr) * (tpr[-1] + tpr[-length(tpr)]) / 2)
  }
} else {
  library(pROC)
  roc_obj <- roc(test_presence, test_predictions, quiet = TRUE)
  auc <- as.numeric(auc(roc_obj))
}

cat(sprintf("\\nResults:\\n"))
cat(sprintf("  Test samples: %d\\n", length(test_presence)))
cat(sprintf("  Positive samples: %d\\n", sum(test_presence == 1)))
cat(sprintf("  Negative samples: %d\\n", sum(test_presence == 0)))
cat(sprintf("  Prediction range: [%.6f, %.6f]\\n", min(test_predictions), max(test_predictions)))
cat(sprintf("  Prediction mean: %.6f\\n", mean(test_predictions)))
cat(sprintf("  AUC: %.6f\\n", auc))

# Save results
results <- list(
  auc = as.numeric(auc),
  n_test = length(test_presence),
  n_positive = sum(test_presence == 1),
  n_negative = sum(test_presence == 0),
  prediction_range = c(min(test_predictions), max(test_predictions)),
  prediction_mean = mean(test_predictions),
  test_predictions = as.numeric(test_predictions),
  test_labels = as.numeric(test_presence)
)

write_json(results, file.path(output_dir, "r_auc_results.json"), auto_unbox = TRUE, pretty = TRUE)
cat(sprintf("\\n✓ Results saved to: %s\\n", file.path(output_dir, "r_auc_results.json")))

cat("\\n================================================================================\\n")
'''

    with open(r_script_path, 'w') as f:
        f.write(r_script_content)

    import os
    os.chmod(r_script_path, 0o755)


def compare_auc_results(py_results, r_results):
    """Compare AUC results from both implementations."""

    print("\n" + "=" * 80)
    print("AUC COMPARISON")
    print("=" * 80)

    if py_results is None or r_results is None:
        print("  ⚠ Missing results from one or both implementations")
        return

    py_auc = py_results['auc']
    r_auc = r_results['auc']

    print(f"\n  Python AUC: {py_auc:.6f}")
    print(f"  R AUC:       {r_auc:.6f}")
    print(f"  Difference:  {abs(py_auc - r_auc):.6f}")

    if py_auc > r_auc:
        print(f"\n  ✓ Python has higher AUC (better performance)")
        print(f"    Python is likely the correct implementation")
    elif r_auc > py_auc:
        print(f"\n  ✓ R has higher AUC (better performance)")
        print(f"    R is likely the correct implementation")
    else:
        print(f"\n  ≈ Both implementations have the same AUC")

    # Additional comparisons
    print(f"\n  Test data comparison:")
    print(f"    Python: {py_results['n_test']} samples ({py_results['n_positive']} pos, {py_results['n_negative']} neg)")
    print(f"    R:       {r_results['n_test']} samples ({r_results['n_positive']} pos, {r_results['n_negative']} neg)")

    print(f"\n  Prediction comparison:")
    py_range = py_results['prediction_range']
    r_range = r_results['prediction_range']
    print(f"    Python range: [{py_range[0]:.6f}, {py_range[1]:.6f}], mean={py_results['prediction_mean']:.6f}")
    print(f"    R range:      [{r_range[0]:.6f}, {r_range[1]:.6f}], mean={r_results['prediction_mean']:.6f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate AUC for both implementations')
    parser.add_argument('--data-dir', default='benchmark_data', help='Data directory')
    parser.add_argument('--output-dir', default='diagnostic_output', help='Output directory')
    parser.add_argument('--python-only', action='store_true', help='Run Python evaluation only')
    parser.add_argument('--r-only', action='store_true', help='Run R evaluation only')
    args = parser.parse_args()

    py_results = None
    r_results = None

    if not args.r_only:
        py_results = evaluate_python_auc(args.data_dir, args.output_dir)

    if not args.python_only:
        r_results = evaluate_r_auc(args.data_dir, args.output_dir)

    if py_results and r_results:
        compare_auc_results(py_results, r_results)

    print("\n" + "=" * 80)
    print("AUC EVALUATION COMPLETE")
    print("=" * 80)
