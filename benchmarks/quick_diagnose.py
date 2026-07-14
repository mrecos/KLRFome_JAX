#!/usr/bin/env python3
"""
Quick diagnostic script to compare Python and R implementations.
Skips expensive steps (raster prediction) and focuses on core differences:
- Kernel matrix values
- IRLS algorithm formulation
- Training data extraction
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
import geopandas as gpd


def save_array(arr, filename):
    """Save array to file."""
    arr_np = np.array(arr)
    np.save(filename, arr_np)
    return arr_np


def quick_diagnose_python(data_dir="benchmark_data", output_dir="diagnostic_output", verbose=True):
    """Quick Python diagnostic - skips raster prediction."""

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("=" * 80)
    print("QUICK PYTHON DIAGNOSTIC (No Raster Prediction)")
    print("=" * 80)

    # Load data
    vprint("\n[1/4] Loading data...")
    vprint("  Reading metadata...")
    data_path = Path(data_dir)
    metadata_file = data_path / "metadata.json"
    with open(metadata_file) as f:
        metadata = json.load(f)
    vprint(f"  ✓ Metadata loaded: {len(metadata['band_names'])} bands")

    vprint("  Loading raster files...")
    raster_files = [data_path / f"{name}.tif" for name in metadata['band_names']]
    for i, rf in enumerate(raster_files, 1):
        vprint(f"    [{i}/{len(raster_files)}] Loading {rf.name}...", end='\r')
        if i == len(raster_files):
            print()  # New line after last file
    raster_stack = RasterStack.from_files(raster_files)
    raster_stack.band_names = metadata['band_names']
    vprint(f"  ✓ Rasters loaded: shape {raster_stack.data.shape}")

    vprint("  Loading site locations...")
    sites_file = data_path / "sites.shp"
    if not sites_file.exists():
        sites_file = data_path / "sites.geojson"
    sites_gdf = gpd.read_file(sites_file)
    vprint(f"  ✓ Sites loaded: {len(sites_gdf)} locations")

    print(f"  ✓ Loaded {len(metadata['band_names'])} rasters, {len(sites_gdf)} sites")

    # Initialize model
    vprint("\n[2/4] Initializing model...")
    vprint("  Creating KLRfome model with:")
    vprint("    sigma=0.5, lambda=0.1, n_rff_features=0 (exact kernel to match R)")
    model = KLRfome(
        sigma=0.5,
        lambda_reg=0.1,
        n_rff_features=0,  # Use exact kernel to match R implementation
        window_size=5,
        seed=42
    )
    vprint("  ✓ Model initialized")

    # Prepare data
    vprint("\n[3/4] Preparing training data...")
    vprint("  Extracting data at site locations...")
    raw_training_data = model.prepare_data(
        raster_stack=raster_stack,
        sites=sites_gdf,
        n_background=50,
        samples_per_location=20,
        site_buffer=0.01,
        background_exclusion_buffer=0.02
    )
    vprint("  ✓ Data extraction complete")
    vprint(f"    Raw data: {raw_training_data.n_locations} locations")

    # Format data to match R's format_site_data (reduces to N_sites, splits train/test, etc.)
    vprint("  Formatting data to match R implementation...")
    # Import from same directory
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
    vprint(f"  ✓ Formatted: {training_data.n_locations} train, {test_data.n_locations} test")

    # Save training data info
    # Calculate average samples per location from collections
    avg_samples = np.mean([len(coll.samples) for coll in training_data.collections]) if training_data.collections else 0
    # Get number of features from first collection
    n_features = len(training_data.collections[0].samples[0]) if training_data.collections and len(training_data.collections[0].samples) > 0 else 0

    train_info = {
        'n_locations': training_data.n_locations,
        'n_sites': training_data.n_sites,
        'n_background': training_data.n_background,
        'n_samples_per_location': float(avg_samples),
        'n_features': n_features
    }
    with open(output_path / "python_training_info.json", 'w') as f:
        json.dump(train_info, f, indent=2)
    print(f"  ✓ Training: {train_info['n_locations']} locations, {train_info['n_features']} features")

    # Save sample data from first few collections
    sample_data = []
    for i, coll in enumerate(training_data.collections[:3]):
        sample_data.append({
            'id': i,
            'n_samples': len(coll.samples),
            'mean': coll.samples.mean(axis=0).tolist(),
            'first_sample': coll.samples[0].tolist() if len(coll.samples) > 0 else []
        })
    with open(output_path / "python_sample_data.json", 'w') as f:
        json.dump(sample_data, f, indent=2)

    # Build kernel matrix
    vprint("\n[4/4] Building kernel matrix and fitting model...")
    vprint("  Building similarity matrix...")
    vprint(f"    Number of collections: {len(training_data.collections)}")
    kernel = model._distribution_kernel
    vprint("    Computing pairwise similarities...")
    if hasattr(kernel, '_use_rff') and kernel._use_rff:
        vprint("    Using RFF approximation (fast)")
        vprint("    Computing mean embeddings...", end=' ', flush=True)
    else:
        vprint("    Using exact kernel (slower)")
        n_collections = len(training_data.collections)
        total_pairs = n_collections * (n_collections + 1) // 2
        vprint(f"    Total pairs to compute: {total_pairs}")
        vprint("    Progress: ", end='', flush=True)

    import time
    start_time = time.time()
    # Apply kernel rounding to match R implementation (round to 3 decimals)
    K = kernel.build_similarity_matrix(
        training_data.collections,
        round_kernel=True,  # Match R's rounding
        kernel_decimals=3
    )
    elapsed = time.time() - start_time
    vprint(f"done ({elapsed:.2f}s)")
    vprint("  ✓ Kernel matrix built")
    K_np = save_array(K, output_path / "python_kernel_matrix.npy")
    vprint(f"  ✓ Saved kernel matrix: {K_np.shape}")

    # Save kernel statistics
    kernel_stats = {
        'shape': list(K_np.shape),
        'mean': float(K_np.mean()),
        'std': float(K_np.std()),
        'min': float(K_np.min()),
        'max': float(K_np.max()),
        'diagonal_mean': float(np.diag(K_np).mean()),
        'off_diagonal_mean': float(K_np[~np.eye(K_np.shape[0], dtype=bool)].mean()),
        'is_symmetric': bool(np.allclose(K_np, K_np.T, atol=1e-6)),
        'has_rounding': not np.allclose(K_np, np.round(K_np, 3))  # Check if values are rounded
    }
    with open(output_path / "python_kernel_stats.json", 'w') as f:
        json.dump(kernel_stats, f, indent=2)
    print(f"  ✓ Kernel matrix: {K_np.shape}, mean={K_np.mean():.6f}")
    print(f"    Values rounded to 3 decimals: {not kernel_stats['has_rounding']}")

    # Get labels
    y = training_data.labels
    save_array(y, output_path / "python_labels.npy")

    # Track IRLS iterations (first 5 iterations only for speed)
    vprint("  Running IRLS (tracking first 5 iterations)...")
    vprint("    Initializing alpha (matching R: 1/N)...")
    alpha_history = []
    iteration_details = []

    n = K.shape[0]
    lambda_reg = model._klr.lambda_reg
    max_iter = 5  # Only track first 5 iterations
    tol = model._klr.tol
    min_prob = model._klr.min_prob

    vprint(f"    Matrix size: {n}x{n}, lambda={lambda_reg}, tol={tol}")
    alpha = jnp.ones(n) / n  # Match R initialization: rep(1/N, N)
    alpha_history.append(np.array(alpha))
    vprint(f"    Initial alpha: mean={np.mean(alpha):.6f}")

    for iteration in range(max_iter):
        vprint(f"    Iteration {iteration + 1}/{max_iter}...", end=' ', flush=True)
        # Compute probabilities
        eta = K @ alpha
        prob = 1 / (1 + jnp.exp(-jnp.clip(eta, -500, 500)))
        prob = jnp.clip(prob, min_prob, 1 - min_prob)

        # IRLS weights
        W = prob * (1 - prob)

        # Working response
        z = eta + (y - prob) / W

        # R IRLS formulation: (K + λ·diag(1/W)) α = z
        diagW_inv = 1.0 / W  # 1/diagW (inverse of diagonal weights)
        lhs = K + lambda_reg * jnp.diag(diagW_inv)
        rhs = z

        try:
            vprint("solving linear system...", end=' ', flush=True)
            alpha_new = jnp.linalg.solve(lhs, rhs)
            vprint("done")
        except Exception as e:
            vprint(f"\n    ❌ Error at iteration {iteration}: {e}")
            break

        alpha_history.append(np.array(alpha_new))

        # Save iteration details
        delta = jnp.max(jnp.abs(alpha_new - alpha))
        iter_detail = {
            'iteration': iteration,
            'alpha_mean': float(jnp.mean(alpha_new)),
            'alpha_std': float(jnp.std(alpha_new)),
            'prob_mean': float(jnp.mean(prob)),
            'delta_max': float(delta),
            'lhs_diag_mean': float(jnp.mean(jnp.diag(lhs))),  # For comparison
            'rhs_mean': float(jnp.mean(rhs))
        }
        iteration_details.append(iter_detail)

        vprint(f"      alpha_mean={iter_detail['alpha_mean']:.6f}, delta={delta:.6e}")
        alpha = alpha_new

        if delta < tol:
            vprint(f"    ✓ Converged at iteration {iteration + 1}")
            break

    # Save alpha history (first 5 iterations)
    vprint("  Saving results...")
    alpha_array = np.array(alpha_history)
    save_array(alpha_array, output_path / "python_alpha_history.npy")
    vprint("    ✓ Alpha history saved")

    with open(output_path / "python_iteration_details.json", 'w') as f:
        json.dump(iteration_details, f, indent=2)
    vprint("    ✓ Iteration details saved")

    print(f"  ✓ IRLS: {len(alpha_history) - 1} iterations tracked")
    print(f"    Final alpha: mean={np.mean(alpha):.6f}, std={np.std(alpha):.6f}")

    # Compute training predictions only
    vprint("  Computing training predictions...")
    eta_final = K @ alpha
    prob_final = 1 / (1 + jnp.exp(-eta_final))
    save_array(prob_final, output_path / "python_training_predictions.npy")
    vprint("    ✓ Training predictions saved")

    print(f"  ✓ Training predictions: mean={np.mean(prob_final):.6f}")

    print("\n" + "=" * 80)
    print("Quick Python diagnostic complete!")
    print(f"Results saved to: {output_path}")
    print("=" * 80)

    return output_path


def quick_diagnose_r(data_dir="benchmark_data", output_dir="diagnostic_output", verbose=True):
    """Quick R diagnostic - skips raster prediction."""

    import subprocess

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("=" * 80)
    print("QUICK R DIAGNOSTIC (No Raster Prediction)")
    print("=" * 80)

    # Run R script
    r_script = Path(__file__).parent / "quick_diagnose_r_workflow.R"

    if not r_script.exists():
        vprint(f"  ⚠ R script not found: {r_script}")
        vprint("  Creating R script...")
        create_quick_r_script(r_script)
        vprint("  ✓ R script created")

    vprint(f"\nRunning R diagnostic script...")
    vprint(f"  Command: Rscript {r_script} {data_dir} {output_dir}")
    vprint("  (Output will appear below)\n")

    # Run with real-time output
    result = subprocess.run(
        ["Rscript", str(r_script), data_dir, output_dir],
        text=True
    )

    if result.returncode != 0:
        print(f"\n  ❌ R script failed with exit code {result.returncode}")
        return None

    print("\n  ✓ R diagnostic complete!")

    return output_path


def create_quick_r_script(r_script_path):
    """Create the quick R diagnostic script."""
    r_script_content = '''#!/usr/bin/env Rscript
# Quick R diagnostic workflow (no raster prediction)

source("benchmarks/klrfome_r_functions.R")

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if(length(args) > 0) args[1] else "benchmark_data"
output_dir <- if(length(args) > 1) args[2] else "diagnostic_output"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("================================================================================\\n")
cat("QUICK R DIAGNOSTIC (No Raster Prediction)\\n")
cat("================================================================================\\n")

# Load and extract data (same as before)
cat("\\n[1/4] Loading data...\\n")
metadata <- fromJSON(file.path(data_dir, "metadata.json"))
rast_stack <- terra::rast(file.path(data_dir, paste0(metadata$band_names, ".tif")))
names(rast_stack) <- metadata$band_names

sites_file <- file.path(data_dir, "sites.shp")
if (!file.exists(sites_file)) sites_file <- file.path(data_dir, "sites.geojson")
sites_sf <- st_read(sites_file, quiet = TRUE)

cat(sprintf("  ✓ Loaded %d rasters, %d sites\\n", length(metadata$band_names), nrow(sites_sf)))

# Extract data
cat("\\n[2/4] Extracting data...\\n")
set.seed(42)
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
cat("\\n[3/4] Formatting data...\\n")
formatted_data <- format_site_data(
  sim_data, N_sites = 10, train_test_split = 0.8,
  sample_fraction = 0.9, background_site_balance = 1
)

train_data <- formatted_data[["train_data"]]
train_presence <- formatted_data[["train_presence"]]

train_info <- list(
  n_locations = length(train_data),
  n_sites = sum(train_presence == 1),
  n_background = sum(train_presence == 0),
  n_samples_per_location = 20,
  n_features = ncol(train_data[[1]])
)
write_json(train_info, file.path(output_dir, "r_training_info.json"), auto_unbox = TRUE, pretty = TRUE)
cat(sprintf("  ✓ Training: %d locations, %d features\\n", length(train_data), ncol(train_data[[1]])))

# Save sample data
sample_data <- list()
for(i in 1:min(3, length(train_data))){
  sample_data[[i]] <- list(
    id = i - 1, n_samples = nrow(train_data[[i]]),
    mean = as.numeric(colMeans(train_data[[i]])),
    first_sample = as.numeric(train_data[[i]][1, ])
  )
}
write_json(sample_data, file.path(output_dir, "r_sample_data.json"), auto_unbox = TRUE, pretty = TRUE)

# Build kernel matrix
cat("\\n[4/4] Building kernel matrix and fitting model...\\n")
sigma <- 0.5
K <- build_K(train_data, sigma = sigma, progress = FALSE, dist_metric = "euclidean")

# Save kernel matrix
write.csv(K, file.path(output_dir, "r_kernel_matrix.csv"), row.names = FALSE)

kernel_stats <- list(
  shape = dim(K), mean = mean(K, na.rm = TRUE), std = sd(K, na.rm = TRUE),
  min = min(K, na.rm = TRUE), max = max(K, na.rm = TRUE),
  diagonal_mean = mean(diag(K), na.rm = TRUE),
  off_diagonal_mean = mean(K[lower.tri(K) | upper.tri(K)], na.rm = TRUE),
  is_symmetric = isSymmetric(K),
  has_rounding = all(K == round(K, 3))  # Check if rounded
)
write_json(kernel_stats, file.path(output_dir, "r_kernel_stats.json"), auto_unbox = TRUE, pretty = TRUE)
cat(sprintf("  ✓ Kernel matrix: %dx%d, mean=%.6f\\n", nrow(K), ncol(K), mean(K)))
cat(sprintf("    Values rounded to 3 decimals: %s\\n", kernel_stats$has_rounding))

# Save labels
write_json(as.numeric(train_presence), file.path(output_dir, "r_labels.json"), auto_unbox = TRUE)

# Track IRLS iterations (first 5 only)
cat("  Running IRLS (tracking first 5 iterations)...\\n")
lambda <- 0.1
N <- nrow(K)
alpha <- rep(1/N, N)  # R initialization
alpha_history <- list()
alpha_history[[1]] <- as.numeric(alpha)

iteration_details <- list()
maxiter <- 5  # Only first 5 iterations

for(iter in 1:maxiter){
  Kalpha <- as.vector(K %*% alpha)
  spec <- 1 + exp(-Kalpha)
  pi <- 1 / spec
  diagW <- pi * (1 - pi)
  z <- Kalpha + ((train_presence - pi) / diagW)

  # R formulation: solve(K + lambda * diag(1/diagW), z)
  alpha_new <- try(solve(K + lambda * Matrix::Diagonal(x = 1/diagW), z), silent = TRUE)

  if(inherits(alpha_new, "try-error")){
    cat(sprintf("    Error at iteration %d\\n", iter))
    break
  }

  alphan <- as.vector(alpha_new)
  alpha_history[[iter + 1]] <- as.numeric(alphan)

  # Save iteration details
  iter_detail <- list(
    iteration = iter - 1,
    alpha_mean = mean(alphan),
    alpha_std = sd(alphan),
    prob_mean = mean(pi),
    delta_max = max(abs(alphan - alpha)),
    lhs_diag_mean = mean(diag(K + lambda * Matrix::Diagonal(x = 1/diagW))),  # For comparison
    rhs_mean = mean(z)
  )
  iteration_details[[iter]] <- iter_detail

  alpha <- alphan
}

# Save alpha history
alpha_matrix <- do.call(rbind, alpha_history)
write.csv(alpha_matrix, file.path(output_dir, "r_alpha_history.csv"), row.names = FALSE)

write_json(iteration_details, file.path(output_dir, "r_iteration_details.json"), auto_unbox = TRUE, pretty = TRUE)

cat(sprintf("  ✓ IRLS: %d iterations tracked\\n", length(alpha_history) - 1))
cat(sprintf("    Final alpha: mean=%.6f, std=%.6f\\n", mean(alpha), sd(alpha)))

# Training predictions only
log_pred <- 1 / (1 + exp(-as.vector(K %*% alpha)))
write_json(as.numeric(log_pred), file.path(output_dir, "r_training_predictions.json"), auto_unbox = TRUE)
cat(sprintf("  ✓ Training predictions: mean=%.6f\\n", mean(log_pred)))

cat("\\n================================================================================\\n")
cat("Quick R diagnostic complete!\\n")
cat(sprintf("Results saved to: %s\\n", output_dir))
cat("================================================================================\\n")
'''

    with open(r_script_path, 'w') as f:
        f.write(r_script_content)

    import os
    os.chmod(r_script_path, 0o755)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Quick diagnostic (no raster prediction)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both Python and R diagnostics
  python benchmarks/quick_diagnose.py

  # Run Python only
  python benchmarks/quick_diagnose.py --python-only

  # Run R only
  python benchmarks/quick_diagnose.py --r-only

  # Custom directories
  python benchmarks/quick_diagnose.py --data-dir my_data --output-dir my_output
        """
    )
    parser.add_argument('--data-dir', default='benchmark_data', help='Data directory')
    parser.add_argument('--output-dir', default='diagnostic_output', help='Output directory')
    parser.add_argument('--python-only', action='store_true', help='Run Python diagnostic only')
    parser.add_argument('--r-only', action='store_true', help='Run R diagnostic only')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    args = parser.parse_args()

    verbose = not args.quiet

    if not args.r_only:
        print("\n" + "="*80)
        print("STARTING PYTHON DIAGNOSTIC")
        print("="*80)
        quick_diagnose_python(args.data_dir, args.output_dir, verbose=verbose)

    if not args.python_only:
        print("\n" + "="*80)
        print("STARTING R DIAGNOSTIC")
        print("="*80)
        quick_diagnose_r(args.data_dir, args.output_dir, verbose=verbose)

    print("\n" + "="*80)
    print("ALL DIAGNOSTICS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nTo compare results, run:")
    print(f"  python benchmarks/compare_diagnostics.py")
