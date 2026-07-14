#!/usr/bin/env Rscript
# R workflow test script for benchmarking (using local functions and modern packages)
#
# This script loads the benchmark data and runs the full KLRfome R workflow:
# 1. Load rasters and site points (using terra and sf)
# 2. Extract data at site locations
# 3. Format site data
# 4. Build similarity kernel
# 5. Fit KLR model
# 6. Predict on rasters
# 7. Report timing and performance metrics
#
# Usage:
#   Rscript benchmarks/test_r_workflow.R [--data-dir DATA_DIR] [--sigma SIGMA] [--lambda LAMBDA]

# Load local functions (no package installation needed)
source("benchmarks/klrfome_r_functions.R")

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Default values
data_dir <- "benchmark_data"
sigma <- 0.5
lambda <- 0.1
dist_metric <- "euclidean"
output_file <- NULL

# Simple argument parsing
i <- 1
while (i <= length(args)) {
  if (args[i] == "--data-dir" && i < length(args)) {
    data_dir <- args[i + 1]
    i <- i + 2
  } else if (args[i] == "--sigma" && i < length(args)) {
    sigma <- as.numeric(args[i + 1])
    i <- i + 2
  } else if (args[i] == "--lambda" && i < length(args)) {
    lambda <- as.numeric(args[i + 1])
    i <- i + 2
  } else if (args[i] == "--output" && i < length(args)) {
    output_file <- args[i + 1]
    i <- i + 2
  } else {
    i <- i + 1
  }
}

# Set seed for reproducibility
set.seed(42)

cat("================================================================================\n")
cat("R KLRfome Workflow Benchmark (Standalone Functions)\n")
cat("================================================================================\n")

# Load metadata
cat("\n[1/6] Loading benchmark data...\n")
start_time <- Sys.time()

metadata_file <- file.path(data_dir, "metadata.json")
if (!file.exists(metadata_file)) {
  stop(paste("Metadata file not found:", metadata_file))
}

metadata <- fromJSON(metadata_file)

# Load rasters using terra (modern replacement for raster)
raster_files <- file.path(data_dir, paste0(metadata$band_names, ".tif"))
rast_stack <- terra::rast(raster_files)
names(rast_stack) <- metadata$band_names

# Load sites using sf (modern replacement for sp/rgdal)
sites_file <- file.path(data_dir, "sites.shp")
if (!file.exists(sites_file)) {
  sites_file <- file.path(data_dir, "sites.geojson")
}

sites_sf <- st_read(sites_file, quiet = TRUE)
load_time <- as.numeric(Sys.time() - start_time, units = "secs")

cat(sprintf("  ✓ Loaded %d rasters (%dx%d)\n",
            length(metadata$band_names),
            ncol(rast_stack),
            nrow(rast_stack)))
cat(sprintf("  ✓ Loaded %d site locations\n", nrow(sites_sf)))
cat(sprintf("  Time: %.3fs\n", load_time))

# Extract data at sites
cat("\n[2/6] Extracting data at site locations...\n")
start_time <- Sys.time()

# Extract values at site points using terra
site_values <- terra::extract(rast_stack, sites_sf)

# Create data frame similar to get_sim_data output
# For each site, create multiple samples (simulating within-site variation)
n_samples_per_site <- 20
site_data_list <- list()

for (i in 1:nrow(sites_sf)) {
  site_vals <- as.numeric(site_values[i, -1])  # Remove ID column
  # Add some noise to simulate within-site variation
  for (j in 1:n_samples_per_site) {
    noise <- rnorm(length(metadata$band_names), mean = 0, sd = 0.1)
    sample_vals <- site_vals + noise * abs(site_vals)
  # Create data frame with available variables
  df_row <- data.frame(
    presence = 1,
    SITENO = paste0("Site", i),
    stringsAsFactors = FALSE
  )
  for(v in 1:length(metadata$band_names)){
    df_row[[metadata$band_names[v]]] <- sample_vals[v]
  }
  site_data_list[[length(site_data_list) + 1]] <- df_row
  }
}

# Sample background locations
n_background <- 50
n_samples_per_bg <- 20
bg_coords <- terra::spatSample(rast_stack, n_background * n_samples_per_bg, as.points = TRUE)
bg_values <- terra::extract(rast_stack, bg_coords)

for (i in 1:nrow(bg_values)) {
  bg_vals <- as.numeric(bg_values[i, -1])  # Remove ID column
  # Create data frame with available variables
  df_row <- data.frame(
    presence = 0,
    SITENO = "background",
    stringsAsFactors = FALSE
  )
  for(v in 1:length(metadata$band_names)){
    df_row[[metadata$band_names[v]]] <- bg_vals[v]
  }
  site_data_list[[length(site_data_list) + 1]] <- df_row
}

# Remove rows with NA (if var3 doesn't exist)
sim_data <- do.call(rbind, site_data_list)
sim_data <- sim_data[complete.cases(sim_data), ]

extract_time <- as.numeric(Sys.time() - start_time, units = "secs")

cat(sprintf("  ✓ Extracted %d samples\n", nrow(sim_data)))
cat(sprintf("  Time: %.3fs\n", extract_time))

# Format data
cat("\n[3/6] Formatting site data...\n")
start_time <- Sys.time()

formatted_data <- format_site_data(
  sim_data,
  N_sites = 10,
  train_test_split = 0.8,
  sample_fraction = 0.9,
  background_site_balance = 1
)

train_data <- formatted_data[["train_data"]]
train_presence <- formatted_data[["train_presence"]]
test_data <- formatted_data[["test_data"]]
test_presence <- formatted_data[["test_presence"]]

format_time <- as.numeric(Sys.time() - start_time, units = "secs")

cat(sprintf("  ✓ Formatted data: %d train, %d test\n",
            length(train_data),
            length(test_data)))
cat(sprintf("  Time: %.3fs\n", format_time))

# Build similarity kernel
cat("\n[4/6] Building similarity kernel...\n")
start_time <- Sys.time()

K <- build_K(train_data, sigma = sigma, progress = FALSE, dist_metric = dist_metric)
build_k_time <- as.numeric(Sys.time() - start_time, units = "secs")

cat(sprintf("  ✓ Built similarity matrix: %dx%d\n", nrow(K), ncol(K)))
cat(sprintf("  Time: %.3fs\n", build_k_time))

# Fit KLR model
cat("\n[5/6] Fitting KLR model...\n")
start_time <- Sys.time()

klr_result <- KLR(K, train_presence, lambda = lambda, maxiter = 100, tol = 0.001, verbose = 1)
fit_time <- as.numeric(Sys.time() - start_time, units = "secs")

cat(sprintf("  ✓ Model fitted\n"))
if (!is.null(klr_result$converged) && klr_result$converged) {
  cat(sprintf("    Converged in %d iterations\n", klr_result$iterations))
} else {
  cat(sprintf("    ⚠ Did not converge after %d iterations\n", klr_result$iterations))
}
cat(sprintf("  Time: %.3fs\n", fit_time))

# Predict on raster
cat("\n[6/6] Predicting on raster...\n")
start_time <- Sys.time()

# Scale prediction rasters
params <- list(
  train_data = train_data,
  alphas_pred = klr_result[["alphas"]],
  sigma = sigma,
  lambda = lambda,
  means = formatted_data$means,
  sds = formatted_data$sds
)

pred_var_stack_scaled <- scale_prediction_rasters(rast_stack, params, verbose = 0)

# Predict (using smaller window for speed in benchmark)
ngb <- 5
pred_rast <- KLR_raster_predict(pred_var_stack_scaled, ngb = ngb, params = params, progress = FALSE)

predict_time <- as.numeric(Sys.time() - start_time, units = "secs")

pred_vals <- values(pred_rast)
pred_vals <- pred_vals[!is.na(pred_vals)]

cat(sprintf("  ✓ Predictions generated: %dx%d\n",
            nrow(pred_rast),
            ncol(pred_rast)))
cat(sprintf("    Range: [%.3f, %.3f]\n",
            min(pred_vals, na.rm = TRUE),
            max(pred_vals, na.rm = TRUE)))
cat(sprintf("    Mean: %.3f\n", mean(pred_vals, na.rm = TRUE)))
cat(sprintf("  Time: %.3fs\n", predict_time))

# Summary
total_time <- load_time + extract_time + format_time + build_k_time + fit_time + predict_time

cat("\n")
cat("================================================================================\n")
cat("TIMING SUMMARY\n")
cat("================================================================================\n")
cat(sprintf("  Load data:          %8.3fs (%5.1f%%)\n", load_time, load_time/total_time*100))
cat(sprintf("  Extract data:       %8.3fs (%5.1f%%)\n", extract_time, extract_time/total_time*100))
cat(sprintf("  Format data:        %8.3fs (%5.1f%%)\n", format_time, format_time/total_time*100))
cat(sprintf("  Build kernel:       %8.3fs (%5.1f%%)\n", build_k_time, build_k_time/total_time*100))
cat(sprintf("  Fit model:          %8.3fs (%5.1f%%)\n", fit_time, fit_time/total_time*100))
cat(sprintf("  Predict:            %8.3fs (%5.1f%%)\n", predict_time, predict_time/total_time*100))
cat(sprintf("  %s\n", paste(rep("-", 60), collapse = "")))
cat(sprintf("  TOTAL:              %8.3fs\n", total_time))
cat("================================================================================\n")

# Save results if output file specified
if (!is.null(output_file)) {
  results <- list(
    load_time = as.numeric(load_time),
    extract_time = as.numeric(extract_time),
    format_time = as.numeric(format_time),
    build_k_time = as.numeric(build_k_time),
    fit_time = as.numeric(fit_time),
    predict_time = as.numeric(predict_time),
    total_time = as.numeric(total_time),
    predictions_shape = as.numeric(c(nrow(pred_rast), ncol(pred_rast))),
    predictions_range = as.numeric(c(min(pred_vals, na.rm = TRUE), max(pred_vals, na.rm = TRUE))),
    predictions_mean = as.numeric(mean(pred_vals, na.rm = TRUE)),
    converged = as.logical(klr_result$converged),
    iterations = as.integer(klr_result$iterations)
  )

  # Use auto_unbox to prevent single values from being wrapped in arrays
  write_json(results, output_file, pretty = TRUE, auto_unbox = TRUE)
  cat(sprintf("\n✓ Results saved to: %s\n", output_file))
}
