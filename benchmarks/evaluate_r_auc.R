#!/usr/bin/env Rscript
# R AUC evaluation script

source("benchmarks/klrfome_r_functions.R")

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if(length(args) > 0) args[1] else "benchmark_data"
output_dir <- if(length(args) > 1) args[2] else "diagnostic_output"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("================================================================================\n")
cat("R AUC EVALUATION\n")
cat("================================================================================\n")

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

cat(sprintf("\nTraining data: %d locations\n", length(train_data)))
cat(sprintf("Test data: %d locations\n", length(test_data)))

# Build kernel and fit
cat("\nFitting model...\n")
sigma <- 0.5
lambda <- 0.1
K <- build_K(train_data, sigma = sigma, progress = FALSE, dist_metric = "euclidean")
klr_result <- KLR(K, train_presence, lambda = lambda, maxiter = 100, tol = 0.001, verbose = 0)
cat(sprintf("  Converged: %s\n", klr_result$converged))
cat(sprintf("  Iterations: %d\n", klr_result$iterations))

# Predict on test data
cat("\nPredicting on test data...\n")
test_predictions <- KLR_predict(test_data, train_data, klr_result$alphas, sigma, progress = FALSE)

# Compute AUC using pROC package
if (!require("pROC", quietly = TRUE)) {
  # Fallback: manual AUC calculation
  cat("  Computing AUC manually (pROC not available)...\n")

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

cat(sprintf("\nResults:\n"))
cat(sprintf("  Test samples: %d\n", length(test_presence)))
cat(sprintf("  Positive samples: %d\n", sum(test_presence == 1)))
cat(sprintf("  Negative samples: %d\n", sum(test_presence == 0)))
cat(sprintf("  Prediction range: [%.6f, %.6f]\n", min(test_predictions), max(test_predictions)))
cat(sprintf("  Prediction mean: %.6f\n", mean(test_predictions)))
cat(sprintf("  AUC: %.6f\n", auc))

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
cat(sprintf("\n✓ Results saved to: %s\n", file.path(output_dir, "r_auc_results.json")))

cat("\n================================================================================\n")
