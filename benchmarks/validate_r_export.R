#!/usr/bin/env Rscript
# R validation export script
# Generates benchmark data and exports formatted collections for Python comparison
#
# Usage:
#   Rscript benchmarks/validate_r_export.R
#
# This script:
#   1. Generates shared raw data (if not exists)
#   2. Formats data using R's format_site_data
#   3. Exports collections for Python to load
#   4. Computes kernel, fits model, makes predictions
#   5. Saves all outputs for comparison

source("benchmarks/klrfome_r_functions.R")

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if(length(args) > 0) args[1] else "benchmark_data"
output_dir <- if(length(args) > 1) args[2] else "diagnostic_output"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("================================================================================\n")
cat("R VALIDATION EXPORT\n")
cat("================================================================================\n")

# Step 1: Generate or load shared raw data
shared_csv <- file.path(output_dir, "shared_raw_data.csv")
if (!file.exists(shared_csv)) {
  cat("\n[1/5] Generating shared raw data...\n")
  
  # Load rasters and sites
  metadata <- fromJSON(file.path(data_dir, "metadata.json"))
  rast_stack <- terra::rast(file.path(data_dir, paste0(metadata$band_names, ".tif")))
  names(rast_stack) <- metadata$band_names
  
  sites_file <- file.path(data_dir, "sites.shp")
  if (!file.exists(sites_file)) sites_file <- file.path(data_dir, "sites.geojson")
  sites_sf <- st_read(sites_file, quiet = TRUE)
  
  # Extract data
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
  write.csv(sim_data, shared_csv, row.names = FALSE)
  cat(sprintf("  Saved %d rows to shared_raw_data.csv\n", nrow(sim_data)))
} else {
  cat("\n[1/5] Loading existing shared raw data...\n")
}

df <- read.csv(shared_csv)
cat(sprintf("  Loaded %d rows (%d sites, %d background)\n", 
            nrow(df), sum(df$presence == 1), sum(df$presence == 0)))

# Step 2: Format data
cat("\n[2/5] Formatting data using R's format_site_data...\n")
set.seed(42)
formatted <- format_site_data(df, N_sites = 10, train_test_split = 0.8,
                               sample_fraction = 0.9, background_site_balance = 1)

train_data <- formatted$train_data
train_presence <- formatted$train_presence

cat(sprintf("  Training: %d collections\n", length(train_data)))
cat(sprintf("  Labels: %s\n", paste(train_presence, collapse = ", ")))

# Step 3: Export collections for Python
cat("\n[3/5] Exporting collections for Python...\n")

for(i in 1:length(train_data)) {
  coll <- train_data[[i]]
  coll_name <- names(train_data)[i]
  label <- train_presence[i]
  
  coll_df <- as.data.frame(coll)
  coll_df$collection_id <- coll_name
  coll_df$collection_index <- i - 1
  coll_df$label <- label
  
  filename <- sprintf("r_collection_%02d.csv", i - 1)
  write.csv(coll_df, file.path(output_dir, filename), row.names = FALSE)
}

manifest <- data.frame(
  index = 0:(length(train_data) - 1),
  id = names(train_data),
  label = train_presence,
  n_samples = sapply(train_data, nrow)
)
write.csv(manifest, file.path(output_dir, "r_collections_manifest.csv"), row.names = FALSE)
cat(sprintf("  Saved %d collection files\n", length(train_data)))

# Step 4: Build kernel and fit model
cat("\n[4/5] Building kernel and fitting model...\n")
sigma <- 0.5
K <- build_K(train_data, sigma = sigma, progress = FALSE, dist_metric = "euclidean")

write.csv(K, file.path(output_dir, "r_rdata_kernel.csv"), row.names = FALSE)
cat(sprintf("  Kernel: %dx%d, mean=%.6f, diag_mean=%.6f\n", 
            nrow(K), ncol(K), mean(K), mean(diag(K))))

result <- KLR(K, train_presence, lambda = 0.1, tol = 0.001, maxiter = 100)
write.csv(matrix(result$alphas, nrow = 1), file.path(output_dir, "r_rdata_alpha.csv"), 
          row.names = FALSE)
cat(sprintf("  Converged: %s in %d iterations\n", result$converge, result$Ts))
cat(sprintf("  Alpha: mean=%.6f\n", mean(result$alphas)))

# Step 5: Compute predictions
cat("\n[5/5] Computing predictions...\n")
train_pred <- 1 / (1 + exp(-as.vector(K %*% result$alphas)))
write_json(as.numeric(train_pred), file.path(output_dir, "r_rdata_predictions.json"))

library(pROC)
auc_val <- as.numeric(auc(roc(train_presence, train_pred, quiet = TRUE)))
cat(sprintf("  Predictions: mean=%.4f, AUC=%.4f\n", mean(train_pred), auc_val))

cat("\n================================================================================\n")
cat("R export complete! Now run:\n")
cat("  python benchmarks/validate_against_r.py\n")
cat("================================================================================\n")

