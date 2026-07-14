#!/usr/bin/env Rscript
# Quick R diagnostic workflow (no raster prediction)

source("benchmarks/klrfome_r_functions.R")

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if(length(args) > 0) args[1] else "benchmark_data"
output_dir <- if(length(args) > 1) args[2] else "diagnostic_output"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("================================================================================\n")
cat("QUICK R DIAGNOSTIC (No Raster Prediction)\n")
cat("================================================================================\n")

# Load and extract data (same as before)
cat("\n[1/4] Loading data...\n")
metadata <- fromJSON(file.path(data_dir, "metadata.json"))
rast_stack <- terra::rast(file.path(data_dir, paste0(metadata$band_names, ".tif")))
names(rast_stack) <- metadata$band_names

sites_file <- file.path(data_dir, "sites.shp")
if (!file.exists(sites_file)) sites_file <- file.path(data_dir, "sites.geojson")
sites_sf <- st_read(sites_file, quiet = TRUE)

cat(sprintf("  ✓ Loaded %d rasters, %d sites\n", length(metadata$band_names), nrow(sites_sf)))

# Extract data
cat("\n[2/4] Extracting data...\n")
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
cat("\n[3/4] Formatting data...\n")
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
cat(sprintf("  ✓ Training: %d locations, %d features\n", length(train_data), ncol(train_data[[1]])))

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
cat("\n[4/4] Building kernel matrix and fitting model...\n")
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
cat(sprintf("  ✓ Kernel matrix: %dx%d, mean=%.6f\n", nrow(K), ncol(K), mean(K)))
cat(sprintf("    Values rounded to 3 decimals: %s\n", kernel_stats$has_rounding))

# Save labels
write_json(as.numeric(train_presence), file.path(output_dir, "r_labels.json"), auto_unbox = TRUE)

# Track IRLS iterations (first 5 only)
cat("  Running IRLS (tracking first 5 iterations)...\n")
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
    cat(sprintf("    Error at iteration %d\n", iter))
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

cat(sprintf("  ✓ IRLS: %d iterations tracked\n", length(alpha_history) - 1))
cat(sprintf("    Final alpha: mean=%.6f, std=%.6f\n", mean(alpha), sd(alpha)))

# Training predictions only
log_pred <- 1 / (1 + exp(-as.vector(K %*% alpha)))
write_json(as.numeric(log_pred), file.path(output_dir, "r_training_predictions.json"), auto_unbox = TRUE)
cat(sprintf("  ✓ Training predictions: mean=%.6f\n", mean(log_pred)))

cat("\n================================================================================\n")
cat("Quick R diagnostic complete!\n")
cat(sprintf("Results saved to: %s\n", output_dir))
cat("================================================================================\n")
