#!/usr/bin/env Rscript
# Diagnostic using SHARED data - matches quick_diagnose_shared.py exactly

source("benchmarks/klrfome_r_functions.R")

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if(length(args) > 0) args[1] else "diagnostic_output"

cat("================================================================================\n")
cat("R DIAGNOSTIC (Using SHARED data)\n")
cat("================================================================================\n")

# Load shared data
shared_csv <- file.path(output_dir, "shared_raw_data.csv")
if (!file.exists(shared_csv)) {
  stop(paste("ERROR:", shared_csv, "not found! Run extract_shared_data.R first."))
}

cat("\n[1/4] Loading shared data...\n")
df <- read.csv(shared_csv)
cat(sprintf("  ✓ Loaded %d rows\n", nrow(df)))
cat(sprintf("    Sites: %d, Background: %d\n", sum(df$presence == 1), sum(df$presence == 0)))

# Format data
cat("\n[2/4] Formatting data...\n")
formatted <- format_site_data(df, N_sites = 10, train_test_split = 0.8,
                               sample_fraction = 0.9, background_site_balance = 1)

train_data <- formatted$train_data
train_presence <- formatted$train_presence

cat(sprintf("  Scaling parameters:\n"))
cat(sprintf("    Means: %s\n", paste(round(formatted$means, 6), collapse = ", ")))
cat(sprintf("    SDs:   %s\n", paste(round(formatted$sds, 6), collapse = ", ")))
cat(sprintf("  ✓ Train: %d locations\n", length(train_data)))

# Save training info
train_info <- list(
  n_locations = length(train_data),
  n_sites = sum(train_presence == 1),
  n_background = sum(train_presence == 0),
  n_samples_per_location = mean(sapply(train_data, nrow)),
  n_features = ncol(train_data[[1]])
)
write_json(train_info, file.path(output_dir, "r_shared_training_info.json"),
           auto_unbox = TRUE, pretty = TRUE)

# Save sample data (first 3 collections)
sample_data <- list()
for(i in 1:min(3, length(train_data))) {
  sample_data[[i]] <- list(
    id = i - 1,
    n_samples = nrow(train_data[[i]]),
    mean = as.numeric(colMeans(train_data[[i]])),
    first_sample = as.numeric(train_data[[i]][1, ])
  )
}
write_json(sample_data, file.path(output_dir, "r_shared_sample_data.json"),
           auto_unbox = TRUE, pretty = TRUE)

# Build kernel
cat("\n[3/4] Building kernel matrix...\n")
sigma <- 0.5
K <- build_K(train_data, sigma = sigma, progress = FALSE, dist_metric = "euclidean")

write.csv(K, file.path(output_dir, "r_shared_kernel.csv"), row.names = FALSE)

kernel_stats <- list(
  shape = dim(K),
  mean = mean(K),
  std = sd(K),
  min = min(K),
  max = max(K),
  diagonal_mean = mean(diag(K)),
  off_diagonal_mean = mean(K[lower.tri(K) | upper.tri(K)])
)
write_json(kernel_stats, file.path(output_dir, "r_shared_kernel_stats.json"),
           auto_unbox = TRUE, pretty = TRUE)

cat(sprintf("  ✓ Kernel: %dx%d, mean=%.6f\n", nrow(K), ncol(K), mean(K)))

# Fit model
cat("\n[4/4] Fitting KLR model...\n")
write_json(as.numeric(train_presence), file.path(output_dir, "r_shared_labels.json"))

result <- KLR(K, train_presence, lambda = 0.1, tol = 0.001, maxiter = 100)

write.csv(matrix(result$alphas, nrow = 1), file.path(output_dir, "r_shared_alpha.csv"),
          row.names = FALSE)

# Training predictions
train_pred <- 1 / (1 + exp(-as.vector(K %*% result$alphas)))
write_json(as.numeric(train_pred), file.path(output_dir, "r_shared_predictions.json"))

cat(sprintf("  ✓ Converged: %s in %d iterations\n", result$converge, result$Ts))
cat(sprintf("  ✓ Alpha: mean=%.6f, std=%.6f\n", mean(result$alphas), sd(result$alphas)))
cat(sprintf("  ✓ Predictions: mean=%.6f\n", mean(train_pred)))

cat("\n================================================================================\n")
cat("Done! Compare with Python outputs.\n")
cat("================================================================================\n")
