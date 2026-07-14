#!/usr/bin/env Rscript
# Run R model on the exported data (for comparison with Python)

source("benchmarks/klrfome_r_functions.R")

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if(length(args) > 0) args[1] else "diagnostic_output"

cat("================================================================================\n")
cat("R MODEL ON EXPORTED DATA\n")
cat("================================================================================\n")

# Load manifest and collections
manifest <- read.csv(file.path(output_dir, "r_collections_manifest.csv"))
cat(sprintf("Loaded manifest: %d collections\n", nrow(manifest)))

train_data <- list()
train_presence <- c()

for(i in 1:nrow(manifest)) {
  idx <- manifest$index[i]
  coll_id <- as.character(manifest$id[i])
  label <- manifest$label[i]

  # Load collection
  coll_df <- read.csv(file.path(output_dir, sprintf("r_collection_%02d.csv", idx)))
  var_cols <- setdiff(colnames(coll_df), c("collection_id", "collection_index", "label"))

  train_data[[coll_id]] <- as.matrix(coll_df[, var_cols])
  train_presence <- c(train_presence, label)
}

cat(sprintf("Labels: %s\n", paste(train_presence, collapse = ", ")))

# Show first 3 collections
cat("\nFirst 3 collections:\n")
for(i in 1:min(3, length(train_data))) {
  coll <- train_data[[i]]
  cat(sprintf("  %d: id=%s, label=%d, n_samples=%d\n",
              i-1, names(train_data)[i], train_presence[i], nrow(coll)))
  cat(sprintf("     mean=%s\n", paste(round(colMeans(coll), 4), collapse = ", ")))
}

# Build kernel
cat("\n[2/4] Building kernel matrix...\n")
sigma <- 0.5
K <- build_K(train_data, sigma = sigma, progress = FALSE, dist_metric = "euclidean")

write.csv(K, file.path(output_dir, "r_rdata_kernel.csv"), row.names = FALSE)

cat(sprintf("  ✓ Kernel: %dx%d, mean=%.6f, diag_mean=%.6f\n",
            nrow(K), ncol(K), mean(K), mean(diag(K))))

kernel_stats <- list(
  shape = dim(K),
  mean = mean(K),
  diag_mean = mean(diag(K)),
  off_diag_mean = mean(K[lower.tri(K) | upper.tri(K)])
)
write_json(kernel_stats, file.path(output_dir, "r_rdata_kernel_stats.json"),
           auto_unbox = TRUE, pretty = TRUE)

# Fit model
cat("\n[3/4] Fitting KLR model...\n")
write_json(as.numeric(train_presence), file.path(output_dir, "r_rdata_labels.json"))

result <- KLR(K, train_presence, lambda = 0.1, tol = 0.001, maxiter = 100)

write.csv(matrix(result$alphas, nrow = 1), file.path(output_dir, "r_rdata_alpha.csv"),
          row.names = FALSE)

cat(sprintf("  ✓ Converged: %s in %d iterations\n", result$converge, result$Ts))
cat(sprintf("  ✓ Alpha: mean=%.6f, std=%.6f\n", mean(result$alphas), sd(result$alphas)))

# Training predictions
cat("\n[4/4] Computing predictions...\n")
train_pred <- 1 / (1 + exp(-as.vector(K %*% result$alphas)))
write_json(as.numeric(train_pred), file.path(output_dir, "r_rdata_predictions.json"))

cat(sprintf("  ✓ Predictions: mean=%.6f, range=[%.4f, %.4f]\n",
            mean(train_pred), min(train_pred), max(train_pred)))

# Compute AUC
library(pROC)
auc_val <- as.numeric(auc(roc(train_presence, train_pred, quiet = TRUE)))
cat(sprintf("  ✓ Training AUC: %.4f\n", auc_val))

cat("\n================================================================================\n")
cat("Done!\n")
cat("================================================================================\n")
