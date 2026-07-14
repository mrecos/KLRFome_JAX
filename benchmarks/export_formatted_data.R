#!/usr/bin/env Rscript
# Export R's formatted data for Python to load directly

source("benchmarks/klrfome_r_functions.R")

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if(length(args) > 0) args[1] else "benchmark_data"
output_dir <- if(length(args) > 1) args[2] else "diagnostic_output"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("================================================================================\n")
cat("EXPORTING R FORMATTED DATA\n")
cat("================================================================================\n")

# Load shared raw data
shared_csv <- file.path(output_dir, "shared_raw_data.csv")
df <- read.csv(shared_csv)
cat(sprintf("Loaded %d rows from shared_raw_data.csv\n", nrow(df)))

# Format data using R's format_site_data
set.seed(42)
formatted <- format_site_data(df, N_sites = 10, train_test_split = 0.8,
                               sample_fraction = 0.9, background_site_balance = 1)

train_data <- formatted$train_data
train_presence <- formatted$train_presence

cat(sprintf("Formatted: %d training collections\n", length(train_data)))
cat(sprintf("Labels: %s\n", paste(train_presence, collapse = ", ")))

# Save each collection as a separate CSV
for(i in 1:length(train_data)) {
  coll <- train_data[[i]]
  coll_name <- names(train_data)[i]
  label <- train_presence[i]

  # Add metadata columns
  coll_df <- as.data.frame(coll)
  coll_df$collection_id <- coll_name
  coll_df$collection_index <- i - 1  # 0-indexed
  coll_df$label <- label

  filename <- sprintf("r_collection_%02d.csv", i - 1)
  write.csv(coll_df, file.path(output_dir, filename), row.names = FALSE)
}

# Save manifest
manifest <- data.frame(
  index = 0:(length(train_data) - 1),
  id = names(train_data),
  label = train_presence,
  n_samples = sapply(train_data, nrow)
)
write.csv(manifest, file.path(output_dir, "r_collections_manifest.csv"), row.names = FALSE)

# Save scaling params
scaling <- list(
  means = as.list(formatted$means),
  sds = as.list(formatted$sds)
)
write_json(scaling, file.path(output_dir, "r_scaling_params.json"), auto_unbox = TRUE, pretty = TRUE)

cat(sprintf("Saved %d collection files and manifest\n", length(train_data)))
cat("================================================================================\n")
