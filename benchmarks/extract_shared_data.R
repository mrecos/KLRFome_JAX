#!/usr/bin/env Rscript
# Extract data in a format that both Python and R can use identically

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if(length(args) > 0) args[1] else "benchmark_data"
output_dir <- if(length(args) > 1) args[2] else "diagnostic_output"

library(terra)
library(sf)
library(jsonlite)

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("================================================================================\n")
cat("EXTRACTING SHARED DATA (for both Python and R)\n")
cat("================================================================================\n")

# Load data
metadata <- fromJSON(file.path(data_dir, "metadata.json"))
rast_stack <- terra::rast(file.path(data_dir, paste0(metadata$band_names, ".tif")))
names(rast_stack) <- metadata$band_names

sites_file <- file.path(data_dir, "sites.shp")
if (!file.exists(sites_file)) sites_file <- file.path(data_dir, "sites.geojson")
sites_sf <- st_read(sites_file, quiet = TRUE)

cat(sprintf("  Loaded %d rasters, %d sites\n", length(metadata$band_names), nrow(sites_sf)))

# Extract data EXACTLY as R does it
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

# Save raw data as CSV (both Python and R can read)
write.csv(sim_data, file.path(output_dir, "shared_raw_data.csv"), row.names = FALSE)

cat(sprintf("  ✓ Saved %d rows to shared_raw_data.csv\n", nrow(sim_data)))
cat(sprintf("    Sites: %d, Background: %d\n", sum(sim_data$presence == 1), sum(sim_data$presence == 0)))
cat("================================================================================\n")
