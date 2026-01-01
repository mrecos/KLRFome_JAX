# KLRfome R Functions - Standalone Implementation
# Extracted from original R package and updated to use modern packages
#
# This file contains all necessary functions for benchmarking, using:
# - terra instead of raster/rgdal/rgeos
# - sf instead of sp/rgeos
# - Modern alternatives for deprecated packages

# ============================================================================
# DEPENDENCIES
# ============================================================================
# Required packages (install with: install.packages(c("terra", "sf", "rdist", "Matrix", "dplyr", "jsonlite")))
# - terra: Modern replacement for raster/rgdal/rgeos
# - sf: Modern replacement for sp/rgeos
# - rdist: Distance calculations
# - Matrix: Sparse matrix operations
# - dplyr: Data manipulation
# - jsonlite: JSON I/O

suppressPackageStartupMessages({
  if (!require("terra", quietly = TRUE)) stop("Please install 'terra': install.packages('terra')")
  if (!require("sf", quietly = TRUE)) stop("Please install 'sf': install.packages('sf')")
  if (!require("rdist", quietly = TRUE)) stop("Please install 'rdist': install.packages('rdist')")
  if (!require("Matrix", quietly = TRUE)) stop("Please install 'Matrix': install.packages('Matrix')")
  if (!require("dplyr", quietly = TRUE)) stop("Please install 'dplyr': install.packages('dplyr')")
  if (!require("jsonlite", quietly = TRUE)) stop("Please install 'jsonlite': install.packages('jsonlite')")
})

# ============================================================================
# KERNEL FUNCTIONS
# ============================================================================

#' get_k - Calculate RBF kernel between two data matrices
get_k <- function(y1, y2, sigma, dist_metric = "euclidean"){
  g <- rdist::cdist(as.matrix(y1), as.matrix(y2), metric = dist_metric)
  g <- exp(-g^2 / (2 * sigma^2))
  return(g)
}

#' tri_swap - Mirror upper triangle to lower triangle
tri_swap <- function(m) {
  m[lower.tri(m)] <- t(m)[lower.tri(m)]
  m
}

#' build_K - Build similarity kernel matrix
build_K <- function(y1, y2 = y1, sigma, progress = TRUE, dist_metric = "euclidean"){
  K <- matrix(nrow = length(y1), ncol = length(y2))
  if(isTRUE(progress)){
    total_iter <- sum(seq(length(y1)-1))
    pb <- txtProgressBar(min = 0, max = total_iter, style = 3)
  }
  iter <- 0
  for(i in 1:length(y1)){
    for(j in i:length(y2)){ 
      g <- get_k(y1[[i]], y2[[j]], sigma, dist_metric = dist_metric)
      k <- round(mean(g, na.rm = TRUE), 3)
      K[i,j] <- k
      if(isTRUE(progress)){setTxtProgressBar(pb, iter)}
      iter <- iter + 1
    }
  }
  if(isTRUE(progress)){close(pb)}
  K <- tri_swap(K)
  return(K)
}

# ============================================================================
# KLR MODEL FITTING
# ============================================================================

#' KLR - Fit Kernel Logistic Regression via IRLS
KLR <- function(K, presence, lambda, maxiter = 100, tol = 0.01, verbose = 1){
  if(is.vector(K)){
    N <- length(K)
  } else if(is.matrix(K)){
    N <- nrow(K)
  }
  alpha <- rep(1/N, N) # initial value
  iter <- 1
  while(TRUE) {
    Kalpha <- as.vector(K %*% alpha)
    spec <- 1 + exp(-Kalpha)
    pi <- 1 / spec
    diagW <- pi * (1 - pi)
    z <- Kalpha + ((presence - pi) / diagW)
    alpha_new <- try(solve(K + lambda * Matrix::Diagonal(x = 1/diagW), z), silent = TRUE)
    if (inherits(alpha_new, "try-error")) {
      cat("Error in calculating solution.\n")
      break
    }
    alphan <- as.vector(alpha_new)
    if(verbose == 2){
      cat("Step ", iter, ". Absolute Relative Approximate Error = ",
          round(abs(sum(alphan - alpha)/sum(alphan))*100, 4), "\n", sep = "")
    }
    if (any(is.nan(alphan)) || all(abs(alphan - alpha) <= tol)) {
      if(verbose %in% c(1,2)){
        cat("Found solution in", iter, "steps.\n")
      }
      break
    }
    else if (iter > maxiter) {
      cat("Maximum iterations for KRR Logit!\n",
          "May be non-optimum solution.\n")
      break
    }
    else {
      alpha <- alphan
      iter <- iter + 1
    }
  }
  log_pred <- 1 / (1 + exp(-as.vector(K %*% alpha_new)))
  return(list(pred = log_pred, alphas = alpha_new, iterations = iter, converged = iter <= maxiter))
}

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

#' KLR_predict - Predict on new data
KLR_predict <- function(test_data, train_data, alphas_pred, sigma, progress = TRUE, dist_metric = "euclidean"){
  kstark <- matrix(nrow = length(test_data), ncol = length(train_data))
  if(isTRUE(progress)){
    total_iter <- length(test_data) * length(train_data)
    pb <- txtProgressBar(min = 0, max = total_iter, style = 3)
  }
  iter <- 0
  for(j in 1:length(test_data)){
    for(i in 1:length(train_data)){
      g_i <- get_k(train_data[[i]], test_data[[j]], sigma, dist_metric = dist_metric)
      k_i <- round(mean(g_i, na.rm = TRUE), 3)
      kstark[j,i] <- k_i
      if(isTRUE(progress)){setTxtProgressBar(pb, iter)}
      iter <- iter + 1
    }
  }
  if(isTRUE(progress)){close(pb)}
  pred <- 1 / (1 + exp(-as.vector(kstark %*% alphas_pred)))
  return(pred)
}

#' KLR_predict_each - Predict on raster using focal windows (terra version)
#' Simplified version that processes cells in batches
KLR_predict_each <- function(rast_stack, ngb, params, progress = FALSE){
  nrows <- nrow(rast_stack)
  ncols <- ncol(rast_stack)
  nbands <- nlyr(rast_stack)
  
  # Create output raster
  pred_rast <- rast(rast_stack, nlyr = 1)
  pred_vals <- rep(NA_real_, ncell(pred_rast))
  
  pad <- floor(ngb / 2)
  
  # Get all values as matrices for faster access
  all_vals <- values(rast_stack, mat = TRUE)  # ncell x nbands matrix
  
  # Process in batches
  batch_size <- 1000
  total_cells <- nrows * ncols
  n_batches <- ceiling(total_cells / batch_size)
  
  for(batch in 1:n_batches){
    batch_start <- (batch - 1) * batch_size + 1
    batch_end <- min(batch * batch_size, total_cells)
    batch_cells <- batch_start:batch_end
    
    test_data_batch <- list()
    valid_indices_batch <- c()
    
    for(cell_idx in batch_cells){
      # Convert cell index to row/col
      row <- ((cell_idx - 1) %/% ncols) + 1
      col <- ((cell_idx - 1) %% ncols) + 1
      
      # Extract window bounds
      row_start <- max(1, row - pad)
      row_end <- min(nrows, row + pad)
      col_start <- max(1, col - pad)
      col_end <- min(ncols, col + pad)
      
      # Extract window values - convert row/col to cell indices
      window_cells <- c()
      for(r in row_start:row_end){
        for(c in col_start:col_end){
          cell <- (r - 1) * ncols + c
          window_cells <- c(window_cells, cell)
        }
      }
      
      # Get values for window cells
      window_vals <- all_vals[window_cells, , drop = FALSE]
      window_vals <- window_vals[complete.cases(window_vals), , drop = FALSE]
      
      if(nrow(window_vals) > 0){
        test_data_batch[[length(test_data_batch) + 1]] <- window_vals
        valid_indices_batch <- c(valid_indices_batch, cell_idx)
      }
    }
    
    # Predict on batch
    if(length(test_data_batch) > 0){
      tpred_batch <- KLR_predict(test_data_batch, params$train_data, params$alphas_pred, 
                                 params$sigma, progress = FALSE)
      pred_vals[valid_indices_batch] <- tpred_batch
    }
    
    if(progress && batch %% 10 == 0){
      cat(sprintf("Processed batch %d of %d (%.1f%%)\n", batch, n_batches, batch/n_batches*100))
    }
  }
  
  values(pred_rast) <- pred_vals
  return(pred_rast)
}

#' KLR_raster_predict - Main raster prediction function (simplified, no parallel)
KLR_raster_predict <- function(rast_stack, ngb, params, progress = TRUE){
  pred_rast <- KLR_predict_each(rast_stack, ngb, params, progress = progress)
  return(pred_rast)
}

# ============================================================================
# DATA FORMATTING FUNCTIONS
# ============================================================================

#' format_site_data - Format data for KLR model
format_site_data <- function(dat, N_sites, train_test_split, background_site_balance, sample_fraction){
  if (!("SITENO" %in% names(dat) & "presence" %in% names(dat))) {
    stop("Data must contain columns named 'presence' and 'SITENO'.")
  }
  if (length(setdiff(colnames(dat), c("presence", "SITENO"))) == 0) {
    stop("Data must contain variable columns in addition to 'presence' and 'SITENO'.")
  }
  variables <- setdiff(colnames(dat), c("presence", "SITENO"))
  if(length(variables) == 0){
    stop("No variable columns found in data.")
  }
  means <- sapply(dat[variables], mean, na.rm = TRUE)
  sds <- sapply(dat[variables], sd, na.rm = TRUE)
  # Scale variables
  dat_scaled <- as.data.frame(scale(dat[, variables, drop = FALSE]))
  colnames(dat_scaled) <- variables
  dat <- cbind(dat_scaled, dat[, c("presence", "SITENO")])
  N_back_bags <- N_sites * background_site_balance
  
  ## Reduce number of sites to N_sites
  sites <- dplyr::filter(dat, presence == 1)
  site_names <- as.character(unique(sites$SITENO))
  N_sites_index <- sample(site_names, N_sites)
  sites <- filter(sites, SITENO %in% N_sites_index)
  
  ### Split Sites Data
  sites_train_index <- sample(N_sites_index, length(N_sites_index) * train_test_split)
  train_sites <- filter(sites, SITENO %in% sites_train_index)
  test_sites <- filter(sites, !SITENO %in% sites_train_index)
  
  ### Split Background Data
  train_background <- filter(dat, presence == 0) %>%
    sample_n(nrow(train_sites) * background_site_balance, replace = TRUE) %>%
    mutate(presence = 0)
  test_background <- filter(dat, presence == 0) %>%
    sample_n(nrow(test_sites) * background_site_balance, replace = TRUE) %>%
    mutate(presence = 0)
  
  ### Tabular data - REDUCE BY [sample_fraction]
  tbl_train_data <- rbind(train_sites, train_background) %>%
    sample_frac(size = sample_fraction)
  tbl_train_presence <- dplyr::select(tbl_train_data, presence)
  tbl_train_presence <- as.numeric(tbl_train_presence$presence)
  tbl_test_data <- rbind(test_sites, test_background) %>%
    sample_frac(size = sample_fraction)
  tbl_test_presence <- dplyr::select(tbl_test_data, presence)
  tbl_test_presence <- as.numeric(tbl_test_presence$presence)
  
  ### Split out background
  train_back_list <- dplyr::filter(tbl_train_data, SITENO == "background") %>%
    dplyr::select(-presence, -SITENO) %>%
    split(sample(N_back_bags, nrow(.), replace = TRUE))
  names(train_back_list) <- sample(paste0("background", 1:N_back_bags), length(train_back_list))
  train_site_list <- dplyr::filter(tbl_train_data, SITENO != "background") %>%
    split(f = .$SITENO) %>%
    lapply(., function(x) x[!(names(x) %in% c("presence", "SITENO"))])
  test_site_list <- group_by(tbl_test_data, SITENO) %>%
    mutate(id = paste0(SITENO, "_", seq_len(n()))) %>%
    split(f = .$id) %>%
    lapply(., function(x) x[!(names(x) %in% c("presence", "SITENO", "id"))])
  
  # Merge site and background bags together
  train_data <- c(train_site_list, train_back_list)
  test_data <- test_site_list
  
  # Shuffle list
  train_data <- sample(train_data, length(train_data))
  test_data <- sample(test_data, length(test_data))
  train_presence <- ifelse(grepl("background", names(train_data)), 0, 1)
  test_presence <- ifelse(grepl("background", names(test_data)), 0, 1)
  
  return(list(train_data = train_data,
              test_data = test_data,
              train_presence = train_presence,
              test_presence = test_presence,
              tbl_train_data = tbl_train_data,
              tbl_train_presence = tbl_train_presence,
              tbl_test_data = tbl_test_data,
              tbl_test_presence = tbl_test_presence,
              means = means,
              sds = sds))
}

#' scale_prediction_rasters - Scale rasters to training data parameters
scale_prediction_rasters <- function(pred_var_stack, params, verbose = 1){
  pred_var_stack_scaled <- pred_var_stack
  for(i in 1:nlyr(pred_var_stack)){
    var_name <- names(pred_var_stack)[i]
    if(verbose == 1){
      cat("Normalizing:", var_name, "\n")
    }
    # Scale using terra
    layer <- pred_var_stack[[i]]
    layer_scaled <- (layer - params$means[var_name]) / params$sds[var_name]
    pred_var_stack_scaled[[i]] <- layer_scaled
  }
  return(pred_var_stack_scaled)
}

