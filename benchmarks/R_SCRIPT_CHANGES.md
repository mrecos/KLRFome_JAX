# R Script Changes Summary

## Overview

The R benchmark script has been completely rewritten to:
1. Use standalone functions extracted from the original R package
2. Replace deprecated packages with modern alternatives
3. Avoid dependency issues that block klrfome package installation

## Key Changes

### 1. Standalone Functions (`klrfome_r_functions.R`)

All necessary KLRfome functions have been extracted from the original R package and included as standalone functions:

- **Kernel functions**: `get_k()`, `build_K()`, `tri_swap()`
- **Model fitting**: `KLR()` - IRLS algorithm for Kernel Logistic Regression
- **Prediction**: `KLR_predict()`, `KLR_predict_each()`, `KLR_raster_predict()`
- **Data formatting**: `format_site_data()`, `scale_prediction_rasters()`

### 2. Package Replacements

| Original Package | Replacement | Reason |
|-----------------|-------------|--------|
| `raster` | `terra` | Deprecated, `terra` is faster and actively maintained |
| `rgdal` | `terra` | Deprecated, functionality included in `terra` |
| `rgeos` | `sf` / `terra` | Deprecated, functionality included in modern packages |
| `sp` | `sf` / `terra` | Being phased out, `sf` is modern standard |
| `klrfome` package | Local functions | Avoids dependency conflicts |

### 3. Updated Test Script (`test_r_workflow.R`)

The test script now:
- Sources local functions from `klrfome_r_functions.R`
- Uses `terra::rast()` instead of `raster::stack()`
- Uses `sf::st_read()` instead of `rgdal::readOGR()`
- Uses `terra::extract()` instead of `raster::extract()`
- Uses `terra::spatSample()` for background sampling

### 4. Function Updates

**`KLR_predict_each()`**:
- Rewritten to use `terra` instead of `raster::getValuesFocal()`
- Processes cells in batches for memory efficiency
- Uses terra's value extraction methods

**`scale_prediction_rasters()`**:
- Updated to work with `terra` SpatRaster objects
- Uses terra's layer indexing instead of raster stack indexing

## Dependencies

### Required (Modern Packages)
- `terra` - Raster operations
- `sf` - Vector operations
- `rdist` - Distance calculations
- `Matrix` - Sparse matrix operations
- `dplyr` - Data manipulation
- `jsonlite` - JSON I/O

### Not Required
- `klrfome` package (uses local functions)
- `raster` (replaced by `terra`)
- `rgdal` (replaced by `terra`)
- `rgeos` (replaced by `sf`/`terra`)
- `sp` (replaced by `sf`/`terra`)
- `NLMR` (not needed for benchmark, only for data generation)

## Installation

```r
install.packages(c("terra", "sf", "rdist", "Matrix", "dplyr", "jsonlite"),
                 repos = "https://cloud.r-project.org")
```

## Usage

The script works exactly the same as before:

```bash
Rscript benchmarks/test_r_workflow.R \
    --data-dir benchmark_data \
    --sigma 0.5 \
    --lambda 0.1 \
    --output r_results.json
```

## Benefits

1. **No dependency conflicts**: Avoids issues with deprecated geospatial packages
2. **Modern packages**: Uses actively maintained, faster packages
3. **Standalone**: No need to install klrfome package
4. **Compatible**: Works with same benchmark data as Python script
5. **Maintainable**: Functions are self-contained and documented

## Testing

To verify the setup works:

```r
# Load functions
source("benchmarks/klrfome_r_functions.R")

# Test basic operations
library(terra)
r <- rast(nrows = 10, ncols = 10, vals = 1:100)
print(r)  # Should work without errors
```

If this works, the benchmark script should run successfully.
