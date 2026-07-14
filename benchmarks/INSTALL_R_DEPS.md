# Installing R Dependencies for Benchmarking

The R benchmark script uses modern R packages to avoid dependency issues with deprecated geospatial packages.

## Required Packages

Install all required packages with:

```r
install.packages(c("terra", "sf", "rdist", "Matrix", "dplyr", "jsonlite"),
                 repos = "https://cloud.r-project.org")
```

## Package Replacements

The original R package used:
- `raster` → **`terra`** (modern, faster, actively maintained)
- `rgdal` → **`terra`** (functionality included)
- `rgeos` → **`sf`** or **`terra`** (functionality included)
- `sp` → **`sf`** or **`terra`** (functionality included)

## Why These Changes?

1. **`raster`/`rgdal`/`rgeos` are deprecated**: These packages are being phased out by the R spatial community
2. **`terra` is the modern replacement**: Faster, more memory efficient, actively maintained
3. **`sf` is the modern replacement for `sp`**: Better integration with modern R workflows

## Verification

After installation, verify packages are available:

```r
library(terra)
library(sf)
library(rdist)
library(Matrix)
library(dplyr)
library(jsonlite)
```

If all load without errors, you're ready to run the benchmark!
