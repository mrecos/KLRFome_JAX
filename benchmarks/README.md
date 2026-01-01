# KLRfome Benchmarks & Validation

This directory contains scripts for validating and benchmarking the Python/JAX implementation against the R reference implementation.

## Quick Validation (Recommended)

The primary validation workflow uses shared data to ensure Python matches R exactly:

```bash
# Step 1: Generate benchmark data
python benchmarks/generate_benchmark_data.py

# Step 2: Export R's formatted data and model outputs
Rscript benchmarks/validate_r_export.R

# Step 3: Run Python validation against R outputs
python benchmarks/validate_against_r.py
```

Expected output:
```
VALIDATION SUMMARY
================================================================================
  Kernel Matrix:  ✓ MATCH
  Alpha Values:   ✓ MATCH
  Predictions:    ✓ MATCH

  Python AUC: 0.XXXX
  R AUC:      0.XXXX

  ✓ ALL COMPONENTS MATCH - Python implementation is validated!
```

## Files Overview

### Primary Validation Scripts (Use These)
| File | Description |
|------|-------------|
| `validate_against_r.py` | **Main Python validation** - compares kernel, alpha, predictions to R |
| `validate_r_export.R` | **R export script** - formats data and saves outputs for Python |
| `generate_benchmark_data.py` | Creates synthetic rasters and sites |

### R Reference Implementation
| File | Description |
|------|-------------|
| `klrfome_r_functions.R` | Core R functions: `KLR`, `build_K`, `format_site_data`, etc. |

### Performance Benchmarks
| File | Description |
|------|-------------|
| `run_benchmark.sh` | Orchestrates full benchmark with timing |
| `test_python_workflow.py` | Times the Python workflow |
| `test_r_workflow.R` | Times the R workflow |
| `benchmark_kernels.py` | Kernel computation benchmarks |
| `benchmark_prediction.py` | Focal prediction benchmarks |

### Legacy Diagnostic Scripts
These were used during development and are retained for reference:
- `quick_diagnose.py`, `quick_diagnose_shared.py` - Python diagnostics
- `quick_diagnose_shared.R`, `diagnose_r_workflow.R` - R diagnostics
- `compare_diagnostics.py`, `compare_shared.py`, `compare_exact.py` - Comparison tools
- `evaluate_auc.py`, `evaluate_r_auc.R` - AUC evaluation
- `export_formatted_data.R`, `diagnose_with_r_data.py` - Data export pipeline

## What the Validation Checks

1. **Kernel Matrix**: Mean embedding kernel with RBF base kernel (σ=0.5), rounded to 3 decimals
2. **Alpha Coefficients**: IRLS optimization with λ=0.1, tolerance=0.001
3. **Predictions**: Probability outputs from sigmoid(K·α)
4. **AUC**: Area under ROC curve for training data

## Key Alignment Findings

During development, we identified these critical alignment points:

### Algorithm Alignment
- **IRLS formulation**: `(K + λ·diag(1/W))·α = z` matches R exactly
- **Initial alpha**: `ones(n)/n` (not zeros)
- **Convergence**: `all(|α_new - α| ≤ tol)` with `tol=0.001`
- **Sigmoid**: Simple `1/(1+exp(-x))` without clipping

### Data Handling
- **Kernel rounding**: Round to 3 decimals to match R's default behavior
- **Coordinate systems**: Use `rasterio.transform.rowcol` correctly (returns row, col)
- **Scaling**: Z-score normalization applied consistently

### Parameters
- **sigma**: 0.5 (RBF kernel width)
- **lambda_reg**: 0.1 (regularization)
- **n_rff_features**: 0 for exact kernel matching, 256 for fast approximation

## Output Files

After running validation:

```
diagnostic_output/
├── r_collections_manifest.csv    # Collection metadata
├── r_collection_00.csv ... NN    # Individual collections
├── r_rdata_kernel.csv            # R's kernel matrix
├── r_rdata_alpha.csv             # R's alpha coefficients
├── r_rdata_predictions.json      # R's predictions
└── shared_raw_data.csv           # Shared raw data
```

## Requirements

### Python
```bash
pip install -e .  # Install klrfome package
```

### R
```r
install.packages(c("terra", "sf", "rdist", "Matrix", "dplyr", "jsonlite", "pROC"))
```

See [INSTALL_R_DEPS.md](INSTALL_R_DEPS.md) for detailed R setup.
