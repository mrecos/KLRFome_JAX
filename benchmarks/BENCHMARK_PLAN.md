# KLRfome Performance Benchmark Plan

This document outlines the plan for benchmarking the Python/JAX implementation against the original R implementation.

## Overview

The benchmark compares computational performance between:
- **Python/JAX implementation**: Modern, GPU-accelerated implementation using JAX
- **R implementation**: Original implementation using base R and optimized packages

## Files

1. **`generate_benchmark_data.py`**: Generates synthetic rasters and site points
2. **`test_python_workflow.py`**: Runs full Python/JAX workflow and measures timing
3. **`test_r_workflow.R`**: Runs full R workflow and measures timing
4. **`run_benchmark.sh`**: Orchestrates the benchmark and generates comparison report

## Workflow Steps

Both implementations follow the same workflow:

1. **Load Data**: Load rasters and site points from disk
2. **Prepare Data**: Extract samples at site locations and background
3. **Build Kernel**: Compute similarity matrix between all locations
4. **Fit Model**: Fit Kernel Logistic Regression model
5. **Predict**: Generate predictions across the entire raster

## Running the Benchmark

### Prerequisites

**Python:**
```bash
# Ensure klrfome package is installed
pip install -e .

# Verify dependencies
python -c "import klrfome; import jax; print('Python setup OK')"
```

**R:**
```bash
# Install required R packages (modern alternatives, no klrfome package needed)
Rscript -e "install.packages(c('terra', 'sf', 'rdist', 'Matrix', 'dplyr', 'jsonlite'), repos='https://cloud.r-project.org')"
```

**Note**: The R script uses standalone functions from `klrfome_r_functions.R` instead of the klrfome package. This avoids dependency issues with deprecated packages (`raster`, `rgdal`, `rgeos`, `sp`). See `INSTALL_R_DEPS.md` for details.

### Quick Start

1. **Generate benchmark data** (if not already generated):
```bash
python benchmarks/generate_benchmark_data.py --output-dir benchmark_data
```

2. **Run full benchmark**:
```bash
bash benchmarks/run_benchmark.sh
```

3. **View results**:
```bash
cat benchmark_report.txt
```

### Manual Execution

**Generate data:**
```bash
python benchmarks/generate_benchmark_data.py \
    --output-dir benchmark_data \
    --cols 200 \
    --rows 200 \
    --n-sites 25 \
    --seed 42
```

**Run Python workflow:**
```bash
python benchmarks/test_python_workflow.py \
    --data-dir benchmark_data \
    --sigma 0.5 \
    --lambda 0.1 \
    --output python_results.json
```

**Run R workflow:**
```bash
Rscript benchmarks/test_r_workflow.R \
    --data-dir benchmark_data \
    --sigma 0.5 \
    --lambda 0.1 \
    --output r_results.json
```

## Benchmark Scenarios

### Scenario 1: Small Dataset (Baseline)
- Raster size: 100x100
- Number of sites: 10
- Background locations: 25
- Samples per location: 10

```bash
python benchmarks/generate_benchmark_data.py \
    --output-dir benchmark_data_small \
    --cols 100 --rows 100 --n-sites 10
```

### Scenario 2: Medium Dataset (Default)
- Raster size: 200x200
- Number of sites: 25
- Background locations: 50
- Samples per location: 20

```bash
python benchmarks/generate_benchmark_data.py \
    --output-dir benchmark_data \
    --cols 200 --rows 200 --n-sites 25
```

### Scenario 3: Large Dataset (Stress Test)
- Raster size: 500x500
- Number of sites: 50
- Background locations: 100
- Samples per location: 30

```bash
python benchmarks/generate_benchmark_data.py \
    --output-dir benchmark_data_large \
    --cols 500 --rows 500 --n-sites 50
```

## Metrics Collected

### Timing Metrics
- **Load time**: Time to load rasters and site points
- **Prepare time**: Time to extract samples and format data
- **Build kernel time**: Time to compute similarity matrix
- **Fit time**: Time to fit KLR model
- **Predict time**: Time to generate predictions across raster
- **Total time**: End-to-end workflow time

### Model Metrics
- Convergence status
- Number of iterations
- Final loss value

### Prediction Metrics
- Prediction raster dimensions
- Value range
- Mean value

## Expected Performance Characteristics

### Python/JAX Advantages
- **GPU acceleration**: Should show significant speedup on GPU for kernel computations
- **JIT compilation**: First run slower, subsequent runs faster
- **Vectorization**: Efficient batch operations
- **Memory efficiency**: Better handling of large arrays

### R Advantages
- **Mature implementation**: Well-optimized base functions
- **Parallel processing**: Built-in parallel support for raster operations
- **Memory management**: Efficient for moderate-sized datasets

## Interpreting Results

### Speedup Calculation
```
Speedup = R_time / Python_time
```

- **Speedup > 1**: Python is faster
- **Speedup < 1**: R is faster
- **Speedup ≈ 1**: Similar performance

### Key Bottlenecks to Watch

1. **Kernel computation**: Should benefit most from GPU/JAX
2. **Raster prediction**: May benefit from batch processing
3. **Data I/O**: Should be similar between implementations
4. **Model fitting**: IRLS algorithm performance

## Troubleshooting

### Python Issues
- **JAX not using GPU**: Check with `jax.devices()`
- **Out of memory**: Reduce batch size or raster size
- **Import errors**: Ensure package is installed: `pip install -e .`

### R Issues
- **Package not found**: Install with `install.packages()` (see `INSTALL_R_DEPS.md`)
- **Memory errors**: Reduce dataset size or batch size in `KLR_predict_each()`
- **Slow performance**: Raster prediction is single-threaded in this implementation
- **Function not found**: Ensure `klrfome_r_functions.R` is in the `benchmarks/` directory

## Next Steps

1. Run baseline benchmarks on multiple systems
2. Profile code to identify bottlenecks
3. Optimize slow components
4. Document performance characteristics
5. Create performance regression tests

## Notes

- Both implementations use the same random seed (42) for reproducibility
- Hyperparameters (sigma, lambda) are matched between implementations
- Raster dimensions and site locations are identical
- Results should be comparable (allowing for numerical precision differences)
- **R implementation uses standalone functions** from `klrfome_r_functions.R` (no package installation needed)
- **Modern R packages used**: `terra` and `sf` replace deprecated `raster`/`rgdal`/`rgeos`/`sp`
