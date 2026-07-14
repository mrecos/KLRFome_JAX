# Key Differences: R vs JAX Implementation of KLRfome

This document outlines the significant changes and improvements in the Python/JAX implementation compared to the original R package.

## Executive Summary

The JAX implementation maintains the core algorithmic approach of the R version while adding JIT compilation, vectorization, optional accelerator support, and approximate feature maps. Performance and numerical robustness depend on the selected model path and must be established by reproducible benchmarks rather than assumed from the framework alone.

---

## 1. Performance & Computational Architecture

### R Implementation
- **Execution**: Single-threaded CPU execution
- **Parallelization**: Manual parallel processing using R's `parallel` package (e.g., `mclapply`, `parLapply`)
- **Memory**: Full kernel matrices stored in memory (O(n²) for n training samples)
- **Speed**: Limited by R's interpreted nature and single-threaded operations

### JAX Implementation
- **Execution**: JIT-compiled functions with automatic GPU acceleration
- **Parallelization**: Automatic vectorization via `jax.vmap` for batch operations
- **Memory**: Random Fourier Features (RFF) reduce cell-pair kernel work, but the current dual KLR path still constructs an N×N bag Gram matrix
- **Speed**: JIT/vectorization can improve throughput; actual speedups are workload- and hardware-dependent
- **JIT Compilation**: Hot paths (kernels, prediction) are compiled to optimized machine code

**Impact**: JAX provides a route to faster execution, but scale claims require measured end-to-end benchmarks, including I/O and dense solver costs.

---

## 2. Kernel Computations

### R Implementation
- **Exact Kernels Only**: Computes full pairwise kernel matrices
- **Computation**: Double loops or vectorized R operations (still CPU-bound)
- **Memory**: Full N×N similarity matrix required (can be prohibitive for large datasets)

### JAX Implementation
- **Dual Approach**:
  - Exact RBF kernel (`RBFKernel`) for small datasets
  - Random Fourier Features (`RandomFourierFeatures`) for scalable approximation
- **Computation**:
  - Exact: JIT-compiled vectorized operations
  - RFF: Linear-time construction of explicit bag features after cell-level mapping
- **Memory**: Explicit RFF bag features are O(ND), but the current dual fit subsequently constructs an O(N²) Gram matrix

**Impact**: RFF reduces representation cost. A primal logistic solver is still required to obtain O(ND) training memory.

---

## 3. Focal Window Prediction

### R Implementation
- **Approach**: Sequential processing of each window
- **Parallelization**: Manual chunking and parallel processing across cores
- **Implementation**: R loops or `apply` functions over raster cells

### JAX Implementation
- **Approach**: Batch processing with `jax.vmap` for automatic vectorization
- **Parallelization**: Automatic parallelization within JAX operations
- **Implementation**: JIT-compiled batch prediction function
- **GPU Support**: Automatically utilizes GPU if available

**Key Code Pattern**:
```python
@staticmethod
@partial(jit, static_argnames=['window_size', 'use_rff'])
def _predict_batch(...):
    def predict_single(coord):
        # Process one window
    return vmap(predict_single)(coords)  # Vectorized over batch
```

**Impact**: Batched prediction can benefit from JAX compilation and accelerators; the realized gain must be benchmarked on representative rasters.

---

## 4. Numerical Stability & Precision

### R Implementation
- **IRLS Solver**: Standard R implementation with base linear algebra
- **Convergence**: Basic tolerance checks
- **Numerical Issues**: Potential for numerical instability with ill-conditioned matrices

### JAX Implementation
- **IRLS Solver**: Retains the R-derived formulation and convergence checks
  - Ridge regularization is included
  - JAX uses float32 by default, so probability clipping must use dtype-effective tolerances
  - Extreme-logit and ill-conditioned cases require explicit stability tests
- **Precision**: JAX uses float32 by default (configurable to float64)
- **Gradient Support**: Automatic differentiation available (though not used in current IRLS)

**Impact**: The implementation is testable and accelerator-compatible, but numerical robustness is an active engineering requirement rather than an established advantage.

---

## 5. Code Architecture & Design

### R Implementation
- **Structure**: Functional R style with separate functions
- **Data**: Lists and data.frames
- **Type Safety**: Dynamic typing, runtime error checking
- **API**: Function-based API (e.g., `klr()`, `build_k()`, `format_data()`)

### JAX Implementation
- **Structure**: Object-oriented with dataclasses and protocols
- **Data**: Typed dataclasses (`SampleCollection`, `TrainingData`, `RasterStack`)
- **Type Safety**: Static type hints with `jaxtyping` for array shapes
- **API**: Both functional and OOP APIs available
  - High-level: `KLRfome` class for end-to-end workflow
  - Low-level: Individual kernel/model classes

**Example Type Safety**:
```python
@dataclass
class SampleCollection:
    samples: Float[Array, "n_samples n_features"]  # Shape-checked at type level
    label: Literal[0, 1]
    id: str
```

**Impact**: Better IDE support, earlier error detection, clearer code documentation.

---

## 6. Testing & Quality Assurance

### R Implementation
- **Testing**: R `testthat` framework
- **Coverage**: Not explicitly tracked in original package
- **Property Testing**: Not implemented

### JAX Implementation
- **Testing**: Comprehensive pytest suite with 23 tests
- **Coverage**: 44% overall, 80-90% for core modules
- **Property Testing**: Hypothesis-based tests for numerical properties
- **Test Types**:
  - Unit tests for kernels, KLR, prediction
  - Integration tests for end-to-end workflows
  - Property-based tests for kernel mathematical properties (symmetry, PSD)

**Impact**: Higher confidence in correctness, especially for numerical computations.

---

## 7. Dependencies & Ecosystem

### R Implementation
- **Core**: Base R + `stats` package
- **Geospatial**: `raster`, `sp`, `rgdal` (older stack)
- **Parallelization**: `parallel` package
- **Visualization**: Base R graphics or `ggplot2`

### JAX Implementation
- **Core**: JAX, NumPy, SciPy
- **Geospatial**: Modern stack (`rasterio`, `geopandas`, `shapely`)
- **Parallelization**: Built into JAX
- **Visualization**: `matplotlib`, `seaborn`
- **Type Checking**: `jaxtyping` for array shape annotations

**Impact**: Modern Python geospatial stack, better integration with contemporary tools.

---

## 8. API Design & Usability

### R Implementation
```r
# R workflow
data <- format_data(site_data, background_data)
K <- build_k(data, sigma = 1.0)
model <- klr(K, labels)
predictions <- klr_raster_predict(model, raster_stack, ngb = 3)
```

### JAX Implementation
```python
# Python/JAX workflow - High-level API
model = KLRfome(sigma=1.0, lambda_reg=0.1, n_rff_features=256)
model.fit(training_data)
predictions = model.predict(raster_stack, window_size=3)

# Or low-level API
kernel = MeanEmbeddingKernel(RandomFourierFeatures(sigma=1.0, n_features=256))
klr = KernelLogisticRegression(lambda_reg=0.1)
fit_result = klr.fit(K, y)
predictor = FocalPredictor(kernel, fit_result.alpha, training_data)
predictions = predictor.predict_raster(raster_stack)
```

**Impact**: More flexible API with both high-level convenience and low-level control.

---

## 9. Memory Efficiency

### R Implementation
- **Kernel Matrix**: Always full N×N matrix in memory
- **Prediction**: Stores intermediate results for each window
- **Scalability**: Limited by memory for large datasets

### JAX Implementation
- **Kernel Matrix**:
  - Exact: Full matrix (same as R)
  - RFF dual path: stores feature maps and then an N×N Gram matrix
  - RFF primal path (planned): stores O(ND) feature data without a Gram matrix
- **Prediction**:
  - Batch processing with configurable batch size
  - Memory-efficient window extraction using `lax.dynamic_slice`
- **Scalability**: Can handle much larger datasets with RFF approximation

**Impact**: Current training still scales quadratically in bag count; the planned primal RFF path removes that bottleneck for M1.

---

## 10. GPU Acceleration

### R Implementation
- **GPU Support**: Not available (CPU-only)

### JAX Implementation
- **GPU Support**: Automatic GPU utilization when available
- **Transparency**: Same code runs on CPU or GPU
- **Performance**: Accelerator speedups are possible for suitable array workloads and must be measured

**Impact**: GPU support expands deployment options; it is not itself a performance guarantee.

---

## 11. Algorithmic Improvements

### Identical Core Algorithm
- Mean embedding computation: **Same**
- IRLS algorithm: **Same lineage**, with stability work tracked explicitly
- Focal window approach: **Same**

### Enhancements
1. **RFF Approximation**: Enables scaling to larger datasets
2. **Batch Processing**: More efficient prediction pipeline
3. **JIT Compilation**: Eliminates Python overhead in hot paths
4. **Type Safety**: Catches shape errors at development time

---

## 12. Limitations & Trade-offs

### What's Missing (Compared to R)
- **Full I/O Integration**: Some rasterio/geopandas integration is still placeholder
- **Cross-Validation**: K-fold CV is placeholder (R version has full implementation)
- **Model Serialization**: Not yet implemented (R version can save/load models)
- **Visualization**: Basic plotting utilities (R version has more comprehensive plots)

### What's New (Not in R)
- **GPU Acceleration**: Automatic GPU support
- **RFF Approximation**: Scalable kernel computation
- **Type Safety**: Static type checking
- **Modern Python Stack**: Better integration with contemporary tools

---

## 13. Performance Benchmarks (Expected)

Based on JAX performance characteristics and R implementation patterns:

| Operation | R (CPU) | JAX (CPU) | JAX (GPU) |
|-----------|---------|-----------|-----------|
| Kernel Matrix (1000×1000) | ~10s | ~1s | ~0.1s |
| RFF Feature Map (1000 samples) | N/A | ~0.1s | ~0.01s |
| Focal Prediction (1000×1000 raster) | ~5-10 min | ~30s | ~3s |
| KLR Fitting (1000 samples) | ~5s | ~0.5s | ~0.05s |

*Note: Actual benchmarks should be run to verify these estimates.*

---

## 14. Migration Guide Considerations

### For R Users
1. **Data Format**: Similar concept but different data structures (dataclasses vs lists)
2. **API**: More object-oriented, but high-level `KLRfome` class provides similar workflow
3. **Hyperparameters**: Same names and meanings (`sigma`, `lambda`/`lambda_reg`)
4. **Results**: Should produce identical results (within numerical precision) for same inputs

### Key Differences to Remember
- Python 0-indexing vs R 1-indexing
- Array shapes: `(n_samples, n_features)` vs R's column-major thinking
- JAX arrays are immutable (unlike R matrices)

---

## Summary

The JAX implementation represents a **modernization** rather than a **replacement** of the core algorithm. The mathematical foundations are identical, but the execution model has been transformed to leverage:

1. **JIT Compilation**: Eliminates Python overhead
2. **GPU Acceleration**: Automatic utilization when available
3. **RFF Approximation**: Enables scaling to larger datasets
4. **Modern Python Stack**: Better integration with contemporary tools
5. **Type Safety**: Earlier error detection and better IDE support

The result is a package that maintains algorithmic fidelity while providing a modern route to measured performance and scalability improvements.

---

## References

- **Original R Package**: https://github.com/mrecos/klrfome
- **JAX Implementation**: https://github.com/mrecos/KLRFome_JAX
- **Original Paper**: Harris, M.D. (2019). KLRfome - Kernel Logistic Regression on Focal Mean Embeddings. Journal of Open Source Software, 4(35), 722.
- **Technical Specification**: See `AI_Context/klrfome_python_refactor_spec.md` in this repository
