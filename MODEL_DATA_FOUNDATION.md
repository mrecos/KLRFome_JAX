# Model and Data Foundation

This document records the implemented contract for the `codex/model-data-foundation`
branch. The statistical rationale and future method roadmap remain in
[`METHODS_ROADMAP_2026-07-13.md`](METHODS_ROADMAP_2026-07-13.md).

## Model architectures

`ModelSpec` makes three formerly coupled choices explicit: the distribution
representation, the decision kernel applied to that representation, and the solver.

| ID | Bag representation | Bag-level decision rule | Solver |
|---|---|---|---|
| M0 | Exact RBF kernel mean embedding | Linear RKHS inner product | Dual KLR |
| M1 | RFF approximation to the RBF mean embedding | Linear | Primal regularized logistic regression |
| M2 | RFF approximation to the RBF mean embedding | RBF on bag embeddings | Dual KLR |
| M3 | Fixed-quantile sliced Wasserstein-2 | RBF on SW2 distance | Dual KLR |

M1 never constructs an N-by-N training Gram matrix. M0 preserves the original
distribution-regression model. M2 separates approximation error from additional
nonlinear decision capacity. M3 represents projected distributional shape rather than
only RKHS mean similarity. The high-level API rejects Wasserstein-1 because an RBF of
the current SW1 distance is retained only as a research utility, not a supported model.

The legacy `KLRfome` arguments map as follows:

- `kernel_type="mean_embedding", n_rff_features=0` maps to M0.
- `kernel_type="mean_embedding", n_rff_features>0` maps to M1.
- adding `embedding_kernel="rbf"` maps an RFF model to M2.
- `kernel_type="wasserstein", wasserstein_p=2` maps to M3.
- an explicit `spec=ModelSpec.m0() ... m3()` is preferred for new work.

All feature scaling, automatic point bandwidth estimation, and bag-level bandwidth
estimation occur inside `fit` and therefore use training folds only. The dual and primal
solvers use stable sigmoid/cross-entropy calculations, float32-effective probability
floors, bounded diagonal jitter, and explicit failure diagnostics.

## Canonical data contract

A `Bag` contains a finite two-dimensional cell-by-feature sample array, binary label,
unique ID, optional cell coordinates, group/site ID, stratum ID, and metadata. A
`BagDataset` adds ordered feature names, CRS, and a declared `presence_background` or
`presence_absence` study design. `SampleCollection` and `TrainingData` are compatibility
aliases to these canonical types.

`TabularBagConfig` supports explicit columns and conservative autodetection. It removes
duplicate `(site ID, x, y)` cells by default, records invalid/duplicate counts, and drops
bags below the configured unique-cell minimum. `RasterSource` validates alignment and
reads windows through Rasterio. Its validity mask is the intersection of all bands.
Polygon extraction retains every covered valid cell; a point requires an explicit buffer
or pixel window.

`align_bags_to_raster` applies the raster mask to tabular site coordinates and uses the
raster values as the canonical covariates. `build_spatial_background_bags` samples valid
raster anchors uniformly, matches the retained site bag-size distribution, and excludes
all site cells. These two paths produce the same `BagDataset` contract.

## Validation contract

`FoldPlan` is immutable and records bag IDs plus every train/test assignment. A bag is
tested exactly once per repeat; group or spatial-block members cannot cross a fold.
Models are cloned for every fold, so preprocessing, bandwidths, and tuning are fitted on
training data only. One-class AUC is reported as undefined (`NaN`), never replaced with
0.5.

## Section 6 comparison

The tracked runner and configuration are:

- `benchmarks/run_section6_comparison.py`
- `benchmarks/section6_comparison_config.json`
- `benchmarks/section6_result_schema.json`

Riverine and upland are fitted separately as presence-background designs. Site CSV cells
are deduplicated, checked against the all-band raster mask, and capped at 120 cells after
dropping bags with fewer than three valid unique cells. Background bags are regenerated
from the GeoTIFF mask; the truncated riverine CSV background is not used. Spatial block
width is twice the 95th-percentile retained site-bag diameter. M0--M3 receive the exact
same bags and folds.

Scores in this design are **relative suitability**, not occurrence probability. A single
physio-shed cannot justify promoting or removing any method. Pooling or partial pooling
across physio-sheds remains explicitly deferred because many strata have few or no sites.
