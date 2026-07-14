# Section 6 M0–M3 Comparison

**Date:** 2026-07-13

This is the first reproducible comparison using the recovered full-area GeoTIFF masks.
It is a **presence–background** analysis, so model scores represent relative suitability,
not occurrence probability. Riverine and upland were fitted separately. No settings or
physio-sheds were pooled.

## Data audit

| Setting | Retained sites | Raster backgrounds | Excluded after raster mask | Spatial CV |
|---|---:|---:|---:|---|
| Riverine | 143 | 143 | 9 sites with fewer than 3 valid unique cells | 5 folds × 2 repeats |
| Upland | 198 | 198 | 1 site with fewer than 3 valid unique cells | 5 folds × 2 repeats |

Site cells were deduplicated by `(SITENO, x, y)`, intersected with the all-band validity
mask, and capped at 120 cells. Background bags were regenerated from uniformly sampled
valid raster anchors and matched to the retained site-bag size distribution. No riverine
CSV background rows were used. Background cells never overlap site cells.

Spatial block width was twice the retained site-bag 95th-percentile diameter: 1,138.44
map units for riverine and 775.02 for upland. Both settings supported all five folds with
both classes in every train and test partition. M0–M3 used identical bags and immutable
fold assignments.

## Results

Values below are means across the ten spatial folds. Timing is the median per fold; the
first JAX compilation is retained in the raw timing records but does not describe warm
execution. All 80 fits converged and every reported fold metric was finite.

### Riverine

| Method | AUC | PR AUC | Boyce | Top-5% lift | Median fit (s) | Median predict (s) |
|---|---:|---:|---:|---:|---:|---:|
| M0 exact KME + dual | 0.6761 | 0.6700 | 0.5727 | 1.272 | 2.701 | 0.683 |
| M1 RFF + primal linear | 0.6847 | 0.6712 | 0.6600 | 1.207 | 0.293 | 0.055 |
| M2 RFF + bag RBF + dual | 0.6910 | 0.6733 | 0.6065 | 1.272 | 0.264 | 0.059 |
| M3 SW2 + RBF + dual | 0.6592 | 0.6530 | 0.5059 | 1.270 | 1.337 | 1.253 |

Paired AUC differences versus M0 (mean, approximate 95% interval): M1 `+0.0087`
`[-0.0067, 0.0240]`; M2 `+0.0149` `[-0.0020, 0.0318]`; M3 `-0.0168`
`[-0.0276, -0.0060]`. M1's paired Boyce difference was `+0.0873`
`[0.0088, 0.1658]`; the other riverine Boyce intervals overlapped zero.

### Upland

| Method | AUC | PR AUC | Boyce | Top-5% lift | Median fit (s) | Median predict (s) |
|---|---:|---:|---:|---:|---:|---:|
| M0 exact KME + dual | 0.7735 | 0.7776 | 0.7978 | 1.900 | 4.917 | 1.221 |
| M1 RFF + primal linear | 0.7663 | 0.7737 | 0.8083 | 1.900 | 0.326 | 0.076 |
| M2 RFF + bag RBF + dual | 0.7761 | 0.7801 | 0.7933 | 1.900 | 0.338 | 0.082 |
| M3 SW2 + RBF + dual | 0.7688 | 0.7739 | 0.8475 | 1.900 | 1.754 | 1.553 |

Paired AUC differences versus M0: M1 `-0.0071` `[-0.0148, 0.0005]`; M2 `+0.0026`
`[-0.0053, 0.0105]`; M3 `-0.0047` `[-0.0099, 0.0005]`. All upland paired
Boyce intervals overlapped zero, including M3's mean `+0.0497` `[-0.0298, 0.1291]`.

## Interpretation

- M1 and M2 provide a clear warm-runtime improvement over M0 while remaining close in
  discrimination. M1 satisfies the intended scaling property because it never constructs
  an N-by-N training Gram matrix.
- M2 has the highest mean AUC in both settings, but its paired intervals overlap zero.
  This is evidence to retain and test it, not evidence to select it as a winner.
- M3 is not supported over M0 by riverine AUC/PR AUC in this run, but upland Boyce is
  directionally higher. It should remain implemented: one physio-shed, fixed SW settings,
  and current bag construction cannot resolve whether shape-sensitive transport geometry
  helps with broader or more realistic data.
- Top-5% lift is coarse at fold sample sizes and tied across most upland methods. It is
  retained because it is operationally meaningful, but should be interpreted with the
  paired continuous metrics.
- Python `tracemalloc` peaks were finite and recorded, but they exclude JAX/XLA native
  allocations. Raw memory fields are therefore diagnostic lower bounds, not total process
  peaks.

No method should be promoted or removed based on this comparison. Pooling or partial
pooling across physio-sheds remains the next major statistical design decision because
many strata contain few or no sites.

The ignored raw result is written to
`site_data/r91_section_6_data/section6_comparison/results.json`; the tracked runner,
configuration, and schema are under `benchmarks/`.
