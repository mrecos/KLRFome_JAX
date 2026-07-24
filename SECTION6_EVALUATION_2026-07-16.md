# Section 6 Presence–Background Evaluation Contract

**Date:** 2026-07-16

**Branch:** `codex/section6-evaluation-revamp`

## Purpose

Section 6 is a presence–background study. Background observations describe mapped availability,
not confirmed archaeological absence, and their sampled prevalence is artificial. Evaluation must
therefore lead with the ability to concentrate held-out sites into a small fraction of mapped
availability rather than with a classification threshold.

## Primary design

- Represent sites, training backgrounds, and mapped availability with identical 7 × 7 focal
  windows.
- Retain 9 × 9 and 11 × 11 common-support designs as explicit sensitivities.
- Retain the original irregular site/background bags as a sensitivity, not as the primary design.
- Fit riverine and upland separately.
- Use one immutable spatial grouped fold plan across all methods in a design.
- Fit scaling, bandwidths, and model parameters on training folds only.
- Score one fixed, uniform sample of 1,000 all-band-valid raster anchors in every fold.
- Convert held-out scores to percentiles of their fold-specific availability scores before pooling
  every bag exactly once within each repeat.

Availability percentiles make independently fitted fold scores comparable without interpreting raw
logistic output as calibrated occurrence probability.

## Metric hierarchy

### Primary

1. Held-out site capture, lift, and capture surplus at the top 5%, 10%, and 20% of mapped
   availability. Capture surplus is capture minus achieved mapped area: the vertical distance
   above random allocation. It is Youden-like but is not called Youden's J because background is
   not confirmed absence.
2. Continuous Boyce index computed on the common availability-percentile scale.
3. Held-out site availability-percentile distributions.

### Secondary

- Kvamme Gain, retained for comparison with archaeological predictive-model literature. Because
  gain equals `1 - 1/lift`, it is not independent model-selection evidence.
- ROC AUC and PR AUC on pooled out-of-fold site/background ranks.

These remain useful discrimination diagnostics, but they depend on constructed background and do
not define the mapped-area operating point.

### Diagnostic

- A negative geometry-only control using bag size and diameter.
- Maps of valid focal-window cell count and distance to the all-band raster-mask boundary.
- A full-window-only sensitivity requiring all 49 cells for sites, backgrounds, and availability.
- A sensitivity that exactly matches the background valid-cell-count distribution to sites.
- Global and Local Moran diagnostics of held-out site percentile shortfall and model disagreement
  with M0. Local permutation p-values are FDR-adjusted. Raw prediction autocorrelation is not a
  performance measure because focal windows and environmental rasters are spatially overlapping.
- Fold-safe maps showing one fold's availability percentiles, training sites, and held-out sites.
- Fit time, availability-prediction time, and peak Python memory.
- Support sensitivity across focal sizes and original irregular bags.

When a method produces tied scores, the requested area budget can expand at the boundary. Results
therefore record the achieved mapped fraction and use that achieved fraction—not the nominal
budget—as the denominator for lift, capture surplus, and gain.

## Reproducibility contract

`benchmarks/run_section6_evaluation.py` owns extraction, fold construction, fitting, prediction, and
metric calculation. `notebooks/05_section6_model_validation.ipynb` reads the resulting schema and
does not redefine folds or metrics. The result records configuration and data fingerprints, the
complete fold plan, bag-level out-of-fold ranks, fixed availability predictions, timings, and
diagnostics.

Generated Section 6 data and results remain under ignored `site_data/`. The tracked notebook can be
regenerated with `benchmarks/build_section6_evaluation_notebook.py`.

## Decision boundary

The evaluation may identify coherent candidates and design pathologies within Section 6. It cannot
promote a final method, remove an implemented representation, calibrate occurrence probability, or
resolve pooling across physio-sheds. Those decisions require additional realistic settings.

## Methodological references

- [Harris (2018), Youden's J and the Kvamme Gain metric](https://matthewdharris.com/2018/04/16/part-1-youdens-j-as-a-replacement-for-the-kvamme-gain-metric-in-the-evaluation-of-archaeological-predictive-models/)
- [Anselin (1995), Local Indicators of Spatial Association—LISA](https://doi.org/10.1111/j.1538-4632.1995.tb00338.x)
