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

1. Held-out site capture, lift, and Kvamme gain at the top 5%, 10%, and 20% of mapped
   availability.
2. Continuous Boyce index computed on the common availability-percentile scale.
3. Held-out site availability-percentile distributions.

### Secondary

- ROC AUC and PR AUC on pooled out-of-fold site/background ranks.

These remain useful discrimination diagnostics, but they depend on constructed background and do
not define the mapped-area operating point.

### Diagnostic

- A negative geometry-only control using bag size and diameter.
- Fold-safe maps showing one fold's availability percentiles, training sites, and held-out sites.
- Fit time, availability-prediction time, and peak Python memory.
- Support sensitivity across focal sizes and original irregular bags.

When a method produces tied scores, the requested area budget can expand at the boundary. Results
therefore record the achieved mapped fraction and use that achieved fraction—not the nominal
budget—as the denominator for lift and gain.

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
