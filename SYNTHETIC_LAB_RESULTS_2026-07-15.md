# Synthetic Methods Laboratory: Core and Targeted Results

**Date:** 2026-07-15

**Configuration SHA-256:** 85d3e88d6ff7b7da19859d01ca812655779f1e504cbef92bc5c41a9fc78b7e41

**Code revision:** 3213877048f2a5a814f1036dc9ebef46a7764a3b

**Result schema:** 1.0

**Raw result:** ignored benchmark_data/synthetic_lab_results.json

**Targeted-v2 configuration SHA-256:**
eab4697e0a4bf12efc65675cdb35a3175321d6265b846dd69d064cf841001455

**Targeted-v2 code revision:** b8cd5cdc6bb83c546660b77ba27c115fdbd2e3d3

**Targeted-v2 result schema:** 1.1

**Targeted-v2 raw result:** ignored benchmark_data/synthetic_lab_targeted_v2_results.json

## Scope

The core run completed 64 independently seeded synthetic cases and produced 1,728 fold-method
records. It compared M0, three M1 random-feature budgets, M2, M3, logistic-regression summaries,
and a random-forest summary baseline on identical grouped folds.

This run tests whether each representation behaves coherently under controlled distributional
signals. It does not rank methods for archaeological application and does not authorize removing
any implemented method.

The targeted-v2 run then completed 68 independently seeded cases, 2,176 fold-method records, and
544 pooled out-of-fold method records. It directly tested the limitations identified by the core
run: null behavior, moment-matched nonlinear geometry, weak signals in small and unequal bags,
spatial dependence, and sparse signals in 30 dimensions. All cases and fits completed without an
explicit failure or nonconvergence diagnostic.

## Targeted-v2 findings

Selected mean pooled out-of-fold ROC AUC results:

| Condition | M0 | M1-128 | M1-512 | M2 | M3 | Best simple baseline |
|---|---:|---:|---:|---:|---:|---:|
| Moment-matched XOR, effect 0.95 | 0.490 | 0.492 | 0.479 | 0.638 | **0.954** | LR-mean 0.551 |
| Mean shift, bag size 3 | 0.473 | 0.469 | 0.475 | 0.449 | 0.495 | RF-mean 0.527 |
| Mean shift, bag size 30 | **0.818** | 0.812 | 0.817 | 0.801 | 0.796 | LR-mean 0.806 |
| Mean shift, spatial range 5 | 0.538 | 0.556 | 0.538 | 0.534 | 0.522 | LR-mean 0.562 |
| Mean shift, unequal bags | 0.631 | 0.627 | 0.623 | 0.646 | **0.664** | LR-mean 0.656 |
| Sparse signal, effect 0.30 | 0.830 | 0.768 | **0.834** | 0.763 | 0.775 | RF-mean 0.826 |

The targeted run supports five conclusions:

1. **M3 captures distribution shape that ordinary summaries and linear mean embeddings miss.** At
   XOR effect 0.95, M3 also achieved PR AUC 0.945, Boyce 0.888, and top-5% lift 1.92. At effect
   0.85 its mean AUC was only 0.579 with an interval overlapping chance, so this sensitivity is not
   universal at weak separation.
2. **M1-512 is the reliable high-fidelity M0 approximation.** Its mean absolute pooled-AUC
   difference from M0 was 0.007, compared with 0.016 for M1-128. M1-128 lost 0.062 AUC relative to
   M0 in the stronger 30-dimensional sparse-signal condition, while M1-512 recovered M0.
3. **Very small bags contain insufficient independent information for every tested method.** Mean
   AUC was near chance at three cells, approximately 0.59--0.64 at five to ten cells, and
   approximately 0.80--0.82 at 30 independent cells. This is not evidence of an implementation
   defect, although shrinkage may reduce estimator variance.
4. **Spatial dependence reduces effective bag size.** With nominal bag size fixed at 30, AUC fell
   from approximately 0.82 without dependence to approximately 0.60 at range 1.5 and approximately
   0.54 at range 5 across methods. Distribution representation does not remove uncertainty caused
   by spatially redundant cells.
5. **Matched unequal bag sizes did not cause a method-specific pathology.** No paired difference
   from M0 was clearly separated from zero in this condition.

Across eight null replicates, every method-level 95% interval included ROC AUC 0.5. The common
method means ranged from 0.554 to 0.572 because all methods saw the same finite synthetic datasets;
there is no clear leakage warning, but a larger null suite would be required for formal calibration.

## Main findings

1. **No clear execution pathology or leakage appeared.** Across five null cases, the largest mean
   method-level departure from ROC AUC 0.5 was 0.078. Five null replicates are a sanity check, not a
   precise bias estimate.
2. **M1 is a credible scalable approximation to M0.** The 128-feature variant had mean relative
   kernel error 0.024 and mean score-rank correlation 0.977. The 512-feature variant improved these
   to 0.013 and 0.991. The 32-feature variant was materially less stable.
3. **M2 adds useful nonlinear bag-level capacity.** It separated the nonlinear-mixture case
   perfectly and improved low-effect variance and correlation cases, but sometimes had larger
   train-test gaps.
4. **M3 provides coherent shape sensitivity.** It improved strong heavy-tail and multimodal cases,
   both correlation effects, low-effect variance, and nonlinear mixtures. It was also the slowest
   predictor among M0-M3.
5. **Simple baselines remain scientifically necessary.** Mean logistic regression matched the
   distribution methods on mean shifts. Mean-plus-standard-deviation logistic regression was best
   on the low variance shift and tied M2/M3 on the original nonlinear mixture.

## M1 fidelity

| Variant | RFF features | Mean relative kernel error | Maximum error | Mean score Spearman | Minimum Spearman | Mean top-5% overlap | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| M1-rff32 | 32 | 0.053 | 0.158 | 0.942 | 0.785 | 0.781 | Low-budget sensitivity only |
| M1 | 128 | 0.024 | 0.104 | 0.977 | 0.893 | 0.885 | Provisional operational default |
| M1-rff512 | 512 | 0.013 | 0.038 | 0.991 | 0.974 | 0.927 | High-fidelity validation option |

The automated notebook initially labeled the whole M1 family REVIEW because it combined these
budgets. They must be gated separately.

## Distribution-signal results

Selected case-mean ROC AUC results:

| Scenario | Effect | M0 | M1-128 | M2 | M3 | Best simple baseline | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Mean shift | 0.35 | 0.868 | 0.868 | 0.854 | 0.859 | LR-mean 0.865 | Ordinary means are sufficient |
| Variance shift | 0.35 | 0.719 | 0.771 | 0.866 | 0.863 | LR-mean-std 0.955 | Scale signal; simple variance summary dominates |
| Correlation shift | 0.35 | 0.679 | 0.707 | 0.786 | 0.738 | LR-mean-std 0.448 | Distribution methods recover dependence |
| Heavy tail | 0.80 | 0.573 | 0.580 | 0.623 | 0.637 | LR-mean-std 0.481 | Modest shape-sensitive advantage |
| Multimodal | 0.90 | 0.639 | 0.641 | 0.637 | 0.710 | LR-mean-std 0.663 | M3 gains under stronger multimodality |
| Sparse signal | 0.35 | 0.873 | 0.851 | 0.844 | 0.872 | LR-mean 0.865 | No current sparse-signal failure |
| Nonlinear mixture | 1.25 | 0.325 | 0.323 | 1.000 | 1.000 | LR-mean-std 1.000 | Nonlinear/shape geometry matters, but variance confounds the test |

The nonlinear-mixture result is geometrically coherent. The mixture embedding lies between two
pure-component groups, so a linear decision rule on M0/M1 embeddings cannot put both outer groups
on one side and the midpoint on the other. M2 can use a nonlinear boundary. M3 and mean-plus-scale
summaries also solve the case because the generated classes differ strongly in within-bag spread.

## Computational observations

Median descriptive timings across the complete run:

| Method | Median fit seconds | Median prediction seconds |
|---|---:|---:|
| M0 | 0.066 | 0.029 |
| M1-128 | 0.048 | 0.017 |
| M1-512 | 0.086 | 0.019 |
| M2 | 0.051 | 0.019 |
| M3 | 0.171 | 0.206 |

These values mix JAX tracing/compilation and steady-state execution. They support only provisional
relative statements; dedicated scaling benchmarks remain necessary.

## Limitations discovered by the run

1. Fold-level Boyce and top-5% lift were too coarse. Each test fold placed only one bag in the top
   5%, and Boyce was defined for only 64% of fold-method rows. Future results must pool all
   out-of-fold predictions within a repeat before computing these metrics.
2. Most primary conditions had only three independently generated cases. Normal intervals with
   three replicates are fragile. Future summaries use small-sample t intervals and retain raw
   replicate values.
3. Bag-size, spatial-dependence, and unequal-bag experiments used a mean-shift effect of 0.75 and
   largely saturated near AUC 1.0. They cannot decide whether shrinkage is useful.
4. The original spatial generator rank-reordered a fixed empirical sample. That preserved too much
   bag information and did not properly express reduced effective sample size. It has been replaced
   with a correlated Gaussian-quantile construction for subsequent runs.
5. The original nonlinear mixture is not moment matched at the bag-summary level. A new
   moment-matched XOR distribution scenario is required to distinguish nonlinear distribution
   geometry from ordinary mean/standard-deviation summaries.

## Provisional method roles

- **M0:** exact lineage-preserving scientific reference.
- **M1-128:** operational scalable default for ordinary low- and moderate-dimensional problems,
  with an explicit sparse/high-dimensional fidelity check.
- **M1-512:** high-fidelity approximation and preferred large-feature validation option.
- **M1-32:** low-resource sensitivity only.
- **M2:** retained nonlinear bag-boundary hypothesis.
- **M3:** validated shape-sensitive transport hypothesis, retained despite higher cost.
- **LR mean, LR mean/std, and RF mean:** required comparison baselines.

## Roadmap decisions

1. Improve M1/M2 random features through adaptive feature budgets, orthogonal random features, or
   another variance-reduced construction. Compare fidelity, time, and memory against both M1-128
   and M1-512 before changing the operational default.
2. Prototype training-fold-only shrinkage or effective-sample-size-aware embeddings for small and
   spatially correlated bags. The experiment must distinguish reduced estimator variance from
   irreducible information loss.
3. Test a leakage-safe hybrid mean-embedding plus Wasserstein kernel. M0 and M3 now have clear,
   complementary strengths on sparse mean signals and moment-matched distribution shape.
4. Defer ARD and grouped kernels until these experiments establish whether feature weighting adds
   value beyond a larger or improved random-feature approximation.

No implemented M0--M3 method should be promoted or removed from this synthetic evidence alone.
