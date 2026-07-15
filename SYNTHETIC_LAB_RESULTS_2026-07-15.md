# Synthetic Methods Laboratory: Core Results

**Date:** 2026-07-15

**Configuration SHA-256:** 85d3e88d6ff7b7da19859d01ca812655779f1e504cbef92bc5c41a9fc78b7e41

**Code revision:** 3213877048f2a5a814f1036dc9ebef46a7764a3b

**Result schema:** 1.0

**Raw result:** ignored benchmark_data/synthetic_lab_results.json

## Scope

The core run completed 64 independently seeded synthetic cases and produced 1,728 fold-method
records. It compared M0, three M1 random-feature budgets, M2, M3, logistic-regression summaries,
and a random-forest summary baseline on identical grouped folds.

This run tests whether each representation behaves coherently under controlled distributional
signals. It does not rank methods for archaeological application and does not authorize removing
any implemented method.

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
- **M1-128:** operational scalable default.
- **M1-512:** high-fidelity approximation check.
- **M1-32:** low-resource sensitivity only.
- **M2:** retained nonlinear bag-boundary hypothesis.
- **M3:** retained shape-sensitive transport hypothesis.
- **LR mean, LR mean/std, and RF mean:** required comparison baselines.

## Next experiment

The tracked benchmarks/synthetic_lab_targeted_v2_config.json addresses unresolved evidence:

- eight null replicates;
- a moment-matched nonlinear XOR scenario;
- weaker mean shifts across small bag sizes;
- corrected spatial dependence under weak signal;
- exactly class-matched unequal bag-size schedules; and
- harder sparse signals in 30 dimensions.

ARD and shrinkage remain deferred until this targeted evidence establishes a specific failure they
can plausibly correct.
