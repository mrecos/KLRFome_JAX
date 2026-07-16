# Representation Efficiency and Bag Uncertainty Sprint

**Date:** 2026-07-15

**Branch:** `codex/representation-extensions`

**Status:** Implemented; extension smoke and 90-case replicated research suites completed

## Purpose

This sprint tests three targeted responses to the synthetic laboratory findings without replacing
the M0 scientific reference:

1. improve the finite-feature approximation used by M1 and M2;
2. represent uncertainty in empirical embeddings estimated from small or dependent bags; and
3. test whether mean-embedding and transport geometries provide complementary information.

These are methodological extensions. They do not change the canonical bag contract, focal raster
prediction semantics, presence-background interpretation, or spatial-validation requirements.

## Implemented extensions

### Orthogonal random features

`RandomFourierFeatures` now supports `scheme="iid"` and `scheme="orthogonal"`. The IID path is the
unchanged compatibility default. Orthogonal random features use sign-corrected QR blocks with
independent chi-distributed radial scaling, and work for arbitrary feature counts and input
dimensions.

The relevant comparison is ORF versus IID at the **same feature budget**. Lower kernel error,
population-reference embedding error, or more stable prediction ranks indicates improved
efficiency. Comparing ORF-128 with IID-512 does not isolate the sampling scheme.

### Shrinkage mean embeddings

M1 and M2 can use `embedding_estimator="shrinkage"`. The empirical RFF mean is multiplied by a
data-derived factor in `[0, 1]` that balances estimated signal magnitude against finite-bag feature
variance. A factor of 1 is the empirical embedding; smaller factors express greater uncertainty.

Uniformly duplicated coordinates do not create false precision. The estimator deduplicates
repeated coordinates for its uncertainty calculation, and an explicitly supplied
`metadata["effective_sample_size"]` can represent known dependence. Scaling and shrinkage are fit
inside each training fold.

This estimator is useful only if it lowers population-reference embedding error for small,
unequal, or dependent bags without creating a systematic penalty for well-sampled bags. Predictive
AUC is a secondary consequence, not the estimator's defining test.

### Experimental M4 hybrid

M4 combines a mean-embedding kernel and the M3 sliced-Wasserstein kernel:

\[
K_{M4}=w K_{mean} + (1-w)K_{SW}, \qquad 0\leq w\leq1.
\]

Each component can be normalized to a comparable diagonal scale. The mean component can use an
exact or RFF embedding. A weight of 1 is the mean-embedding endpoint and a weight of 0 is the
transport endpoint. The high-level legacy argument mapping intentionally cannot select M4; it must
be requested with an explicit `ModelSpec` because it remains experimental.

When a weight grid is configured in the laboratory, grouped inner cross-validation selects the
weight using only the outer training data. Selection is deterministic and the outer test fold is
never used for tuning. Endpoint tests verify that unnormalized exact-mean M4 reproduces M0 at
`w=1` and M3 at `w=0`.

## Diagnostic contract

Synthetic result schema 1.2 adds `embedding_mse` to each fold-method record. Independent large
reference bags are generated from the same known case distribution, transformed by the fitted
training preprocessor, and represented by their unshrunk empirical RFF mean. This is a controlled estimate of
finite-bag representation error, not an empirical-data metric.

Model diagnostics and archives record:

- RFF sampling scheme;
- empirical or shrinkage estimator;
- effective-size mode and fitted shrinkage summaries;
- M4 component choice, component scales, and selected weight; and
- the exact random-feature and projection state required for round-trip prediction.

## Experiment tiers

### Extension smoke

`benchmarks/synthetic_lab_extensions_smoke_config.json` is a four-case execution gate. It covers a
null case, two bag sizes under a weak mean shift, and a moment-matched XOR case. It exercises IID,
ORF, shrinkage, M3, and nested M4 selection. It is not large enough for method conclusions.

Acceptance result on 2026-07-15:

- 4 cases and 48 fold-method rows completed;
- all configured methods returned finite outputs;
- all permutation and uniform-duplication invariance checks passed;
- population-reference error was recorded for every RFF-based fold; and
- M4 selected weights from the configured `{0, 0.5, 1}` grid without outer-fold leakage.

### Replicated extension suite

`benchmarks/synthetic_lab_extensions_config.json` is a 90-case research configuration. It emphasizes
null calibration, equal-budget IID/ORF comparison, small and unequal bags, spatial dependence,
sparse features, moment-matched nonlinear shape, variance, and correlation changes. Its cost is
deliberately higher because M4 performs nested validation.

The notebook mode `extensions_smoke` should be run first. Change it to `extensions` only after the
smoke output is interpretable. Generated results remain under ignored `benchmark_data/`.

The completed 90-case interpretation is recorded in
[Synthetic Laboratory Extension Results](SYNTHETIC_LAB_EXTENSION_RESULTS_2026-07-16.md). ORF improved
equal-budget fidelity, nominal shrinkage helped small bags, coordinate dependence exposed the limits
of unique-cell count, and M4 remained experimental.

A bounded pilot of four full-size cases (null, moment-matched XOR, three-cell mean shift, and
spatially dependent mean shift) also completed on 2026-07-15. All 192 fold-method records had
finite AUC and PR AUC. Across this deliberately small pilot, shrinkage had the lowest mean
population-reference embedding error (0.0511 versus approximately 0.0548--0.0550 for the empirical
RFF variants), while M4 selected all three candidate weights. These values verify that the research
diagnostics behave as designed; four cases are not evidence for promotion.

## Advancement gates

- **ORF:** lower error than IID at equal feature count across replicated cases, without null drift.
- **Shrinkage:** lower small/dependent-bag embedding error, no material large-bag penalty, and stable
  invariance behavior.
- **M4:** coherent scenario-specific weight selection plus repeated outer-fold gains; no claim based
  on inner weights alone.
- **All extensions:** finite fits, round-trip predictions, PSD diagnostics, immutable outer folds,
  and no preprocessing or tuning leakage.

No extension is promoted or removed based on synthetic results alone. Successful candidates return
to mapped Section 6 validation before any default changes. ARD/grouped kernels remain the next
unimplemented direction for sparse-signal problems.

## Focused spatial follow-up

The replicated run motivated two bounded implementation changes:

- coordinate-aware effective sample size for shrinkage, using the variance of the equally weighted
  spatial mean under an explicit exponential correlation range; and
- cached bandwidth-free ORF draws keyed by seed, input dimension, and feature count, with the
  training-fold bandwidth applied after cache retrieval.

The focused `synthetic_lab_spatial_shrinkage_config.json` run is the gate before Section 6 changes.
It must recover nominal bag size at zero correlation, reduce effective size monotonically as the
configured range increases, and report paired population-reference embedding MSE against nominal
shrinkage. The correlation range remains explicit; estimating it from empirical bags is deferred.

## Method references

- Yu et al. (2016), [Orthogonal Random Features](https://proceedings.neurips.cc/paper/2016/hash/53adaf494dc89ef7196d73636eb2451b-Abstract.html).
- Griffith (2005), [Effective Geographic Sample Size in the Presence of Spatial Autocorrelation](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8306.2005.00484.x).
- Muandet et al. (2014), [Kernel Mean Estimation and Stein's Effect](https://proceedings.mlr.press/v32/muandet14.html).
- Wolfer and Alquier (2025), [Variance-Aware Estimation of Kernel Mean Embedding](https://www.jmlr.org/papers/v26/23-0161.html).
- Rakotomamonjy et al. (2008), [SimpleMKL](https://www.jmlr.org/papers/v9/rakotomamonjy08a.html). The present M4 uses a small leakage-safe grid rather than a learned SimpleMKL optimizer.
