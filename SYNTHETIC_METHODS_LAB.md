# Synthetic Methods Laboratory

The synthetic laboratory tests when KLRfome's four reference representations differ under known
bag-level data-generating processes. It is an evaluation and reproducibility layer, not a new model.

## Questions

- How closely does M1 approximate M0 as the random-feature budget increases?
- Does M2 help on nonlinear functionals of a distribution, or mainly increase overfitting risk?
- Which scale, tail, multimodal, or dependence changes favor M3?
- How do small, unequal, duplicated, or spatially dependent cell samples affect each method?
- Do conclusions remain stable across seeds and paired folds?

## Scenarios

`klrfome.data.synthetic` generates canonical `BagDataset` objects for null, mean, variance,
heavy-tail, moment-matched multimodal, correlation, sparse-signal, and nonlinear-mixture problems.
The nonlinear-mixture case contrasts pure component bags with bags containing the same components
as a mixture. It is designed to probe nonlinear functionals of an embedding rather than a simple
mean shift.

Cell order is explicitly a nuisance. Uniformly repeating every cell preserves the empirical
distribution, whereas repeating only selected cells changes probability mass and is therefore an
intentional distributional change. Spatial dependence is induced with a Gaussian copula that
uses the estimated marginal distribution while allowing realized bag summaries to vary as
effective information decreases.

The targeted v2 suite adds a moment-matched XOR problem. Every bag has the same population
feature-wise means and variances, while Gaussian-versus-bimodal shape states form an XOR pattern
across two features. It is designed to test nonlinear distribution geometry without giving
mean-plus-standard-deviation logistic regression a direct signal.

## Running the laboratory

Run the fast CI-sized configuration:

```bash
python benchmarks/run_synthetic_methods_lab.py \
  --config benchmarks/synthetic_lab_smoke_config.json
```

List the cases in the larger research configuration without fitting:

```bash
python benchmarks/run_synthetic_methods_lab.py \
  --config benchmarks/synthetic_lab_config.json \
  --list-cases
```

Run the research configuration:

```bash
python benchmarks/run_synthetic_methods_lab.py \
  --config benchmarks/synthetic_lab_config.json
```

Run the targeted follow-up:

```bash
python benchmarks/run_synthetic_methods_lab.py \
  --config benchmarks/synthetic_lab_targeted_v2_config.json
```

Run one or more zero-based cases reported by `--list-cases`:

```bash
python benchmarks/run_synthetic_methods_lab.py \
  --config benchmarks/synthetic_lab_config.json \
  --case-index 0 --case-index 4
```

Raw output is written under ignored `benchmark_data/`. The runner records the configuration hash,
scientific dataset fingerprint, exact fold assignments, runtime environment, method diagnostics,
paired differences, pooled out-of-fold predictions, and optional invariance checks. Result schema
1.1 calculates ranking metrics after pooling every held-out bag within a repeat; fold metrics remain
available as diagnostics but are not treated as independent scientific replicates. Undefined
metrics are JSON `null` rather than non-standard `NaN`.

The first complete core-run interpretation is recorded in
[Synthetic Laboratory Core Results](SYNTHETIC_LAB_RESULTS_2026-07-15.md).

## Reproducibility and fitted archives

`klrfome.utils.serialization` stores fitted models as versioned `.klrfome` ZIP archives containing
only `manifest.json` and `arrays.npz`. Loading uses `allow_pickle=False`; archives do not execute
Python objects. M0 and M3 retain the reference bag state needed for prediction. M1 stores only its
preprocessor, random-feature state, and primal coefficients. M2 stores its random-feature state,
reference bag embeddings, decision bandwidth, and dual coefficients.

Round-trip predictions are tested for M0--M3 and for the public `KLRfome` facade.

## Interpretation

Synthetic success establishes that a method can identify a controlled distributional difference;
it does not establish superiority for archaeological data. Synthetic failure on the specific signal
a representation claims to capture is stronger evidence for revision than a tie on one empirical
subset. No method is promoted or removed from this laboratory alone.

The next extension is selected conditionally:

- inadequate M1 fidelity suggests improved random features;
- small-bag instability suggests shrinkage embeddings;
- sparse-signal failure suggests ARD or grouped kernels;
- reliable nonlinear gains support further M2 work; and
- shape-sensitive gains support further M3 or hybrid-kernel work.
