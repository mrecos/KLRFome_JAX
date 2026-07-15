# Synthetic Methods Laboratory

The synthetic laboratory tests when KLRfome's reference representations and experimental
extensions differ under known bag-level data-generating processes. It is an evaluation and
reproducibility layer, not a model-ranking contest.

## Questions

- How closely does M1 approximate M0 as the random-feature budget increases?
- Does M2 help on nonlinear functionals of a distribution, or mainly increase overfitting risk?
- Which scale, tail, multimodal, or dependence changes favor M3?
- How do small, unequal, duplicated, or spatially dependent cell samples affect each method?
- Do conclusions remain stable across seeds and paired folds?
- Do orthogonal random features improve M1 fidelity at a fixed feature budget?
- Does shrinkage reduce finite-bag embedding error when bags are small or spatially dependent?
- Can a leakage-safe hybrid of mean-embedding and transport kernels adapt to different signals?

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

Validate the representation extensions before starting their research suite:

```bash
python benchmarks/run_synthetic_methods_lab.py \
  --config benchmarks/synthetic_lab_extensions_smoke_config.json

python benchmarks/run_synthetic_methods_lab.py \
  --config benchmarks/synthetic_lab_extensions_config.json
```

Run one or more zero-based cases reported by `--list-cases`:

```bash
python benchmarks/run_synthetic_methods_lab.py \
  --config benchmarks/synthetic_lab_config.json \
  --case-index 0 --case-index 4
```

Raw output is written under ignored `benchmark_data/`. The runner records the configuration hash,
scientific dataset fingerprint, exact fold assignments, runtime environment, method diagnostics,
paired differences, pooled out-of-fold predictions, optional invariance checks, population-reference
embedding error, and fitted representation diagnostics. Result schema 1.2 calculates ranking
metrics after pooling every held-out bag within a repeat; fold metrics remain diagnostics and are
not independent scientific replicates. Undefined metrics are JSON `null`, not non-standard `NaN`.

For extension runs, the runner creates independent large reference bags from the known synthetic
distribution. Their fitted-preprocessor, unshrunk RFF means estimate the population target used for
`embedding_mse`. M4 mixture weights are chosen by grouped inner validation using only the outer
training fold. The outer test fold therefore remains untouched by weight selection.

The first complete core-run interpretation is recorded in
[Synthetic Laboratory Core Results](SYNTHETIC_LAB_RESULTS_2026-07-15.md).

## Reproducibility and fitted archives

`klrfome.utils.serialization` stores fitted models as versioned `.klrfome` ZIP archives containing
only `manifest.json` and `arrays.npz`. Loading uses `allow_pickle=False`; archives do not execute
Python objects. M0 and M3 retain the reference bag state needed for prediction. M1 stores only its
preprocessor, random-feature state, and primal coefficients. M2 stores its random-feature state,
reference bag embeddings, decision bandwidth, and dual coefficients.

Round-trip predictions are tested for M0--M4 and for the public `KLRfome` facade. M1, M2, and M4
archives retain the exact IID or orthogonal feature state and any fitted shrinkage factors.

## Interpretation

Synthetic success establishes that a method can identify a controlled distributional difference;
it does not establish superiority for archaeological data. Synthetic failure on the specific signal
a representation claims to capture is stronger evidence for revision than a tie on one empirical
subset. No method is promoted or removed from this laboratory alone.

Extension decisions are conditional:

- retain orthogonal random features only when they reduce approximation error at the same budget;
- retain shrinkage only when it reduces small/dependent-bag embedding error without harming larger bags;
- sparse-signal failure suggests ARD or grouped kernels;
- reliable nonlinear gains support further M2 work; and
- retain M4 only when selected weights and outer-fold gains repeat coherently by scenario.

The implementation and acceptance design are recorded in
[Representation Extensions Sprint](REPRESENTATION_EXTENSIONS_2026-07-15.md).
