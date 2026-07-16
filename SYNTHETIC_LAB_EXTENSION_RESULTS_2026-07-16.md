# Synthetic Laboratory Extension Results

**Date:** 2026-07-16

**Configuration:** `benchmarks/synthetic_lab_extensions_config.json`

**Scope:** 90 synthetic cases, 4,320 fold-method fits, and 1,080 pooled
out-of-fold method results

## Result summary

The replicated extension suite completed with finite AUC and PR AUC throughout. Null behavior was
centered near chance: method-level mean null AUC ranged from 0.480 to 0.516.

At 128 frequencies, orthogonal random features (ORF) improved the approximation to M0 relative to
IID features:

| Diagnostic | ORF-128 | IID-128 |
|---|---:|---:|
| Relative kernel error | 0.0229 | 0.0248 |
| Kernel correlation | 0.9956 | 0.9677 |
| Prediction-score Spearman correlation | 0.9714 | 0.9508 |

IID-512 still produced the strongest mean rank agreement (0.9816), as expected from its larger
feature budget. The equal-budget result supports ORF as the more efficient approximation, not as a
replacement for checking larger feature budgets.

Nominal shrinkage reduced population-reference embedding MSE most clearly for very small bags.
The mean paired MSE improvement was 0.01369 at bag size 3, 0.00507 at size 5, 0.00147 at size 10,
and 0.00017 at size 30. This is coherent finite-bag behavior.

The spatial-dependence cases exposed a limitation: nominal shrinkage uses the number of unique
cells, so its shrinkage factor moved closer to 1 as simulated dependence increased. Unique-cell
count prevents duplicated coordinates from creating false precision, but it cannot recover the
information loss caused by correlation among distinct nearby cells.

M4 selected scenario-coherent component weights, but 82% of selections were at a pure-component
endpoint and it did not consistently outperform the better endpoint. M4 therefore remains
experimental. The result supports retaining nonlinear bag-level kernels and Wasserstein methods as
separate candidates rather than promoting the hybrid.

## Decisions

- Retain ORF as the preferred equal-budget RFF construction candidate.
- Retain shrinkage for small-bag research, but replace nominal cell count with a coordinate-aware
  effective size before empirical validation.
- Cache bandwidth-free ORF draws so folds with the same seed, feature count, and input dimension do
  not repeatedly perform QR construction; fold-specific bandwidth scaling remains unchanged.
- Keep M4 experimental and out of the focused follow-up.
- Do not change Section 6 yet. First run the focused spatial-shrinkage laboratory and review its
  representation-level diagnostics.

## Focused follow-up

`benchmarks/synthetic_lab_spatial_shrinkage_config.json` contains 40 replicated cases. It compares
M0, IID-128, ORF-128, nominal ORF shrinkage, and coordinate-aware ORF shrinkage across the
independent limit, small bags, and increasing spatial correlation ranges.

For coordinates \(s_i\), the spatial effective size matches the variance of the unweighted bag
mean under an exponential correlation model:

\[
n_{eff}=\frac{n^2}{\mathbf{1}^{T}R\mathbf{1}}, \qquad
R_{ij}=\exp\left(-\frac{\lVert s_i-s_j\rVert}{\rho}\right).
\]

The correlation range \(\rho\) is explicit model input or bag metadata. This sprint does not claim
that it can be estimated reliably from a single bag. The primary decision statistic is the paired
change in population-reference embedding MSE; AUC is secondary.

This variance-matching use of an effective geographic sample size follows the general rationale in
Griffith (2005), [Effective Geographic Sample Size in the Presence of Spatial
Autocorrelation](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8306.2005.00484.x).
The ORF construction follows Yu et al. (2016), [Orthogonal Random
Features](https://papers.neurips.cc/paper_files/paper/2016/hash/53adaf494dc89ef7196d73636eb2451b-Abstract.html).
