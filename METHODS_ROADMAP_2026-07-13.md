# KLRFome Methods Status and Roadmap

**Date:** 2026-07-13
**Status:** Working methodological reference; no implementation changes are authorized by this document

## Purpose

This document records the current understanding of KLRFome's statistical rationale,
methodological history, implemented alternatives, evidence, and recommended research roadmap.
It is intended to prevent implementation work from outrunning the scientific design while more
representative spatial data are being recovered.

The central conclusion is:

> The original distribution-regression rationale remains sound. The newer representations have
> not yet demonstrated a reliable improvement, but the current data and evaluation design are not
> sufficient to rule them out. Preserve the exact mean-embedding model as the scientific reference,
> retain the scalable and transport-based alternatives as explicit experimental models, and compare
> them under controlled synthetic tests and later under realistic spatial validation.

## 1. The Scientific Problem

The modeling unit is a site, not an individual raster cell. Each site contains a bag of cells with
measured environmental covariates, and those cells are strongly correlated because they come from
the same spatial unit.

Two common simplifications are unsatisfactory:

1. Reducing a site to a centroid, mean, or other single summary discards within-site variation,
   multimodality, tails, and potentially important environmental mixtures.
2. Treating every cell as an independent labeled observation creates pseudoreplication and can
   severely overfit because the response is defined at the site level while nearby cells are highly
   correlated.

KLRFome instead represents each site as an empirical distribution:

\[
P_i = \frac{1}{n_i}\sum_{a=1}^{n_i}\delta_{x_{ia}},
\]

where site \(i\) contains \(n_i\) cells and each \(x_{ia}\) is a vector of environmental variables.
The model receives one response label per bag. A distribution kernel makes bags comparable without
pretending that their cells are independent response observations.

### Original kernel mean-embedding model

For a cell-level positive-definite kernel \(k\), the empirical mean embedding of bag \(P_i\) is

\[
\widehat{\mu}_{P_i}=\frac{1}{n_i}\sum_a \phi(x_{ia}).
\]

The original bag kernel is the inner product between two embeddings:

\[
K(P_i,P_j)
= \left\langle\widehat{\mu}_{P_i},\widehat{\mu}_{P_j}\right\rangle
= \frac{1}{n_i n_j}\sum_{a,b} k(x_{ia},x_{jb}).
\]

With a characteristic kernel such as an appropriately parameterized Gaussian RBF, this is not
merely a comparison of ordinary means. At the population level, the embedding can distinguish full
probability distributions. In finite samples, however, its effective resolution depends on bag
size, kernel bandwidth, covariate scaling, and any feature approximation.

The kernel logistic regression layer then predicts one site-level label from similarities among
bags. This preserves the original statistical intent of distribution regression while avoiding
cell-level pseudoreplication.

## 2. Methodological Journey

### Phase 1: R implementation and faithful Python/JAX port

The project began in R as kernel logistic regression over exact distribution-kernel values. It was
ported to Python to make the implementation more accessible and to JAX to improve vectorization,
compilation, accelerator support, and computational efficiency.

The faithful reference method remains:

- exact empirical kernel mean embedding using averaged pairwise cell-level RBF similarities;
- a linear inner product in the embedding's reproducing-kernel Hilbert space; and
- kernel logistic regression fit with an IRLS-style solver derived from the R workflow.

This method is the lineage-preserving baseline against which later representations should be
compared.

### Phase 2: Sliced-Wasserstein representation

Sliced Wasserstein was added to compare bags through distributions of one-dimensional random
projections. For each projection direction, cells are projected, their empirical quantiles are
compared, and distances are aggregated over directions. The current high-level implementation uses
a fixed quantile grid so unequal bag sizes can be compared consistently.

This is a genuinely different distribution geometry. It may be sensitive to changes in spread,
quantiles, and support in ways that differ from a kernel mean embedding. It is therefore an
alternative scientific hypothesis, not simply a faster implementation of the original kernel.

The principled default is the order-2 version with an RBF kernel on the resulting distance. The
current order-1 form should not be treated as equally safe without revision: squaring an L1-style
distance inside a Gaussian kernel is not guaranteed to produce a positive-semidefinite Gram matrix.

### Phase 3: Correctness and data-pipeline repairs

Recent work addressed issues including covariate scaling, bandwidth interpretation, degenerate
bags, reconstructed spatial background bags, and fixed-grid quantiles. These changes were necessary
for meaningful comparison, but some pipeline behavior has become specialized to the data currently
available.

### Phase 4: Presence-background evaluation

The current negative class is better described as sampled background than as confirmed absence.
Consequently, model output should currently be interpreted as relative suitability or relative
intensity, not as calibrated probability of occurrence across Pennsylvania.

Presence-only measures such as continuous Boyce index, area capture, and lift are useful, but they
do not cure sampling-domain confounding. High discrimination can arise when presences and background
come from different geographic or environmental sampling domains.

### Phase 5: Random Fourier features and bag-level nonlinear kernels

Phase-free random Fourier features were added to approximate the cell-level RBF embedding:

\[
z(x)=[\cos(Wx),\sin(Wx)], \qquad
\widehat{\mu}_{P_i}\approx\frac{1}{n_i}\sum_a z(x_{ia}).
\]

Two decision geometries can then be applied:

- a linear kernel on the approximate bag embeddings; and
- an RBF kernel on distances between the bag embeddings.

The linear version is an approximation to the original scientific model. The RBF-on-embeddings
version adds another nonlinear layer at the bag level and is therefore a capacity extension rather
than a neutral speed optimization.

### Current interpretation of the bake-off

The most recent committed validation output reported broadly useful discrimination, including a
mean-embedding test AUC around 0.72 and a lower sliced-Wasserstein result in one split. Later commit
messages describe multi-seed results in which the newer methods did not outperform the original
method. However, the committed notebook does not contain a fully executed, reproducible record of
the final multi-seed comparison.

More importantly, the current comparison uses a small and compromised evaluation setting:

- site-present data retain site grouping and measured covariates;
- the available non-site/background samples cover only a limited riverine portion of the much
  larger study area;
- spatial collinearity and original sampling structure were lost in the background CSV;
- background bags had to be reconstructed;
- the held-out sample contains only a modest number of sites; and
- expected future inputs are likely to be aligned rasters and polygon GIS files, not the current
  CSV layout.

The supported conclusion is therefore **no detected improvement under the current data and
evaluation**, not **evidence that the newer representations are inferior in the intended statewide
setting**.

## 3. Current Method Inventory and Status

| ID | Representation and decision kernel | Scientific role | Current status |
|---|---|---|---|
| M0 | Exact RBF mean embedding + linear bag kernel + KLR | Canonical reference and continuity with the R model | Implemented; retain |
| M1 | RFF mean embedding + linear bag kernel | Scalable approximation to M0 | Implemented; retain and validate approximation error |
| M2 | RFF mean embedding + RBF bag kernel | Added nonlinear capacity among distributions | Implemented in lower-level/tabular workflow; needs a clean public comparison interface |
| M3 | Fixed-quantile sliced-Wasserstein-2 distance + RBF kernel + KLR | Alternative transport-based distribution geometry | Implemented; retain as experimental |

### What appears complete for the current methods

The core mathematical paths for M0, M1, and M3 exist, and the full current test suite passes. Core
kernel, logistic-regression, and Wasserstein modules have meaningful unit-test coverage. The code is
sufficient for controlled experimentation with the methods currently represented.

“Complete” should nevertheless be qualified. The project is not yet complete as a robust,
production-ready statewide spatial modeling system because several cross-cutting concerns remain:

- high-level API parity among M0-M3;
- spatially appropriate and leakage-free model selection;
- realistic raster and polygon ingestion semantics;
- robust numerical behavior under extreme logits;
- scalable training beyond materializing a dense bag-by-bag Gram matrix;
- complete model persistence and reproducible fitted-state restoration;
- operational parallel prediction; and
- stronger tests for I/O, serialization, prediction orchestration, and visualization.

### Important corrections to the prior narrative

1. A mean embedding is not an ordinary covariate mean. With a characteristic cell kernel it is a
   full-distribution representation in principle, although finite samples and approximations limit
   it in practice.
2. Sliced Wasserstein is not the only method that “sees distribution shape.” It imposes a different
   geometry and may emphasize different discrepancies.
3. A high-level Wasserstein focal/raster predictor exists. What is missing is equivalent support in
   every specialized large tabular prediction workflow and a fully uniform method-selection API.
4. RFF removes the expensive exact cell-pair comparison when constructing embeddings, but the
   present downstream KLR path can still materialize an \(N\times N\) Gram matrix and solve a dense
   system. It therefore does not by itself make training linear in the number of bags.
5. The available bake-off is too limited to eliminate M2 or M3.

## 4. Known Technical and Statistical Risks

### Numerical stability

The current JAX configuration generally uses 32-bit floating point. IRLS probability clipping at a
tolerance far below float32 resolution does not prevent exact zero weights after sigmoid saturation.
Extreme logits can therefore produce singular systems or NaN coefficients. A stable sigmoid exists
elsewhere in the code but is not consistently used in the fit path.

This is a data-independent correctness issue and should be fixed before trusting difficult fits.

### Sliced-Wasserstein order-1 kernel validity

The current Gaussian form based on the square of an order-1 distance can produce indefinite Gram
matrices. Until this is corrected and tested, use order 2 as the supported route. A potential order-1
alternative is a Laplace-style kernel \(\exp(-d/\sigma)\), subject to a documented validity review.

### Cross-validation validity

Current validation utilities are not yet a definitive scientific comparison framework:

- random folds do not address spatial dependence;
- preprocessing and hyperparameter selection must occur inside each training fold;
- some non-stratified splitting logic can omit remainder observations;
- one-class folds can make AUC undefined;
- Wasserstein training and test transformations should use the identical fixed-quantile definition;
- all methods must receive exactly the same site-level folds; and
- paired uncertainty across folds/repeats matters more than isolated point estimates.

### Input and spatial semantics

The current high-level workflow does not yet fully enforce CRS agreement, raster alignment, band
ordering, nodata handling, or expected variable metadata. Polygon sites can be reduced to centroids
and square neighborhoods instead of being represented by their complete cell sets. A documented
site-buffer option is not consistently honored. These are important, but their final design depends
on realistic source data.

### Presence-background interpretation

Without verified absences or a defensible sampling design, intercepts and absolute probabilities are
not identified in the ordinary presence/absence sense. Predictions should be labeled as relative
suitability unless prevalence or observation-process information supports calibration.

## 5. Priority Split

### Data-independent work that can proceed now

1. **Numerical correctness**
   - stabilize sigmoid, weights, and linear solves;
   - define float32-appropriate tolerances;
   - add adversarial tests for extreme logits and ill-conditioned Gram matrices;
   - restrict or repair the sliced-Wasserstein order-1 kernel.

2. **A unified representation/decision interface**
   - separate bag representation from the decision kernel and solver;
   - expose M0-M3 through the same fit, predict, and evaluation contract;
   - make the scientific distinction between approximation and added capacity explicit in names and
     documentation.

3. **Synthetic methodological tests**
   - create simulations with known distribution differences;
   - verify invariance to cell order and sensible behavior under duplicated cells;
   - evaluate unequal and small bag sizes;
   - distinguish approximation error from classification error.

4. **Validation framework mechanics**
   - implement fold objects that can later carry spatial groups or buffers;
   - put scaling, feature construction, bandwidth selection, and regularization tuning inside folds;
   - ensure complete observation coverage and graceful handling of invalid folds;
   - report paired differences and uncertainty.

5. **True scalable RFF fitting**
   - when using explicit RFF bag embeddings, fit logistic regression directly in the feature space;
   - avoid constructing a dense Gram matrix when the selected model does not require one;
   - retain equivalence tests against dual KLR on small problems.

6. **Persistence and reproducibility**
   - serialize fitted preprocessing, representation parameters, solver state, variable order, and
     random seeds;
   - round-trip test predictions;
   - record data fingerprints and model configuration in evaluation artifacts.

7. **Method documentation**
   - document the estimand, bag unit, formulas, approximations, and interpretation;
   - label experimental methods and unsupported combinations;
   - keep the exact reference path available even if a scalable method becomes the default.

### Data-dependent work to defer or design provisionally

1. Define the actual site unit from polygons, buffers, archaeological boundaries, or another
   defensible spatial object.
2. Rebuild GIS ingestion around realistic aligned rasters, polygons, CRS metadata, nodata rules, and
   explicit environmental-variable ordering.
3. Design statewide background or absence sampling to reflect the target domain and observation
   process.
4. Resolve pooling across physio-sheds. Many physio-shed × setting strata contain few or no sites,
   so completely separate models may be unidentified or unstable. Compare principled partial
   pooling, hierarchical/multitask models, and scientifically justified aggregation while retaining
   physio-shed and upland/riverine structure. Do not silently pool strata merely to increase sample
   size.
5. Choose bag construction, neighborhood scale, and multi-scale windows using ecological and survey
   meaning rather than the constraints of the current CSV.
6. Conduct definitive spatially blocked or buffered validation.
7. Tune and rank M0-M3 on realistic data.
8. Assess calibration only when the sampling design permits a probabilistic interpretation.
9. Benchmark statewide memory, runtime, tiling, and parallel prediction on representative rasters.

## 6. New Methods Roadmap

### Stage 0: Freeze the estimand and study-design vocabulary

Every run should explicitly declare whether it is using true presence/absence data or
presence/background data. Documentation, output labels, metrics, and calibration claims must follow
that declaration.

Deliverables:

- a small study-design/configuration object or equivalent explicit metadata;
- output language distinguishing occurrence probability from relative suitability;
- documented assumptions about the spatial sampling frame.

### Stage 1: Freeze four reference models

Keep M0-M3 available behind one comparison interface. Do not silently replace M0 with an
approximation, and do not present M2 as merely a faster M1.

Deliverables:

- identical input/output API;
- common fold assignment and preprocessing contract;
- model cards or concise method documentation for M0-M3;
- deterministic seeds and configuration capture.

### Stage 2: Build a synthetic distribution-regression laboratory

Generate bag-level datasets where classes differ in controlled ways:

- location/mean shift;
- variance or scale;
- skewness and tail weight;
- unimodal versus multimodal mixtures with similar low-order moments;
- correlation/dependence structure among covariates;
- sparse signal in one covariate among many noise covariates;
- unequal and very small bag sizes;
- spatially structured versus exchangeable cells; and
- irrelevant duplication or cell-order changes.

Questions to answer:

- Can each representation recover the intended difference?
- How many cells, RFF features, projections, and quantiles are required?
- When does M1 approximate M0 closely enough?
- Does M2 improve nonlinear separation or only overfit?
- Which distribution changes favor M3, and are they plausible for the application?

### Stage 3: Establish a fair evaluation protocol

Use repeated, paired, site-level folds. When realistic coordinates and statewide background become
available, replace or supplement ordinary folds with spatial blocks or buffered leave-location-out
validation.

All learned preprocessing and hyperparameters must be trained inside each fold. Report:

- ROC AUC when valid;
- precision-recall metrics when class balance makes them informative;
- presence-only ranking metrics for presence/background designs;
- calibration only when identifiable;
- paired fold/repeat differences between methods;
- uncertainty intervals;
- fit and prediction time; and
- peak memory.

Method decisions should be based on repeatable improvements that exceed sampling uncertainty, not a
single split or best seed.

### Stage 4: Extend the original kernel lineage first

Before moving to heavily learned representations, test interpretable extensions close to the
original model:

1. **ARD bandwidths:** separate cell-kernel length scales by covariate, selected within folds.
2. **Grouped or multiple kernels:** combine interpretable covariate groups such as terrain,
   hydrology, soils, and climate.
3. **Shrinkage mean embeddings:** reduce noisy empirical embeddings for small bags.
4. **Primal RFF logistic regression:** obtain the intended scale improvement without changing the M1
   scientific representation.
5. **Improved feature sampling:** evaluate orthogonal random features or quasi-Monte Carlo features
   if ordinary RFF requires too many features.

These offer a comparatively direct path to improved precision or scale while preserving the
distribution-regression rationale.

### Stage 5: Add spatial structure when the data support it

The current bag representation is permutation invariant and normally discards internal spatial
arrangement. This is appropriate if only the environmental distribution matters, but it cannot
distinguish two sites with the same environmental histogram and different spatial organization.

Candidate extensions:

- append coordinates relative to the site centroid, with separate scaling from environmental
  covariates;
- use a product kernel combining environmental and relative-spatial similarity;
- compute multiple embeddings for nested spatial scales or distance bands;
- compare full-polygon bags with buffered context bags; and
- use local spatial summaries only when they have ecological or archaeological meaning.

These choices should wait for credible polygons, raster alignment, and sampling geometry.

### Stage 6: Advance transport methods only after a fair M3 test

If M3 shows complementary value in synthetic tests or realistic spatial validation, consider:

- orthogonal or quasi-Monte Carlo projection directions for lower variance;
- max-sliced Wasserstein or learned projections when discriminative directions are rare;
- Sinkhorn divergences when multivariate transport structure justifies the extra cost;
- unbalanced transport if bags differ because of meaningful mass or coverage effects; and
- a hybrid or multiple-kernel model combining KME and sliced-Wasserstein similarities.

A hybrid should be preferred over prematurely replacing M0 when methods capture complementary
signals.

### Stage 7: Consider learned representations only with enough independent sites

Deep Sets, Set Transformers, patch encoders, or pretrained Earth-observation embeddings may provide
richer representations. They also add substantial capacity and are easy to overfit when there are
many cells but few independent labeled sites.

A cautious order is:

1. frozen pretrained Earth-observation embeddings used as additional covariates;
2. small, strongly regularized permutation-invariant bag encoders;
3. learned projection or attention models; and
4. spatial patch models only when the number and geographic diversity of sites justify them.

The effective supervised sample size is the number of independent sites, not the number of cells.

## 7. Decision Gates

### Gate A: Mathematical and numerical correctness

Proceed to substantive model comparison only when:

- Gram matrices for supported kernels pass symmetry and positive-semidefinite diagnostics within
  numerical tolerance;
- extreme-logit fits remain finite;
- train and prediction representations are mathematically identical; and
- all bag-size edge cases are defined and tested.

### Gate B: Synthetic identifiability

Retain an experimental representation for real-data testing when it succeeds on at least the
distribution differences it is designed to detect. Failure on controlled tasks is a stronger reason
for revision than a tie on the current real-data subset.

### Gate C: Realistic spatial data readiness

Begin definitive comparison only when:

- site geometries and covariate rasters have verified CRS and alignment;
- the target statewide prediction domain is explicit;
- background or absence sampling is defensible for that domain;
- folds reflect spatial dependence and the intended transfer distance; and
- preprocessing can be reproduced within folds.

### Gate D: Method promotion

Promote a method from experimental to recommended only if it provides a repeatable advantage in one
or more of:

- predictive discrimination with paired uncertainty;
- performance on scientifically relevant distribution shifts;
- computational scale or memory;
- robustness to bag size and sampling noise; or
- useful representation richness that changes substantive conclusions.

The advantage must be weighed against complexity, tuning burden, interpretability, and failure
modes.

## 8. Recommended Execution Order

1. Preserve M0 and document it as the canonical scientific reference.
2. Correct numerical and kernel-validity risks.
3. Create a common M0-M3 comparison interface and leakage-free validation mechanics.
4. Build the synthetic distribution-regression test suite.
5. Implement truly primal RFF logistic fitting and benchmark approximation error against M0.
6. Complete reproducible serialization and experiment metadata.
7. Recover or acquire representative statewide rasters, polygons, and sampling information.
8. Redesign GIS ingestion and site/background construction around those data.
9. Run paired spatial validation of M0-M3.
10. Test ARD, grouped kernels, shrinkage embeddings, multi-scale spatial representations, and then
    more advanced transport or learned methods only as justified by earlier results.

## 9. Research Basis for Candidate Extensions

The following sources provide the main methodological basis for the roadmap:

- Szabó et al., [Learning Theory for Distribution Regression](https://jmlr.org/beta/papers/v17/14-510.html)
- Nishiyama and Fukumizu, [Characteristic Kernels and Infinitely Divisible Distributions](https://jmlr.org/papers/v17/14-132.html)
- Muandet et al., [Kernel Mean Shrinkage Estimators](https://www.jmlr.org/papers/v17/14-195.html)
- Rudi et al., [FALKON: An Optimal Large Scale Kernel Method](https://papers.nips.cc/paper/6978-falkon-an-optimal-large-scale-kernel-method)
- Yu et al., [Orthogonal Random Features](https://arxiv.org/abs/1610.09072)
- Avron et al., [Quasi-Monte Carlo Feature Maps for Shift-Invariant Kernels](https://jmlr.org/papers/v17/14-538.html)
- Le et al., [Fastfood: Approximate Kernel Expansions in Loglinear Time](https://proceedings.mlr.press/v28/le13.html)
- Feydy et al., [Interpolating between Optimal Transport and MMD using Sinkhorn Divergences](https://proceedings.mlr.press/v89/feydy19a)
- Zaheer et al., [Deep Sets](https://proceedings.nips.cc/paper_files/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf)
- Lee et al., [Set Transformer](https://proceedings.mlr.press/v97/lee19d)
- Roberts et al., [Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure](https://www.biom.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf/at_download/file)
- Valavi et al., [blockCV: spatial and environmental blocking for ecological and cross-validation](https://doi.org/10.1111/2041-210X.13107)
- Warton and Shepherd, [Poisson point process models solve the pseudo-absence problem for presence-only data in ecology](https://arxiv.org/abs/1011.3319)
- Fithian and Hastie, [Finite-sample equivalence in statistical models for presence-only data](https://arxiv.org/abs/1207.6950)

## 10. Standing Methodological Position

Until stronger data and evaluation are available:

- do not remove an implemented method solely because it failed to outperform on the current data;
- use M0 as the canonical, interpretable reference;
- use M1 as the preferred scalable approximation once approximation fidelity is demonstrated;
- treat M2 as an added-capacity experiment;
- treat M3 as an alternative distribution geometry, with order 2 as the supported default;
- describe current presence/background outputs as relative suitability;
- avoid data-format-specific structural changes that cannot be validated against likely raster and
  polygon inputs; and
- distinguish speed approximations from changes to the scientific model in code, experiments, and
  documentation.

This position should be revisited after the synthetic test suite and again after representative GIS
data support a spatially defensible model comparison.
