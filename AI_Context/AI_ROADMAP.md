# AI Roadmap for KLRfome — Learned Representations for Distribution-Based Landscape Modelling

> **Historical strategy note:** This document records exploratory directions. The authoritative
> current method status, corrections, and staged decisions are in
> `METHODS_ROADMAP_2026-07-13.md`. In particular, an RBF kernel mean embedding is a
> full-distribution representation in principle, and current RFF dual fitting still constructs a
> dense bag-level Gram matrix.

> **Status:** strategy / design document (not an implementation plan with deadlines).
> **Scope:** how to evolve KLRfome from a fixed-kernel distribution classifier toward
> "AI for ecological/archaeological modelling" without throwing away what already works.
> **Audience:** project maintainer; assumes familiarity with the current kernel + KLR
> pipeline (`klrfome/kernels`, `klrfome/models/klr.py`, `klrfome/prediction/focal.py`).

---

## 0. TL;DR

KLRfome is **already** a shallow version of "AI for ecological modelling": it learns a
decision boundary in a reproducing-kernel feature space. The frontier is not a different
architecture — it is **making the representation learned instead of fixed**, one piece at
a time, while keeping the honest presence-only evaluation harness you already built.

The single highest-leverage, lowest-regret move is:

> **Swap the representation (raw covariates → pretrained geospatial embeddings or a learned
> set encoder), keep the kernel + KLR head, keep the evaluation.**

Everything below is an elaboration of that, with concrete entry points into the existing
codebase.

---

## 1. The problem, stated abstractly

Every method in this space — yours included — factors into three pieces:

| Piece | What it does | KLRfome today |
|------|---------------|----------------|
| **1. Representation** | location/neighborhood → vector or object | bag of z-scored covariate cells (`SampleCollection`) |
| **2. Comparison** | similarity between two such objects | kernel mean embedding (`MeanEmbeddingKernel`) or sliced-Wasserstein (`WassersteinKernel`) |
| **3. Decision** | similarity-to-known-sites → suitability score | Kernel Logistic Regression (`KernelLogisticRegression`) |

The "large learned multi-D space that understands landscape variation and scale" is what
you get when **piece 1 becomes learned** rather than specified by hand and compared with a
fixed RBF. That is the entire conceptual move. KLRfome sits at the *fixed representation +
fixed kernel* end of the AI spectrum; the rest of this document walks toward the *learned*
end.

### 1.1 The name for this problem

Formally, KLRfome solves **distribution regression / distribution classification**
(Szabó, Sriperumbudur, Póczos, Gretton — *"Learning Theory for Distribution Regression"*,
JMLR 2016). The input is not a point but a *probability distribution* — the bag of cells in
a focal window — and the label applies to the whole distribution. This subfield has its own
generalization theory (two-stage sampling: finitely many bags, finitely many samples per
bag), and it is the correct theoretical anchor. Most "AI for species distribution modelling"
work silently assumes **point** inputs and therefore misses the distributional structure that
KLRfome is built around. That distributional framing is a genuine asset, not an accident.

---

## 2. The spectrum of learned representation

Ordered from closest-to-current to most-deeply-learned. Each rung is independently adoptable.

### Rung 1 — Metric / kernel learning (minimal step)
Keep the KLR head; learn the feature-space geometry instead of fixing it.

- **ARD length-scales.** One RBF length-scale per covariate instead of a single global
  `sigma`. With ~22 z-scored covariates of heterogeneous informativeness, a global bandwidth
  is a blunt instrument. RFF implementation: scale input dim `j` by `1/ℓ_j` *before*
  projecting. Cheap, interpretable (length-scales rank feature relevance), often a real AUC
  gain. This is the "baby" version of deep kernel learning.
- **Deep Kernel Learning** (Wilson et al. 2016). Put a small MLP `φ_θ` in front of the RBF:
  `k(x, y) = RBF(φ_θ(x), φ_θ(y))`, train `θ` and kernel params jointly. Still a kernel
  machine; still PSD; still drops into the existing `build_similarity_matrix` contract.

**Risk:** low. **Data demand:** low. **Interpretability:** high (ARD) to medium (DKL).

### Rung 2 — Learned set / distribution encoders (the natural KLRfome upgrade)
Your mean embedding `μ = mean(φ(cells))` is a *fixed* permutation-invariant pooling.
Replace it with a *learned* permutation-invariant encoder `bag of cells → vector`:

- **Deep Sets** (Zaheer et al. 2017): `ρ( Σ_i φ(x_i) )` — learnable per-cell map `φ` and
  post-pool map `ρ`. The mean embedding is the special case `ρ = identity`, `φ` = RFF.
- **Set Transformer** (Lee et al. 2019): attention-based pooling that learns *which* cells
  and *which* feature interactions matter within the window — captures multi-modality and
  interactions a mean cannot.

This is the most on-target rung **for KLRfome specifically**, because the project is already
a distribution-comparison engine. You are swapping a hand-chosen pooling for a learned one
while keeping the rest of the skeleton.

**Risk:** medium. **Data demand:** medium (needs enough bags to train the encoder, or
pretrain/borrow). **Interpretability:** medium (attention weights are inspectable).

### Rung 3 — Spatial encoders (CNN / ViT over the window)
A representational gap worth naming explicitly:

> The bag-of-cells kernel is **permutation-invariant**, so it discards the *spatial
> arrangement* of cells within a window. A site where "wetland abuts upland" is
> indistinguishable from one where the same cells are scattered.

A CNN or vision-transformer over the window **patch** captures texture, gradients, edges, and
configuration — information the current method provably cannot see. This is the most
defensible place where deep learning adds **new signal** rather than just parameters. It does,
however, change the input object from a *set* to an *image patch*, so it sits slightly outside
the current `SampleCollection` abstraction (see §6).

**Risk:** medium-high. **Data demand:** high if trained from scratch; low if using a
pretrained backbone (Rung 4). **Interpretability:** lower (saliency/Grad-CAM needed).

### Rung 4 — Geospatial foundation models (the direct answer to "learned landscape space")
Pretrained on planet-scale imagery / time-series; emit a per-location embedding that already
encodes vegetation, hydrology, terrain, seasonality, and multi-scale context:

- **Google AlphaEarth / Satellite Embedding** — 64-D learned embedding per 10 m pixel, global.
- **Prithvi** (NASA + IBM), **Clay**, **SatMAE**, **Presto** (pixel time-series), **DOFA**.

You replace the raw covariate columns with these embeddings and feed them straight into the
existing kernel/KLR machinery. This is the most literal realization of "a large learned
multi-D space that understands landscape variation and scale."

**Risk:** low-medium (mostly data engineering: alignment, resolution, CRS). **Data demand:**
low (representation is borrowed). **Interpretability:** low (opaque embeddings; needs post-hoc
attribution).

```
   FIXED ◄──────────────────────────────────────────────► LEARNED
   RBF/RFF      ARD / DKL      Deep Sets /       CNN/ViT      Foundation-model
   (today)                    Set Transformer    patches      embeddings
   Rung 0        Rung 1          Rung 2           Rung 3         Rung 4
```

---

## 3. The pragmatic punchline: frozen embeddings + light head

> **For small-N, presence-only problems — i.e. archaeology — the winning recipe is a frozen
> pretrained representation plus a lightweight classifier, NOT an end-to-end deep net.**

You almost never have enough sites (tens to low hundreds) to train a deep network without
overfitting. But you *can* feed someone else's pretrained embedding into KLRfome's kernel +
KLR. This keeps:

- your **honest evaluation** (spatial CV, presence-only metrics),
- your **presence-only framing** (Boyce, gain, lift, area-budget thresholds),
- your **interpretability story** (permutation importance, calibration),

and upgrades only the representation. **KLRfome is not replaced; it becomes the head on a
foundation-model backbone.** This is the single recommendation to act on first.

---

## 4. The ecological-modelling analog: deep SDM and multi-task learning

The closest live analog to KLRfome is deep **species distribution modelling (SDM)**.

- **SINR — Spatial Implicit Neural Representations** (Cole et al. 2023). One network trained
  on millions of presence-only iNaturalist observations learns an implicit field
  `(lat, lon, environment) → presence`, **jointly across thousands of species**. The "AI" is
  that co-training forces a *shared* representation of habitat: a species with 20 records
  borrows geometry learned from species with 20,000.

**The archaeological translation is the interesting bit:** jointly model many **site types /
periods / cultures** so a rare site type borrows landscape structure from common ones.
Multi-task / shared-representation learning is where AI earns its keep on **sparse labels** —
which is the defining constraint of archaeological predictive modelling. If/when multiple site
classes exist in the data, this is the highest-upside direction.

Your **"multiscalar"** instinct appears here too: modern SDMs use multi-resolution context
(pixel + neighborhood + region). That is your focal-window idea generalized into a scale
pyramid — extract bags at several window sizes and concatenate/encode across scales.

---

## 5. Caveats that get *worse*, not better, with more AI

These are not reasons to avoid AI; they are reasons the existing KLRfome discipline must be
carried forward unchanged.

1. **Presence-only stays presence-only.** A bigger embedding does not manufacture true
   absences. Boyce / gain / lift / PU framing remain mandatory. Deep models are *more* prone
   to learning **sampling bias** (where surveyors actually went) as if it were signal.
2. **Spatial autocorrelation makes evaluation harder.** A 64-D embedding makes train/test
   leakage subtler, not rarer. Spatially-blocked / leave-site-out CV becomes **more**
   important. (This is also the fix for the current broken `cross_validate` default — see
   the review notes / §7.)
3. **Interpretability.** Archaeology usually needs a defensible "why here." Foundation
   embeddings are opaque; budget for post-hoc attribution (permutation importance, SHAP,
   partial dependence) from day one.
4. **Data hunger vs. N.** End-to-end deep nets want thousands of labeled bags. With ~100
   sites, well-regularized kernels frequently **beat** deep nets outright. The value of AI
   here is almost entirely in the **pretrained representation**, not in training depth
   yourself. Resist training big models from scratch on small site counts.

`Rule of thumb:` if a proposed AI upgrade adds trainable parameters faster than it adds
labeled bags, it is probably the wrong upgrade for this problem.

---

## 6. Concrete entry points into the codebase

The current architecture's biggest asset is that representation, comparison, and decision are
**separable**, so AI can be adopted one piece at a time and measured on the same harness.

### 6.1 A `LearnedDistributionKernel` drop-in
Add a fourth kernel beside `MeanEmbeddingKernel` and `WassersteinKernel`, conforming to the
same interface so it slots into `fit` / `predict` / evaluation unchanged:

```python
# klrfome/kernels/learned.py  (sketch)
class LearnedDistributionKernel:
    """Bag -> vector via a learned permutation-invariant encoder (Deep Sets /
    Set Transformer), then a linear or RBF kernel on the encoded vectors."""

    def encode(self, samples):            # (m, d) -> (E,)
        ...                               # learned pooling

    def build_similarity_matrix(self, collections, **kw):   # (N, N), same contract
        Z = jnp.stack([self.encode(c.samples) for c in collections])
        return Z @ Z.T                    # or exp(-||z_i - z_j||^2 / 2σ^2)

    def compute_new_to_training(self, new_samples, training_collections):
        ...                               # matches WassersteinKernel signature
```

Required contract (already implied by `api.py` / `focal.py`):
`build_similarity_matrix(collections) -> (N, N)` and a new-vs-train path used by the focal
predictor. With this, mean-embedding / Wasserstein / learned become a clean **fixed-vs-learned
ablation on identical evaluation**.

### 6.2 Foundation-model embeddings as covariates
No new kernel needed. Replace the per-cell covariate vector with a pretrained embedding at
ingest time (`klrfome/data/tabular.py::bags_from_dataframe`, or the raster ingest in
`klrfome/io`). Everything downstream — `median_sigma`, RFF, KLR, focal predict — works
unchanged because it is agnostic to what the `d` covariate columns *mean*.

### 6.3 Spatial encoder path
A CNN/ViT over window patches breaks the set abstraction: the input becomes
`(window_size, window_size, n_bands)` instead of `(m, d)`. Cleanest integration is to treat
the encoder output as the bag's encoded vector and reuse the `LearnedDistributionKernel`
contract (§6.1), bypassing per-cell pooling.

### 6.4 Multi-task head
KLR is single-task. Multi-task would generalize the decision layer to shared-representation +
per-task heads. This is the largest architectural change and should wait until (a) multiple
site classes exist and (b) a learned representation (Rung 2+) is already in place.

---

## 7. Suggested sequencing (no dates — gated by results, not calendar)

Each step is independently shippable and individually measurable on the existing
presence-only / spatial-CV harness. Stop whenever the marginal AUC/Boyce gain stops paying
for the added complexity and lost interpretability.

1. **Foundation: honest evaluation first.** Land spatially-blocked / leave-site-out CV (this
   also replaces the broken `stratified` `cross_validate` default). *No AI is trustworthy
   until evaluation is.* This gates everything else.
2. **ARD length-scales** on the RFF path (Rung 1). Cheap, interpretable, likely a real gain.
3. **Foundation-model embeddings as covariates** (Rung 4 representation, §6.2). Highest
   leverage per unit effort; reuses the entire existing stack.
4. **`LearnedDistributionKernel`** with a small Deep Sets / Set Transformer encoder
   (Rung 2, §6.1). The clean fixed-vs-learned ablation.
5. **Spatial encoder** (Rung 3) only if §4 shows that within-window *configuration* carries
   signal the set encoder is missing.
6. **Multi-task / shared representation** (deep-SDM analog, §4) if and when multiple site
   classes exist — the sparse-label payoff is largest here.

---

## 8. Why KLRfome is well-positioned

Most of the field fused representation and decision into one black box and therefore cannot
cleanly measure where their gains come from. KLRfome kept the three pieces separable and
built an honest presence-only evaluation around them. That means you can:

- adopt "AI" **incrementally** (one rung at a time),
- **ablate** fixed vs learned on identical metrics,
- preserve the presence-only / spatial-autocorrelation rigor that most deep-SDM work skips.

The distributional framing (§1.1) is ahead of where most point-based deep SDM sits. The work
ahead is to upgrade the *representation* up the spectrum in §2 while holding that rigor fixed.

---

## 9. Reading list / anchors

- Szabó, Sriperumbudur, Póczos, Gretton (2016). *Learning Theory for Distribution Regression.* JMLR.
- Zaheer et al. (2017). *Deep Sets.* NeurIPS.
- Lee et al. (2019). *Set Transformer.* ICML.
- Wilson et al. (2016). *Deep Kernel Learning.* AISTATS.
- Rahimi & Recht (2007). *Random Features for Large-Scale Kernel Machines.* NeurIPS. *(current RFF basis)*
- Sutherland & Schneider (2015). *On the Error of Random Fourier Features.* UAI. *(use sin/cos pair, not random phase)*
- Cole et al. (2023). *Spatial Implicit Neural Representations for Global-Scale Species Mapping (SINR).* ICML.
- Hirzel et al. (2006). *Continuous Boyce Index — evaluating habitat suitability models.* *(already implemented)*
- Geospatial foundation models: AlphaEarth / Satellite Embedding, Prithvi, Clay, SatMAE, Presto, DOFA.
