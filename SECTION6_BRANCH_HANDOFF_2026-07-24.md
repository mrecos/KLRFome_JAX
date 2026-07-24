# Section 6 Evaluation Branch Handoff

**Date:** 2026-07-24  
**Branch:** `codex/section6-evaluation-revamp`  
**Pull request:** Draft PR #10  
**Status:** Paused after the completed Section 6 geometry evaluation

## Purpose of this branch

This branch reorganizes Section 6 validation around the actual
presence–background study design. It treats model output as relative suitability,
uses mapped raster availability to define operating points, and makes geometry,
support, and spatial failure diagnostics explicit.

The branch does not rank methods conclusively, pool physio-sheds, or extend the
analysis to 9 × 9 and 11 × 11 focal supports.

## Implemented work

- Reframed the primary operating-point metrics as:
  - held-out site capture at 5%, 10%, and 20% of mapped availability;
  - lift relative to random allocation; and
  - capture surplus, defined as capture minus achieved mapped-area fraction.
- Retained Kvamme Gain as a secondary archaeological crosswalk. It is explicitly
  documented as a monotonic transformation of lift rather than independent
  model-selection evidence.
- Kept ROC AUC and PR AUC secondary because constructed background is not
  confirmed absence.
- Added a `geometry` evaluation mode containing:
  - the primary common 7 × 7 focal design;
  - a full-window sensitivity requiring all 49 cells for sites, backgrounds, and
    mapped availability; and
  - a sensitivity that exactly matches the background valid-cell-count
    distribution to retained site bags.
- Added maps and audit fields for valid focal-window cell count and anchor
  distance to the all-band raster-mask boundary.
- Added exploratory global and local Moran diagnostics for:
  - held-out site percentile shortfall; and
  - spatial disagreement between each method and M0.
- Used eight-neighbor spatial weights, 999 permutations, and
  Benjamini–Hochberg adjustment for local diagnostics.
- Regenerated the Section 6 notebook as the reporting surface for the runner.
- Updated the README, benchmark documentation, result schema, configuration, and
  tests.

## Latest Section 6 run

The completed run used `geometry` mode, five spatial grouped folds, two repeats,
and all M0–M3, LR, RF, and geometry-control methods. Raw results remain ignored
under:

`site_data/r91_section_6_data/section6_evaluation/results.json`

The raw JSON is approximately 231 MB because it retains fold predictions and
local spatial diagnostics.

### Geometry audit

| Setting | Primary sites | Full-window sites | Count-matched sites | Full-window availability |
|---|---:|---:|---:|---:|
| Riverine | 142 | 70 | 141 | 619 |
| Upland | 198 | 129 | 198 | 925 |

The full-window design removes many sites and therefore changes the evaluated
site population. Its performance differences are not a clean estimate of the
effect of geometry alone.

### Riverine result

- The primary geometry-only control was weak: mean ROC AUC 0.551 and 10%
  capture surplus 0.068.
- Exact cell-count matching reduced it to approximately random:
  ROC AUC 0.495 and capture surplus 0.017.
- Under cell-count matching, the substantive methods retained modest positive
  10% capture surplus:
  - M0: 0.090
  - M1: 0.094
  - M2: 0.080
  - M3: 0.076
  - RF: 0.105
- The riverine signal therefore appears modest but is not explained primarily
  by bag cell count.

### Upland result

- The primary geometry-only control was clearly informative:
  ROC AUC 0.641 and 10% capture surplus 0.205.
- Exact cell-count matching reduced, but did not eliminate, the geometry signal:
  ROC AUC 0.610 and capture surplus 0.085. Footprint diameter or boundary
  proximity therefore remains class-associated.
- Count matching left substantive performance essentially unchanged:
  - M0: capture surplus 0.318
  - M1: 0.321
  - M2: 0.318
  - M3: 0.311
  - RF: 0.391
- The environmental prediction signal persists after controlling cell count,
  but part of the upland sampling structure remains associated with the raster
  mask or upland–riverine interface.
- The full-window geometry control was near random, but that design retained
  only 129 of 198 sites and thus evaluates a more interior subset.

## Spatial diagnostic findings

- Held-out site shortfall had positive global spatial autocorrelation for the
  substantive primary models in both settings, generally Moran's I around
  0.09–0.15.
- The primary designs did not yield FDR-significant local failure clusters,
  suggesting diffuse regional structure rather than a few isolated hotspots.
- The count-matched riverine design did identify a small number of local
  high–high failure clusters. Sites `36BU0105` and `36LH0038` recurred across
  more than one method/repeat and are useful inspection targets.
- Method disagreement was strongly spatially structured. RF–M0 disagreement
  had mean Moran's I of approximately 0.253 in riverine and 0.352 in upland.
  These zones should be examined as coherent differences in model behavior,
  not dismissed as numerical noise.

## Interpretation

The geometry investigation does not invalidate M0–M3 or RF. Exact cell-count
matching preserves their substantive performance while weakening the negative
geometry control, especially in riverine.

The unresolved issue is narrower: the upland mask boundary or footprint shape
still contains class information. Some of that information may be
archaeologically meaningful proximity to the upland–riverine interface rather
than a nuisance. It should be represented explicitly if meaningful, not enter
implicitly through missing focal cells or bag diameter.

No method should be promoted or removed from this single physio-shed.

## Restart point and next priorities

1. Separate distance to the meaningful upland–riverine interface from distance
   to the outer physio-shed boundary and arbitrary nodata edges.
2. Add a targeted control or matching design for distance-to-boundary and
   footprint diameter, not only valid-cell count.
3. Inspect recurrent held-out failure sites and RF–M0 disagreement zones against
   environmental features and data quality.
4. Compact or compress the result contract before applying these diagnostics to
   additional settings.
5. Only after resolving the boundary question, decide whether the broader
   9 × 9 and 11 × 11 focal-support run is worthwhile.
6. Keep pooling or partial pooling across physio-sheds deferred and explicit.

## Reproduction

Run the current geometry evaluation:

```bash
python benchmarks/run_section6_evaluation.py --mode geometry
```

Regenerate the reporting notebook:

```bash
python benchmarks/build_section6_evaluation_notebook.py
```

Open and run:

`notebooks/05_section6_model_validation.ipynb`

The last full tracked test run passed 152 tests. Focused Section 6, ingestion,
Black, Ruff, Mypy, and notebook-compilation checks also passed.
