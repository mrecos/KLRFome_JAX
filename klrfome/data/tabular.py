"""Helpers for building KLRfome bags from tabular (already-extracted) site data.

This is the data shape used by the original R ``format_site_data`` workflow and by
the real archaeological CSVs (one row per raster cell, with a ``presence`` flag and
a ``SITENO`` group id, plus covariate columns).

Keeping this logic in the package lets analysis notebooks stay short: a notebook
calls ``bags_from_dataframe`` / ``stratified_bag_split`` / ``scale_bags`` and then
uses the normal kernels + KLR, rather than re-implementing data wrangling inline
(which is what made the previous notebooks balloon).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import jax.numpy as jnp

from .formats import SampleCollection

# Column names that are never covariates even if numeric.
_NON_COVARIATE = {
    "", "unnamed: 0", "presence", "siteno", "site_no", "site", "site_id",
    "x", "y", "lon", "lat", "long", "longitude", "latitude",
    "easting", "northing", "coords_x1", "coords_x2", "fid", "id", "cell", "rowid",
}


def detect_columns(df) -> Tuple[str, str, List[str]]:
    """Return (presence_col, siteno_col, covariate_cols) from a dataframe.

    Covariates are numeric columns that are not the presence/siteno/coordinate ids.
    """
    import pandas as pd

    cols = list(df.columns)
    lower = {c: str(c).lower() for c in cols}
    pcol = next((c for c in cols if lower[c] == "presence"), None)
    scol = next((c for c in cols if lower[c] in ("siteno", "site_no", "site", "site_id")), None)
    if pcol is None or scol is None:
        raise ValueError(f"Need 'presence' and 'SITENO' columns; found {cols[:20]}")
    covars = [
        c for c in cols
        if lower[c] not in _NON_COVARIATE and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not covars:
        raise ValueError("No numeric covariate columns detected.")
    return pcol, scol, covars


def detect_xy(df) -> Tuple[str, str]:
    """Return (x_col, y_col) coordinate columns, or raise if none found."""
    lower = {str(c).lower(): c for c in df.columns}
    for xn, yn in (("x", "y"), ("lon", "lat"), ("longitude", "latitude"),
                   ("easting", "northing"), ("coords_x1", "coords_x2")):
        if xn in lower and yn in lower:
            return lower[xn], lower[yn]
    raise ValueError(f"No x/y coordinate columns found in {list(df.columns)[:20]}")


def bags_from_dataframe(
    df,
    presence_col: Optional[str] = None,
    siteno_col: Optional[str] = None,
    covariates: Optional[Sequence[str]] = None,
    n_background_bags: Optional[int] = None,
    bag_cap: int = 120,
    min_bag: int = 3,
    background: str = "random",
    xy_cols: Optional[Tuple[str, str]] = None,
    bg_bag_size: Optional[int] = None,
    seed: int = 42,
) -> List[SampleCollection]:
    """Build SampleCollections from a tabular site dataframe.

    - Each ``SITENO`` with ``presence == 1`` becomes one site bag (label 1).
    - Background bags (label 0) are built according to ``background``:
        * ``"random"`` (default): ``presence == 0`` rows are randomly partitioned
          into ``n_background_bags`` bags. Fast, but each bag is just the regional
          marginal (no spatial autocorrelation) -- which lets shape-sensitive
          kernels trivially separate "scattered pile" from real (compact) sites,
          inflating AUC. Use only as a baseline.
        * ``"spatial"``: each background bag is a compact **k-nearest-neighbour
          patch** in (x, y) space around a random seed cell, so it carries the same
          within-bag spatial co-correlation a real site does. This removes the
          sampling artifact and is the honest setting. Requires x/y columns.
    - ``n_background_bags`` defaults to the number of site bags (balance = 1).
    - ``bg_bag_size`` is the target cells per background bag; defaults to the median
      site-bag size so the two classes are distributionally comparable. Capped at
      ``bag_cap``.
    - Bags larger than ``bag_cap`` are subsampled; bags smaller than ``min_bag`` dropped.

    Returns a single shuffled list of SampleCollection (sites + background).
    """
    rng = np.random.default_rng(seed)
    if presence_col is None or siteno_col is None or covariates is None:
        presence_col, siteno_col, covariates = detect_columns(df)
    covariates = list(covariates)

    sites_df = df[[presence_col, siteno_col] + covariates].dropna()
    sites_df = sites_df[sites_df[presence_col] == 1]

    def _cap(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        if a.shape[0] > bag_cap:
            a = a[rng.choice(a.shape[0], bag_cap, replace=False)]
        return a

    site_ids = sorted(sites_df[siteno_col].unique())
    collections: List[SampleCollection] = []
    for sid in site_ids:
        a = _cap(sites_df[sites_df[siteno_col] == sid][covariates].to_numpy())
        if a.shape[0] >= min_bag:
            collections.append(SampleCollection(jnp.asarray(a), 1, f"site_{sid}"))

    n_site_bags = len([c for c in collections if c.label == 1])
    n_back = n_background_bags or max(2, n_site_bags)
    site_sizes = [c.n_samples for c in collections if c.label == 1]
    target = bg_bag_size or (int(np.median(site_sizes)) if site_sizes else bag_cap)
    target = int(min(target, bag_cap))

    if background == "spatial":
        xcol, ycol = xy_cols if xy_cols else detect_xy(df)
        bgxy = df[df[presence_col] == 0][[xcol, ycol] + covariates].dropna()
        coords = bgxy[[xcol, ycol]].to_numpy(dtype=np.float64)
        x_bg = bgxy[covariates].to_numpy(dtype=np.float64)
        if coords.shape[0] < max(min_bag, target):
            raise ValueError("Not enough background cells for spatial bagging.")
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        k = int(min(target, coords.shape[0]))
        seeds = rng.choice(coords.shape[0], size=min(n_back, coords.shape[0]), replace=False)
        for s in seeds:
            _, idx = tree.query(coords[s], k=k)
            idx = np.atleast_1d(idx)
            a = x_bg[idx]
            if a.shape[0] >= min_bag:
                collections.append(SampleCollection(jnp.asarray(a), 0, f"background_{int(s)}"))
    else:
        bgv = df[df[presence_col] == 0][covariates].dropna().to_numpy()
        assign = rng.integers(0, n_back, size=bgv.shape[0])
        for g in range(n_back):
            a = _cap(bgv[assign == g])
            if a.shape[0] >= min_bag:
                collections.append(SampleCollection(jnp.asarray(a), 0, f"background_{g}"))

    rng.shuffle(collections)
    return collections


def stratified_bag_split(
    collections: List[SampleCollection],
    test_fraction: float = 0.3,
    seed: int = 42,
) -> Tuple[List[SampleCollection], List[SampleCollection]]:
    """Split bags into train/test, stratified by label."""
    rng = np.random.default_rng(seed)
    pos = [c for c in collections if c.label == 1]
    neg = [c for c in collections if c.label == 0]

    def _take(group):
        idx = np.arange(len(group)); rng.shuffle(idx)
        nt = max(1, int(round(len(group) * test_fraction)))
        test = [group[i] for i in idx[:nt]]
        train = [group[i] for i in idx[nt:]]
        return train, test

    tr_p, te_p = _take(pos)
    tr_n, te_n = _take(neg)
    train = tr_p + tr_n
    test = te_p + te_n
    rng.shuffle(train); rng.shuffle(test)
    return train, test


def scale_bags(
    train: List[SampleCollection],
    test: Optional[List[SampleCollection]] = None,
) -> Tuple[List[SampleCollection], Optional[List[SampleCollection]], np.ndarray, np.ndarray]:
    """Z-score bags using statistics pooled over the TRAIN bags only.

    Returns (train_scaled, test_scaled, mean, std). ``test`` may be None.
    """
    allx = np.concatenate([np.asarray(c.samples) for c in train], axis=0)
    mean = allx.mean(axis=0)
    std = allx.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)

    def _apply(bags):
        return [
            SampleCollection(jnp.asarray((np.asarray(c.samples) - mean) / std), c.label, c.id)
            for c in bags
        ]

    train_s = _apply(train)
    test_s = _apply(test) if test is not None else None
    return train_s, test_s, mean, std


def labels_of(collections: List[SampleCollection]) -> np.ndarray:
    """Label vector for a list of bags."""
    return np.array([c.label for c in collections])


def fit_mean_embedding(
    train: List[SampleCollection],
    sigma: float,
    n_features: int = 256,
    lambda_reg: float = 0.1,
    seed: int = 42,
) -> dict:
    """Fit a mean-embedding (RFF) KLR and return a reusable fitted model dict.

    Returns {'rff', 'Etr' (n_train x D train embeddings), 'alpha', 'sigma'} so the
    same model can score held-out bags AND a prediction surface without refitting.
    """
    from ..kernels.rff import RandomFourierFeatures
    from ..models.klr import KernelLogisticRegression

    rff = RandomFourierFeatures(sigma=sigma, n_features=n_features, seed=seed)
    rff._initialize_weights(int(train[0].samples.shape[1]))
    e_tr = np.stack([
        np.asarray(jnp.mean(rff.feature_map(jnp.asarray(c.samples)), axis=0)) for c in train
    ])
    klr = KernelLogisticRegression(lambda_reg=lambda_reg, tol=0.001)
    fit = klr.fit(jnp.asarray(e_tr @ e_tr.T), jnp.asarray(labels_of(train)))
    return {"rff": rff, "Etr": e_tr, "alpha": np.asarray(fit.alpha), "sigma": float(sigma)}


def mean_embedding_predict(model: dict, bags: List[SampleCollection]) -> np.ndarray:
    """Probabilities for a list of bags under a fitted mean-embedding model."""
    e = np.stack([
        np.asarray(jnp.mean(model["rff"].feature_map(jnp.asarray(c.samples)), axis=0)) for c in bags
    ])
    return 1.0 / (1.0 + np.exp(-((e @ model["Etr"].T) @ model["alpha"])))


def predict_xy_surface(
    model: dict,
    centers_xy: np.ndarray,
    cells_xy: np.ndarray,
    cells_x_scaled: np.ndarray,
    k: int = 16,
    batch_size: int = 5000,
) -> np.ndarray:
    """Score each prediction point by its k-NN neighbourhood patch (focal predict).

    centers_xy: (M, 2) locations to score.
    cells_xy / cells_x_scaled: (C, 2) coords and (C, d) SCALED covariates of the
    cell pool the neighbourhoods are drawn from (use the same mean/std as training).
    Processed in batches so the whole surface (hundreds of thousands of cells) can
    be scored without exhausting memory. Returns (M,) probabilities.

    Note: you can only predict where covariate cells exist. If the cell pool is
    spatially restricted (e.g. background sampled only along rivers), the surface
    is necessarily blank elsewhere -- that is data coverage, not model failure.
    """
    from scipy.spatial import cKDTree

    centers_xy = np.atleast_2d(np.asarray(centers_xy))
    tree = cKDTree(cells_xy)
    proj = model["Etr"].T @ model["alpha"]           # (D,) collapse train side once
    out = np.empty(centers_xy.shape[0])
    for start in range(0, centers_xy.shape[0], batch_size):
        chunk = centers_xy[start:start + batch_size]
        _, idx = tree.query(chunk, k=k)
        idx = np.atleast_2d(idx)
        bags = cells_x_scaled[idx]                   # (b, k, d)
        b, kk, d = bags.shape
        phi = np.asarray(model["rff"].feature_map(jnp.asarray(bags.reshape(b * kk, d))))
        emb = phi.reshape(b, kk, -1).mean(axis=1)    # (b, D)
        out[start:start + b] = 1.0 / (1.0 + np.exp(-(emb @ proj)))
    return out


def site_centroids(df, presence_col=None, siteno_col=None, xy_cols=None):
    """Per-SITENO (x, y) centroid for presence==1 rows. Returns a DataFrame indexed by SITENO."""
    if presence_col is None or siteno_col is None:
        presence_col, siteno_col, _ = detect_columns(df)
    xcol, ycol = xy_cols if xy_cols else detect_xy(df)
    s = df[df[presence_col] == 1].groupby(siteno_col)[[xcol, ycol]].mean()
    s.columns = ["x", "y"]
    return s


def site_ids_in(bags: List[SampleCollection]) -> set:
    """SITENO strings present as site bags (id 'site_<SITENO>') in a bag list."""
    return {c.id.split("site_", 1)[1] for c in bags if c.label == 1 and c.id.startswith("site_")}


def optimal_threshold(y, probs, metric: str = "tss"):
    """Threshold maximizing TSS (Youden's J = sens+spec-1) or F1.

    Returns (threshold, score, (TP, FP, TN, FN)).
    """
    y = np.asarray(y).astype(int)
    probs = np.asarray(probs)
    cands = np.unique(np.concatenate([[0.0, 1.0], probs]))
    best = (0.5, -np.inf, (0, 0, 0, 0))
    for t in cands:
        pred = (probs >= t).astype(int)
        TP = int(((pred == 1) & (y == 1)).sum()); FP = int(((pred == 1) & (y == 0)).sum())
        TN = int(((pred == 0) & (y == 0)).sum()); FN = int(((pred == 0) & (y == 1)).sum())
        sens = TP / (TP + FN) if (TP + FN) else 0.0
        spec = TN / (TN + FP) if (TN + FP) else 0.0
        if metric == "f1":
            score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
        else:
            score = sens + spec - 1.0
        if score > best[1]:
            best = (float(t), float(score), (TP, FP, TN, FN))
    return best


# ---------------------------------------------------------------------------
# Spatial-domain restriction.
#
# IMPORTANT: the model can only validly score locations that fall inside the
# region the BACKGROUND was sampled from. If background covers only part of the
# study area, sites outside that footprint are extrapolation and the prediction
# surface is undefined there. We make that restriction explicit (and loud) rather
# than silently scoring out-of-domain sites. Revisit when full-area background
# becomes available.
# ---------------------------------------------------------------------------
def background_bbox(df, presence_col=None, xy_cols=None, margin: float = 0.0):
    """Bounding box (xmin, xmax, ymin, ymax) of the background (presence==0) cells."""
    if presence_col is None:
        presence_col, _, _ = detect_columns(df)
    xcol, ycol = xy_cols if xy_cols else detect_xy(df)
    bg = df[df[presence_col] == 0]
    if len(bg) == 0:
        raise ValueError("No background (presence==0) rows to define a domain.")
    xmin, xmax = float(bg[xcol].min()), float(bg[xcol].max())
    ymin, ymax = float(bg[ycol].min()), float(bg[ycol].max())
    if margin:
        dx, dy = (xmax - xmin) * margin, (ymax - ymin) * margin
        xmin, xmax, ymin, ymax = xmin - dx, xmax + dx, ymin - dy, ymax + dy
    return (xmin, xmax, ymin, ymax)


def restrict_to_background_domain(df, presence_col=None, siteno_col=None, xy_cols=None, margin: float = 0.0):
    """Clip the dataframe to the background bounding box. Returns (df_clip, info).

    info reports the bbox and how many site cells / site locations were dropped, so
    the restriction can be surfaced loudly in a notebook (never a silent bias).
    """
    if presence_col is None or siteno_col is None:
        presence_col, siteno_col, _ = detect_columns(df)
    xcol, ycol = xy_cols if xy_cols else detect_xy(df)
    bbox = background_bbox(df, presence_col, (xcol, ycol), margin=margin)
    xmin, xmax, ymin, ymax = bbox
    inside = (df[xcol] >= xmin) & (df[xcol] <= xmax) & (df[ycol] >= ymin) & (df[ycol] <= ymax)
    clip = df[inside].copy()

    sites_before = df[df[presence_col] == 1]
    sites_after = clip[clip[presence_col] == 1]
    info = {
        "bbox": bbox,
        "n_site_before": int(len(sites_before)),
        "n_site_after": int(len(sites_after)),
        "n_site_dropped": int(len(sites_before) - len(sites_after)),
        "n_siteno_before": int(sites_before[siteno_col].nunique()),
        "n_siteno_after": int(sites_after[siteno_col].nunique()),
        "n_background": int((clip[presence_col] == 0).sum()),
    }
    return clip, info


# ---------------------------------------------------------------------------
# Diagnostics: capture/gain, calibration, permutation importance.
# ---------------------------------------------------------------------------
def capture_curve(probs, is_site):
    """Cumulative-capture (gain) curve: fraction of sites captured vs fraction of area.

    Cells are ranked by predicted probability (descending). Returns
    (area_fraction, captured_fraction), both length-N arrays.
    """
    probs = np.asarray(probs)
    is_site = np.asarray(is_site).astype(int)
    order = np.argsort(-probs)
    s = is_site[order]
    total = max(1, int(s.sum()))
    captured = np.cumsum(s) / total
    area = np.arange(1, len(s) + 1) / len(s)
    return area, captured


def reliability_curve(probs, y, n_bins: int = 10):
    """Binned calibration: returns (bin_center, mean_pred, observed_freq, count) per non-empty bin."""
    probs = np.asarray(probs)
    y = np.asarray(y).astype(int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    bc, mp, of, ct = [], [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            bc.append((edges[b] + edges[b + 1]) / 2)
            mp.append(float(probs[m].mean()))
            of.append(float(y[m].mean()))
            ct.append(int(m.sum()))
    return np.array(bc), np.array(mp), np.array(of), np.array(ct)


def permutation_importance(model: dict, test_bags, feature_names, n_repeats: int = 5, seed: int = 42):
    """AUC drop when each covariate is shuffled across the held-out cells.

    Returns (baseline_auc, importances) where importances[j] aligns with feature_names[j].
    """
    from ..utils.validation import compute_roc_auc

    rng = np.random.default_rng(seed)
    y = labels_of(test_bags)
    base = compute_roc_auc(mean_embedding_predict(model, test_bags), y)
    pool = np.concatenate([np.asarray(b.samples) for b in test_bags], axis=0)
    d = pool.shape[1]
    imp = np.zeros(d)
    for j in range(d):
        drops = []
        for _ in range(n_repeats):
            permed = []
            for b in test_bags:
                s = np.asarray(b.samples).copy()
                s[:, j] = rng.choice(pool[:, j], size=s.shape[0], replace=True)
                permed.append(SampleCollection(jnp.asarray(s), b.label, b.id))
            drops.append(base - compute_roc_auc(mean_embedding_predict(model, permed), y))
        imp[j] = float(np.mean(drops))
    return float(base), imp


# ---------------------------------------------------------------------------
# Prospection-appropriate (presence-only) evaluation.
#
# This is presence/background (positive-unlabeled), not binary classification:
# "background" is unlabeled, not true absence, and the sampled prevalence is not
# the landscape base rate. So specificity / precision / accuracy / Youden's J are
# biased and should NOT drive the operating point. Instead choose the threshold by
# AREA budget and judge the model by how efficiently it concentrates sites (gain /
# lift) and by the presence-only Continuous Boyce Index.
# ---------------------------------------------------------------------------
def threshold_for_area(landscape_probs, area_fraction: float) -> float:
    """Probability cutoff that flags the top ``area_fraction`` of the landscape."""
    p = np.asarray(landscape_probs)
    a = float(np.clip(area_fraction, 1e-9, 1.0))
    return float(np.quantile(p, 1.0 - a))


def capture_gain_table(landscape_probs, target_probs,
                       area_fractions=(0.05, 0.10, 0.20, 0.30)):
    """Operating points by AREA budget (the prospection-appropriate view).

    landscape_probs : predicted P over all landscape cells (defines the cutoff).
    target_probs    : predicted P at the target sites (ideally HELD-OUT) we want to
                      capture; capture = sensitivity at each area budget.

    Returns a list of dicts with: area, threshold, capture, gain, lift.
      gain = Kvamme's gain = 1 - area/capture   (0 = random, ->1 = efficient)
      lift = capture/area                        (enrichment over random)
    """
    land = np.asarray(landscape_probs)
    tgt = np.asarray(target_probs)
    rows = []
    for a in area_fractions:
        thr = threshold_for_area(land, a)
        capture = float((tgt >= thr).mean()) if len(tgt) else float("nan")
        gain = (1.0 - a / capture) if capture > 0 else float("nan")
        lift = (capture / a) if a > 0 else float("nan")
        rows.append({"area": float(a), "threshold": thr,
                     "capture": capture, "gain": gain, "lift": lift})
    return rows


def continuous_boyce_index(presence_probs, background_probs, n_windows: int = 20, window: float = 0.1):
    """Continuous Boyce Index (Hirzel et al. 2006) — presence-only model evaluation.

    Needs only presences and the background (available) distribution; no true
    absences. Moving windows across the suitability range; per window compute the
    predicted-to-expected ratio F = (presence fraction)/(area fraction); CBI is the
    Spearman correlation between F and window suitability.

    Range [-1, 1]: >0 good (sites concentrate at high suitability), ~0 random,
    <0 counter-predictive. Returns (cbi, window_midpoints, F_values).
    """
    from scipy.stats import spearmanr

    pres = np.asarray(presence_probs)
    bg = np.asarray(background_probs)
    lo = float(min(pres.min(), bg.min()))
    hi = float(max(pres.max(), bg.max()))
    if hi <= lo:
        return float("nan"), np.array([]), np.array([])
    w = window * (hi - lo)
    starts = np.linspace(lo, hi - w, n_windows)
    mids, F = [], []
    for s in starts:
        e = s + w
        Pi = float(((pres >= s) & (pres < e)).mean())
        Ei = float(((bg >= s) & (bg < e)).mean())
        if Ei > 0:
            mids.append((s + e) / 2.0)
            F.append(Pi / Ei)
    if len(F) < 3:
        return float("nan"), np.array(mids), np.array(F)
    cbi = float(spearmanr(mids, F).correlation)
    return cbi, np.array(mids), np.array(F)


def median_sigma(
    collections: List[SampleCollection],
    n_cells: int = 1000,
    seed: int = 42,
) -> float:
    """Median pairwise Euclidean distance between sampled cells.

    This is the standard RBF bandwidth heuristic and the single most important
    knob for KLRfome on real data: the default ``sigma=0.5`` is tuned for ~2-3
    features and *saturates to zero* once you have many covariates (for d z-scored
    features a typical squared distance is ~2d, so ``exp(-2d / 2 sigma^2)`` underflows).
    Setting ``sigma`` near this median keeps the kernel in its informative range.
    """
    rng = np.random.default_rng(seed)
    cells = np.concatenate([np.asarray(c.samples) for c in collections], axis=0)
    k = min(n_cells, cells.shape[0])
    sample = cells[rng.choice(cells.shape[0], k, replace=False)]
    # pairwise distances on the subsample (upper triangle)
    diff = sample[:, None, :] - sample[None, :, :]
    d = np.sqrt(np.maximum((diff ** 2).sum(-1), 0))
    iu = np.triu_indices(k, k=1)
    return float(np.median(d[iu]))


def mean_embedding_heldout(
    train: List[SampleCollection],
    test: List[SampleCollection],
    sigma: float,
    n_features: int = 256,
    lambda_reg: float = 0.1,
    seed: int = 42,
):
    """Fit a mean-embedding (RFF) KLR on train bags, score held-out test bags.

    Returns (auc, test_probabilities, test_labels). Vectorized and fast: each bag
    is reduced to a single mean RFF embedding, so the kernel is E @ E.T.
    """
    from ..kernels.rff import RandomFourierFeatures
    from ..models.klr import KernelLogisticRegression
    from ..utils.validation import compute_roc_auc

    rff = RandomFourierFeatures(sigma=sigma, n_features=n_features, seed=seed)
    rff._initialize_weights(int(train[0].samples.shape[1]))

    def embed(bags):
        return np.stack([
            np.asarray(jnp.mean(rff.feature_map(jnp.asarray(c.samples)), axis=0))
            for c in bags
        ])

    e_tr, e_te = embed(train), embed(test)
    y_tr = labels_of(train); y_te = labels_of(test)
    klr = KernelLogisticRegression(lambda_reg=lambda_reg, tol=0.001)
    fit = klr.fit(jnp.asarray(e_tr @ e_tr.T), jnp.asarray(y_tr))
    alpha = np.asarray(fit.alpha)
    probs = 1.0 / (1.0 + np.exp(-((e_te @ e_tr.T) @ alpha)))
    return compute_roc_auc(probs, y_te), probs, y_te


def wasserstein_heldout(
    train: List[SampleCollection],
    test: List[SampleCollection],
    sigma: Optional[float] = None,
    n_projections: int = 100,
    p: int = 2,
    n_quantiles: int = 128,
    lambda_reg: float = 0.1,
    seed: int = 42,
):
    """Fit a Sliced-Wasserstein KLR on train bags, score held-out test bags.

    Uses the single-global-Q quantile representation (``n_quantiles``): every bag
    is summarized by Q sorted-quantile points per projection, so the train matrix
    and the test cross-matrix are each a single uniform-shape, JIT-compiled op
    (no per-pair recompilation). This is the fast, recommended path.

    If ``sigma`` is None it is calibrated to the median pairwise SW distance.
    Returns (auc, test_probabilities, test_labels, sigma_used).
    """
    from ..kernels.wasserstein import SlicedWassersteinDistance, estimate_sigma_from_distances
    from ..models.klr import KernelLogisticRegression
    from ..utils.validation import compute_roc_auc

    sw = SlicedWassersteinDistance(n_projections=n_projections, p=p, seed=seed)
    y_tr = labels_of(train); y_te = labels_of(test)

    d_tr = np.asarray(sw.pairwise_distances_quantile(train, n_quantiles))  # (N, N)
    if sigma is None:
        sigma = float(estimate_sigma_from_distances(jnp.asarray(d_tr), 50.0))

    k_tr = np.exp(-d_tr ** 2 / (2.0 * sigma ** 2))
    klr = KernelLogisticRegression(lambda_reg=lambda_reg, tol=0.001)
    fit = klr.fit(jnp.asarray(k_tr), jnp.asarray(y_tr))
    alpha = np.asarray(fit.alpha)

    d_te = np.asarray(sw.cross_distances_quantile(test, train, n_quantiles))  # (Ntest, Ntrain)
    k_te = np.exp(-d_te ** 2 / (2.0 * sigma ** 2))
    probs = 1.0 / (1.0 + np.exp(-(k_te @ alpha)))
    return compute_roc_auc(probs, y_te), probs, y_te, sigma
