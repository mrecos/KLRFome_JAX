#!/usr/bin/env python3
"""Build the streamlined Section 6 evaluation notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/05_section6_model_validation.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    markdown(
        """
# Section 6 presence–background evaluation

This notebook is the reporting surface for one reproducible evaluation runner. The primary design
uses the same **7 × 7 focal support** for sites, training backgrounds, and mapped availability.
Every held-out bag is ranked against the fixed raster-availability sample scored by its own fitted
fold, and folds are pooled exactly once within each repeat.

| Priority | Evidence | Interpretation |
|---|---|---|
| Primary | capture / lift / capture surplus at 5%, 10%, and 20% of mapped availability | How efficiently high-suitability area captures held-out sites |
| Primary | Continuous Boyce and held-out site availability percentiles | Whether sites concentrate toward higher mapped suitability |
| Secondary | ROC AUC and PR AUC | Discrimination against constructed background, not true absence |
| Secondary | Kvamme Gain | Archaeological comparability; mathematically redundant with lift |
| Diagnostic | geometry controls, Moran/LISA, timing, memory | Detect support, spatial, and computational pathologies |

Scores are relative suitability. They are not occurrence probabilities. No method is promoted or
removed from one physio-shed.
"""
    ),
    markdown(
        """
## 1. Configuration and execution

The first run executes the geometry investigation if no current result exists. Set
`RERUN_EVALUATION = True` to replace it. Geometry mode compares the primary 7 × 7 design with an
all-49-cell design and an exactly matched site/background cell-count design. Keep full mode deferred
until these controls are interpretable; full mode additionally adds 9 × 9, 11 × 11, and the
original-irregular support sensitivity.
"""
    ),
    code(
        """
from pathlib import Path
import json
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.metrics import roc_curve

from klrfome.utils.reproducibility import configuration_fingerprint


def repository_root(start=Path.cwd()):
    for candidate in [start, *start.parents]:
        if (candidate / 'pyproject.toml').exists() and (candidate / 'klrfome').exists():
            return candidate
    raise RuntimeError('Run this notebook from inside the KLRFome_JAX repository.')


ROOT = repository_root()
CONFIG_PATH = ROOT / 'benchmarks/section6_evaluation_config.json'
CONFIG = json.loads(CONFIG_PATH.read_text())
RESULT_PATH = ROOT / CONFIG['output']

EVALUATION_MODE = 'geometry'  # 'primary', 'geometry', or 'full'
RERUN_EVALUATION = False
SELECTED_METHODS = None  # e.g. ['M0', 'M1', 'LR-mean', 'RF-mean'] for a bounded run

result_is_current = False
if RESULT_PATH.exists():
    existing = json.loads(RESULT_PATH.read_text())
    result_is_current = (
        existing.get('configuration_sha256') == configuration_fingerprint(CONFIG)
        and existing.get('mode') == EVALUATION_MODE
    )

if RERUN_EVALUATION or not result_is_current:
    command = [
        sys.executable,
        str(ROOT / 'benchmarks/run_section6_evaluation.py'),
        '--config',
        str(CONFIG_PATH),
        '--mode',
        EVALUATION_MODE,
    ]
    if SELECTED_METHODS:
        command.extend(['--methods', *SELECTED_METHODS])
    print('Running:', ' '.join(command))
    subprocess.run(command, cwd=ROOT, check=True)
else:
    print(f'Using existing result: {RESULT_PATH}')
"""
    ),
    code(
        """
RESULT = json.loads(RESULT_PATH.read_text())
assert RESULT['schema_version'] == '2.0'
assert RESULT['configuration_sha256']
assert RESULT['settings']

print('Mode:', RESULT['mode'])
print('Settings:', ', '.join(RESULT['settings']))
print('Primary support:', f"{CONFIG['primary_window']} × {CONFIG['primary_window']}")
print('Availability anchors per setting:', CONFIG['availability_sample_size'])
print('Fold design:', f"{CONFIG['n_splits']} folds × {CONFIG['n_repeats']} repeats")
display(pd.DataFrame([
    {'priority': priority, 'metrics': '; '.join(metrics)}
    for priority, metrics in RESULT['metric_hierarchy'].items()
]))
"""
    ),
    markdown(
        """
## 2. Data and support audit

Interpret metrics only after confirming that both settings have retained focal sites, matched focal
backgrounds, a fixed availability sample, and spatial grouped folds. Exclusions remain explicit.
"""
    ),
    code(
        """
audit_rows = []
for setting, payload in RESULT['settings'].items():
    audit = payload['data_audit']
    common = audit['common_focal']
    available = audit['availability']
    controls = audit['geometry_sensitivities']
    full = controls[f"focal_{CONFIG['primary_window']}_full_window"]
    matched = controls[f"focal_{CONFIG['primary_window']}_matched_counts"]
    audit_rows.append({
        'setting': setting,
        'retained_sites': common['n_site_bags'],
        'training_backgrounds': common['n_background_bags'],
        'excluded_sites': len(common['excluded_site_ids']),
        'availability_anchors': available['n_retained'],
        'availability_rejected': available['n_rejected'],
        'full_window_sites': full['n_site_bags'],
        'full_window_backgrounds': full['n_background_bags'],
        'full_window_availability': full['n_availability_bags'],
        'count_matched_sites': matched['n_site_bags'],
        'count_matched_backgrounds': matched['n_background_bags'],
        'spatial_block_width': audit['shared_block_width'],
        'support_windows': ', '.join(map(str, common['window_sizes'])),
    })
audit_table = pd.DataFrame(audit_rows)
display(audit_table.style.format({'spatial_block_width': '{:,.1f}'}))

for setting, payload in RESULT['settings'].items():
    for design in payload['designs'].values():
        assignments = design['fold_plan']['assignments']
        assert all(not set(row['train_ids']) & set(row['test_ids']) for row in assignments)
        assert all(not set(row['train_groups']) & set(row['test_groups']) for row in assignments)
print('Fold leakage checks: PASS')
"""
    ),
    markdown(
        """
## 3. One tidy result table

All later sections use the same pooled out-of-fold rows. Each bag occurs once per repeat; no plot
recomputes a different fold plan or metric definition.
"""
    ),
    code(
        """
PRIMARY_DESIGN = f"focal_{CONFIG['primary_window']}"


def pooled_frame(result):
    rows = []
    for setting, setting_payload in result['settings'].items():
        for design, design_payload in setting_payload['designs'].items():
            for row in design_payload.get('pooled_repeat_results', []):
                rows.append({'setting': setting, 'design': design, **row})
    return pd.DataFrame(rows)


POOLED = pooled_frame(RESULT)
PRIMARY = POOLED[POOLED['design'] == PRIMARY_DESIGN].copy()
if PRIMARY.empty:
    raise RuntimeError(f'No pooled results found for {PRIMARY_DESIGN}.')
settings = list(RESULT['settings'])
methods = [method for method in PRIMARY['method'].unique() if method != 'NEG-geometry']

coverage = PRIMARY.groupby(['setting', 'method', 'repeat'], as_index=False).agg(
    rows=('n_observations', 'size'),
    observations=('n_observations', 'first'),
)
assert (coverage['rows'] == 1).all()
display(coverage.head())
"""
    ),
    markdown(
        """
## 4. Geometry gate

`NEG-geometry` sees only bag cell count and diameter. In the common focal design it should lose the
class-support shortcut. Strong geometry performance is a sampling-design warning, not a useful
archaeological model.
"""
    ),
    code(
        """
geometry = PRIMARY[PRIMARY['method'] == 'NEG-geometry']
geometry_summary = geometry.groupby('setting', as_index=False).agg(
    capture_10=('capture_10_percent', 'mean'),
    lift_10=('lift_10_percent', 'mean'),
    boyce=('boyce', 'mean'),
    auc_secondary=('auc_secondary', 'mean'),
)
display(geometry_summary.style.format({
    'capture_10': '{:.1%}', 'lift_10': '{:.2f}', 'boyce': '{:.3f}', 'auc_secondary': '{:.3f}'
}))
display(Markdown(
    'Read this table as a gate: values near random support the focal design; systematic enrichment '
    'means support or sampling geometry still carries class information.'
))
"""
    ),
    markdown(
        """
### 4.1 Map support and raster-mask proximity

Valid-cell count is the realized sample size inside each nominal 7 × 7 focal window. Distance is
measured from the anchor to the nearest all-band-invalid cell or raster edge. These maps diagnose
where support loss is spatially organized; neither quantity is an archaeological predictor.
"""
    ),
    code(
        """
fig, axes = plt.subplots(len(settings), 2, figsize=(14, 6 * len(settings)), squeeze=False)
for row_index, setting in enumerate(settings):
    design = RESULT['settings'][setting]['designs'][PRIMARY_DESIGN]
    fold = next(
        row for row in design['fold_results']
        if row['method'] == 'M0' and row['repeat'] == 1 and row['fold'] == 1
    )
    availability = pd.DataFrame(fold['availability_predictions']).drop_duplicates('bag_id')
    bags = pd.DataFrame(design['bag_index'])
    sites = bags[bags['label'] == 1]

    for axis, column, title, colorbar_label in [
        (axes[row_index, 0], 'valid_cell_count', 'valid cells in 7 × 7 window', 'Valid cells'),
        (axes[row_index, 1], 'distance_to_mask_boundary', 'distance to mask boundary', 'CRS units'),
    ]:
        plotted = axis.scatter(
            availability['x'], availability['y'], c=availability[column], cmap='viridis',
            s=12, linewidths=0, rasterized=True,
        )
        axis.scatter(
            sites['x'], sites['y'], c=sites[column], cmap='viridis', marker='*', s=70,
            edgecolors='black', linewidths=0.4,
        )
        axis.set(title=f'{setting.title()}: {title}', xlabel='X', ylabel='Y')
        axis.set_aspect('equal')
        plt.colorbar(plotted, ax=axis, fraction=0.035, pad=0.02, label=colorbar_label)
plt.tight_layout()
plt.show()

support_distribution = []
for setting in settings:
    bags = pd.DataFrame(RESULT['settings'][setting]['designs'][PRIMARY_DESIGN]['bag_index'])
    for label, name in [(1, 'site'), (0, 'background')]:
        values = bags[bags['label'] == label]
        support_distribution.append({
            'setting': setting,
            'class': name,
            'n_bags': len(values),
            'mean_valid_cells': values['valid_cell_count'].mean(),
            'min_valid_cells': values['valid_cell_count'].min(),
            'full_window_fraction': (values['valid_cell_count'] == CONFIG['primary_window'] ** 2).mean(),
            'median_boundary_distance': values['distance_to_mask_boundary'].median(),
        })
display(pd.DataFrame(support_distribution).style.format({
    'mean_valid_cells': '{:.1f}', 'full_window_fraction': '{:.1%}',
    'median_boundary_distance': '{:,.1f}',
}))
"""
    ),
    markdown(
        """
### 4.2 Geometry-controlled performance

The full-window design removes mask-related cell loss entirely. The matched-count design preserves
the 7 × 7 landscape geometry but makes the background cell-count distribution exactly equal to the
retained site distribution. Performance that persists in both controls is less plausibly explained
by bag size alone. Differences are descriptive because filtering changes the evaluated bags.
"""
    ),
    code(
        """
FULL_WINDOW_DESIGN = f"focal_{CONFIG['primary_window']}_full_window"
MATCHED_COUNT_DESIGN = f"focal_{CONFIG['primary_window']}_matched_counts"
geometry_designs = [PRIMARY_DESIGN, FULL_WINDOW_DESIGN, MATCHED_COUNT_DESIGN]
controlled = POOLED[POOLED['design'].isin(geometry_designs)].copy()

controlled_summary = controlled.groupby(
    ['setting', 'design', 'method'], as_index=False
).agg(
    n_sites=('n_sites', 'mean'),
    capture_10=('capture_10_percent', 'mean'),
    capture_surplus_10=('capture_surplus_10_percent', 'mean'),
    lift_10=('lift_10_percent', 'mean'),
    boyce=('boyce', 'mean'),
    site_percentile=('site_percentile_median', 'mean'),
    auc_secondary=('auc_secondary', 'mean'),
)
display(controlled_summary.style.format({
    'n_sites': '{:.0f}', 'capture_10': '{:.1%}', 'capture_surplus_10': '{:.1%}',
    'lift_10': '{:.2f}', 'boyce': '{:.3f}', 'site_percentile': '{:.1%}',
    'auc_secondary': '{:.3f}',
}))

control_delta = controlled_summary.pivot(
    index=['setting', 'method'], columns='design', values='capture_surplus_10'
).reset_index()
for design, name in [
    (FULL_WINDOW_DESIGN, 'delta_full_window'),
    (MATCHED_COUNT_DESIGN, 'delta_matched_counts'),
]:
    if design in control_delta:
        control_delta[name] = control_delta[design] - control_delta[PRIMARY_DESIGN]
display(control_delta.style.format({
    column: '{:+.1%}' for column in ['delta_full_window', 'delta_matched_counts']
    if column in control_delta
}))
"""
    ),
    markdown(
        """
## 5. Primary evaluation: mapped-area capture, lift, and capture surplus

The curve answers the operational question directly: *what fraction of held-out sites is captured
when the top x% of mapped availability is designated?* The diagonal is random allocation.
"""
    ),
    code(
        """
capture_rows = []
for row in PRIMARY.to_dict('records'):
    for point in row['capture_curve']:
        capture_rows.append({
            'setting': row['setting'],
            'method': row['method'],
            'repeat': row['repeat'],
            **point,
        })
CAPTURE = pd.DataFrame(capture_rows)
capture_summary = CAPTURE.groupby(
    ['setting', 'method', 'area_fraction'], as_index=False
).agg(
    achieved_area=('achieved_area_fraction', 'mean'),
    capture=('capture', 'mean'),
    lift=('lift', 'mean'),
    capture_surplus=('capture_surplus', 'mean'),
    gain=('gain', 'mean'),
)

fig, axes = plt.subplots(1, len(settings), figsize=(7 * len(settings), 5), squeeze=False)
for axis, setting in zip(axes[0], settings):
    subset = capture_summary[capture_summary['setting'] == setting]
    for method in methods:
        values = subset[subset['method'] == method].sort_values('achieved_area')
        if not values.empty:
            axis.plot(values['achieved_area'], values['capture'], marker='o', label=method)
    axis.plot([0, 0.20], [0, 0.20], color='black', linestyle='--', label='random')
    axis.set(title=f'{setting.title()}: held-out site capture', xlabel='Mapped availability selected', ylabel='Sites captured')
    axis.set_xlim(0, 0.205)
    axis.set_ylim(0, 1.02)
    axis.xaxis.set_major_formatter(lambda value, _: f'{value:.0%}')
    axis.yaxis.set_major_formatter(lambda value, _: f'{value:.0%}')
    axis.grid(alpha=0.2)
axes[0, -1].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

headline = PRIMARY.groupby(['setting', 'method'], as_index=False).agg(
    area_5=('achieved_area_5_percent', 'mean'),
    capture_5=('capture_5_percent', 'mean'),
    lift_5=('lift_5_percent', 'mean'),
    surplus_5=('capture_surplus_5_percent', 'mean'),
    area_10=('achieved_area_10_percent', 'mean'),
    capture_10=('capture_10_percent', 'mean'),
    lift_10=('lift_10_percent', 'mean'),
    surplus_10=('capture_surplus_10_percent', 'mean'),
    area_20=('achieved_area_20_percent', 'mean'),
    capture_20=('capture_20_percent', 'mean'),
    lift_20=('lift_20_percent', 'mean'),
    surplus_20=('capture_surplus_20_percent', 'mean'),
)
display(headline.style.format({
    'area_5': '{:.1%}', 'area_10': '{:.1%}', 'area_20': '{:.1%}',
    'capture_5': '{:.1%}', 'capture_10': '{:.1%}', 'capture_20': '{:.1%}',
    'lift_5': '{:.2f}', 'lift_10': '{:.2f}', 'lift_20': '{:.2f}',
    'surplus_5': '{:+.1%}', 'surplus_10': '{:+.1%}', 'surplus_20': '{:+.1%}',
}))

assert np.allclose(
    PRIMARY['gain_10_percent'], 1.0 - 1.0 / PRIMARY['lift_10_percent'], equal_nan=True
)
kvamme_rows = PRIMARY.assign(
    gain_reconstructed_from_lift=1.0 - 1.0 / PRIMARY['lift_10_percent']
)
kvamme = kvamme_rows.groupby(['setting', 'method'], as_index=False).agg(
    lift_10=('lift_10_percent', 'mean'),
    kvamme_gain_10=('gain_10_percent', 'mean'),
    gain_reconstructed_from_lift=('gain_reconstructed_from_lift', 'mean'),
)
display(Markdown(
    '**Kvamme Gain crosswalk (secondary):** retained for archaeological comparability. '
    'The reconstructed column confirms that it contains no ordering information beyond lift.'
))
display(kvamme.style.format({
    'lift_10': '{:.2f}', 'kvamme_gain_10': '{:.3f}',
    'gain_reconstructed_from_lift': '{:.3f}',
}))
"""
    ),
    markdown(
        """
## 6. Presence-only rank diagnostics

Boyce asks whether observed-to-expected site frequency rises with suitability. Site percentiles are
more literal: 0.90 means a held-out site outranked 90% of mapped availability under its fold model.
Undefined Boyce values remain missing rather than being replaced.
"""
    ),
    code(
        """
rank_summary = PRIMARY.groupby(['setting', 'method'], as_index=False).agg(
    boyce=('boyce', 'mean'),
    median_site_percentile=('site_percentile_median', 'mean'),
    q25_site_percentile=('site_percentile_q25', 'mean'),
    q75_site_percentile=('site_percentile_q75', 'mean'),
)
display(rank_summary.style.format({
    'boyce': '{:.3f}',
    'median_site_percentile': '{:.1%}',
    'q25_site_percentile': '{:.1%}',
    'q75_site_percentile': '{:.1%}',
}))

fig, axes = plt.subplots(1, len(settings), figsize=(7 * len(settings), 4.5), squeeze=False)
for axis, setting in zip(axes[0], settings):
    values = PRIMARY[PRIMARY['setting'] == setting]
    plot_methods = [method for method in methods if method in set(values['method'])]
    distributions = [
        np.concatenate(values[values['method'] == method]['site_percentiles'].map(np.asarray).to_list())
        for method in plot_methods
    ]
    axis.boxplot(distributions, tick_labels=plot_methods, showfliers=False)
    axis.axhline(0.5, color='black', linestyle='--', linewidth=1)
    axis.set(title=f'{setting.title()}: held-out site ranks', ylabel='Availability percentile')
    axis.tick_params(axis='x', rotation=45)
    axis.yaxis.set_major_formatter(lambda value, _: f'{value:.0%}')
    axis.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
## 7. Spatial diagnostics of failures and model disagreement

Global and Local Moran statistics are applied to **held-out site percentile shortfall**
(`1 - site availability percentile`), not to the raw suitability surface. High–high local clusters
therefore identify neighboring sites that the model repeatedly ranks poorly. Local permutation
p-values are FDR-adjusted. A second diagnostic maps spatially coherent method disagreement with M0.

These are exploratory diagnostics, not performance scores. The focal predictions are expected to
be autocorrelated, and the k-nearest-neighbor graph uses Euclidean distance; river-network distance
would be preferable when an appropriate network is available.
"""
    ),
    code(
        """
failure_rows = []
for row in PRIMARY.to_dict('records'):
    diagnostic = row['site_shortfall_spatial_diagnostic']
    local = diagnostic['local']
    failure_rows.append({
        'setting': row['setting'],
        'method': row['method'],
        'repeat': row['repeat'],
        'global_moran_i': diagnostic['global_moran_i'],
        'global_p_value': diagnostic['global_p_value'],
        'fdr_local_clusters': sum(item['significant_fdr'] for item in local),
        'fdr_high_high_failures': sum(
            item['significant_fdr'] and item['cluster'] == 'high-high' for item in local
        ),
    })
FAILURE_SPATIAL = pd.DataFrame(failure_rows)
display(FAILURE_SPATIAL.groupby(['setting', 'method'], as_index=False).agg(
    moran_i=('global_moran_i', 'mean'),
    permutation_p=('global_p_value', 'mean'),
    local_clusters=('fdr_local_clusters', 'mean'),
    high_high_failures=('fdr_high_high_failures', 'mean'),
).style.format({
    'moran_i': '{:.3f}', 'permutation_p': '{:.3f}',
    'local_clusters': '{:.1f}', 'high_high_failures': '{:.1f}',
}))

SPATIAL_METHOD = 'M0'
SPATIAL_REPEAT = 1
cluster_colors = {
    'high-high': '#d73027', 'low-low': '#4575b4',
    'high-low': '#fdae61', 'low-high': '#74add1',
}
fig, axes = plt.subplots(1, len(settings), figsize=(7 * len(settings), 6), squeeze=False)
for axis, setting in zip(axes[0], settings):
    row = PRIMARY[
        (PRIMARY['setting'] == setting)
        & (PRIMARY['method'] == SPATIAL_METHOD)
        & (PRIMARY['repeat'] == SPATIAL_REPEAT)
    ].iloc[0]
    local = pd.DataFrame(row['site_shortfall_spatial_diagnostic']['local'])
    axis.scatter(local['x'], local['y'], color='lightgray', s=24, label='not FDR-significant')
    for cluster, values in local[local['significant_fdr']].groupby('cluster'):
        axis.scatter(
            values['x'], values['y'], color=cluster_colors[cluster], s=45,
            edgecolors='black', linewidths=0.3, label=cluster,
        )
    axis.set(
        title=f'{setting.title()}: {SPATIAL_METHOD} held-out site shortfall LISA',
        xlabel='X', ylabel='Y',
    )
    axis.set_aspect('equal')
    axis.legend(fontsize=8)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
disagreement_rows = []
for setting in settings:
    diagnostics = RESULT['settings'][setting]['designs'][PRIMARY_DESIGN].get(
        'availability_disagreement_vs_M0', []
    )
    for diagnostic in diagnostics:
        disagreement_rows.append({
            'setting': setting,
            'method': diagnostic['method'],
            'repeat': diagnostic['repeat'],
            'global_moran_i': diagnostic['global_moran_i'],
            'global_p_value': diagnostic['global_p_value'],
            'fdr_local_clusters': sum(
                item['significant_fdr'] for item in diagnostic['local']
            ),
        })
DISAGREEMENT = pd.DataFrame(disagreement_rows)
if DISAGREEMENT.empty:
    display(Markdown('Model-disagreement diagnostics require M0 and at least one comparison method.'))
else:
    display(DISAGREEMENT.groupby(['setting', 'method'], as_index=False).agg(
        moran_i=('global_moran_i', 'mean'),
        permutation_p=('global_p_value', 'mean'),
        local_clusters=('fdr_local_clusters', 'mean'),
    ).style.format({
        'moran_i': '{:.3f}', 'permutation_p': '{:.3f}', 'local_clusters': '{:.1f}',
    }))

    DISAGREEMENT_METHOD = 'RF-mean' if 'RF-mean' in set(DISAGREEMENT['method']) else DISAGREEMENT['method'].iloc[0]
    fig, axes = plt.subplots(1, len(settings), figsize=(7 * len(settings), 6), squeeze=False)
    plotted = False
    for axis, setting in zip(axes[0], settings):
        diagnostics = RESULT['settings'][setting]['designs'][PRIMARY_DESIGN][
            'availability_disagreement_vs_M0'
        ]
        diagnostic = next(
            row for row in diagnostics
            if row['method'] == DISAGREEMENT_METHOD and row['repeat'] == SPATIAL_REPEAT
        )
        local = pd.DataFrame(diagnostic['local'])
        limit = max(abs(local['value'].min()), abs(local['value'].max()))
        scatter = axis.scatter(
            local['x'], local['y'], c=local['value'], cmap='coolwarm', vmin=-limit, vmax=limit,
            s=14, linewidths=0, rasterized=True,
        )
        significant = local[local['significant_fdr']]
        axis.scatter(
            significant['x'], significant['y'], facecolors='none', edgecolors='black',
            s=32, linewidths=0.7, label='FDR-significant local association',
        )
        axis.set(
            title=f'{setting.title()}: {DISAGREEMENT_METHOD} − M0 rank disagreement',
            xlabel='X', ylabel='Y',
        )
        axis.set_aspect('equal')
        axis.legend(fontsize=8)
        plt.colorbar(scatter, ax=axis, fraction=0.035, pad=0.02, label='Availability-percentile difference')
    plt.tight_layout()
    plt.show()
"""
    ),
    markdown(
        """
## 8. Secondary discrimination: ROC AUC and PR AUC

These plots compare sites with constructed background bags. They are useful diagnostics, but they
do not define the mapped-area operating point and do not turn background into confirmed absence.
"""
    ),
    code(
        """
secondary = PRIMARY.groupby(['setting', 'method'], as_index=False).agg(
    auc=('auc_secondary', 'mean'),
    pr_auc=('pr_auc_secondary', 'mean'),
)
display(secondary.style.format({'auc': '{:.3f}', 'pr_auc': '{:.3f}'}))

fig, axes = plt.subplots(1, len(settings), figsize=(7 * len(settings), 5), squeeze=False)
for axis, setting in zip(axes[0], settings):
    subset = PRIMARY[(PRIMARY['setting'] == setting) & (PRIMARY['method'] != 'NEG-geometry')]
    for method, rows in subset.groupby('method'):
        curves = []
        grid = np.linspace(0, 1, 101)
        for row in rows.to_dict('records'):
            fpr, tpr, _ = roc_curve(row['labels'], row['availability_percentiles'])
            curves.append(np.interp(grid, fpr, tpr))
        axis.plot(grid, np.mean(curves, axis=0), label=f"{method} ({rows['auc_secondary'].mean():.3f})")
    axis.plot([0, 1], [0, 1], color='black', linestyle='--')
    axis.set(title=f'{setting.title()}: secondary ROC', xlabel='Background false-positive rate', ylabel='Site true-positive rate')
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
## 9. Fold-safe validation maps

Each panel uses one fitted fold only. Availability colors are fold-specific percentiles; triangles
are training sites and stars are sites withheld from that model. These maps are diagnostic views of
the fixed uniform availability sample, not final full-data prediction products.
"""
    ),
    code(
        """
MAP_METHOD = 'M0'
MAP_REPEAT = 1
MAP_FOLD = 1

fig, axes = plt.subplots(1, len(settings), figsize=(8 * len(settings), 7), squeeze=False)
for axis, setting in zip(axes[0], settings):
    design = RESULT['settings'][setting]['designs'][PRIMARY_DESIGN]
    fold = next(
        row for row in design['fold_results']
        if row['method'] == MAP_METHOD and row['repeat'] == MAP_REPEAT and row['fold'] == MAP_FOLD
    )
    assignment = next(
        row for row in design['fold_plan']['assignments']
        if row['repeat'] == MAP_REPEAT and row['fold'] == MAP_FOLD
    )
    availability = pd.DataFrame(fold['availability_predictions'])
    bags = pd.DataFrame(design['bag_index'])
    train_sites = bags[(bags['label'] == 1) & bags['bag_id'].isin(assignment['train_ids'])]
    heldout_sites = bags[(bags['label'] == 1) & bags['bag_id'].isin(assignment['test_ids'])]
    scatter = axis.scatter(
        availability['x'], availability['y'],
        c=availability['availability_percentile'], cmap='viridis', s=14,
        vmin=0, vmax=1, linewidths=0, rasterized=True,
    )
    axis.scatter(train_sites['x'], train_sites['y'], marker='^', s=45, facecolors='none', edgecolors='white', linewidths=1.2, label='train sites')
    axis.scatter(heldout_sites['x'], heldout_sites['y'], marker='*', s=110, color='red', edgecolors='black', linewidths=0.5, label='held-out sites')
    axis.set(title=f'{setting.title()} — {MAP_METHOD}, repeat {MAP_REPEAT}, fold {MAP_FOLD}', xlabel='X', ylabel='Y')
    axis.set_aspect('equal')
    axis.legend(loc='best')
    plt.colorbar(scatter, ax=axis, fraction=0.035, pad=0.02, label='Availability percentile')
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
## 10. Compute diagnostics

Timing includes all folds in one repeat, including fixed-availability prediction. Python peak memory
does not include every device allocation, so compare it as a relative diagnostic.
"""
    ),
    code(
        """
compute = PRIMARY.groupby(['setting', 'method'], as_index=False).agg(
    fit_seconds=('fit_seconds', 'mean'),
    predict_seconds=('predict_seconds', 'mean'),
    peak_python_memory_mb=('peak_python_memory_mb', 'max'),
)
display(compute.style.format({
    'fit_seconds': '{:,.2f}',
    'predict_seconds': '{:,.2f}',
    'peak_python_memory_mb': '{:,.1f}',
}))
"""
    ),
    markdown(
        """
## 11. Support sensitivity

Geometry mode intentionally omits the broader scale sweep. Re-run with `EVALUATION_MODE = 'full'` and
`RERUN_EVALUATION = True` to compare 7 × 7, 9 × 9, 11 × 11, and the original irregular bags after
the primary result is understood.
"""
    ),
    code(
        """
support_designs = [
    f'focal_{window}' for window in CONFIG['support_windows']
    if window != CONFIG['primary_window']
] + ['original_irregular']
support_rows = POOLED[POOLED['design'].isin(support_designs)]
if support_rows.empty:
    display(Markdown('**Not run.** This is correct for the primary evaluation mode.'))
else:
    support_summary = support_rows.groupby(['setting', 'design', 'method'], as_index=False).agg(
        capture_10=('capture_10_percent', 'mean'),
        lift_10=('lift_10_percent', 'mean'),
        boyce=('boyce', 'mean'),
        auc_secondary=('auc_secondary', 'mean'),
    )
    display(support_summary.style.format({
        'capture_10': '{:.1%}', 'lift_10': '{:.2f}', 'boyce': '{:.3f}', 'auc_secondary': '{:.3f}'
    }))
"""
    ),
    markdown(
        """
## 12. Structured interpretation

This table names descriptive leaders without converting one Section 6 run into a promotion rule.
Agreement across mapped-area capture, capture surplus, Boyce, site ranks, geometry controls, and
failure maps is stronger evidence than a small AUC difference alone.
"""
    ),
    code(
        """
candidate_methods = [method for method in methods if method in set(PRIMARY['method'])]
summary = PRIMARY[PRIMARY['method'].isin(candidate_methods)].groupby(
    ['setting', 'method'], as_index=False
).agg(
    capture_10=('capture_10_percent', 'mean'),
    capture_surplus_10=('capture_surplus_10_percent', 'mean'),
    lift_10=('lift_10_percent', 'mean'),
    boyce=('boyce', 'mean'),
    site_percentile=('site_percentile_median', 'mean'),
    auc_secondary=('auc_secondary', 'mean'),
)

interpretation_rows = []
for setting, values in summary.groupby('setting'):
    for metric, label in [
        ('capture_10', '10% mapped-area capture'),
        ('capture_surplus_10', '10% mapped-area capture surplus'),
        ('boyce', 'Continuous Boyce'),
        ('site_percentile', 'median held-out site percentile'),
        ('auc_secondary', 'secondary ROC AUC'),
    ]:
        available = values.dropna(subset=[metric])
        if not available.empty:
            winner = available.loc[available[metric].idxmax()]
            interpretation_rows.append({
                'setting': setting,
                'evidence': label,
                'descriptive_leader': winner['method'],
                'value': winner[metric],
                'decision': 'retain for cross-setting validation; no promotion from Section 6 alone',
            })
display(pd.DataFrame(interpretation_rows).style.format({'value': '{:.3f}'}))
"""
    ),
    markdown(
        """
## 13. Experimental appendix boundary

ORF seed sensitivity, nominal shrinkage, and spatial shrinkage remain experimental extensions. Their
completed diagnostic notebook outputs are preserved, but they are not mixed into the primary
M0–M3 plus LR/RF comparison. This keeps the main evaluation question stable while those extensions
await more settings and more realistic spatial data.

**End-to-end reading order:** audit → geometry maps and controlled designs → capture/lift/surplus →
Boyce/site ranks → spatial failure and disagreement diagnostics → fold-safe maps → secondary
AUC/PR → compute → optional support sensitivity.
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3 (KLRFome)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
