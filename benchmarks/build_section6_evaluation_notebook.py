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
| Primary | capture / lift / gain at 5%, 10%, and 20% of mapped availability | How efficiently high-suitability area captures held-out sites |
| Primary | Continuous Boyce and held-out site availability percentiles | Whether sites concentrate toward higher mapped suitability |
| Secondary | ROC AUC and PR AUC | Discrimination against constructed background, not true absence |
| Diagnostic | geometry-only control, timing, memory, support sensitivity | Detect design pathologies and computational tradeoffs |

Scores are relative suitability. They are not occurrence probabilities. No method is promoted or
removed from one physio-shed.
"""
    ),
    markdown(
        """
## 1. Configuration and execution

The first run executes the primary comparison if no result exists. Set `RERUN_EVALUATION = True`
to replace it. Use `EVALUATION_MODE = 'full'` only after the primary run is interpretable; full mode
adds 9 × 9, 11 × 11, and original-irregular support sensitivities.
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


def repository_root(start=Path.cwd()):
    for candidate in [start, *start.parents]:
        if (candidate / 'pyproject.toml').exists() and (candidate / 'klrfome').exists():
            return candidate
    raise RuntimeError('Run this notebook from inside the KLRFome_JAX repository.')


ROOT = repository_root()
CONFIG_PATH = ROOT / 'benchmarks/section6_evaluation_config.json'
CONFIG = json.loads(CONFIG_PATH.read_text())
RESULT_PATH = ROOT / CONFIG['output']

EVALUATION_MODE = 'primary'  # 'primary' or 'full'
RERUN_EVALUATION = False
SELECTED_METHODS = None  # e.g. ['M0', 'M1', 'LR-mean', 'RF-mean'] for a bounded run

if RERUN_EVALUATION or not RESULT_PATH.exists():
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
    audit_rows.append({
        'setting': setting,
        'retained_sites': common['n_site_bags'],
        'training_backgrounds': common['n_background_bags'],
        'excluded_sites': len(common['excluded_site_ids']),
        'availability_anchors': available['n_retained'],
        'availability_rejected': available['n_rejected'],
        'spatial_block_width': audit['shared_block_width'],
        'support_windows': ', '.join(map(str, common['window_sizes'])),
    })
audit_table = pd.DataFrame(audit_rows)
display(audit_table.style.format({'spatial_block_width': '{:,.1f}'}))

for setting, payload in RESULT['settings'].items():
    primary = payload['designs'][f"focal_{CONFIG['primary_window']}"]
    assignments = primary['fold_plan']['assignments']
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
## 5. Primary evaluation: mapped-area capture and lift

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
    gain=('gain', 'mean'),
)

methods = [method for method in PRIMARY['method'].unique() if method != 'NEG-geometry']
settings = list(RESULT['settings'])
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
    area_10=('achieved_area_10_percent', 'mean'),
    capture_10=('capture_10_percent', 'mean'),
    lift_10=('lift_10_percent', 'mean'),
    area_20=('achieved_area_20_percent', 'mean'),
    capture_20=('capture_20_percent', 'mean'),
    lift_20=('lift_20_percent', 'mean'),
)
display(headline.style.format({
    'area_5': '{:.1%}', 'area_10': '{:.1%}', 'area_20': '{:.1%}',
    'capture_5': '{:.1%}', 'capture_10': '{:.1%}', 'capture_20': '{:.1%}',
    'lift_5': '{:.2f}', 'lift_10': '{:.2f}', 'lift_20': '{:.2f}',
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
## 7. Secondary discrimination: ROC AUC and PR AUC

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
## 8. Fold-safe validation maps

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
## 9. Compute diagnostics

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
## 10. Support sensitivity

Primary mode intentionally omits this section. Re-run with `EVALUATION_MODE = 'full'` and
`RERUN_EVALUATION = True` to compare 7 × 7, 9 × 9, 11 × 11, and the original irregular bags after
the primary result is understood.
"""
    ),
    code(
        """
support_rows = POOLED[POOLED['design'] != PRIMARY_DESIGN]
if support_rows.empty:
    display(Markdown('**Not run.** This is correct for the primary evaluation mode.'))
else:
    support_summary = POOLED.groupby(['setting', 'design', 'method'], as_index=False).agg(
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
## 11. Structured interpretation

This table names descriptive leaders without converting one Section 6 run into a promotion rule.
Agreement across mapped-area capture, Boyce, site ranks, and maps is stronger evidence than a small
AUC difference alone.
"""
    ),
    code(
        """
candidate_methods = [method for method in methods if method in set(PRIMARY['method'])]
summary = PRIMARY[PRIMARY['method'].isin(candidate_methods)].groupby(
    ['setting', 'method'], as_index=False
).agg(
    capture_10=('capture_10_percent', 'mean'),
    lift_10=('lift_10_percent', 'mean'),
    boyce=('boyce', 'mean'),
    site_percentile=('site_percentile_median', 'mean'),
    auc_secondary=('auc_secondary', 'mean'),
)

interpretation_rows = []
for setting, values in summary.groupby('setting'):
    for metric, label in [
        ('capture_10', '10% mapped-area capture'),
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
## 12. Experimental appendix boundary

ORF seed sensitivity, nominal shrinkage, and spatial shrinkage remain experimental extensions. Their
completed diagnostic notebook outputs are preserved, but they are not mixed into the primary
M0–M3 plus LR/RF comparison. This keeps the main evaluation question stable while those extensions
await more settings and more realistic spatial data.

**End-to-end reading order:** audit → geometry gate → capture/lift → Boyce/site ranks → maps →
secondary AUC/PR → compute → optional support sensitivity.
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
