#!/usr/bin/env python3
"""Summarize declared completed development runs from saved reports only.

No data arrays are opened, checkpoints deserialized, or new model scores computed.
All declared runs remain separate; unequal budgets are not pooled as seed repeats.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/phase1-matplotlib')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from fluvius_cgan.training import _json, _sha256


FIELDS = ('normalized_profile_sw1', 'daily_energy_w1_kwh', 'diversity_ratio',
          'mean_profile_rmse_kwh', 'condition_mean_error_rmse_kwh')
TITLES = ('Normalized profile SW1', 'Daily-energy W1 (kWh)', 'Diversity ratio',
          'Macro mean-profile RMSE (kWh)', 'Condition-mean RMSE (kWh)')
SELECTION_FIELDS = ('indices', 'conditions', 'index_seed', 'directions_seed', 'directions_count')
BASELINE_COLORS = {'mean': '#555555', 'resample': '#9559A3', 'copula': '#A56C39'}


def _read(path):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON value in {path}: {value}')
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def _verified(directory, required):
    path = directory / 'manifest.json'
    manifest = _read(path)
    if manifest.get('status') != 'completed':
        raise ValueError(f'Requires a completed manifest: {directory}')
    hashes = manifest.get('artifact_hashes', {})
    for name in required:
        if name not in hashes or _sha256(directory / name) != hashes[name]:
            raise ValueError(f'Artifact hash mismatch: {directory / name}')
    return manifest


def _shared(manifest, reference):
    if manifest['data_hashes'] != reference['data_hashes']:
        raise ValueError('Development datasets differ')
    for name in SELECTION_FIELDS:
        if manifest['selection'].get(name) != reference['selection'].get(name):
            raise ValueError(f'Development selection differs at {name}')
    for name in ('minimum', 'maximum'):
        if manifest['metric_normalization'][name] != reference['metric_normalization'][name]:
            raise ValueError(f'Development physical normalization differs at {name}')


def _metrics(scores):
    return {name: scores['macro'][name] if name in scores['macro'] else scores['diagnostics'][name]
            for name in FIELDS}


def _stage(row):
    return {'step': row['step'], 'row_presentations': row['row_presentations'],
            'elapsed_seconds': row['elapsed_seconds'], **_metrics({
                'macro': row['validation'], 'diagnostics': row['diagnostics']})}


def _conditions(scores, reference_counts=None):
    values = scores['per_condition']
    if set(values) != {str(label) for label in range(64)}:
        raise ValueError('All 64 conditions must be reported')
    counts = {label: row['n_real'] for label, row in values.items()}
    if any(row['n_real'] < 1 or row['n_real'] != row['n_fake'] for row in values.values()):
        raise ValueError('Requires positive matched condition sample counts')
    if reference_counts is not None and counts != reference_counts:
        raise ValueError('Condition sample counts differ')
    return counts


def _csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _plot(entries, baselines, output, post_initial=False):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    xlabel = ('Training updates (GAN pairs; flow optimizer steps)'
              if any(entry['method'] == 'conditional_flow_matching' for entry in entries)
              else 'Generator/discriminator update pairs')
    for axis, metric, title in zip(axes.flat, FIELDS, TITLES):
        for entry in entries:
            history = entry['history'][1:] if post_initial else entry['history']
            if not history:
                continue
            line, = axis.plot([row['step'] for row in history],
                [_stage(row)[metric] for row in history], marker='.',
                label=f"{entry['name']} (seed {entry['config']['seed']})")
            selected = entry['best']
            if selected['step'] >= history[0]['step']:
                axis.scatter([selected['step']], [selected[metric]], marker='*', s=110,
                    color=line.get_color(), edgecolor='black', linewidth=.4, zorder=3)
        for kind, scores in baselines.items():
            value = _metrics(scores)[metric]
            if value is not None:
                axis.axhline(value, linestyle='--', alpha=.8, color=BASELINE_COLORS[kind],
                             label=f'{kind} (matched selection draw)')
        if metric == 'diversity_ratio':
            axis.axhline(1., color='black', linestyle=':', linewidth=1, label='Equal variance')
        axis.set(title=title, xlabel=xlabel)
        axis.grid(alpha=.2)
        axis.margins(y=.08)
    axes.flat[-1].set_axis_off()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc='center', fontsize=8)
    title = 'Post-test development: fixed inner-validation rows; stars mark selected checkpoints'
    if post_initial:
        title += '\nDetail excludes the first recorded checkpoint; complete curves are supplied separately'
    fig.suptitle(title)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def summarize(runs, baselines, output, generation_seed=20261022):
    runs, baselines, output = [Path(path).resolve() for path in runs], Path(baselines).resolve(), Path(output)
    if not runs or len(runs) != len(set(runs)):
        raise ValueError('Declare a nonempty list of distinct run directories')
    if output.exists():
        raise FileExistsError(output)
    baseline_manifest = _verified(baselines, ('mean.json', 'resample.json', 'copula.json', 'summary.json'))
    baseline_scores, baseline_reports = {}, {}
    counts = None
    for kind in ('mean', 'resample', 'copula'):
        report = _read(baselines / f'{kind}.json')
        matches = [draw['metrics'] for draw in report['draws'] if draw['generation_seed'] == generation_seed]
        if len(matches) != 1 or report['method'] != kind:
            raise ValueError(f'Requires one matched selection draw for baseline {kind}')
        counts = _conditions(matches[0], counts)
        for draw in report['draws']:
            _conditions(draw['metrics'], counts)
        baseline_scores[kind], baseline_reports[kind] = matches[0], report
    entries, sources = [], {str(baselines / 'manifest.json'): _sha256(baselines / 'manifest.json')}
    for index, run in enumerate(runs):
        manifest = _verified(run, ('best.pt', 'last.pt', 'history.json', 'best_validation.json'))
        _shared(manifest, baseline_manifest)
        if manifest['selection'].get('generation_seed') != generation_seed:
            raise ValueError('Training and baseline selection generation seeds differ')
        history, scores = _read(run / 'history.json'), _read(run / 'best_validation.json')
        _conditions(scores, counts)
        steps = [row['step'] for row in history]
        if not steps or steps != sorted(set(steps)) or steps[-1] != manifest['completed_steps']:
            raise ValueError(f'Incomplete or unordered history: {run}')
        if steps[-1] != manifest['config']['steps']:
            raise ValueError(f'Run did not finish its declared update budget: {run}')
        if history[-1]['row_presentations'] != manifest['row_presentations']:
            raise ValueError(f'History and manifest training compute disagree: {run}')
        selected = min(history, key=lambda row: row['validation']['normalized_profile_sw1'])
        if (selected['step'] != manifest['best_step'] or
                selected['validation'] != scores['macro'] or
                selected['diagnostics'] != scores['diagnostics'] or
                selected['validation']['normalized_profile_sw1'] != manifest['best_validation_sw1']):
            raise ValueError(f'Selected report disagrees with history/manifest: {run}')
        name = run.name if sum(path.name == run.name for path in runs) == 1 else f'{index + 1}:{run.name}'
        entry = {'name': name, 'directory': str(run), 'config': manifest['config'],
            'method': manifest.get('method', 'conditional_gan'),
            'initial': _stage(history[0]), 'best': _stage(selected), 'last': _stage(history[-1]),
            'completed_steps': manifest['completed_steps'], 'row_presentations': manifest['row_presentations'],
            'training_pass_equivalents': manifest['row_presentations'] / manifest['available_training_rows'],
            'wall_seconds': manifest['elapsed_seconds'],
            'best_at_budget_boundary': selected['step'] == history[-1]['step'],
            'history': history, 'best_metrics': scores}
        entries.append(entry)
        sources[str(run / 'manifest.json')] = _sha256(run / 'manifest.json')
    result = {'schema_version': 1, 'role': 'post-test development',
        'notes': ['All declared completed runs are reported separately; no cross-budget seed pooling.',
                  'Best is selected on these validation curves, not an independent performance estimate.',
                  'No convergence, superiority, calibrated significance, or publication-readiness claim is made.',
                  'Baseline draws are sampling replicates, not independent training runs.'],
        'generation_seed_for_comparison': generation_seed, 'data_hashes': baseline_manifest['data_hashes'],
        'selection': baseline_manifest['selection'], 'metric_normalization': baseline_manifest['metric_normalization'],
        'runs': entries, 'baseline_reports': baseline_reports}
    condition_rows, checkpoint_rows, curve_rows = [], [], []
    normalizer = baseline_manifest['metric_normalization']
    denominator = normalizer['maximum'] - normalizer['minimum'] or 1.
    def add_conditions(name, seed, stage, scores):
        for label in sorted(scores['per_condition'], key=int):
            record = scores['per_condition'][label]
            metrics = dict(record['metrics'], normalized_profile_sw1=record['metrics']['profile_sw1_kwh'] / denominator)
            for metric, value in metrics.items():
                condition_rows.append({'name': name, 'generation_seed': seed, 'stage': stage,
                    'condition': label, 'n_real': record['n_real'], 'n_fake': record['n_fake'],
                    'metric': metric, 'value': value})
    for entry in entries:
        for stage in ('initial', 'best', 'last'):
            checkpoint_rows.append({'name': entry['name'], 'training_seed': entry['config']['seed'],
                'stage': stage, **entry[stage], 'budget_steps': entry['completed_steps'],
                'total_row_presentations': entry['row_presentations'], 'wall_seconds': entry['wall_seconds'],
                'training_pass_equivalents': entry['training_pass_equivalents'],
                'best_at_budget_boundary': entry['best_at_budget_boundary']})
        for row in entry['history']:
            curve_rows.append({'name': entry['name'], 'training_seed': entry['config']['seed'],
                **_stage(row), 'method': entry['method'],
                'loss_g': row.get('loss_g'), 'loss_d': row.get('loss_d'),
                'loss_velocity_mse': row.get('loss_velocity_mse'),
                'common_hour_error_fraction': row['diagnostics']['common_hour_error_fraction']})
        add_conditions(entry['name'], generation_seed, 'selected checkpoint', entry['best_metrics'])
    for kind, report in baseline_reports.items():
        for draw in report['draws']:
            add_conditions(kind, draw['generation_seed'], 'baseline sampling draw', draw['metrics'])
    output.mkdir(parents=True, exist_ok=False)
    _json(output / 'summary.json', result)
    _csv(output / 'checkpoints.csv', checkpoint_rows)
    _csv(output / 'learning_curves.csv', curve_rows)
    _csv(output / 'conditions.csv', condition_rows)
    _plot(entries, baseline_scores, output / 'learning_curves.png')
    if all(len(entry['history']) > 1 for entry in entries):
        _plot(entries, baseline_scores, output / 'learning_curves_detail.png', post_initial=True)
    _json(output / 'manifest.json', {'schema_version': 1, 'role': result['role'], 'status': 'completed',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'declared_run_directories': [str(path) for path in runs], 'baseline_directory': str(baselines),
        'source_manifest_hashes': sources, 'script_sha256': _sha256(Path(__file__)),
        'artifact_hashes': {path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file()}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--baselines', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generation-seed', type=int, default=20261022)
    args = parser.parse_args()
    result = summarize(args.runs, args.baselines, args.output, args.generation_seed)
    print(f"Saved {len(result['runs'])} declared development runs to {args.output}")


if __name__ == '__main__':
    main()
