"""Render Phase 1 PNG/PDF figures and CSV tables from saved JSON reports only.

Usage: python scripts/render_phase1.py --runs RUNS --evaluations EVALUATIONS --output NEW_DIR
No data arrays, model checkpoints, fitting or evaluation are accessed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/phase1-matplotlib')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


SEEDS = (11, 22, 33, 44, 55)
METHODS = ('conditional', 'unconditional', 'mean', 'resample', 'copula')
NAMES = {'conditional': 'Conditional GAN', 'unconditional': 'Unconditional GAN',
         'mean': 'Conditional mean', 'resample': 'Train resampling', 'copula': 'Gaussian copula'}
COLORS = {'conditional': '#2463A6', 'unconditional': '#C66A24', 'mean': '#AA862A',
          'resample': '#687E34', 'copula': '#B35285', 'real': '#252525'}
MARKERS = {'conditional': 'o', 'unconditional': '^', 'mean': 'D', 'resample': 's', 'copula': 'P'}
STYLES = {'real': '-', 'conditional': '-', 'resample': '-.', 'copula': '--'}
PRIMARY = 'normalized_profile_sw1'
CONDITIONS = tuple(range(64))
REPRESENTATIVE = tuple(range(0, 64, 8))


def _sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def _read(path):
    if Path(path).suffix != '.json':
        raise ValueError('Renderer input must be saved JSON')
    def invalid(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def _number(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _stats(values):
    available = [float(v) for v in values if v is not None]
    return (float(np.mean(available)) if available else None,
            float(np.std(available, ddof=1)) if len(available) > 1 else None)


def _units(metric):
    if metric == PRIMARY or 'ratio' in metric or 'acf' in metric:
        return 'dimensionless'
    if metric.endswith('_kw'):
        return 'kW'
    if metric.endswith('_kwh2'):
        return 'kWh^2'
    if metric.endswith('_kwh'):
        return 'kWh'
    return 'count or diagnostic'


def _denominator(report):
    normalizer = report['normalization']
    if normalizer.get('fitted_on') != 'train':
        raise ValueError('Normalization must be train-fitted')
    minimum = _number(normalizer['minimum'], 'normalization minimum')
    maximum = _number(normalizer['maximum'], 'normalization maximum')
    if maximum < minimum:
        raise ValueError('Invalid normalization bounds')
    return maximum - minimum or 1.


def _inputs(runs, evaluations):
    reports, paths = {}, {}
    report_directory = evaluations / 'reports' if (evaluations / 'reports').is_dir() else evaluations
    for path in sorted(report_directory.glob('*.json')):
        value = _read(path)
        if not isinstance(value, dict) or 'replicates' not in value or 'macro_summary' not in value:
            continue
        method = value.get('method')
        if method not in METHODS:
            raise ValueError(f'Unknown evaluation method: {method}')
        seed = value['model']['config']['seed'] if method in METHODS[:2] else None
        key = (method, seed)
        if key in reports:
            raise ValueError(f'Duplicate evaluation for {key}')
        if value.get('generation_seeds') != [31001, 31002, 31003]:
            raise ValueError('Expected all three fixed reporting generation seeds 31001/31002/31003')
        if [r['generation_seed'] for r in value['replicates']] != value['generation_seeds']:
            raise ValueError('Evaluation replicate seeds disagree with metadata')
        _denominator(value)
        for replicate in value['replicates']:
            conditions = replicate['metrics']['per_condition']
            if set(conditions) != {str(c) for c in CONDITIONS}:
                raise ValueError(f'{key}: all 64 conditions must be present')
            for condition in conditions.values():
                if condition['n_real'] < 1 or condition['n_fake'] != condition['n_real']:
                    raise ValueError('Expected positive matched real/generated condition counts')
                for metric, result in condition['metrics'].items():
                    if result is not None and not isinstance(result, bool):
                        _number(result, metric)
            _number(replicate['metrics']['macro'][PRIMARY], PRIMARY)
        reports[key], paths[key] = value, path
    expected = {(m, s) for m in METHODS[:2] for s in SEEDS} | {(m, None) for m in METHODS[2:]}
    if set(reports) != expected:
        raise ValueError(f'Expected exactly 13 evaluations; missing={expected - set(reports)}, extra={set(reports) - expected}')
    exemplar = reports[('conditional', 11)]
    for key, value in reports.items():
        for field in ('split', 'data_hashes', 'directions_sha256', 'protocol_sha256', 'normalization'):
            if value.get(field) != exemplar.get(field):
                raise ValueError(f'{key}: evaluation population/protocol mismatch in {field}')
        for rep, reference in zip(value['replicates'], exemplar['replicates']):
            for label in CONDITIONS:
                if rep['metrics']['per_condition'][str(label)]['n_real'] != reference['metrics']['per_condition'][str(label)]['n_real']:
                    raise ValueError('Evaluation real condition counts differ')
    histories, history_paths = {}, []
    for method in METHODS[:2]:
        for seed in SEEDS:
            directory = runs / method / f'seed_{seed}'
            manifest, history = _read(directory / 'manifest.json'), _read(directory / 'history.json')
            if manifest.get('status') != 'completed' or manifest.get('completed_epochs') != manifest['config']['epochs']:
                raise ValueError(f'Incomplete training run: {method}, seed {seed}')
            if [row['epoch'] for row in history] != list(range(1, manifest['completed_epochs'] + 1)):
                raise ValueError('Training history is incomplete')
            if _sha(directory / 'history.json') != manifest.get('artifact_hashes', {}).get('history.json'):
                raise ValueError('Training history hash mismatch')
            if not any(row['validation_sw1'] is not None for row in history):
                raise ValueError('Training history has no validation scores')
            for row in history:
                if row['validation_sw1'] is not None:
                    _number(row['validation_sw1'], 'validation SW1')
            report_hash = reports[(method, seed)].get('checkpoint_sha256')
            if report_hash is not None and report_hash != manifest['artifact_hashes'].get('best.pt'):
                raise ValueError('Evaluated checkpoint differs from the selected run artifact')
            histories[(method, seed)] = (history, manifest['best_epoch'])
            history_paths += [directory / 'history.json', directory / 'manifest.json']
    inference_path = evaluations / 'inference.json'
    if not inference_path.exists() and evaluations.name == 'reports':
        inference_path = evaluations.parent / 'inference.json'
    inference = _read(inference_path) if inference_path.exists() else None
    if inference is not None:
        inference_protocol = inference.get('provenance', {}).get('protocol', {}).get('sha256')
        if not inference_protocol or inference_protocol != exemplar.get('protocol_sha256'):
            raise ValueError('Inference protocol differs from evaluation reports')
        for name, expected_hash in inference.get('evaluation_report_hashes', {}).items():
            matches = [path for path in paths.values() if name in (path.name, path.stem)]
            if len(matches) != 1 or _sha(matches[0]) != expected_hash:
                raise ValueError('Inference evaluation report hash mismatch')
    return reports, paths, histories, history_paths, inference, inference_path


def _csv(path, rows):
    fields = list(dict.fromkeys(field for row in rows for field in row))
    if not fields:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tables(output, reports):
    metrics, draws, conditions, technologies = [], [], [], []
    for (method, seed), report in reports.items():
        identity = {'method': method, 'training_seed': seed, 'split': report['split']}
        for metric, values in report['macro_summary'].items():
            metrics.append({**identity, 'metric': metric, 'units': _units(metric),
                            'macro_mean': values['mean'], 'sampling_sd': values['sampling_sd'],
                            'valid_generation_replicates': values['valid_replicates']})
        for rep in report['replicates']:
            for metric, value in rep['metrics']['macro'].items():
                draws.append({**identity, 'generation_seed': rep['generation_seed'],
                              'metric': metric, 'units': _units(metric), 'macro_value': value})
        for label in CONDITIONS:
            entries = [r['metrics']['per_condition'][str(label)] for r in report['replicates']]
            row = {**identity, 'condition': label, 'technology': label // 8,
                   'heat_pump': label // 32, 'electric_vehicle': (label // 16) % 2,
                   'photovoltaic': (label // 8) % 2, 'weekend': (label // 4) % 2,
                   'season': ('winter', 'spring', 'summer', 'autumn')[label % 4],
                   'n_real': entries[0]['n_real'], 'n_fake_per_draw': entries[0]['n_fake']}
            for metric in entries[0]['metrics']:
                row[metric], row[f'{metric}_sampling_sd'] = _stats([e['metrics'][metric] for e in entries])
            row[PRIMARY] = row['profile_sw1_kwh'] / _denominator(report)
            row[f'{PRIMARY}_sampling_sd'] = (row['profile_sw1_kwh_sampling_sd'] / _denominator(report)
                                           if row['profile_sw1_kwh_sampling_sd'] is not None else None)
            conditions.append(row)
        for group in range(8):
            scores = [float(np.mean([rep['metrics']['per_condition'][str(label)]['metrics']['profile_sw1_kwh']
                                    for label in range(group * 8, (group + 1) * 8)])) / _denominator(report)
                      for rep in report['replicates']]
            mean, sd = _stats(scores)
            technologies.append({**identity, 'technology': group, 'n_calendar_conditions': 8,
                                 PRIMARY: mean, 'sampling_sd': sd})
    summary = []
    for method in METHODS:
        selected = [r for (m, _), r in reports.items() if m == method]
        values = [r['macro_summary'][PRIMARY]['mean'] for r in selected]
        mean, sd = _stats(values)
        summary.append({'method': method, 'mean_normalized_profile_sw1': mean,
                        'n_training_seeds': len(selected) if method in METHODS[:2] else None,
                        'training_seed_sd': sd if method in METHODS[:2] else None,
                        'sampling_sd': selected[0]['macro_summary'][PRIMARY]['sampling_sd'] if method in METHODS[2:] else None,
                        'n_generation_draws_per_model': 3,
                        'uncertainty': 'SD across five training-seed means' if method in METHODS[:2] else 'SD across three generation draws'})
    for filename, rows in [('metrics.csv', metrics), ('replicate_metrics.csv', draws),
                           ('condition_metrics.csv', conditions), ('technology_metrics.csv', technologies),
                           ('primary_summary.csv', summary)]:
        _csv(output / filename, rows)
    return technologies


def _style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#252525', 'text.color': '#252525',
                         'xtick.color': '#252525', 'ytick.color': '#252525',
                         'axes.linewidth': .8, 'lines.linewidth': 1.5,
                         'savefig.facecolor': 'white', 'pdf.fonttype': 42})


def _save(fig, output, name, note):
    fig.text(.02, .015, note, ha='left', va='bottom', fontsize=8)
    fig.savefig(output / f'{name}.png', dpi=180, bbox_inches='tight', pad_inches=.15)
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight', pad_inches=.15)
    plt.close(fig)


def _learning(output, histories):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True, sharey=True)
    seed_colors = list(COLORS[m] for m in METHODS)
    for ax, method in zip(axes, METHODS[:2]):
        for index, seed in enumerate(SEEDS):
            history, best_epoch = histories[(method, seed)]
            scored = [row for row in history if row['validation_sw1'] is not None]
            ax.plot([r['epoch'] for r in scored], [r['validation_sw1'] for r in scored],
                    color=seed_colors[index], marker=list(MARKERS.values())[index], markersize=3.5,
                    label=f'Seed {seed}')
            best = next(r for r in history if r['epoch'] == best_epoch)
            ax.plot(best_epoch, best['validation_sw1'], marker='*', color=seed_colors[index], markersize=10)
        ax.set(title=NAMES[method], xlabel='Training epoch')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y', color='#E5E5E5', linewidth=.6)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Normalized conditional SW1 (dimensionless)')
    axes[1].legend(loc='best', frameon=False, fontsize=8)
    fig.suptitle('Validation checkpoint selection: all ten training runs', y=.98)
    fig.subplots_adjust(bottom=.23, top=.84, wspace=.12)
    _save(fig, output, 'learning_curves',
          'Fixed validation rows and generated draws for every run. Lower is better. Stars mark selected epochs; only saved selection epochs are plotted.')


def _point(ax, method, position, points):
    offsets = np.linspace(-.12, .12, len(points)) if len(points) > 1 else np.zeros(1)
    ax.scatter(points, position + offsets, color=COLORS[method],
               marker=MARKERS[method], s=28, alpha=.8, zorder=3)
    if len(points) == 1:
        return
    mean, sd = _stats(points)
    ax.errorbar(mean, position, xerr=sd, fmt='D', color='#252525', markersize=4,
                capsize=3, elinewidth=1.2, zorder=4)


def _primary(output, reports):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True, width_ratios=(1, 1))
    for position, method in enumerate(METHODS[:2]):
        _point(axes[0], method, position, [reports[(method, seed)]['macro_summary'][PRIMARY]['mean'] for seed in SEEDS])
    for position, method in enumerate(METHODS[2:]):
        _point(axes[1], method, position, [r['metrics']['macro'][PRIMARY] for r in reports[(method, None)]['replicates']])
    for ax, methods, title in zip(axes, (METHODS[:2], METHODS[2:]),
                                  ('Five neural training seeds per arm', 'Three generation draws per baseline')):
        ax.set_yticks(range(len(methods)), [NAMES[m] for m in methods])
        ax.set_title(title)
        ax.set_xlabel('Condition-macro normalized SW1 (dimensionless)')
        ax.set_xlim(left=0)
        ax.set_ylim(len(methods) - .5, -.5)
        ax.grid(axis='x', color='#E5E5E5', linewidth=.6)
    fig.suptitle(f'Primary endpoint on the saved {next(iter(reports.values()))["split"]} evaluation', y=.98)
    fig.subplots_adjust(bottom=.27, top=.82, wspace=.68)
    _save(fig, output, 'primary_sw1',
          'Left: each point averages three generation draws; black diamond ±1 training-seed SD (n=5).\n'
          'Right: each point is one generation draw; black diamond ±1 sampling SD (n=3). These spreads are different quantities; neither is a confidence interval.')


def _technology(output, technologies):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5), sharex=True, sharey=True)
    for group, ax in enumerate(axes.flat):
        for position, method in enumerate(METHODS):
            rows = [r for r in technologies if r['technology'] == group and r['method'] == method]
            points = [r[PRIMARY] for r in rows]
            _point(ax, method, position, points)
        ax.set(title=f'HP={group // 4}, EV={(group // 2) % 2}, PV={group % 2}')
        ax.set_yticks(range(5), [NAMES[m] for m in METHODS], fontsize=8)
        ax.set_ylim(4.5, -.5)
        ax.set_xlim(left=0)
        ax.grid(axis='x', color='#E5E5E5', linewidth=.6)
        if group >= 4:
            ax.set_xlabel('Normalized SW1', fontsize=9)
    fig.suptitle('Technology groups: equal weight across eight calendar conditions', y=.98)
    fig.subplots_adjust(left=.14, bottom=.2, top=.88, hspace=.28, wspace=.18)
    _save(fig, output, 'technology_errors',
          'Each technology group averages weekday/weekend × four seasons, then three generation draws. GAN points show all five training seeds;\n'
          'black diamonds ±1 training-seed SD. Baseline points show three-draw means (no training-seed interval). SW1 is dimensionless.\n'
          'HP: heat pump; EV: electric vehicle; PV: photovoltaic. Lower errors are better.')


def _profiles(output, reports):
    methods = ('conditional', 'resample', 'copula')
    selected = {method: next(rep for rep in reports[(method, 11 if method == 'conditional' else None)]['replicates']
                              if rep['generation_seed'] == 31001) for method in methods}
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5), sharex=True, sharey=True)
    for group, (label, ax) in enumerate(zip(REPRESENTATIVE, axes.flat)):
        reference = selected['conditional']['curves'][str(label)]
        for method in ('real', *methods):
            curve = reference if method == 'real' else selected[method]['curves'][str(label)]
            for field in ('real_mean_kwh', 'real_quantiles_kwh'):
                if not np.allclose(curve[field], reference[field], rtol=1e-12, atol=1e-12):
                    raise ValueError('Representative profile reports disagree on real curves')
            probabilities = curve['quantile_probabilities']
            low = next(i for i, p in enumerate(probabilities) if np.isclose(p, .05))
            high = next(i for i, p in enumerate(probabilities) if np.isclose(p, .95))
            prefix = 'real' if method == 'real' else 'fake'
            mean = np.asarray(curve[f'{prefix}_mean_kwh'], dtype=float)
            quantiles = np.asarray(curve[f'{prefix}_quantiles_kwh'], dtype=float)
            if mean.shape != (24,) or quantiles.shape != (len(probabilities), 24) or not np.isfinite(mean).all() or not np.isfinite(quantiles).all():
                raise ValueError('Representative hourly curves must be finite [24] summaries')
            if np.any(quantiles[low] > quantiles[high]):
                raise ValueError('Invalid pointwise quantile bounds')
            ax.fill_between(range(24), quantiles[low], quantiles[high], color=COLORS[method], alpha=.08)
            ax.plot(range(24), mean, color=COLORS[method], linestyle=STYLES[method],
                    label='Real held-out' if method == 'real' else NAMES[method])
        ax.set(title=f'HP={group // 4}, EV={(group // 2) % 2}, PV={group % 2}; label {label}',
               xticks=[0, 6, 12, 18, 23], xlim=(0, 23))
        ax.grid(axis='y', color='#E5E5E5', linewidth=.6)
        if group >= 4:
            ax.set_xlabel('Local hour')
    fig.supylabel('Net hourly energy (kWh)', x=.015, fontsize=10)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=4, loc='upper center', bbox_to_anchor=(.5, .93), frameon=False)
    fig.suptitle('Fixed representative conditions: winter weekdays, every technology group', y=.99)
    fig.subplots_adjust(left=.08, bottom=.17, top=.8, hspace=.35, wspace=.14)
    _save(fig, output, 'representative_profiles',
          'Lines: hourly means; shading: pointwise 5th–95th profile percentiles (not confidence intervals). Shared scales across panels.\n'
          'Conditional GAN training seed 11; fixed generation seed 31001 for all methods. Conditions g×8 are fixed in advance, without outcome-based selection.')


def _inference(output, inference):
    intervals, training, proximity = [], [], []
    for comparator, result in inference.get('paired_cluster_comparisons', {}).items():
        for metric, observed in result['observed_difference'].items():
            bounds = result['ci95'][metric]
            intervals.append({'comparator': comparator, 'metric': metric, 'units': _units(metric),
                'conditional_minus_comparator': observed, 'ci95_lower': bounds[0] if bounds else None,
                'ci95_upper': bounds[1] if bounds else None,
                'n_bootstrap': result['metadata'].get('n_bootstrap'),
                'uncertainty_scope': result['metadata'].get('uncertainty_scope'),
                'ci_unavailable_reason': result['metadata'].get('ci_unavailable_reason')})
    for comparator, metrics in inference.get('paired_training_seed_differences', {}).items():
        for metric, result in metrics.items():
            for draw in result['per_training_seed']:
                training.append({'comparator': comparator, 'metric': metric, 'units': _units(metric), **draw,
                                 'mean_difference': result['mean_difference'],
                                 'training_seed_sd': result['training_seed_sd']})
    for method, result in inference.get('nearest_neighbor_diagnostics', {}).items():
        for condition, entry in result['per_condition'].items():
            for sample in ('generated_to_train', 'heldout_to_train'):
                values = entry[sample]
                proximity.append({'method': method, 'condition': condition, 'sample': sample,
                                  'n_train': entry['n_train'],
                                  **{key: value for key, value in values.items() if key != 'quantiles_rmse_kwh'},
                                  **{f'rmse_kwh_quantile_{key}': value for key, value in values['quantiles_rmse_kwh'].items()}})
    for name, rows in [('paired_cluster_intervals.csv', intervals), ('paired_training_seed_differences.csv', training),
                       ('nearest_neighbor_diagnostics.csv', proximity)]:
        _csv(output / name, rows)
    if not intervals:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), sharey=True)
    for ax, metric, title in zip(axes, (PRIMARY, 'daily_energy_w1_kwh'),
                                 ('Normalized profile SW1 difference', 'Daily-energy W1 difference (kWh)')):
        for index, comparator in enumerate(('resample', 'copula')):
            row = next(r for r in intervals if r['comparator'] == comparator and r['metric'] == metric)
            if row['ci95_lower'] is not None:
                ax.hlines(index, row['ci95_lower'], row['ci95_upper'], color=COLORS[comparator], linewidth=2)
            ax.scatter(row['conditional_minus_comparator'], index, color=COLORS[comparator],
                       marker=MARKERS[comparator], zorder=3)
        ax.axvline(0, color='#555555', linestyle='--', linewidth=.8)
        ax.set(title=title, xlabel='Conditional GAN minus comparator; positive = worse',
               yticks=[0, 1], yticklabels=[NAMES['resample'], NAMES['copula']], ylim=(1.5, -.5))
    exemplar = inference.get('exemplar', {})
    fig.suptitle('Paired technology-stratified meter-cluster 95% intervals', y=.98)
    fig.subplots_adjust(left=.16, bottom=.3, top=.76, wspace=.3)
    _save(fig, output, 'paired_cluster_differences',
          f'Fixed training seed {exemplar.get("training_seed", "unreported")}, generation seed {exemplar.get("generation_seed", "unreported")}. '
          'Intervals resample whole real meter histories within technology.\n'
          'They exclude training-seed, generation-draw and future-year/weather uncertainty. Unavailable intervals remain blank in the CSV.')


def render(runs, evaluations, output):
    runs, evaluations, output = Path(runs).resolve(), Path(evaluations).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output must be a new directory: {output}')
    reports, paths, histories, history_paths, inference, inference_path = _inputs(runs, evaluations)
    output.parent.mkdir(parents=True, exist_ok=True)
    _style()
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        stage = Path(temporary)
        technologies = _tables(stage, reports)
        _learning(stage, histories)
        _primary(stage, reports)
        _technology(stage, technologies)
        _profiles(stage, reports)
        if inference is not None:
            _inference(stage, inference)
        source_paths = [*paths.values(), *history_paths, *([inference_path] if inference is not None else [])]
        manifest = {'schema_version': 1, 'source_access': 'saved JSON reports only; no profile arrays or checkpoints',
                    'evaluation_reports': [str(p) for p in paths.values()],
                    'source_json_sha256': {str(path): _sha(path) for path in source_paths},
                    'renderer_sha256': _sha(Path(__file__)), 'matplotlib': matplotlib.__version__,
                    'split': next(iter(reports.values()))['split'],
                    'training_seeds': list(SEEDS), 'generation_seeds': [31001, 31002, 31003],
                    'representative_conditions': list(REPRESENTATIVE), 'profile_training_seed': 11,
                    'profile_generation_seed': 31001, 'inference_included': inference is not None,
                    'uncertainty': {'neural': 'SD across five training-seed means, each averaging three generation draws',
                                    'baseline': 'sampling SD across three generation draws; no training-seed SD',
                                    'profile_bands': 'pointwise 5th–95th profile percentiles, not confidence intervals',
                                    'cluster_intervals': 'paired fixed-model/fixed-draw whole-real-meter percentile 95%'},
                    'output_sha256': {path.name: _sha(path) for path in sorted(stage.iterdir())}}
        (stage / 'render_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--evaluations', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = render(args.runs, args.evaluations, args.output)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print(f'render_phase1: error: {error}', file=sys.stderr)
        return 2
    print(f'Rendered {len(manifest["evaluation_reports"])} saved evaluations to {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
