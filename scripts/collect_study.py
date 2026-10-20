"""Collect completed validation runs without recomputing or selecting results.

Each --run is NAME=PATH_TO_SEED_DIRECTORY. All attempted arms should be included;
the collector never ranks, filters, averages away failures, or reads test data.
"""
import argparse
import json
from pathlib import Path

from fluvius_cgan.release import sha256_file


def collect(name, directory):
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    report = json.loads((directory / 'validation_report.json').read_text())
    if report['split'] != 'val' or summary['test_used']:
        raise ValueError('This collector is restricted to validation-only runs')
    if report['checkpoint_sha256'] != sha256_file(directory / 'best.pt'):
        raise ValueError('The evaluated checkpoint has changed')
    history = json.loads((directory / 'history.json').read_text())
    if len(history) != summary['epochs_completed']:
        raise ValueError('History and completed epoch counts disagree')
    selection = {k: v for k, v in summary['selection'].items() if k != 'validation_indices'}
    primary = ('pointwise_wasserstein_kwh', 'daily_energy_wasserstein_kwh',
               'peak_power_wasserstein_kw', 'ramp_wasserstein_kw', 'diversity_ratio',
               'mean_profile_rmse_kwh', 'pointwise_variance_rmse_kwh2')
    methods = {method: {'macro': values['macro']} for method, values in report['methods'].items()}
    methods['cgan']['per_condition'] = {
        label: {**{key: values[key] for key in primary},
                'acf_wasserstein': {lag: acf['wasserstein']
                                    for lag, acf in values['autocorrelation'].items()}}
        for label, values in report['methods']['cgan']['per_condition'].items()}
    return {
        'name': name, 'config': config, 'checkpoint_sha256': report['checkpoint_sha256'],
        'best_epoch': summary['best_epoch'], 'epochs_completed': summary['epochs_completed'],
        'selection_metric': summary['validation_metric'],
        'best_selection_score': summary['best_validation_score'],
        'training_selection': selection, 'scaler': summary['scaler'],
        'data_hashes': summary['data_hashes'], 'provenance': summary['provenance'],
        'elapsed_seconds': summary['elapsed_seconds'],
        'generator_updates': summary.get('generator_updates', sum(
            row.get('generator_updates', (selection['train_profiles_used'] + config['batch_size'] - 1)
                    // config['batch_size']) for row in history)),
        'class_exposure_counts': summary.get('class_exposure_counts'),
        'evaluation': {k: v for k, v in report.items() if k != 'methods'},
        'methods': methods,
        'history': [{k: v for k, v in row.items() if k != 'class_exposure_counts'} for row in history],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='NAME=SEED_DIRECTORY')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()
    pairs = [item.split('=', 1) for item in args.run]
    if any(len(pair) != 2 or not all(pair) for pair in pairs):
        parser.error('Each run must be NAME=SEED_DIRECTORY')
    if len({pair[0] for pair in pairs}) != len(pairs):
        parser.error('Run names must be unique')
    report = {
        'schema_version': 1, 'split': 'val', 'test_used': False,
        'source_trace_independence': 'See docs/DUPLICATE_REMEDIATION.md: original-split screens '
                                    'contain detected exact full-trace train/validation aliases.',
        'interpretation': 'Exploratory method-screening evidence, including unsuccessful arms. '
                          'Compare only arms with matched declared budgets, samplers and selection criteria. '
                          'Not a final test, significance claim, or publication-readiness certificate.',
        'runs': [collect(name, directory) for name, directory in pairs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w' if args.replace else 'x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(f'Collected {len(pairs)} validation runs: {args.output}')


if __name__ == '__main__':
    main()
