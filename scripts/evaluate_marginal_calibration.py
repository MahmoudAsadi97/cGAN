#!/usr/bin/env python3
"""Evaluate frozen TRAIN-only marginal calibration on existing development draws.

This is a separate adapted generator, not a revised score for the reference
cGAN. Source CDFs use independent fixed generator draws; target quantiles use
inner TRAIN only. Checkpoints, calibration settings and evaluation draws are
declared before scoring. No original TEST profiles are read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import re
import subprocess
import time

import numpy as np
import scipy
import torch

from fluvius_cgan import development
from fluvius_cgan.baselines import fit_baseline
from fluvius_cgan.calibration import fit_calibration, load_calibration
from fluvius_cgan.training import _json, _read_data, _sha256

import evaluate_development_candidates as diagnostics


ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = {'train.npz', 'val.npz', 'scaler.json', 'metadata.json'}
CALIBRATION_RULES = {'source_cdf': 'deterministic_mid_ecdf',
                     'target_inverse': 'empirical_step',
                     'constant_source_policy': 'fail_if_target_varies'}
EXPORT_METRICS = ('export_cell_fraction', 'export_day_fraction',
                  'negative_daily_energy_fraction', 'mean_export_magnitude_kwh_per_day')


def _seed(value):
    return type(value) is int and 0 <= value < 2**32


def _plan(path):
    plan = diagnostics._read(path)
    if plan.get('schema_version') != 1 or plan.get('phase') != 1:
        raise ValueError('Expected a phase-1 calibration plan with schema_version 1')
    calibration = plan.get('calibration', {})
    if (set(calibration) != set(CALIBRATION_RULES) | {'seed', 'profiles_per_condition'}
            or any(calibration[key] != value for key, value in CALIBRATION_RULES.items())):
        raise ValueError('Unsupported calibration rules')
    if not _seed(calibration['seed']):
        raise ValueError('Calibration seed must be an unsigned 32-bit integer')
    count = calibration['profiles_per_condition']
    if type(count) is not int or count < 2:
        raise ValueError('profiles_per_condition must be an integer of at least 2')
    seeds = plan.get('generation_seeds')
    if (not isinstance(seeds, list) or not seeds or not all(_seed(seed) for seed in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError('Generation seeds must be distinct unsigned 32-bit integers')
    if calibration['seed'] in seeds:
        raise ValueError('Calibration seed must differ from every evaluation seed')
    hashes = plan.get('data_hashes', {})
    if set(hashes) != DATA_FILES:
        raise ValueError('Plan data hashes must bind the four development files')
    data = Path(plan['data']).resolve()
    for name, digest in hashes.items():
        if _sha256(data / name) != digest:
            raise ValueError(f'Plan data hash mismatch: {name}')
    entries = plan.get('candidates')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Plan requires a nonempty candidate list')
    paths, checkpoints, reference_ids = set(), set(), set()
    for entry in entries:
        path = Path(entry['run_directory']).resolve()
        if not re.fullmatch(r'candidate_[0-9]{3}', entry.get('reference_method_id', '')):
            raise ValueError('Invalid reference_method_id')
        if (path in paths or entry['checkpoint_sha256'] in checkpoints
                or entry['reference_method_id'] in reference_ids):
            raise ValueError('Candidate directories, checkpoints and reference IDs must be distinct')
        paths.add(path)
        checkpoints.add(entry['checkpoint_sha256'])
        reference_ids.add(entry['reference_method_id'])
        if _sha256(path / 'best.pt') != entry['checkpoint_sha256']:
            raise ValueError(f'Plan checkpoint hash mismatch: {path}')
        if _sha256(path / 'manifest.json') != entry['manifest_sha256']:
            raise ValueError(f'Plan candidate manifest hash mismatch: {path}')
    reference = Path(plan['reference']['directory']).resolve()
    if _sha256(reference / 'manifest.json') != plan['reference']['manifest_sha256']:
        raise ValueError('Plan reference manifest hash mismatch')
    return plan


def _reference(plan, candidates, hashes, selection, scaler, real, labels):
    directory = Path(plan['reference']['directory']).resolve()
    required = ['summary.json', 'baseline_mean.json', 'baseline_resample.json', 'baseline_copula.json']
    required.extend(f"{entry['reference_method_id']}.json" for entry in plan['candidates'])
    manifest = diagnostics._verify(directory, required)
    diagnostics._shared(manifest, hashes, selection, scaler)
    if (manifest.get('schema_version') != 1 or manifest.get('test_performance_access') is not False
            or manifest.get('real_profile_bytes_sha256') != diagnostics._digest(real, '<f8')
            or manifest.get('label_bytes_sha256') != diagnostics._digest(labels, '<i8')):
        raise ValueError('Reference provenance or selection array hashes differ')
    reports = {}
    for entry, (record, _, _) in zip(plan['candidates'], candidates):
        method_id = entry['reference_method_id']
        report = diagnostics._read(directory / f'{method_id}.json')
        identity_fields = ('method', 'checkpoint_sha256', 'manifest_sha256', 'training_seed',
                           'selected_step', 'completed_steps', 'config', 'training_source_sha256')
        if (report.get('method_id') != method_id
                or any(report.get(key) != record[key] for key in identity_fields)
                or Path(report.get('run_directory', '')).resolve() != Path(record['run_directory'])):
            raise ValueError(f'Reference candidate identity differs: {method_id}')
        reports[record['method_id']] = report
    for kind in ('mean', 'resample', 'copula'):
        method_id = f'baseline_{kind}'
        report = diagnostics._read(directory / f'{method_id}.json')
        if report.get('method_id') != method_id or report.get('method') != kind:
            raise ValueError('Reference baseline method identity differs')
        reports[method_id] = report
    draws = {}
    for method_id, report in reports.items():
        for seed in plan['generation_seeds']:
            matches = [draw for draw in report['draws'] if draw['generation_seed'] == seed]
            if len(matches) != 1:
                raise ValueError('Reference must contain each evaluation draw exactly once')
            draws[method_id, seed] = matches[0]
    return draws, manifest


def _summary(report):
    draws = report['draws']
    return {'method': report['method'], 'training_seed': report['training_seed'],
        'metrics': {key: diagnostics._stats([draw['metrics']['macro'][key] for draw in draws])
                    for key in diagnostics.SUMMARY_METRICS},
        'tail_errors': {key: diagnostics._stats([draw['diagnostics']['macro'][key] for draw in draws])
                        for key in diagnostics.TAIL_ERRORS},
        'source_pv0': {side: {key: diagnostics._stats([
            draw['diagnostics']['source_pv0'][side][key] for draw in draws]) for key in EXPORT_METRICS}
                      for side in ('real', 'fake')},
        'quantized_exact_training_match_fraction': diagnostics._stats([
            draw['diagnostics']['copying']['quantized_exact_match_fraction'] for draw in draws])}


def evaluate(plan_path, output):
    """Fit and score every declared adaptation, retaining failures and controls."""
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = _plan(plan_path)
    data = Path(plan['data']).resolve()
    training, validation, scaler, metadata, hashes = _read_data(data)
    if hashes != plan['data_hashes']:
        raise ValueError('Plan data hashes changed while loading')
    if metadata.get('development', {}).get('source_partition_role') != 'original_train':
        raise ValueError('Requires inner development data sourced from original TRAIN')
    candidates = [diagnostics._candidate(Path(entry['run_directory']).resolve(), i)
                  for i, entry in enumerate(plan['candidates'])]
    selection = candidates[0][1]['selection']
    calibration, seeds = plan['calibration'], plan['generation_seeds']
    for record, manifest, _ in candidates:
        diagnostics._shared(manifest, hashes, selection, scaler)
        if manifest['selection']['generation_seed'] in [calibration['seed'], *seeds]:
            raise ValueError('Calibration and evaluation seeds must exclude every checkpoint selection seed')
    if selection.get('directions_seed') != 2026 or selection.get('directions_count') != 128:
        raise ValueError('Unsupported selection projection settings')
    indices = np.asarray(selection['indices'])
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
            or len(np.unique(indices)) != len(indices)
            or np.any((indices < 0) | (indices >= len(validation['x'])))):
        raise ValueError('Invalid saved selection indices')
    real = validation['x'][indices].astype(np.float64)
    labels = validation['y'][indices]
    if np.unique(labels).tolist() != selection['conditions']:
        raise ValueError('Selection condition coverage differs from data')
    reference_draws, reference_manifest = _reference(plan, candidates, hashes, selection, scaler, real, labels)
    source_labels = np.repeat(np.unique(training['y']), calibration['profiles_per_condition'])
    sources = [Path(__file__).resolve(), Path(diagnostics.__file__).resolve(),
               *sorted((ROOT / 'src' / 'fluvius_cgan').glob('*.py'))]
    manifest = {'schema_version': 1, 'phase': 1, 'status': 'running',
        'role': 'post-test development fixed conditional marginal calibration',
        'test_performance_access': False, 'reselection': False, 'clipping': False,
        'fitted_on': 'inner train and independent draws from frozen selected generators',
        'evaluated_on': 'previously used fixed inner validation rows',
        'plan_path': str(plan_path), 'plan_sha256': _sha256(plan_path),
        'data_directory': str(data), 'data_hashes': hashes, 'selection': selection,
        'real_profile_bytes_sha256': diagnostics._digest(real, '<f8'),
        'label_bytes_sha256': diagnostics._digest(labels, '<i8'),
        'calibration': calibration, 'generation_seeds': seeds,
        'generation_seed_role': 'Paired sampling draws; not new meters or training replicates.',
        'candidates': [record for record, _, _ in candidates],
        'reference': plan['reference'], 'reference_schema_version': reference_manifest['schema_version'],
        'reference_source_sha256': reference_manifest['source_sha256'],
        'metric_normalization': {'fitted_on': 'train', 'minimum': scaler['minimum'], 'maximum': scaler['maximum']},
        'source_sha256': {str(path.relative_to(ROOT)): _sha256(path) for path in sources},
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                                   capture_output=True, text=True).stdout.strip(),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'torch': str(torch.__version__), 'generation_device': 'cpu', 'torch_threads': 1,
            'thread_environment': {key: os.environ.get(key) for key in
                                   ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}},
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'completed_draws': 0}
    output.mkdir(parents=True, exist_ok=False)
    _json(output / 'plan.json', plan)
    _json(output / 'manifest.json', manifest)
    previous_threads, previous_rng = torch.get_num_threads(), torch.get_rng_state().clone()
    torch.set_num_threads(1)
    summary = {'schema_version': 1, 'role': manifest['role'], 'generation_seeds': seeds,
        'selection_profiles': len(real), 'conditions': selection['conditions'],
        'calibration': calibration,
        'uncertainty_note': 'Sampling SD describes paired generation draws only. This is post-test development, not new independent validation.',
        'interpretation': 'Empirical inverse quantiles enforce observed hourly marginal support and truncate unseen tails. They do not establish joint profile support, privacy, publication readiness, or recovered appliance labels.',
        'methods': {}}
    rows, condition_rows = [], []
    try:
        index = diagnostics.CopyIndex(training['x'])
        manifest['copy_reference'] = {'partition': 'full inner TRAIN', 'rows': index.training_rows,
                                      'unique_quantized_rows': len(index.rows)}
        _json(output / 'manifest.json', manifest)

        def score(report, seed, fake, elapsed, reference_id=None, mapping=None):
            scores = development.physical_metrics(real, fake, labels, scaler['maximum'] - scaler['minimum'])
            tails = diagnostics.tail_diagnostics(real, fake, labels, index.matches(fake))
            generated_hash = diagnostics._digest(fake, '<f8')
            if reference_id is not None:
                previous = reference_draws[reference_id, seed]
                if (previous['generated_profile_bytes_sha256'] != generated_hash
                        or previous['metrics'] != scores):
                    raise ValueError(f'Regenerated reference draw differs: {reference_id}, seed {seed}')
            draw = {'generation_seed': seed, 'generation_seconds': elapsed,
                'generated_profile_bytes_sha256': generated_hash, 'metrics': scores, 'diagnostics': tails,
                'reference_draw_match': reference_id is not None}
            if mapping is not None:
                draw['calibration_diagnostics'] = mapping
            report['draws'].append(draw)
            _json(output / f"{report['method_id']}.json", report)
            row = {key: report[key] for key in ('method_id', 'method', 'training_seed')}
            row.update(generation_seed=seed, **scores['macro'], **tails['macro'],
                       quantized_exact_training_match_fraction=tails['copying']['quantized_exact_match_fraction'])
            row.update({f'pv0_{side}_{key}': tails['source_pv0'][side][key]
                        for side in ('real', 'fake') for key in EXPORT_METRICS})
            rows.append(row)
            for label, values in tails['per_condition'].items():
                condition_rows.append({key: report[key] for key in ('method_id', 'method', 'training_seed')} |
                    {'generation_seed': seed, 'condition': int(label), 'source_pv_flag': values['source_pv_flag'],
                     **values['errors'],
                     'quantized_exact_training_match_fraction': values['quantized_exact_training_match_fraction'],
                     **{f'{side}_{key}': values[side][key] for side in ('real', 'fake') for key in EXPORT_METRICS}})
            manifest['completed_draws'] += 1
            _json(output / 'manifest.json', manifest)
            print(f"{report['method_id']} draw={seed} normalized_SW1="
                  f"{scores['macro']['normalized_profile_sw1']:.8f}", flush=True)

        for record, _, generate in candidates:
            prefix = record['method_id']
            started = time.perf_counter()
            source = generate(Path(record['checkpoint']), source_labels, calibration['seed'])
            generation_seconds = time.perf_counter() - started
            started = time.perf_counter()
            fitted = fit_calibration(training['x'], training['y'], source, source_labels)
            fit_seconds = time.perf_counter() - started
            calibration_path = output / f'{prefix}_calibration.npz'
            started = time.perf_counter()
            fitted.save(calibration_path)
            calibration_sha256 = _sha256(calibration_path)
            fitted = load_calibration(calibration_path, calibration_sha256)
            serialization_seconds = time.perf_counter() - started
            fit = {'schema_version': 1, 'fitted_on': 'inner train',
                'source_fit': 'Independent generator draws from the frozen selected checkpoint; no evaluation-batch ranks.',
                'candidate': record, 'checkpoint_sha256': record['checkpoint_sha256'],
                'data_hashes': hashes, 'calibration_seed': calibration['seed'],
                'calibration': calibration, 'source_rows': len(source), 'target_rows': len(training['x']),
                'source_profiles_per_condition': calibration['profiles_per_condition'],
                'target_profiles_per_condition': {str(int(label)): int(np.sum(training['y'] == label))
                                                  for label in np.unique(training['y'])},
                'source_profile_bytes_sha256': diagnostics._digest(source, '<f8'),
                'source_label_bytes_sha256': diagnostics._digest(source_labels, '<i8'),
                'target_profile_bytes_sha256': diagnostics._digest(training['x'], '<f8'),
                'target_label_bytes_sha256': diagnostics._digest(training['y'], '<i8'),
                'calibration_artifact': calibration_path.name, 'calibration_sha256': calibration_sha256,
                'source_generation_seconds': generation_seconds, 'fit_seconds': fit_seconds,
                'save_and_verified_reload_seconds': serialization_seconds,
                'mapping': 'Q_inner_train((count(source < x) + count(source <= x))/(2*m)); Q uses ceil(n*u)-1 with endpoint index handling.',
                'limitation': 'Deterministic ties; no marginal extrapolation. Weak coordinate ordering is retained, but new ties can alter rank correlations and hourly extrema can form unrealistic joint profiles.'}
            _json(output / f'{prefix}_fit.json', fit)
            del source
            raw = dict(record, method_id=f'{prefix}_raw', draws=[], calibration_applied=False)
            calibrated = dict(record, method_id=f'{prefix}_calibrated',
                method=f"{record['method']}_marginal_calibration", draws=[], calibration_applied=True,
                calibration_artifact=calibration_path.name, calibration_sha256=calibration_sha256)
            for seed in seeds:
                started = time.perf_counter()
                fake = generate(Path(record['checkpoint']), labels, seed)
                elapsed = time.perf_counter() - started
                score(raw, seed, fake, elapsed, reference_id=prefix)
                started = time.perf_counter()
                mapped, mapping = fitted.apply(fake, labels)
                mapping_seconds = time.perf_counter() - started
                score(calibrated, seed, mapped, elapsed + mapping_seconds, mapping=mapping)
                calibrated['draws'][-1]['mapping_seconds'] = mapping_seconds
                _json(output / f'{prefix}_calibrated.json', calibrated)
            summary['methods'][raw['method_id']] = _summary(raw)
            summary['methods'][calibrated['method_id']] = _summary(calibrated)
        for kind in ('mean', 'resample', 'copula'):
            model = fit_baseline(kind, training['x'], training['y'])
            report = {'method_id': f'baseline_{kind}', 'method': kind, 'training_seed': None,
                      'fitted_on': 'inner train', 'draws': []}
            for seed in seeds:
                started = time.perf_counter()
                fake = model.sample(labels, seed)
                elapsed = time.perf_counter() - started
                score(report, seed, fake, elapsed, reference_id=report['method_id'])
            summary['methods'][report['method_id']] = _summary(report)
        _json(output / 'summary.json', summary)
        diagnostics._csv(output / 'metrics.csv', rows)
        diagnostics._csv(output / 'condition_diagnostics.csv', condition_rows)
        manifest.update(status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat(),
            artifact_hashes={path.name: _sha256(path) for path in sorted(output.iterdir())
                             if path.is_file() and path.name != 'manifest.json'})
        _json(output / 'manifest.json', manifest)
        return summary
    except Exception as error:
        manifest.update(status='failed', finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        error=f'{type(error).__name__}: {error}', artifact_hashes={})
        # Bind partial maps and reports as well as completed draws. Receipt
        # failures must not replace the original scientific or runtime error.
        try:
            for path in sorted(output.iterdir()):
                if path.is_file() and path.name != 'manifest.json':
                    try:
                        manifest['artifact_hashes'][path.name] = _sha256(path)
                    except Exception as receipt_error:
                        manifest.setdefault('artifact_hash_errors', {})[path.name] = (
                            f'{type(receipt_error).__name__}: {receipt_error}')
            _json(output / 'manifest.json', manifest)
        except Exception as receipt_error:
            error.add_note(f'Could not finish the failure receipt: {type(receipt_error).__name__}: {receipt_error}')
        raise
    finally:
        torch.set_num_threads(previous_threads)
        torch.set_rng_state(previous_rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.plan, args.output)


if __name__ == '__main__':
    main()
