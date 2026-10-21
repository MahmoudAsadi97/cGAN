#!/usr/bin/env python3
"""Paired stronger-copula controls and exact same-condition TRAIN proximity.

No checkpoint or calibration is refitted. Historical neural draws are
regenerated and checked before adding descriptive nearest-TRAIN diagnostics.
All new baseline fitting uses the original-TRAIN-derived inner fitting split.
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
from fluvius_cgan.calibration import load_calibration
from fluvius_cgan.proximity import ConditionalNearestNeighbors
from fluvius_cgan.strong_baselines import fit_strong_baseline
from fluvius_cgan.training import _json, _read_data, _sha256

import evaluate_development_candidates as diagnostics
import evaluate_marginal_calibration as calibration_evaluation


ROOT = Path(__file__).resolve().parents[1]
PROBABILITIES = [.01, .05, .5, .95, .99]
NN_METRICS = ('mean_rms_kwh', 'mean_normalized_rms', 'below_real_p05_fraction',
              'below_real_median_fraction')


def _plan(path):
    plan = diagnostics._read(path)
    if (type(plan.get('schema_version')) is not int or plan['schema_version'] != 1
            or type(plan.get('phase')) is not int or plan['phase'] != 1):
        raise ValueError('Requires a phase-1 strong-baseline plan with schema_version 1')
    seeds = plan.get('generation_seeds')
    if (not isinstance(seeds, list) or not seeds
            or not all(calibration_evaluation._seed(seed) for seed in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError('Generation seeds must be distinct unsigned 32-bit integers')
    if set(plan.get('data_hashes', {})) != calibration_evaluation.DATA_FILES:
        raise ValueError('Plan must bind the four development data hashes')
    data = Path(plan['data']).resolve()
    for name, expected in plan['data_hashes'].items():
        if _sha256(data / name) != expected:
            raise ValueError(f'Plan data hash mismatch: {name}')
    methods = plan.get('methods')
    if not isinstance(methods, list) or not methods:
        raise ValueError('Declare a nonempty list of stronger baseline methods')
    identifiers, specifications = set(), set()
    for entry in methods:
        if not isinstance(entry, dict) or not re.fullmatch(r'[a-z][a-z0-9_]*', entry.get('method_id', '')):
            raise ValueError('Invalid stronger-baseline method_id')
        kind = entry.get('kind')
        if kind == 'beta':
            if set(entry) != {'method_id', 'kind', 'fit_seed'}:
                raise ValueError('Beta baseline requires only method_id, kind and fit_seed')
            if not calibration_evaluation._seed(entry['fit_seed']) or entry['fit_seed'] in seeds:
                raise ValueError('Beta fit seed must be unsigned 32-bit and distinct from evaluation seeds')
            specification = kind, entry['fit_seed']
        elif kind == 'student_t':
            df = entry.get('df')
            if (set(entry) != {'method_id', 'kind', 'df'} or type(df) not in (int, float)
                    or not np.isfinite(df) or df <= 2):
                raise ValueError('Student-t baseline df must be finite and greater than 2')
            specification = kind, df
        else:
            raise ValueError('Unknown stronger baseline kind')
        if entry['method_id'] in identifiers or specification in specifications:
            raise ValueError('Stronger baseline IDs and specifications must be distinct')
        identifiers.add(entry['method_id'])
        specifications.add(specification)
    for key in ('reference', 'calibration_reference'):
        record = plan[key]
        if _sha256(Path(record['directory']) / 'manifest.json') != record['manifest_sha256']:
            raise ValueError(f'{key} reference manifest hash mismatch')
    return plan


def _draws(report, seeds):
    result = {}
    for seed in seeds:
        matches = [draw for draw in report['draws'] if draw['generation_seed'] == seed]
        if len(matches) != 1:
            raise ValueError('Reference must contain each evaluation draw exactly once')
        result[seed] = matches[0]
    return result


def _prepare(plan, training, validation, hashes, scaler):
    directory = Path(plan['calibration_reference']['directory']).resolve()
    manifest = diagnostics._verify(directory, ('plan.json', 'summary.json'))
    if (manifest.get('schema_version') != 1 or manifest.get('test_performance_access') is not False
            or manifest.get('reselection') is not False or manifest.get('clipping') is not False):
        raise ValueError('Calibration reference provenance differs')
    # The artifact hash binds the saved plan. The original input-plan hash can
    # differ because its JSON was reformatted when copied into the run.
    original = calibration_evaluation._plan(directory / 'plan.json')
    if (original['data_hashes'] != hashes or original['calibration'] != manifest['calibration']
            or Path(original['reference']['directory']).resolve() != Path(plan['reference']['directory']).resolve()
            or original['reference']['manifest_sha256'] != plan['reference']['manifest_sha256']):
        raise ValueError('Calibration and original reference declarations differ')
    if manifest['calibration']['seed'] in plan['generation_seeds']:
        raise ValueError('Evaluation seeds must exclude the calibration fitting seed')
    candidates = [diagnostics._candidate(Path(entry['run_directory']).resolve(), i)
                  for i, entry in enumerate(original['candidates'])]
    if [record for record, _, _ in candidates] != manifest.get('candidates'):
        raise ValueError('Calibration reference candidate identity differs')
    selection = manifest['selection']
    for _, candidate_manifest, _ in candidates:
        diagnostics._shared(candidate_manifest, hashes, selection, scaler)
        if candidate_manifest['selection']['generation_seed'] in plan['generation_seeds']:
            raise ValueError('Evaluation seeds must exclude the checkpoint selection seed')
    diagnostics._shared(manifest, hashes, selection, scaler)
    if selection.get('directions_seed') != 2026 or selection.get('directions_count') != 128:
        raise ValueError('Unsupported selection projection settings')
    indices = np.asarray(selection['indices'])
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
            or len(np.unique(indices)) != len(indices)
            or np.any((indices < 0) | (indices >= len(validation['x'])))):
        raise ValueError('Invalid saved selection indices')
    real, labels = validation['x'][indices].astype(np.float64), validation['y'][indices]
    if (np.unique(labels).tolist() != selection['conditions']
            or manifest['real_profile_bytes_sha256'] != diagnostics._digest(real, '<f8')
            or manifest['label_bytes_sha256'] != diagnostics._digest(labels, '<i8')):
        raise ValueError('Calibration reference selection arrays differ')
    paired_plan = dict(original, generation_seeds=plan['generation_seeds'])
    reference_draws, reference_manifest = calibration_evaluation._reference(
        paired_plan, candidates, hashes, selection, scaler, real, labels)
    neural = []
    for record, _, generate in candidates:
        prefix = record['method_id']
        reports = []
        for suffix in ('raw', 'calibrated'):
            name = f'{prefix}_{suffix}'
            if f'{name}.json' not in manifest['artifact_hashes']:
                raise ValueError('Calibration reference omitted a required draw report')
            report = diagnostics._read(directory / f'{name}.json')
            identity = ('checkpoint_sha256', 'manifest_sha256', 'training_seed', 'config',
                        'selected_step', 'completed_steps', 'training_source_sha256')
            method = record['method'] + ('_marginal_calibration' if suffix == 'calibrated' else '')
            if (report.get('method_id') != name or report.get('method') != method
                    or report.get('calibration_applied') is not (suffix == 'calibrated')
                    or any(report.get(key) != record[key] for key in identity)):
                raise ValueError('Calibration draw report candidate identity differs')
            reference_draws.update({(name, seed): draw for seed, draw in
                                   _draws(report, plan['generation_seeds']).items()})
            reports.append(report)
        name = reports[1]['calibration_artifact']
        if (Path(name).name != name
                or manifest['artifact_hashes'].get(name) != reports[1]['calibration_sha256']):
            raise ValueError('Calibration artifact declaration differs')
        neural.append((record, generate, directory / name, reports[1]['calibration_sha256']))
    return real, labels, selection, neural, reference_draws, manifest, reference_manifest


def _proximity(values, labels, scale, reference=None):
    per_condition = {}
    for label in np.unique(labels):
        group = values[labels == label]
        quantiles = np.quantile(group, PROBABILITIES)
        row = {'profiles': len(group), 'mean_rms_kwh': float(group.mean()),
               'mean_normalized_rms': float(group.mean() / scale)}
        for probability, value in zip(PROBABILITIES, quantiles):
            row[f'q{round(100 * probability):02d}_rms_kwh'] = float(value)
            row[f'q{round(100 * probability):02d}_normalized_rms'] = float(value / scale)
        if reference is not None:
            thresholds = np.quantile(reference[labels == label], [.05, .5])
            row.update(real_p05_rms_kwh=float(thresholds[0]), real_median_rms_kwh=float(thresholds[1]),
                below_real_p05_fraction=float(np.mean(group < thresholds[0])),
                below_real_median_fraction=float(np.mean(group < thresholds[1])))
        per_condition[str(int(label))] = row
    fields = [key for key in next(iter(per_condition.values())) if key != 'profiles']
    return {'definition': 'Exact nearest same-condition inner-TRAIN RMS Euclidean distance across 24 hourly kWh coordinates.',
        'normalization_denominator_kwh': scale,
        'quantile_probabilities': PROBABILITIES, 'quantile_method': 'numpy linear interpolation',
        'threshold_comparison': 'Strictly below the corresponding condition real inner-validation distance quantile.',
        'interpretation': 'Descriptive proximity to fitting profiles, not a privacy guarantee or proof of novelty. Close realistic profiles may be legitimate; resampling has zero distance by construction.',
        'distance_bytes_sha256': diagnostics._digest(values, '<f8'),
        'per_condition': per_condition,
        'macro': {key: float(np.mean([row[key] for row in per_condition.values()])) for key in fields}}


def _summary(report):
    result = calibration_evaluation._summary(report)
    result['proximity'] = {key: diagnostics._stats([draw['proximity']['macro'][key] for draw in report['draws']])
                           for key in NN_METRICS}
    return result


def evaluate(plan_path, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = _plan(plan_path)
    training, validation, scaler, metadata, hashes = _read_data(Path(plan['data']))
    if hashes != plan['data_hashes']:
        raise ValueError('Plan data hashes changed while loading')
    if metadata.get('development', {}).get('source_partition_role') != 'original_train':
        raise ValueError('Requires inner development data sourced from original TRAIN')
    real, labels, selection, neural, previous, cal_manifest, reference_manifest = _prepare(
        plan, training, validation, hashes, scaler)
    sources = [Path(__file__).resolve(), Path(diagnostics.__file__).resolve(),
               Path(calibration_evaluation.__file__).resolve(), *sorted((ROOT / 'src/fluvius_cgan').glob('*.py'))]
    scale = scaler['maximum'] - scaler['minimum']
    manifest = {'schema_version': 1, 'phase': 1, 'status': 'running',
        'role': 'post-test development stronger copula controls and TRAIN proximity',
        'test_performance_access': False, 'reselection': False, 'clipping': False, 'calibration_refitted': False,
        'data_directory': str(Path(plan['data']).resolve()), 'data_hashes': hashes,
        'plan_path': str(plan_path), 'plan_sha256': _sha256(plan_path),
        'selection': selection, 'real_profile_bytes_sha256': diagnostics._digest(real, '<f8'),
        'label_bytes_sha256': diagnostics._digest(labels, '<i8'), 'generation_seeds': plan['generation_seeds'],
        'generation_seed_role': 'Paired development sampling draws, not new meters or training replicates.',
        'methods': plan['methods'], 'candidates': [record for record, _, _, _ in neural],
        'reference': plan['reference'], 'calibration_reference': plan['calibration_reference'],
        'reference_source_sha256': reference_manifest['source_sha256'],
        'calibration_reference_source_sha256': cal_manifest['source_sha256'],
        'metric_normalization': {'fitted_on': 'train', 'minimum': scaler['minimum'], 'maximum': scaler['maximum']},
        'nearest_neighbor': {'reference': 'same-condition full inner TRAIN', 'exact': True,
            'distance': 'sqrt(sum_hour((query-training)**2)/24)', 'max_query_block_rows': 128,
            'normalized_denominator_kwh': scale or 1., 'interpretation': 'Descriptive proximity; not privacy or novelty evidence.'},
        'source_sha256': {str(path.relative_to(ROOT)): _sha256(path) for path in sources},
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                                   capture_output=True, text=True).stdout.strip(),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'torch': str(torch.__version__), 'generation_device': 'cpu', 'torch_threads': 1,
            'thread_environment': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}},
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'completed_draws': 0}
    output.mkdir(parents=True, exist_ok=False)
    _json(output / 'plan.json', plan)
    _json(output / 'manifest.json', manifest)
    previous_threads, previous_rng = torch.get_num_threads(), torch.get_rng_state().clone()
    torch.set_num_threads(1)
    summary = {'schema_version': 1, 'role': manifest['role'], 'generation_seeds': plan['generation_seeds'],
        'selection_profiles': len(real), 'conditions': selection['conditions'],
        'uncertainty_note': 'Sampling SD across paired generation draws only; no newly independent validation or calibrated inferential claim.',
        'methods': {}}
    rows, condition_rows = [], []
    try:
        tick = time.perf_counter()
        nearest = ConditionalNearestNeighbors(training['x'], training['y'])
        real_distances = nearest.distances(real, labels)
        real_report = {'schema_version': 1, 'partition': 'fixed inner-validation selection',
            'reference_partition': 'full inner TRAIN', 'data_hashes': hashes,
            'real_profile_bytes_sha256': manifest['real_profile_bytes_sha256'],
            'label_bytes_sha256': manifest['label_bytes_sha256'],
            'nearest_neighbor_seconds_including_index': time.perf_counter() - tick,
            'proximity': _proximity(real_distances, labels, scale or 1.)}
        _json(output / 'real_proximity.json', real_report)
        summary['real_proximity'] = real_report['proximity']['macro']
        copies = diagnostics.CopyIndex(training['x'])
        manifest['nearest_neighbor']['training_rows'] = nearest.training_rows
        manifest['nearest_neighbor']['training_condition_counts'] = {str(k): v for k, v in nearest.counts.items()}
        _json(output / 'manifest.json', manifest)

        def score(report, seed, fake, generation_seconds, reference_id=None):
            tick = time.perf_counter()
            scores = development.physical_metrics(real, fake, labels, scale)
            tails = diagnostics.tail_diagnostics(real, fake, labels, copies.matches(fake))
            metric_seconds = time.perf_counter() - tick
            generated_hash = diagnostics._digest(fake, '<f8')
            if reference_id is not None:
                old = previous[reference_id, seed]
                if old['generated_profile_bytes_sha256'] != generated_hash or old['metrics'] != scores:
                    raise ValueError(f'Regenerated reference draw differs: {reference_id}, seed {seed}')
            tick = time.perf_counter()
            distances = nearest.distances(fake, labels)
            proximity = _proximity(distances, labels, scale or 1., real_distances)
            proximity_seconds = time.perf_counter() - tick
            draw = {'generation_seed': seed, 'generation_seconds': generation_seconds,
                'physical_metric_seconds': metric_seconds, 'nearest_neighbor_seconds': proximity_seconds,
                'generated_profile_bytes_sha256': generated_hash, 'metrics': scores,
                'diagnostics': tails, 'proximity': proximity, 'reference_draw_match': reference_id is not None}
            report['draws'].append(draw)
            _json(output / f"{report['method_id']}.json", report)
            identity = {key: report[key] for key in ('method_id', 'method', 'training_seed')}
            rows.append(identity | {'generation_seed': seed, **scores['macro'], **tails['macro'],
                **{f'nn_{key}': value for key, value in proximity['macro'].items()},
                'quantized_exact_training_match_fraction': tails['copying']['quantized_exact_match_fraction'],
                **{f'pv0_{side}_{key}': tails['source_pv0'][side][key] for side in ('real', 'fake')
                   for key in calibration_evaluation.EXPORT_METRICS}})
            for label, values in tails['per_condition'].items():
                condition_rows.append(identity | {'generation_seed': seed, 'condition': int(label),
                    'source_pv_flag': values['source_pv_flag'], **values['errors'],
                    **{f'nn_{key}': value for key, value in proximity['per_condition'][label].items()},
                    'quantized_exact_training_match_fraction': values['quantized_exact_training_match_fraction'],
                    **{f'{side}_{key}': values[side][key] for side in ('real', 'fake')
                       for key in calibration_evaluation.EXPORT_METRICS}})
            manifest['completed_draws'] += 1
            _json(output / 'manifest.json', manifest)
            print(f"{report['method_id']} draw={seed} normalized_SW1={scores['macro']['normalized_profile_sw1']:.8f} "
                  f"mean_TRAIN_RMS={proximity['macro']['mean_rms_kwh']:.6f}", flush=True)

        for record, generate, artifact, expected_hash in neural:
            fitted = load_calibration(artifact, expected_hash)
            prefix = record['method_id']
            raw = dict(record, method_id=f'{prefix}_raw', draws=[], calibration_applied=False)
            calibrated = dict(record, method_id=f'{prefix}_calibrated',
                method=f"{record['method']}_marginal_calibration", draws=[], calibration_applied=True,
                calibration_artifact=str(artifact), calibration_sha256=expected_hash)
            for seed in plan['generation_seeds']:
                tick = time.perf_counter()
                fake = generate(Path(record['checkpoint']), labels, seed)
                elapsed = time.perf_counter() - tick
                score(raw, seed, fake, elapsed, raw['method_id'])
                tick = time.perf_counter()
                mapped, _ = fitted.apply(fake, labels)
                mapping_seconds = time.perf_counter() - tick
                score(calibrated, seed, mapped, elapsed + mapping_seconds, calibrated['method_id'])
            summary['methods'][raw['method_id']] = _summary(raw)
            summary['methods'][calibrated['method_id']] = _summary(calibrated)
        for kind in ('mean', 'resample', 'copula'):
            tick = time.perf_counter()
            fitted = fit_baseline(kind, training['x'], training['y'])
            report = {'method_id': f'baseline_{kind}', 'method': kind, 'training_seed': None,
                'fitted_on': 'inner train', 'fit_seconds': time.perf_counter() - tick, 'draws': []}
            for seed in plan['generation_seeds']:
                tick = time.perf_counter()
                fake = fitted.sample(labels, seed)
                elapsed = time.perf_counter() - tick
                score(report, seed, fake, elapsed, report['method_id'])
            summary['methods'][report['method_id']] = _summary(report)
        for entry in plan['methods']:
            tick = time.perf_counter()
            arguments = {key: value for key, value in entry.items() if key not in ('kind', 'method_id')}
            fitted = fit_strong_baseline(entry['kind'], training['x'], training['y'], **arguments)
            fit_seconds = time.perf_counter() - tick
            fit_metadata = fitted.metadata()
            report = {'method_id': f"strong_{entry['method_id']}", 'method': fit_metadata['method'],
                'training_seed': None, 'fitted_on': 'inner train', 'specification': entry,
                'fit_seconds': fit_seconds, 'fit_metadata': fit_metadata, 'data_hashes': hashes, 'draws': []}
            _json(output / f"{report['method_id']}_fit.json", {key: value for key, value in report.items() if key != 'draws'})
            for seed in plan['generation_seeds']:
                tick = time.perf_counter()
                fake = fitted.sample(labels, seed)
                elapsed = time.perf_counter() - tick
                score(report, seed, fake, elapsed)
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
            error.add_note(f'Could not finish failure receipt: {type(receipt_error).__name__}: {receipt_error}')
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
