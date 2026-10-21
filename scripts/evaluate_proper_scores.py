#!/usr/bin/env python3
"""Plan-bound proper scores for every eligible inner-validation profile.

This runner archives IID generated ensembles, their nested prefixes, corrected
energy/variogram score vectors and calendar-balanced meter contributions. It
computes no intervals and performs no model or calibration selection. Reused
validation meters remain postselection development evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from fluvius_cgan.baselines import fit_baseline
from fluvius_cgan.calibration import load_calibration
from fluvius_cgan.evaluation import conditional_metrics, projection_directions
from fluvius_cgan.meter_uncertainty import meter_contributions
from fluvius_cgan.mixture_baselines import METHOD as MIXTURE_METHOD, load_mixture_baseline
from fluvius_cgan.proper_scores import fit_ensemble_scores
from fluvius_cgan.strong_baselines import fit_strong_baseline
from fluvius_cgan.training import _json, _read_data, _sha256

import evaluate_development_candidates as diagnostics

ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = {'train.npz', 'val.npz', 'scaler.json', 'metadata.json'}
SCORES = ('energy_score', 'variogram_score')
INTERPRETATION = (
    'Reused inner-validation postselection development evidence, conditional on fixed fitted models. '
    'Generation replicates assess finite-ensemble variation; they are not training replicates or new meters. '
    'Identical integer seeds across samplers do not establish matched random numbers. '
    'No intervals, selection adjustment, new-meter coverage or publication threshold is supplied.')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _seed(value):
    return type(value) is int and 0 <= value < 2**32


def _read(path):
    value = diagnostics._read(path)
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def _digest(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order='C')).hexdigest()


def _safe_name(value):
    return isinstance(value, str) and value not in ('', '.', '..') and Path(value).name == value


def _bound_hash(path, expected, description):
    if (not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected)
            or _sha256(path) != expected):
        raise ValueError(f'{description} hash mismatch: {path}')


def _plan(path):
    plan = _read(path)
    if any(type(plan.get(key)) is not int or plan[key] != 1 for key in ('schema_version', 'phase')):
        raise ValueError('Requires a phase-1 proper-score plan with schema_version 1')
    if set(plan.get('data_hashes', {})) != DATA_FILES:
        raise ValueError('Plan must bind exactly the four development data hashes')
    seeds = plan.get('generation_seeds')
    if not isinstance(seeds, list) or not seeds or not all(_seed(seed) for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Generation seeds must be distinct unsigned 32-bit integers')
    full, nested = plan.get('ensemble_size'), plan.get('nested_size')
    if type(full) is not int or type(nested) is not int or not 2 <= nested < full:
        raise ValueError('Require 2 <= nested_size < ensemble_size')
    models = plan.get('models')
    if not isinstance(models, list) or not models:
        raise ValueError('Plan must declare a nonempty model list')
    ids = set()
    for entry in models:
        if (not isinstance(entry, dict) or not isinstance(entry.get('id'), str)
                or not re.fullmatch('[a-z][a-z0-9_]*', entry['id']) or entry['id'] in ids):
            raise ValueError('Model IDs must be distinct safe lowercase names')
        ids.add(entry['id'])
        kind = entry.get('kind')
        if kind == 'baseline':
            if set(entry) != {'id', 'kind', 'baseline'} or entry['baseline'] not in {'mean', 'resample', 'copula'}:
                raise ValueError('Invalid baseline model declaration')
        elif kind == 'strong':
            if (set(entry) != {'id', 'kind', 'baseline', 'fit_seed', 'df'}
                    or entry['baseline'] not in {'beta', 'student_t'} or not _seed(entry['fit_seed'])):
                raise ValueError('Invalid stronger-baseline declaration')
            if entry['baseline'] == 'beta':
                if entry['df'] is not None or entry['fit_seed'] in seeds:
                    raise ValueError('Beta fitting seed must be separate from generation and df must be null')
            elif type(entry['df']) not in (int, float) or not np.isfinite(entry['df']) or entry['df'] <= 2:
                raise ValueError('Student-t df must be finite and greater than two')
        elif kind in {'mixture', 'neural'}:
            expected = {'id', 'kind', 'run_directory', 'manifest_sha256',
                        'model_sha256' if kind == 'mixture' else 'checkpoint_sha256'}
            if kind == 'neural' and 'calibration' in entry:
                expected.add('calibration')
                calibration = entry['calibration']
                if (not isinstance(calibration, dict) or set(calibration) != {
                        'study_directory', 'manifest_sha256', 'fit_artifact', 'fit_sha256',
                        'calibration_artifact', 'calibration_sha256'}
                        or not all(_safe_name(calibration.get(key)) for key in ('fit_artifact', 'calibration_artifact'))):
                    raise ValueError('Invalid frozen calibration binding')
            if set(entry) != expected or not isinstance(entry['run_directory'], str):
                raise ValueError('Invalid saved model binding')
        else:
            raise ValueError('Unknown model kind')
    data = Path(plan['data']).resolve()
    for name, expected in plan['data_hashes'].items():
        _bound_hash(data / name, expected, 'Plan data')
    return plan


def _calendar_audit(values, split):
    dates = values['date'].astype('datetime64[D]')
    expected = np.arange('2024-01-01', '2025-01-01', dtype='datetime64[D]')
    excluded = np.array(['2024-03-31', '2024-10-27'], dtype='datetime64[D]')
    expected = expected[~np.isin(expected, excluded)]
    weekend = ((dates.astype(np.int64) + 3) % 7 >= 5).astype(np.int64)
    month = dates.astype('datetime64[M]').astype(np.int64) % 12 + 1
    calendar = weekend * 4 + (month % 12) // 3
    if not np.array_equal(values['y'] % 8, calendar):
        raise ValueError(f'{split} condition labels disagree with the calendar')
    if not np.array_equal(np.unique(values['y']), np.arange(64)):
        raise ValueError(f'{split} must support all 64 conditions')
    identities = np.rec.fromarrays([values['meter_id'], values['date']])
    if len(np.unique(identities)) != len(dates):
        raise ValueError(f'{split} has duplicate meter/date rows')
    meters, inverse = np.unique(values['meter_id'], return_inverse=True)
    counts = np.bincount(inverse)
    if not np.all(counts == 364):
        raise ValueError(f'{split} requires exactly 364 dates for every meter')
    order = np.argsort(inverse, kind='stable')
    strata = []
    for index, meter in enumerate(meters):
        rows = order[index * 364:(index + 1) * 364]
        if not np.array_equal(np.sort(dates[rows]), expected):
            raise ValueError(f'{split} meter {meter!r} does not contain the complete eligible 2024 calendar')
        equipment = np.unique(values['y'][rows] // 8)
        if len(equipment) != 1 or len(np.unique(values['y'][rows] % 8)) != 8:
            raise ValueError(f'{split} meter {meter!r} has inconsistent equipment or incomplete calendar bins')
        strata.append(int(equipment[0]))
    return {'profiles': len(dates), 'meters': len(meters), 'dates_per_meter': 364,
            'first_date': '2024-01-01', 'last_date': '2024-12-31',
            'excluded_dates': excluded.astype('U10').tolist(),
            'calendar_sha256': _digest(expected.astype(np.int64), '<i8'),
            'calendar_bins_per_meter': 8,
            'condition_counts': {str(label): int(np.sum(values['y'] == label)) for label in range(64)},
            'meters_per_equipment_stratum': {str(label): int(np.sum(np.asarray(strata) == label)) for label in range(8)}}


def _data(plan):
    result = _read_data(Path(plan['data']).resolve())
    training, validation, _, metadata, hashes = result
    if hashes != plan['data_hashes']:
        raise ValueError('Data changed after plan binding')
    if metadata.get('development', {}).get('source_partition_role') != 'original_train':
        raise ValueError('Requires inner development data sourced from original TRAIN')
    return result


def _calibration(entry, candidate, training, hashes, seeds):
    directory = Path(entry['study_directory']).resolve()
    _bound_hash(directory / 'manifest.json', entry['manifest_sha256'], 'Calibration study manifest')
    manifest = diagnostics._verify(directory, (entry['fit_artifact'], entry['calibration_artifact']))
    for field, artifact in [('fit_sha256', 'fit_artifact'), ('calibration_sha256', 'calibration_artifact')]:
        _bound_hash(directory / entry[artifact], entry[field], 'Calibration artifact')
    record = _read(directory / entry['fit_artifact'])
    source_candidate = record.get('candidate', {})
    if (manifest.get('data_hashes') != hashes or record.get('data_hashes') != hashes
            or record.get('fitted_on') != 'inner train'
            or record.get('checkpoint_sha256') != candidate['checkpoint_sha256']
            or source_candidate.get('checkpoint_sha256') != candidate['checkpoint_sha256']
            or source_candidate.get('manifest_sha256') != candidate['manifest_sha256']
            or record.get('calibration_artifact') != entry['calibration_artifact']
            or record.get('calibration_sha256') != entry['calibration_sha256']
            or record.get('calibration') != manifest.get('calibration')
            or record.get('calibration_seed') != manifest.get('calibration', {}).get('seed')
            or not _seed(record.get('calibration_seed')) or record['calibration_seed'] in seeds
            or record.get('target_rows') != len(training['x'])
            or record.get('target_profile_bytes_sha256') != _digest(training['x'], '<f8')
            or record.get('target_label_bytes_sha256') != _digest(training['y'], '<i8')):
        raise ValueError('Frozen calibration fit disagrees with checkpoint, TRAIN or independent seeds')
    fitted = load_calibration(directory / entry['calibration_artifact'], entry['calibration_sha256'])
    if fitted.conditions != tuple(range(64)) or fitted.profile_dim != 24:
        raise ValueError('Frozen calibration condition coverage differs from the evaluation')
    source_count = record.get('source_profiles_per_condition')
    target_counts = {str(label): int(np.sum(training['y'] == label)) for label in range(64)}
    if (type(source_count) is not int or source_count < 1 or record.get('source_rows') != 64 * source_count
            or record.get('target_profiles_per_condition') != target_counts):
        raise ValueError('Frozen calibration fit counts disagree with TRAIN or source ensemble')
    for label, (source, target) in fitted._groups.items():
        if len(source) != source_count or not np.array_equal(target, np.sort(training['x'][training['y'] == label], axis=0)):
            raise ValueError('Frozen calibration arrays disagree with their TRAIN/source fit receipt')
    return fitted, {'binding': entry, 'fit_record': record,
                    'study_source_sha256': manifest.get('source_sha256', {})}


def _bind_model(entry, training, hashes, scaler, seeds):
    kind = entry['kind']
    metadata = {'model_id': entry['id'], 'kind': kind, 'binding': entry, 'fitted_on': 'inner train',
                'training_profile_bytes_sha256': _digest(training['x'], '<f8'),
                'training_label_bytes_sha256': _digest(training['y'], '<i8'),
                'training_profiles': len(training['x']), 'training_seed': None,
                'generation_seed_role': 'Independent IID ensemble generation, not refitting or training replication'}
    if kind == 'baseline':
        fitted = fit_baseline(entry['baseline'], training['x'], training['y'])
        metadata.update(method=entry['baseline'], fitting='deterministic full inner-TRAIN fit')
        return fitted.sample, metadata
    if kind == 'strong':
        fitted = fit_strong_baseline(entry['baseline'], training['x'], training['y'],
                                    fit_seed=entry['fit_seed'], df=entry['df'])
        metadata.update(method=fitted.metadata()['method'], fit_metadata=fitted.metadata(),
                        fitting_seed=fitted.metadata()['fit_seed'])
        return fitted.sample, metadata
    directory = Path(entry['run_directory']).resolve()
    _bound_hash(directory / 'manifest.json', entry['manifest_sha256'], 'Model manifest')
    if kind == 'mixture':
        receipt = diagnostics._verify(directory, ('model.npz', 'fit_metadata.json'))
        _bound_hash(directory / 'model.npz', entry['model_sha256'], 'Mixture model')
        fitted = load_mixture_baseline(directory / 'model.npz', entry['model_sha256'])
        fit = fitted.metadata()
        if (receipt.get('method') != MIXTURE_METHOD or receipt.get('data_hashes') != hashes
                or receipt.get('fitted_on') != 'inner train'
                or receipt.get('validation_performance_access') is not False
                or receipt.get('test_performance_access') is not False
                or receipt.get('model_artifact') != 'model.npz'
                or receipt.get('model_sha256') != entry['model_sha256']
                or receipt.get('config') != fit['config'] or _read(directory / 'fit_metadata.json') != fit
                or fit['source_profile_bytes_sha256'] != metadata['training_profile_bytes_sha256']
                or fit['source_label_bytes_sha256'] != metadata['training_label_bytes_sha256']
                or fit['training_profiles'] != len(training['x']) or fitted.conditions != tuple(range(64))):
            raise ValueError('Saved mixture fit disagrees with completed receipt or TRAIN')
        metadata.update(method=MIXTURE_METHOD, fitting_seed=fit['config']['fit_seed'],
                        fit_metadata=fit, fit_source_sha256=receipt.get('source_sha256', {}))
        return fitted.sample, metadata
    candidate, manifest, generate = diagnostics._candidate(directory, 0)
    _bound_hash(Path(candidate['checkpoint']), entry['checkpoint_sha256'], 'Neural checkpoint')
    if candidate['manifest_sha256'] != entry['manifest_sha256'] or manifest['data_hashes'] != hashes:
        raise ValueError('Neural model binding differs from plan or development data')
    diagnostics._shared(manifest, hashes, manifest['selection'], scaler)
    if manifest['selection']['generation_seed'] in seeds:
        raise ValueError('Generation seeds must exclude the checkpoint selection seed')
    metadata.update(method=candidate['method'], candidate=candidate,
                    training_seed=candidate['training_seed'], fitted_on='frozen selected inner-TRAIN checkpoint')
    calibration = None
    if 'calibration' in entry:
        calibration, calibration_record = _calibration(entry['calibration'], candidate, training, hashes, seeds)
        if calibration_record['fit_record']['calibration_seed'] == manifest['selection']['generation_seed']:
            raise ValueError('Calibration fit reused the checkpoint selection seed')
        metadata.update(method=candidate['method'] + '_marginal_calibration', calibration=calibration_record)
    generation_sources = [(Path(candidate['checkpoint']), entry['checkpoint_sha256'])]
    if 'marginal_artifact' in candidate:
        generation_sources.append((Path(candidate['marginal_artifact']), candidate['marginals']['sha256']))
    def sample(labels, seed):
        for source, expected in generation_sources:
            _bound_hash(source, expected, 'Generation source before draw')
        generated = generate(Path(candidate['checkpoint']), labels, seed)
        for source, expected in generation_sources:
            _bound_hash(source, expected, 'Generation source after draw')
        return calibration.apply(generated, labels)[0] if calibration is not None else generated
    return sample, metadata


@contextmanager
def _single_thread():
    threads, torch_rng = torch.get_num_threads(), torch.get_rng_state().clone()
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(torch_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)


def _inventory(directory):
    hashes, errors = {}, {}
    try:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path != directory / 'manifest.json':
                name = str(path.relative_to(directory))
                try:
                    hashes[name] = _sha256(path)
                except Exception as error:
                    errors[name] = f'{type(error).__name__}: {error}'
    except Exception as error:
        errors['<directory>'] = f'{type(error).__name__}: {error}'
    return {'artifact_hashes': hashes, **({'artifact_hash_errors': errors} if errors else {})}


def _npz(path, **arrays):
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)


def _save_receipt(path, receipt):
    try:
        _json(path, receipt)
    except Exception as error:
        receipt['status'] = 'failed'
        receipt.setdefault('error', f'Receipt write failed: {type(error).__name__}: {error}')
        receipt.setdefault('receipt_write_errors', []).append(f'{type(error).__name__}: {error}')


def _score_draw(generated, validation, ensemble_size, nested_size):
    """Fit each condition/size score function once and restore source row order."""
    columns = [f'{score}_{count}' for count in (nested_size, ensemble_size) for score in SCORES]
    vectors = {name: np.empty(len(validation['x']), dtype=np.float64) for name in columns}
    condition_report = {}
    for condition in range(64):
        indices = np.flatnonzero(validation['y'] == condition)
        ensemble = generated[condition * ensemble_size:(condition + 1) * ensemble_size]
        record = {'real_profiles': len(indices), 'scores': {}}
        for count in (nested_size, ensemble_size):
            fitted = fit_ensemble_scores(ensemble[:count], block_size=256)
            scores = fitted.score(validation['x'][indices])
            for name in SCORES:
                vectors[f'{name}_{count}'][indices] = scores[name]
            record['scores'][str(count)] = {'score_function': fitted.metadata(),
                                           'means': {name: float(value.mean()) for name, value in scores.items()}}
        condition_report[str(condition)] = record
    contributions = meter_contributions(np.column_stack([vectors[name] for name in columns]),
                                         validation['meter_id'], validation['y'])
    arrays = {**vectors, 'row_index': np.arange(len(validation['x']), dtype=np.int64),
              'y': validation['y'], 'meter_ids': np.asarray(contributions['meter_ids']),
              'meter_strata': contributions['strata'], 'meter_values': contributions['values'],
              'meter_bin_counts': contributions['bin_counts'], 'meter_bin_sums': contributions['bin_sums'],
              'score_columns': np.asarray(columns)}
    strata_means = {str(group): contributions['values'][contributions['strata'] == group].mean(axis=0).tolist()
                    for group in range(8)}
    report = {'score_columns': columns, 'per_condition': condition_report,
              'per_stratum_meter_means': strata_means,
              'macro_meter_balanced_scores': dict(zip(columns, np.mean(list(strata_means.values()), axis=0).tolist())),
              'meter_aggregation': 'equal calendar bins within meter, then equal meters within equipment stratum, then equal strata',
              'profile_rows': len(validation['x']), 'meters': len(contributions['meter_ids'])}
    return arrays, report


def _physical_metrics(generated, generated_labels, validation, scaler):
    """Retain the historical metrics with all real rows and separate labels."""
    directions = projection_directions(24, count=128, seed=2026)
    metrics = conditional_metrics(validation['x'], generated, validation['y'], generated_labels,
                                  interval_hours=1., directions=directions)
    training_range = scaler['maximum'] - scaler['minimum']
    denominator = training_range or 1.
    for name in ('macro', 'weighted'):
        metrics[name]['normalized_profile_sw1'] = metrics[name]['profile_sw1_kwh'] / denominator
    for record in metrics['per_condition'].values():
        record['metrics']['normalized_profile_sw1'] = record['metrics']['profile_sw1_kwh'] / denominator
    metrics['metric_valid_conditions']['normalized_profile_sw1'] = 64
    specification = {'ensemble': 'main ensemble only', 'directions_count': 128, 'directions_seed': 2026,
                     'directions_sha256': _digest(directions, '<f8'), 'interval_hours': 1.,
                     'normalization': {'fitted_on': 'train', 'minimum': scaler['minimum'],
                                       'maximum': scaler['maximum'], 'denominator': denominator,
                                       'constant_range_fallback': 1.}}
    return metrics, specification


def _evaluate_model(entry, plan, output, provenance):
    output = Path(output)
    receipt = dict(provenance, model=entry, status='running', draws=[], failed_draws=[],
                   started_at_utc=_now(), worker_pid=os.getpid(), worker_threads=1)
    started = time.perf_counter()
    try:
        output.mkdir(exist_ok=False)
        _save_receipt(output / 'manifest.json', receipt)
        with _single_thread():
            training, validation, scaler, _, hashes = _data(plan)
            sample, metadata = _bind_model(entry, training, hashes, scaler, plan['generation_seeds'])
            _json(output / 'model_metadata.json', metadata)
            receipt['model_metadata_sha256'] = _sha256(output / 'model_metadata.json')
            labels = np.repeat(np.arange(64, dtype=np.int64), plan['ensemble_size'])
            for seed in plan['generation_seeds']:
                draw = {'model_id': entry['id'], 'generation_seed': seed, 'status': 'running',
                        'ensemble_size': plan['ensemble_size'], 'nested_size': plan['nested_size'],
                        'data_rows_sha256': provenance['data_rows_sha256'],
                        'model_metadata_sha256': receipt['model_metadata_sha256']}
                tick = time.perf_counter()
                try:
                    generated = diagnostics._profiles(sample(labels, seed))
                    if len(generated) != len(labels):
                        raise ValueError('Generated profiles differ from the declared label count')
                    draw['generation_seconds'] = time.perf_counter() - tick
                    generated_path = output / f'generated_{seed}.npz'
                    _npz(generated_path, x=generated, y=labels)
                    draw.update(generated_artifact=generated_path.name, generated_sha256=_sha256(generated_path),
                                generated_profile_bytes_sha256=_digest(generated, '<f8'),
                                generated_label_bytes_sha256=_digest(labels, '<i8'))
                    tick = time.perf_counter()
                    arrays, report = _score_draw(generated, validation, plan['ensemble_size'], plan['nested_size'])
                    arrays['data_rows_sha256'] = np.asarray(provenance['data_rows_sha256'])
                    score_path = output / f'profile_scores_{seed}.npz'
                    _npz(score_path, **arrays)
                    draw.update(report, status='completed', scoring_seconds=time.perf_counter() - tick,
                                score_artifact=score_path.name, score_sha256=_sha256(score_path))
                    tick = time.perf_counter()
                    metrics, specification = _physical_metrics(generated, labels, validation, scaler)
                    draw.update(physical_metrics=metrics, physical_metric_specification=specification,
                                physical_metric_seconds=time.perf_counter() - tick)
                except Exception as error:
                    draw.update(status='failed', error=f'{type(error).__name__}: {error}')
                    receipt.setdefault('error', draw['error'])
                    receipt['failed_draws'].append(seed)
                _save_receipt(output / f'draw_{seed}.json', draw)
                if draw['status'] != 'completed' and seed not in receipt['failed_draws']:
                    receipt['failed_draws'].append(seed)
                    receipt.setdefault('error', draw['error'])
                receipt['draws'].append(draw)
                _save_receipt(output / 'manifest.json', receipt)
            if receipt['failed_draws'] or receipt.get('receipt_write_errors'):
                receipt['status'] = 'failed'
            else:
                receipt['status'] = 'completed'
    except Exception as error:
        receipt['status'] = 'failed'
        receipt.setdefault('error', f'{type(error).__name__}: {error}')
    receipt.update(finished_at_utc=_now(), elapsed_seconds=time.perf_counter() - started, **_inventory(output))
    if receipt.get('artifact_hash_errors'):
        receipt['status'] = 'failed'
        receipt.setdefault('error', 'Artifact inventory failed; some output hashes are unavailable')
    _save_receipt(output / 'manifest.json', receipt)
    print(f"{entry['id']}: {receipt['status']}", flush=True)
    return receipt


def _row_artifact(validation, path):
    profile_hashes = np.stack([np.frombuffer(hashlib.sha256(np.asarray(row, dtype='<f8').tobytes()).digest(), dtype=np.uint8)
                               for row in validation['x']])
    _npz(path, row_index=np.arange(len(validation['x']), dtype=np.int64),
         y=validation['y'], meter_id=validation['meter_id'], date=validation['date'],
         profile_row_sha256=profile_hashes)


def _worker_failure(entry, plan, directory, provenance, error):
    """Recover partial worker receipts before recording a coordinator failure."""
    directory.mkdir(exist_ok=True)
    receipt, recovery_errors = dict(provenance, model=entry), []
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        try:
            partial = _read(manifest_path)
            if (partial.get('model') != entry or partial.get('data_hashes') != provenance['data_hashes']
                    or partial.get('plan_sha256') != provenance['plan_sha256']):
                raise ValueError('Partial worker receipt differs from its declared model or plan')
            receipt.update(partial)
        except Exception as recovery_error:
            recovery_errors.append(f'manifest.json: {type(recovery_error).__name__}: {recovery_error}')
    draws = {}
    def recover(draw):
        if (not isinstance(draw, dict) or draw.get('model_id') != entry['id']
                or not _seed(draw.get('generation_seed'))
                or draw['generation_seed'] not in plan['generation_seeds']
                or draw.get('status') not in {'running', 'completed', 'failed'}):
            raise ValueError('Invalid partial draw receipt')
        draws[draw['generation_seed']] = draw
    partial_draws = receipt.get('draws', [])
    if not isinstance(partial_draws, list):
        recovery_errors.append('manifest.json: draw collection is not a list; recovering individual receipts')
        partial_draws = []
    for draw in partial_draws:
        try:
            recover(draw)
        except Exception as recovery_error:
            recovery_errors.append(f'manifest draw: {type(recovery_error).__name__}: {recovery_error}')
    for seed in plan['generation_seeds']:
        path = directory / f'draw_{seed}.json'
        if path.exists():
            try:
                draw = _read(path)
                if draw.get('generation_seed') != seed:
                    raise ValueError('Draw receipt generation seed differs from its filename')
                recover(draw)
            except Exception as recovery_error:
                recovery_errors.append(f'{path.name}: {type(recovery_error).__name__}: {recovery_error}')
    worker_error = f'{type(error).__name__}: {error}'
    receipt.update(status='failed', worker_error=worker_error, finished_at_utc=_now(),
                   draws=[draws[seed] for seed in plan['generation_seeds'] if seed in draws],
                   failed_draws=[seed for seed in plan['generation_seeds']
                                 if seed in draws and draws[seed]['status'] != 'completed'],
                   interrupted_generation_seeds=[seed for seed in plan['generation_seeds'] if seed not in draws],
                   **_inventory(directory))
    receipt.setdefault('error', worker_error)
    if recovery_errors:
        receipt['recovery_errors'] = recovery_errors
    _save_receipt(manifest_path, receipt)
    return receipt


def evaluate(plan_path, output, workers=4):
    """Execute every declared model once, retaining failures and all partial files."""
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('workers must be an integer from one to four')
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan_sha256 = _sha256(plan_path)
    plan = _plan(plan_path)
    training, validation, _, _, hashes = _data(plan)
    train_calendar = _calendar_audit(training, 'TRAIN')
    val_calendar = _calendar_audit(validation, 'VAL')
    _bound_hash(plan_path, plan_sha256, 'Plan preparation')
    sources = [Path(__file__).resolve(), Path(diagnostics.__file__).resolve(),
               *sorted((ROOT / 'src/fluvius_cgan').glob('*.py'))]
    provenance = {'schema_version': 1, 'phase': 1,
        'role': 'full inner-validation proper-score development supplement',
        'data_directory': str(Path(plan['data']).resolve()), 'data_hashes': hashes,
        'plan_path': str(plan_path), 'plan_sha256': plan_sha256,
        'test_performance_access': False, 'intervals_computed': False,
        'training_calendar': train_calendar, 'validation_calendar': val_calendar,
        'validation_profile_bytes_sha256': _digest(validation['x'], '<f8'),
        'validation_label_bytes_sha256': _digest(validation['y'], '<i8'),
        'generation_seeds': plan['generation_seeds'], 'ensemble_size': plan['ensemble_size'],
        'nested_size': plan['nested_size'], 'nested_rule': 'first nested_size members within each condition ensemble',
        'generation_labels': '64 ascending conditions, each repeated ensemble_size times',
        'ensemble_design': 'IID members within each condition; no antithetic or quasi-Monte Carlo sampling',
        'source_sha256': {str(path.relative_to(ROOT)): _sha256(path) for path in sources},
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                                   capture_output=True, text=True).stdout.strip(),
        'runtime': {'python': platform.python_version(), 'workers': workers, 'worker_threads': 1,
            'device': 'cpu', 'process_mode': 'spawn' if workers > 1 else 'serial',
            'libraries': {name: importlib.metadata.version(name) for name in
                          ('numpy', 'scipy', 'torch', 'scikit-learn', 'threadpoolctl', 'joblib')}},
        'interpretation': INTERPRETATION}
    manifest = dict(provenance, status='running', models=plan['models'], completed_models=0,
                    failed_models=[], started_at_utc=_now())
    output.mkdir(parents=True, exist_ok=False)
    summary = {'schema_version': 1, 'interpretation': INTERPRETATION, 'models': {}}
    try:
        (output / 'plan.json').write_bytes(plan_path.read_bytes())
        _bound_hash(output / 'plan.json', provenance['plan_sha256'], 'Copied plan')
        _row_artifact(validation, output / 'data_rows.npz')
        provenance['data_rows_sha256'] = _sha256(output / 'data_rows.npz')
        manifest['data_rows_sha256'] = provenance['data_rows_sha256']
        _json(output / 'manifest.json', manifest)
        del training, validation
        def retain(entry, result):
            summary['models'][entry['id']] = {
                'status': result['status'], 'model': entry,
                'completed_draws': sum(draw['status'] == 'completed' for draw in result['draws']),
                'failed_draws': result['failed_draws'], 'draws': result['draws'],
                **{key: result[key] for key in ('error', 'receipt_write_errors', 'worker_error',
                                               'recovery_errors', 'interrupted_generation_seeds') if key in result}}
            if result['status'] == 'completed':
                manifest['completed_models'] += 1
            else:
                manifest['failed_models'].append(entry['id'])
            _json(output / 'summary.json', summary)
            _json(output / 'manifest.json', manifest)
        if workers == 1:
            for entry in plan['models']:
                retain(entry, _evaluate_model(entry, plan, output / entry['id'], provenance))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as executor:
                jobs = {executor.submit(_evaluate_model, entry, plan, output / entry['id'], provenance): entry
                        for entry in plan['models']}
                for future in as_completed(jobs):
                    entry = jobs[future]
                    try:
                        result = future.result()
                    except Exception as error:
                        result = _worker_failure(entry, plan, output / entry['id'], provenance, error)
                    retain(entry, result)
        if manifest['failed_models']:
            raise RuntimeError('Declared proper-score models failed: ' + ', '.join(manifest['failed_models']))
        for name, expected in hashes.items():
            _bound_hash(Path(plan['data']) / name, expected, 'Final data')
        manifest.update(finished_at_utc=_now(), **_inventory(output))
        if manifest.get('artifact_hash_errors'):
            raise RuntimeError('Artifact inventory failed; some output hashes are unavailable')
        manifest['status'] = 'completed'
        _json(output / 'manifest.json', manifest)
        return manifest
    except Exception as error:
        manifest.update(status='failed', finished_at_utc=_now(), error=f'{type(error).__name__}: {error}',
                        **_inventory(output))
        try:
            _json(output / 'manifest.json', manifest)
        except Exception as receipt_error:
            error.add_note(f'Could not finish failure receipt: {receipt_error}')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    evaluate(args.plan, args.output, args.workers)


if __name__ == '__main__':
    main()
