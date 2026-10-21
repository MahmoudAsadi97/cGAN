#!/usr/bin/env python3
"""Declared TRAIN-only mixture copulas beside hash-verified historical reports.

Historical draws and scores are reused verbatim. New arms fit independently,
save and verify their numeric model artifacts, then score the fixed development
selection. Every declared arm runs once; convergence warnings remain in its fit
metadata and failures do not prevent the other arms from finishing.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time

import numpy as np
import scipy
import sklearn
import torch
from threadpoolctl import threadpool_limits

from fluvius_cgan import development
from fluvius_cgan.mixture_baselines import fit_mixture_baseline, load_mixture_baseline
from fluvius_cgan.proximity import ConditionalNearestNeighbors
from fluvius_cgan.training import _json, _read_data, _sha256

import evaluate_development_candidates as diagnostics
import evaluate_strong_baselines as strong


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {'n_components', 'fit_seed', 'n_init', 'max_iter', 'tol', 'reg_covar'}
METHOD = 'conditional_normal_score_gaussian_mixture_pit'


def _now():
    return datetime.now(timezone.utc).isoformat()


def _plan(path):
    plan = diagnostics._read(path)
    if any(type(plan.get(key)) is not int or plan[key] != 1 for key in ('schema_version', 'phase')):
        raise ValueError('Requires a phase-1 mixture plan with schema_version 1')
    seeds = plan.get('generation_seeds')
    if (not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds))
            or not all(strong.calibration_evaluation._seed(seed) for seed in seeds)):
        raise ValueError('Generation seeds must be distinct unsigned 32-bit integers')
    if set(plan.get('data_hashes', {})) != strong.calibration_evaluation.DATA_FILES:
        raise ValueError('Plan must bind the four development data hashes')
    for name, expected in plan['data_hashes'].items():
        if _sha256(Path(plan['data']) / name) != expected:
            raise ValueError(f'Plan data hash mismatch: {name}')
    entries = plan.get('methods')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Declare a nonempty list of mixture methods')
    identifiers, components, settings = set(), set(), None
    for entry in entries:
        if (not isinstance(entry, dict) or set(entry) != CONFIG | {'method_id'}
                or not re.fullmatch(r'[a-z][a-z0-9_]*', entry.get('method_id', ''))):
            raise ValueError('Each mixture requires a safe method_id and every fixed fit setting')
        for key in ('n_components', 'n_init', 'max_iter'):
            if type(entry[key]) is not int or entry[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('tol', 'reg_covar'):
            if type(entry[key]) not in (int, float) or not np.isfinite(entry[key]) or entry[key] <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if (not strong.calibration_evaluation._seed(entry['fit_seed'])
                or entry['fit_seed'] in seeds):
            raise ValueError('Mixture fit seed must be unsigned 32-bit and distinct from evaluation seeds')
        current = {key: entry[key] for key in CONFIG - {'n_components'}}
        if settings is not None and current != settings:
            raise ValueError('Mixture arms must share the declared fit settings except n_components')
        settings = current
        if entry['method_id'] in identifiers or entry['n_components'] in components:
            raise ValueError('Mixture method IDs and component counts must be distinct')
        identifiers.add(entry['method_id'])
        components.add(entry['n_components'])
    reference = plan['reference']
    if _sha256(Path(reference['directory']) / 'manifest.json') != reference['manifest_sha256']:
        raise ValueError('Plan reference manifest hash mismatch')
    return plan


def _reference(plan, training, validation, hashes, scaler):
    directory = Path(plan['reference']['directory']).resolve()
    manifest = diagnostics._verify(directory, ('plan.json', 'summary.json', 'real_proximity.json',
                                               'metrics.csv', 'condition_diagnostics.csv'))
    if (manifest.get('schema_version') != 1 or manifest.get('phase') != 1
            or any(manifest.get(key) is not False for key in
                   ('test_performance_access', 'reselection', 'clipping', 'calibration_refitted'))):
        raise ValueError('Historical reference provenance differs')
    original = diagnostics._read(directory / 'plan.json')
    if (original['data_hashes'] != hashes or original['methods'] != manifest['methods']
            or original['generation_seeds'] != manifest['generation_seeds']
            or original['reference'] != manifest['reference']
            or original['calibration_reference'] != manifest['calibration_reference']):
        raise ValueError('Historical reference plan differs from manifest')
    if plan['generation_seeds'] != manifest['generation_seeds']:
        raise ValueError('Evaluation seeds must match the historical reference draws exactly')
    selection = manifest['selection']
    diagnostics._shared(manifest, hashes, selection, scaler)
    if (selection.get('directions_seed') != 2026 or selection.get('directions_count') != 128
            or selection.get('generation_seed') in plan['generation_seeds']
            or any(entry['fit_seed'] == selection.get('generation_seed') for entry in plan['methods'])):
        raise ValueError('Invalid projection settings or overlapping checkpoint-selection seed')
    indices = np.asarray(selection['indices'])
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
            or len(np.unique(indices)) != len(indices)
            or np.any((indices < 0) | (indices >= len(validation['x'])))):
        raise ValueError('Invalid historical selection indices')
    real, labels = validation['x'][indices].astype(np.float64), validation['y'][indices]
    if (np.unique(labels).tolist() != selection['conditions']
            or diagnostics._digest(real, '<f8') != manifest['real_profile_bytes_sha256']
            or diagnostics._digest(labels, '<i8') != manifest['label_bytes_sha256']):
        raise ValueError('Historical selection arrays differ')
    expected = {}
    for candidate in manifest['candidates']:
        for suffix in ('raw', 'calibrated'):
            method_id = f"{candidate['method_id']}_{suffix}"
            fields = {key: candidate[key] for key in ('checkpoint_sha256', 'manifest_sha256',
                'training_seed', 'config', 'selected_step', 'completed_steps', 'training_source_sha256')}
            fields.update(method=candidate['method'] + ('_marginal_calibration' if suffix == 'calibrated' else ''),
                          calibration_applied=suffix == 'calibrated')
            expected[method_id] = fields
    for kind in ('mean', 'resample', 'copula'):
        expected[f'baseline_{kind}'] = {'method': kind, 'training_seed': None, 'fitted_on': 'inner train'}
    for entry in manifest['methods']:
        expected[f"strong_{entry['method_id']}"] = {'method': 'conditional_empirical_beta_copula'
            if entry['kind'] == 'beta' else 'conditional_student_t_copula',
            'training_seed': None, 'fitted_on': 'inner train', 'specification': entry, 'data_hashes': hashes}
    summary = diagnostics._read(directory / 'summary.json')
    if (set(summary['methods']) != set(expected) or summary['generation_seeds'] != plan['generation_seeds']
            or summary['selection_profiles'] != len(real) or summary['conditions'] != selection['conditions']):
        raise ValueError('Historical summary draw or method identities differ')
    reports = {}
    for method_id, fields in expected.items():
        filename = f'{method_id}.json'
        if filename not in manifest['artifact_hashes']:
            raise ValueError('Historical reference omitted a required report hash')
        report = diagnostics._read(directory / filename)
        if report.get('method_id') != method_id or any(report.get(key) != value for key, value in fields.items()):
            raise ValueError(f'Historical report identity differs: {method_id}')
        draws = strong._draws(report, plan['generation_seeds'])
        if len(report['draws']) != len(draws):
            raise ValueError('Historical report contains undeclared or duplicate draws')
        for draw in draws.values():
            if not re.fullmatch('[0-9a-f]{64}', draw.get('generated_profile_bytes_sha256', '')):
                raise ValueError('Historical draw array hash is missing or invalid')
            if (set(draw['proximity']['per_condition']) != set(map(str, selection['conditions']))
                    or not re.fullmatch('[0-9a-f]{64}', draw['proximity'].get('distance_bytes_sha256', ''))):
                raise ValueError('Historical proximity draw identity differs')
        if strong._summary(report) != summary['methods'][method_id]:
            raise ValueError(f'Historical report scores differ from summary: {method_id}')
        reports[method_id] = report
    if manifest['completed_draws'] != sum(len(report['draws']) for report in reports.values()):
        raise ValueError('Historical completed draw count differs')
    real_report = diagnostics._read(directory / 'real_proximity.json')
    if (real_report['data_hashes'] != hashes
            or real_report['real_profile_bytes_sha256'] != manifest['real_profile_bytes_sha256']
            or real_report['label_bytes_sha256'] != manifest['label_bytes_sha256']
            or real_report['proximity']['macro'] != summary['real_proximity']):
        raise ValueError('Historical real proximity provenance differs')
    filenames = {'plan.json', 'manifest.json', 'summary.json', 'metrics.csv', 'condition_diagnostics.csv',
                 'reference_plan.json', 'reference_manifest.json', 'reference_summary.json',
                 'real_proximity.json', 'selection.npz'} | {f'{name}.json' for name in reports}
    for entry in plan['methods']:
        names = {f"{entry['method_id']}{suffix}" for suffix in
                 ('.json', '_fit.json', '_model.npz', '_job.json', '_worker.json', '.log')}
        if filenames & names:
            raise ValueError('Mixture method_id collides with another artifact identity')
        filenames |= names
    return real, labels, selection, reports, summary, manifest, real_report


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


def _inventory(output):
    hashes, errors = {}, {}
    try:
        paths = sorted(Path(output).iterdir())
    except Exception as error:
        return {'artifact_hashes': {}, 'artifact_hash_errors': {'<directory>': f'{type(error).__name__}: {error}'}}
    for path in paths:
        if path.is_file() and path.name != 'manifest.json':
            try:
                hashes[path.name] = _sha256(path)
            except Exception as error:
                errors[path.name] = f'{type(error).__name__}: {error}'
    return {'artifact_hashes': hashes, **({'artifact_hash_errors': errors} if errors else {})}


def _marginal_support(fake, labels, training, training_labels):
    groups = {}
    for label in np.unique(labels):
        values, target = fake[labels == label], training[training_labels == label]
        minimum, maximum = target.min(axis=0), target.max(axis=0)
        groups[str(int(label))] = {'profiles': len(values), 'training_profiles': len(target),
            'zero_counts': (values == 0).sum(axis=0).tolist(),
            'negative_counts': (values < 0).sum(axis=0).tolist(),
            'at_training_min_counts': (values == minimum).sum(axis=0).tolist(),
            'at_training_max_counts': (values == maximum).sum(axis=0).tolist(),
            'outside_training_support_counts': [int(np.sum(~np.isin(values[:, hour], target[:, hour])))
                                                for hour in range(values.shape[1])],
            'training_zero_counts': (target == 0).sum(axis=0).tolist(),
            'training_negative_counts': (target < 0).sum(axis=0).tolist(),
            'training_min_counts': (target == minimum).sum(axis=0).tolist(),
            'training_max_counts': (target == maximum).sum(axis=0).tolist()}
    return {'schema_version': 1, 'per_condition': groups,
        'interpretation': 'Per-hour counts use their respective generated or TRAIN row denominator. '
                         'Observed hourly support and atoms do not establish joint-profile support or privacy.'}


def _run_method(job):
    output, entry = Path(job['output']), job['specification']
    method_id = entry['method_id']
    report = {'schema_version': 1, 'method_id': method_id, 'method': METHOD, 'training_seed': None,
        'status': 'running', 'specification': entry, 'fitted_on': 'inner train',
        'data_hashes': job['data_hashes'], 'draws': [], 'started_at_utc': _now()}
    _json(output / f'{method_id}.json', report)
    started = time.perf_counter()
    try:
        with _single_thread():
            training, validation, scaler, metadata, hashes = _read_data(Path(job['data']))
            if hashes != job['data_hashes'] or metadata.get('development', {}).get('source_partition_role') != 'original_train':
                raise ValueError('Worker TRAIN provenance differs from plan')
            indices = np.asarray(job['selection']['indices'])
            real, labels = validation['x'][indices].astype(np.float64), validation['y'][indices]
            if (diagnostics._digest(real, '<f8') != job['real_profile_bytes_sha256']
                    or diagnostics._digest(labels, '<i8') != job['label_bytes_sha256']
                    or _sha256(output / 'selection.npz') != job['selection_artifact_sha256']):
                raise ValueError('Worker evaluation selection differs')
            with np.load(output / 'selection.npz', allow_pickle=False) as saved:
                real_distances = saved['real_distances']
            if (real_distances.shape != (len(real),) or not np.isfinite(real_distances).all()
                    or np.any(real_distances < 0)
                    or diagnostics._digest(real_distances, '<f8') != job['real_distance_bytes_sha256']):
                raise ValueError('Worker real-distance reference differs')
            tick = time.perf_counter()
            fitted = fit_mixture_baseline(training['x'], training['y'], **{key: entry[key] for key in CONFIG})
            report.update(fit_seconds=time.perf_counter() - tick, fit_metadata=fitted.metadata())
            if report['fit_metadata'].get('method') != METHOD:
                raise ValueError('Fitted mixture method differs from declared construction')
            model_path = output / f'{method_id}_model.npz'
            tick = time.perf_counter()
            fitted.save(model_path)
            report.update(model_artifact=model_path.name, model_sha256=_sha256(model_path))
            _json(output / f'{method_id}_fit.json', {key: value for key, value in report.items() if key != 'draws'})
            fitted = load_mixture_baseline(model_path, report['model_sha256'])
            if fitted.metadata() != report['fit_metadata']:
                raise ValueError('Reloaded mixture metadata differs from fitted model')
            report['save_and_verified_reload_seconds'] = time.perf_counter() - tick
            nearest = ConditionalNearestNeighbors(training['x'], training['y'])
            copies = diagnostics.CopyIndex(training['x'])
            scale = scaler['maximum'] - scaler['minimum']
            for seed in job['generation_seeds']:
                tick = time.perf_counter()
                fake = fitted.sample(labels, seed)
                elapsed = time.perf_counter() - tick
                tick = time.perf_counter()
                metrics = development.physical_metrics(real, fake, labels, scale)
                tails = diagnostics.tail_diagnostics(real, fake, labels, copies.matches(fake))
                metric_seconds = time.perf_counter() - tick
                tick = time.perf_counter()
                proximity = strong._proximity(nearest.distances(fake, labels), labels, scale or 1., real_distances)
                report['draws'].append({'generation_seed': seed, 'generation_seconds': elapsed,
                    'physical_metric_seconds': metric_seconds, 'nearest_neighbor_seconds': time.perf_counter() - tick,
                    'generated_profile_bytes_sha256': diagnostics._digest(fake, '<f8'),
                    'metrics': metrics, 'diagnostics': tails, 'proximity': proximity,
                    'marginal_support': _marginal_support(fake, labels, training['x'], training['y']),
                    'reference_draw_match': False})
                _json(output / f'{method_id}.json', report)
                print(f"{method_id} draw={seed} normalized_SW1={metrics['macro']['normalized_profile_sw1']:.8f}", flush=True)
            report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    report.update(finished_at_utc=_now(), elapsed_seconds=time.perf_counter() - started)
    _json(output / f'{method_id}.json', report)
    return report


def _subprocess(job_path):
    output, name = job_path.parent, job_path.stem.removesuffix('_job')
    path = output / f'{name}.json'
    report = {'method_id': name, 'method': METHOD, 'training_seed': None, 'status': 'failed', 'draws': []}
    environment = os.environ.copy()
    environment.update({key: '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(job_path)]
    record = {'command': command, 'returncode': None, 'single_thread_environment':
              {key: environment[key] for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}
    try:
        job = diagnostics._read(job_path)
        report['specification'] = job['specification']
        with (output / f'{name}.log').open('x') as log:
            completed = subprocess.run(command, env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record['returncode'] = completed.returncode
        if path.exists():
            report = diagnostics._read(path)
        if completed.returncode != 0 or report.get('status') != 'completed':
            report.update(status='failed', error=report.get('error', f'Worker exited with code {completed.returncode}'))
    except Exception as error:
        # Launch, log and report-reading errors are failures of this arm only.
        # Keep a readable partial report and account for completed siblings.
        record['error'] = f'{type(error).__name__}: {error}'
        try:
            if path.exists():
                report = diagnostics._read(path)
        except Exception as receipt_error:
            record['partial_report_read_error'] = str(receipt_error)
        report.update(status='failed', error=record['error'])
    for destination, value in ((path, report), (output / f'{name}_worker.json', record)):
        try:
            _json(destination, value)
        except Exception as error:
            report.setdefault('receipt_errors', {})[destination.name] = f'{type(error).__name__}: {error}'
    return report


def _tables(reports):
    rows, conditions = [], []
    for report in reports.values():
        identity = {key: report[key] for key in ('method_id', 'method', 'training_seed')}
        for draw in report['draws']:
            metrics, tails, proximity = draw['metrics'], draw['diagnostics'], draw['proximity']
            rows.append(identity | {'generation_seed': draw['generation_seed'], **metrics['macro'], **tails['macro'],
                **{f'nn_{key}': value for key, value in proximity['macro'].items()},
                'quantized_exact_training_match_fraction': tails['copying']['quantized_exact_match_fraction'],
                **{f'pv0_{side}_{key}': tails['source_pv0'][side][key] for side in ('real', 'fake')
                   for key in strong.calibration_evaluation.EXPORT_METRICS}})
            for label, values in tails['per_condition'].items():
                conditions.append(identity | {'generation_seed': draw['generation_seed'], 'condition': int(label),
                    'source_pv_flag': values['source_pv_flag'], **values['errors'],
                    **{f'nn_{key}': value for key, value in proximity['per_condition'][label].items()},
                    'quantized_exact_training_match_fraction': values['quantized_exact_training_match_fraction'],
                    **{f'{side}_{key}': values[side][key] for side in ('real', 'fake')
                       for key in strong.calibration_evaluation.EXPORT_METRICS}})
    return rows, conditions


def evaluate(plan_path, output, workers=4):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('workers must be an integer between 1 and 4')
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = _plan(plan_path)
    training, validation, scaler, metadata, hashes = _read_data(Path(plan['data']))
    if hashes != plan['data_hashes'] or metadata.get('development', {}).get('source_partition_role') != 'original_train':
        raise ValueError('Plan data hashes or inner-TRAIN provenance differ')
    real, labels, selection, reports, previous_summary, reference, real_report = _reference(
        plan, training, validation, hashes, scaler)
    sources = [Path(__file__).resolve(), Path(diagnostics.__file__).resolve(), Path(strong.__file__).resolve(),
        Path(strong.calibration_evaluation.__file__).resolve(), *sorted((ROOT / 'src/fluvius_cgan').glob('*.py'))]
    manifest = {'schema_version': 1, 'phase': 1, 'status': 'running',
        'role': 'post-test development normal-score mixture copula screen', 'test_performance_access': False,
        'reselection': False, 'clipping': False, 'historical_models_regenerated': False,
        'physical_clipping': False, 'clipping_definition': 'The clipping flag refers to generated physical profiles.',
        'probability_roundoff_clamp': {'interval': [0., 1.],
            'operation': 'numpy.clip on the analytic mixture marginal CDF only',
            'purpose': 'Floating-point roundoff at probability endpoints; no physical-value clipping.'},
        'data_directory': str(Path(plan['data']).resolve()), 'data_hashes': hashes,
        'plan_path': str(plan_path), 'plan_sha256': _sha256(plan_path), 'methods': plan['methods'],
        'reference': plan['reference'], 'reference_source_sha256': reference['source_sha256'],
        'selection': selection, 'real_profile_bytes_sha256': diagnostics._digest(real, '<f8'),
        'label_bytes_sha256': diagnostics._digest(labels, '<i8'), 'generation_seeds': plan['generation_seeds'],
        'generation_seed_role': 'Paired reused development draws; not independent meters or training replicates.',
        'metric_normalization': {'fitted_on': 'train', 'minimum': scaler['minimum'], 'maximum': scaler['maximum']},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                    'sklearn': sklearn.__version__, 'torch': str(torch.__version__), 'workers': workers,
                    'threads_per_worker': 1, 'device': 'cpu'},
        'source_sha256': {str(path.relative_to(ROOT)): _sha256(path) for path in sources},
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                                   capture_output=True, text=True).stdout.strip(),
        'started_at_utc': _now(), 'completed_draws': 0,
        'reused_draws': sum(len(report['draws']) for report in reports.values()),
        'historical_report_sha256': {f'{name}.json': reference['artifact_hashes'][f'{name}.json'] for name in reports}}
    output.mkdir(parents=True, exist_ok=False)
    _json(output / 'plan.json', plan)
    _json(output / 'manifest.json', manifest)
    try:
        with _single_thread():
            reference_dir = Path(plan['reference']['directory'])
            for name in ('plan.json', 'manifest.json', 'summary.json'):
                (output / f'reference_{name}').write_bytes((reference_dir / name).read_bytes())
            if _sha256(output / 'reference_manifest.json') != plan['reference']['manifest_sha256']:
                raise ValueError('Historical manifest changed while copying')
            for name in [*(f'{key}.json' for key in reports), 'real_proximity.json']:
                (output / name).write_bytes((reference_dir / name).read_bytes())
                if _sha256(output / name) != reference['artifact_hashes'][name]:
                    raise ValueError('Historical report changed while copying')
            nearest = ConditionalNearestNeighbors(training['x'], training['y'])
            real_distances = nearest.distances(real, labels)
            if strong._proximity(real_distances, labels, (scaler['maximum'] - scaler['minimum']) or 1.) != real_report['proximity']:
                raise ValueError('Recomputed real proximity differs from historical reference')
            with (output / 'selection.npz').open('xb') as handle:
                np.savez_compressed(handle, real_distances=real_distances)
            selection_hash = _sha256(output / 'selection.npz')
            jobs = []
            for entry in plan['methods']:
                job = {'specification': entry, 'data': str(Path(plan['data']).resolve()), 'data_hashes': hashes,
                    'output': str(output), 'selection': selection, 'generation_seeds': plan['generation_seeds'],
                    'real_profile_bytes_sha256': manifest['real_profile_bytes_sha256'],
                    'label_bytes_sha256': manifest['label_bytes_sha256'], 'selection_artifact_sha256': selection_hash,
                    'real_distance_bytes_sha256': diagnostics._digest(real_distances, '<f8')}
                path = output / f"{entry['method_id']}_job.json"
                _json(path, job)
                jobs.append((job, path))
            del training, validation, nearest
            if workers == 1:
                results = [_run_method(job) for job, _ in jobs]
            else:
                with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
                    results = list(pool.map(_subprocess, [path for _, path in jobs]))
            summary = {'schema_version': 1, 'role': manifest['role'], 'generation_seeds': plan['generation_seeds'],
                'selection_profiles': len(real), 'conditions': selection['conditions'],
                'uncertainty_note': 'Paired sampling variation on reused inner validation; no independent validation or calibrated inference.',
                'historical_report_reuse': 'Verified saved reports copied verbatim; historical models were neither refitted nor regenerated.',
                'real_proximity': previous_summary['real_proximity'], 'methods': dict(previous_summary['methods'])}
            failed = []
            for report in results:
                name = report['method_id']
                reports[name] = report
                manifest['completed_draws'] += len(report['draws'])
                if report['status'] == 'completed':
                    summary['methods'][name] = strong._summary(report)
                else:
                    failed.append(name)
                    summary['methods'][name] = {'method': METHOD, 'training_seed': None, 'status': 'failed',
                                               'error': report['error'], 'completed_draws': len(report['draws'])}
            _json(output / 'summary.json', summary)
            rows, conditions = _tables(reports)
            diagnostics._csv(output / 'metrics.csv', rows)
            diagnostics._csv(output / 'condition_diagnostics.csv', conditions)
            if failed:
                raise RuntimeError(f"Declared mixture arms failed: {', '.join(failed)}")
            manifest.update(status='completed', finished_at_utc=_now(), **_inventory(output))
            _json(output / 'manifest.json', manifest)
            return summary
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}', finished_at_utc=_now(),
                        **_inventory(output))
        try:
            _json(output / 'manifest.json', manifest)
        except Exception as receipt_error:
            error.add_note(f'Could not finish failure receipt: {receipt_error}')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        report = _run_method(diagnostics._read(args.worker))
        raise SystemExit(0 if report['status'] == 'completed' else 1)
    if args.plan is None or args.output is None:
        parser.error('--plan and --output are required')
    evaluate(args.plan, args.output, args.workers)


if __name__ == '__main__':
    main()
