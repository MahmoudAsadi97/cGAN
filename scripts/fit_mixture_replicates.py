#!/usr/bin/env python3
"""Fit a bounded, declared mixture-replicate plan on inner TRAIN only.

The four development files are hash-bound, but only TRAIN profiles are loaded.
No validation selection, generated draws, or performance scores are computed.
Every declared job runs once, sequentially with one CPU thread. A returned fit
keeps the core's complete initialization ledger and TRAIN-objective selection
rule. A mid-fit exception preserves the job error and available artifacts; the
core does not expose partial initialization states. Nonconvergence never
triggers retries or an extension of the declared budget.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import re
import subprocess
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from fluvius_cgan.data import load_split
from fluvius_cgan.mixture_baselines import (
    METHOD, _configuration, fit_mixture_baseline, load_mixture_baseline,
)
from fluvius_cgan.training import _json, _sha256


ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = {'train.npz', 'val.npz', 'scaler.json', 'metadata.json'}
FIT_FIELDS = {'n_components', 'fit_seed', 'n_init', 'max_iter', 'tol', 'reg_covar'}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path, contents=None):
    def nonfinite(value):
        raise ValueError(f'Nonfinite JSON value in {path}: {value}')
    value = json.loads(Path(path).read_bytes() if contents is None else contents, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def _digest(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order='C')).hexdigest()


def _prepare(plan_path):
    plan_bytes = plan_path.read_bytes()
    plan = _read(plan_path, plan_bytes)
    if any(type(plan.get(key)) is not int or plan[key] != 1 for key in ('schema_version', 'phase')):
        raise ValueError('Requires a phase-1 fit-only plan with schema_version 1')
    if set(plan.get('data_hashes', {})) != DATA_FILES:
        raise ValueError('Plan must bind exactly the four development data hashes')
    data = Path(plan['data']).resolve()
    for name, expected in plan['data_hashes'].items():
        if _sha256(data / name) != expected:
            raise ValueError(f'Plan data hash mismatch: {name}')
    jobs = plan.get('jobs')
    if not isinstance(jobs, dict) or not jobs:
        raise ValueError('Plan must declare a nonempty set of jobs')
    seeds, settings = set(), None
    for name, config in jobs.items():
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', name):
            raise ValueError('Invalid job name')
        if not isinstance(config, dict) or set(config) != FIT_FIELDS:
            raise ValueError('Every job must declare exactly the six fixed mixture fit settings')
        validated = _configuration(**config)
        if validated['fit_seed'] in seeds:
            raise ValueError('Replicate fitting seeds must be distinct')
        seeds.add(validated['fit_seed'])
        current = {key: value for key, value in validated.items() if key != 'fit_seed'}
        if settings is not None and current != settings:
            raise ValueError('Replicate jobs must share the same settings except fit_seed')
        settings = current
    training = load_split(data / 'train.npz')
    metadata, scaler = _read(data / 'metadata.json'), _read(data / 'scaler.json')
    if (training['x'].shape[1] != 24 or metadata.get('resolution', 24) != 24
            or metadata.get('condition_dim', 64) != 64):
        raise ValueError('Prepared TRAIN must describe 24 hourly coordinates and 64 conditions')
    if metadata.get('development', {}).get('source_partition_role') != 'original_train':
        raise ValueError('Requires inner development TRAIN sourced from original TRAIN')
    declared = metadata.get('splits', {}).get('train', {}).get('sha256')
    if declared is not None and declared != plan['data_hashes']['train.npz']:
        raise ValueError('Prepared TRAIN hash disagrees with metadata')
    if (scaler.get('fitted_on') != 'train'
            or scaler.get('minimum') != float(training['x'].min())
            or scaler.get('maximum') != float(training['x'].max())):
        raise ValueError('Scaler must describe the full TRAIN partition only')
    if any(_sha256(data / name) != expected for name, expected in plan['data_hashes'].items()):
        raise ValueError('Plan data hashes changed while loading TRAIN')
    return plan, training, plan_bytes


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


def _inventory(directory, recursive=False):
    hashes, errors = {}, {}
    try:
        paths = sorted(directory.rglob('*') if recursive else directory.iterdir())
        for path in paths:
            if path.is_file() and path != directory / 'manifest.json':
                name = str(path.relative_to(directory))
                try:
                    hashes[name] = _sha256(path)
                except Exception as error:
                    errors[name] = f'{type(error).__name__}: {error}'
    except Exception as error:
        errors['<directory>'] = f'{type(error).__name__}: {error}'
    return {'artifact_hashes': hashes, **({'artifact_hash_errors': errors} if errors else {})}


def _fit_job(name, config, output, training, provenance):
    receipt = dict(provenance, job_name=name, config=config, status='running', started_at_utc=_now())
    started = time.perf_counter()
    try:
        output.mkdir(exist_ok=False)
        _json(output / 'manifest.json', receipt)
        tick = time.perf_counter()
        fitted = fit_mixture_baseline(training['x'], training['y'], **config)
        metadata = fitted.metadata()
        receipt['fit_seconds'] = time.perf_counter() - tick
        if (metadata['method'] != METHOD or metadata['config'] != config
                or metadata['source_profile_bytes_sha256'] != provenance['training_profile_bytes_sha256']
                or metadata['source_label_bytes_sha256'] != provenance['training_label_bytes_sha256']
                or metadata['training_profiles'] != len(training['x'])):
            raise ValueError('Fitted model metadata differs from the declared TRAIN inputs or settings')
        _json(output / 'fit_metadata.json', metadata)
        model_path = output / 'model.npz'
        tick = time.perf_counter()
        fitted.save(model_path)
        receipt.update(model_artifact=model_path.name, model_sha256=_sha256(model_path))
        # Record the original expected hash before attempting the verified reload.
        _json(output / 'manifest.json', receipt)
        reloaded = load_mixture_baseline(model_path, receipt['model_sha256'])
        if reloaded.metadata() != metadata:
            raise ValueError('Reloaded mixture metadata differs from the fitted model')
        receipt.update(status='completed', save_and_verified_reload_seconds=time.perf_counter() - tick)
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
    receipt.update(finished_at_utc=_now(), elapsed_seconds=time.perf_counter() - started,
                   **_inventory(output))
    if receipt.get('artifact_hash_errors'):
        receipt['status'] = 'failed'
        receipt.setdefault('error', 'Artifact inventory failed; see artifact_hash_errors')
    try:
        _json(output / 'manifest.json', receipt)
    except Exception as error:
        receipt.update(status='failed')
        receipt.setdefault('error', f'Could not write job receipt: {type(error).__name__}: {error}')
        receipt['receipt_write_error'] = f'{type(error).__name__}: {error}'
    print(f"{name}: {receipt['status']}", flush=True)
    return receipt


def run(plan_path, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan, training, plan_bytes = _prepare(plan_path)
    sources = [Path(__file__).resolve(), *sorted((ROOT / 'src/fluvius_cgan').glob('*.py'))]
    provenance = {'schema_version': 1, 'phase': 1, 'method': METHOD,
        'role': 'post-test development TRAIN-only mixture replicate fitting',
        'fitted_on': 'inner train', 'validation_performance_access': False, 'test_performance_access': False,
        'validation_file_access': 'SHA-256 bytes only; no validation archive or profiles are parsed',
        'selection': 'Core selects the largest final TRAIN average log likelihood; exact ties retain earliest initialization.',
        'retries': False, 'budget_extensions': False,
        'data_directory': str(Path(plan['data']).resolve()), 'data_hashes': plan['data_hashes'],
        'training_rows': len(training['x']), 'conditions': np.unique(training['y']).tolist(),
        'training_profile_bytes_sha256': _digest(training['x'], '<f8'),
        'training_label_bytes_sha256': _digest(training['y'], '<i8'),
        'plan_path': str(plan_path), 'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
        'plan_snapshot': 'The exact initial bytes parsed for execution are copied into plan.json.',
        'initialization_ledger': 'Complete when fitting returns. Mid-fit exceptions preserve the whole-job '
                                 'error and available artifacts, but the core exposes no partial initialization states.',
        'source_sha256': {str(path.relative_to(ROOT)): _sha256(path) for path in sources},
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                                   capture_output=True, text=True).stdout.strip(),
        'runtime': {'python': platform.python_version(), 'workers': 1, 'threads': 1, 'device': 'cpu',
            'libraries': {name: importlib.metadata.version(name) for name in
                          ('numpy', 'scipy', 'scikit-learn', 'torch', 'threadpoolctl', 'joblib')}},
        'interpretation': 'Fit-only numerical and replicate evidence. No held-out performance, '
                          'new independent validation, or appliance labels are inferred.'}
    manifest = dict(provenance, status='running', jobs=plan['jobs'], completed_jobs=0, failed_jobs=[],
                    started_at_utc=_now())
    output.mkdir(parents=True, exist_ok=False)
    summary = {'schema_version': 1, 'role': provenance['role'], 'jobs': {}}
    try:
        (output / 'plan.json').write_bytes(plan_bytes)
        if _sha256(output / 'plan.json') != provenance['plan_sha256']:
            raise ValueError('Saved plan snapshot differs from the parsed input bytes')
        _json(output / 'manifest.json', manifest)
        with _single_thread():
            for name, config in plan['jobs'].items():
                receipt = _fit_job(name, config, output / name, training, provenance)
                summary['jobs'][name] = {key: receipt[key] for key in
                    ('status', 'config', 'started_at_utc', 'finished_at_utc', 'elapsed_seconds')}
                summary['jobs'][name].update({key: receipt[key] for key in
                    ('error', 'model_sha256', 'fit_seconds', 'receipt_write_error', 'artifact_hash_errors') if key in receipt})
                if receipt['status'] == 'completed':
                    manifest['completed_jobs'] += 1
                else:
                    manifest['failed_jobs'].append(name)
                _json(output / 'summary.json', summary)
                _json(output / 'manifest.json', manifest)
        if manifest['failed_jobs']:
            raise RuntimeError(f"Declared mixture fitting jobs failed: {', '.join(manifest['failed_jobs'])}")
        manifest.update(**_inventory(output, recursive=True))
        if manifest.get('artifact_hash_errors'):
            raise RuntimeError('Artifact inventory failed; see artifact_hash_errors')
        manifest.update(status='completed', finished_at_utc=_now())
        _json(output / 'manifest.json', manifest)
        return manifest
    except Exception as error:
        manifest.update(status='failed', finished_at_utc=_now(), error=f'{type(error).__name__}: {error}',
                        **_inventory(output, recursive=True))
        try:
            _json(output / 'manifest.json', manifest)
        except Exception as receipt_error:
            error.add_note(f'Could not finish failure receipt: {receipt_error}')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.output)


if __name__ == '__main__':
    main()
