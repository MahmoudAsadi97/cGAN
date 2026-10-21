#!/usr/bin/env python3
"""Score training-only comparators on the development trainer's fixed VAL rows.

This is post-test development, not an independent confirmatory evaluation.
The multiple generation draws measure sampling variability, not training seeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy

from fluvius_cgan.baselines import fit_baseline
from fluvius_cgan.development import physical_metrics
from fluvius_cgan.training import _json, _read_data, _sha256


ROOT = Path(__file__).resolve().parents[1]
METRICS = ('normalized_profile_sw1', 'daily_energy_w1_kwh',
           'mean_profile_rmse_kwh', 'diversity_ratio')


def _digest(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order='C')).hexdigest()


def evaluate(data, output, seeds=(20261022, 20261023, 20261024), per_condition=256):
    data, output, seeds = Path(data), Path(output), list(seeds)
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError('Generation seeds must be unsigned 32-bit integers')
    if len(seeds) != len(set(seeds)):
        raise ValueError('Generation seeds must be distinct')
    if type(per_condition) is not int or not 1 <= per_condition <= 1024:
        raise ValueError('per_condition must be an integer in [1, 1024]')
    if output.exists():
        raise FileExistsError(output)
    training, validation, scaler, metadata, hashes = _read_data(data)
    if metadata.get('development', {}).get('source_partition_role') != 'original_train':
        raise ValueError('Requires an inner development split sourced from original TRAIN')
    rng = np.random.default_rng(20261021)
    indices = np.concatenate([rng.choice(np.flatnonzero(validation['y'] == label),
        min(per_condition, int(np.sum(validation['y'] == label))), replace=False)
        for label in np.unique(training['y'])])
    real = validation['x'][indices].astype(np.float64)
    labels = validation['y'][indices]
    sources = [Path(__file__).resolve(), *sorted((ROOT / 'src' / 'fluvius_cgan').glob('*.py'))]
    manifest = {
        'schema_version': 1, 'status': 'running', 'phase': 1,
        'role': 'post-test development baseline comparison', 'test_performance_access': False,
        'data_directory': str(data.resolve()), 'data_hashes': hashes,
        'fitted_on': 'inner train', 'evaluated_on': 'fixed inner validation rows',
        'prior_exposure': 'Both inner partitions originate in previously used TRAIN; no new untouched holdout.',
        'selection': {'index_seed': 20261021, 'indices': indices.tolist(),
            'conditions': np.unique(labels).tolist(), 'per_condition': per_condition,
            'directions_seed': 2026, 'directions_count': 128,
            'real_profile_bytes_sha256': _digest(real, '<f8'),
            'label_bytes_sha256': _digest(labels, '<i8'),
            'array_hash_format': 'C-order little-endian float64 profiles and int64 labels'},
        'generation_seeds': seeds,
        'generation_seed_role': 'independent sampling draws; not training seeds or independent datasets',
        'metric_normalization': {'fitted_on': 'inner train',
            'minimum': scaler['minimum'], 'maximum': scaler['maximum']},
        'source_hashes': {str(path.relative_to(ROOT)): _sha256(path) for path in sources},
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
            check=True, capture_output=True, text=True).stdout.strip(),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'thread_environment': {name: os.environ.get(name)
            for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}},
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    output.mkdir(parents=True, exist_ok=False)
    _json(output / 'manifest.json', manifest)
    summary = {'schema_version': 1, 'role': manifest['role'], 'selection_rows': len(indices),
        'generation_seeds': seeds, 'sampling_sd_ddof': 1 if len(seeds) > 1 else 0,
        'uncertainty_note': 'SD describes generation draws only, not estimation uncertainty or training variation.',
        'methods': {}}
    try:
        for kind in ('mean', 'resample', 'copula'):
            model = fit_baseline(kind, training['x'], training['y'])
            draws = []
            for seed in seeds:
                fake = model.sample(labels, seed)
                scores = physical_metrics(real, fake, labels, scaler['maximum'] - scaler['minimum'])
                draws.append({'generation_seed': seed, 'metrics': scores,
                    'generated_profile_bytes_sha256': _digest(fake, '<f8')})
                print(f"method={kind} draw={seed} normalized_SW1="
                      f"{scores['macro']['normalized_profile_sw1']:.8f}", flush=True)
            _json(output / f'{kind}.json', {'method': kind, 'fitted_on': 'inner train', 'draws': draws})
            summary['methods'][kind] = {
                name: {'mean': float(np.mean([draw['metrics']['macro'][name] for draw in draws])),
                       'sampling_sd': float(np.std([draw['metrics']['macro'][name] for draw in draws],
                            ddof=summary['sampling_sd_ddof']))}
                for name in METRICS}
        _json(output / 'summary.json', summary)
        manifest.update(status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat(),
            artifact_hashes={name: _sha256(output / name) for name in
                ('mean.json', 'resample.json', 'copula.json', 'summary.json')})
        _json(output / 'manifest.json', manifest)
        return summary
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        _json(output / 'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261022, 20261023, 20261024])
    parser.add_argument('--per-condition', type=int, default=256)
    args = parser.parse_args()
    result = evaluate(args.data, args.output, args.seeds, args.per_condition)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
