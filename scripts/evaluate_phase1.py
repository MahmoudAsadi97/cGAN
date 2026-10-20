#!/usr/bin/env python3
"""Run the frozen Phase 1 final evaluation, preserving every subprocess result.

This command accesses test performance. It does not train or select models.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from fluvius_cgan.study import (
    DATA_FILES, GENERATION_SEEDS, TRAINING_SEEDS, _selected_final_checkpoint,
    _test_protocol, sha256, write_json,
)


ROOT = Path(__file__).resolve().parents[1]
METHODS = ('conditional', 'unconditional', 'mean', 'resample', 'copula')
METRICS = ('normalized_profile_sw1', 'daily_energy_w1_kwh')


def _json(path):
    def invalid(value):
        raise ValueError(f'nonfinite JSON value in {path}: {value}')
    value = json.loads(Path(path).read_text(), parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError(f'expected a JSON object: {path}')
    return value


def _time():
    return datetime.now(timezone.utc).isoformat()


def preflight(data, protocol, runs):
    """Verify frozen settings and ten completed selected artifacts without metrics."""
    import torch
    frozen = _test_protocol(data, protocol, GENERATION_SEEDS)
    expected = {'schema_version': 1, 'training_seeds': TRAINING_SEEDS,
                'projection_count': 128, 'projection_seed': 2026}
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError(f'frozen Phase 1 protocol differs at {key}')
    if frozen.get('methods') != list(METHODS):
        raise ValueError('frozen protocol must declare all five Phase 1 methods')
    uncertainty = frozen.get('uncertainty', {})
    for key, value in {
        'bootstrap_replicates': 499, 'bootstrap_seed': 20261020,
        'exemplar_training_seed': 11, 'exemplar_generation_seed': 31001,
        'comparators': ['resample', 'copula'], 'metrics': list(METRICS),
        'nearest_neighbor_query_cap_per_condition': 256, 'nearest_neighbor_seed': 41001,
    }.items():
        if uncertainty.get(key) != value:
            raise ValueError(f'frozen Phase 1 uncertainty differs at {key}')
    checkpoints, manifests = {}, {}
    for method in METHODS[:2]:
        for seed in TRAINING_SEEDS:
            checkpoint = Path(runs) / method / f'seed_{seed}' / 'best.pt'
            manifest_path = checkpoint.parent / 'manifest.json'
            manifest = _json(manifest_path)
            if manifest.get('status') != 'completed':
                raise ValueError(f'run is not completed: {manifest_path}')
            config = manifest.get('config', {})
            if config.get('seed') != seed or config.get('unconditional') is not (method == 'unconditional'):
                raise ValueError(f'run identity differs from its declared arm/seed: {manifest_path}')
            for name in ('train.npz', 'val.npz', 'scaler.json', 'metadata.json'):
                if manifest.get('data_hashes', {}).get(name) != frozen['data_hashes'][name]:
                    raise ValueError(f'completed run dataset mismatch: {manifest_path}: {name}')
            digest = sha256(checkpoint)
            if digest != manifest.get('artifact_hashes', {}).get('best.pt'):
                raise ValueError(f'selected artifact hash mismatch: {checkpoint}')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            _selected_final_checkpoint(checkpoint, payload, frozen)
            key = f'{method}_seed_{seed}'
            checkpoints[key] = {'path': str(checkpoint.resolve()), 'sha256': digest}
            manifests[key] = {'path': str(manifest_path.resolve()), 'sha256': sha256(manifest_path)}
    sources = [Path(__file__).resolve(), *sorted((ROOT / 'src' / 'fluvius_cgan').glob('*.py'))]
    provenance = {
        'protocol': {'path': str(Path(protocol).resolve()), 'sha256': sha256(protocol)},
        'data_hashes': frozen['data_hashes'], 'checkpoints': checkpoints,
        'run_manifests': manifests,
        'source_hashes': {str(path.relative_to(ROOT)): sha256(path) for path in sources},
    }
    return frozen, provenance


def _command_environment():
    environment = os.environ.copy()
    environment.update({name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    return environment


def _run_logged(name, command, output):
    stdout_path = output / 'logs' / f'{name}.stdout.log'
    stderr_path = output / 'logs' / f'{name}.stderr.log'
    record = {'name': name, 'command': command, 'started_at_utc': _time(),
              'stdout': str(stdout_path), 'stderr': str(stderr_path)}
    with stdout_path.open('x') as stdout, stderr_path.open('x') as stderr:
        try:
            process = subprocess.run(command, env=_command_environment(), stdout=stdout,
                                     stderr=stderr, cwd=ROOT, check=False)
            record['returncode'] = process.returncode
        except Exception as error:
            record.update(returncode=None, error=f'{type(error).__name__}: {error}')
            stderr.write(record['error'] + '\n')
    record['finished_at_utc'] = _time()
    return record


def _report_paths(output):
    names = [f'{method}_seed_{seed}' for method in METHODS[:2] for seed in TRAINING_SEEDS]
    names += list(METHODS[2:])
    return {name: output / 'reports' / f'{name}.json' for name in names}


def run_final(args):
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError(f'final evaluation requires a new or empty output directory: {args.output}')
    frozen, provenance = preflight(args.data, args.protocol, args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'logs').mkdir()
    (args.output / 'reports').mkdir()
    write_json(args.output / 'execution_plan.json', {
        'schema_version': 1, 'started_at_utc': _time(), 'workers': args.workers,
        'provenance': provenance, 'thread_environment': {name: '1' for name in
            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
    })
    write_json(args.output / 'protocol.snapshot.json', frozen)
    paths = _report_paths(args.output)
    jobs = []
    for name, path in paths.items():
        method = provenance['checkpoints'][name]['path'] if name in provenance['checkpoints'] else name
        command = [sys.executable, '-m', 'fluvius_cgan', 'evaluate', '--data', str(args.data),
                   '--method', method, '--split', 'test', '--protocol', str(args.protocol),
                   '--seeds', *map(str, frozen['generation_seeds']), '--output', str(path)]
        jobs.append((name, command))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_logged, name, command, args.output) for name, command in jobs]
        records = [future.result() for future in futures]
    for record in records:
        path = paths[record['name']]
        record['report_sha256'] = sha256(path) if path.is_file() else None
    write_json(args.output / 'evaluation_execution.json', {
        'workers': args.workers, 'jobs': records, 'finished_at_utc': _time(),
    })
    if any(record['returncode'] != 0 or record['report_sha256'] is None for record in records):
        raise ValueError('final evaluation contains failures; all logs and completed reports are preserved')
    command = [sys.executable, '-m', 'fluvius_cgan', 'summarize', '--reports',
               *map(str, paths.values()), '--output', str(args.output / 'summary.json')]
    summary = _run_logged('summarize', command, args.output)
    write_json(args.output / 'summary_execution.json', summary)
    if summary['returncode'] != 0 or not (args.output / 'summary.json').is_file():
        raise ValueError('summary failed; see preserved summary logs')
    command = [sys.executable, str(Path(__file__).resolve()), '--data', str(args.data),
               '--protocol', str(args.protocol), '--runs', str(args.runs),
               '--output', str(args.output), '--inference-only']
    inference = _run_logged('inference', command, args.output)
    write_json(args.output / 'inference_execution.json', inference)
    if inference['returncode'] != 0 or not (args.output / 'inference.json').is_file():
        raise ValueError('uncertainty/memorization stage failed; see preserved inference logs')
    write_json(args.output / 'completed.json', {
        'completed_at_utc': _time(), 'protocol_sha256': provenance['protocol']['sha256'],
        'summary_sha256': sha256(args.output / 'summary.json'),
        'inference_sha256': sha256(args.output / 'inference.json'),
    })


def training_seed_differences(reports, seeds):
    """Paired point differences of generation means; SD is across trained seeds."""
    lookup = {}
    for report in reports:
        seed = (report.get('model') or {}).get('config', {}).get('seed')
        key = (report['method'], seed)
        if key in lookup:
            raise ValueError(f'duplicate report for method/seed: {key}')
        lookup[key] = report
    result = {}
    for comparator in METHODS[1:]:
        metrics = {}
        for metric in METRICS:
            paired = []
            for seed in seeds:
                a = lookup[('conditional', seed)]['macro_summary'][metric]['mean']
                b = lookup[(comparator, seed if comparator == 'unconditional' else None)]['macro_summary'][metric]['mean']
                paired.append({'training_seed': seed, 'a': a, 'b': b, 'difference': a - b})
            differences = [row['difference'] for row in paired]
            metrics[metric] = {'per_training_seed': paired,
                               'mean_difference': statistics.mean(differences),
                               'training_seed_sd': statistics.stdev(differences),
                               'n_training_seeds': len(seeds)}
        result[comparator] = metrics
    return result


def _access_event(output, event, protocol_hash, **extra):
    with (output / 'test_access.jsonl').open('a') as handle:
        handle.write(json.dumps({'event': event, 'time_utc': _time(),
                                 'protocol_sha256': protocol_hash, **extra}, allow_nan=False) + '\n')


def _load_reports(output, frozen, provenance):
    reports, hashes = [], {}
    execution = _json(output / 'evaluation_execution.json')
    expected = {row['name']: row['report_sha256'] for row in execution['jobs'] if row['returncode'] == 0}
    for name, path in _report_paths(output).items():
        if sha256(path) != expected.get(name):
            raise ValueError(f'evaluation report changed or did not complete: {path}')
        report = _json(path)
        if report.get('protocol_sha256') != provenance['protocol']['sha256'] or report.get('split') != 'test':
            raise ValueError(f'evaluation report is not bound to frozen test protocol: {path}')
        if report.get('generation_seeds') != frozen['generation_seeds']:
            raise ValueError(f'generation seeds differ in saved report: {path}')
        expected_method = name.split('_seed_')[0]
        if report.get('method') != expected_method:
            raise ValueError(f'method differs in saved report: {path}')
        checkpoint = provenance['checkpoints'].get(name)
        if checkpoint and report.get('checkpoint_sha256') != checkpoint['sha256']:
            raise ValueError(f'checkpoint differs in saved report: {path}')
        if checkpoint and report.get('model', {}).get('config', {}).get('seed') != int(name.split('_seed_')[1]):
            raise ValueError(f'training seed differs in saved report: {path}')
        expected_data = {filename: frozen['data_hashes'][filename] for filename in
                         ('train.npz', 'test.npz', 'scaler.json', 'metadata.json')}
        if report.get('data_hashes') != expected_data:
            raise ValueError(f'dataset differs or is incomplete in saved report: {path}')
        reports.append(report)
        hashes[name] = sha256(path)
    return reports, hashes


def run_inference(args):
    """One process, one train/test load, five fixed exemplar generated samples."""
    protocol_hash = sha256(args.protocol)
    _access_event(args.output, 'test_inference_started', protocol_hash)
    try:
        if (args.output / 'inference.json').exists():
            raise FileExistsError(args.output / 'inference.json')
        frozen, provenance = preflight(args.data, args.protocol, args.runs)
        plan = _json(args.output / 'execution_plan.json')
        if provenance != plan['provenance']:
            raise ValueError('protocol, inputs, checkpoints or source changed since evaluation preflight')
        if not (args.output / 'summary.json').is_file():
            raise ValueError('inference requires completed evaluation summary')
        reports, report_hashes = _load_reports(args.output, frozen, provenance)
        import numpy as np
        import scipy
        import torch
        from fluvius_cgan.baselines import fit_baseline
        from fluvius_cgan.inference import nearest_neighbor_diagnostic, paired_cluster_comparison
        from fluvius_cgan.study import _verified_data
        from fluvius_cgan.training import generate
        torch.set_num_threads(1)
        train, real, metadata, scaler, hashes = _verified_data(args.data, 'test')
        uncertainty = frozen['uncertainty']
        training_seed = uncertainty['exemplar_training_seed']
        generation_seed = uncertainty['exemplar_generation_seed']
        samples = {}
        for method in METHODS[:2]:
            checkpoint = provenance['checkpoints'][f'{method}_seed_{training_seed}']['path']
            samples[method] = generate(checkpoint, real['y'], generation_seed)
        for method in METHODS[2:]:
            model = fit_baseline(method, train['x'], train['y'])
            samples[method] = model.sample(real['y'], generation_seed)
            del model
        # y = ((technology * 2) + weekend) * 4 + season.
        technology = real['y'] // 8
        training_range = scaler['maximum'] - scaler['minimum']
        denominator = training_range or 1.0
        comparisons = {}
        for method in uncertainty['comparators']:
            comparison = paired_cluster_comparison(
                real['x'], samples['conditional'], samples[method], real['y'],
                real['meter_id'], technology,
                n_bootstrap=uncertainty['bootstrap_replicates'], seed=uncertainty['bootstrap_seed'])
            comparison['observed_difference']['normalized_profile_sw1'] = (
                comparison['observed_difference']['profile_sw1_kwh'] / denominator)
            physical_ci = comparison['ci95']['profile_sw1_kwh']
            comparison['ci95']['normalized_profile_sw1'] = (
                [endpoint / denominator for endpoint in physical_ci] if physical_ci is not None else None)
            comparison['metadata']['normalization'] = {
                'fitted_on': 'train', 'training_range_kwh': training_range,
                'denominator_kwh': denominator, 'zero_range_fallback': 1.0,
                'definition': 'physical profile SW1 A-minus-B and interval endpoints divided by frozen training-only range',
            }
            comparisons[method] = comparison
        proximity = {method: nearest_neighbor_diagnostic(
            train['x'], train['y'], real['x'], real['y'], fake, real['y'],
            max_queries_per_condition=uncertainty['nearest_neighbor_query_cap_per_condition'],
            seed=uncertainty['nearest_neighbor_seed']) for method, fake in samples.items()}
        if preflight(args.data, args.protocol, args.runs)[1] != provenance:
            raise ValueError('inputs or source changed during inference')
        if hashes != {name: frozen['data_hashes'][name] for name in hashes}:
            raise ValueError('loaded inference dataset differs from frozen protocol')
        value = {
            'schema_version': 1, 'phase': 1, 'split': 'test', 'test_performance_access': True,
            'provenance': provenance, 'evaluation_report_hashes': report_hashes,
            'source_sha256': metadata.get('source_sha256'),
            'runtime': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                        'torch': torch.__version__, 'num_threads': 1},
            'exemplar': {'training_seed': training_seed, 'generation_seed': generation_seed,
                         'scope': 'declared seed-11 fitted model and fixed generation draw; interval is not for five-model mean'},
            'paired_cluster_comparisons': comparisons,
            'paired_training_seed_differences': training_seed_differences(reports, frozen['training_seeds']),
            'training_seed_difference_definition': 'A=conditional minus comparator, using means over three matched generation seeds; '
                'sample SD over five paired training seeds; baseline fits and generation estimates fixed across those pairs',
            'nearest_neighbor_diagnostics': proximity,
        }
        write_json(args.output / 'inference.json', value)
        _access_event(args.output, 'test_inference_completed', protocol_hash,
                      report_sha256=sha256(args.output / 'inference.json'))
    except Exception as error:
        _access_event(args.output, 'test_inference_failed', protocol_hash,
                      error_type=type(error).__name__, message=str(error))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'protocol', 'runs', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=3)
    parser.add_argument('--inference-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ('data', 'protocol', 'runs', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    try:
        run_inference(args) if args.inference_only else run_final(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'evaluate_phase1: {type(error).__name__}: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
