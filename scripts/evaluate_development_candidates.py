#!/usr/bin/env python3
"""Fresh-draw diagnostics on the already used inner-validation selection.

This is post-test development. New latent draws assess sampling sensitivity;
they do not create new held-out meters or independent confirmation. No model is
retrained, reselected, clipped or calibrated by this evaluator.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy
import torch

from fluvius_cgan import development, flow, reflection
from fluvius_cgan.baselines import fit_baseline
from fluvius_cgan.marginals import load_marginals
from fluvius_cgan.training import _json, _read_data, _sha256


ROOT = Path(__file__).resolve().parents[1]
QUANTILES = [.01, .05, .5, .95, .99]
SELECTION_FIELDS = ('indices', 'conditions', 'index_seed', 'directions_seed', 'directions_count')
SUMMARY_METRICS = ('normalized_profile_sw1', 'daily_energy_w1_kwh', 'peak_w1_kw',
                   'ramp_w1_kw', 'mean_profile_rmse_kwh', 'diversity_ratio',
                   'acf_lag1_w1', 'acf_lag6_w1')
TAIL_ERRORS = ('pointwise_quantile_mae_kwh', 'daily_energy_quantile_mae_kwh',
               'peak_quantile_mae_kw', 'absolute_ramp_quantile_mae_kw',
               'covariance_rmse_kwh2', 'export_cell_fraction_abs_error',
               'export_day_fraction_abs_error')


def _read(path):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON value in {path}: {value}')
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def _digest(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order='C')).hexdigest()


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _profiles(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 24 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected nonempty finite physical [N,24] profiles')
    return values


def _row_bytes(values):
    values = _profiles(values)
    if np.max(np.abs(values)) > np.finfo(np.float32).max:
        raise ValueError('Profiles exceed finite float32 copy-check precision')
    array = np.ascontiguousarray(values, dtype='<f4')
    return array.view(np.dtype((np.void, 24 * 4))).ravel()


class CopyIndex:
    """Full-TRAIN membership after explicit little-endian float32 quantization."""

    def __init__(self, training):
        self.training_rows = len(training)
        self.rows = np.unique(_row_bytes(training))

    def matches(self, generated):
        rows = _row_bytes(generated)
        positions = np.searchsorted(self.rows, rows)
        matched = positions < len(self.rows)
        matched[matched] = self.rows[positions[matched]] == rows[matched]
        return matched


def _export(values):
    if not len(values):
        return {'profiles': 0, 'export_cell_fraction': None, 'export_day_fraction': None,
                'negative_daily_energy_fraction': None, 'mean_export_magnitude_kwh_per_day': None}
    return {'profiles': len(values), 'export_cell_fraction': float(np.mean(values < 0)),
            'export_day_fraction': float(np.mean(np.any(values < 0, axis=1))),
            'negative_daily_energy_fraction': float(np.mean(values.sum(axis=1) < 0)),
            'mean_export_magnitude_kwh_per_day': float(np.maximum(-values, 0).sum(axis=1).mean())}


def _description(values):
    return {'pointwise_quantiles_kwh': np.quantile(values, QUANTILES, axis=0).tolist(),
            'daily_energy_quantiles_kwh': np.quantile(values.sum(axis=1), QUANTILES).tolist(),
            'peak_quantiles_kw': np.quantile(values.max(axis=1), QUANTILES).tolist(),
            'signed_ramp_quantiles_kw': np.quantile(np.diff(values, axis=1), QUANTILES).tolist(),
            'absolute_ramp_quantiles_kw': np.quantile(np.abs(np.diff(values, axis=1)), QUANTILES).tolist(),
            **_export(values)}


def tail_diagnostics(real, fake, labels, copies):
    real, fake = _profiles(real), _profiles(fake)
    labels, copies = np.asarray(labels), np.asarray(copies)
    if real.shape != fake.shape or labels.shape != (len(real),) or copies.shape != (len(real),):
        raise ValueError('Tail diagnostics require aligned real/fake profiles, labels and copy flags')
    if labels.dtype.kind not in 'iu' or np.any((labels < 0) | (labels >= 64)) or copies.dtype != np.bool_:
        raise ValueError('Expected condition integers in [0,64) and boolean copy flags')
    conditions = {}
    for label in np.unique(labels):
        mask = labels == label
        a, b = real[mask], fake[mask]
        first, second = _description(a), _description(b)
        # Remove a shared offset first so repeated decimal constants keep zero covariance.
        centered_a, centered_b = a - a[:1], b - b[:1]
        centered_a -= centered_a.mean(axis=0)
        centered_b -= centered_b.mean(axis=0)
        covariance_a = centered_a.T @ centered_a / len(a)
        covariance_b = centered_b.T @ centered_b / len(b)
        difference = covariance_b - covariance_a
        denominator = float(np.linalg.norm(covariance_a))
        errors = {
            'pointwise_quantile_mae_kwh': float(np.mean(np.abs(np.asarray(first['pointwise_quantiles_kwh']) - second['pointwise_quantiles_kwh']))),
            'daily_energy_quantile_mae_kwh': float(np.mean(np.abs(np.asarray(first['daily_energy_quantiles_kwh']) - second['daily_energy_quantiles_kwh']))),
            'peak_quantile_mae_kw': float(np.mean(np.abs(np.asarray(first['peak_quantiles_kw']) - second['peak_quantiles_kw']))),
            'absolute_ramp_quantile_mae_kw': float(np.mean(np.abs(np.asarray(first['absolute_ramp_quantiles_kw']) - second['absolute_ramp_quantiles_kw']))),
            'covariance_rmse_kwh2': float(np.sqrt(np.mean(difference ** 2))),
            'covariance_relative_frobenius': float(np.linalg.norm(difference) / denominator) if denominator else None,
            'export_cell_fraction_abs_error': abs(first['export_cell_fraction'] - second['export_cell_fraction']),
            'export_day_fraction_abs_error': abs(first['export_day_fraction'] - second['export_day_fraction']),
        }
        conditions[str(int(label))] = {'source_pv_flag': int(label // 8 % 2),
            'real': first, 'fake': second, 'errors': errors,
            'quantized_exact_training_matches': int(copies[mask].sum()),
            'quantized_exact_training_match_fraction': float(copies[mask].mean())}
    pv0 = labels // 8 % 2 == 0
    return {'quantile_probabilities': QUANTILES, 'quantile_method': 'numpy linear interpolation',
        'interval_hours': 1., 'covariance_ddof': 0,
        'ramp_definition': 'signed or absolute adjacent-hour power differences; no daily wrap',
        'macro': {key: float(np.mean([value['errors'][key] for value in conditions.values()])) for key in TAIL_ERRORS},
        'per_condition': conditions,
        'copying': {'quantized_exact_matches': int(copies.sum()),
            'quantized_exact_match_fraction': float(copies.mean()),
            'definition': 'Full 24-coordinate row-byte equality against any inner-TRAIN row after little-endian float32 quantization; signed-zero byte patterns remain distinct.',
            'interpretation': 'A precision-specific copying diagnostic; absence of exact matches does not establish novelty or privacy.'},
        'source_pv0': {'real': _export(real[pv0]), 'fake': _export(fake[pv0]),
            'interpretation': 'Descriptive behavior for source PV=0 groups, not a physical-impossibility or equipment-ground-truth test.'}}


def _verify(directory, required):
    path = directory / 'manifest.json'
    manifest = _read(path)
    if manifest.get('status') != 'completed':
        raise ValueError(f'Requires a completed manifest: {directory}')
    hashes = manifest.get('artifact_hashes', {})
    if not set(required) <= set(hashes):
        raise ValueError(f'Missing artifact hashes: {directory}')
    for name, digest in hashes.items():
        if Path(name).name != name or _sha256(directory / name) != digest:
            raise ValueError(f'Artifact hash mismatch: {directory / name}')
    return manifest


def _shared(manifest, hashes, selection, scaler):
    if manifest['data_hashes'] != hashes:
        raise ValueError('Development datasets differ')
    for key in SELECTION_FIELDS:
        if manifest['selection'].get(key) != selection.get(key):
            raise ValueError(f'Development selection differs at {key}')
    if any(manifest['metric_normalization'][key] != scaler[key] for key in ('minimum', 'maximum')):
        raise ValueError('Development physical normalization differs')


def _candidate(directory, index):
    manifest = _verify(directory, ('best.pt', 'last.pt', 'history.json', 'best_validation.json'))
    schema = manifest.get('schema_version')
    if schema == development.SCHEMA:
        method, implementation = 'conditional_cgan', development
    elif schema == flow.SCHEMA and manifest.get('method') == 'conditional_flow_matching':
        _verify(directory, ('solver_sensitivity.json',))
        method, implementation = 'conditional_flow_matching', flow
    elif schema == reflection.SCHEMA and manifest.get('method') == 'conditional_reflection_copula':
        artifact = manifest.get('marginals', {}).get('artifact')
        if not isinstance(artifact, str) or Path(artifact).name != artifact or artifact in ('', '.', '..'):
            raise ValueError('Invalid reflection marginal artifact name')
        _verify(directory, (artifact, 'best_probability_diagnostics.json',
                            'independent_marginals.json', 'solver_sensitivity.json'))
        method, implementation = 'conditional_reflection_copula', reflection
    else:
        raise ValueError(f'Unsupported candidate checkpoint schema: {directory}')
    path = directory / 'best.pt'
    if implementation is reflection:
        checkpoint, _, fitted_marginals = reflection._load(path)
    else:
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    current, previous = implementation._config(manifest['config']), implementation._config(checkpoint['config'])
    if (checkpoint.get('schema_version') != schema or checkpoint['step'] != manifest['best_step']
            or checkpoint['best_step'] != checkpoint['step']
            or checkpoint['best_score'] != manifest['best_validation_sw1']
            or checkpoint['data_hashes'] != manifest['data_hashes']
            or checkpoint['selection'] != manifest['selection']
            or (implementation is not reflection and checkpoint['representation'] != manifest['representation'])
            or {k: v for k, v in current.items() if k != 'steps'} !=
               {k: v for k, v in previous.items() if k != 'steps'}):
        raise ValueError(f'Selected checkpoint and manifest disagree: {directory}')
    if manifest['completed_steps'] != current['steps'] or checkpoint['step'] > current['steps']:
        raise ValueError(f'Candidate did not complete its declared budget: {directory}')
    if implementation is reflection:
        marginal = checkpoint['marginals']
        if (marginal != manifest.get('marginals')
                or marginal['sha256'] != manifest['artifact_hashes'][marginal['artifact']]
                or sum(group['training_profiles'] for group in fitted_marginals.summary()['per_condition'].values())
                   != manifest['available_training_rows']):
            raise ValueError('Reflection checkpoint marginal fit differs from manifest')
        source_report = _read(directory / 'independent_marginals.json')
        if (source_report.get('method') != 'independent_marginals'
                or source_report.get('marginals') != marginal
                or source_report.get('data_hashes') != manifest['data_hashes']
                or source_report.get('generation_seed') != manifest['selection']['generation_seed']
                or source_report.get('conditions') != manifest['selection']['conditions']
                or source_report.get('selection_rows') != len(manifest['selection']['indices'])
                or source_report.get('used_for_selection') is not False):
            raise ValueError('Saved reflection independent-marginal report disagrees with fit or selection')
        probability = _read(directory / 'best_probability_diagnostics.json')
        if (probability['solver'].get('method') != 'reflected Euler'
                or probability['solver'].get('steps') != current['solver_steps']
                or probability['solver'].get('nfe') != current['solver_steps']
                or probability['solver'].get('max_time') != current['max_time']
                or probability['probabilities'].get('profiles') != len(manifest['selection']['indices'])
                or sorted(map(int, probability['probabilities']['per_condition'])) != manifest['selection']['conditions']):
            raise ValueError('Saved reflection probability diagnostics disagree with selection solver')
        sensitivity = _read(directory / 'solver_sensitivity.json')
        if (sensitivity.get('checkpoint_sha256') != _sha256(path)
                or sensitivity.get('selected_checkpoint_step') != checkpoint['step']
                or sensitivity.get('generation_seed') != manifest['selection']['generation_seed']
                or sensitivity.get('common_initial_uniforms') is not True
                or sensitivity.get('used_for_selection') is not False
                or sensitivity['selection_solver'].get('steps') != current['solver_steps']
                or sensitivity['selection_solver'].get('nfe') != current['solver_steps']
                or sensitivity['sensitivity_solver'].get('steps') != 2 * current['solver_steps']
                or sensitivity['sensitivity_solver'].get('nfe') != 2 * current['solver_steps']
                or sensitivity['selection_solver'].get('metrics') != _read(directory / 'best_validation.json')
                or sensitivity['selection_solver'].get('diagnostics') != probability):
            raise ValueError('Saved reflection solver sensitivity disagrees with selected checkpoint')
    prior = None
    if implementation is flow:
        prior = checkpoint.get('source_prior')
        if prior != manifest.get('source_prior'):
            raise ValueError(f'Selected checkpoint and manifest source prior disagree: {directory}')
        validate_prior = getattr(implementation, '_checkpoint_prior', None)
        if validate_prior is not None:
            validate_prior(checkpoint, current)
        elif prior is not None or current.get('source_prior', 'isotropic') != 'isotropic':
            raise ValueError('Installed flow implementation cannot validate this source prior')
        if prior is not None:
            if prior['kind'] == 'training_correlation':
                fit = prior.get('fit') or {}
                if fit.get('training_npz_sha256') != manifest['data_hashes']['train.npz']:
                    raise ValueError('Source prior training hash differs from candidate TRAIN')
                if (fit.get('fitted_on') != 'inner train'
                        or fit.get('rows') != manifest['available_training_rows']
                        or fit.get('condition_specific') is not False):
                    raise ValueError('Source prior must describe the full unconditional inner-TRAIN fit')
            _verify(directory, ('prior_baseline.json',))
            source_report = _read(directory / 'prior_baseline.json')
            if (source_report.get('method') != 'source_only'
                    or source_report.get('source_prior') != prior
                    or source_report.get('data_hashes') != manifest['data_hashes']
                    or source_report.get('generation_seed') != manifest['selection']['generation_seed']
                    or source_report.get('conditions') != manifest['selection']['conditions']
                    or source_report.get('selection_rows') != len(manifest['selection']['indices'])
                    or source_report.get('used_for_selection') is not False):
                raise ValueError('Saved source-only report disagrees with candidate source prior or selection')
    record = {'method_id': f'candidate_{index:03d}', 'method': method,
        'run_directory': str(directory), 'manifest_sha256': _sha256(directory / 'manifest.json'),
        'checkpoint': str(path), 'checkpoint_sha256': _sha256(path),
        'training_seed': current['seed'], 'config': current,
        'selected_step': checkpoint['step'], 'completed_steps': manifest['completed_steps'],
        'training_runtime': manifest['runtime'], 'training_source_sha256': manifest['source_sha256']}
    if implementation is flow:
        record.update(representation=checkpoint['representation'], source_prior=prior,
                      source_prior_sha256=_json_digest(prior))
    elif implementation is reflection:
        record.update(marginals=checkpoint['marginals'],
                      marginal_artifact=str(directory / checkpoint['marginals']['artifact']))
    return record, manifest, implementation.generate


def _source_controls(candidates):
    """Deduplicate matched Gaussian controls, never count them as training replicates."""
    controls = {}
    for candidate, _, _ in candidates:
        if candidate['method'] != 'conditional_flow_matching':
            continue
        prior, representation = candidate['source_prior'], candidate['representation']
        kind = 'isotropic' if prior is None else prior['kind']
        # The generator applies this float32 factor and the physical inverse affine map.
        # Metadata differences (including legacy None) do not create a different draw.
        factor = np.eye(24) if prior is None else np.asarray(prior['cholesky'])
        distribution = {'source_prior_kind': kind, 'factor_float32_sha256': _digest(factor, '<f4'),
                        'offset_float64_sha256': _digest(representation['offset'], '<f8'),
                        'scale_float64_sha256': _digest(representation['scale'], '<f8')}
        key = _json_digest(distribution)
        if key not in controls:
            controls[key] = {'method_id': f'source_{len(controls):03d}',
                'method': f'source_only_{kind}', 'training_seed': None,
                'source_prior_kind': kind, 'source_prior': prior,
                'source_prior_sha256': candidate['source_prior_sha256'],
                'representation': representation, 'source_distribution': distribution,
                'source_distribution_sha256': key, 'originating_candidates': [],
                'interpretation': 'Common unconditional Gaussian source with the candidate physical inverse transform; no learned map or equipment conditioning. Equivalent sources across training seeds share one diagnostic, not independent replicates.'}
        controls[key]['originating_candidates'].append({field: candidate[field] for field in
            ('method_id', 'run_directory', 'manifest_sha256', 'checkpoint_sha256', 'source_prior_sha256')})
    return list(controls.values())


def _independent_controls(candidates):
    """One decoder-only uniform control per identical frozen marginal artifact."""
    controls = {}
    for candidate, _, _ in candidates:
        if candidate['method'] != 'conditional_reflection_copula':
            continue
        marginal = candidate['marginals']
        distribution = {'marginal_artifact_sha256': marginal['sha256'],
            'training_npz_sha256': marginal['training_npz_sha256'],
            'algorithm': 'numpy default_rng(seed).random([N,24],dtype=float32); conditional empirical step inverse'}
        key = _json_digest(distribution)
        if key not in controls:
            controls[key] = {'method_id': f'independent_{len(controls):03d}',
                'method': 'independent_marginals', 'training_seed': None, 'fitted_on': 'inner train',
                'marginals': marginal, 'marginal_artifact': candidate['marginal_artifact'],
                'source_distribution': distribution, 'source_distribution_sha256': key,
                'originating_candidates': [],
                'interpretation': 'Independent uniforms through the same conditional empirical inverse; no learned dependence. Shared fitted marginals across training seeds create one control, not extra replicates.'}
        controls[key]['originating_candidates'].append({field: candidate[field] for field in
            ('method_id', 'run_directory', 'manifest_sha256', 'checkpoint_sha256')})
    return list(controls.values())


def _stats(values):
    if any(value is None for value in values):
        return {'mean': None, 'sampling_sd': None, 'defined_draws': sum(value is not None for value in values)}
    return {'mean': float(np.mean(values)), 'sampling_sd': float(np.std(values, ddof=1 if len(values) > 1 else 0)),
            'defined_draws': len(values)}


def _csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(data, runs, output, seeds=(20261023, 20261024), baselines=None):
    data, output = Path(data).resolve(), Path(output).resolve()
    runs, seeds = [Path(path).resolve() for path in runs], list(seeds)
    if output.exists():
        raise FileExistsError(output)
    if not runs or len(runs) != len(set(runs)):
        raise ValueError('Declare a nonempty list of distinct candidate directories')
    if (not seeds or len(seeds) != len(set(seeds))
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError('Generation seeds must be distinct unsigned 32-bit integers')
    training, validation, scaler, metadata, hashes = _read_data(data)
    if metadata.get('development', {}).get('source_partition_role') != 'original_train':
        raise ValueError('Requires an inner development split sourced from original TRAIN')
    candidates = [_candidate(path, index) for index, path in enumerate(runs)]
    selection = candidates[0][1]['selection']
    for _, manifest, _ in candidates:
        _shared(manifest, hashes, selection, scaler)
        if manifest['selection']['generation_seed'] in seeds:
            raise ValueError('Fresh generation draws must exclude the checkpoint selection seed')
    if selection.get('directions_seed') != 2026 or selection.get('directions_count') != 128:
        raise ValueError('Unsupported selection projection settings')
    indices = np.asarray(selection['indices'])
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
            or len(indices) != len(np.unique(indices)) or np.any((indices < 0) | (indices >= len(validation['x'])))):
        raise ValueError('Invalid saved selection indices')
    real, labels = validation['x'][indices].astype(np.float64), validation['y'][indices]
    if np.unique(labels).tolist() != selection['conditions']:
        raise ValueError('Saved selection condition coverage disagrees with data')
    source_controls = _source_controls(candidates)
    independent_controls = _independent_controls(candidates)
    existing_baseline_draws = {}
    baseline_source = None
    if baselines is not None:
        baseline_dir = Path(baselines).resolve()
        baseline_manifest = _verify(baseline_dir, ('mean.json', 'resample.json', 'copula.json', 'summary.json'))
        _shared(baseline_manifest, hashes, selection, scaler)
        if (baseline_manifest['selection']['real_profile_bytes_sha256'] != _digest(real, '<f8')
                or baseline_manifest['selection']['label_bytes_sha256'] != _digest(labels, '<i8')):
            raise ValueError('Existing baseline selection array hashes differ')
        for kind in ('mean', 'resample', 'copula'):
            report = _read(baseline_dir / f'{kind}.json')
            if report['method'] != kind:
                raise ValueError('Baseline report method mismatch')
            for seed in seeds:
                matches = [draw for draw in report['draws'] if draw['generation_seed'] == seed]
                if len(matches) != 1:
                    raise ValueError('Existing baseline reports must contain each fresh draw exactly once')
                existing_baseline_draws[kind, seed] = matches[0]
        baseline_source = {'directory': str(baseline_dir),
                           'manifest_sha256': _sha256(baseline_dir / 'manifest.json')}
    sources = [Path(__file__), *sorted((ROOT / 'src' / 'fluvius_cgan').glob('*.py'))]
    manifest = {'schema_version': 1, 'phase': 1, 'status': 'running',
        'role': 'post-test development fresh-generation diagnostics', 'test_performance_access': False,
        'data_directory': str(data), 'data_hashes': hashes, 'selection': selection,
        'real_profile_bytes_sha256': _digest(real, '<f8'), 'label_bytes_sha256': _digest(labels, '<i8'),
        'generation_seeds': seeds, 'generation_seed_role': 'fresh sampling draws, not independent meters or training seeds',
        'methods': [record for record, _, _ in candidates] + source_controls + independent_controls,
        'existing_baseline_reports': baseline_source,
        'source_controls': 'One matched source-only control per declared prior kind, applied float32 factor and physical inverse affine transform; legacy missing priors mean isotropic. Use identical labels, order and fresh seeds.',
        'baseline_fit': 'Fit each mean/resample/copula once on the identical full inner TRAIN; generate the same fresh draws.',
        'metric_normalization': {'fitted_on': 'train', 'minimum': scaler['minimum'], 'maximum': scaler['maximum']},
        'clipping': False, 'reselection': False, 'quantile_probabilities': QUANTILES,
        'source_sha256': {str(path.relative_to(ROOT)): _sha256(path) for path in sources},
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
            check=True, capture_output=True, text=True).stdout.strip(),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'torch': str(torch.__version__), 'generation_device': 'cpu', 'torch_threads': 1,
            'thread_environment': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}},
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'completed_draws': 0}
    output.mkdir(parents=True, exist_ok=False)
    _json(output / 'manifest.json', manifest)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    summary = {'schema_version': 1, 'role': manifest['role'], 'generation_seeds': seeds,
        'selection_profiles': len(real), 'conditions': selection['conditions'],
        'uncertainty_note': 'Sampling SD is across fresh generation draws only; no confidence, privacy, or untouched-validation claim.',
        'methods': {}}
    rows, condition_rows = [], []
    try:
        index = CopyIndex(training['x'])
        manifest['copy_reference'] = {'partition': 'full inner TRAIN', 'rows': index.training_rows,
                                       'unique_quantized_rows': len(index.rows)}
        _json(output / 'manifest.json', manifest)
        tasks = [(record, lambda seed, r=record, f=generate: f(Path(r['checkpoint']), labels, seed))
                 for record, _, generate in candidates]
        tasks.extend((record, lambda seed, r=record: flow.source_samples(
            r['representation'], labels, seed, r['source_prior'])) for record in source_controls)
        for record in independent_controls:
            fitted = load_marginals(record['marginal_artifact'], record['marginals']['sha256'])
            tasks.append((record, lambda seed, decoder=fitted: reflection.independent_marginals(decoder, labels, seed)))
        for kind in ('mean', 'resample', 'copula'):
            model = fit_baseline(kind, training['x'], training['y'])
            record = {'method_id': f'baseline_{kind}', 'method': kind, 'training_seed': None, 'fitted_on': 'inner train'}
            tasks.append((record, lambda seed, fitted=model: fitted.sample(labels, seed)))
        for record, sample in tasks:
            method_id, method = record['method_id'], record['method']
            report = dict(record, draws=[])
            for seed in seeds:
                started = time.perf_counter()
                fake = sample(seed)
                generation_seconds = time.perf_counter() - started
                scores = development.physical_metrics(real, fake, labels, scaler['maximum'] - scaler['minimum'])
                diagnostics = tail_diagnostics(real, fake, labels, index.matches(fake))
                generated_hash = _digest(fake, '<f8')
                existing = existing_baseline_draws.get((method, seed))
                if existing is not None and (existing['generated_profile_bytes_sha256'] != generated_hash
                                             or existing['metrics'] != scores):
                    raise ValueError(f'Regenerated baseline differs from existing report: {method}, seed {seed}')
                draw = {'generation_seed': seed, 'generation_seconds_including_checkpoint_load': generation_seconds,
                    'generated_profile_bytes_sha256': generated_hash, 'metrics': scores, 'diagnostics': diagnostics,
                    'existing_baseline_report_match': True if existing is not None else None}
                report['draws'].append(draw)
                _json(output / f'{method_id}.json', report)
                row = {'method_id': method_id, 'method': method, 'training_seed': record['training_seed'],
                       'generation_seed': seed, **scores['macro'], **diagnostics['macro'],
                       'quantized_exact_training_match_fraction': diagnostics['copying']['quantized_exact_match_fraction']}
                rows.append(row)
                for label, values in diagnostics['per_condition'].items():
                    condition_rows.append({'method_id': method_id, 'method': method, 'training_seed': record['training_seed'],
                        'generation_seed': seed, 'condition': int(label), 'source_pv_flag': values['source_pv_flag'],
                        **values['errors'], 'quantized_exact_training_match_fraction': values['quantized_exact_training_match_fraction'],
                        **{f'{side}_{key}': values[side][key] for side in ('real', 'fake') for key in
                           ('profiles', 'export_cell_fraction', 'export_day_fraction', 'negative_daily_energy_fraction', 'mean_export_magnitude_kwh_per_day')}})
                manifest['completed_draws'] += 1
                _json(output / 'manifest.json', manifest)
                print(f"{method_id} draw={seed} normalized_SW1={scores['macro']['normalized_profile_sw1']:.8f} "
                      f"quantized_copy_fraction={diagnostics['copying']['quantized_exact_match_fraction']:.4f}", flush=True)
            summary['methods'][method_id] = {'method': method, 'training_seed': record['training_seed'],
                'metrics': {key: _stats([draw['metrics']['macro'][key] for draw in report['draws']]) for key in SUMMARY_METRICS},
                'tail_errors': {key: _stats([draw['diagnostics']['macro'][key] for draw in report['draws']]) for key in TAIL_ERRORS},
                'quantized_exact_training_match_fraction': _stats([
                    draw['diagnostics']['copying']['quantized_exact_match_fraction'] for draw in report['draws']])}
        _json(output / 'summary.json', summary)
        _csv(output / 'metrics.csv', rows)
        _csv(output / 'condition_diagnostics.csv', condition_rows)
        manifest.update(status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat(),
            artifact_hashes={path.name: _sha256(path) for path in sorted(output.iterdir())
                             if path.is_file() and path.name != 'manifest.json'})
        _json(output / 'manifest.json', manifest)
        return summary
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}',
                        finished_at_utc=datetime.now(timezone.utc).isoformat(), artifact_hashes={})
        # Preserve both partial scientific artifacts and the original failure.
        # A secondary read/write problem must not replace that exception.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baselines', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261023, 20261024])
    args = parser.parse_args()
    evaluate(args.data, args.runs, args.output, args.seeds, args.baselines)


if __name__ == '__main__':
    main()
