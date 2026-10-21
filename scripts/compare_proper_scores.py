#!/usr/bin/env python3
"""Hash-bound fixed-fit proper-score comparisons on reused validation meters.

Average per-profile scores across fixed fits and three ensembles. No generated
profiles are concatenated, models reselected, or score functions refitted.
Every interval is postselection descriptive, conditional on fixed score
functions and ignoring selection and ensemble-generation error.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from fluvius_cgan.meter_uncertainty import meter_contributions, stratified_studentized_bootstrap, SCOPE
from fluvius_cgan.training import _json, _sha256
import evaluate_proper_scores as evaluation

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / 'reports/phase1-development/PROPER_SCORE_COMPARISON_DESIGN.json'
EXPECTED_SEEDS = [20261111, 20261112, 20261113]
CGAN_SEEDS = [301, 401, 501, 601, 701]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    value = json.loads(Path(path).read_text(), parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def _safe(name):
    return isinstance(name, str) and re.fullmatch('[a-z][a-z0-9_]*', name) is not None


def _bound(path, expected):
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected) or _sha256(path) != expected:
        raise ValueError(f'Artifact hash mismatch: {path}')


def _inventory(directory):
    return {str(path.relative_to(directory)): _sha256(path) for path in sorted(directory.rglob('*'))
            if path.is_file() and path != directory / 'manifest.json'}


def _partial_inventory(directory):
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


def _completed(directory, expected=None):
    if expected is not None:
        _bound(directory / 'manifest.json', expected)
    manifest = _read(directory / 'manifest.json')
    if manifest.get('status') != 'completed' or manifest.get('artifact_hash_errors'):
        raise ValueError(f'Requires a completed, fully hashed study: {directory}')
    hashes = manifest.get('artifact_hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Completed receipt lacks artifact hashes')
    for name, digest in hashes.items():
        path = directory / name
        if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Artifact inventory escapes its study directory')
        _bound(path, digest)
    if _inventory(directory) != hashes:
        raise ValueError('Artifact inventory differs from the completed receipt')
    return manifest


def _plan(path):
    plan = _read(path)
    if any(type(plan.get(key)) is not int or plan[key] != 1 for key in ('schema_version', 'phase')):
        raise ValueError('Requires a phase-1 comparison plan')
    binding = plan.get('design', {})
    if set(binding) != {'path', 'sha256'}:
        raise ValueError('Comparison plan must bind its prospective design')
    _bound(Path(binding['path']), binding['sha256'])
    if binding['sha256'] != _sha256(DESIGN):
        raise ValueError('Bound design differs from the frozen repository comparison design')
    design = _read(binding['path'])
    if design.get('status') != 'design_only_not_executable':
        raise ValueError('Invalid prospective comparison design')
    for key in ('role', 'comparisons', 'bootstrap', 'metrics', 'ensemble_sizes', 'generation_seeds'):
        if plan.get(key) != design.get(key):
            raise ValueError(f'Comparison plan changed prospective design: {key}')
    if (plan['generation_seeds'] != EXPECTED_SEEDS or plan['ensemble_sizes'] != [512, 1024]
            or plan['metrics'] != ['energy_score', 'variogram_score']
            or plan['bootstrap'] != {'seed': 20261131, 'replicates': 1999, 'level': .95}):
        raise ValueError('Comparison settings differ from the frozen replication specification')
    groups = plan.get('groups')
    if not isinstance(groups, list) or len(groups) != len(design['groups']):
        raise ValueError('Comparison groups differ from prospective design')
    refs = []
    for group, original in zip(groups, design['groups']):
        if (set(group) != {'id', 'role', 'members'} or not _safe(group['id'])
                or group['id'] != original['id'] or group['role'] != original['role']
                or not isinstance(group['members'], list)
                or any(set(member) != {'study_id', 'model_id'} for member in group['members'])
                or [member['model_id'] for member in group['members']] != original['model_ids']):
            raise ValueError('Model groups changed or omitted a prospectively declared model')
        for member in group['members']:
            if not _safe(member['study_id']) or not _safe(member['model_id']):
                raise ValueError('Invalid model/study ID')
            refs.append((member['study_id'], member['model_id']))
    if len(refs) != len(set(refs)):
        raise ValueError('A model is assigned to multiple groups')
    studies = plan.get('studies')
    if (not isinstance(studies, list) or not studies or any(set(study) != {'id', 'directory', 'manifest_sha256'}
            or not _safe(study['id']) for study in studies)
            or len({study['id'] for study in studies}) != len(studies)):
        raise ValueError('Declare distinct completed study bindings')
    if set(member[0] for member in refs) != {study['id'] for study in studies}:
        raise ValueError('Model membership and study bindings disagree')
    return plan


def _array_file(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def _macro(values, strata):
    return np.mean([values[strata == group].mean(axis=0) for group in range(8)], axis=0)


def _same(actual, expected, message):
    if not np.array_equal(actual, expected):
        raise ValueError(message)


def _load_studies(plan):
    models, common, reference_rows = {}, None, None
    columns = [f'{score}_{count}' for count in plan['ensemble_sizes'] for score in plan['metrics']]
    for binding in plan['studies']:
        directory = Path(binding['directory']).resolve()
        manifest = _completed(directory, binding['manifest_sha256'])
        if (manifest.get('phase') != 1 or manifest.get('schema_version') != 1
                or manifest.get('test_performance_access') is not False or manifest.get('intervals_computed') is not False
                or manifest.get('generation_seeds') != plan['generation_seeds']
                or [manifest.get('nested_size'), manifest.get('ensemble_size')] != plan['ensemble_sizes']
                or manifest.get('failed_models') or manifest.get('completed_models') != len(manifest.get('models', []))):
            raise ValueError('Proper study completion or evaluation settings disagree with comparison plan')
        _bound(directory / 'plan.json', manifest['plan_sha256'])
        source_plan = _read(directory / 'plan.json')
        model_ids = [entry['id'] for entry in manifest.get('models', [])]
        if not model_ids or len(model_ids) != len(set(model_ids)):
            raise ValueError('Proper study model IDs must be nonempty and unique')
        for key in ('models', 'data_hashes', 'generation_seeds', 'ensemble_size', 'nested_size'):
            if source_plan.get(key) != manifest.get(key):
                raise ValueError(f'Proper study copied plan disagrees at {key}')
        for name in ('proper_scores', 'meter_uncertainty'):
            source = f'src/fluvius_cgan/{name}.py'
            if manifest.get('source_sha256', {}).get(source) != _sha256(ROOT / source):
                raise ValueError('Scoring/aggregation source revision differs from the verified comparison core')
        _bound(directory / 'data_rows.npz', manifest['data_rows_sha256'])
        rows = _array_file(directory / 'data_rows.npz')
        if set(rows) != {'row_index', 'y', 'meter_id', 'date', 'profile_row_sha256'}:
            raise ValueError('Unexpected validation row artifact schema')
        count = len(rows['y'])
        if (rows['profile_row_sha256'].shape != (count, 32) or rows['profile_row_sha256'].dtype != np.uint8
                or not np.array_equal(rows['row_index'], np.arange(count))):
            raise ValueError('Invalid validation row indices/profile hashes')
        if evaluation._calendar_audit(rows, 'VAL') != manifest['validation_calendar']:
            raise ValueError('Validation calendar audit differs from saved proper study')
        identity = {key: manifest[key] for key in ('data_hashes', 'validation_profile_bytes_sha256',
                    'validation_label_bytes_sha256', 'validation_calendar')}
        if common is None:
            common, reference_rows = identity, rows
        else:
            if identity != common:
                raise ValueError('Proper studies use different validation datasets')
            for key in rows:
                _same(rows[key], reference_rows[key], 'Proper study row ordering/identity differs')
        study_summary = _read(directory / 'summary.json')
        if set(study_summary.get('models', {})) != {entry['id'] for entry in manifest['models']}:
            raise ValueError('Proper study summary omitted or added a model')
        for entry in manifest['models']:
            model_id = entry['id']
            if not _safe(model_id):
                raise ValueError('Unsafe model ID in proper study')
            target = directory / model_id
            receipt = _completed(target)
            if (receipt.get('model') != entry or receipt.get('failed_draws')
                    or receipt.get('data_hashes') != manifest['data_hashes']
                    or receipt.get('data_rows_sha256') != manifest['data_rows_sha256']
                    or receipt.get('generation_seeds') != plan['generation_seeds']):
                raise ValueError('Model receipt disagrees with completed proper study')
            _bound(target / 'model_metadata.json', receipt['model_metadata_sha256'])
            metadata = _read(target / 'model_metadata.json')
            if metadata.get('model_id') != model_id or metadata.get('binding') != entry:
                raise ValueError('Saved model metadata disagrees with model binding')
            draws = receipt.get('draws', [])
            summary_model = study_summary['models'][model_id]
            if ([draw.get('generation_seed') for draw in draws] != plan['generation_seeds']
                    or summary_model.get('draws') != draws or summary_model.get('status') != 'completed'
                    or summary_model.get('completed_draws') != 3 or summary_model.get('failed_draws')):
                raise ValueError('Every model requires all three completed declared generation draws')
            profiles, meters = [], []
            for seed, draw in zip(plan['generation_seeds'], draws):
                if _read(target / f'draw_{seed}.json') != draw:
                    raise ValueError('Draw receipt differs from model manifest')
                if (draw.get('status') != 'completed' or draw.get('model_id') != model_id
                        or draw.get('data_rows_sha256') != manifest['data_rows_sha256']
                        or draw.get('model_metadata_sha256') != receipt['model_metadata_sha256']
                        or draw.get('score_columns') != columns
                        or [draw.get('nested_size'), draw.get('ensemble_size')] != plan['ensemble_sizes']):
                    raise ValueError('Draw metadata/score specification mismatch')
                for kind in ('score', 'generated'):
                    name = draw[f'{kind}_artifact']
                    if Path(name).name != name:
                        raise ValueError('Invalid draw artifact path')
                    _bound(target / name, draw[f'{kind}_sha256'])
                arrays = _array_file(target / draw['score_artifact'])
                _same(arrays['row_index'], rows['row_index'], 'Score rows differ')
                _same(arrays['y'], rows['y'], 'Score conditions differ')
                _same(arrays['score_columns'], columns, 'Score column order differs')
                if arrays['data_rows_sha256'].shape != () or arrays['data_rows_sha256'].item() != manifest['data_rows_sha256']:
                    raise ValueError('Score artifact is bound to different validation rows')
                matrix = np.column_stack([arrays[name] for name in columns])
                if matrix.shape != (count, len(columns)) or not np.isfinite(matrix).all():
                    raise ValueError('Invalid finite per-profile score matrix')
                contribution = meter_contributions(matrix, rows['meter_id'], rows['y'])
                for saved, field in (('meter_ids', 'meter_ids'), ('meter_strata', 'strata'), ('meter_values', 'values'),
                                      ('meter_bin_counts', 'bin_counts'), ('meter_bin_sums', 'bin_sums')):
                    _same(arrays[saved], contribution[field], 'Stored meter aggregation differs from per-profile scores')
                stratum_means = {str(group): contribution['values'][contribution['strata'] == group].mean(axis=0).tolist()
                                 for group in range(8)}
                if (draw.get('per_stratum_meter_means') != stratum_means
                        or draw.get('macro_meter_balanced_scores') != dict(zip(columns, np.mean(list(stratum_means.values()), axis=0).tolist()))
                        or draw.get('profile_rows') != count or draw.get('meters') != len(contribution['meter_ids'])):
                    raise ValueError('Draw summary disagrees with reconstructed scores')
                if set(draw.get('per_condition', {})) != {str(label) for label in range(64)}:
                    raise ValueError('Draw lacks all condition summaries')
                for label in range(64):
                    if draw['per_condition'][str(label)]['real_profiles'] != int(np.sum(rows['y'] == label)):
                        raise ValueError('Draw condition counts differ from validation rows')
                profiles.append(matrix)
                meters.append(contribution['values'])
            key = f"{binding['id']}:{model_id}"
            models[key] = {'metadata': metadata, 'profiles': profiles, 'meters': meters,
                           'profile_mean': np.mean(profiles, axis=0)}
    declared = {f"{member['study_id']}:{member['model_id']}" for group in plan['groups'] for member in group['members']}
    if set(models) != declared:
        raise ValueError('Every completed proper-study model must appear exactly once in comparison groups')
    return models, reference_rows, common, columns


def _members(group):
    return [f"{member['study_id']}:{member['model_id']}" for member in group['members']]


def _groups(plan, models):
    roles = {group['role']: group for group in plan['groups'] if group['role'] not in ('control', 'contextual_flow')}
    for role in ('replicated_mixture', 'gaussian', 'calibrated_cgan', 'raw_cgan'):
        if role not in roles:
            raise ValueError('Missing primary replication group')
    mixtures = [models[key]['metadata'] for key in _members(roles['replicated_mixture'])]
    if sorted(item.get('fitting_seed') for item in mixtures) != list(range(20261101, 20261106)):
        raise ValueError('Mixture replication seed set differs')
    for item in mixtures:
        config = item.get('fit_metadata', {}).get('config', {})
        if item.get('kind') != 'mixture' or config.get('fit_seed') != item['fitting_seed'] or any(config.get(k) != v for k, v in
                {'n_components': 8, 'n_init': 2, 'max_iter': 300, 'tol': .001, 'reg_covar': .001}.items()):
            raise ValueError('Mixture recipe differs from frozen replication')
    gaussian = [models[key]['metadata'] for key in _members(roles['gaussian'])]
    if len(gaussian) != 1 or gaussian[0].get('method') != 'copula' or gaussian[0].get('kind') != 'baseline':
        raise ValueError('Gaussian comparator is not the declared baseline')
    paired = {}
    for role, method in [('raw_cgan', 'conditional_cgan'), ('calibrated_cgan', 'conditional_cgan_marginal_calibration')]:
        items = {models[key]['metadata'].get('training_seed'): key for key in _members(roles[role])}
        if sorted(items) != CGAN_SEEDS or len(_members(roles[role])) != 5:
            raise ValueError('cGAN replication must retain exactly the five declared training seeds')
        if any(models[key]['metadata'].get('method') != method or models[key]['metadata'].get('kind') != 'neural'
               for key in items.values()):
            raise ValueError('Raw/calibrated cGAN group method mismatch')
        paired[role] = items
    for seed in CGAN_SEEDS:
        raw, calibrated = (models[paired[role][seed]]['metadata'] for role in ('raw_cgan', 'calibrated_cgan'))
        for name in ('checkpoint_sha256', 'manifest_sha256'):
            value = raw['candidate'].get(name)
            if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value) or value != calibrated['candidate'].get(name):
                raise ValueError('Calibrated/raw cGAN pairing requires identical saved checkpoints')
        if calibrated.get('calibration', {}).get('fit_record', {}).get('calibration_seed') != 20261121:
            raise ValueError('cGAN calibration seed differs from frozen replication')
    for group in plan['groups']:
        if group['role'] == 'control':
            members = _members(group)
            if len(members) != 1:
                raise ValueError('Each classical control is one declared fitted score function')
            item = models[members[0]]['metadata']
            identity = group['id']
            if identity in ('mean', 'resample'):
                valid = item.get('kind') == 'baseline' and item.get('method') == identity
            elif identity == 'beta':
                valid = (item.get('kind') == 'strong' and item.get('method') == 'conditional_empirical_beta_copula'
                         and item.get('fit_metadata', {}).get('df') is None
                         and item.get('fit_metadata', {}).get('fit_seed') == 20261026)
            else:
                valid = (identity in ('student_t4', 'student_t8', 'student_t16')
                         and item.get('kind') == 'strong' and item.get('method') == 'conditional_student_t_copula'
                         and item.get('fit_metadata', {}).get('df') == int(identity.removeprefix('student_t')))
            if not valid:
                raise ValueError('Classical control method or fit recipe differs from its prospective identity')
        if group['role'] == 'contextual_flow':
            metadata = [models[key]['metadata'] for key in _members(group)]
            if len(metadata) != 2 or sorted(item.get('training_seed') for item in metadata) != [101, 202]:
                raise ValueError('Contextual flow group must retain both existing training seeds')
            calibrated = group['id'] == 'flow_cal'
            method = 'conditional_flow_matching' + ('_marginal_calibration' if calibrated else '')
            if any(item.get('method') != method or ('calibration' in item) != calibrated for item in metadata):
                raise ValueError('Contextual flow group contains a different method')
    flow_groups = {group['id']: group for group in plan['groups'] if group['role'] == 'contextual_flow'}
    raw_flow = {models[key]['metadata']['training_seed']: models[key]['metadata'] for key in _members(flow_groups['flow_raw'])}
    for key in _members(flow_groups['flow_cal']):
        item = models[key]['metadata']
        for name in ('checkpoint_sha256', 'manifest_sha256'):
            value = raw_flow[item['training_seed']].get('candidate', {}).get(name)
            if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value) or value != item.get('candidate', {}).get(name):
                raise ValueError('Contextual calibrated/raw flow requires matching checkpoints')
    return paired


def _csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _npz(path, **arrays):
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)


def compare(plan_path, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan_snapshot = plan_path.read_bytes()
    plan_hash = hashlib.sha256(plan_snapshot).hexdigest()
    plan = _plan(plan_path)
    models, rows, common, names = _load_studies(plan)
    paired = _groups(plan, models)
    _bound(plan_path, plan_hash)
    base = meter_contributions(np.zeros(len(rows['y'])), rows['meter_id'], rows['y'])
    strata = base['strata']
    manifest = {'schema_version': 1, 'phase': 1, 'status': 'running', 'role': plan['role'], 'scope': SCOPE,
        'plan_sha256': plan_hash, 'design': plan['design'], 'studies': plan['studies'],
        'data_identity': common, 'test_performance_access': False, 'started_at_utc': _now(),
        'source_sha256': {str(path.relative_to(ROOT)): _sha256(path) for path in
                         [Path(__file__).resolve(), ROOT / 'src/fluvius_cgan/meter_uncertainty.py']},
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                                   capture_output=True, text=True).stdout.strip()}
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / 'plan.json').write_bytes(plan_snapshot)
        _bound(output / 'plan.json', plan_hash)
        _json(output / 'manifest.json', manifest)
        summary = {'schema_version': 1, 'scope': SCOPE, 'models': {}, 'groups': {},
                   'comparisons': {}, 'paired_cgan_fits': [], 'score_columns': []}
        matrix_parts, descriptors, offsets = [], [], {}
        def register(level, identity, meter_values, **extra):
            offset = len(descriptors)
            offsets[level, identity] = offset
            matrix_parts.append(meter_values)
            for column, name in enumerate(names):
                metric, size = name.rsplit('_', 1)
                descriptor = {'column_id': f'{level}:{identity}:{name}', 'level': level, 'id': identity,
                              'metric': metric, 'ensemble_size': int(size), **extra}
                descriptors.append(descriptor)
                summary['score_columns'].append(descriptor)
            return _macro(meter_values, strata)
        def aggregate(profiles):
            result = meter_contributions(profiles, rows['meter_id'], rows['y'])
            _same(result['meter_ids'], base['meter_ids'], 'Derived meter ordering changed')
            return result['values']
        def save_profiles(name, values):
            contribution = meter_contributions(values, rows['meter_id'], rows['y'])
            _npz(output / name, row_index=rows['row_index'], y=rows['y'], values=values,
                 score_columns=np.asarray(names), meter_ids=np.asarray(contribution['meter_ids']),
                 meter_strata=contribution['strata'], meter_values=contribution['values'],
                 meter_bin_counts=contribution['bin_counts'], meter_bin_sums=contribution['bin_sums'])
        fit_rows = []
        for group in plan['groups']:
            for key in _members(group):
                record = models[key]
                metadata = record['metadata']
                macro_draws = []
                for seed, meter in zip(plan['generation_seeds'], record['meters']):
                    macro_draws.append(register('model_draw', f'{key}:{seed}', meter, group_id=group['id'],
                        training_seed=metadata.get('training_seed'), fitting_seed=metadata.get('fitting_seed'), generation_seed=seed))
                meter_mean = aggregate(record['profile_mean'])
                mean = register('model_average', key, meter_mean, group_id=group['id'],
                    training_seed=metadata.get('training_seed'), fitting_seed=metadata.get('fitting_seed'))
                deviation = np.std(macro_draws, axis=0, ddof=1)
                summary['models'][key] = {'group_id': group['id'], 'role': group['role'],
                    'training_seed': metadata.get('training_seed'), 'fitting_seed': metadata.get('fitting_seed'),
                    'generation_mean': dict(zip(names, mean.tolist())), 'generation_sd': dict(zip(names, deviation.tolist())),
                    'draws': {str(seed): dict(zip(names, values.tolist())) for seed, values in zip(plan['generation_seeds'], macro_draws)}}
                for index, name in enumerate(names):
                    fit_rows.append({'unit': 'model', 'model': key, 'group': group['id'], 'role': group['role'], 'score': name,
                        'training_seed': metadata.get('training_seed'), 'fitting_seed': metadata.get('fitting_seed'),
                        'generation_mean': mean[index], 'within_fit_generation_sd': deviation[index],
                        'generation_replicates': 3, 'scope': SCOPE})
        group_profiles = {}
        for group in plan['groups']:
            keys = _members(group)
            profiles = np.mean([models[key]['profile_mean'] for key in keys], axis=0)
            group_profiles[group['id']] = profiles
            mean = register('group_average', group['id'], aggregate(profiles), group_id=group['id'], fit_count=len(keys), role=group['role'])
            fit_means = np.array([[summary['models'][key]['generation_mean'][name] for name in names] for key in keys])
            deviation = np.std(fit_means, axis=0, ddof=1).tolist() if len(keys) > 1 else [None] * len(names)
            summary['groups'][group['id']] = {'role': group['role'], 'fit_count': len(keys), 'models': keys,
                'mean': dict(zip(names, mean.tolist())), 'fit_seed_sd': dict(zip(names, deviation)),
                'within_fit_generation_sd': {key: summary['models'][key]['generation_sd'] for key in keys},
                'interpretation': 'SD across fixed generation-averaged fits; one-fit SD is undefined. Contextual flow has two fits, not five.'}
        for seed in CGAN_SEEDS:
            raw_key, cal_key = paired['raw_cgan'][seed], paired['calibrated_cgan'][seed]
            raw, cal = models[raw_key], models[cal_key]
            draws = []
            for generation, raw_profiles, cal_profiles in zip(plan['generation_seeds'], raw['profiles'], cal['profiles']):
                draws.append(register('paired_cgan_draw', f'{seed}:{generation}', aggregate(cal_profiles - raw_profiles),
                                      training_seed=seed, generation_seed=generation))
            profiles = cal['profile_mean'] - raw['profile_mean']
            mean = register('paired_cgan_average', str(seed), aggregate(profiles), training_seed=seed)
            save_profiles(f'paired_cgan_s{seed}_profiles.npz', profiles)
            summary['paired_cgan_fits'].append({'training_seed': seed, 'left': cal_key, 'right': raw_key,
                'mean': dict(zip(names, mean.tolist())), 'generation_sd': dict(zip(names, np.std(draws, axis=0, ddof=1).tolist()))})
            for index, name in enumerate(names):
                fit_rows.append({'unit': 'paired_cgan_difference', 'model': f'{cal_key} minus {raw_key}',
                    'group': 'cgan_cal_vs_cgan_raw', 'role': 'paired_training_seed', 'score': name,
                    'training_seed': seed, 'fitting_seed': None, 'generation_mean': mean[index],
                    'within_fit_generation_sd': summary['paired_cgan_fits'][-1]['generation_sd'][name],
                    'generation_replicates': 3, 'scope': SCOPE})
        for comparison in plan['comparisons']:
            profiles = group_profiles[comparison['left']] - group_profiles[comparison['right']]
            mean = register('comparison', comparison['id'], aggregate(profiles), role=comparison['role'], priority=comparison['priority'])
            save_profiles(f"comparison_{comparison['id']}_profiles.npz", profiles)
            summary['comparisons'][comparison['id']] = {**comparison, 'mean': dict(zip(names, mean.tolist())),
                'left_fit_seed_sd': summary['groups'][comparison['left']]['fit_seed_sd'],
                'right_fit_seed_sd': summary['groups'][comparison['right']]['fit_seed_sd'],
                'paired_fit_seed_sd': dict(zip(names, np.std([[pair['mean'][name] for name in names]
                    for pair in summary['paired_cgan_fits']], axis=0, ddof=1).tolist())) if comparison['priority'] == 3 else None}
        meter_matrix = np.column_stack(matrix_parts)
        _npz(output / 'meter_contributions.npz', meter_ids=np.asarray(base['meter_ids']), strata=strata,
             values=meter_matrix, column_ids=np.asarray([record['column_id'] for record in descriptors]),
             bin_counts=base['bin_counts'])
        _json(output / 'summary.json', summary)
        bootstrap = stratified_studentized_bootstrap(meter_matrix, strata, **plan['bootstrap'])
        _json(output / 'bootstrap.json', bootstrap)
        all_rows = []
        for index, record in enumerate(descriptors):
            interval = bootstrap['interval'][index]
            all_rows.append({**record, 'estimate': bootstrap['estimate'][index], 'standard_error': bootstrap['standard_error'][index],
                'interval_level': bootstrap['level'], 'interval_scope': bootstrap['interval_scope'],
                'interval_lower': interval[0] if interval is not None else None,
                'interval_upper': interval[1] if interval is not None else None,
                'status': bootstrap['status'][index], 'zero_bootstrap_se_count': bootstrap['bootstrap_zero_standard_error_counts'][index],
                'multiplicity_sha256': bootstrap['multiplicity_sha256'], 'scope': SCOPE})
        _csv(output / 'scores.csv', all_rows)
        _csv(output / 'groups.csv', [row | {'fit_seed_sd': summary['groups'][row['id']]['fit_seed_sd'][f"{row['metric']}_{row['ensemble_size']}"]}
                                    for row in all_rows if row['level'] == 'group_average'])
        _csv(output / 'comparisons.csv', [row | {'left': summary['comparisons'][row['id']]['left'],
                                               'right': summary['comparisons'][row['id']]['right'],
            'left_fit_seed_sd': summary['comparisons'][row['id']]['left_fit_seed_sd'][f"{row['metric']}_{row['ensemble_size']}"],
            'right_fit_seed_sd': summary['comparisons'][row['id']]['right_fit_seed_sd'][f"{row['metric']}_{row['ensemble_size']}"],
            'paired_fit_seed_sd': (summary['comparisons'][row['id']]['paired_fit_seed_sd'] or {}).get(f"{row['metric']}_{row['ensemble_size']}")}
                                         for row in all_rows if row['level'] == 'comparison'])
        _csv(output / 'fit_variation.csv', fit_rows)
        manifest.update(status='completed', finished_at_utc=_now(), score_columns=len(descriptors),
                        multiplicity_sha256=bootstrap['multiplicity_sha256'], artifact_hashes=_inventory(output))
        _json(output / 'manifest.json', manifest)
        return manifest
    except Exception as error:
        manifest.update(status='failed', finished_at_utc=_now(), error=f'{type(error).__name__}: {error}',
                        **_partial_inventory(output))
        try:
            _json(output / 'manifest.json', manifest)
        except Exception as receipt_error:
            error.add_note(f'Could not write comparison failure receipt: {receipt_error}')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.plan, args.output)


if __name__ == '__main__':
    main()
