"""Bound comparisons preserve every declared fit, draw and bootstrap failure."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from fluvius_cgan.meter_uncertainty import meter_contributions

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / 'reports/phase1-development/PROPER_SCORE_COMPARISON_DESIGN.json'


def module(name):
    path = ROOT / 'scripts' / f'{name}.py'
    assert path.exists(), f'{name} is not implemented'
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    previous = sys.path.copy()
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(result)
    finally:
        sys.path[:] = previous
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def read(path):
    return json.loads(path.read_text())


def inventory(directory):
    return {str(path.relative_to(directory)): digest(path) for path in directory.rglob('*')
            if path.is_file() and path != directory / 'manifest.json'}


@pytest.fixture
def fixture(tmp_path):
    directory = tmp_path / 'proper'
    directory.mkdir()
    design = read(DESIGN)
    dates = np.arange('2024-01-01', '2025-01-01', dtype='datetime64[D]')
    dates = dates[~np.isin(dates, np.array(['2024-03-31', '2024-10-27'], dtype='datetime64[D]'))]
    weekend = ((dates.astype(np.int64) + 3) % 7 >= 5).astype(np.int64)
    months = dates.astype('datetime64[M]').astype(np.int64) % 12 + 1
    calendar = weekend * 4 + (months % 12) // 3
    ids = np.repeat([f'meter_{g}_{m}' for g in range(8) for m in range(3)], 364)
    labels = np.concatenate([g * 8 + calendar for g in range(8) for _ in range(3)])
    rows = dict(row_index=np.arange(len(ids)), y=labels, meter_id=ids,
                date=np.tile(dates.astype('U10'), 24), profile_row_sha256=np.zeros((len(ids), 32), dtype=np.uint8))
    np.savez_compressed(directory / 'data_rows.npz', **rows)
    row_hash = digest(directory / 'data_rows.npz')
    entries, metadata = [], {}
    for group in design['groups']:
        for model_id in group['model_ids']:
            role = group['role']
            seed = int(model_id.rsplit('_s', 1)[1]) if '_s' in model_id else None
            kind = 'mixture' if role == 'replicated_mixture' else 'neural' if seed is not None else 'baseline'
            if model_id == 'beta' or model_id.startswith('student_t'):
                kind = 'strong'
            method = {'replicated_mixture': 'conditional_gaussian_mixture_copula', 'gaussian': 'copula',
                      'raw_cgan': 'conditional_cgan', 'calibrated_cgan': 'conditional_cgan_marginal_calibration',
                      'contextual_flow': 'conditional_flow_matching' + ('_marginal_calibration' if '_cal_' in model_id else '')}.get(role, model_id)
            if model_id == 'beta':
                method = 'conditional_empirical_beta_copula'
            elif model_id.startswith('student_t'):
                method = 'conditional_student_t_copula'
            entry = {'id': model_id, 'kind': kind}
            record = {'model_id': model_id, 'kind': kind, 'method': method,
                      'binding': entry, 'training_seed': seed if kind == 'neural' else None}
            if role == 'replicated_mixture':
                record.update(fitting_seed=seed, fit_metadata={'config': {'n_components': 8,
                    'fit_seed': seed, 'n_init': 2, 'max_iter': 300, 'tol': .001, 'reg_covar': .001}})
            if kind == 'strong':
                record['fit_metadata'] = {'df': int(model_id.removeprefix('student_t')) if model_id.startswith('student_t') else None,
                                          'fit_seed': 20261026 if model_id == 'beta' else None}
            if role in ('raw_cgan', 'calibrated_cgan', 'contextual_flow'):
                record['candidate'] = {'checkpoint_sha256': f'{seed:064x}', 'manifest_sha256': f'{seed + 1:064x}'}
                if role == 'calibrated_cgan' or '_cal_' in model_id:
                    record['calibration'] = {'fit_record': {'calibration_seed': 20261121 if role == 'calibrated_cgan' else 20261025}}
            entries.append(entry)
            metadata[model_id] = record
    proper_plan = {'schema_version': 1, 'phase': 1, 'generation_seeds': design['generation_seeds'],
                   'ensemble_size': 1024, 'nested_size': 512, 'models': entries,
                   'data_hashes': {key: str(index) * 64 for index, key in enumerate(
                       ['train.npz', 'val.npz', 'scaler.json', 'metadata.json'])}}
    write(directory / 'plan.json', proper_plan)
    provenance = {'schema_version': 1, 'phase': 1, 'data_hashes': proper_plan['data_hashes'],
                  'plan_sha256': digest(directory / 'plan.json'), 'data_rows_sha256': row_hash,
                  'generation_seeds': design['generation_seeds'], 'ensemble_size': 1024, 'nested_size': 512,
                  'test_performance_access': False, 'intervals_computed': False,
                  'validation_calendar': module('evaluate_proper_scores')._calendar_audit(rows, 'VAL'),
                  'validation_profile_bytes_sha256': 'a' * 64,
                  'validation_label_bytes_sha256': hashlib.sha256(labels.astype('<i8').tobytes()).hexdigest(),
                  'source_sha256': {f'src/fluvius_cgan/{name}.py': digest(ROOT / 'src/fluvius_cgan' / f'{name}.py')
                                    for name in ('proper_scores', 'meter_uncertainty')}}
    columns = ['energy_score_512', 'variogram_score_512', 'energy_score_1024', 'variogram_score_1024']
    summary = {'schema_version': 1, 'models': {}}
    for index, entry in enumerate(entries):
        target = directory / entry['id']
        target.mkdir()
        write(target / 'model_metadata.json', metadata[entry['id']])
        draws = []
        for replicate, seed in enumerate(design['generation_seeds']):
            energy = .2 * index + .05 * replicate + (1 + .02 * index) * np.repeat(np.arange(24) / 10., 364)
            vectors = np.column_stack([energy + .01, np.zeros(len(ids)), energy, np.zeros(len(ids))])
            contribution = meter_contributions(vectors, ids, labels)
            score_path = target / f'profile_scores_{seed}.npz'
            np.savez_compressed(score_path, **dict(zip(columns, vectors.T)), row_index=rows['row_index'], y=labels,
                data_rows_sha256=np.asarray(row_hash), score_columns=np.asarray(columns),
                meter_ids=np.asarray(contribution['meter_ids']), meter_strata=contribution['strata'],
                meter_values=contribution['values'], meter_bin_counts=contribution['bin_counts'],
                meter_bin_sums=contribution['bin_sums'])
            generated = target / f'generated_{seed}.npz'
            np.savez_compressed(generated, synthetic_fixture=np.asarray(True))
            strata_means = {str(g): contribution['values'][contribution['strata'] == g].mean(axis=0).tolist() for g in range(8)}
            draw = {'status': 'completed', 'model_id': entry['id'], 'generation_seed': seed,
                    'ensemble_size': 1024, 'nested_size': 512, 'data_rows_sha256': row_hash,
                    'model_metadata_sha256': digest(target / 'model_metadata.json'),
                    'score_artifact': score_path.name, 'score_sha256': digest(score_path),
                    'generated_artifact': generated.name, 'generated_sha256': digest(generated),
                    'score_columns': columns, 'profile_rows': len(ids), 'meters': 24,
                    'per_stratum_meter_means': strata_means,
                    'macro_meter_balanced_scores': dict(zip(columns, np.mean(list(strata_means.values()), axis=0))),
                    'per_condition': {str(c): {'real_profiles': int(np.sum(labels == c))} for c in range(64)}}
            write(target / f'draw_{seed}.json', draw)
            draws.append(draw)
        receipt = dict(provenance, status='completed', model=entry, draws=draws, failed_draws=[],
                       model_metadata_sha256=digest(target / 'model_metadata.json'), artifact_hashes=inventory(target))
        write(target / 'manifest.json', receipt)
        summary['models'][entry['id']] = {'status': 'completed', 'model': entry,
                                          'completed_draws': 3, 'failed_draws': [], 'draws': draws}
    write(directory / 'summary.json', summary)
    write(directory / 'manifest.json', dict(provenance, status='completed', models=entries,
          completed_models=len(entries), failed_models=[], artifact_hashes=inventory(directory)))
    plan = {key: design[key] for key in ('schema_version', 'phase', 'role', 'comparisons', 'bootstrap',
                                        'metrics', 'ensemble_sizes', 'generation_seeds')}
    plan.update(design={'path': str(DESIGN), 'sha256': digest(DESIGN)},
                studies=[{'id': 'replication', 'directory': str(directory), 'manifest_sha256': digest(directory / 'manifest.json')}],
                groups=[{'id': g['id'], 'role': g['role'], 'members': [{'study_id': 'replication', 'model_id': m}
                        for m in g['model_ids']]} for g in design['groups']])
    path = tmp_path / 'comparison_plan.json'
    write(path, plan)
    return directory, path


def test_every_fit_draw_control_and_comparison_shares_one_bootstrap(fixture, tmp_path):
    _, plan = fixture
    output = tmp_path / 'comparison'
    result = module('compare_proper_scores').compare(plan, output)
    assert result['status'] == 'completed' and result['artifact_hashes'] == inventory(output)
    summary = read(output / 'summary.json')
    assert len(summary['models']) == 26
    assert summary['groups']['gaussian']['fit_seed_sd']['energy_score_1024'] is None
    assert summary['groups']['flow_raw']['fit_count'] == 2
    assert summary['groups']['mixture_k8']['fit_count'] == 5
    assert len(summary['paired_cgan_fits']) == 5
    assert set(summary['comparisons']) == {entry['id'] for entry in read(plan)['comparisons']}
    assert summary['groups']['mixture_k8']['fit_seed_sd']['energy_score_1024'] == pytest.approx(.223 * np.sqrt(2.5))
    first = summary['models']['replication:mixture_k8_s20261101']
    assert first['generation_sd']['energy_score_1024'] == pytest.approx(.05)
    assert first['generation_mean']['energy_score_1024'] == pytest.approx(1.2)
    assert summary['comparisons']['cgan_cal_vs_cgan_raw']['mean']['energy_score_1024'] == pytest.approx(-1.115)
    bootstrap = read(output / 'bootstrap.json')
    assert bootstrap['replicates'] == 1999 and bootstrap['seed'] == 20261131
    assert bootstrap['columns'] == len(summary['score_columns'])
    assert 'degenerate_observed_variance' in bootstrap['status'] and 'ok' in bootstrap['status']
    assert result['multiplicity_sha256'] == bootstrap['multiplicity_sha256']
    for name in ('scores.csv', 'groups.csv', 'comparisons.csv', 'fit_variation.csv'):
        assert 'postselection' in (output / name).read_text()
    with np.load(output / 'comparison_cgan_cal_vs_cgan_raw_profiles.npz', allow_pickle=False) as saved:
        assert saved['values'].shape == (24 * 364, 4)
    before = (output / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        module('compare_proper_scores').compare(plan, output)
    assert (output / 'manifest.json').read_bytes() == before


def test_changed_artifacts_or_omitted_models_fail_before_output(fixture, tmp_path):
    directory, path = fixture
    runner = module('compare_proper_scores')
    original = path.read_bytes()
    plan = read(path)
    plan['groups'][0]['members'].pop()
    write(path, plan)
    with pytest.raises(ValueError):
        runner.compare(path, tmp_path / 'omitted')
    assert not (tmp_path / 'omitted').exists()
    path.write_bytes(original)
    with (directory / 'gaussian/profile_scores_20261111.npz').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        runner.compare(path, tmp_path / 'tampered')
    assert not (tmp_path / 'tampered').exists()


def test_runtime_failure_retains_failed_receipt_and_declared_plan(fixture, tmp_path, monkeypatch):
    _, plan = fixture
    runner = module('compare_proper_scores')
    def failure(*args, **kwargs):
        raise RuntimeError('synthetic bootstrap execution failure')
    monkeypatch.setattr(runner, 'stratified_studentized_bootstrap', failure)
    output = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='synthetic bootstrap'):
        runner.compare(plan, output)
    receipt = read(output / 'manifest.json')
    assert receipt['status'] == 'failed' and receipt['artifact_hashes'] == inventory(output)
    assert (output / 'plan.json').exists()


def rebind(directory, plan_path, model_id):
    target = directory / model_id
    receipt = read(target / 'manifest.json')
    metadata_hash = digest(target / 'model_metadata.json')
    receipt['model_metadata_sha256'] = metadata_hash
    for draw in receipt['draws']:
        draw['model_metadata_sha256'] = metadata_hash
        draw['score_sha256'] = digest(target / draw['score_artifact'])
        write(target / f"draw_{draw['generation_seed']}.json", draw)
    receipt['artifact_hashes'] = inventory(target)
    write(target / 'manifest.json', receipt)
    summary = read(directory / 'summary.json')
    summary['models'][model_id]['draws'] = receipt['draws']
    write(directory / 'summary.json', summary)
    root = read(directory / 'manifest.json')
    root['artifact_hashes'] = inventory(directory)
    write(directory / 'manifest.json', root)
    plan = read(plan_path)
    plan['studies'][0]['manifest_sha256'] = digest(directory / 'manifest.json')
    write(plan_path, plan)


@pytest.mark.parametrize('change', ['student_df', 'flow_kind', 'paired_checkpoint', 'meter_values'])
def test_rebound_semantic_corruption_still_fails_closed(fixture, tmp_path, change):
    directory, plan = fixture
    name = {'student_df': 'student_t4', 'flow_kind': 'flow_raw_s101',
            'paired_checkpoint': 'cgan_cal_s301', 'meter_values': 'gaussian'}[change]
    metadata_path = directory / name / 'model_metadata.json'
    metadata = read(metadata_path)
    if change == 'student_df':
        metadata['fit_metadata']['df'] = 8
    elif change == 'flow_kind':
        metadata['method'] += '_marginal_calibration'
    elif change == 'paired_checkpoint':
        metadata['candidate']['checkpoint_sha256'] = 'f' * 64
    else:
        path = directory / name / 'profile_scores_20261111.npz'
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        arrays['meter_values'][0, 0] += 1
        np.savez_compressed(path, **arrays)
    write(metadata_path, metadata)
    rebind(directory, plan, name)
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        module('compare_proper_scores').compare(plan, output)
    assert not output.exists()


def test_rewriting_design_cannot_reverse_a_prespecified_comparison(fixture, tmp_path):
    _, plan_path = fixture
    design = read(DESIGN)
    first = design['comparisons'][0]
    first['left'], first['right'] = first['right'], first['left']
    altered = tmp_path / 'altered_design.json'
    write(altered, design)
    plan = read(plan_path)
    plan['comparisons'] = design['comparisons']
    plan['design'] = {'path': str(altered), 'sha256': digest(altered)}
    write(plan_path, plan)
    with pytest.raises(ValueError, match='design'):
        module('compare_proper_scores').compare(plan_path, tmp_path / 'altered')


def test_failure_inventory_error_does_not_replace_original_exception(fixture, tmp_path, monkeypatch):
    _, plan = fixture
    runner = module('compare_proper_scores')
    output = tmp_path / 'failed_hash'
    original_hash = runner._sha256
    def broken_hash(path):
        if Path(path).name == 'unhashable.bin':
            raise OSError('synthetic hash failure')
        return original_hash(path)
    def fail(*args, **kwargs):
        (output / 'unhashable.bin').write_bytes(b'partial')
        raise RuntimeError('original bootstrap failure')
    monkeypatch.setattr(runner, '_sha256', broken_hash)
    monkeypatch.setattr(runner, 'stratified_studentized_bootstrap', fail)
    with pytest.raises(RuntimeError, match='original bootstrap failure'):
        runner.compare(plan, output)
    receipt = read(output / 'manifest.json')
    assert receipt['status'] == 'failed'
    assert receipt['error'] == 'RuntimeError: original bootstrap failure'
    assert 'unhashable.bin' in receipt['artifact_hash_errors']
