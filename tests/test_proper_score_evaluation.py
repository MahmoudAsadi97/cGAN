"""Full-cohort score evaluation preserves dates, row order and model bindings."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fluvius_cgan.proper_scores import fit_ensemble_scores
from fluvius_cgan.meter_uncertainty import meter_contributions
from fluvius_cgan.evaluation import conditional_metrics, projection_directions

ROOT = Path(__file__).resolve().parents[1]


def runner():
    path = ROOT / 'scripts/evaluate_proper_scores.py'
    assert path.exists(), 'The full-cohort proper-score runner is not implemented'
    spec = importlib.util.spec_from_file_location('proper_score_runner', path)
    result = importlib.util.module_from_spec(spec)
    old = sys.path.copy()
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(result)
    finally:
        sys.path[:] = old
    return result


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(directory):
    return {str(path.relative_to(directory)): digest(path) for path in directory.rglob('*')
            if path.is_file() and path != directory / 'manifest.json'}


def save_plan(path, plan):
    plan['data_hashes'] = {name: digest(Path(plan['data']) / name)
                           for name in ('train.npz', 'val.npz', 'scaler.json', 'metadata.json')}
    path.write_text(json.dumps(plan))


@pytest.fixture
def study(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    dates = np.arange('2024-01-01', '2025-01-01', dtype='datetime64[D]')
    dates = dates[~np.isin(dates, np.array(['2024-03-31', '2024-10-27'], dtype='datetime64[D]'))]
    weekend = ((dates.astype(np.int64) + 3) % 7 >= 5).astype(np.int64)
    month = dates.astype('datetime64[M]').astype(np.int64) % 12 + 1
    calendar = weekend * 4 + (month % 12) // 3
    rng = np.random.default_rng(52)
    for split, repeats in [('train', 1), ('val', 2)]:
        ids, conditions, days = [], [], []
        for stratum in range(8):
            for meter in range(repeats):
                ids.extend([f'{split}_{stratum}_{meter}'] * 364)
                conditions.extend(stratum * 8 + calendar)
                days.extend(dates.astype('U10'))
        values = rng.uniform(-.5, .5, (len(ids), 24)).astype(np.float32)
        if split == 'val':
            values += 3
        permutation = rng.permutation(len(values))
        np.savez(data / f'{split}.npz', x=values[permutation],
                 y=np.asarray(conditions, dtype=np.int64)[permutation],
                 meter_id=np.asarray(ids)[permutation], date=np.asarray(days)[permutation])
    with np.load(data / 'train.npz') as saved:
        scaler = dict(minimum=float(saved['x'].min()), maximum=float(saved['x'].max()), fitted_on='train')
    (data / 'scaler.json').write_text(json.dumps(scaler))
    (data / 'metadata.json').write_text(json.dumps({'resolution': 24, 'condition_dim': 64,
        'development': {'source_partition_role': 'original_train'}}))
    (data / 'test.npz').write_bytes(b'TEST MUST NOT BE READ')
    plan = {'schema_version': 1, 'phase': 1, 'data': str(data), 'generation_seeds': [61, 62],
            'ensemble_size': 4, 'nested_size': 2,
            'models': [{'id': 'mean', 'kind': 'baseline', 'baseline': 'mean'},
                       {'id': 'resample', 'kind': 'baseline', 'baseline': 'resample'}]}
    path = tmp_path / 'plan.json'
    save_plan(path, plan)
    return data, path


def test_full_cohort_scores_keep_input_order_and_nested_ensemble_members(study, tmp_path):
    data, plan = study
    evaluate = runner()
    output = tmp_path / 'scores'
    result = evaluate.evaluate(plan, output, workers=1)
    assert result['status'] == 'completed' and result['completed_models'] == 2
    manifest = read(output / 'manifest.json')
    assert manifest['artifact_hashes'] == inventory(output)
    assert manifest['validation_calendar']['profiles'] == 16 * 364
    assert manifest['validation_calendar']['dates_per_meter'] == 364
    assert manifest['test_performance_access'] is False and manifest['intervals_computed'] is False
    with np.load(data / 'val.npz') as val, np.load(output / 'data_rows.npz') as rows:
        for key in ('meter_id', 'date', 'y'):
            np.testing.assert_array_equal(rows[key], val[key])
        assert rows['profile_row_sha256'].shape == (len(val['x']), 32)
        for name in ('mean', 'resample'):
            receipt = read(output / name / 'manifest.json')
            assert receipt['status'] == 'completed' and len(receipt['draws']) == 2
            assert receipt['artifact_hashes'] == inventory(output / name)
            with np.load(output / name / 'generated_61.npz') as generated:
                np.testing.assert_array_equal(generated['y'], np.repeat(np.arange(64), 4))
                assert np.max(generated['x']) < .5
                with np.load(output / name / 'profile_scores_61.npz') as scores:
                    np.testing.assert_array_equal(scores['row_index'], np.arange(len(val['x'])))
                    for condition in (0, 7, 23, 63):
                        positions = np.flatnonzero(val['y'] == condition)
                        ensemble = generated['x'][generated['y'] == condition]
                        for count in (2, 4):
                            expected = fit_ensemble_scores(ensemble[:count]).score(val['x'][positions])
                            for score in ('energy_score', 'variogram_score'):
                                np.testing.assert_allclose(scores[f'{score}_{count}'][positions], expected[score],
                                                           rtol=1e-13, atol=1e-13)
                    columns = ['energy_score_2', 'variogram_score_2', 'energy_score_4', 'variogram_score_4']
                    expected = meter_contributions(np.column_stack([scores[key] for key in columns]),
                                                   val['meter_id'], val['y'])
                    np.testing.assert_array_equal(scores['meter_ids'], expected['meter_ids'])
                    np.testing.assert_allclose(scores['meter_values'], expected['values'], rtol=0, atol=0)
                    np.testing.assert_array_equal(scores['meter_bin_counts'], expected['bin_counts'])
    before = (output / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        evaluate.evaluate(plan, output, workers=1)
    assert before == (output / 'manifest.json').read_bytes()


@pytest.mark.parametrize('change', ['missing_day', 'wrong_calendar', 'changed_stratum', 'duplicate', 'overlap'])
def test_bad_calendars_and_meter_leakage_fail_before_output(study, tmp_path, change):
    data, path = study
    with np.load(data / 'val.npz') as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if change == 'missing_day':
        arrays = {key: value[1:] for key, value in arrays.items()}
    elif change == 'wrong_calendar':
        arrays['y'][0] = (arrays['y'][0] // 8) * 8 + (arrays['y'][0] + 1) % 8
    elif change == 'changed_stratum':
        arrays['y'][0] = (arrays['y'][0] + 8) % 64
    elif change == 'duplicate':
        for value in arrays.values():
            value[0] = value[1]
    else:
        with np.load(data / 'train.npz') as saved:
            arrays['meter_id'][0] = saved['meter_id'][0]
    np.savez(data / 'val.npz', **arrays)
    save_plan(path, read(path))
    with pytest.raises(ValueError):
        runner().evaluate(path, tmp_path / 'rejected', workers=1)
    assert not (tmp_path / 'rejected').exists()


@pytest.mark.parametrize('mutation', [lambda p: p.update(ensemble_size=1),
    lambda p: p.update(nested_size=4), lambda p: p.update(generation_seeds=[61, 61]),
    lambda p: p.update(generation_seeds=[True]), lambda p: p['models'][0].update(id='../escape'),
    lambda p: p['models'][1].update(id='mean'), lambda p: p.update(models=[]),
    lambda p: p['data_hashes'].update({'val.npz': '0' * 64})])
def test_invalid_plan_fails_before_output(study, tmp_path, mutation):
    _, path = study
    plan = read(path)
    mutation(plan)
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        runner().evaluate(path, tmp_path / 'rejected', workers=1)
    assert not (tmp_path / 'rejected').exists()


def test_strong_baselines_fit_train_only_and_retain_fit_metadata(study, tmp_path):
    _, path = study
    plan = read(path)
    plan['models'] = [{'id': 'beta', 'kind': 'strong', 'baseline': 'beta', 'fit_seed': 912, 'df': None},
                      {'id': 't4', 'kind': 'strong', 'baseline': 'student_t', 'fit_seed': 912, 'df': 4},
                      {'id': 'copula', 'kind': 'baseline', 'baseline': 'copula'}]
    plan['generation_seeds'] = [61]
    path.write_text(json.dumps(plan))
    runner().evaluate(path, tmp_path / 'scores', workers=1)
    for name in ('beta', 't4', 'copula'):
        with np.load(tmp_path / 'scores' / name / 'generated_61.npz') as saved:
            assert np.max(saved['x']) < .5
        assert read(tmp_path / 'scores' / name / 'model_metadata.json')['fitted_on'] == 'inner train'


def test_main_ensemble_physical_metrics_use_all_real_rows_and_train_range(study, tmp_path):
    data, path = study
    plan = read(path)
    plan.update(models=plan['models'][:1], generation_seeds=[61])
    path.write_text(json.dumps(plan))
    output = tmp_path / 'scores'
    runner().evaluate(path, output, workers=1)
    report = read(output / 'mean/draw_61.json')
    with np.load(data / 'val.npz') as real, np.load(output / 'mean/generated_61.npz') as fake:
        expected = conditional_metrics(real['x'], fake['x'], real['y'], fake['y'],
                                       directions=projection_directions(24, count=128, seed=2026))
    actual = report['physical_metrics']
    scale = read(data / 'scaler.json')
    denominator = scale['maximum'] - scale['minimum']
    for summary in ('macro', 'weighted'):
        assert actual[summary]['normalized_profile_sw1'] == pytest.approx(
            expected[summary]['profile_sw1_kwh'] / denominator)
        for key, value in expected[summary].items():
            assert actual[summary][key] == value
    assert actual['per_condition']['0']['n_real'] > actual['per_condition']['0']['n_fake']
    assert report['physical_metric_specification']['directions_count'] == 128
    assert report['physical_metric_specification']['directions_seed'] == 2026
    assert report['physical_metric_specification']['normalization']['fitted_on'] == 'train'


def test_one_failed_draw_preserves_partial_hashes_and_other_models_continue(study, tmp_path, monkeypatch):
    _, path = study
    evaluate = runner()
    score = evaluate._score_draw
    count = 0
    def fail_second(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError('intentional scoring failure')
        return score(*args, **kwargs)
    monkeypatch.setattr(evaluate, '_score_draw', fail_second)
    output = tmp_path / 'partial'
    with pytest.raises(RuntimeError, match='failed'):
        evaluate.evaluate(path, output, workers=1)
    manifest = read(output / 'manifest.json')
    assert manifest['failed_models'] == ['mean'] and manifest['completed_models'] == 1
    assert manifest['artifact_hashes'] == inventory(output)
    first, second = read(output / 'mean/manifest.json'), read(output / 'resample/manifest.json')
    assert first['status'] == 'failed' and 'intentional scoring failure' in first['error']
    assert second['status'] == 'completed'
    assert first['artifact_hashes'] == inventory(output / 'mean')
    assert 'generated_62.npz' in first['artifact_hashes'] and 'profile_scores_61.npz' in first['artifact_hashes']


def test_spawned_worker_cli_preserves_single_thread_and_generated_labels(study, tmp_path):
    _, path = study
    plan = read(path)
    plan['generation_seeds'] = [61]
    path.write_text(json.dumps(plan))
    output = tmp_path / 'spawned'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_proper_scores.py'),
        '--plan', str(path), '--output', str(output), '--workers', '2'], cwd=ROOT,
        capture_output=True, text=True, timeout=60,
        env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    assert result.returncode == 0, result.stdout + result.stderr
    assert read(output / 'manifest.json')['runtime']['process_mode'] == 'spawn'
    for name in ('mean', 'resample'):
        receipt = read(output / name / 'manifest.json')
        assert receipt['worker_threads'] == 1 and receipt['worker_pid'] != os.getpid()
        assert receipt['status'] == 'completed'


def test_plan_changed_during_preparation_is_rejected(study, tmp_path, monkeypatch):
    _, path = study
    evaluate = runner()
    audit = evaluate._calendar_audit
    def change_plan(values, split):
        result = audit(values, split)
        if split == 'VAL':
            plan = read(path)
            plan['generation_seeds'] = [93]
            path.write_text(json.dumps(plan))
        return result
    monkeypatch.setattr(evaluate, '_calendar_audit', change_plan)
    with pytest.raises(ValueError, match='[Pp]lan.*hash'):
        evaluate.evaluate(path, tmp_path / 'rejected', workers=1)
    assert not (tmp_path / 'rejected').exists()


def test_failure_receipt_io_and_hash_errors_do_not_replace_original_error(study, tmp_path, monkeypatch):
    _, path = study
    plan = read(path)
    plan.update(models=plan['models'][:1], generation_seeds=[61])
    path.write_text(json.dumps(plan))
    evaluate = runner()
    write, sha = evaluate._json, evaluate._sha256
    failed = False
    def fail_score(*args, **kwargs):
        nonlocal failed
        failed = True
        raise RuntimeError('original scoring error')
    def fail_receipt(target, value):
        if target.name == 'draw_61.json':
            raise OSError('receipt unavailable')
        return write(target, value)
    def fail_hash(target):
        if failed and target.name == 'generated_61.npz':
            raise OSError('hash unavailable')
        assert target.name != 'test.npz', 'TEST bytes must never be opened'
        return sha(target)
    monkeypatch.setattr(evaluate, '_score_draw', fail_score)
    monkeypatch.setattr(evaluate, '_json', fail_receipt)
    monkeypatch.setattr(evaluate, '_sha256', fail_hash)
    output = tmp_path / 'partial'
    with pytest.raises(RuntimeError, match='failed'):
        evaluate.evaluate(path, output, workers=1)
    receipt = read(output / 'mean/manifest.json')
    assert receipt['error'] == 'RuntimeError: original scoring error'
    assert 'generated_61.npz' in receipt['artifact_hash_errors']
    assert receipt['draws'][0]['receipt_write_errors'] == ['OSError: receipt unavailable']
    assert 'mean/generated_61.npz' in read(output / 'manifest.json')['artifact_hash_errors']


@pytest.mark.parametrize('failure_location', ['model', 'root'])
def test_inventory_hash_errors_prevent_completed_receipts(study, tmp_path, monkeypatch, failure_location):
    _, path = study
    plan = read(path)
    plan['generation_seeds'] = [61]
    path.write_text(json.dumps(plan))
    evaluate = runner()
    inventory_function = evaluate._inventory
    output = tmp_path / 'scores'
    def fail_inventory(directory):
        result = inventory_function(directory)
        target = directory == output if failure_location == 'root' else directory == output / 'mean'
        if target:
            result['artifact_hash_errors'] = {'unreadable': 'OSError: synthetic inventory error'}
        return result
    monkeypatch.setattr(evaluate, '_inventory', fail_inventory)
    with pytest.raises(RuntimeError, match='failed|[Ii]nventory'):
        evaluate.evaluate(path, output, workers=1)
    receipt = read(output / 'manifest.json')
    assert receipt['status'] == 'failed'
    assert read(output / 'resample/manifest.json')['status'] == 'completed'
    if failure_location == 'model':
        first = read(output / 'mean/manifest.json')
        assert first['status'] == 'failed' and first['artifact_hash_errors']
        assert receipt['failed_models'] == ['mean']
    else:
        assert receipt['artifact_hash_errors']


def test_coordinator_preserves_completed_draw_when_worker_future_fails(study, tmp_path, monkeypatch):
    _, path = study
    evaluate = runner()
    class Future:
        def __init__(self, receipt, fail):
            self.receipt, self.fail = receipt, fail

        def result(self):
            if self.fail:
                raise RuntimeError('worker lost after first draw')
            return self.receipt

    class Executor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, entry, plan, output, provenance):
            fail = entry['id'] == 'mean'
            worker_plan = dict(plan, generation_seeds=plan['generation_seeds'][:1]) if fail else plan
            receipt = function(entry, worker_plan, output, provenance)
            if fail:
                receipt['status'] = 'running'
                evaluate._json(output / 'manifest.json', receipt)
            return Future(receipt, fail)

    monkeypatch.setattr(evaluate, 'ProcessPoolExecutor', Executor)
    monkeypatch.setattr(evaluate, 'as_completed', lambda jobs: list(jobs))
    output = tmp_path / 'partial'
    with pytest.raises(RuntimeError, match='failed'):
        evaluate.evaluate(path, output, workers=2)
    receipt = read(output / 'mean/manifest.json')
    assert receipt['status'] == 'failed'
    assert 'worker lost after first draw' in receipt['worker_error']
    assert len(receipt['draws']) == 1 and receipt['draws'][0]['status'] == 'completed'
    assert receipt['draws'][0]['generation_seed'] == 61
    assert read(output / 'summary.json')['models']['mean']['completed_draws'] == 1
    assert receipt['artifact_hashes'] == inventory(output / 'mean')
    assert read(output / 'resample/manifest.json')['status'] == 'completed'


def test_worker_recovery_uses_draw_files_when_partial_draw_list_is_malformed(tmp_path):
    evaluate = runner()
    entry = {'id': 'mean', 'kind': 'baseline', 'baseline': 'mean'}
    provenance = {'data_hashes': {}, 'plan_sha256': 'a' * 64}
    evaluate._json(tmp_path / 'manifest.json', dict(provenance, model=entry, draws=None,
        error='RuntimeError: earlier model error'))
    evaluate._json(tmp_path / 'draw_61.json', {'model_id': 'mean', 'generation_seed': 61, 'status': 'completed'})
    result = evaluate._worker_failure(entry, {'generation_seeds': [61, 62]}, tmp_path,
                                      provenance, RuntimeError('worker disappeared'))
    assert result['status'] == 'failed' and result['error'] == 'RuntimeError: earlier model error'
    assert result['worker_error'] == 'RuntimeError: worker disappeared'
    assert result['draws'][0]['generation_seed'] == 61
    assert result['interrupted_generation_seeds'] == [62] and result['recovery_errors']


def test_mixture_saved_fit_binding_does_not_refit_and_rejects_changed_model(study, tmp_path, monkeypatch):
    from fluvius_cgan.mixture_baselines import fit_mixture_baseline
    data, path = study
    evaluate = runner()
    model_dir = tmp_path / 'mixture'
    model_dir.mkdir()
    with np.load(data / 'train.npz') as saved:
        fitted = fit_mixture_baseline(saved['x'], saved['y'], 1, n_init=1, max_iter=1)
    fitted.save(model_dir / 'model.npz')
    (model_dir / 'fit_metadata.json').write_text(json.dumps(fitted.metadata()))
    manifest = {'status': 'completed', 'method': fitted.metadata()['method'],
        'fitted_on': 'inner train', 'data_hashes': read(path)['data_hashes'],
        'config': fitted.metadata()['config'], 'model_artifact': 'model.npz',
        'model_sha256': digest(model_dir / 'model.npz'),
        'validation_performance_access': False, 'test_performance_access': False,
        'artifact_hashes': inventory(model_dir)}
    (model_dir / 'manifest.json').write_text(json.dumps(manifest))
    plan = read(path)
    plan['models'] = [{'id': 'mixture', 'kind': 'mixture', 'run_directory': str(model_dir),
        'manifest_sha256': digest(model_dir / 'manifest.json'), 'model_sha256': digest(model_dir / 'model.npz')}]
    plan['generation_seeds'] = [61]
    path.write_text(json.dumps(plan))
    monkeypatch.setattr('fluvius_cgan.mixture_baselines.fit_mixture_baseline',
                        lambda *a, **k: pytest.fail('Saved mixtures must never be refitted'))
    evaluate.evaluate(path, tmp_path / 'valid', workers=1)
    with (model_dir / 'model.npz').open('ab') as handle:
        handle.write(b'tampered')
    with pytest.raises(RuntimeError, match='failed'):
        evaluate.evaluate(path, tmp_path / 'invalid', workers=1)
    assert 'hash' in read(tmp_path / 'invalid/mixture/manifest.json')['error'].lower()


@pytest.mark.parametrize('kind', ['cgan', 'flow'])
def test_neural_binding_and_frozen_calibration_use_selected_checkpoint_only(study, tmp_path, kind, monkeypatch):
    from fluvius_cgan import development, flow
    from fluvius_cgan.calibration import fit_calibration
    data, path = study
    evaluate = runner()
    implementation = development if kind == 'cgan' else flow
    run = tmp_path / kind
    config = {'steps': 1, 'batch_size': 8, 'validation_per_condition': 2}
    config.update({'normalization': 'standard'} if kind == 'cgan' else {'solver_steps': 1})
    implementation.train_probe(data, run, config)
    candidate, _, _ = evaluate.diagnostics._candidate(run, 0)
    labels = np.repeat(np.arange(64), 5)
    source = implementation.generate(run / 'best.pt', labels, 900)
    with np.load(data / 'train.npz') as saved:
        training = {key: saved[key] for key in saved.files}
    fitted = fit_calibration(training['x'], training['y'], source, labels)
    calibration = tmp_path / 'calibration'
    calibration.mkdir()
    fitted.save(calibration / 'candidate_000_calibration.npz')
    fit_record = {'schema_version': 1, 'fitted_on': 'inner train', 'candidate': candidate,
        'checkpoint_sha256': candidate['checkpoint_sha256'], 'data_hashes': read(path)['data_hashes'],
        'calibration_seed': 900, 'calibration': {'seed': 900, 'profiles_per_condition': 5},
        'source_rows': len(source), 'target_rows': len(training['x']),
        'source_profiles_per_condition': 5,
        'target_profiles_per_condition': {str(label): int(np.sum(training['y'] == label)) for label in range(64)},
        'source_profile_bytes_sha256': hashlib.sha256(source.astype('<f8').tobytes()).hexdigest(),
        'source_label_bytes_sha256': hashlib.sha256(labels.astype('<i8').tobytes()).hexdigest(),
        'target_profile_bytes_sha256': hashlib.sha256(training['x'].astype('<f8').tobytes()).hexdigest(),
        'target_label_bytes_sha256': hashlib.sha256(training['y'].astype('<i8').tobytes()).hexdigest(),
        'calibration_artifact': 'candidate_000_calibration.npz',
        'calibration_sha256': digest(calibration / 'candidate_000_calibration.npz')}
    (calibration / 'candidate_000_fit.json').write_text(json.dumps(fit_record))
    receipt = {'status': 'completed', 'data_hashes': read(path)['data_hashes'],
               'calibration': fit_record['calibration'], 'artifact_hashes': inventory(calibration)}
    (calibration / 'manifest.json').write_text(json.dumps(receipt))
    raw = {'id': 'raw', 'kind': 'neural', 'run_directory': str(run),
           'manifest_sha256': digest(run / 'manifest.json'), 'checkpoint_sha256': digest(run / 'best.pt')}
    calibrated = dict(raw, id='calibrated', calibration={'study_directory': str(calibration),
        'manifest_sha256': digest(calibration / 'manifest.json'), 'fit_artifact': 'candidate_000_fit.json',
        'fit_sha256': digest(calibration / 'candidate_000_fit.json'),
        'calibration_artifact': 'candidate_000_calibration.npz',
        'calibration_sha256': digest(calibration / 'candidate_000_calibration.npz')})
    plan = read(path)
    plan.update(models=[raw, calibrated], generation_seeds=[61])
    path.write_text(json.dumps(plan))
    evaluate.evaluate(path, tmp_path / 'scores', workers=1)
    expected = implementation.generate(run / 'best.pt', np.repeat(np.arange(64), 4), 61)
    with np.load(tmp_path / 'scores/raw/generated_61.npz') as saved:
        np.testing.assert_array_equal(saved['x'], expected)
    mapped, _ = fitted.apply(expected, np.repeat(np.arange(64), 4))
    with np.load(tmp_path / 'scores/calibrated/generated_61.npz') as saved:
        np.testing.assert_array_equal(saved['x'], mapped)
    calibrated['calibration']['fit_sha256'] = '0' * 64
    path.write_text(json.dumps(plan))
    with pytest.raises(RuntimeError, match='failed'):
        evaluate.evaluate(path, tmp_path / 'bad_binding', workers=1)
    assert read(tmp_path / 'bad_binding/raw/manifest.json')['status'] == 'completed'
    assert read(tmp_path / 'bad_binding/calibrated/manifest.json')['status'] == 'failed'
    sample, _ = evaluate._bind_model(raw, training, plan['data_hashes'], read(data / 'scaler.json'), [61])
    checkpoint = run / 'best.pt'
    original_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(original_bytes + b'changed after binding')
    with pytest.raises(ValueError, match='hash'):
        sample(np.repeat(np.arange(64), 4), 61)
    checkpoint.write_bytes(original_bytes)
    generate = implementation.generate
    def changed_during_generation(*args, **kwargs):
        result = generate(*args, **kwargs)
        checkpoint.write_bytes(original_bytes + b'changed during generation')
        return result
    monkeypatch.setattr(implementation, 'generate', changed_during_generation)
    sample, _ = evaluate._bind_model(raw, training, plan['data_hashes'], read(data / 'scaler.json'), [61])
    with pytest.raises(ValueError, match='hash'):
        sample(np.repeat(np.arange(64), 4), 61)
