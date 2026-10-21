"""Fit-only mixture replicates never parse validation or score held-out rows."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def runner():
    path = ROOT / 'scripts/fit_mixture_replicates.py'
    assert path.exists(), 'The bounded TRAIN-only replicate runner is not implemented'
    spec = importlib.util.spec_from_file_location('mixture_replicate_runner', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(directory):
    return {str(path.relative_to(directory)): digest(path) for path in directory.rglob('*')
            if path.is_file() and path != directory / 'manifest.json'}


@pytest.fixture
def study(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    rng = np.random.default_rng(17)
    x = rng.normal(.2, .4, (40, 24)).astype(np.float32)
    x[:, 0] = 0
    y = np.repeat(np.array([0, 8], dtype=np.int64), 20)
    np.savez(data / 'train.npz', x=x, y=y, meter_id=np.array([f'm{i}' for i in range(40)]),
             date=np.full(40, '2024-01-01'))
    (data / 'val.npz').write_bytes(b'VALIDATION MAY BE HASHED BUT MUST NEVER BE PARSED')
    (data / 'test.npz').write_bytes(b'TEST MUST NEVER BE READ')
    (data / 'scaler.json').write_text(json.dumps({'minimum': float(x.min()),
        'maximum': float(x.max()), 'fitted_on': 'train'}))
    (data / 'metadata.json').write_text(json.dumps({'resolution': 24, 'condition_dim': 64,
        'development': {'source_partition_role': 'original_train'}}))
    config = dict(n_components=2, n_init=2, max_iter=1, tol=.001, reg_covar=.001)
    plan = {'schema_version': 1, 'phase': 1, 'data': str(data),
        'data_hashes': {name: digest(data / name) for name in ('train.npz', 'val.npz', 'scaler.json', 'metadata.json')},
        'jobs': {'first': config | {'fit_seed': 20261027}, 'second': config | {'fit_seed': 20261101}}}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    return data, path


def test_fit_only_preserves_each_model_and_every_initialization_without_validation(study, tmp_path, monkeypatch):
    data, plan = study
    fit = runner()
    loaded = []
    load = fit.load_split
    def train_only(path):
        assert Path(path).name == 'train.npz', 'Only TRAIN profiles may be loaded'
        loaded.append(Path(path).name)
        return load(path)
    monkeypatch.setattr(fit, 'load_split', train_only)
    output = tmp_path / 'fits'
    torch_rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    result = fit.run(plan, output)
    manifest = read(output / 'manifest.json')
    assert manifest['status'] == result['status'] == 'completed'
    assert manifest['completed_jobs'] == 2 and manifest['failed_jobs'] == []
    assert manifest['validation_performance_access'] is False and manifest['test_performance_access'] is False
    assert manifest['plan_sha256'] == digest(plan)
    assert manifest['data_hashes'] == read(plan)['data_hashes']
    assert manifest['artifact_hashes'] == inventory(output)
    assert loaded == ['train.npz']
    with np.load(data / 'train.npz') as saved:
        x, y = saved['x'], saved['y']
    for name, config in read(plan)['jobs'].items():
        receipt = read(output / name / 'manifest.json')
        metadata = read(output / name / 'fit_metadata.json')
        assert receipt['status'] == 'completed' and receipt['config'] == config
        assert receipt['plan_sha256'] == digest(plan) and receipt['data_hashes'] == manifest['data_hashes']
        assert receipt['artifact_hashes'] == inventory(output / name)
        assert receipt['model_sha256'] == digest(output / name / 'model.npz')
        restored = fit.load_mixture_baseline(output / name / 'model.npz', receipt['model_sha256'])
        assert restored.metadata() == metadata
        assert metadata['config'] == config
        assert metadata['source_profile_bytes_sha256'] == hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()
        assert metadata['source_label_bytes_sha256'] == hashlib.sha256(y.astype('<i8').tobytes()).hexdigest()
        assert all(len(group['initializations']) == 2 for group in metadata['per_condition'].values())
        assert all(attempt['n_iter'] == 1 and not attempt['converged'] and attempt['warnings']
                   for group in metadata['per_condition'].values() for attempt in group['initializations'])
        assert 'metrics' not in receipt and 'draws' not in receipt
    assert torch.equal(torch_rng, torch.get_rng_state()) and torch.get_num_threads() == threads
    after = np.random.get_state()
    assert numpy_rng[0] == after[0] and np.array_equal(numpy_rng[1], after[1]) and numpy_rng[2:] == after[2:]
    assert random.getstate() == python_rng
    previous = (output / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        fit.run(plan, output)
    assert (output / 'manifest.json').read_bytes() == previous


@pytest.mark.parametrize('mutation,match', [
    (lambda p: p.update(phase=2), 'phase'),
    (lambda p: p['data_hashes'].update({'train.npz': '0' * 64}), 'hash'),
    (lambda p: p.update(jobs={}), 'jobs'),
    (lambda p: p['jobs'].update({'../escape': p['jobs'].pop('first')}), 'name'),
    (lambda p: p['jobs']['first'].update(max_iter=0), 'max_iter'),
    (lambda p: p['jobs']['first'].update(fit_seed=True), '[Ss]eed'),
    (lambda p: p['jobs']['second'].update(fit_seed=p['jobs']['first']['fit_seed']), 'distinct'),
    (lambda p: p['jobs']['second'].update(max_iter=2), 'same|share'),
])
def test_invalid_plan_fails_before_output(study, tmp_path, mutation, match):
    _, path = study
    plan = read(path)
    mutation(plan)
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=match):
        runner().run(path, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()


@pytest.mark.parametrize('change', ['bad_train', 'source_role', 'scaler'])
def test_malformed_training_provenance_fails_before_output(study, tmp_path, change):
    data, path = study
    plan = read(path)
    if change == 'bad_train':
        np.savez(data / 'train.npz', x=np.full((2, 24), np.nan), y=np.zeros(2))
    elif change == 'source_role':
        (data / 'metadata.json').write_text(json.dumps({'development': {'source_partition_role': 'original_test'}}))
    else:
        (data / 'scaler.json').write_text(json.dumps({'minimum': 0., 'maximum': 999., 'fitted_on': 'val'}))
    plan['data_hashes'] = {name: digest(data / name) for name in plan['data_hashes']}
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        runner().run(path, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()


def test_failed_first_saved_model_retains_receipt_and_next_job_finishes(study, tmp_path, monkeypatch):
    _, plan = study
    fit = runner()
    load = fit.load_mixture_baseline
    def corrupt(path, expected_sha256):
        if Path(path).parent.name == 'first':
            torch.manual_seed(8123)
            with Path(path).open('ab') as handle:
                handle.write(b'corruption')
        return load(path, expected_sha256)
    monkeypatch.setattr(fit, 'load_mixture_baseline', corrupt)
    output = tmp_path / 'partial'
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    with pytest.raises(RuntimeError, match='failed'):
        fit.run(plan, output)
    manifest = read(output / 'manifest.json')
    assert manifest['status'] == 'failed' and manifest['completed_jobs'] == 1
    assert manifest['failed_jobs'] == ['first']
    first, second = read(output / 'first/manifest.json'), read(output / 'second/manifest.json')
    assert first['status'] == 'failed' and 'SHA' in first['error']
    assert second['status'] == 'completed'
    assert first['artifact_hashes'] == inventory(output / 'first')
    assert second['artifact_hashes'] == inventory(output / 'second')
    assert manifest['artifact_hashes'] == inventory(output)
    assert read(output / 'summary.json')['jobs']['first']['status'] == 'failed'
    assert torch.equal(rng, torch.get_rng_state()) and torch.get_num_threads() == threads


def test_metadata_reload_mismatch_is_a_failed_job(study, tmp_path, monkeypatch):
    _, plan = study
    fit = runner()
    original = fit.load_mixture_baseline
    class Changed:
        def __init__(self, model):
            self.model = model
        def metadata(self):
            metadata = self.model.metadata()
            metadata['training_profiles'] += 1
            return metadata
    def change(path, expected_sha256):
        result = original(path, expected_sha256)
        return Changed(result) if Path(path).parent.name == 'first' else result
    monkeypatch.setattr(fit, 'load_mixture_baseline', change)
    with pytest.raises(RuntimeError, match='failed'):
        fit.run(plan, tmp_path / 'changed')
    assert 'metadata' in read(tmp_path / 'changed/first/manifest.json')['error']
    assert read(tmp_path / 'changed/second/manifest.json')['status'] == 'completed'


@pytest.mark.parametrize('scope', ['job', 'root'])
def test_inventory_errors_never_claim_completion_and_other_jobs_are_retained(study, tmp_path, monkeypatch, scope):
    _, plan = study
    fit = runner()
    original = fit._inventory
    output = tmp_path / 'inventory_failure'
    def incomplete(directory, recursive=False):
        result = original(directory, recursive)
        if (scope == 'job' and directory == output / 'first') or (scope == 'root' and directory == output):
            result['artifact_hash_errors'] = {'synthetic': 'PermissionError: could not hash an artifact'}
        return result
    monkeypatch.setattr(fit, '_inventory', incomplete)
    with pytest.raises(RuntimeError, match='inventory|failed'):
        fit.run(plan, output)
    manifest = read(output / 'manifest.json')
    assert manifest['status'] == 'failed'
    assert read(output / 'second/manifest.json')['status'] == 'completed'
    if scope == 'job':
        assert manifest['failed_jobs'] == ['first']
        first = read(output / 'first/manifest.json')
        assert first['status'] == 'failed' and 'synthetic' in first['artifact_hash_errors']
    else:
        assert manifest['completed_jobs'] == 2
        assert 'synthetic' in manifest['artifact_hash_errors']


def test_plan_receipt_binds_initial_parsed_bytes_even_if_input_path_changes(study, tmp_path, monkeypatch):
    _, plan = study
    fit = runner()
    initial = plan.read_bytes()
    original = fit.load_split
    def replace_plan_during_loading(path):
        training = original(path)
        changed = read(plan)
        changed['jobs']['first']['fit_seed'] = 41
        plan.write_text(json.dumps(changed))
        return training
    monkeypatch.setattr(fit, 'load_split', replace_plan_during_loading)
    output = tmp_path / 'bound_snapshot'
    fit.run(plan, output)
    manifest = read(output / 'manifest.json')
    expected_sha = hashlib.sha256(initial).hexdigest()
    assert plan.read_bytes() != initial
    assert (output / 'plan.json').read_bytes() == initial
    assert manifest['plan_sha256'] == expected_sha
    assert read(output / 'first/manifest.json')['plan_sha256'] == expected_sha
    assert read(output / 'first/fit_metadata.json')['config'] == json.loads(initial)['jobs']['first']
