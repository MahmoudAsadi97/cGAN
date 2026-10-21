"""Frozen calibration must bind its fitting inputs and preserve paired controls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

from fluvius_cgan import development, flow


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    path = ROOT / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    old = sys.path.copy()
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(result)
    finally:
        sys.path[:] = old
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


@pytest.fixture(scope='module')
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp('calibration-study')
    data = root / 'data'
    data.mkdir()
    rng = np.random.default_rng(91)
    labels = np.repeat(np.array([0, 8], dtype=np.int64), 12)
    for split, offset in [('train', 0.), ('val', 10.)]:
        # Disjoint numeric supports make accidental VAL-fitted quantiles visible.
        x = (rng.uniform(.1, 1., (24, 24)) + offset).astype(np.float32)
        np.savez(data / f'{split}.npz', x=x, y=labels,
                 meter_id=np.array([f'{split}{i}' for i in range(24)]),
                 date=np.full(24, '2024-01-01'))
    with np.load(data / 'train.npz') as train:
        scaler = {'minimum': float(train['x'].min()), 'maximum': float(train['x'].max()),
                  'fitted_on': 'train'}
    (data / 'scaler.json').write_text(json.dumps(scaler))
    (data / 'metadata.json').write_text(json.dumps({'resolution': 24, 'condition_dim': 64,
        'development': {'source_partition_role': 'original_train'}}))
    (data / 'test.npz').write_bytes(b'FORBIDDEN TEST DATA')
    common = {'steps': 1, 'batch_size': 8, 'validation_per_condition': 4}
    cgan, cflow = root / 'cgan', root / 'flow'
    development.train_probe(data, cgan, common | {'normalization': 'standard'})
    flow.train_probe(data, cflow, common | {'solver_steps': 1})
    reference = root / 'reference'
    module('evaluate_development_candidates').evaluate(data, [cgan, cflow], reference)
    plan = {'schema_version': 1, 'phase': 1, 'data': str(data),
        'data_hashes': {name: digest(data / name) for name in
                        ('train.npz', 'val.npz', 'scaler.json', 'metadata.json')},
        'calibration': {'seed': 20261025, 'profiles_per_condition': 32,
            'source_cdf': 'deterministic_mid_ecdf', 'target_inverse': 'empirical_step',
            'constant_source_policy': 'fail_if_target_varies'},
        'generation_seeds': [20261023, 20261024],
        'candidates': [{'run_directory': str(run), 'checkpoint_sha256': digest(run / 'best.pt'),
                        'manifest_sha256': digest(run / 'manifest.json'),
                        'reference_method_id': f'candidate_{i:03d}'} for i, run in enumerate((cgan, cflow))],
        'reference': {'directory': str(reference), 'manifest_sha256': digest(reference / 'manifest.json')}}
    plan_path = root / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    return root, plan_path


def test_calibration_uses_train_support_and_preserves_raw_controls(study, tmp_path):
    root, plan_path = study
    runner = module('evaluate_marginal_calibration')
    output = tmp_path / 'result'
    result = runner.evaluate(plan_path, output)
    assert set(result['methods']) == {
        'candidate_000_raw', 'candidate_000_calibrated',
        'candidate_001_raw', 'candidate_001_calibrated',
        'baseline_mean', 'baseline_resample', 'baseline_copula'}
    manifest = read(output / 'manifest.json')
    assert manifest['status'] == 'completed' and manifest['completed_draws'] == 14
    assert manifest['test_performance_access'] is False
    assert manifest['reselection'] is False and manifest['clipping'] is False
    assert manifest['plan_sha256'] == digest(plan_path)
    assert manifest['data_hashes'] == read(plan_path)['data_hashes']
    assert manifest['calibration']['seed'] == 20261025
    assert (root / 'data' / 'test.npz').read_bytes() == b'FORBIDDEN TEST DATA'
    for name, expected in manifest['artifact_hashes'].items():
        assert digest(output / name) == expected
    with np.load(root / 'data' / 'train.npz') as archive:
        training, train_labels = archive['x'], archive['y']
    with np.load(root / 'data' / 'val.npz') as archive:
        labels = archive['y'][manifest['selection']['indices']]
    for i, (name, generate) in enumerate((('cgan', development.generate), ('flow', flow.generate))):
        prefix = f'candidate_{i:03d}'
        raw = read(output / f'{prefix}_raw.json')
        calibrated = read(output / f'{prefix}_calibrated.json')
        reference = read(root / 'reference' / f'{prefix}.json')
        fit = read(output / f'{prefix}_fit.json')
        assert fit['fitted_on'] == 'inner train'
        assert fit['target_rows'] == len(training)
        assert fit['source_rows'] == 64
        assert fit['calibration_seed'] == 20261025
        assert fit['checkpoint_sha256'] == digest(root / name / 'best.pt')
        source_labels = np.repeat(np.unique(train_labels), 32)
        expected_source = generate(root / name / 'best.pt', source_labels, 20261025)
        assert fit['source_profile_bytes_sha256'] == hashlib.sha256(expected_source.astype('<f8').tobytes()).hexdigest()
        fitted = runner.load_calibration(output / fit['calibration_artifact'], fit['calibration_sha256'])
        for draw, changed, previous in zip(raw['draws'], calibrated['draws'], reference['draws']):
            assert draw['generated_profile_bytes_sha256'] == previous['generated_profile_bytes_sha256']
            assert draw['metrics'] == previous['metrics']
            assert draw['reference_draw_match'] is True
            fake = generate(root / name / 'best.pt', labels, draw['generation_seed'])
            mapped, diagnostics = fitted.apply(fake, labels)
            assert changed['generated_profile_bytes_sha256'] == hashlib.sha256(mapped.astype('<f8').tobytes()).hexdigest()
            assert changed['calibration_diagnostics'] == diagnostics
            assert np.max(mapped) < 1. and np.min(mapped) >= .1
            for condition in np.unique(labels):
                for hour in range(24):
                    assert np.isin(mapped[labels == condition, hour], training[train_labels == condition, hour]).all()
        assert all(draw['reference_draw_match'] for draw in read(output / 'baseline_copula.json')['draws'])
    original = (output / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        runner.evaluate(plan_path, output)
    assert (output / 'manifest.json').read_bytes() == original


@pytest.mark.parametrize('mutation,match', [
    (lambda p: p['calibration'].update(seed=20261023), 'seed'),
    (lambda p: p['calibration'].update(seed=20261022), 'selection'),
    (lambda p: p.update(generation_seeds=[20261022]), 'selection'),
    (lambda p: p.update(generation_seeds=[20261023, 20261023]), 'distinct'),
    (lambda p: p['calibration'].update(profiles_per_condition=0), 'profiles_per_condition'),
    (lambda p: p['calibration'].update(source_cdf='batch_ranks'), 'calibration'),
    (lambda p: p['data_hashes'].update({'train.npz': '0' * 64}), 'data hash'),
    (lambda p: p['candidates'][0].update(checkpoint_sha256='0' * 64), 'checkpoint hash'),
    (lambda p: p['candidates'][0].update(manifest_sha256='0' * 64), 'manifest hash'),
    (lambda p: p['reference'].update(manifest_sha256='0' * 64), 'reference manifest hash'),
    (lambda p: p['candidates'][0].update(reference_method_id='../candidate_000'), 'reference_method_id'),
])
def test_plan_rejects_provenance_and_seed_violations_before_output(study, tmp_path, mutation, match):
    _, original = study
    plan = read(original)
    mutation(plan)
    changed = tmp_path / 'plan.json'
    changed.write_text(json.dumps(plan))
    output = tmp_path / 'rejected'
    with pytest.raises(ValueError, match=match):
        module('evaluate_marginal_calibration').evaluate(changed, output)
    assert not output.exists()


def test_failed_generation_keeps_receipt_and_restores_torch_state(study, tmp_path, monkeypatch):
    _, plan = study
    runner = module('evaluate_marginal_calibration')
    def fail(*args, **kwargs):
        torch.manual_seed(83)
        raise FloatingPointError('synthetic generation failure')
    monkeypatch.setattr(development, 'generate', fail)
    before_threads, before_rng = torch.get_num_threads(), torch.get_rng_state().clone()
    with pytest.raises(FloatingPointError, match='synthetic generation failure'):
        runner.evaluate(plan, tmp_path / 'failed')
    manifest = read(tmp_path / 'failed' / 'manifest.json')
    assert manifest['status'] == 'failed' and manifest['completed_draws'] == 0
    assert 'synthetic generation failure' in manifest['error']
    assert torch.get_num_threads() == before_threads
    assert torch.equal(torch.get_rng_state(), before_rng)


def test_failure_after_scoring_binds_every_retained_artifact(study, tmp_path, monkeypatch):
    _, plan = study
    runner = module('evaluate_marginal_calibration')
    original = development.generate
    def fail_second_draw(checkpoint, labels, seed):
        if seed == 20261024:
            raise FloatingPointError('second draw failed')
        return original(checkpoint, labels, seed)
    monkeypatch.setattr(development, 'generate', fail_second_draw)
    output = tmp_path / 'partial'
    with pytest.raises(FloatingPointError, match='second draw failed'):
        runner.evaluate(plan, output)
    manifest = read(output / 'manifest.json')
    assert manifest['status'] == 'failed' and manifest['completed_draws'] == 2
    retained = {path.name: digest(path) for path in output.iterdir()
                if path.is_file() and path.name != 'manifest.json'}
    assert set(retained) == {'plan.json', 'candidate_000_calibration.npz',
        'candidate_000_fit.json', 'candidate_000_raw.json', 'candidate_000_calibrated.json'}
    assert manifest['artifact_hashes'] == retained


def test_corrupted_saved_calibration_aborts_before_scoring(study, tmp_path, monkeypatch):
    _, plan = study
    runner = module('evaluate_marginal_calibration')
    load = runner.load_calibration
    def corrupt(path, expected_sha256):
        with Path(path).open('ab') as handle:
            handle.write(b'corruption')
        return load(path, expected_sha256)
    monkeypatch.setattr(runner, 'load_calibration', corrupt)
    with pytest.raises(ValueError, match='hash|SHA'):
        runner.evaluate(plan, tmp_path / 'corrupt')
    manifest = read(tmp_path / 'corrupt' / 'manifest.json')
    assert manifest['status'] == 'failed' and manifest['completed_draws'] == 0


def test_reference_report_corruption_is_rejected(study, tmp_path):
    root, original = study
    reference = tmp_path / 'reference'
    shutil.copytree(root / 'reference', reference)
    with (reference / 'candidate_000.json').open('a') as handle:
        handle.write(' ')
    plan = read(original)
    plan['reference']['directory'] = str(reference)
    changed = tmp_path / 'plan.json'
    changed.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='Artifact hash mismatch'):
        module('evaluate_marginal_calibration').evaluate(changed, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()


def test_reference_cannot_substitute_a_different_frozen_generator(study, tmp_path):
    _, original = study
    plan = read(original)
    plan['candidates'] = plan['candidates'][:1]
    plan['candidates'][0]['reference_method_id'] = 'candidate_001'
    changed = tmp_path / 'plan.json'
    changed.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='Reference candidate identity differs'):
        module('evaluate_marginal_calibration').evaluate(changed, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()


def test_duplicate_generator_declaration_is_not_a_training_replicate(study, tmp_path):
    _, original = study
    plan = read(original)
    plan['candidates'].append(plan['candidates'][0].copy())
    changed = tmp_path / 'plan.json'
    changed.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='must be distinct'):
        module('evaluate_marginal_calibration').evaluate(changed, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()
