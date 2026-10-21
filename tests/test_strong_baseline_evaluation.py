"""Stronger controls retain paired model provenance and expose TRAIN proximity."""
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
    previous = sys.path.copy()
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(result)
    finally:
        sys.path[:] = previous
    return result


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope='module')
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp('strong-baseline-study')
    data = root / 'data'
    data.mkdir()
    rng = np.random.default_rng(93)
    labels = np.repeat(np.array([0, 8], dtype=np.int64), 12)
    for split, offset in [('train', 0.), ('val', 3.)]:
        x = (rng.uniform(.1, 1., (24, 24)) + offset).astype(np.float32)
        np.savez(data / f'{split}.npz', x=x, y=labels,
                 meter_id=np.array([f'{split}{i}' for i in range(24)]),
                 date=np.full(24, '2024-01-01'))
    with np.load(data / 'train.npz') as archive:
        scaler = {'minimum': float(archive['x'].min()), 'maximum': float(archive['x'].max()),
                  'fitted_on': 'train'}
    (data / 'scaler.json').write_text(json.dumps(scaler))
    (data / 'metadata.json').write_text(json.dumps({'resolution': 24, 'condition_dim': 64,
        'development': {'source_partition_role': 'original_train'}}))
    (data / 'test.npz').write_bytes(b'FORBIDDEN TEST DATA')
    common = {'steps': 1, 'batch_size': 8, 'validation_per_condition': 4}
    cgan, cflow = root / 'cgan', root / 'flow'
    development.train_probe(data, cgan, common | {'normalization': 'standard'})
    flow.train_probe(data, cflow, common | {'solver_steps': 1})
    reference, calibrated = root / 'reference', root / 'calibrated'
    module('evaluate_development_candidates').evaluate(data, [cgan, cflow], reference)
    data_hashes = {name: digest(data / name) for name in
                   ('train.npz', 'val.npz', 'scaler.json', 'metadata.json')}
    calibration_plan = {'schema_version': 1, 'phase': 1, 'data': str(data), 'data_hashes': data_hashes,
        'calibration': {'seed': 20261025, 'profiles_per_condition': 32,
            'source_cdf': 'deterministic_mid_ecdf', 'target_inverse': 'empirical_step',
            'constant_source_policy': 'fail_if_target_varies'},
        'generation_seeds': [20261023, 20261024],
        'candidates': [{'run_directory': str(run), 'checkpoint_sha256': digest(run / 'best.pt'),
            'manifest_sha256': digest(run / 'manifest.json'), 'reference_method_id': f'candidate_{i:03d}'}
            for i, run in enumerate((cgan, cflow))],
        'reference': {'directory': str(reference), 'manifest_sha256': digest(reference / 'manifest.json')}}
    calibration_path = root / 'calibration_plan.json'
    calibration_path.write_text(json.dumps(calibration_plan))
    module('evaluate_marginal_calibration').evaluate(calibration_path, calibrated)
    plan = {'schema_version': 1, 'phase': 1, 'data': str(data), 'data_hashes': data_hashes,
        'generation_seeds': [20261023, 20261024],
        'methods': [{'method_id': 'beta', 'kind': 'beta', 'fit_seed': 20261026},
                    *[{'method_id': f't_df{df}', 'kind': 'student_t', 'df': df} for df in (4, 8, 16)]],
        'reference': calibration_plan['reference'],
        'calibration_reference': {'directory': str(calibrated), 'manifest_sha256': digest(calibrated / 'manifest.json')}}
    path = root / 'plan.json'
    path.write_text(json.dumps(plan))
    return root, path


def test_stronger_controls_and_reused_generators_have_verified_proximity(study, tmp_path):
    root, plan = study
    runner = module('evaluate_strong_baselines')
    output = tmp_path / 'result'
    result = runner.evaluate(plan, output)
    expected = {'strong_beta', 'strong_t_df4', 'strong_t_df8', 'strong_t_df16',
        'baseline_mean', 'baseline_resample', 'baseline_copula',
        'candidate_000_raw', 'candidate_000_calibrated',
        'candidate_001_raw', 'candidate_001_calibrated'}
    assert set(result['methods']) == expected
    manifest = read(output / 'manifest.json')
    assert manifest['status'] == 'completed' and manifest['completed_draws'] == 22
    assert manifest['test_performance_access'] is False and manifest['reselection'] is False
    assert manifest['plan_sha256'] == digest(plan)
    assert manifest['data_hashes'] == read(plan)['data_hashes']
    assert manifest['calibration_refitted'] is False
    assert (root / 'data' / 'test.npz').read_bytes() == b'FORBIDDEN TEST DATA'
    for name, expected_hash in manifest['artifact_hashes'].items():
        assert digest(output / name) == expected_hash
    real = read(output / 'real_proximity.json')
    assert real['proximity']['macro']['mean_rms_kwh'] > 2.
    report = read(output / 'baseline_resample.json')
    assert all(draw['reference_draw_match'] for draw in report['draws'])
    assert all(draw['proximity']['macro']['mean_rms_kwh'] == 0. for draw in report['draws'])
    assert all(draw['proximity']['macro']['below_real_p05_fraction'] == 1. for draw in report['draws'])
    for prefix in ('candidate_000', 'candidate_001'):
        for suffix in ('raw', 'calibrated'):
            name = f'{prefix}_{suffix}'
            actual = read(output / f'{name}.json')
            prior = read(root / 'calibrated' / f'{name}.json')
            for draw, old in zip(actual['draws'], prior['draws']):
                assert draw['reference_draw_match'] is True
                assert draw['generated_profile_bytes_sha256'] == old['generated_profile_bytes_sha256']
                assert draw['metrics'] == old['metrics']
                assert draw['proximity']['distance_bytes_sha256']
                assert set(draw['proximity']['per_condition']) == {'0', '8'}
    beta = read(output / 'strong_beta.json')
    assert beta['fit_metadata']['fit_seed'] == 20261026
    assert beta['fitted_on'] == 'inner train'
    assert beta['fit_seconds'] >= 0
    assert all(not draw['reference_draw_match'] for draw in beta['draws'])
    original = (output / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        runner.evaluate(plan, output)
    assert (output / 'manifest.json').read_bytes() == original


@pytest.mark.parametrize('mutation,match', [
    (lambda p: p.update(generation_seeds=[20261023, 20261023]), 'distinct'),
    (lambda p: p.update(generation_seeds=[20261022]), 'selection'),
    (lambda p: p.update(generation_seeds=[20261025]), 'calibration'),
    (lambda p: p['methods'][0].update(fit_seed=20261023), 'fit seed'),
    (lambda p: p['methods'][0].update(method_id='../beta'), 'method_id'),
    (lambda p: p['methods'].append(p['methods'][0].copy()), 'distinct'),
    (lambda p: p['methods'][1].update(df=2), 'df'),
    (lambda p: p['data_hashes'].update({'train.npz': '0' * 64}), 'data hash'),
    (lambda p: p['reference'].update(manifest_sha256='0' * 64), 'reference manifest hash'),
    (lambda p: p['calibration_reference'].update(manifest_sha256='0' * 64), 'reference manifest hash'),
])
def test_plan_and_reference_violations_fail_before_output(study, tmp_path, mutation, match):
    _, original = study
    plan = read(original)
    mutation(plan)
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=match):
        module('evaluate_strong_baselines').evaluate(path, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()


def test_saved_calibration_tampering_is_rejected_before_scoring(study, tmp_path):
    root, original = study
    changed_reference = tmp_path / 'calibrated'
    shutil.copytree(root / 'calibrated', changed_reference)
    with (changed_reference / 'candidate_000_calibration.npz').open('ab') as handle:
        handle.write(b'tamper')
    plan = read(original)
    plan['calibration_reference']['directory'] = str(changed_reference)
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='Artifact hash mismatch'):
        module('evaluate_strong_baselines').evaluate(path, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()


def test_frozen_checkpoint_tampering_is_rejected_before_scoring(study, tmp_path):
    root, plan = study
    checkpoint = root / 'cgan' / 'best.pt'
    original = checkpoint.read_bytes()
    try:
        checkpoint.write_bytes(original + b'tamper')
        with pytest.raises(ValueError, match='checkpoint hash mismatch'):
            module('evaluate_strong_baselines').evaluate(plan, tmp_path / 'rejected')
        assert not (tmp_path / 'rejected').exists()
    finally:
        checkpoint.write_bytes(original)


def test_partial_failure_retains_hashed_draws_and_restores_runtime(study, tmp_path, monkeypatch):
    _, plan = study
    runner = module('evaluate_strong_baselines')
    original = development.generate
    def fail_second(checkpoint, labels, seed):
        if seed == 20261024:
            torch.manual_seed(987)
            raise FloatingPointError('second model draw failed')
        return original(checkpoint, labels, seed)
    monkeypatch.setattr(development, 'generate', fail_second)
    threads, rng = torch.get_num_threads(), torch.get_rng_state().clone()
    output = tmp_path / 'failed'
    with pytest.raises(FloatingPointError, match='second model draw failed'):
        runner.evaluate(plan, output)
    manifest = read(output / 'manifest.json')
    assert manifest['status'] == 'failed' and manifest['completed_draws'] >= 2
    assert manifest['artifact_hashes'] == {path.name: digest(path) for path in output.iterdir()
        if path.is_file() and path.name != 'manifest.json'}
    assert torch.get_num_threads() == threads and torch.equal(torch.get_rng_state(), rng)
