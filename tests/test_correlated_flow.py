import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluvius_cgan import flow


def make_data(path):
    path.mkdir()
    rng = np.random.default_rng(8)
    common = rng.normal(size=(30, 1))
    training = (common + rng.normal(scale=.3, size=(30, 24))).astype(np.float32)
    training[:, 23] = 4.
    for split, x in [('train', training), ('val', rng.normal(20., 5., (20, 24)).astype(np.float32))]:
        np.savez(path / f'{split}.npz', x=x, y=np.arange(len(x), dtype=np.int64) % 2,
                 meter_id=np.array([f'{split}{i}' for i in range(len(x))]),
                 date=np.full(len(x), '2024-01-01'))
    (path / 'scaler.json').write_text(json.dumps({'minimum': float(training.min()),
        'maximum': float(training.max()), 'fitted_on': 'train'}))
    (path / 'metadata.json').write_text(json.dumps({'resolution': 24, 'condition_dim': 64,
        'development': {'source_partition_role': 'original_train'}}))
    (path / 'test.npz').write_bytes(b'FORBIDDEN')
    return training


def same(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for first, second in zip(a, b):
            same(first, second)
    else:
        assert a == b


def test_fitted_source_preserves_unit_variance_and_empirical_correlation(tmp_path):
    training = make_data(tmp_path / 'data')
    prior = flow.fit_source_prior(training, 'training_correlation')
    expected = np.eye(24)
    expected[:23, :23] = .99 * np.corrcoef(training[:, :23].astype(np.float64), rowvar=False) + .01 * np.eye(23)
    correlation = np.asarray(prior['correlation'])
    factor = np.asarray(prior['cholesky'])
    np.testing.assert_allclose(correlation, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(factor @ factor.T, expected, atol=1e-12)
    np.testing.assert_array_equal(np.diag(correlation), np.ones(24))
    assert np.linalg.eigvalsh(correlation).min() >= .00999999
    assert prior['source_mean'] == [0.] * 24
    assert prior['fit']['constant_columns'] == [23]
    assert prior['fit']['rows'] == len(training)
    base = torch.from_numpy(np.random.default_rng(412).standard_normal((100000, 24), dtype=np.float32))
    transformed = flow._source_noise(base, flow._prior_factor(prior))
    torch.testing.assert_close(transformed, base @ torch.tensor(factor, dtype=torch.float32).T, rtol=0, atol=0)
    np.testing.assert_allclose(np.cov(transformed.numpy(), rowvar=False, ddof=0), expected, atol=.02)


def test_all_constant_source_is_identity_and_invalid_kind_is_rejected():
    prior = flow.fit_source_prior(np.full((1, 24), 3.), 'training_correlation')
    np.testing.assert_array_equal(prior['correlation'], np.eye(24))
    assert prior['fit']['constant_columns'] == list(range(24))
    with pytest.raises(ValueError):
        flow.fit_source_prior(np.ones((2, 24)), 'unknown')


def test_isotropic_default_matches_explicit_prior_and_legacy_generation(tmp_path):
    make_data(tmp_path / 'data')
    config = {'steps': 3, 'batch_size': 7, 'validation_every': 1,
              'validation_per_condition': 3, 'solver_steps': 2, 'seed': 101}
    old = flow.train_probe(tmp_path / 'data', tmp_path / 'old', config | {'source_prior': 'isotropic'})
    new = flow.train_probe(tmp_path / 'data', tmp_path / 'new', config)
    a = torch.load(tmp_path / 'old' / 'last.pt', weights_only=True)
    b = torch.load(tmp_path / 'new' / 'last.pt', weights_only=True)
    for key in ('velocity_model', 'optimizer', 'model_rng_state', 'noise_rng_state',
                'time_rng_state', 'sampler_rng_state', 'sampler_order'):
        same(a[key], b[key])
    assert old['best_validation_sw1'] == new['best_validation_sw1']
    labels = np.array([0, 1, 63], dtype=np.int64)
    legacy = torch.load(tmp_path / 'old' / 'best.pt', weights_only=True)
    legacy.pop('source_prior')
    legacy['config'].pop('source_prior')
    torch.save(legacy, tmp_path / 'legacy.pt')
    np.testing.assert_array_equal(flow.generate(tmp_path / 'old' / 'best.pt', labels, 42),
                                  flow.generate(tmp_path / 'legacy.pt', labels, 42))
    baseline = json.loads((tmp_path / 'new' / 'prior_baseline.json').read_text())
    assert baseline['method'] == 'source_only'
    assert baseline['source_prior']['kind'] == 'isotropic'
    assert baseline['used_for_selection'] is False


def test_correlated_training_is_train_only_deterministic_and_rng_isolated(tmp_path):
    training = make_data(tmp_path / 'data')
    config = {'steps': 4, 'batch_size': 8, 'validation_every': 1,
        'validation_per_condition': 3, 'solver_steps': 2, 'source_prior': 'training_correlation'}
    state = torch.get_rng_state().clone()
    a = flow.train_probe(tmp_path / 'data', tmp_path / 'a', config)
    b = flow.train_probe(tmp_path / 'data', tmp_path / 'b', config | {'validation_every': 2})
    assert torch.equal(state, torch.get_rng_state())
    ca = torch.load(tmp_path / 'a' / 'last.pt', weights_only=True)
    cb = torch.load(tmp_path / 'b' / 'last.pt', weights_only=True)
    for key in ('velocity_model', 'optimizer', 'noise_rng_state', 'time_rng_state',
                'sampler_rng_state', 'sampler_order'):
        same(ca[key], cb[key])
    expected = flow.fit_source_prior(training, 'training_correlation')
    assert a['source_prior']['correlation'] == expected['correlation']
    assert a['source_prior']['fit']['training_npz_sha256'] == a['data_hashes']['train.npz']
    assert 'test.npz' not in a['data_hashes']
    labels = np.array([0, 1, 0], dtype=np.int64)
    samples = flow.generate(tmp_path / 'a' / 'best.pt', labels, 123)
    np.testing.assert_array_equal(samples, flow.generate(tmp_path / 'a' / 'best.pt', labels, 123))
    assert torch.equal(state, torch.get_rng_state())
    sensitivity = json.loads((tmp_path / 'a' / 'solver_sensitivity.json').read_text())
    assert sensitivity['selection_solver']['steps'] == 2
    assert sensitivity['sensitivity_solver']['steps'] == 4
    assert sensitivity['selection_solver']['metrics'] == json.loads((tmp_path / 'a' / 'best_validation.json').read_text())
    baseline = json.loads((tmp_path / 'a' / 'prior_baseline.json').read_text())
    assert baseline['source_prior'] == a['source_prior']
    assert baseline['generation_seed'] == a['selection']['generation_seed']
    assert baseline['conditions'] == a['selection']['conditions']


def test_correlated_checkpoint_requires_valid_fitted_factor(tmp_path):
    make_data(tmp_path / 'data')
    flow.train_probe(tmp_path / 'data', tmp_path / 'run', {'steps': 1, 'solver_steps': 1,
        'validation_per_condition': 2, 'source_prior': 'training_correlation'})
    state = torch.load(tmp_path / 'run' / 'best.pt', weights_only=True)
    missing = dict(state)
    missing.pop('source_prior')
    torch.save(missing, tmp_path / 'missing.pt')
    with pytest.raises(ValueError, match='source prior'):
        flow.generate(tmp_path / 'missing.pt', np.array([0]), 4)
    state['source_prior']['cholesky'][0][0] *= 2
    torch.save(state, tmp_path / 'bad.pt')
    with pytest.raises(ValueError, match='source prior'):
        flow.generate(tmp_path / 'bad.pt', np.array([0]), 4)
