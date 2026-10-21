"""Reflected trajectories, inverse sampling and TRAIN-only reproducibility."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def module():
    from fluvius_cgan import reflection
    return reflection


def test_multiple_reflections_and_collision_right_derivatives():
    reflection = module()
    y = torch.tensor([[-4., -3., -2.2, -.2, 0., .2, 1., 1.2, 2., 2.2, 5.]], dtype=torch.float64)
    expected = torch.tensor([[0., 1., .2, .2, 0., .2, 1., .8, 0., .2, 1.]], dtype=torch.float64)
    torch.testing.assert_close(reflection.reflect(y), expected)
    velocities = torch.tensor([[-2., -2., 2., 2., -2., -2., -2., -2., 2., -2., 2.]], dtype=torch.float64)
    position, derivative = reflection.reflected_bridge(y, velocities, torch.zeros((1, 1), dtype=torch.float64))
    torch.testing.assert_close(position, expected)
    right = (reflection.reflect(y + 1e-7 * velocities) - reflection.reflect(y)) / 1e-7
    torch.testing.assert_close(derivative, right, atol=1e-8, rtol=1e-8)
    assert derivative[0, 0] > 0 and derivative[0, 1] < 0


def test_reflected_velocity_matches_finite_difference_away_from_collisions():
    reflection = module()
    u = torch.tensor([[.1, .7], [.4, .8]], dtype=torch.float64)
    v = torch.tensor([[-3.2, 4.1], [8.7, -6.4]], dtype=torch.float64)
    t = torch.tensor([[.33], [1.13]], dtype=torch.float64)
    position, velocity = reflection.reflected_bridge(u, v, t)
    central = (reflection.reflect(u + (t + 1e-7) * v)
               - reflection.reflect(u + (t - 1e-7) * v)) / 2e-7
    torch.testing.assert_close(velocity, central, rtol=1e-8, atol=1e-8)
    assert torch.all((position >= 0) & (position <= 1))


def test_zero_velocity_preserves_uniform_initial_values_and_counts_nfe():
    reflection = module()
    initial = torch.tensor([[.1, .7], [.4, .8]], dtype=torch.float64)
    class Zero:
        calls = 0
        def __call__(self, x, t, labels):
            self.calls += 1
            return torch.zeros_like(x)
    model = Zero()
    final, diagnostic = reflection.reflected_euler(model, initial, torch.tensor([0, 8]), 64)
    torch.testing.assert_close(final, initial, rtol=0, atol=0)
    assert model.calls == diagnostic['nfe'] == 64
    assert diagnostic['reflection_coordinate_counts'] == [0] * 64
    assert diagnostic['time_grid'][0] == 1.5 and diagnostic['time_grid'][-1] == 0


def test_reverse_solver_recovers_analytic_nonlinear_copula_dependence():
    # Wrong time direction or a second minus sign destroys this nonzero moment.
    reflection = module()
    rng = np.random.default_rng(771)
    initial = torch.from_numpy(rng.random((32768, 2)))
    labels = torch.zeros(len(initial), dtype=torch.int64)
    class Oracle:
        def __call__(self, x, t, labels):
            amplitude = .8 * torch.exp(-torch.pi ** 2 * t ** 2)
            cosine, sine = torch.cos(torch.pi * x), torch.sin(torch.pi * x)
            denominator = 1 + amplitude * cosine[:, :1] * cosine[:, 1:]
            gradient = -torch.pi * amplitude * torch.stack(
                (sine[:, 0] * cosine[:, 1], cosine[:, 0] * sine[:, 1]), dim=1) / denominator
            return -t * gradient
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        coarse, _ = reflection.reflected_euler(Oracle(), initial, labels, 64)
        fine, _ = reflection.reflected_euler(Oracle(), initial, labels, 128)
    finally:
        torch.set_num_threads(previous)
    for sample in (coarse, fine):
        moment = (torch.cos(torch.pi * sample[:, 0]) * torch.cos(torch.pi * sample[:, 1])).mean().item()
        assert abs(moment - .2) < .012
        assert torch.max(torch.abs(sample.mean(dim=0) - .5)) < .006
        assert torch.max(torch.abs(sample.var(dim=0, unbiased=False) - 1 / 12)) < .003
    assert torch.sqrt(torch.mean((coarse - fine) ** 2)) < .002


def test_pooled_conditioning_changes_only_velocity_labels():
    reflection = module()
    class Conditional:
        def __call__(self, x, t, labels):
            return labels[:, None].to(x) * torch.ones_like(x) / 100
    initial = torch.full((2, 2), .5)
    labels = torch.tensor([0, 8])
    conditional, _ = reflection.reflected_euler(Conditional(), initial, labels, 8)
    pooled, _ = reflection.reflected_euler(Conditional(), initial, labels, 8, condition_mode='pooled')
    assert conditional[1, 0] < conditional[0, 0]
    torch.testing.assert_close(pooled, initial)


def test_solver_recovers_opposite_nonlinear_dependence_with_zero_covariance():
    # Frequency two has zero population linear covariance despite dependence.
    reflection = module()
    rng = np.random.default_rng(880)
    initial = torch.from_numpy(np.tile(rng.random((16384, 2)), (2, 1)))
    labels = torch.cat((torch.zeros(16384, dtype=torch.int64), torch.full((16384,), 8, dtype=torch.int64)))
    class Oracle:
        def __call__(self, x, t, condition):
            sign = torch.where(condition[:, None] == 0, 1., -1.).to(x)
            amplitude = sign * .8 * torch.exp(-4 * torch.pi ** 2 * t ** 2)
            cosine, sine = torch.cos(2 * torch.pi * x), torch.sin(2 * torch.pi * x)
            denominator = 1 + amplitude * cosine[:, :1] * cosine[:, 1:]
            gradient = -2 * torch.pi * amplitude * torch.stack(
                (sine[:, 0] * cosine[:, 1], cosine[:, 0] * sine[:, 1]), dim=1) / denominator
            return -t * gradient
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        sample, _ = reflection.reflected_euler(Oracle(), initial, labels, 128)
    finally:
        torch.set_num_threads(previous)
    for group, expected in zip(torch.split(sample, 16384), (.2, -.2)):
        moment = (torch.cos(2 * torch.pi * group[:, 0]) * torch.cos(2 * torch.pi * group[:, 1])).mean().item()
        assert abs(moment - expected) < .015
        assert abs(((group[:, 0] - .5) * (group[:, 1] - .5)).mean().item()) < .003
        assert torch.max(torch.abs(group.mean(dim=0) - .5)) < .008


def data(root):
    root.mkdir()
    rng = np.random.default_rng(20)
    training = rng.uniform(-.5, 1., (25, 24)).astype(np.float32)
    training[:, 0] = 0
    training[:12, 1] = 0
    for name, values in [('train', training), ('val', rng.uniform(8., 10., (20, 24)).astype(np.float32))]:
        np.savez(root / f'{name}.npz', x=values, y=np.arange(len(values), dtype=np.int64) % 2,
                 meter_id=np.array([f'{name}_{i}' for i in range(len(values))]),
                 date=np.full(len(values), '2024-01-01'))
    (root / 'scaler.json').write_text(json.dumps({'minimum': float(training.min()),
        'maximum': float(training.max()), 'fitted_on': 'train'}))
    (root / 'metadata.json').write_text(json.dumps({'resolution': 24, 'condition_dim': 64,
        'development': {'source_partition_role': 'original_train'}}))
    (root / 'test.npz').write_bytes(b'TEST MUST NEVER BE OPENED')
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


def test_training_is_train_only_deterministic_and_independent_of_validation_cadence(tmp_path):
    reflection = module()
    training = data(tmp_path / 'data')
    config = {'steps': 4, 'batch_size': 7, 'seed': 101, 'validation_every': 1,
              'validation_per_condition': 4, 'solver_steps': 2}
    original_rng = torch.get_rng_state().clone()
    first = reflection.train_probe(tmp_path / 'data', tmp_path / 'first', config)
    second = reflection.train_probe(tmp_path / 'data', tmp_path / 'second', config | {'validation_every': 2})
    assert torch.equal(original_rng, torch.get_rng_state())
    assert first['status'] == second['status'] == 'completed'
    assert first['row_presentations'] == 25
    assert first['architecture']['trainable_parameters'] == 23896
    assert first['marginals']['training_npz_sha256'] == first['data_hashes']['train.npz']
    assert first['test_performance_access'] is False
    for name, expected in first['artifact_hashes'].items():
        assert hashlib.sha256((tmp_path / 'first' / name).read_bytes()).hexdigest() == expected
    a = torch.load(tmp_path / 'first' / 'last.pt', weights_only=True)
    b = torch.load(tmp_path / 'second' / 'last.pt', weights_only=True)
    for name in ('velocity_model', 'optimizer', 'model_rng_state', 'noise_rng_state',
                 'time_rng_state', 'sampler_rng_state', 'jitter_rng_state', 'sampler_order'):
        same(a[name], b[name])
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    checkpoint = tmp_path / 'first' / 'best.pt'
    generated = reflection.generate(checkpoint, labels, 300)
    np.testing.assert_array_equal(generated, reflection.generate(checkpoint, labels, 300))
    probabilities, diagnostic = reflection.generate_probabilities(checkpoint, labels, 300)
    assert probabilities.shape == generated.shape == (4, 24)
    assert diagnostic['solver']['nfe'] == 2
    assert diagnostic['probabilities']['per_condition']['0']['profiles'] == 2
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    for i, label in enumerate(labels):
        for hour in range(24):
            assert generated[i, hour] in training[np.arange(25) % 2 == label, hour]
    assert torch.equal(original_rng, torch.get_rng_state())
    sensitivity = json.loads((tmp_path / 'first' / 'solver_sensitivity.json').read_text())
    assert sensitivity['selection_solver']['nfe'] == 2
    assert sensitivity['sensitivity_solver']['nfe'] == 4
    assert sensitivity['selection_solver']['metrics'] == json.loads((tmp_path / 'first' / 'best_validation.json').read_text())
    assert sensitivity['used_for_selection'] is False
    assert json.loads((tmp_path / 'first' / 'independent_marginals.json').read_text())['used_for_selection'] is False
    with pytest.raises(FileExistsError):
        reflection.train_probe(tmp_path / 'data', tmp_path / 'first', config)
    with (tmp_path / 'first' / 'marginals.npz').open('ab') as handle:
        handle.write(b'tamper')
    with pytest.raises(ValueError, match='SHA|hash'):
        reflection.generate(checkpoint, labels, 300)


def test_failed_training_binds_partial_artifacts_and_restores_runtime(tmp_path, monkeypatch):
    reflection = module()
    data(tmp_path / 'data')
    original = reflection.Velocity.forward
    def nonfinite(model, x, t, labels):
        return original(model, x, t, labels) * float('nan')
    monkeypatch.setattr(reflection.Velocity, 'forward', nonfinite)
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    output = tmp_path / 'failed'
    with pytest.raises(FloatingPointError, match='Nonfinite reflection regression loss'):
        reflection.train_probe(tmp_path / 'data', output, {'steps': 1})
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['artifact_hashes']['marginals.npz'] == hashlib.sha256((output / 'marginals.npz').read_bytes()).hexdigest()
    assert torch.equal(rng, torch.get_rng_state()) and torch.get_num_threads() == threads


@pytest.mark.parametrize('mutation', ['training_hash', 'unsafe_path', 'summary'])
def test_checkpoint_marginal_provenance_is_checked_before_generation(tmp_path, mutation):
    reflection = module()
    data(tmp_path / 'data')
    run = tmp_path / 'run'
    reflection.train_probe(tmp_path / 'data', run, {'steps': 1, 'validation_per_condition': 2, 'solver_steps': 1})
    state = torch.load(run / 'best.pt', weights_only=True)
    if mutation == 'training_hash':
        state['marginals']['training_npz_sha256'] = '0' * 64
    elif mutation == 'unsafe_path':
        state['marginals']['artifact'] = '../marginals.npz'
    else:
        state['marginals']['summary']['profile_dim'] = 23
    torch.save(state, run / 'best.pt')
    with pytest.raises(ValueError, match='provenance|summary'):
        reflection.generate(run / 'best.pt', np.array([0, 1]), 300)


@pytest.mark.parametrize('bad', [{'device': 'cuda'}, {'num_threads': 2}, {'learning_rate': float('nan')},
    {'solver_steps': 0}, {'unknown': 1}, {'seed': -1}, {'max_time': 0},
    {'gradient_clip_norm': 0}, {'condition_mode': 'ignored'}, {'steps': True}])
def test_invalid_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        module()._config(bad)


@pytest.mark.parametrize('bad', [torch.tensor([[float('nan'), .1]]), torch.tensor([[-.1, .1]]),
                               torch.tensor([[1.1, .1]]), torch.tensor([[1, 0]])])
def test_solver_rejects_nonprobability_initial_states(bad):
    with pytest.raises(ValueError):
        module().reflected_euler(lambda x, t, labels: torch.zeros_like(x), bad, torch.tensor([0]), 1)
