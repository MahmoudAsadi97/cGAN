"""Conditional flow objective, integration, provenance and reproducibility tests."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def _module():
    from fluvius_cgan import flow
    return flow


def test_heun_recovers_analytic_independent_gaussian_bridge_with_second_order_accuracy():
    flow = _module()
    z = torch.linspace(-2., 2., 120, dtype=torch.float64).reshape(5, 24)
    labels = torch.arange(5)
    mu = torch.linspace(-1., 1., 24, dtype=torch.float64)
    sigma = 1.7

    class GaussianVelocity:
        def __init__(self):
            self.calls = 0

        def __call__(self, x, t, labels):
            self.calls += 1
            variance = (1 - t) ** 2 + (t * sigma) ** 2
            return mu + (t * sigma ** 2 - (1 - t)) / variance * (x - t * mu)

    coarse, fine = GaussianVelocity(), GaussianVelocity()
    a = flow.heun(coarse, z, labels, 16)
    b = flow.heun(fine, z, labels, 32)
    truth = mu + sigma * z
    assert coarse.calls == 32 and fine.calls == 64
    assert torch.max(torch.abs(b - truth)) < .002
    assert torch.linalg.vector_norm(b - truth) < .3 * torch.linalg.vector_norm(a - truth)
    torch.testing.assert_close(z, torch.linspace(-2., 2., 120, dtype=torch.float64).reshape(5, 24))


def test_linear_bridge_endpoints_and_velocity():
    flow = _module()
    x = torch.arange(72, dtype=torch.float32).reshape(3, 24) / 10
    z = -x
    t = torch.tensor([[0.], [.5], [1.]])
    position, velocity = flow.linear_bridge(x, z, t)
    torch.testing.assert_close(position[0], z[0])
    torch.testing.assert_close(position[1], torch.zeros(24))
    torch.testing.assert_close(position[2], x[2])
    torch.testing.assert_close(velocity, x - z)


def _data(root):
    root.mkdir()
    rng = np.random.default_rng(20)
    training = rng.normal(.5, .8, (25, 24)).astype(np.float32)
    for name, x in [('train', training), ('val', rng.normal(8., 2., (20, 24)).astype(np.float32))]:
        np.savez(root / f'{name}.npz', x=x, y=np.arange(len(x), dtype=np.int64) % 2,
                 meter_id=np.array([f'{name}_{i}' for i in range(len(x))]),
                 date=np.full(len(x), '2024-01-01'))
    (root / 'scaler.json').write_text(json.dumps({'minimum': float(training.min()),
        'maximum': float(training.max()), 'fitted_on': 'train'}))
    (root / 'metadata.json').write_text(json.dumps({'resolution': 24, 'condition_dim': 64,
        'development': {'source_partition_role': 'original_train'}}))
    (root / 'test.npz').write_bytes(b'TEST MUST NEVER BE OPENED')
    return training


def _same(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _same(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for first, second in zip(a, b):
            _same(first, second)
    else:
        assert a == b


def test_training_is_deterministic_and_selection_does_not_change_trajectory(tmp_path):
    flow = _module()
    training = _data(tmp_path / 'data')
    c = {'steps': 4, 'batch_size': 7, 'seed': 101, 'validation_every': 1,
         'validation_per_condition': 4, 'solver_steps': 2}
    global_rng = torch.get_rng_state().clone()
    first = flow.train_probe(tmp_path / 'data', tmp_path / 'first', c)
    second = flow.train_probe(tmp_path / 'data', tmp_path / 'second', c | {'validation_every': 2})
    assert torch.equal(global_rng, torch.get_rng_state())
    assert first['status'] == second['status'] == 'completed'
    assert first['row_presentations'] == 25
    assert first['architecture']['trainable_parameters'] == 23896
    assert first['representation']['offset'] == [float(training.astype(np.float64).mean())] * 24
    assert first['representation']['scale'] == [float(training.astype(np.float64).std())] * 24
    assert set(first['data_hashes']) == {'train.npz', 'val.npz', 'scaler.json', 'metadata.json'}
    a = torch.load(tmp_path / 'first' / 'last.pt', weights_only=True)
    b = torch.load(tmp_path / 'second' / 'last.pt', weights_only=True)
    for name in ('velocity_model', 'optimizer', 'model_rng_state', 'noise_rng_state',
                 'time_rng_state', 'sampler_rng_state', 'sampler_order'):
        _same(a[name], b[name])
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    generated = flow.generate(tmp_path / 'first' / 'best.pt', labels, 300)
    np.testing.assert_array_equal(generated, flow.generate(tmp_path / 'first' / 'best.pt', labels, 300))
    assert torch.equal(global_rng, torch.get_rng_state())
    assert generated.shape == (4, 24) and np.isfinite(generated).all()
    sensitivity = json.loads((tmp_path / 'first' / 'solver_sensitivity.json').read_text())
    assert sensitivity['selected_checkpoint_step'] == first['best_step']
    assert sensitivity['selection_solver']['steps'] == 2
    assert sensitivity['sensitivity_solver']['steps'] == 4
    assert sensitivity['selection_solver']['nfe'] == 4
    assert sensitivity['sensitivity_solver']['nfe'] == 8
    assert sensitivity['used_for_selection'] is False
    assert sensitivity['selection_solver']['metrics'] == json.loads(
        (tmp_path / 'first' / 'best_validation.json').read_text())
    with pytest.raises(FileExistsError):
        flow.train_probe(tmp_path / 'data', tmp_path / 'first', c)


def test_generation_inverts_scale_without_clipping(tmp_path):
    flow = _module()
    with torch.random.fork_rng(devices=[]):
        model = flow.Velocity()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
    representation = {'offset': [10.] * 24, 'scale': [3.] * 24}
    checkpoint = tmp_path / 'identity.pt'
    torch.save({'schema_version': flow.SCHEMA, 'velocity_model': model.state_dict(),
        'representation': representation, 'config': flow._config({})}, checkpoint)
    labels = np.array([0, 63], dtype=np.int64)
    got = flow.generate(checkpoint, labels, 700, solver_steps=3)
    expected = np.random.default_rng(700).standard_normal((2, 24), dtype=np.float32).astype(np.float64) * 3 + 10
    np.testing.assert_array_equal(got, expected)


def test_velocity_rejects_misaligned_time_and_invalid_condition_labels():
    flow = _module()
    with torch.random.fork_rng(devices=[]):
        model = flow.Velocity()
    x, t = torch.zeros(2, 24), torch.full((2, 1), .5)
    labels = torch.tensor([0, 63], dtype=torch.int64)
    assert model(x, t, labels).shape == x.shape
    with pytest.raises(ValueError, match='time'):
        model(x, t[:, 0], labels)
    for bad in (torch.tensor([0, 64]), torch.tensor([-1, 63]),
                torch.tensor([0., 63.]), torch.tensor([0])):
        with pytest.raises(ValueError):
            model(x, t, bad)


def test_training_failure_is_recorded_and_runtime_state_is_restored(tmp_path, monkeypatch):
    flow = _module()
    _data(tmp_path / 'data')
    original_forward = flow.Velocity.forward
    def nonfinite_forward(model, x, t, labels):
        return original_forward(model, x, t, labels) * float('nan')
    monkeypatch.setattr(flow.Velocity, 'forward', nonfinite_forward)
    global_rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    output = tmp_path / 'failed'
    with pytest.raises(FloatingPointError, match='Nonfinite flow regression loss'):
        flow.train_probe(tmp_path / 'data', output, {'steps': 1})
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['error'] == 'FloatingPointError: Nonfinite flow regression loss'
    assert manifest['test_performance_access'] is False
    assert not (output / 'best.pt').exists()
    assert torch.equal(global_rng, torch.get_rng_state())
    assert torch.get_num_threads() == threads
    with pytest.raises(FileExistsError):
        flow.train_probe(tmp_path / 'data', output, {'steps': 1})


@pytest.mark.parametrize('bad', [{'device': 'cuda'}, {'num_threads': 2}, {'learning_rate': float('nan')},
                               {'solver_steps': 0}, {'unknown': 1}, {'seed': -1}])
def test_invalid_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        _module()._config(bad)
