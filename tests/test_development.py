import json

import numpy as np
import pytest
import torch

from fluvius_cgan.development import fit_transform, transform, physical_metrics, train_probe, generate


@pytest.mark.parametrize('kind', ['minmax', 'standard', 'hourly_standard'])
def test_scaling_is_invertible_without_clipping_or_validation_refit(kind):
    x = np.arange(120, dtype=np.float64).reshape(5, 24) - 30
    spec = fit_transform(x, kind)
    heldout = x * 100 - 1000
    encoded = transform(heldout, spec)
    np.testing.assert_allclose(transform(encoded, spec, inverse=True), heldout, atol=1e-10)
    assert spec['fitted_on'] == 'train'
    if kind == 'standard':
        assert spec['offset'] == [float(x.mean())] * 24
        assert spec['scale'] == [float(x.std())] * 24


def test_constant_coordinates_remain_finite_and_physical_metric_is_common():
    x = np.full((4, 24), -3.)
    spec = fit_transform(x, 'hourly_standard')
    np.testing.assert_array_equal(transform(x, spec), 0)
    np.testing.assert_array_equal(transform(transform(x, spec), spec, inverse=True), x)
    y = np.array([0, 0, 1, 1])
    score = physical_metrics(x, x + 2., y, training_range=10.)
    assert score['macro']['daily_energy_w1_kwh'] == pytest.approx(48.)
    assert score['macro']['normalized_profile_sw1'] == pytest.approx(score['macro']['profile_sw1_kwh'] / 10.)


def test_probe_reproducibility_checkpoint_generation_and_test_nonaccess(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    rng = np.random.default_rng(12)
    for split in ['train', 'val']:
        x = rng.normal(1., .2, (20, 24)).astype(np.float32)
        np.savez(data / f'{split}.npz', x=x, y=np.arange(20, dtype=np.int64) % 2,
                 meter_id=np.array([f'{split}{i}' for i in range(20)]), date=np.full(20, '2024-01-01'))
    with np.load(data / 'train.npz') as values:
        scaler = {'minimum': float(values['x'].min()), 'maximum': float(values['x'].max()), 'fitted_on': 'train'}
    (data / 'scaler.json').write_text(json.dumps(scaler))
    (data / 'metadata.json').write_text(json.dumps({'resolution': 24, 'condition_dim': 64}))
    (data / 'test.npz').write_bytes(b'FORBIDDEN')
    config = {'steps': 2, 'batch_size': 7, 'validation_every': 1, 'validation_per_condition': 4,
              'normalization': 'standard', 'seed': 101}
    a, b = tmp_path / 'a', tmp_path / 'b'
    first, second = train_probe(data, a, config), train_probe(data, b, config)
    assert first['status'] == second['status'] == 'completed'
    assert first['row_presentations'] == 14
    assert first['completed_steps'] == 2
    assert 'test.npz' not in first['data_hashes']
    ca = torch.load(a / 'last.pt', weights_only=True)
    cb = torch.load(b / 'last.pt', weights_only=True)
    for key in ca['generator']:
        assert torch.equal(ca['generator'][key], cb['generator'][key])
    labels = np.array([0, 1, 0], dtype=np.int64)
    state = torch.get_rng_state().clone()
    values = generate(a / 'best.pt', labels, 300)
    assert torch.equal(state, torch.get_rng_state())
    np.testing.assert_array_equal(values, generate(a / 'best.pt', labels, 300))
    assert np.isfinite(values).all() and values.shape == (3, 24)
    with pytest.raises(FileExistsError):
        train_probe(data, a, config)
