import hashlib
import json

import numpy as np
import pytest
import torch

from fluvius_cgan.training import generate, train


def prepared(tmp_path):
    directory = tmp_path / 'data'
    directory.mkdir()
    rng = np.random.default_rng(91)
    for split, n in [('train', 18), ('val', 10)]:
        x = rng.uniform(-2, 4, (n, 24)).astype(np.float32)
        np.savez(directory / f'{split}.npz', x=x, y=np.arange(n, dtype=np.int64) % 2,
                 meter_id=np.array([f'{split}_{i}' for i in range(n)]),
                 date=np.full(n, '2024-01-01'))
    # The trainer must never read or hash the test split.
    (directory / 'test.npz').write_bytes(b'heldout forbidden')
    with np.load(directory / 'train.npz') as values:
        scaler = {'minimum': float(values['x'].min()), 'maximum': float(values['x'].max()), 'fitted_on': 'train'}
    (directory / 'scaler.json').write_text(json.dumps(scaler))
    (directory / 'metadata.json').write_text(json.dumps({'resolution': 24, 'condition_dim': 64}))
    return directory


def config(epochs=2, **kwargs):
    return dict(epochs=epochs, batch_size=7, seed=11, num_threads=1,
                validation_per_condition=4, validation_every=1, **kwargs)


def checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def test_reference_optimizers_artifacts_fixed_selection_and_no_test_access(tmp_path):
    data = prepared(tmp_path)
    run = tmp_path / 'run'
    train(data, run, config())
    saved = checkpoint(run / 'last.pt')
    for name, lr in [('optimizer_g', 1e-4), ('optimizer_d', 2e-4)]:
        assert saved[name]['param_groups'][0]['lr'] == lr
        assert saved[name]['param_groups'][0]['betas'] == (.9, .999)
        assert all(int(state['step']) == 6 for state in saved[name]['state'].values())
    history = json.loads((run / 'history.json').read_text())
    assert [row['epoch'] for row in history] == [1, 2]
    assert all(np.isfinite(row['loss_g']) and np.isfinite(row['loss_d']) and np.isfinite(row['validation_sw1']) for row in history)
    assert all(row['seconds'] >= 0 for row in history)
    manifest = json.loads((run / 'manifest.json').read_text())
    assert set(manifest['data_hashes']) == {'train.npz', 'val.npz', 'scaler.json', 'metadata.json'}
    assert manifest['data_hashes']['train.npz'] == hashlib.sha256((data / 'train.npz').read_bytes()).hexdigest()
    assert manifest['selection']['conditions'] == [0, 1]
    assert manifest['selection']['noise_seed'] != 11
    assert manifest['nondefault_parameters']['batch_size'] == 7
    assert (run / 'best.pt').exists()


def test_cpu_epoch_boundary_resume_matches_uninterrupted(tmp_path):
    data = prepared(tmp_path)
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    train(data, full, config(3))
    train(data, resumed, config(1))
    first_history = json.loads((resumed / 'history.json').read_text())[0]
    train(data, resumed, config(3), resume=True)
    a, b = checkpoint(full / 'last.pt'), checkpoint(resumed / 'last.pt')
    for name in ['generator', 'discriminator']:
        for key in a[name]:
            assert torch.equal(a[name][key], b[name][key]), (name, key)
    for left, right in zip(a['history'], b['history']):
        for field in ['epoch', 'loss_g', 'loss_d', 'validation_sw1']:
            assert left[field] == right[field]
    assert b['history'][0] == first_history
    assert a['best_epoch'] == b['best_epoch']
    torch.testing.assert_close(a['noise_rng_state'], b['noise_rng_state'], rtol=0, atol=0)


def test_generation_is_physical_reproducible_batched_and_rng_isolated(tmp_path):
    data = prepared(tmp_path)
    run = tmp_path / 'run'
    train(data, run, config(1))
    saved = checkpoint(run / 'last.pt')
    from fluvius_cgan.models import Generator
    model = Generator()
    model.load_state_dict(saved['generator'])
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    state = torch.get_rng_state().clone()
    output = generate(run / 'last.pt', labels, seed=177, batch_size=2)
    assert torch.equal(state, torch.get_rng_state())
    assert output.shape == (11, 24) and np.isfinite(output).all()
    np.testing.assert_array_equal(output, generate(run / 'last.pt', labels, seed=177, batch_size=2))
    for batch_size in [1, 3, 5, 11]:
        np.testing.assert_allclose(output, generate(run / 'last.pt', labels, seed=177, batch_size=batch_size), rtol=1e-6, atol=1e-6)
    assert not np.array_equal(output, generate(run / 'last.pt', labels, seed=178, batch_size=2))
    zero = {key: torch.zeros_like(value) for key, value in saved['generator'].items()}
    saved['generator'] = zero
    torch.save(saved, run / 'zero.pt')
    assert np.all(generate(run / 'zero.pt', labels, seed=1) == saved['scaler']['minimum'])


def test_overwrite_and_incompatible_resume_fail_without_history_changes(tmp_path):
    data = prepared(tmp_path)
    run = tmp_path / 'run'
    train(data, run, config(1))
    before = (run / 'history.json').read_bytes()
    with pytest.raises(FileExistsError):
        train(data, run, config(2))
    altered = config(2)
    altered['batch_size'] = 8
    with pytest.raises(ValueError, match='config'):
        train(data, run, altered, resume=True)
    (data / 'metadata.json').write_text('{}')
    with pytest.raises(ValueError, match='data|hash'):
        train(data, run, config(2), resume=True)
    assert (run / 'history.json').read_bytes() == before


def test_scaler_fitted_on_training_only_and_support_validation(tmp_path):
    data = prepared(tmp_path)
    scaler = json.loads((data / 'scaler.json').read_text())
    scaler['maximum'] = 10000
    (data / 'scaler.json').write_text(json.dumps(scaler))
    with pytest.raises(ValueError, match='scaler|train'):
        train(data, tmp_path / 'bad', config(1))
    with np.load(data / 'train.npz') as values:
        scaler['maximum'] = float(values['x'].max())
    (data / 'scaler.json').write_text(json.dumps(scaler))
    with np.load(data / 'val.npz') as values:
        arrays = dict(values)
    arrays['y'][:] = 0
    np.savez(data / 'val.npz', **arrays)
    with pytest.raises(ValueError, match='support|condition'):
        train(data, tmp_path / 'unsupported', config(1))


@pytest.mark.parametrize('field,value', [('epochs', 0), ('batch_size', 0), ('validation_per_condition', 1025), ('latent_dim', 25), ('surprise', True)])
def test_invalid_config_rejected(tmp_path, field, value):
    data = prepared(tmp_path)
    params = config(1)
    params[field] = value
    with pytest.raises(ValueError):
        train(data, tmp_path / 'bad', params)


def test_unconditional_comparator_keeps_conditional_selection(tmp_path):
    data = prepared(tmp_path)
    train(data, tmp_path / 'unconditional', config(1, unconditional=True))
    saved = checkpoint(tmp_path / 'unconditional/last.pt')
    assert saved['generator']['model.0.weight'].shape == (64, 24)
    assert saved['selection']['conditions'] == [0, 1]
    assert generate(tmp_path / 'unconditional/last.pt', [0, 63], seed=42).shape == (2, 24)


def test_sparse_validation_schedule_resume_and_best_epoch(tmp_path):
    data = prepared(tmp_path)
    params = config(6)
    params['validation_every'] = 5
    train(data, tmp_path / 'full', params)
    short = {**params, 'epochs': 3}
    train(data, tmp_path / 'resumed', short)
    train(data, tmp_path / 'resumed', params, resume=True)
    full, resumed = checkpoint(tmp_path / 'full/last.pt'), checkpoint(tmp_path / 'resumed/last.pt')
    assert [row['epoch'] for row in full['history'] if row['validation_sw1'] is not None] == [1, 5]
    for key in full['generator']:
        assert torch.equal(full['generator'][key], resumed['generator'][key])
    expected_best = min((row for row in full['history'] if row['validation_sw1'] is not None), key=lambda row: row['validation_sw1'])
    assert checkpoint(tmp_path / 'resumed/best.pt')['epoch'] == expected_best['epoch']


def test_history_tampering_and_declared_data_hashes_rejected(tmp_path):
    data = prepared(tmp_path)
    run = tmp_path / 'run'
    train(data, run, config(1))
    (run / 'history.json').write_text('[]')
    with pytest.raises(ValueError, match='history'):
        train(data, run, config(2), resume=True)
    (data / 'metadata.json').write_text(json.dumps({'resolution': 24, 'condition_dim': 64, 'splits': {'train': {'sha256': '0' * 64}}}))
    with pytest.raises(ValueError, match='hash'):
        train(data, tmp_path / 'bad_hash', config(1))


def test_constant_train_range_and_empty_generation_are_finite(tmp_path):
    data = prepared(tmp_path)
    with np.load(data / 'train.npz') as archive:
        values = dict(archive)
    values['x'][:] = -2
    np.savez(data / 'train.npz', **values)
    (data / 'scaler.json').write_text(json.dumps({'minimum': -2., 'maximum': -2., 'fitted_on': 'train'}))
    train(data, tmp_path / 'run', config(1))
    empty = generate(tmp_path / 'run/last.pt', np.empty(0, dtype=np.int64), seed=1)
    assert empty.shape == (0, 24)
    assert np.isfinite(generate(tmp_path / 'run/last.pt', [0, 1], seed=1)).all()


def test_selection_is_shared_across_training_seeds_and_comparator_arms(tmp_path):
    from fluvius_cgan.evaluation import projection_directions
    data = prepared(tmp_path)
    train(data, tmp_path / 'first', config(1))
    params = config(1, unconditional=True)
    params['seed'] = 22
    train(data, tmp_path / 'second', params)
    first = checkpoint(tmp_path / 'first/last.pt')['selection']
    second = checkpoint(tmp_path / 'second/last.pt')['selection']
    assert first == second
    assert first['index_seed'] == 2027 and first['noise_seed'] == 2028
    assert first['directions_sha256'] == hashlib.sha256(projection_directions(24).astype('<f8').tobytes()).hexdigest()
