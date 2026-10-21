"""Tests for exact continuation and explicit device behavior."""
import json

import numpy as np
import pytest
import torch


from fluvius_cgan import development


@pytest.fixture
def data(tmp_path):
    path = tmp_path / 'data'
    path.mkdir()
    rng = np.random.default_rng(12)
    for split in ['train', 'val']:
        x = rng.normal(1., .2, (20, 24)).astype(np.float32)
        np.savez(path / f'{split}.npz', x=x, y=np.arange(20, dtype=np.int64) % 2,
                 meter_id=np.array([f'{split}{i}' for i in range(20)]), date=np.full(20, '2024-01-01'))
    with np.load(path / 'train.npz') as values:
        scaler = {'minimum': float(values['x'].min()), 'maximum': float(values['x'].max()), 'fitted_on': 'train'}
    (path / 'scaler.json').write_text(json.dumps(scaler))
    (path / 'metadata.json').write_text(json.dumps({'resolution': 24, 'condition_dim': 64}))
    (path / 'test.npz').write_bytes(b'FORBIDDEN')
    return path


CONFIG = {'steps': 6, 'batch_size': 7, 'validation_every': 1, 'validation_per_condition': 4,
          'normalization': 'standard', 'seed': 101}


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for first, second in zip(a, b):
            assert_tree_equal(first, second)
    else:
        assert a == b


def test_resume_matches_uninterrupted_cpu_weights_optimizer_rng_selection(data, tmp_path):
    full, partial, resumed = [tmp_path / name for name in ('full', 'partial', 'resumed')]
    development.train_probe(data, full, CONFIG)
    development.train_probe(data, partial, CONFIG | {'steps': 3})
    original_hashes = {p.name: development._sha256(p) for p in partial.iterdir() if p.is_file()}
    manifest = development.train_probe(data, resumed, CONFIG, resume=partial / 'last.pt')
    a, b = [torch.load(path / 'last.pt', weights_only=True) for path in (full, resumed)]
    for key in ('generator', 'discriminator', 'optimizer_g', 'optimizer_d', 'model_rng_state',
                'noise_rng_state', 'sampler_rng_state', 'sampler_order', 'sampler_cursor',
                'passes_started', 'row_presentations', 'best_step', 'best_score'):
        assert_tree_equal(a[key], b[key])
    assert_tree_equal([{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in a['history']],
                      [{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in b['history']])
    assert_tree_equal(torch.load(full / 'best.pt', weights_only=True)['generator'],
                      torch.load(resumed / 'best.pt', weights_only=True)['generator'])
    assert manifest['resume']['checkpoint_sha256'] == original_hashes['last.pt']
    assert manifest['completed_steps'] == 6
    assert manifest['resumed_from_step'] == 3
    assert original_hashes == {p.name: development._sha256(p) for p in partial.iterdir() if p.is_file()}


def test_resume_rejects_changed_config_tampered_artifacts_and_nonextension(data, tmp_path):
    partial = tmp_path / 'partial'
    development.train_probe(data, partial, CONFIG | {'steps': 3})
    for change in ({'seed': 102}, {'normalization': 'minmax'}, {'steps': 3}, {'batch_size': 8}, {'beta1': .5}):
        with pytest.raises(ValueError):
            development.train_probe(data, tmp_path / 'invalid', CONFIG | change,
                                    resume=partial / 'last.pt')
        assert not (tmp_path / 'invalid').exists()
    with (partial / 'best_validation.json').open('a') as handle:
        handle.write(' ')
    with pytest.raises(ValueError, match='hash'):
        development.train_probe(data, tmp_path / 'invalid', CONFIG, resume=partial / 'last.pt')
    assert not (tmp_path / 'invalid').exists()


def test_cpu_default_and_explicit_cpu_have_identical_trajectory(data, tmp_path):
    from fluvius_cgan.development import train_probe as original_train
    original_train(data, tmp_path / 'original', CONFIG)
    development.train_probe(data, tmp_path / 'current', CONFIG | {'device': 'cpu'})
    a = torch.load(tmp_path / 'original' / 'last.pt', weights_only=True)
    b = torch.load(tmp_path / 'current' / 'last.pt', weights_only=True)
    for key in ('generator', 'discriminator', 'optimizer_g', 'optimizer_d', 'model_rng_state',
                'noise_rng_state', 'sampler_rng_state'):
        assert_tree_equal(a[key], b[key])


def test_explicit_cuda_never_silently_falls_back(data, tmp_path):
    if torch.cuda.is_available():
        pytest.skip('Unavailable-device behavior requires a CPU-only worker')
    with pytest.raises(RuntimeError, match='CUDA'):
        development.train_probe(data, tmp_path / 'cuda', CONFIG | {'device': 'cuda'})
    assert not (tmp_path / 'cuda').exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
def test_cuda_checkpoint_can_generate_on_cpu(data, tmp_path):
    manifest = development.train_probe(data, tmp_path / 'cuda', CONFIG | {'steps': 2, 'device': 'cuda'})
    assert manifest['runtime']['device'] == 'cuda'
    assert manifest['runtime']['deterministic_algorithms'] is True
    fake = development.generate(tmp_path / 'cuda' / 'best.pt', np.array([0, 1]), 99)
    assert fake.shape == (2, 24) and np.isfinite(fake).all()
