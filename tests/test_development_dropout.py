"""Controlled changes to the existing discriminator dropout layer."""
import json

import numpy as np
import pytest
import torch

from fluvius_cgan import development as original

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
    return path


CONFIG = {'steps': 3, 'batch_size': 7, 'validation_every': 1, 'validation_per_condition': 4,
          'normalization': 'standard', 'seed': 101}


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for first, second in zip(a, b):
            equal_tree(first, second)
    else:
        assert a == b


@pytest.mark.parametrize('value', [True, False, -0.1, 1, float('nan'), float('inf'), '.2', None])
def test_invalid_dropout_is_rejected(value):
    with pytest.raises(ValueError, match='discriminator_dropout'):
        development._config({'discriminator_dropout': value})


def test_default_dropout_preserves_original_cpu_trajectory(data, tmp_path):
    original.train_probe(data, tmp_path / 'original', CONFIG)
    result = development.train_probe(data, tmp_path / 'current', CONFIG)
    a = torch.load(tmp_path / 'original' / 'last.pt', weights_only=True)
    b = torch.load(tmp_path / 'current' / 'last.pt', weights_only=True)
    for key in ('generator', 'discriminator', 'optimizer_g', 'optimizer_d', 'model_rng_state',
                'noise_rng_state', 'sampler_rng_state', 'sampler_order', 'best_step', 'best_score'):
        equal_tree(a[key], b[key])
    assert result['discriminator_dropout'] == .4
    assert b['config']['discriminator_dropout'] == .4


@pytest.mark.parametrize('probability', [0, .2])
def test_changed_dropout_changes_training_and_cannot_resume_original(data, tmp_path, probability):
    development.train_probe(data, tmp_path / 'reference', CONFIG)
    result = development.train_probe(data, tmp_path / 'alternative',
                                     CONFIG | {'discriminator_dropout': probability})
    a = torch.load(tmp_path / 'reference' / 'last.pt', weights_only=True)
    b = torch.load(tmp_path / 'alternative' / 'last.pt', weights_only=True)
    assert result['discriminator_dropout'] == probability
    assert any(not torch.equal(a['discriminator'][key], b['discriminator'][key])
               for key in a['discriminator'])
    assert a['discriminator'].keys() == b['discriminator'].keys()
    with pytest.raises(ValueError, match='config differs'):
        development.train_probe(data, tmp_path / 'invalid',
            CONFIG | {'steps': 6, 'discriminator_dropout': probability},
            resume=tmp_path / 'reference' / 'last.pt')
    assert not (tmp_path / 'invalid').exists()
