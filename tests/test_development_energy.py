"""Mathematical and synthetic-trajectory checks for the optional energy loss."""
import json

import numpy as np
import pytest
import torch

from fluvius_cgan import development


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def data(tmp_path):
    path = tmp_path / 'data'
    path.mkdir()
    rng = np.random.default_rng(12)
    for split in ['train', 'val']:
        x = rng.normal(1., .2, (20, 24)).astype(np.float32)
        np.savez(path / f'{split}.npz', x=x, y=np.arange(20, dtype=np.int64) % 2,
                 meter_id=np.array([f'{split}{i}' for i in range(20)]),
                 date=np.full(20, '2024-01-01'))
    with np.load(path / 'train.npz') as values:
        scaler = {'minimum': float(values['x'].min()),
                  'maximum': float(values['x'].max()), 'fitted_on': 'train'}
    (path / 'scaler.json').write_text(json.dumps(scaler))
    (path / 'metadata.json').write_text(json.dumps({'resolution': 24, 'condition_dim': 64}))
    (path / 'test.npz').write_bytes(b'FORBIDDEN')
    return path


CONFIG = {'steps': 3, 'batch_size': 7, 'validation_every': 1,
          'validation_per_condition': 4, 'normalization': 'standard', 'seed': 101}
TRAJECTORY_KEYS = ('generator', 'discriminator', 'optimizer_g', 'optimizer_d',
                   'model_rng_state', 'noise_rng_state', 'sampler_rng_state',
                   'sampler_order', 'sampler_cursor', 'passes_started',
                   'row_presentations', 'best_step', 'best_score')


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


def _strip_energy_metadata(path):
    """Exercise actual legacy serialization, including history/hash validation.

    Historical preintegration CPU parity is recorded separately. This fixture
    reconstructs the old checkpoint/manifest format from a zero-weight trajectory.
    """
    def strip_history(history):
        for row in history:
            row.pop('loss_adversarial', None)
            row.pop('loss_energy', None)
    for name in ('last.pt', 'best.pt'):
        state = torch.load(path / name, weights_only=True)
        state['config'].pop('energy_weight')
        strip_history(state['history'])
        development._save(path / name, state)
    history = json.loads((path / 'history.json').read_text())
    strip_history(history)
    development._json(path / 'history.json', history)
    manifest = json.loads((path / 'manifest.json').read_text())
    manifest['config'].pop('energy_weight')
    manifest.pop('generator_objective')
    manifest['artifact_hashes'] = {name: development._sha256(path / name) for name in
        ('last.pt', 'best.pt', 'history.json', 'best_validation.json')}
    development._json(path / 'manifest.json', manifest)
    assert 'energy_weight' not in torch.load(path / 'last.pt', weights_only=True)['config']
    assert all('loss_energy' not in row and 'loss_adversarial' not in row for row in history)


def scalar_reference(real, fake, labels):
    """Independent scalar implementation, including every cross-sample pair."""
    values = []
    for label in labels.unique():
        x, y = real[labels == label], fake[labels == label]
        n = len(x)
        if n < 2:
            continue
        cross = sum(torch.linalg.vector_norm(a - b) for a in x for b in y) / n**2
        within_x = sum(torch.linalg.vector_norm(x[i] - x[j])
                       for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
        within_y = sum(torch.linalg.vector_norm(y[i] - y[j])
                       for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
        values.append(2 * cross - within_x - within_y)
    return torch.stack(values).mean() if values else fake.sum() * 0


def test_diagonal_exclusion_all_cross_pairs_and_negative_estimates():
    real = torch.tensor([[0.], [2.]], dtype=torch.float64)
    fake = real.clone().requires_grad_()
    result = development._conditional_energy(real, fake, torch.zeros(2, dtype=torch.long))
    # Cross mean=1 (including paired zeros); RR and FF means=2 each.
    assert result.item() == pytest.approx(-2.)
    result.backward()
    assert torch.isfinite(fake.grad).all()


def test_equal_condition_weight_and_singleton_exclusion_match_scalar_formula():
    real = torch.tensor([[0., 1.], [2., 0.], [10., 2.], [12., 0.],
                         [15., 3.], [1e6, -1e6]], dtype=torch.float64)
    fake = torch.tensor([[1., 1.], [4., 2.], [9., 0.], [14., 1.],
                         [14., 5.], [-1e6, 1e6]], dtype=torch.float64)
    labels = torch.tensor([1, 1, 5, 5, 5, 9])
    result = development._conditional_energy(real, fake, labels)
    assert result.item() == pytest.approx(scalar_reference(real, fake, labels).item(), abs=1e-12)
    terms = [development._conditional_energy(real[labels == c], fake[labels == c],
             labels[labels == c]).item() for c in (1, 5)]
    assert result.item() == pytest.approx(np.mean(terms), abs=1e-12)
    assert result.item() != pytest.approx(np.average(terms, weights=[2, 3]), abs=1e-8)


def test_symmetry_permutation_and_no_cross_condition_interaction():
    rng = torch.Generator().manual_seed(241)
    real = torch.randn(7, 3, generator=rng, dtype=torch.float64)
    fake = torch.randn(7, 3, generator=rng, dtype=torch.float64)
    labels = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    loss = development._conditional_energy(real, fake, labels)
    assert loss.item() == pytest.approx(development._conditional_energy(fake, real, labels).item(), abs=1e-12)
    shifted_real, shifted_fake = real.clone(), fake.clone()
    shifted_real[labels == 1] += 1000
    shifted_fake[labels == 1] += 1000
    assert loss.item() == pytest.approx(
        development._conditional_energy(shifted_real, shifted_fake, labels).item(), abs=1e-12)
    order = torch.tensor([6, 3, 1, 4, 2, 5, 0])
    assert loss.item() == pytest.approx(
        development._conditional_energy(real[order], fake[order], labels[order]).item(), abs=1e-12)
    # RF pairs have no privileged row pairing.
    fake_order = torch.tensor([1, 0, 4, 2, 3, 6, 5])
    assert loss.item() == pytest.approx(
        development._conditional_energy(real, fake[fake_order], labels).item(), abs=1e-12)


@pytest.mark.parametrize('count', [1, 4])
def test_only_singletons_return_differentiable_zero(count):
    fake = torch.arange(count * 3, dtype=torch.float64).reshape(count, 3).requires_grad_()
    real = fake.detach() + 100.
    result = development._conditional_energy(real, fake, torch.arange(count))
    assert result.item() == 0.
    result.backward()
    assert torch.equal(fake.grad, torch.zeros_like(fake))


def test_fake_gradient_matches_finite_differences_and_real_is_detached():
    rng = torch.Generator().manual_seed(242)
    real = torch.randn(6, 3, generator=rng, dtype=torch.float64).requires_grad_()
    fake = torch.randn(6, 3, generator=rng, dtype=torch.float64).requires_grad_()
    labels = torch.tensor([0, 0, 1, 1, 1, 2])
    assert torch.autograd.gradcheck(lambda y: development._conditional_energy(real, y, labels),
                                   (fake,), eps=1e-6, atol=1e-5, rtol=1e-4)
    development._conditional_energy(real, fake, labels).backward()
    assert real.grad is None
    assert torch.equal(fake.grad[-1], torch.zeros(3, dtype=torch.float64))
    assert fake.grad[:-1].abs().sum() > 0


def test_normal_population_energy_and_gradient_encourage_missing_spread():
    # For X~N(0,1), Y~N(0,s²), population energy and its derivative are known.
    # Repeated independent synthetic profile draws check U-statistic scaling and gradient.
    rng = torch.Generator().manual_seed(243)
    labels = torch.zeros(256, dtype=torch.long)
    sigma = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    observed_loss, observed_gradient = [], []
    for _ in range(64):
        real = torch.randn(256, 1, generator=rng, dtype=torch.float64)
        base_noise = torch.randn(256, 1, generator=rng, dtype=torch.float64)
        loss = development._conditional_energy(real, sigma * base_noise, labels)
        observed_loss.append(loss.item())
        observed_gradient.append(torch.autograd.grad(loss, sigma)[0].item())
    factor = np.sqrt(2 / np.pi)
    expected_loss = factor * (2 * np.sqrt(1 + .5**2) - np.sqrt(2) * (1 + .5))
    expected_gradient = factor * (2 * .5 / np.sqrt(1 + .5**2) - np.sqrt(2))
    assert np.mean(observed_loss) == pytest.approx(expected_loss, abs=.015)
    assert np.mean(observed_gradient) == pytest.approx(expected_gradient, abs=.025)
    assert np.mean(observed_gradient) < 0  # Gradient descent increases sigma.


@pytest.mark.parametrize('value', [True, False, -0.1, float('nan'), float('inf'), '1', None])
def test_invalid_energy_weight_is_rejected(value):
    with pytest.raises(ValueError, match='energy_weight'):
        development._config({'energy_weight': value})


def test_default_and_explicit_zero_weight_skip_auxiliary_and_match_trajectory(data, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Default must not evaluate energy loss')
    monkeypatch.setattr(development, '_conditional_energy', forbidden)
    development.train_probe(data, tmp_path / 'original', CONFIG)
    result = development.train_probe(data, tmp_path / 'current', CONFIG | {'energy_weight': 0.})
    a = torch.load(tmp_path / 'original' / 'last.pt', weights_only=True)
    b = torch.load(tmp_path / 'current' / 'last.pt', weights_only=True)
    for key in TRAJECTORY_KEYS:
        equal_tree(a[key], b[key])
    for old, new in zip(a['history'], b['history']):
        equal_tree({k: v for k, v in old.items() if k != 'elapsed_seconds'},
                   {k: v for k, v in new.items() if k in old and k != 'elapsed_seconds'})
        assert new['loss_adversarial'] == new['loss_g']
        assert new['loss_energy'] == 0.
    assert result['config']['energy_weight'] == 0.


def test_enabled_energy_uses_same_fake_draw_records_components_and_changes_training(data, tmp_path, monkeypatch):
    calls = []
    discriminator_inputs = []
    class RecordingDiscriminator(development.Discriminator):
        def forward(self, profiles, labels):
            discriminator_inputs.append((profiles.data_ptr(), labels.data_ptr()))
            return super().forward(profiles, labels)
    monkeypatch.setattr(development, 'Discriminator', RecordingDiscriminator)
    original_energy = development._conditional_energy
    def record(real, fake, labels):
        assert not real.requires_grad
        assert fake.requires_grad
        assert discriminator_inputs[-1] == (fake.data_ptr(), labels.data_ptr())
        calls.append((len(real), fake.data_ptr()))
        return original_energy(real, fake, labels)
    monkeypatch.setattr(development, '_conditional_energy', record)
    development.train_probe(data, tmp_path / 'reference', CONFIG)
    result = development.train_probe(data, tmp_path / 'energy', CONFIG | {'energy_weight': 1.})
    a = torch.load(tmp_path / 'reference' / 'last.pt', weights_only=True)
    b = torch.load(tmp_path / 'energy' / 'last.pt', weights_only=True)
    assert len(calls) == CONFIG['steps']
    assert [count for count, _ in calls] == [7, 7, 6]
    # Same sampler/noise/model RNG consumption, despite a changed optimization path.
    for key in ('noise_rng_state', 'sampler_rng_state', 'model_rng_state', 'sampler_order'):
        equal_tree(a[key], b[key])
    assert any(not torch.equal(a['generator'][key], b['generator'][key]) for key in a['generator'])
    for row in b['history']:
        assert row['loss_g'] == pytest.approx(row['loss_adversarial'] + row['loss_energy'], abs=2e-6)
    assert result['config']['energy_weight'] == 1.
    assert 'test.npz' not in result['data_hashes']


def test_energy_resume_matches_full_trajectory_and_changed_weight_is_rejected(data, tmp_path):
    config = CONFIG | {'energy_weight': 1., 'steps': 6}
    development.train_probe(data, tmp_path / 'full', config)
    development.train_probe(data, tmp_path / 'partial', config | {'steps': 3})
    development.train_probe(data, tmp_path / 'resumed', config,
                            resume=tmp_path / 'partial' / 'last.pt')
    a = torch.load(tmp_path / 'full' / 'last.pt', weights_only=True)
    b = torch.load(tmp_path / 'resumed' / 'last.pt', weights_only=True)
    for key in TRAJECTORY_KEYS:
        equal_tree(a[key], b[key])
    equal_tree([{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in a['history']],
               [{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in b['history']])
    with pytest.raises(ValueError, match='config differs'):
        development.train_probe(data, tmp_path / 'invalid', config | {'energy_weight': .5},
                                resume=tmp_path / 'partial' / 'last.pt')
    assert not (tmp_path / 'invalid').exists()


def test_legacy_no_energy_checkpoint_can_resume_without_trajectory_changes(data, tmp_path):
    config = CONFIG | {'steps': 6}
    development.train_probe(data, tmp_path / 'full', config)
    development.train_probe(data, tmp_path / 'partial', CONFIG)
    _strip_energy_metadata(tmp_path / 'partial')
    development.train_probe(data, tmp_path / 'resumed', config,
                            resume=tmp_path / 'partial' / 'last.pt')
    a = torch.load(tmp_path / 'full' / 'last.pt', weights_only=True)
    b = torch.load(tmp_path / 'resumed' / 'last.pt', weights_only=True)
    for key in TRAJECTORY_KEYS:
        equal_tree(a[key], b[key])


def test_unavailable_cuda_fails_instead_of_cpu_fallback(data, tmp_path):
    if torch.cuda.is_available():
        pytest.skip('Requires worker without CUDA')
    with pytest.raises(RuntimeError, match='CUDA'):
        development.train_probe(data, tmp_path / 'cuda', CONFIG | {'energy_weight': 1., 'device': 'cuda'})
    assert not (tmp_path / 'cuda').exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
def test_cuda_energy_is_deterministic_or_raises_for_unsupported_kernel():
    # A missing deterministic kernel must be an explicit error, not a fallback.
    device = development._device('cuda')
    outputs = []
    for _ in range(2):
        real = torch.arange(24., device=device).reshape(6, 4)
        fake = (real + .125).requires_grad_()
        with development._execution(device):
            try:
                loss = development._conditional_energy(real, fake,
                    torch.tensor([0, 0, 0, 1, 1, 1], device=device))
                loss.backward()
            except RuntimeError as error:
                assert 'deterministic' in str(error).lower()
                return
        assert fake.grad.device.type == 'cuda'
        outputs.append((loss.detach().cpu(), fake.grad.cpu()))
    equal_tree(outputs[0], outputs[1])
