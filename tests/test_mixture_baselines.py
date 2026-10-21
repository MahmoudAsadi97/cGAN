"""Distributional and provenance tests for the conditional mixture control."""
import hashlib
import importlib
import importlib.util
import json

import numpy as np
import pytest
from scipy.special import ndtr


def module():
    assert importlib.util.find_spec('fluvius_cgan.mixture_baselines') is not None
    return importlib.import_module('fluvius_cgan.mixture_baselines')


def fit(values, labels, components=2, **kwargs):
    return module().fit_mixture_baseline(values, labels, components, **kwargs)


def example():
    first = np.column_stack([np.linspace(-2., 3., 40), np.zeros(40), np.arange(40.)])
    second = np.column_stack([np.linspace(10., 14., 40), np.full(40, 3.), np.arange(40.)[::-1]])
    return np.concatenate([first, second]), np.repeat([0, 7], 40)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mixture_marginal_pit_uses_all_components_and_covariance_diagonal():
    weights = np.array([.3, .7])
    means = np.array([[-2., 1.], [3., -1.]])
    covariance = np.array([[[4., 1.7], [1.7, 1.]], [[1., -.5], [-.5, 9.]]])
    values = np.array([[-2., 1.], [0., 0.], [3., -1.]])
    expected = .3 * ndtr((values - means[0]) / [2., 1.])
    expected += .7 * ndtr((values - means[1]) / [1., 3.])
    np.testing.assert_allclose(module()._mixture_marginal_cdf(values, weights, means, covariance),
                               expected, atol=1e-15, rtol=0)
    assert not np.allclose(expected, ndtr((values - means[0]) / [2., 1.]))


def test_known_gaussian_mixture_pit_is_uniform_in_each_coordinate():
    rng = np.random.default_rng(528)
    weights = np.array([.17, .83])
    means = np.array([[-3., 2.], [2., -1.]])
    covariance = np.array([[[.5, .2], [.2, 1.]], [[2., -.9], [-.9, 1.]]])
    selected = rng.choice(2, size=90000, p=weights)
    normal = rng.standard_normal((len(selected), 2))
    latent = np.empty_like(normal)
    for component in range(2):
        mask = selected == component
        latent[mask] = normal[mask] @ np.linalg.cholesky(covariance[component]).T + means[component]
    probability = module()._mixture_marginal_cdf(latent, weights, means, covariance)
    for hour in range(2):
        frequencies = np.histogram(probability[:, hour], bins=np.linspace(0., 1., 11))[0] / len(latent)
        np.testing.assert_allclose(frequencies, .1, atol=.004, rtol=0)


def test_single_component_matches_its_gaussian_copula_exactly():
    values = np.column_stack([np.arange(80.), np.sin(np.arange(80.)), np.ones(80)])
    fitted = fit(values, np.zeros(80), components=1)
    group = fitted._groups[0]
    rng = np.random.default_rng(84)
    rng.choice(1, size=500, p=group.weights)
    noise = rng.standard_normal((500, 2))
    latent = noise @ np.linalg.cholesky(group.covariances[0]).T + group.means[0]
    probability = ndtr((latent - group.means[0]) / np.sqrt(np.diag(group.covariances[0])))
    indices = np.clip(np.ceil(80 * probability).astype(int) - 1, 0, 79)
    expected = np.ones((500, 3))
    target = np.sort(values, axis=0)
    expected[:, :2] = target[indices, np.arange(2)]
    np.testing.assert_array_equal(fitted.sample(np.zeros(500), seed=84), expected)


def test_mixture_captures_nonlinear_dependence_a_single_gaussian_cannot():
    horizontal = np.linspace(-1., 1., 601)
    values = np.column_stack([horizontal, horizontal ** 2])
    labels = np.zeros(len(values), dtype=int)
    single = fit(values, labels, components=1).sample(np.zeros(12000), seed=49)
    mixture = fit(values, labels, components=4).sample(np.zeros(12000), seed=49)
    single_error = np.mean(np.abs(single[:, 1] - single[:, 0] ** 2))
    mixture_error = np.mean(np.abs(mixture[:, 1] - mixture[:, 0] ** 2))
    assert single_error > .25
    assert mixture_error < .14
    assert mixture_error < .5 * single_error


def test_sample_retains_condition_order_constants_signed_atoms_and_marginals():
    values, labels = example()
    values[:40, 0] = np.repeat([-2., 0., 5.], [8, 28, 4])
    original = values.copy()
    fitted = fit(values, labels)
    requested = np.tile([7, 0], 25000)
    generated = fitted.sample(requested, seed=142)
    assert generated.shape == (50000, 3) and generated.dtype == np.float64
    np.testing.assert_array_equal(generated[requested == 0, 1], 0.)
    np.testing.assert_array_equal(generated[requested == 7, 1], 3.)
    for condition in (0, 7):
        for hour in range(3):
            assert np.isin(generated[requested == condition, hour], original[labels == condition, hour]).all()
    for atom, frequency in [(-2., .2), (0., .7), (5., .1)]:
        assert abs(np.mean(generated[requested == 0, 0] == atom) - frequency) < .012
    values[:] = 999
    labels[:] = 63
    np.testing.assert_array_equal(fitted.sample(requested, seed=142), generated)
    assert fitted.sample([], seed=142).shape == (0, 3)


def test_constant_only_conditions_skip_em_even_if_components_exceed_rows():
    fitted = fit([[-2., 0.], [7., 4.], [7., 4.]], [0, 7, 7], components=8)
    np.testing.assert_array_equal(fitted.sample([7, 0, 7], seed=3), [[7., 4.], [-2., 0.], [7., 4.]])
    for record in fitted.metadata()['per_condition'].values():
        assert record['status'] == 'constant'
        assert record['initializations'] == []
        assert record['selected_init'] is None


def test_rank_stream_is_fixed_across_component_counts_and_em_stream_is_separate():
    values = np.column_stack([np.repeat([0., 1., 2.], 20)] * 2 + [np.ones(60)])
    labels = np.zeros(60)
    first = fit(values, labels, components=1, fit_seed=18).metadata()['per_condition']['0']
    second = fit(values, labels, components=2, fit_seed=18).metadata()['per_condition']['0']
    changed = fit(values, labels, components=2, fit_seed=19).metadata()['per_condition']['0']
    assert first['ranks_sha256'] == second['ranks_sha256']
    assert first['normal_scores_sha256'] == second['normal_scores_sha256']
    assert first['ranks_sha256'] != changed['ranks_sha256']
    assert [r['seed'] for r in first['initializations']] == [r['seed'] for r in second['initializations']]
    assert len(set(r['seed'] for r in second['initializations'])) == 2


def test_every_initialization_and_train_only_selection_are_recorded():
    values, labels = example()
    fitted = fit(values, labels, components=3, max_iter=1)
    metadata = fitted.metadata()
    assert metadata['method'] == 'conditional_normal_score_gaussian_mixture_pit'
    assert metadata['copula_maximum_likelihood_fit'] is False
    assert metadata['config'] == dict(n_components=3, fit_seed=20261027, n_init=2,
                                      max_iter=1, tol=.001, reg_covar=.001)
    assert metadata['source_profile_bytes_sha256'] == hashlib.sha256(values.astype('<f8').tobytes()).hexdigest()
    assert metadata['source_label_bytes_sha256'] == hashlib.sha256(labels.astype('<i8').tobytes()).hexdigest()
    for group in metadata['per_condition'].values():
        attempts = group['initializations']
        assert len(attempts) == 2
        assert group['selected_init'] == int(np.argmax([record['final_score'] for record in attempts]))
        assert all(not record['converged'] and record['n_iter'] == 1 for record in attempts)
        assert all(any(warning['category'] == 'ConvergenceWarning' for warning in record['warnings'])
                   for record in attempts)
        np.testing.assert_allclose(np.sum(group['effective_occupancies']), 40., atol=1e-10)
        assert min(group['covariance_min_eigenvalues']) >= .001 - 1e-12
    json.dumps(metadata, allow_nan=False)
    metadata['per_condition']['0']['selected_init'] = -999
    assert fitted.metadata()['per_condition']['0']['selected_init'] >= 0


@pytest.mark.parametrize('scores,selected', [([5., 6.], 1), ([5., 5.], 0)])
def test_final_train_score_overrides_lower_bound_and_exact_ties_keep_earliest(monkeypatch, scores, selected):
    # Deliberately separate final likelihood from the previous EM lower bound.
    # Ordinary converged fits can order both alike and hide the wrong selector.
    class FixedFit:
        counter = 0

        def __init__(self, **kwargs):
            self.index = FixedFit.counter
            FixedFit.counter += 1
            self.lower_bound_ = [10., -10.][self.index]
            self.converged_, self.n_iter_ = True, 1
            self.weights_ = np.ones(1)
            self.means_ = np.array([[float(self.index), 0.]])
            self.covariances_ = np.array([np.eye(2)])

        def fit(self, values):
            return self

        def score(self, values):
            return scores[self.index]

        def predict_proba(self, values):
            return np.ones((len(values), 1))

    monkeypatch.setattr(module(), 'GaussianMixture', FixedFit)
    fitted = fit([[0., 0.], [1., 1.], [2., 4.], [3., 9.]], [0] * 4, components=1)
    assert fitted.metadata()['per_condition']['0']['selected_init'] == selected
    assert fitted._groups[0].means[0, 0] == selected


def test_fitting_and_sampling_are_deterministic_and_preserve_global_rng():
    values, labels = example()
    global_state = np.random.get_state()
    first = fit(values, labels, fit_seed=42)
    second = fit(values, labels, fit_seed=42)
    assert first.metadata() == second.metadata()
    np.testing.assert_array_equal(first.sample([0, 7] * 150, 6), second.sample([0, 7] * 150, 6))
    assert not np.array_equal(first.sample([0] * 100, 6), first.sample([0] * 100, 7))
    after = np.random.get_state()
    assert global_state[0] == after[0] and global_state[2:] == after[2:]
    np.testing.assert_array_equal(global_state[1], after[1])
    for group in first._groups.values():
        for array in (group.target, group.weights, group.means, group.covariances):
            with pytest.raises(ValueError):
                array.flags.writeable = True


def test_save_load_is_hash_bound_numeric_only_and_sample_exact(tmp_path):
    values, labels = example()
    fitted = fit(values, labels)
    path = tmp_path / 'mixture.npz'
    fitted.save(path)
    with np.load(path, allow_pickle=False) as stored:
        assert all(stored[key].dtype.kind != 'O' for key in stored.files)
    loaded = module().load_mixture_baseline(path, digest(path))
    assert loaded.metadata() == fitted.metadata()
    np.testing.assert_array_equal(loaded.sample([7, 0] * 100, 91), fitted.sample([7, 0] * 100, 91))
    with pytest.raises(FileExistsError):
        fitted.save(path)
    with pytest.raises(ValueError, match='SHA-256'):
        module().load_mixture_baseline(path, '0' * 64)
    with pytest.raises(ValueError, match='SHA-256'):
        module().load_mixture_baseline(path, None)


def test_constant_artifact_roundtrip_and_repeated_fit_artifacts_are_identical(tmp_path):
    first = fit([[-2., 0.], [7., 4.], [7., 4.]], [0, 7, 7], components=8)
    second = fit([[-2., 0.], [7., 4.], [7., 4.]], [0, 7, 7], components=8)
    first_path, second_path = tmp_path / 'first.npz', tmp_path / 'second.npz'
    first.save(first_path)
    second.save(second_path)
    assert digest(first_path) == digest(second_path)
    loaded = module().load_mixture_baseline(first_path, digest(first_path))
    np.testing.assert_array_equal(loaded.sample([7, 0], 17), [[7., 4.], [-2., 0.]])


@pytest.mark.parametrize('key,value', [('selection_rule', 'best validation score'),
                                      ('covariance_regularization', 'eigenvalue clipping'),
                                      ('sklearn_n_init_per_attempt', 2),
                                      ('probability_transform', 'sampled component CDF')])
def test_loader_rejects_changed_estimator_declarations(tmp_path, key, value):
    path = tmp_path / 'mixture.npz'
    fit([[0.], [1.], [2.]], [0, 0, 0]).save(path)
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    metadata = json.loads(arrays['metadata'].item())
    metadata[key] = value
    arrays['metadata'] = np.asarray(json.dumps(metadata))
    with path.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    with pytest.raises(ValueError, match='declaration'):
        module().load_mixture_baseline(path, digest(path))


@pytest.mark.parametrize('change', ['target', 'covariance', 'weights', 'condition', 'schema', 'extra'])
def test_hash_updated_but_malformed_or_inconsistent_artifact_is_rejected(tmp_path, change):
    values, labels = example()
    path = tmp_path / 'mixture.npz'
    fit(values, labels).save(path)
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    if change == 'target':
        arrays['target_0'][0, 0] -= 1
    elif change == 'covariance':
        arrays['covariances_0'][0, 0, 0] = -1
    elif change == 'weights':
        arrays['weights_0'][0] = .99
    elif change == 'extra':
        arrays['untracked'] = np.array([1.])
    else:
        metadata = json.loads(arrays['metadata'].item())
        metadata['conditions' if change == 'condition' else 'schema_version'] = [0, 0] if change == 'condition' else 99
        arrays['metadata'] = np.asarray(json.dumps(metadata))
    with path.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    with pytest.raises(ValueError):
        module().load_mixture_baseline(path, digest(path))


def test_rare_components_and_extreme_physical_values_stay_finite():
    values = np.array([[-1e200, 0.], [1e200, 1.], [0., 0.], [0., 0.]])
    fitted = fit(values, np.zeros(4), components=4)
    generated = fitted.sample(np.zeros(10000), seed=80)
    assert np.isfinite(generated).all()
    assert np.isin(generated[:, 0], values[:, 0]).all()
    assert min(fitted.metadata()['per_condition']['0']['weights']) > 0


@pytest.mark.parametrize('values,labels', [
    ([[np.nan]], [0]), ([[np.inf]], [0]), ([[1 + 2j]], [0]),
    ([[True]], [0]), ([['1']], [0]), (np.empty((0, 2)), []),
    (np.empty((1, 0)), [0]), ([1., 2.], [0, 0]),
    ([[1.]], [-1]), ([[1.]], [64]), ([[1.]], [.1]),
    ([[1.]], [np.inf]), ([[1.]], ['0']), ([[1.]], [True]),
    ([[1.]], [0, 1]),
])
def test_invalid_training_inputs_fail(values, labels):
    with pytest.raises(ValueError):
        fit(values, labels)


@pytest.mark.parametrize('options', [
    {'n_components': 0}, {'n_components': True}, {'n_components': 1.5},
    {'n_init': 0}, {'n_init': True}, {'max_iter': 0}, {'max_iter': 1.5},
    {'tol': 0}, {'tol': -1}, {'tol': np.inf}, {'tol': np.nan}, {'tol': '1'},
    {'reg_covar': 0}, {'reg_covar': -1}, {'reg_covar': True}, {'reg_covar': np.inf},
    {'fit_seed': -1}, {'fit_seed': 2**32}, {'fit_seed': True}, {'fit_seed': 1.5},
])
def test_invalid_configuration_fails(options):
    with pytest.raises(ValueError):
        module().fit_mixture_baseline([[0.], [1.]], [0, 0], **{'n_components': 1, **options})


def test_more_components_than_nonconstant_rows_fails():
    with pytest.raises(ValueError, match='components'):
        fit([[0.], [1.]], [0, 0], components=3)


@pytest.mark.parametrize('labels,seed', [([1], 0), ([True], 0), ([.1], 0), ([64], 0),
                                       ([0], -1), ([0], True), ([0], 2**32)])
def test_invalid_sample_arguments_fail(labels, seed):
    fitted = fit([[1.]], [0])
    with pytest.raises(ValueError):
        fitted.sample(labels, seed)
