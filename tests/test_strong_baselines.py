"""Synthetic distribution checks for additional TRAIN-only copula controls."""
import hashlib
import importlib
import importlib.util
import json

import numpy as np
import pytest


def fit(kind, values, labels, **kwargs):
    # This assertion also gives a feature failure before the module exists.
    assert importlib.util.find_spec('fluvius_cgan.strong_baselines') is not None
    module = importlib.import_module('fluvius_cgan.strong_baselines')
    return module.fit_strong_baseline(kind, values, labels, **kwargs)


def example():
    values = np.array([[-2., 0., 4.], [-1., 0., 5.], [0., 0., 6.],
                       [10., 3., 1.], [11., 3., 2.], [12., 3., 3.]])
    return values, np.repeat([0, 1], 3)


@pytest.mark.parametrize('kind,options', [('beta', {}), ('student_t', {'df': 4}),
                                         ('student_t', {'df': 8}), ('student_t', {'df': 16})])
def test_sampling_preserves_condition_order_constants_support_and_fit_ownership(kind, options):
    values, labels = example()
    original = values.copy()
    fitted = fit(kind, values, labels, **options)
    requested = np.tile([1, 0], 300)
    actual = fitted.sample(requested, seed=9)
    assert actual.shape == (600, 3) and actual.dtype == np.float64
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[requested == 0, 1], 0.)
    np.testing.assert_array_equal(actual[requested == 1, 1], 3.)
    for condition in (0, 1):
        for hour in range(3):
            assert np.isin(actual[requested == condition, hour],
                           original[labels == condition, hour]).all()
    np.testing.assert_array_equal(values, original)
    values[:] = 999
    labels[:] = 55
    np.testing.assert_array_equal(fitted.sample(requested, seed=9), actual)
    assert not np.array_equal(fitted.sample(requested, seed=10), actual)
    assert fitted.sample([], seed=9).shape == (0, 3)
    with pytest.raises(ValueError, match='condition'):
        fitted.sample([2], seed=9)


@pytest.mark.parametrize('kind,options', [('beta', {}), ('student_t', {'df': 4})])
def test_single_row_and_fully_constant_conditions(kind, options):
    fitted = fit(kind, [[-2., 7.], [4., 9.], [4., 9.]], [0, 1, 1], **options)
    np.testing.assert_array_equal(fitted.sample([1, 0, 1], seed=9),
                                  [[4., 9.], [-2., 7.], [4., 9.]])


def test_beta_ties_have_seeded_independent_full_rank_permutations():
    tied = np.repeat([0., 1., 2.], 20)
    values = np.column_stack([tied, tied, np.ones(len(tied))])
    labels = np.zeros(len(values), dtype=int)
    fitted = fit('beta', values, labels, fit_seed=7)
    ranks = fitted._groups[0].ranks
    for hour in range(3):
        np.testing.assert_array_equal(np.sort(ranks[:, hour]), np.arange(1, len(tied) + 1))
        order = np.argsort(ranks[:, hour])
        assert np.all(np.diff(values[order, hour]) >= 0)
    assert not np.array_equal(ranks[:, 0], ranks[:, 1])
    np.testing.assert_array_equal(ranks, fit('beta', values, labels, fit_seed=7)._groups[0].ranks)
    assert not np.array_equal(ranks, fit('beta', values, labels, fit_seed=8)._groups[0].ranks)
    with pytest.raises(ValueError):
        ranks[0, 0] = 99
    with pytest.raises(ValueError):
        ranks.flags.writeable = True


def test_beta_uses_one_common_row_to_preserve_nonlinear_dependence():
    values = np.linspace(-1., 1., 1001)
    training = np.column_stack([values, values ** 2])
    fitted = fit('beta', training, np.zeros(len(training), dtype=int))
    generated = fitted.sample(np.zeros(20000, dtype=int), seed=19)
    # Independent selection by hour destroys this U-shaped relation.
    assert np.mean(np.abs(generated[:, 1] - generated[:, 0] ** 2)) < .04


@pytest.mark.parametrize('kind,options', [('beta', {}), ('student_t', {'df': 4}),
                                         ('student_t', {'df': 8}), ('student_t', {'df': 16})])
def test_uniform_copula_margins_decode_to_exact_empirical_atoms_in_law(kind, options):
    first = np.repeat([-2., 0., 5.], [2, 7, 1])
    values = np.column_stack([first, np.arange(10.)])
    fitted = fit(kind, values, np.zeros(10, dtype=int), **options)
    generated = fitted.sample(np.zeros(60000, dtype=int), seed=143)
    for value, probability in [(-2., .2), (0., .7), (5., .1)]:
        assert abs(np.mean(generated[:, 0] == value) - probability) < .012
    # Atoms represent equal probability bins, testing uniformity without mocks.
    frequency = np.bincount(generated[:, 1].astype(int), minlength=10) / len(generated)
    np.testing.assert_allclose(frequency, .1, atol=.006, rtol=0)


def test_student_t_shared_chi_square_produces_tail_dependence_at_zero_correlation():
    a, b = np.meshgrid(np.arange(41.), np.arange(41.), indexing='ij')
    training = np.column_stack([a.ravel(), b.ravel()])
    fitted = fit('student_t', training, np.zeros(len(training), dtype=int), df=4)
    factor = fitted._groups[0].factor
    np.testing.assert_allclose(factor @ factor.T, np.eye(2), atol=1e-14)
    generated = fitted.sample(np.zeros(120000, dtype=int), seed=289)
    upper = generated >= 39
    np.testing.assert_allclose(upper.mean(axis=0), 2 / 41, atol=.003, rtol=0)
    # Separate per-coordinate chi-square draws give about (2/41)^2 = .00238.
    assert np.mean(upper.all(axis=1)) > .0045


def test_student_score_correlation_is_positive_definite_after_declared_shrinkage():
    values = np.arange(100.)
    training = np.column_stack([values, 2 * values, -values, np.ones(100)])
    fitted = fit('student_t', training, np.zeros(100, dtype=int), df=4)
    factor = fitted._groups[0].factor
    correlation = factor @ factor.T
    np.testing.assert_allclose(np.diag(correlation), 1., atol=1e-14)
    assert np.linalg.eigvalsh(correlation).min() > .0099
    assert abs(correlation[0, 1] - .99) < 1e-12
    assert abs(correlation[0, 2] + .99) < 1e-12


def test_metadata_is_independent_json_safe_and_binds_inputs_fit_and_parameters():
    values = np.column_stack([np.repeat([0., 1.], 20), np.arange(40.)])
    labels = np.zeros(40, dtype=int)
    beta = fit('beta', values, labels, fit_seed=5)
    metadata = beta.metadata()
    json.dumps(metadata, allow_nan=False)
    assert metadata['fit_seed'] == 5 and metadata['df'] is None
    assert metadata['method'] == 'conditional_empirical_beta_copula'
    assert metadata['source_profile_bytes_sha256'] == hashlib.sha256(values.astype('<f8').tobytes()).hexdigest()
    assert metadata['source_label_bytes_sha256'] == hashlib.sha256(labels.astype('<i8').tobytes()).hexdigest()
    assert metadata == fit('beta', values, labels, fit_seed=5).metadata()
    different = fit('beta', values, labels, fit_seed=6).metadata()
    assert metadata['per_condition']['0']['ranks_sha256'] != different['per_condition']['0']['ranks_sha256']
    changed = values.copy()
    changed[0, 0] = -1
    assert metadata['source_profile_bytes_sha256'] != fit('beta', changed, labels, fit_seed=5).metadata()['source_profile_bytes_sha256']
    metadata['per_condition']['0']['training_profiles'] = -1
    assert beta.metadata()['per_condition']['0']['training_profiles'] == 40
    student = fit('student_t', values, labels, df=4)
    description = student.metadata()
    assert description['fit_seed'] is None and description['df'] == 4
    assert description['estimator'] == 'average_rank_t_score_pearson_moment'
    assert description['identity_shrinkage'] == .01
    assert description['eigenvalue_floor'] == 1e-8
    assert description['maximum_likelihood_fit'] is False
    assert len(description['per_condition']['0']['correlation_factor_sha256']) == 64
    assert description == fit('student_t', values, labels, fit_seed=999, df=4).metadata()
    assert description['per_condition']['0']['correlation_factor_sha256'] != fit('student_t', values, labels, df=16).metadata()['per_condition']['0']['correlation_factor_sha256']


@pytest.mark.parametrize('values,labels', [
    ([[np.nan]], [0]), ([[np.inf]], [0]), ([[1 + 2j]], [0]),
    ([[True]], [0]), ([['1']], [0]), (np.empty((0, 2)), []),
    (np.empty((1, 0)), [0]), ([1., 2.], [0, 0]),
    ([[1.]], [-1]), ([[1.]], [64]), ([[1.]], [.1]),
    ([[1.]], [np.inf]), ([[1.]], ['0']), ([[1.]], [True]),
    ([[1.]], [0, 1]),
])
def test_invalid_fitting_inputs_fail(values, labels):
    with pytest.raises(ValueError):
        fit('beta', values, labels)


@pytest.mark.parametrize('kind,options', [
    ('unknown', {}), ('beta', {'df': 4}), ('student_t', {}),
    ('student_t', {'df': 2}), ('student_t', {'df': -1}),
    ('student_t', {'df': np.inf}), ('student_t', {'df': np.nan}),
    ('student_t', {'df': True}), ('student_t', {'df': '4'}),
    ('student_t', {'df': 10**400}),
    ('student_t', {'df': np.longdouble('1e400')}),
    ('beta', {'fit_seed': -1}), ('beta', {'fit_seed': 2**32}),
    ('beta', {'fit_seed': 4.5}), ('beta', {'fit_seed': True}),
])
def test_invalid_configuration_fails(kind, options):
    with pytest.raises(ValueError):
        fit(kind, [[1.]], [0], **options)


def test_student_accepts_finite_noninteger_degrees_of_freedom():
    fitted = fit('student_t', [[0.], [1.], [2.]], [0, 0, 0], df=2.5)
    assert fitted.metadata()['df'] == 2.5
    assert np.isin(fitted.sample([0] * 100, seed=19), [0., 1., 2.]).all()


@pytest.mark.parametrize('labels', [[True], ['0'], [.5], [-1], [64], [np.nan], [[0]]])
def test_invalid_sampling_labels_fail(labels):
    fitted = fit('beta', [[1.]], [0])
    with pytest.raises(ValueError):
        fitted.sample(labels, seed=2)


@pytest.mark.parametrize('seed', [-1, 2**32, True, '2', 3.5, None])
def test_invalid_sampling_seed_fails(seed):
    fitted = fit('beta', [[1.]], [0])
    with pytest.raises(ValueError):
        fitted.sample([0], seed=seed)


def test_original_baseline_draw_hashes_are_unchanged_after_strong_baseline_use():
    from fluvius_cgan.baselines import fit_baseline
    values, labels = example()
    requested = np.tile([1, 0], 200)
    fit('beta', values, labels).sample(requested, seed=17)
    fit('student_t', values, labels, df=4).sample(requested, seed=17)
    # Recorded from untouched original code before this module was implemented.
    expected = {
        'mean': '121bd98eeb131a41a2d71618c6b83d0a671e2f7443019189c1d8f4b4a44d9c29',
        'resample': 'f0dec1bb0013d2475c66df0ca61091586a780758a0f1fc34f186aeb5c9ba274d',
        'copula': 'feaee84e1b8eb7b7433235747ee7813e1ce019dd0b8314f5695d4323ba615a07',
    }
    for kind, digest in expected.items():
        generated = fit_baseline(kind, values, labels).sample(requested, seed=17)
        assert hashlib.sha256(generated.astype('<f8').tobytes()).hexdigest() == digest
