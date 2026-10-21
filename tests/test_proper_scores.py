"""Finite-ensemble corrections must estimate the declared population scores."""
from dataclasses import FrozenInstanceError
import hashlib
from itertools import product
import json
import math

import numpy as np
import pytest
from scipy.special import ndtri


def core():
    from fluvius_cgan import proper_scores
    return proper_scores


def pair_spec(dimension):
    pairs = [(first, second) for first in range(dimension) for second in range(first + 1, dimension)]
    weights = np.array([1. / (second - first) for first, second in pairs])
    return pairs, weights / weights.sum()


def direct_scores(ensemble, observations, corrected=True):
    count, dimension = ensemble.shape
    pairs, weights = pair_spec(dimension)
    distance = lambda a, b: math.sqrt(sum(float(x - y) ** 2 for x, y in zip(a, b)) / dimension)
    denominator = count * (count - 1) if corrected else count * count
    offset = sum(distance(ensemble[i], ensemble[j]) for i in range(count) for j in range(count)
                 if i != j) / (2 * denominator)
    features = np.array([[math.sqrt(abs(row[h] - row[k])) for h, k in pairs] for row in ensemble])
    variogram_offset = np.dot(weights, features.var(axis=0, ddof=1)) / count if corrected else 0.
    return {
        'energy_score': np.array([sum(distance(row, observation) for row in ensemble) / count - offset
                                  for observation in observations]),
        'variogram_score': np.array([np.dot(weights,
            (np.array([math.sqrt(abs(observation[h] - observation[k])) for h, k in pairs])
             - features.mean(axis=0)) ** 2) - variogram_offset for observation in observations])}


@pytest.mark.parametrize('members', [2, 3])
def test_exact_finite_support_enumeration_is_unbiased_for_both_scores(members):
    # Keeping diagonal pairs or omitting the sample-variance correction fails this.
    support = np.array([[0., 0., 0.], [1., 4., -1.], [-2., 1., 3.]])
    probability = np.array([.2, .3, .5])
    observations = np.array([[.3, -.2, 1.5], [5., 4., 0.]])
    distance = lambda a, b: np.linalg.norm(a - b) / np.sqrt(3.)
    expected_energy = np.array([
        sum(probability[i] * distance(support[i], observation) for i in range(3))
        - .5 * sum(probability[i] * probability[j] * distance(support[i], support[j])
                   for i in range(3) for j in range(3)) for observation in observations])
    pairs, weights = pair_spec(3)
    means = np.array([sum(probability[i] * np.sqrt(abs(support[i, h] - support[i, k]))
                         for i in range(3)) for h, k in pairs])
    expected_variogram = np.array([sum(weights[p] * (np.sqrt(abs(row[h] - row[k])) - means[p]) ** 2
                                      for p, (h, k) in enumerate(pairs)) for row in observations])
    expectation = {name: np.zeros(len(observations)) for name in ('energy_score', 'variogram_score')}
    for indices in product(range(3), repeat=members):
        mass = np.prod(probability[list(indices)])
        scores = core().fit_ensemble_scores(support[list(indices)], block_size=2).score(observations)
        for name in expectation:
            expectation[name] += mass * scores[name]
    np.testing.assert_allclose(expectation['energy_score'], expected_energy, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(expectation['variogram_score'], expected_variogram, rtol=1e-13, atol=1e-13)


def test_tiled_scores_equal_direct_ordered_pair_formulas_and_are_batch_invariant():
    rng = np.random.default_rng(19)
    ensemble, observations = rng.normal(size=(7, 5)), rng.normal(size=(9, 5))
    fitted = core().fit_ensemble_scores(ensemble, block_size=3)
    expected, actual = direct_scores(ensemble, observations), fitted.score(observations)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-13, atol=1e-13)
        singles = np.concatenate([fitted.score(row[None])[name] for row in observations])
        np.testing.assert_allclose(singles, actual[name], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(fitted.score(observations[::-1])[name], actual[name][::-1], rtol=1e-13, atol=1e-13)


def test_corrected_scores_have_declared_empirical_distribution_offsets():
    ensemble = np.array([[-2., 0., 1.], [0., 3., -1.], [2., 0., 0.]])
    observations = np.array([[0., 0., 0.], [1., 3., 9.], [-5., 0., 3.]])
    fitted = core().fit_ensemble_scores(ensemble, block_size=2)
    actual = fitted.score(observations)
    empirical = direct_scores(ensemble, observations, corrected=False)
    metadata = fitted.metadata()
    np.testing.assert_allclose(empirical['energy_score'] - actual['energy_score'],
                              metadata['energy_empirical_distribution_offset'], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(empirical['variogram_score'] - actual['variogram_score'],
                              metadata['variogram_empirical_distribution_offset'], rtol=1e-13, atol=1e-13)
    assert metadata['energy_empirical_distribution_offset'] == pytest.approx(
        metadata['energy_off_diagonal_pair_mean'] / (2 * len(ensemble)))


def test_negative_corrected_variogram_is_retained():
    ensemble, observation = np.array([[0., 0.], [0., 4.]]), np.array([[0., 1.]])
    fitted = core().fit_ensemble_scores(ensemble)
    assert fitted.score(observation)['variogram_score'][0] == pytest.approx(-1.)
    assert fitted.metadata()['variogram_empirical_distribution_offset'] == pytest.approx(1.)


def test_variogram_variance_is_stable_for_large_features_with_small_spread():
    features = 1e8 + np.arange(4.)
    ensemble = np.column_stack([np.zeros(4), features ** 2])
    observation = np.array([[0., (1e8 + 1.5) ** 2]])
    fitted = core().fit_ensemble_scores(ensemble, block_size=1)
    assert fitted.score(observation)['variogram_score'][0] == pytest.approx(-5. / 12., abs=1e-12)


def test_distance_is_stable_under_large_common_coordinate_offset():
    ensemble = np.array([[0., 1., -2.], [3., 4., 2.], [-1., 5., 4.]])
    observations = np.array([[1., 2., 3.], [-1., 0., 1.]])
    before = core().fit_ensemble_scores(ensemble, block_size=2).score(observations)
    after = core().fit_ensemble_scores(ensemble + 2. ** 40, block_size=2).score(observations + 2. ** 40)
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])


def test_constant_profiles_and_signed_linear_scaling_keep_kwh_units():
    ensemble = np.array([[-1., -1., -1.], [1., 1., 1.]])
    observations = np.array([[0., 0., 0.], [2., 2., 2.]])
    constant = core().fit_ensemble_scores(ensemble).score(observations)
    np.testing.assert_allclose(constant['energy_score'], [0., 1.], atol=1e-15)
    np.testing.assert_array_equal(constant['variogram_score'], [0., 0.])
    ensemble = np.array([[1., -2., 4.], [0., 3., 2.], [-2., 0., 1.]])
    actual = core().fit_ensemble_scores(ensemble).score(observations)
    for scale in (-3., .25):
        scaled = core().fit_ensemble_scores(scale * ensemble + 17.).score(scale * observations + 17.)
        for name in actual:
            np.testing.assert_allclose(scaled[name], abs(scale) * actual[name], rtol=1e-13, atol=1e-13)


def test_fine_normal_quantile_quadrature_matches_absolute_moment_identities():
    # This is deterministic quadrature, not an IID unbiasedness/coverage claim.
    count, sigma = 2048, 2.
    quantiles = sigma * ndtri((np.arange(count) + .5) / count)
    ensemble = np.column_stack([quantiles, -quantiles])
    fitted = core().fit_ensemble_scores(ensemble, block_size=128)
    scores = fitted.score(np.zeros((1, 2)))
    expected_energy = sigma * (np.sqrt(2.) - 1.) / np.sqrt(np.pi)
    expected_variogram = (2. ** .75 * np.sqrt(sigma) * math.gamma(.75) / np.sqrt(np.pi)) ** 2
    assert scores['energy_score'][0] == pytest.approx(expected_energy, abs=.003)
    assert scores['variogram_score'][0] == pytest.approx(expected_variogram, abs=.003)


def test_distance_calls_respect_tile_bound_without_replacing_real_calculation(monkeypatch):
    module = core()
    original = module.cdist
    original_features = module._variogram_features
    sizes = []
    feature_sizes = []
    def observed_distance(first, second, *args, **kwargs):
        sizes.append((len(first), len(second)))
        return original(first, second, *args, **kwargs)
    def observed_features(rows, first, second):
        feature_sizes.append((len(rows), len(first)))
        return original_features(rows, first, second)
    monkeypatch.setattr(module, 'cdist', observed_distance)
    monkeypatch.setattr(module, '_variogram_features', observed_features)
    rng = np.random.default_rng(77)
    ensemble, observations = rng.normal(size=(13, 6)), rng.normal(size=(11, 6))
    actual = module.fit_ensemble_scores(ensemble, block_size=3).score(observations)
    assert len(sizes) > 2
    assert all(0 < first <= 3 and 0 < second <= 3 for first, second in sizes)
    assert len(feature_sizes) > 2
    assert all(0 < rows <= 3 and 0 < pairs <= 3 for rows, pairs in feature_sizes)
    for name, expected in direct_scores(ensemble, observations).items():
        np.testing.assert_allclose(actual[name], expected, rtol=1e-13, atol=1e-13)


def test_fitted_scores_own_data_and_return_fresh_metadata_and_arrays():
    ensemble = np.array([[-1., 0.], [1., 2.], [0., -3.]])
    original = ensemble.copy()
    observations = np.array([[0., 1.], [-1., 2.]])
    fitted = core().fit_ensemble_scores(ensemble, block_size=2)
    expected = fitted.score(observations)
    metadata = fitted.metadata()
    json.dumps(metadata, allow_nan=False)
    assert metadata['ensemble_size'] == 3 and metadata['profile_dim'] == 2
    assert metadata['ensemble_sha256'] == hashlib.sha256(original.astype('<f8').tobytes()).hexdigest()
    metadata['ensemble_size'] = 999
    assert fitted.metadata()['ensemble_size'] == 3
    ensemble[:] = 999
    for name, values in fitted.score(observations).items():
        np.testing.assert_array_equal(values, expected[name])
        values[:] = 123
        np.testing.assert_array_equal(fitted.score(observations)[name], expected[name])
    with pytest.raises(FrozenInstanceError):
        fitted.block_size = 999
    np.testing.assert_array_equal(observations, [[0., 1.], [-1., 2.]])


@pytest.mark.parametrize('ensemble', [
    [[0., 1.]], [[0.], [1.]], np.empty((0, 2)),
    [[0., np.nan], [1., 2.]], [[0., np.inf], [1., 2.]],
    [[True, False], [False, True]], [[0., 1.j], [1., 0.]], [0., 1.],
])
def test_invalid_ensemble_fails(ensemble):
    with pytest.raises(ValueError):
        core().fit_ensemble_scores(ensemble)


@pytest.mark.parametrize('block_size', [0, -1, True, 2.5, None])
def test_invalid_block_size_fails(block_size):
    with pytest.raises(ValueError):
        core().fit_ensemble_scores([[0., 1.], [1., 0.]], block_size=block_size)


@pytest.mark.parametrize('observations', [
    [[0.]], [[0., 1., 2.]], np.empty((0, 2)), [[np.nan, 0.]],
    [[np.inf, 0.]], [0., 1.], [[False, True]],
])
def test_invalid_observations_fail(observations):
    fitted = core().fit_ensemble_scores([[0., 1.], [1., 0.]])
    with pytest.raises(ValueError):
        fitted.score(observations)


def test_nonfinite_derived_distance_or_moment_fails_explicitly():
    with pytest.raises(FloatingPointError):
        core().fit_ensemble_scores([[-1e308, 0.], [1e308, 0.]])
    fitted = core().fit_ensemble_scores([[0., 1.], [1., 0.]])
    with pytest.raises(FloatingPointError):
        fitted.score([[1e308, -1e308]])
