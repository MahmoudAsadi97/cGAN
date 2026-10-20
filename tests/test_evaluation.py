"""Distribution checks with independently calculable physical expectations."""

import numpy as np
import pytest
from scipy.stats import wasserstein_distance


def evaluation():
    # Keeping imports inside tests makes an absent implementation a clear red.
    from fluvius_cgan import evaluation as module
    return module


def test_projection_directions_are_fixed_unit_vectors():
    directions = evaluation().projection_directions(24, count=135)
    assert directions.shape == (135, 24)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1)
    np.testing.assert_array_equal(directions, evaluation().projection_directions(24, 135))
    assert not np.array_equal(directions, evaluation().projection_directions(24, 135, 7))


@pytest.mark.parametrize('dimension,count', [(0, 2), (2, 0), (1.5, 2), (2, True)])
def test_projection_dimensions_must_be_positive_integers(dimension, count):
    with pytest.raises(ValueError):
        evaluation().projection_directions(dimension, count)


def test_sliced_wasserstein_uses_empirical_mass_for_unequal_samples():
    real = np.array([[0., 0.], [2., 4.]])
    fake = np.array([[0., 0.], [1., 2.], [4., 8.]])
    directions = np.eye(2)
    expected = np.mean([wasserstein_distance(real[:, j], fake[:, j]) for j in range(2)])
    assert evaluation().sliced_wasserstein(real, fake, directions) == pytest.approx(expected)
    assert evaluation().sliced_wasserstein(real[::-1], fake[[2, 0, 1]], directions) == pytest.approx(expected)


def test_sliced_wasserstein_translations_are_in_physical_units():
    real = np.arange(40.).reshape(10, 4)
    directions = np.array([[.5, .5, .5, .5], [1., 0., 0., 0.]])
    assert evaluation().sliced_wasserstein(real, real + 3, directions) == pytest.approx(4.5)


@pytest.mark.parametrize('directions', [np.zeros((1, 2)), [[2., 0.]], [[np.nan, 1.]], [[1.]], np.empty((0, 2))])
def test_sliced_wasserstein_rejects_nonunit_or_invalid_directions(directions):
    with pytest.raises(ValueError):
        evaluation().sliced_wasserstein([[0., 1.]], [[0., 1.]], directions)


def test_profile_metrics_conversions_and_known_distribution_shift():
    real = np.array([[0., 1., 3.], [0., 1., 3.]])
    fake = real + 2
    metrics = evaluation().profile_metrics(real, fake, interval_hours=.5, directions=np.eye(3))
    expected = {
        'slot_w1_kwh': 2., 'daily_energy_w1_kwh': 6., 'peak_w1_kw': 4.,
        'ramp_w1_kw': 0., 'mean_profile_rmse_kwh': 2., 'variance_mae_kwh2': 0.,
        'diversity_ratio': 1., 'acf_lag1_w1': 0., 'profile_sw1_kwh': 2.,
        'real_constant_profiles': 0, 'fake_constant_profiles': 0,
        'real_zero_diversity': True,
    }
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value)
    assert 'acf_lag6_w1' not in metrics


def test_ramps_have_no_midnight_wrap_and_measure_signed_kw_changes():
    real = [[0., 1., 3.]]
    fake = [[3., 1., 0.]]
    # Signed ramps [2, 4] versus [-4, -2] have W1 of six kW.
    assert evaluation().profile_metrics(real, fake, .5)['ramp_w1_kw'] == 6
    assert evaluation().profile_metrics(real, [[0., 2., 6.]], .5)['ramp_w1_kw'] == 3


def test_scaled_variance_and_diversity_are_population_slot_variances():
    real = np.array([[0., 0.], [2., 4.]])
    metrics = evaluation().profile_metrics(real, 2 * real)
    assert metrics['variance_mae_kwh2'] == pytest.approx(7.5)
    assert metrics['diversity_ratio'] == pytest.approx(4.)
    assert metrics['real_zero_diversity'] is False


def test_acf_is_biased_full_mean_with_constant_profiles_explicit():
    real = np.array([[1., 2., 3., 4., 5., 6., 7.], [2.] * 7])
    fake = np.array([[1., 2., 3., 4., 5., 6., 7.], [0., 0., 0., 0., 0., 0., 1.]])
    metrics = evaluation().profile_metrics(real, fake)
    centered = fake[1] - fake[1].mean()
    for lag in [1, 6]:
        acf = np.dot(centered[:-lag], centered[lag:]) / np.dot(centered, centered)
        real_centered = real[0] - real[0].mean()
        common = np.dot(real_centered[:-lag], real_centered[lag:]) / np.dot(real_centered, real_centered)
        expected = wasserstein_distance([common, 0], [common, acf])
        assert metrics[f'acf_lag{lag}_w1'] == pytest.approx(expected)
    assert metrics['real_constant_profiles'] == 1
    assert metrics['fake_constant_profiles'] == 0
    assert 'acf_lag6_w1' not in evaluation().profile_metrics(real, fake, .5)


def test_zero_diversity_is_finite_and_diagnosed():
    metrics = evaluation().profile_metrics(np.zeros((2, 7)), np.ones((3, 7)))
    assert metrics['diversity_ratio'] == 1
    assert metrics['real_zero_diversity'] is True
    metrics = evaluation().profile_metrics(np.zeros((2, 7)), np.arange(21).reshape(3, 7))
    assert metrics['diversity_ratio'] is None  # undefined ratio serializes as JSON null
    assert metrics['real_constant_profiles'] == 2
    assert all(np.isfinite(value) for key, value in metrics.items() if key != 'diversity_ratio')


def test_metrics_are_row_order_invariant_and_allow_unequal_sample_counts():
    rng = np.random.default_rng(19)
    real, fake = rng.normal(size=(11, 24)), rng.normal(size=(17, 24))
    first = evaluation().profile_metrics(real, fake)
    shuffled = evaluation().profile_metrics(real[::-1], fake[::-1])
    assert shuffled == pytest.approx(first)


@pytest.mark.parametrize('real,fake,interval', [
    ([], [[1]], 1), ([[1]], [[1, 2]], 1), ([[np.nan, 0]], [[0, 0]], 1),
    ([[0, np.inf]], [[0, 0]], 1), ([1, 2], [[1, 2]], 1),
    ([[1, 2]], [[1, 2]], 0), ([[1, 2]], [[1, 2]], np.nan),
    (np.empty((1, 0)), np.empty((1, 0)), 1),
])
def test_metrics_reject_invalid_profiles_or_intervals(real, fake, interval):
    with pytest.raises(ValueError):
        evaluation().profile_metrics(real, fake, interval)


def test_single_slot_profiles_have_zero_ramp_and_acf():
    metrics = evaluation().profile_metrics([[1.]], [[2.]])
    assert metrics['ramp_w1_kw'] == 0
    assert metrics['acf_lag1_w1'] == 0


def test_conditional_macro_and_real_count_weighting_exclude_diagnostics():
    real = np.array([[0., 1.], [0., 1.], [0., 1.], [1., 2.]])
    fake = np.array([[2., 3.], [5., 6.], [5., 6.]])
    result = evaluation().conditional_metrics(real, fake, ['a', 'a', 'a', 'b'], ['a', 'b', 'b'])
    assert result['per_condition']['a']['n_real'] == 3
    assert result['per_condition']['a']['n_fake'] == 1
    assert result['per_condition']['b']['n_real'] == 1
    assert result['macro']['slot_w1_kwh'] == pytest.approx(3.)
    assert result['weighted']['slot_w1_kwh'] == pytest.approx(2.5)
    assert 'real_constant_profiles' not in result['macro']
    assert 'fake_constant_profiles' not in result['weighted']
    assert 'real_zero_diversity' not in result['macro']


@pytest.mark.parametrize('real_labels,fake_labels', [
    (['a', 'b'], ['a', 'c']), (['a', 'b'], ['a', 'a']),
    (['a'], ['a', 'b']), ([None, 'b'], ['a', 'b']),
    ([np.nan, 'b'], ['a', 'b']), ([['a'], ['b']], ['a', 'b']),
    (['', 'b'], ['a', 'b']), ([1, '1'], [1, '1']),
])
def test_conditional_metrics_reject_missing_unsupported_or_ambiguous_labels(real_labels, fake_labels):
    with pytest.raises(ValueError):
        evaluation().conditional_metrics([[0, 1], [1, 2]], [[0, 1], [1, 2]], real_labels, fake_labels)


def test_conditional_undefined_diversity_is_excluded_with_valid_count():
    real = [[0., 0.], [0., 0.], [0., 1.], [2., 3.]]
    fake = [[0., 0.], [1., 1.], [0., 1.], [4., 5.]]
    result = evaluation().conditional_metrics(real, fake, ['a', 'a', 'b', 'b'], ['a', 'a', 'b', 'b'])
    assert result['per_condition']['a']['metrics']['diversity_ratio'] is None
    assert result['macro']['diversity_ratio'] == 4
    assert result['weighted']['diversity_ratio'] == 4
    assert result['metric_valid_conditions']['diversity_ratio'] == 1
    assert result['metric_valid_conditions']['slot_w1_kwh'] == 2
    all_invalid = evaluation().conditional_metrics(real[:2], fake[:2], ['a', 'a'], ['a', 'a'])
    assert all_invalid['macro']['diversity_ratio'] is None
    assert all_invalid['weighted']['diversity_ratio'] is None
    assert all_invalid['metric_valid_conditions']['diversity_ratio'] == 0


def test_decimal_constant_profiles_do_not_gain_roundoff_acf_or_diversity():
    real = np.full((3, 24), .1)
    fake = np.full((5, 24), .3)
    metrics = evaluation().profile_metrics(real, fake)
    assert metrics['real_constant_profiles'] == 3
    assert metrics['fake_constant_profiles'] == 5
    assert metrics['real_zero_diversity'] is True
    assert metrics['diversity_ratio'] == 1
    assert metrics['acf_lag1_w1'] == 0
    assert metrics['acf_lag6_w1'] == 0
    assert metrics['variance_mae_kwh2'] == 0
