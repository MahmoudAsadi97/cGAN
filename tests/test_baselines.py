import numpy as np
import pytest

from fluvius_cgan.baselines import fit_baseline


def training_data():
    x = np.array([[-2, 0, 4], [-1, 0, 5], [0, 0, 6],
                  [10, 3, 1], [11, 3, 2], [12, 3, 3]], dtype=float)
    return x, np.repeat([0, 1], 3)


@pytest.mark.parametrize("kind", ["mean", "resample", "copula"])
def test_seeded_sampling_preserves_order_support_and_constants(kind):
    x, y = training_data()
    fitted = fit_baseline(kind, x, y)
    labels = np.tile([1, 0], 200)
    a = fitted.sample(labels, seed=17)
    np.testing.assert_array_equal(a, fitted.sample(labels, seed=17))
    assert a.shape == (400, 3)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a[labels == 0, 1], 0)
    np.testing.assert_array_equal(a[labels == 1, 1], 3)
    assert np.all(a[labels == 0, 0] <= 0)
    assert np.all(a[labels == 1, 0] >= 10)
    with pytest.raises(ValueError, match="condition"):
        fitted.sample(np.array([2]), seed=17)


def test_mean_control_has_zero_within_condition_variance():
    x, y = training_data()
    fitted = fit_baseline("mean", x, y)
    np.testing.assert_array_equal(
        fitted.sample([1, 0, 1], seed=1),
        np.array([[11, 3, 2], [-1, 0, 5], [11, 3, 2]]))


def test_resampling_draws_complete_training_profiles():
    x, y = training_data()
    fitted = fit_baseline("resample", x, y)
    generated = fitted.sample(np.zeros(1000, dtype=int), seed=8)
    assert {tuple(row) for row in generated} == {tuple(row) for row in x[:3]}
    x[:] = 999
    assert not np.any(fitted.sample([0], seed=8) == 999)


def test_copula_preserves_ties_empirical_support_and_strong_dependence():
    values = np.repeat(np.arange(-5, 6), 10)
    x = np.column_stack([values, 2 * values, np.zeros(len(values))])
    fitted = fit_baseline("copula", x, np.zeros(len(x), dtype=int))
    generated = fitted.sample(np.zeros(5000, dtype=int), seed=12)
    for j in range(3):
        assert np.isin(generated[:, j], x[:, j]).all()
    assert np.corrcoef(generated[:, :2].T)[0, 1] > 0.95
    assert (generated[:, 0] == 0).sum() > 200
    assert generated[:, 0].min() < 0


def test_copula_supports_a_fully_constant_condition():
    fitted = fit_baseline("copula", [[-2, 0], [-2, 0]], [4, 4])
    np.testing.assert_array_equal(fitted.sample([4, 4, 4], seed=9),
                                  [[-2, 0]] * 3)


@pytest.mark.parametrize("x,y", [
    ([[np.nan, 0]], [0]),
    ([[1, 2]], [0, 1]),
    ([[1, 2]], [0.5]),
    (np.empty((0, 2)), []),
    ([[1, 2]], [-1]),
])
def test_invalid_training_inputs_fail(x, y):
    with pytest.raises(ValueError):
        fit_baseline("copula", x, y)


def test_unknown_baseline_and_invalid_generated_labels_fail():
    x, y = training_data()
    with pytest.raises(ValueError, match="kind"):
        fit_baseline("unknown", x, y)
    fitted = fit_baseline("resample", x, y)
    with pytest.raises(ValueError):
        fitted.sample([0.1], seed=1)
