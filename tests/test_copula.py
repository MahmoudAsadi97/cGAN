"""Synthetic-only scientific checks for the train-only fresh-sample baseline."""
import numpy as np
import pytest
from scipy.special import ndtri

from fluvius_cgan.copula import (
    ConditionalGaussianCopula, _inverse_quantiles, _normal_scores,
    _repair_correlation, fit_conditional_copula,
)


def example(width=3):
    rng = np.random.default_rng(18)
    x = rng.normal(size=(120, width))
    x[60:] = 20 + 3 * x[60:]
    return x, np.repeat([0, 8], 60)


def test_midrank_normal_scores_are_finite_and_order_invariant_with_ties():
    x = np.array([[0., 3.], [0., 3.], [2., 3.], [5., 3.]])
    z = _normal_scores(x)
    np.testing.assert_allclose(z[:, 0], ndtri(np.array([1.5, 1.5, 3., 4.]) / 4 - .5 / 4))
    np.testing.assert_array_equal(z[:, 1], 0.)
    permutation = [3, 0, 2, 1]
    np.testing.assert_array_equal(_normal_scores(x[permutation]), z[permutation])


@pytest.mark.parametrize("width", [1, 24, 96])
def test_selected_training_subset_support_signed_units_and_fresh_draws(width):
    x, labels = example(width)
    before = x.copy()
    model = fit_conditional_copula(x, labels, quantile_knots=None)
    np.testing.assert_array_equal(model.supported_labels, [0, 8])
    assert model.train_counts == {0: 60, 8: 60}
    requested = np.tile([8, 0, 0, 8], 20)
    generated = model.generate(requested, seed=19, batch_size=13)
    assert generated.shape == (80, width) and np.isfinite(generated).all()
    for label in [0, 8]:
        training = x[labels == label]
        samples = generated[requested == label]
        assert np.all(samples >= training.min(axis=0))
        assert np.all(samples <= training.max(axis=0))
    assert generated[requested == 8].min() > 1
    assert (generated[requested == 0] < 0).any()
    assert all(not np.any(np.all(row == x, axis=1)) for row in generated)
    np.testing.assert_array_equal(x, before)
    x[:] = 1e6  # Fitted parameters own their memory.
    np.testing.assert_array_equal(model.generate(requested, seed=19, batch_size=13), generated)
    with pytest.raises(ValueError, match="unsupported"):
        model.generate(np.array([1]))


def test_conditional_fit_is_isolated_from_other_conditions():
    x, labels = example()
    full = fit_conditional_copula(x, labels)
    selected = fit_conditional_copula(x[:60], labels[:60])
    for name in ["quantiles", "correlation", "cholesky"]:
        np.testing.assert_array_equal(getattr(full.models[0], name), getattr(selected.models[0], name))
    with pytest.raises(ValueError, match="unsupported"):
        selected.generate(np.array([8]))


def test_constant_columns_singletons_and_singular_perfect_dependence():
    x = np.column_stack([np.arange(8.), -np.arange(8.), np.full(8, -2.)])
    x = np.vstack([x, [99., -90., 7.]])
    labels = np.r_[np.zeros(8, dtype=int), 8]
    model = fit_conditional_copula(x, labels, shrinkage=0)
    generated = model.generate(np.array([0] * 100 + [8] * 3), seed=7)
    np.testing.assert_array_equal(generated[:100, 2], -2.)
    np.testing.assert_array_equal(generated[100:], np.repeat(x[-1:], 3, axis=0))
    np.testing.assert_allclose(model.models[0].correlation[:2, 2], 0, atol=1e-12)
    assert np.isfinite(generated).all()
    assert np.linalg.eigvalsh(model.models[0].correlation).min() > 0


def test_psd_repair_handles_indefinite_asymmetric_input():
    bad = np.array([[1., 1.8, -.8], [1.9, 1., .8], [-.8, .8, 1.]])
    result = _repair_correlation(bad)
    assert np.linalg.eigvalsh(result).min() > 0
    np.testing.assert_array_equal(np.diag(result), 1.)
    np.testing.assert_allclose(result, result.T, rtol=0, atol=0)
    assert np.isfinite(np.linalg.cholesky(result)).all()


def test_exact_and_compressed_marginal_quantiles_match_linear_definition():
    x = np.array([[-10., 1.], [-2., 1.], [0., 2.], [1., 2.], [80., 9.]])
    y = np.zeros(len(x), dtype=int)
    exact = fit_conditional_copula(x, y, quantile_knots=None)
    compressed = fit_conditional_copula(x, y, quantile_knots=3)
    np.testing.assert_array_equal(exact.models[0].quantiles, np.sort(x, axis=0))
    np.testing.assert_allclose(compressed.models[0].quantiles,
                               np.quantile(x, [0., .5, 1.], axis=0, method="linear"))
    probabilities = np.array([0., .13, .5, .99, 1.])
    inverse = _inverse_quantiles(np.repeat(probabilities[:, None], 2, axis=1), exact.models[0].quantiles)
    np.testing.assert_allclose(inverse, np.quantile(x, probabilities, axis=0, method="linear"))
    np.testing.assert_array_equal(inverse[[0, -1]], np.array([x.min(axis=0), x.max(axis=0)]))


def test_default_knots_bound_model_storage_and_preserve_endpoints():
    x = np.random.default_rng(1).normal(size=(400, 3))
    model = fit_conditional_copula(x, np.zeros(len(x), dtype=int))
    assert model.models[0].quantiles.shape == (257, 3)
    np.testing.assert_array_equal(model.models[0].quantiles[0], x.min(axis=0))
    np.testing.assert_array_equal(model.models[0].quantiles[-1], x.max(axis=0))


def test_joint_dependence_is_fitted_and_propagated_to_fresh_samples():
    rng = np.random.default_rng(8)
    latent = rng.normal(size=(1500, 3))
    latent[:, 1] = .85 * latent[:, 0] + np.sqrt(1 - .85 ** 2) * latent[:, 1]
    x = np.column_stack([latent[:, 0], np.exp(latent[:, 1]), latent[:, 2]])
    model = fit_conditional_copula(x, np.zeros(len(x), dtype=int), quantile_knots=None)
    assert model.models[0].correlation[0, 1] > .8
    fresh = model.generate(np.zeros(5000, dtype=int), seed=9)
    recovered = np.corrcoef(_normal_scores(fresh), rowvar=False)
    assert recovered[0, 1] > .8
    assert abs(recovered[0, 2]) < .1
    np.testing.assert_allclose(np.quantile(fresh, [.1, .5, .9], axis=0),
                               np.quantile(x, [.1, .5, .9], axis=0), atol=.25, rtol=.1)


def test_seed_batch_order_and_npz_replay(tmp_path):
    x, y = example()
    model = fit_conditional_copula(x, y)
    labels = np.tile([8, 0, 8, 0, 0], 100)
    first = model.generate(labels, seed=42, batch_size=17)
    np.testing.assert_array_equal(model.generate(labels, seed=42, batch_size=17), first)
    np.testing.assert_allclose(model.generate(labels, seed=42, batch_size=63), first, rtol=1e-13, atol=1e-13)
    assert not np.array_equal(model.generate(labels, seed=43, batch_size=17), first)
    assert model.generate(np.array([], dtype=int)).shape == (0, 3)
    path = tmp_path / "fitted.npz"
    model.save(path)
    loaded = ConditionalGaussianCopula.load(path)
    assert loaded.train_counts == model.train_counts and loaded.quantile_knots == 257
    np.testing.assert_array_equal(loaded.generate(labels, seed=42, batch_size=17), first)


@pytest.mark.parametrize("knots", [None, 2, 257])
def test_singleton_and_exact_margin_archives_replay(tmp_path, knots):
    x = np.array([[1., -2.], [5., -3.], [20., 90.]])
    y = np.array([0, 0, 8])
    model = fit_conditional_copula(x, y, quantile_knots=knots)
    path = tmp_path / "copula.npz"
    model.save(path)
    loaded = ConditionalGaussianCopula.load(path)
    assert loaded.quantile_knots == knots
    np.testing.assert_array_equal(loaded.generate(y), model.generate(y))


@pytest.mark.parametrize("invalid", ["nonfinite", "complex"])
def test_invalid_numeric_archive_is_rejected(tmp_path, invalid):
    x, y = example()
    path = tmp_path / "invalid.npz"
    fit_conditional_copula(x, y).save(path)
    with np.load(path, allow_pickle=False) as saved:
        contents = {key: saved[key] for key in saved.files}
    if invalid == "complex":
        contents["quantiles_0"] = contents["quantiles_0"].astype(complex) + 1j
    else:
        contents["quantiles_0"][0, 0] = np.nan
    np.savez(path, **contents)
    with pytest.raises(ValueError, match="parameters"):
        ConditionalGaussianCopula.load(path)


@pytest.mark.parametrize("x,y", [
    (np.empty((0, 3)), np.array([], dtype=int)),
    (np.empty((2, 0)), np.array([0, 0])),
    (np.ones(3), np.array([0, 0, 0])),
    ([[np.nan]], [0]), ([[np.inf]], [0]), ([[1j]], [0]), ([[True]], [0]),
    ([["1"]], [0]), ([[1.]], [0.]), ([[1.]], [-1]), ([[1.]], [64]),
    ([[1.]], [[0]]), ([[1.]], [True]), ([[1.]], [0, 0]),
])
def test_invalid_training_shapes_types_and_values(x, y):
    with pytest.raises(ValueError):
        fit_conditional_copula(x, y)


@pytest.mark.parametrize("kwargs", [
    {"condition_dim": 0}, {"condition_dim": True}, {"condition_dim": 1.5},
    {"quantile_knots": 1}, {"quantile_knots": True}, {"quantile_knots": 2.5},
    {"shrinkage": -1}, {"shrinkage": 1.1}, {"shrinkage": np.nan}, {"shrinkage": True},
])
def test_invalid_fit_settings(kwargs):
    with pytest.raises(ValueError):
        fit_conditional_copula([[1.], [2.]], [0, 0], **kwargs)


@pytest.mark.parametrize("labels,kwargs", [
    ([0.], {}), ([True], {}), ([64], {}), ([-1], {}), ([[0]], {}),
    ([0], {"batch_size": 0}), ([0], {"batch_size": True}), ([0], {"batch_size": 1.5}),
    ([0], {"seed": -1}), ([0], {"seed": True}), ([0], {"seed": 1.5}),
])
def test_invalid_generation_inputs(labels, kwargs):
    model = fit_conditional_copula([[1.], [2.]], [0, 0])
    with pytest.raises(ValueError):
        model.generate(labels, **kwargs)
