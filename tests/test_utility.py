"""Temporal alignment, training isolation and ridge utility checks on toy arrays."""
import copy

import numpy as np
import pytest

from fluvius_cgan.utility import RidgeProfileForecaster, _feature_batch, utility_comparison


def test_features_are_chronological_and_never_cross_daily_boundaries():
    profiles = np.stack((np.arange(24), 1000 + np.arange(24)))
    features, targets = _feature_batch(profiles, np.array([3, 51]), lags=3)
    assert features.shape == (42, 72)
    np.testing.assert_array_equal(features[0, :3], [0, 1, 2])
    np.testing.assert_array_equal(features[20, :3], [20, 21, 22])
    np.testing.assert_array_equal(features[21, :3], [1000, 1001, 1002])
    np.testing.assert_array_equal(targets, np.r_[np.arange(3, 24), 1000 + np.arange(3, 24)])
    phase = 2 * np.pi * 3 / 24
    np.testing.assert_allclose(features[0, 3:7], [np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)])
    assert features[0, 7 + 3] == features[21, 7 + 51] == 1
    np.testing.assert_array_equal(features[:, 7:71].sum(axis=1), np.ones(42))
    np.testing.assert_array_equal(features[:, -1], np.ones(42))


@pytest.mark.parametrize("resolution,lags", [(24, 3), (96, 12)])
def test_defaults_and_window_counts(resolution, lags):
    x = np.arange(4 * resolution, dtype=np.float64).reshape(4, resolution)
    labels = np.array([0, 0, 8, 8])
    model = RidgeProfileForecaster().fit(x, labels)
    assert model.lags_ == lags
    assert model.fit_summary_["n_windows"] == 4 * (resolution - lags)
    assert model.fit_summary_["ridge_alpha"] == 1
    assert model.predict(x, labels).shape == (4, resolution - lags)
    report = model.evaluate(x, labels)
    assert report["overall"]["n_windows"] == 4 * (resolution - lags)
    assert report["per_condition"]["8"]["n_windows"] == 2 * (resolution - lags)


def test_streaming_fit_matches_independent_dense_ridge_and_batch_sizes():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(19, 24))
    labels = rng.integers(0, 4, size=len(x))
    # Independent reference: explicitly enumerate each within-day target window.
    rows, targets = [], []
    for profile, label in zip(x, labels):
        for target in range(3, 24):
            phase = 2 * np.pi * target / 24
            onehot = np.eye(64)[label]
            rows.append(np.r_[profile[target - 3:target], np.sin(phase), np.cos(phase),
                              np.sin(2 * phase), np.cos(2 * phase), onehot, 1.])
            targets.append(profile[target])
    features = np.asarray(rows)
    penalty = np.eye(features.shape[1])
    penalty[-1, -1] = 0
    expected = np.linalg.solve(features.T @ features + penalty, features.T @ targets)
    for size in (1, 7, 256):
        model = RidgeProfileForecaster(batch_profiles=size).fit(x, labels)
        np.testing.assert_allclose(model.coef_, expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(model.predict(x, labels).ravel(), features @ expected, rtol=1e-10, atol=1e-10)


def test_evaluation_cannot_refit_or_change_training_coefficients():
    rng = np.random.default_rng(3)
    train = rng.normal(size=(12, 24))
    labels = np.arange(12) % 2
    model = RidgeProfileForecaster(batch_profiles=2).fit(train, labels)
    before = copy.deepcopy(model.__dict__)
    first = model.evaluate(rng.normal(size=(4, 24)), np.array([0, 1, 0, 1]))
    second = model.evaluate(100 + rng.normal(size=(4, 24)), np.array([0, 1, 0, 1]))
    assert first["overall"]["rmse_kwh"] != second["overall"]["rmse_kwh"]
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(model.__dict__[key], value)
        else:
            assert model.__dict__[key] == value


def test_target_and_future_values_do_not_enter_their_own_prediction():
    rng = np.random.default_rng(5)
    train = rng.normal(size=(25, 24))
    model = RidgeProfileForecaster(lags=3).fit(train, np.zeros(25, dtype=int))
    heldout = rng.normal(size=(2, 24))
    labels = np.zeros(2, dtype=int)
    prediction = model.predict(heldout, labels)
    changed = heldout.copy()
    changed[0, 10:] += 10000
    changed_prediction = model.predict(changed, labels)
    # Output index7 predicts slot10: that slot itself is not an input yet.
    np.testing.assert_array_equal(changed_prediction[0, :8], prediction[0, :8])
    np.testing.assert_array_equal(changed_prediction[1], prediction[1])


def test_known_signed_ar_relation_generalizes_and_beats_persistence():
    def ar_profiles(seed, count):
        rng = np.random.default_rng(seed)
        x = np.empty((count, 24))
        x[:, 0] = rng.normal(0, 4, size=count)
        for t in range(1, 24):
            x[:, t] = -.7 * x[:, t - 1] - .2
        return x
    train, heldout = ar_profiles(4, 50), ar_profiles(8, 12)
    model = RidgeProfileForecaster(lags=1).fit(train, np.zeros(len(train), dtype=int))
    report = model.evaluate(heldout, np.zeros(len(heldout), dtype=int))
    assert model.coef_[0] == pytest.approx(-.7, abs=.002)
    assert report["overall"]["rmse_kwh"] < .005
    assert report["overall"]["rmse_kwh"] < report["persistence"]["overall"]["rmse_kwh"] / 100
    assert model.predict(heldout, np.zeros(len(heldout), dtype=int)).min() < 0


def test_persistence_alignment_and_macro_is_mean_of_condition_rmse():
    train = np.ones((4, 24))
    labels = np.array([0, 0, 1, 1])
    model = RidgeProfileForecaster().fit(train, labels)
    heldout = np.stack([np.arange(24), 3 * np.arange(24), 3 * np.arange(24)])
    report = model.evaluate(heldout, np.array([0, 1, 1]))["persistence"]
    assert report["per_condition"]["0"]["rmse_kwh"] == 1
    assert report["per_condition"]["1"]["rmse_kwh"] == 3
    assert report["macro"] == {"rmse_kwh": 2., "mae_kwh": 2.}
    assert report["overall"]["rmse_kwh"] == pytest.approx(np.sqrt(19 / 3))
    assert report["overall"]["mae_kwh"] == pytest.approx(7 / 3)


def test_intercept_is_unpenalized_even_for_constant_negative_profiles():
    x = np.full((3, 24), -2.)
    model = RidgeProfileForecaster(batch_profiles=1).fit(x, np.array([0, 1, 1]))
    np.testing.assert_allclose(model.coef_[:-1], 0, atol=1e-11)
    assert model.coef_[-1] == pytest.approx(-2., abs=1e-11)
    assert model.evaluate(x, np.array([0, 1, 1]))["overall"]["rmse_kwh"] < 1e-12


def test_comparison_uses_same_holdout_and_reports_training_sizes():
    rng = np.random.default_rng(9)
    train = rng.normal(size=(8, 24))
    labels = np.arange(8) % 2
    heldout = rng.normal(size=(5, 24))
    heldout_labels = np.arange(5) % 2
    report = utility_comparison(train, labels, train.copy(), labels.copy(), heldout, heldout_labels, batch_profiles=2)
    assert report["methods"]["train_real"] == report["methods"]["train_synthetic"]
    assert report["forecast_horizon_hours"] == 1
    assert report["training"]["real"]["n_profiles"] == 8
    assert all(method["overall"]["n_windows"] == 5 * 21 for method in report["methods"].values())
    assert "persistence" not in report["methods"]["train_real"]
    assert "not recursive" in report["interpretation"]


@pytest.mark.parametrize("operation", ["predict", "evaluate"])
def test_unfitted_model_rejected(operation):
    with pytest.raises(ValueError, match="fitted"):
        getattr(RidgeProfileForecaster(), operation)(np.zeros((2, 24)), np.zeros(2, dtype=int))


@pytest.mark.parametrize("kwargs", [{"lags": 0}, {"lags": True}, {"lags": 1.5},
                                     {"batch_profiles": 0}, {"batch_profiles": True}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        RidgeProfileForecaster(**kwargs)


@pytest.mark.parametrize("x,labels", [
    (np.zeros((0, 24)), np.array([], dtype=int)),
    (np.zeros((2, 12)), np.array([0, 0])),
    (np.full((2, 24), np.nan), np.array([0, 0])),
    (np.zeros((2, 24)), np.array([0., 1.])),
    (np.zeros((2, 24)), np.array([True, False])),
    (np.zeros((2, 24)), np.array([0, 64])),
    (np.zeros((2, 24)), np.array([-1, 0])),
    (np.zeros((2, 24)), np.array([0])),
])
def test_invalid_training_inputs_rejected(x, labels):
    with pytest.raises(ValueError):
        RidgeProfileForecaster().fit(x, labels)


def test_resolution_lag_and_unknown_conditions_fail_closed():
    model = RidgeProfileForecaster().fit(np.zeros((2, 24)), np.array([0, 0]))
    with pytest.raises(ValueError, match="unsupported"):
        model.evaluate(np.zeros((1, 24)), np.array([1]))
    with pytest.raises(ValueError, match="resolution"):
        model.predict(np.zeros((1, 96)), np.array([0]))
    with pytest.raises(ValueError, match="lags"):
        RidgeProfileForecaster(lags=24).fit(np.zeros((2, 24)), np.array([0, 0]))


def test_feature_materialization_is_bounded_by_profile_batch(monkeypatch):
    import fluvius_cgan.utility as utility
    original = utility._feature_batch
    observed = []

    def observe(profiles, labels, lags):
        observed.append(len(profiles))
        assert len(profiles) <= 3
        return original(profiles, labels, lags)

    monkeypatch.setattr(utility, "_feature_batch", observe)
    x = np.arange(11 * 24, dtype=np.float32).reshape(11, 24)
    labels = np.zeros(11, dtype=int)
    model = RidgeProfileForecaster(batch_profiles=3).fit(x, labels)
    model.evaluate(x, labels)
    model.predict(x, labels)
    assert observed == [3, 3, 3, 2] * 3
