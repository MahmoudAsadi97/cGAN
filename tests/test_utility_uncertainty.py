import json

import numpy as np
import pytest

from fluvius_cgan.utility import RidgeProfileForecaster
from fluvius_cgan.utility_uncertainty import paired_utility_meter_bootstrap


def example():
    rng = np.random.default_rng(7)
    # Unequal selected day counts per meter; two meters per technology stratum.
    ids = np.array(["a", "a", "b", "b", "b", "c", "c", "d", "d", "d"])
    labels = np.array([0, 1, 0, 1, 1, 8, 9, 8, 9, 9])
    return rng.normal(size=(10, 3)), rng.normal(size=(10, 3)), labels, ids


def macro(errors, labels):
    return np.array([
        np.mean([np.sqrt(np.mean(errors[labels == label] ** 2)) for label in np.unique(labels)]),
        np.mean([np.mean(np.abs(errors[labels == label])) for label in np.unique(labels)]),
    ])


def test_matches_manual_paired_whole_meter_draws_and_macro_not_pooled_rmse():
    a, b, labels, ids = example()
    result = paired_utility_meter_bootstrap(a, b, labels, ids, n_bootstrap=37, seed=14, confidence=.8)
    rng = np.random.default_rng(14)
    differences = []
    for _ in range(37):
        selected = np.r_[rng.integers(0, 2, size=2), rng.integers(2, 4, size=2)]
        rows = np.concatenate([np.flatnonzero(ids == np.unique(ids)[meter]) for meter in selected])
        differences.append(macro(a[rows], labels[rows]) - macro(b[rows], labels[rows]))
    bounds = np.quantile(differences, [.1, .9], axis=0)
    for ki, metric in enumerate(["rmse_kwh", "mae_kwh"]):
        assert result["point_estimates"]["candidate"][metric] == pytest.approx(macro(a, labels)[ki])
        assert result["point_estimates"]["comparator"][metric] == pytest.approx(macro(b, labels)[ki])
        assert result["intervals"][metric] == pytest.approx({
            "estimate": macro(a, labels)[ki] - macro(b, labels)[ki],
            "lower": bounds[0, ki], "upper": bounds[1, ki]})
    assert result["point_estimates"]["candidate"]["rmse_kwh"] != pytest.approx(np.sqrt(np.mean(a ** 2)))
    assert result["stratum_meter_counts"] == {"0": 2, "1": 2}
    assert result["n_valid_bootstrap"] == 37
    json.dumps(result, allow_nan=False)


def test_agrees_with_forecaster_evaluation_on_identical_windows():
    rng = np.random.default_rng(16)
    x = rng.normal(size=(10, 24))
    _, _, labels, ids = example()
    a = RidgeProfileForecaster().fit(x, labels)
    b = RidgeProfileForecaster().fit(x * .2 + 1, labels)
    real = rng.normal(size=(10, 24))
    result = paired_utility_meter_bootstrap(
        a.predict(real, labels) - real[:, 3:], b.predict(real, labels) - real[:, 3:],
        labels, ids, n_bootstrap=5)
    assert result["point_estimates"]["candidate"] == pytest.approx(a.evaluate(real, labels)["macro"])
    assert result["point_estimates"]["comparator"] == pytest.approx(b.evaluate(real, labels)["macro"])


def test_determinism_pairing_zero_error_and_no_mutation():
    a, _, labels, ids = example()
    original = a.copy()
    result = paired_utility_meter_bootstrap(a, a, labels, ids, n_bootstrap=11)
    assert result == paired_utility_meter_bootstrap(a, a, labels, ids, n_bootstrap=11)
    assert all(row == {"estimate": 0., "lower": 0., "upper": 0.} for row in result["intervals"].values())
    np.testing.assert_array_equal(a, original)
    zero = paired_utility_meter_bootstrap(np.zeros_like(a), np.zeros_like(a), labels, ids, n_bootstrap=3)
    assert zero["point_estimates"]["candidate"] == {"rmse_kwh": 0., "mae_kwh": 0.}


def test_missing_condition_withholds_all_intervals_without_rejection_resampling():
    a, b, labels, ids = example()
    labels = labels.copy(); labels[0] = 2
    result = paired_utility_meter_bootstrap(a, b, labels, ids, n_bootstrap=100, seed=1)
    assert result["n_invalid_bootstrap"] > 0
    assert result["n_valid_bootstrap"] + result["n_invalid_bootstrap"] == 100
    assert result["missing_condition_draws"]["2"] > 0
    assert result["interval_status"] == "withheld_missing_conditions"
    assert all(row["lower"] is None and row["upper"] is None for row in result["intervals"].values())


@pytest.mark.parametrize("name,value,match", [
    ("n_bootstrap", 1, "at least two"), ("n_bootstrap", True, "at least two"),
    ("seed", -1, "nonnegative integer"), ("seed", False, "nonnegative integer"),
    ("confidence", True, "between zero and one"), ("confidence", 1., "between zero and one"),
    ("confidence", float("nan"), "between zero and one"),
])
def test_invalid_settings(name, value, match):
    with pytest.raises(ValueError, match=match):
        paired_utility_meter_bootstrap(*example(), **{name: value})


@pytest.mark.parametrize("change,match", [
    (lambda a, b, y, ids: (a[:, :2], b, y, ids), "identical rows"),
    (lambda a, b, y, ids: (a * np.nan, b, y, ids), "finite numeric"),
    (lambda a, b, y, ids: (a.astype(bool), b, y, ids), "finite numeric"),
    (lambda a, b, y, ids: (a, b, y.astype(float), ids), "integer condition"),
    (lambda a, b, y, ids: (a, b, y + 64, ids), "integer condition"),
    (lambda a, b, y, ids: (a, b, y, np.array([""] * len(ids))), "empty identifiers"),
    (lambda a, b, y, ids: (a, b, y, np.array(["a"] * len(ids))), "at least two"),
    (lambda a, b, y, ids: (a, b, y, np.roll(ids, 1)), "exactly one stratum"),
])
def test_invalid_inputs(change, match):
    with pytest.raises(ValueError, match=match):
        paired_utility_meter_bootstrap(*change(*example()), n_bootstrap=3)


def test_overflow_is_explicit_and_not_serialized_as_nonfinite():
    a, b, labels, ids = example()
    with pytest.raises(ValueError, match="nonfinite utility error"):
        paired_utility_meter_bootstrap(np.full_like(a, 1e300), b, labels, ids, n_bootstrap=3)


def test_finite_per_meter_statistics_cannot_overflow_when_combined():
    with pytest.raises(ValueError, match="nonfinite utility error statistics in a meter draw"):
        paired_utility_meter_bootstrap(
            np.full((2, 1), 1e154), np.zeros((2, 1)), np.array([0, 0]),
            np.array(["a", "b"]), n_bootstrap=2, seed=0)
