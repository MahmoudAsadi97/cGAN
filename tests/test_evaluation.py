"""Checks for physical-unit metrics and honest small-sample diagnostics."""

import numpy as np
import pytest

from energy_cgan.evaluation import (
    conditional_bootstrap,
    evaluate_conditional,
    evaluate_profiles,
)


@pytest.fixture
def profiles():
    return np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])


def test_identical_distributions_have_zero_distances(profiles):
    result = evaluate_profiles(profiles, profiles[::-1])
    keys = (
        "per_slot_wasserstein",
        "daily_energy_wasserstein_kwh",
        "peak_power_wasserstein_kw",
        "ramp_power_wasserstein_kw",
        "mean_profile_mae",
        "std_profile_mae",
        "autocorrelation_mae_lag_1",
        "autocorrelation_mae_lag_quarter_day",
    )
    for key in keys:
        assert result[key] == pytest.approx(0.0)
    assert result["negative_fraction"] == 0.0


@pytest.mark.parametrize("unit,energy,power", [("kWh", 4.0, 4.0), ("kW", 1.0, 1.0)])
def test_unit_conversion_is_physically_correct(profiles, unit, energy, power):
    result = evaluate_profiles(profiles, profiles + 1.0, unit=unit)
    assert result["per_slot_wasserstein"] == pytest.approx(1.0)
    assert result["daily_energy_wasserstein_kwh"] == pytest.approx(energy)
    assert result["peak_power_wasserstein_kw"] == pytest.approx(power)
    assert result["ramp_power_wasserstein_kw"] == pytest.approx(0.0)
    assert result["mean_profile_mae"] == pytest.approx(1.0)
    assert result["std_profile_mae"] == pytest.approx(0.0)
    assert result["metadata"]["profile_duration_hours"] == pytest.approx(1.0)
    assert result["metadata"]["profile_value_unit"] == unit


def test_ramps_are_in_power_units():
    real = np.array([[0.0, 1.0, 2.0]])
    fake = np.array([[0.0, 2.0, 4.0]])
    assert evaluate_profiles(real, fake)["ramp_power_wasserstein_kw"] == pytest.approx(4.0)
    assert evaluate_profiles(real, fake, unit="kW")["ramp_power_wasserstein_kw"] == pytest.approx(1.0)


def test_negative_fraction_counts_generated_values(profiles):
    fake = profiles.copy()
    fake[0, 0] = -1.0
    result = evaluate_profiles(profiles, fake)
    assert result["negative_fraction"] == pytest.approx(1 / 8)


def test_unequal_distribution_sizes_are_supported(profiles):
    result = evaluate_profiles(profiles, np.repeat(profiles, 3, axis=0))
    assert result["per_slot_wasserstein"] == pytest.approx(0.0)
    assert result["metadata"]["n_real"] == 2
    assert result["metadata"]["n_fake"] == 6


def test_constant_profiles_do_not_claim_valid_autocorrelation():
    result = evaluate_profiles(np.ones((2, 8)), np.ones((2, 8)))
    assert result["autocorrelation_mae_lag_1"] is None
    assert result["metadata"]["autocorrelation_valid_real"] == 0
    assert result["warnings"]


def test_autocorrelation_compares_mean_temporal_structure():
    alternating = np.array([[0.0, 1.0] * 4])
    blocked = np.array([[0.0] * 4 + [1.0] * 4])
    result = evaluate_profiles(alternating, blocked)
    assert result["autocorrelation_mae_lag_1"] == pytest.approx(1.5)
    assert result["autocorrelation_mae_lag_quarter_day"] == pytest.approx(0.5)
    assert result["metadata"]["autocorrelation_lag_quarter_day"] == 2


def test_profile_spread_metric_detects_collapsed_variation():
    real = np.array([[0.0] * 4, [2.0] * 4])
    fake = np.ones((2, 4))
    result = evaluate_profiles(real, fake)
    assert result["mean_profile_mae"] == pytest.approx(0.0)
    assert result["std_profile_mae"] == pytest.approx(1.0)


def test_single_slot_has_no_ramp_or_autocorrelation():
    result = evaluate_profiles(np.array([[1.0], [2.0]]), np.array([[2.0]]))
    assert result["ramp_power_wasserstein_kw"] is None
    assert result["autocorrelation_mae_lag_1"] is None


@pytest.mark.parametrize(
    "real,fake",
    [
        (np.ones(4), np.ones((2, 4))),
        (np.ones((0, 4)), np.ones((2, 4))),
        (np.ones((2, 0)), np.ones((2, 0))),
        (np.ones((2, 3)), np.ones((2, 4))),
        (np.array([[np.nan, 1.0]]), np.ones((2, 2))),
        (np.ones((2, 2)), np.array([[np.inf, 1.0]])),
        (np.ones((2, 2), dtype=complex), np.ones((2, 2))),
    ],
)
def test_invalid_profiles_are_rejected(real, fake):
    with pytest.raises(ValueError):
        evaluate_profiles(real, fake)


@pytest.mark.parametrize("interval", [0, -0.25, np.inf, np.nan])
def test_invalid_interval_rejected(profiles, interval):
    with pytest.raises(ValueError, match="interval_hours"):
        evaluate_profiles(profiles, profiles, interval_hours=interval)


def test_unknown_unit_rejected(profiles):
    with pytest.raises(ValueError, match="unit"):
        evaluate_profiles(profiles, profiles, unit="Wh")


def test_nearest_neighbor_detects_exact_training_copy():
    train = np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
    holdout = train + 10.0
    result = evaluate_profiles(holdout, train.copy(), reference_train=train)
    diagnostic = result["nearest_neighbor"]
    assert diagnostic["mean_rmse_to_train"] == 0.0
    assert diagnostic["mean_rmse_to_holdout"] > 0.0
    assert diagnostic["train_to_holdout_rmse_ratio"] == 0.0
    assert "privacy" in diagnostic["caveat"].lower()
    assert diagnostic["n_reference_train"] == diagnostic["n_reference_holdout"]


def test_nearest_neighbor_zero_denominator_is_not_infinity(profiles):
    result = evaluate_profiles(profiles, profiles, reference_train=profiles)
    assert result["nearest_neighbor"]["train_to_holdout_rmse_ratio"] is None


def test_nearest_neighbor_rejects_invalid_training_reference(profiles):
    with pytest.raises(ValueError):
        evaluate_profiles(profiles, profiles, reference_train=np.ones((2, 5)))
    with pytest.raises(ValueError):
        evaluate_profiles(profiles, profiles, reference_train=np.ones((0, 4)))


def test_nearest_neighbor_is_seeded_and_capped():
    rng = np.random.default_rng(7)
    real = rng.normal(size=(2003, 2))
    fake = rng.normal(size=(2004, 2))
    train = rng.normal(size=(2005, 2))
    first = evaluate_profiles(real, fake, reference_train=train)["nearest_neighbor"]
    second = evaluate_profiles(real, fake, reference_train=train)["nearest_neighbor"]
    assert first == second
    assert first["n_fake"] == first["n_reference_train"] == first["n_reference_holdout"] == 2000


def test_bootstrap_preserves_requested_labels_and_is_reproducible():
    train = np.arange(24.0).reshape(6, 4)
    labels = np.array(["workday", "workday", "workday", "weekend", "weekend", "weekend"])
    requested = np.array(["weekend", "workday", "weekend", "workday"])
    first = conditional_bootstrap(train, labels, requested, seed=19)
    second = conditional_bootstrap(train, labels, requested, seed=19)
    np.testing.assert_array_equal(first, second)
    for row, label in zip(first, requested):
        assert any(np.array_equal(row, source) for source in train[labels == label])
    first[0, 0] = -999.0
    assert -999.0 not in train


def test_bootstrap_rejects_unsupported_conditions():
    with pytest.raises(ValueError, match="unsupported"):
        conditional_bootstrap(np.ones((2, 4)), np.array([0, 0]), np.array([1]))


def test_bootstrap_rejects_empty_request_and_misaligned_labels():
    with pytest.raises(ValueError):
        conditional_bootstrap(np.ones((2, 4)), np.array([0]), np.array([0]))
    with pytest.raises(ValueError):
        conditional_bootstrap(np.ones((2, 4)), np.array([0, 0]), np.array([]))


def test_conditional_evaluation_exposes_counts_and_small_support():
    real = np.arange(12.0).reshape(3, 4)
    labels = np.array([0, 0, 1])
    result = evaluate_conditional(real, real.copy(), labels)
    assert result["global"]["per_slot_wasserstein"] == 0.0
    assert result["per_condition"]["0"]["n_real"] == 2
    assert result["per_condition"]["0"]["support_status"] == "limited"
    assert result["per_condition"]["1"]["support_status"] == "insufficient"
    assert result["per_condition"]["1"]["warnings"]


def test_conditional_training_reference_is_filtered_by_condition():
    real = np.arange(16.0).reshape(4, 4)
    labels = np.array([0, 0, 1, 1])
    result = evaluate_conditional(real + 2, real, labels, reference_train=real, train_labels=labels)
    for condition in ("0", "1"):
        diagnostic = result["per_condition"][condition]["metrics"]["nearest_neighbor"]
        assert diagnostic["n_reference_train"] == 2
        assert diagnostic["mean_rmse_to_train"] == 0.0


def test_conditional_missing_training_condition_is_explicit():
    real = np.arange(16.0).reshape(4, 4)
    result = evaluate_conditional(
        real, real, np.array([0, 0, 1, 1]),
        reference_train=real[:2], train_labels=np.array([0, 0]),
    )
    missing = result["per_condition"]["1"]
    assert missing["n_train"] == 0
    assert missing["support_status"] == "unsupported"
    assert missing["warnings"]
    assert "nearest_neighbor" not in missing["metrics"]


def test_conditional_missing_holdout_condition_is_explicit():
    train = np.arange(16.0).reshape(4, 4)
    real = train[:2]
    result = evaluate_conditional(
        real, real.copy(), np.array([0, 0]),
        reference_train=train, train_labels=np.array([0, 0, 1, 1]),
    )
    assert set(result["per_condition"]) == {"0", "1"}
    absent = result["per_condition"]["1"]
    assert absent["n_real"] == absent["n_fake"] == 0
    assert absent["n_train"] == 2
    assert absent["support_status"] == "absent"
    assert absent["metrics"] is None
    assert absent["warnings"]
    assert result["per_condition"]["0"]["metrics"]["per_slot_wasserstein"] == 0.0


def test_conditional_single_training_profile_has_insufficient_support():
    real = np.arange(16.0).reshape(4, 4)
    result = evaluate_conditional(
        real, real, np.zeros(4, dtype=int),
        reference_train=real[:1], train_labels=np.array([0]),
    )
    assert result["per_condition"]["0"]["support_status"] == "insufficient"


def test_conditional_shapes_and_training_labels_are_validated(profiles):
    with pytest.raises(ValueError):
        evaluate_conditional(profiles, profiles[:1], np.array([0, 0]))
    with pytest.raises(ValueError):
        evaluate_conditional(profiles, profiles, np.array([0]))
    with pytest.raises(ValueError):
        evaluate_conditional(profiles, profiles, np.array([0, 0]), train_labels=np.array([0, 0]))
    with pytest.raises(ValueError):
        evaluate_conditional(
            profiles, profiles, np.array([0, 0]),
            reference_train=profiles, train_labels=np.array([0]),
        )
