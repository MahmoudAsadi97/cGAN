"""Scientific metric and input-contract checks in physical units."""

import numpy as np
import pytest

from fluvius_cgan.evaluation import (
    evaluate_conditional,
    evaluate_profiles,
    mean_baseline,
    meter_bootstrap_confidence_intervals,
    nearest_neighbor_diagnostics,
    resample_baseline,
)


def test_identity_distributions_have_zero_errors():
    x = np.array([[0., 1., 2., 3.], [1., 0., 3., 2.]])
    result = evaluate_profiles(x, x[::-1])
    for name in (
        "pointwise_wasserstein_kwh", "daily_energy_wasserstein_kwh",
        "peak_power_wasserstein_kw", "ramp_wasserstein_kw",
        "mean_profile_rmse_kwh", "pointwise_variance_rmse_kwh2",
    ):
        assert result[name] == pytest.approx(0.)
    assert result["diversity_ratio"] == pytest.approx(1.)
    assert result["n_real"] == result["n_generated"] == 2
    for lag in result["autocorrelation"].values():
        assert lag["wasserstein"] == pytest.approx(0.)
        assert lag["mean_difference_abs"] == pytest.approx(0.)


def test_known_energy_peak_ramp_units_and_signed_ramps():
    real = np.array([[0., 1., 2.]])
    generated = np.array([[1., 3., 5.]])
    result = evaluate_profiles(real, generated, interval_hours=.25)
    assert result["pointwise_wasserstein_kwh"] == pytest.approx(2.)
    assert result["daily_energy_wasserstein_kwh"] == pytest.approx(6.)
    assert result["peak_power_wasserstein_kw"] == pytest.approx(12.)
    assert result["ramp_wasserstein_kw"] == pytest.approx(4.)
    assert result["real_mean_daily_energy_kwh"] == pytest.approx(3.)
    assert result["generated_mean_peak_power_kw"] == pytest.approx(20.)
    assert result["mean_profile_rmse_kwh"] == pytest.approx(np.sqrt(14 / 3))
    assert result["diversity_ratio"] is None
    descending = evaluate_profiles([[0., 1., 2.]], [[2., 1., 0.]])
    assert descending["daily_energy_wasserstein_kwh"] == 0
    assert descending["ramp_wasserstein_kw"] == pytest.approx(8.)


def test_acf_is_per_profile_with_documented_constant_convention():
    result = evaluate_profiles([[0., 1., 0., 1.], [2., 2., 2., 2.]],
                               [[0., 1., 0., 1.], [2., 2., 2., 2.]])
    assert result["autocorrelation"]["1"]["real_mean"] == pytest.approx(-.375)
    assert result["real_constant_profile_count"] == 1
    assert set(evaluate_profiles(np.ones((2, 96)), np.ones((3, 96)))["autocorrelation"]) == {"1", "4", "24"}
    assert set(evaluate_profiles(np.ones((2, 24)), np.ones((3, 24)), 1.)["autocorrelation"]) == {"1", "6"}


def test_single_interval_has_explicit_unavailable_temporal_metrics():
    result = evaluate_profiles([[1.]], [[2.]])
    assert result["ramp_wasserstein_kw"] is None
    assert result["autocorrelation"] == {}


def test_conditional_metrics_macro_and_counts():
    result = evaluate_conditional([[0., 0.], [1., 1.], [1., 1.]], [0, 1, 1],
                                  [[2., 2.], [2., 2.]], [0, 1])
    assert result["counts"] == {"real": {"0": 1, "1": 2}, "generated": {"0": 1, "1": 1}}
    assert result["macro"]["pointwise_wasserstein_kwh"] == pytest.approx(1.5)
    assert result["per_condition"]["1"]["n_real"] == 2
    assert result["n_conditions"] == 2


def test_conditional_missing_classes_raise():
    with pytest.raises(ValueError, match="support"):
        evaluate_conditional([[0., 0.], [1., 1.]], [0, 1], [[0., 0.]], [0])


@pytest.mark.parametrize("bad", [[], [1., 2.], [[np.nan, 1.]], [[np.inf, 1.]],
                                 np.empty((0, 2)), np.empty((2, 0)), [["x", "y"]],
                                 [[1+1j, 2]], [[True, False]]])
def test_invalid_profiles_raise(bad):
    with pytest.raises(ValueError):
        evaluate_profiles(bad, [[0., 1.]])


@pytest.mark.parametrize("hours", [0, -.25, np.inf, np.nan, True])
def test_invalid_interval_raises(hours):
    with pytest.raises(ValueError):
        evaluate_profiles([[0., 1.]], [[0., 1.]], hours)


def test_different_widths_raise():
    with pytest.raises(ValueError, match="width"):
        evaluate_profiles([[0., 1.]], [[0., 1., 2.]])


@pytest.mark.parametrize("labels", [[0., 1.], [-1, 1], [0, 64], [False, True], [0], [[0, 1]]])
def test_invalid_labels_raise(labels):
    with pytest.raises(ValueError):
        resample_baseline([[0., 1.], [2., 3.]], labels, [0])


def test_conditional_baselines_are_deterministic_and_keep_label_order():
    train = np.array([[0., 0.], [2., 2.], [10., 20.]])
    train_labels = np.array([0, 0, 1])
    requested = np.array([1, 0, 1, 0, 0, 0, 0, 0])
    a = resample_baseline(train, train_labels, requested, seed=15)
    np.testing.assert_array_equal(a, resample_baseline(train, train_labels, requested, seed=15))
    np.testing.assert_array_equal(a[requested == 1], [[10., 20.], [10., 20.]])
    assert np.isin(a[requested == 0], [0., 2.]).all()
    np.testing.assert_array_equal(mean_baseline(train, train_labels, [1, 0]), [[10., 20.], [1., 1.]])
    assert not np.shares_memory(a, train)


@pytest.mark.parametrize("baseline", [mean_baseline, resample_baseline])
def test_baseline_unsupported_conditions_raise(baseline):
    with pytest.raises(ValueError, match="unsupported"):
        baseline([[0., 1.]], [0], [1])


def test_nearest_neighbors_detect_copies_in_rms_units():
    result = nearest_neighbor_diagnostics([[0., 0.], [4., 4.]],
                                          [[0., 0.], [1., 1.]], [[2., 2.]])
    assert result["generated_exact_copy_fraction"] == pytest.approx(.5)
    assert result["generated_to_train_mean_rms_kwh"] == pytest.approx(.5)
    assert result["heldout_to_train_mean_rms_kwh"] == pytest.approx(2.)
    assert result["sample_counts"] == {"train": 2, "generated": 2, "heldout": 1}


def test_nearest_neighbor_sampling_is_bounded_and_deterministic():
    x = np.arange(120, dtype=float).reshape(60, 2)
    a = nearest_neighbor_diagnostics(x, x, x + .1, max_samples=7, seed=12)
    assert a == nearest_neighbor_diagnostics(x, x, x + .1, max_samples=7, seed=12)
    assert a["sample_counts"] == {"train": 7, "generated": 7, "heldout": 7}
    assert a["total_counts"] == {"train": 60, "generated": 60, "heldout": 60}
    assert a["reference_scope"] == "sampled_training_profiles"
    with pytest.raises(ValueError):
        nearest_neighbor_diagnostics(x, x, x, max_samples=0)


def test_bootstrap_clusters_whole_meters_and_is_deterministic():
    real = np.array([[0., 0.], [0., 0.], [2., 2.], [2., 2.]])
    generated = np.ones((3, 2))
    kwargs = {"n_bootstrap": 30, "seed": 10,
              "metrics": ["pointwise_wasserstein_kwh", "mean_profile_rmse_kwh"]}
    result = meter_bootstrap_confidence_intervals(real, generated, ["a", "a", "b", "b"], **kwargs)
    assert result == meter_bootstrap_confidence_intervals(real, generated, ["a", "a", "b", "b"], **kwargs)
    assert result["n_meters"] == 2
    assert result["resampling_unit"] == "real_meter"
    assert result["generated_sample_policy"] == "fixed"
    assert result["intervals"]["pointwise_wasserstein_kwh"]["lower"] == pytest.approx(1.)
    assert result["intervals"]["pointwise_wasserstein_kwh"]["upper"] == pytest.approx(1.)
    with pytest.raises(ValueError, match="two"):
        meter_bootstrap_confidence_intervals(real, generated, ["a"] * 4, **kwargs)
