"""Paired cluster inference must preserve meters, conditions, and fixed references."""

import json

import numpy as np
import pytest

from fluvius_cgan.evaluation import evaluate_conditional
from fluvius_cgan.uncertainty import paired_condition_macro_meter_bootstrap


METRICS = (
    "pointwise_wasserstein_kwh",
    "daily_energy_wasserstein_kwh",
    "ramp_wasserstein_kw",
)


def example():
    # Every meter has both conditions, with unequal days per meter and condition.
    return {
        "real": np.array([[-2., 1., 0.], [3., 0., 2.], [1., 1., 1.],
                          [4., -1., 2.], [1., 0., 2.], [2., 4., 0.],
                          [2., 2., 3.], [-1., 0., 0.]]),
        "real_labels": np.array([0, 1, 0, 1, 0, 1, 1, 1]),
        "real_meter_ids": np.array(["a", "a", "b", "b", "c", "c", "c", "c"]),
        "generated": np.array([[1., -1., 3.], [0., 1., 1.], [2., 2., 2.]]),
        "generated_labels": np.array([1, 0, 0]),
        "comparator": np.array([[3., 2., 1.], [0., 0., 0.], [1., 2., 3.],
                                [1., 2., 3.], [-3., -2., -1.]]),
        "comparator_labels": np.array([0, 1, 1, 1, 0]),
    }


def conditional_errors(data, indices, method, hours):
    return evaluate_conditional(
        data["real"][indices], data["real_labels"][indices],
        data[method], data[f"{method}_labels"], interval_hours=hours,
    )["macro"]


def test_matches_existing_metrics_and_manual_paired_whole_meter_bootstrap():
    data = example()
    n_bootstrap, seed, confidence, hours = 29, 17, .8, .5
    result = paired_condition_macro_meter_bootstrap(
        **data, n_bootstrap=n_bootstrap, seed=seed, confidence=confidence,
        interval_hours=hours,
    )
    points = {method: conditional_errors(data, slice(None), method, hours)
              for method in ("generated", "comparator")}
    groups = [np.flatnonzero(data["real_meter_ids"] == meter)
              for meter in np.unique(data["real_meter_ids"])]
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[meter] for meter in chosen])
        generated = conditional_errors(data, indices, "generated", hours)
        comparator = conditional_errors(data, indices, "comparator", hours)
        differences.append([generated[metric] - comparator[metric] for metric in METRICS])
    quantiles = np.quantile(differences, [.1, .9], axis=0)
    for i, metric in enumerate(METRICS):
        for method in points:
            assert result["point_estimates"][method][metric] == pytest.approx(points[method][metric])
        assert result["intervals"][metric] == pytest.approx({
            "estimate": points["generated"][metric] - points["comparator"][metric],
            "lower": quantiles[0, i], "upper": quantiles[1, i],
        })
    assert result["n_valid_bootstrap"] == n_bootstrap
    assert result["n_invalid_bootstrap"] == 0
    assert result["interval_status"] == "complete"
    assert result["condition_weighting"] == "equal_over_original_observed_support"
    assert result["counts"]["real"] == {"0": 3, "1": 5}
    assert result["counts"]["generated"] == {"0": 2, "1": 1}
    assert result["counts"]["comparator"] == {"0": 2, "1": 3}
    json.dumps(result, allow_nan=False)


def test_is_deterministic_and_does_not_mutate_inputs():
    data = example()
    original = {key: value.copy() for key, value in data.items()}
    result = paired_condition_macro_meter_bootstrap(**data, n_bootstrap=11, seed=3)
    assert result == paired_condition_macro_meter_bootstrap(**data, n_bootstrap=11, seed=3)
    for key in data:
        np.testing.assert_array_equal(data[key], original[key])


def test_stratified_bootstrap_matches_manual_fixed_meter_counts_per_stratum():
    data = example()
    data["real_meter_ids"] = np.repeat(["a", "b", "c", "d"], 2)
    strata = np.repeat(["north", "south"], 4)
    n_bootstrap, seed = 19, 31
    result = paired_condition_macro_meter_bootstrap(
        **data, n_bootstrap=n_bootstrap, seed=seed, real_meter_strata=strata,
    )
    meters = np.unique(data["real_meter_ids"])
    groups = [np.flatnonzero(data["real_meter_ids"] == meter) for meter in meters]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        # Preserve two sampled meters from each stratum, including repetitions.
        selected = np.concatenate((rng.integers(0, 2, size=2), rng.integers(2, 4, size=2)))
        indices = np.concatenate([groups[meter] for meter in selected])
        generated = conditional_errors(data, indices, "generated", .25)
        comparator = conditional_errors(data, indices, "comparator", .25)
        draws.append([generated[metric] - comparator[metric] for metric in METRICS])
    bounds = np.quantile(draws, [.025, .975], axis=0)
    for i, metric in enumerate(METRICS):
        assert result["intervals"][metric]["lower"] == pytest.approx(bounds[0, i])
        assert result["intervals"][metric]["upper"] == pytest.approx(bounds[1, i])
    assert result["resampling_scheme"] == "within_stratum_real_meter"
    assert result["stratum_meter_counts"] == {"north": 2, "south": 2}
    assert "nominal coverage is not guaranteed" in result["coverage_interpretation"]


@pytest.mark.parametrize("strata,match", [
    ([0, 1, 0, 0, 0, 0, 0, 0], "exactly one stratum"),
    ([0, 0, 1, 1, 1, 1, 1, 1], "at least two distinct meters per stratum"),
    ([0] * 7, "matching real rows"),
    ([0.] * 8, "string/integer"),
    ([""] * 8, "empty strata"),
])
def test_strata_require_consistent_meter_assignment_and_replicated_meters(strata, match):
    with pytest.raises(ValueError, match=match):
        paired_condition_macro_meter_bootstrap(**example(), real_meter_strata=strata, n_bootstrap=2)


def test_reference_order_and_duplicate_frequency_preserve_distribution_not_row_pairing():
    data = example()
    original = paired_condition_macro_meter_bootstrap(**data, n_bootstrap=7)
    # Duplication changes counts, but not a reference's empirical distribution.
    for method in ("generated", "comparator"):
        data[method] = np.repeat(data[method][::-1], 3, axis=0)
        data[f"{method}_labels"] = np.repeat(data[f"{method}_labels"][::-1], 3)
    result = paired_condition_macro_meter_bootstrap(**data, n_bootstrap=7)
    for metric in METRICS:
        assert result["intervals"][metric] == pytest.approx(original["intervals"][metric])


def test_identical_references_have_zero_paired_difference_in_every_draw():
    data = example()
    data["comparator"] = data["generated"][::-1]
    data["comparator_labels"] = data["generated_labels"][::-1]
    result = paired_condition_macro_meter_bootstrap(**data, n_bootstrap=13)
    for interval in result["intervals"].values():
        assert interval == pytest.approx({"estimate": 0., "lower": 0., "upper": 0.})


def test_difference_direction_and_equal_condition_weight_are_explicit():
    # Unequal condition frequencies must not turn macro into a pooled metric.
    real = np.zeros((6, 2))
    result = paired_condition_macro_meter_bootstrap(
        real, [0, 1, 1, 0, 1, 1], [[0., 0.], [0., 0.]], [0, 1],
        [[2., 2.], [0., 0.]], [0, 1], ["a", "a", "a", "b", "b", "b"],
        n_bootstrap=5,
    )
    assert result["intervals"][METRICS[0]] == {"estimate": -1., "lower": -1., "upper": -1.}
    assert result["intervals"][METRICS[1]] == {"estimate": -2., "lower": -2., "upper": -2.}
    assert result["negative_difference_favors"] == "generated"


def test_missing_conditions_withhold_intervals_and_report_exact_draw_counts():
    n_bootstrap, seed = 23, 5
    result = paired_condition_macro_meter_bootstrap(
        [[0., 0.], [2., 2.]], [0, 1], [[0., 0.], [2., 2.]], [0, 1],
        [[1., 1.], [3., 3.]], [0, 1], ["a", "b"],
        n_bootstrap=n_bootstrap, seed=seed,
    )
    rng = np.random.default_rng(seed)
    selections = [rng.integers(0, 2, size=2) for _ in range(n_bootstrap)]
    missing = {str(condition): sum(condition not in chosen for chosen in selections)
               for condition in (0, 1)}
    n_invalid = sum(missing.values())
    assert result["missing_condition_draw_counts"] == missing
    assert result["n_invalid_bootstrap"] == n_invalid > 0
    assert result["n_valid_bootstrap"] == n_bootstrap - n_invalid
    assert result["interval_status"] == "unavailable_missing_condition_draws"
    assert result["intervals"][METRICS[0]]["estimate"] == -1.
    for interval in result["intervals"].values():
        assert interval["lower"] is None and interval["upper"] is None
    json.dumps(result, allow_nan=False)


def test_all_draws_missing_support_are_reported_without_empty_quantiles():
    # Seed 0 draws [2, 1, 1] then [0, 0, 0], so no draw has all three conditions.
    profiles = np.array([[0., 0.], [1., 1.], [2., 2.]])
    result = paired_condition_macro_meter_bootstrap(
        profiles, [0, 1, 2], profiles, [0, 1, 2], profiles + 1, [0, 1, 2],
        [0, 1, 2], n_bootstrap=2, seed=0,
    )
    assert result["n_valid_bootstrap"] == 0
    assert result["n_invalid_bootstrap"] == 2
    # Several absent conditions in one draw still count as only one invalid draw.
    assert result["missing_condition_draw_counts"] == {"0": 1, "1": 1, "2": 1}
    assert all(row["lower"] is None and row["upper"] is None
               for row in result["intervals"].values())


@pytest.mark.parametrize("method", ["generated", "comparator"])
def test_condition_support_must_match_for_each_reference(method):
    data = example()
    data[f"{method}_labels"] = np.full(len(data[method]), 0)
    with pytest.raises(ValueError, match=f"condition support differs for {method}"):
        paired_condition_macro_meter_bootstrap(**data, n_bootstrap=2)


@pytest.mark.parametrize("key,value,match", [
    ("n_bootstrap", 1, "n_bootstrap"),
    ("n_bootstrap", 2.5, "n_bootstrap"),
    ("n_bootstrap", True, "n_bootstrap"),
    ("confidence", 1., "confidence"),
    ("confidence", np.nan, "confidence"),
    ("confidence", "invalid", "confidence"),
    ("confidence", True, "confidence"),
    ("seed", -1, "seed"),
    ("seed", True, "seed"),
    ("interval_hours", 0, "interval_hours"),
    ("real_meter_ids", ["a"] * 8, "at least two"),
    ("real_meter_ids", ["a"] * 7, "matching real rows"),
    ("real_meter_ids", ["a"] * 7 + [" "], "empty identifiers"),
    ("real_labels", [0] * 7, "profile count"),
    ("generated", [[np.nan, 0., 0.]] * 3, "finite"),
    ("comparator", [[0., 0.]] * 5, "same width"),
])
def test_invalid_inputs_raise_before_bootstrapping(key, value, match):
    arguments = {**example(), "n_bootstrap": 2, key: value}
    with pytest.raises(ValueError, match=match):
        paired_condition_macro_meter_bootstrap(**arguments)


def test_single_interval_is_rejected_because_ramps_are_unavailable():
    data = example()
    for method in ("real", "generated", "comparator"):
        data[method] = data[method][:, :1]
    with pytest.raises(ValueError, match="at least two intervals"):
        paired_condition_macro_meter_bootstrap(**data, n_bootstrap=2)
