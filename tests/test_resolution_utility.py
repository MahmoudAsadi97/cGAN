"""Synthetic-only common-hourly utility and frozen-error bootstrap checks."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

from fluvius_cgan.calibration import ConditionalMarginalCalibrator
from fluvius_cgan.utility import RidgeProfileForecaster

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_module("evaluate_resolution_utility", ROOT / "scripts/evaluate_resolution_utility.py")
pair_tests = load_module("resolution_utility_pair_fixtures", ROOT / "tests/test_compare_resolutions.py")
experiment = pair_tests.experiment
forbid_test_access = pair_tests.forbid_test_access


def test_blocked_native_map_then_sum_and_hash_preserve_order_and_float64_24():
    class FixedMap:
        def __init__(self):
            self.calls = []

        def transform(self, values, labels):
            self.calls.append((len(values), labels.copy()))
            return values.astype(np.float64) ** 2 + labels[:, None] / 32 + np.arange(values.shape[1]) / 1000

    labels = np.array([0, 8, 4, 0, 8])
    values = np.random.default_rng(1).normal(size=(5, 96)).astype(np.float32)
    mapping = FixedMap()
    actual, native_hash = evaluation._hourly_training(values, labels, 96, mapping, batch_profiles=2)
    expected_native = values.astype(np.float64) ** 2 + labels[:, None] / 32 + np.arange(96) / 1000
    np.testing.assert_array_equal(actual, evaluation.hourly_sum(expected_native))
    assert actual.dtype == np.float32
    assert native_hash == evaluation._profile_digest(expected_native)
    assert [size for size, _ in mapping.calls] == [2, 2, 1]
    np.testing.assert_array_equal(np.concatenate([part for _, part in mapping.calls]), labels)
    raw, raw_hash = evaluation._hourly_training(values, labels, 96, batch_profiles=2)
    np.testing.assert_array_equal(raw, evaluation.hourly_sum(values))
    assert raw_hash == evaluation._profile_digest(values)
    direct = values[:, :24]
    mapped24, _ = evaluation._hourly_training(direct, labels, 24, FixedMap(), batch_profiles=2)
    assert mapped24.dtype == np.float64
    np.testing.assert_array_equal(mapped24, direct.astype(np.float64) ** 2 + labels[:, None] / 32 + np.arange(24) / 1000)
    raw24, _ = evaluation._hourly_training(direct, labels, 24, batch_profiles=2)
    np.testing.assert_array_equal(raw24, direct)
    assert np.any(raw24 < 0)


def test_full_train_hourly_fits_saved_maps_errors_bootstrap_and_replay(experiment, forbid_test_access, monkeypatch):
    import fluvius_cgan.calibration as calibration

    monkeypatch.setattr(calibration, "fit", lambda *a, **k: pytest.fail("utility evaluation refitted a map"))
    train, val = experiment["bundles"][24][:2]
    original_fit = RidgeProfileForecaster.fit
    original_generate = evaluation._generate_verified
    original_copula = evaluation.fit_conditional_copula
    original_bootstrap = evaluation.paired_utility_meter_bootstrap
    fitted_pools, generated_pools, bootstrap_calls, copula_calls = [], [], [], []

    def fit_spy(self, values, labels):
        assert np.asarray(values).shape == train["x"].shape
        np.testing.assert_array_equal(labels, train["y"])
        assert self.lags == 3 and self.batch_profiles == 256
        fitted_pools.append(np.asarray(values).copy())
        return original_fit(self, values, labels)

    def generate_spy(path, details, labels, seed):
        np.testing.assert_array_equal(labels, train["y"])
        assert seed == 2026
        result = original_generate(path, details, labels, seed)
        generated_pools.append(result.copy())
        return result

    def copula_spy(values, labels, **kwargs):
        np.testing.assert_array_equal(values, train["x"])
        np.testing.assert_array_equal(labels, train["y"])
        assert kwargs == {"quantile_knots": 1025, "shrinkage": .01}
        copula_calls.append(kwargs)
        return original_copula(values, labels, **kwargs)

    def bootstrap_spy(candidate, comparator, labels, meter_ids, **kwargs):
        assert kwargs == {"n_bootstrap": 500, "confidence": .95, "seed": 42027}
        bootstrap_calls.append((candidate.copy(), comparator.copy(), labels.copy(), meter_ids.copy()))
        return original_bootstrap(candidate, comparator, labels, meter_ids, **kwargs)

    monkeypatch.setattr(RidgeProfileForecaster, "fit", fit_spy)
    monkeypatch.setattr(evaluation, "_generate_verified", generate_spy)
    monkeypatch.setattr(evaluation, "fit_conditional_copula", copula_spy)
    monkeypatch.setattr(evaluation, "paired_utility_meter_bootstrap", bootstrap_spy)
    options = {"map96": experiment["maps"][96], "map24": experiment["maps"][24],
               "candidate96": experiment["checkpoints"][96], "candidate_map96": experiment["maps"][96],
               "samples_per_condition": 2}
    result = evaluation.evaluate(*pair_tests.arguments(experiment), **options)
    replay = evaluation.evaluate(*pair_tests.arguments(experiment), **options)
    for key in ("training", "methods", "pipelines", "paired_uncertainty", "validation_selection"):
        assert result[key] == replay[key]
    assert len(generated_pools) == 6  # Once per model per replay, never per map chunk.
    assert len(copula_calls) == 2
    names = list(result["training"])
    assert names == ["training_real24", "bce96_raw", "bce96_calibrated", "bce24_raw", "bce24_calibrated",
                     "candidate96_raw", "candidate96_calibrated", "conditional_resampling24", "gaussian_copula24"]
    pools = dict(zip(names, fitted_pools[:len(names)]))
    selected = np.array(result["validation_selection"]["indices"])
    vx, vy = val["x"][selected], val["y"][selected]
    manual_rng = np.random.default_rng(2026)
    np.testing.assert_array_equal(selected, np.concatenate([
        manual_rng.choice(np.flatnonzero(val["y"] == label), 2, replace=False) for label in np.unique(val["y"])]))
    np.testing.assert_array_equal(pools["training_real24"], train["x"])
    for index, width in enumerate((96, 24)):
        raw = generated_pools[index]
        mapping = ConditionalMarginalCalibrator.load(experiment["maps"][width])
        mapped = mapping.transform(raw, train["y"])
        expected = evaluation.hourly_sum(mapped) if width == 96 else mapped
        np.testing.assert_array_equal(pools[f"bce{width}_calibrated"], expected)
    manual_errors = {}
    for name, pool in pools.items():
        context = nullcontext() if threadpool_limits is None else threadpool_limits(limits=1, user_api="blas")
        with context:
            model = original_fit(RidgeProfileForecaster(lags=3, batch_profiles=256), pool, train["y"])
            manual = model.evaluate(vx, vy)
            manual_errors[name] = model.predict(vx, vy) - vx[:, 3:].astype(np.float64)
        persistence = manual.pop("persistence")
        assert result["methods"][name] == manual
        assert result["methods"]["persistence"] == persistence
        assert result["training"][name]["coefficients_sha256"] == evaluation._digest(model.coef_, "<f8")
        assert result["training"][name]["profiles_sha256"] == evaluation._profile_digest(pool)
        assert result["training"][name]["validation_errors_sha256"] == evaluation._profile_digest(manual_errors[name])
    expected_pairs = [("bce24_raw", "bce96_raw"), ("bce24_calibrated", "bce96_calibrated")]
    expected_pairs += [(method, control) for method in ("bce96_calibrated", "bce24_calibrated", "candidate96_calibrated")
                       for control in ("training_real24", "conditional_resampling24")]
    assert len(bootstrap_calls) == 2 * len(expected_pairs)
    assert list(result["paired_uncertainty"]) == [a + "_minus_" + b for a, b in expected_pairs]
    for (candidate, comparator), (first, second, labels, meters) in zip(expected_pairs, bootstrap_calls):
        np.testing.assert_array_equal(first, manual_errors[candidate])
        np.testing.assert_array_equal(second, manual_errors[comparator])
        np.testing.assert_array_equal(labels, vy)
        np.testing.assert_array_equal(meters, val["meter_id"][selected])
        ci = result["paired_uncertainty"][candidate + "_minus_" + comparator]["bootstrap"]
        assert ci["windows_per_profile"] == 21 and ci["n_bootstrap"] == 500
        for metric in ("rmse_kwh", "mae_kwh"):
            assert ci["point_estimates"]["candidate"][metric] == pytest.approx(result["methods"][candidate]["macro"][metric])
            assert ci["point_estimates"]["comparator"][metric] == pytest.approx(result["methods"][comparator]["macro"][metric])
    assert result["task"]["forecast_horizon_hours"] == 1.
    assert result["task"]["lags"] == 3 and len(result["task"]["feature_names"]) == 72
    assert all(fit["n_profiles"] == len(train["x"]) and fit["n_windows"] == 21 * len(train["x"])
               for fit in result["training"].values())
    assert result["utility_training_selection"]["cap_applied"] is False
    assert result["test_used"] is False and result["split"] == "val"
    json.dumps(result, allow_nan=False)


def test_wrong_map_fails_before_generation_or_any_ridge_fit(experiment, forbid_test_access, monkeypatch):
    monkeypatch.setattr(evaluation, "_generate_verified", lambda *a, **k: pytest.fail("generation before map verification"))
    monkeypatch.setattr(RidgeProfileForecaster, "fit", lambda *a, **k: pytest.fail("ridge fit before map verification"))
    with pytest.raises(ValueError, match="calibration"):
        evaluation.evaluate(*pair_tests.arguments(experiment), map96=experiment["maps"][96], map24=experiment["maps"][96])


@pytest.mark.parametrize("kwargs", [{"samples_per_condition": 0}, {"samples_per_condition": True},
                                    {"seed": -1}, {"seed": 2**32}, {"map96": "one-map"},
                                    {"candidate_map96": "orphan-map"}])
def test_invalid_options_fail_before_prepared_reads(monkeypatch, kwargs):
    monkeypatch.setattr(evaluation, "_prepared_pair", lambda *a: pytest.fail("invalid options read prepared files"))
    with pytest.raises(ValueError):
        evaluation.evaluate("a", "b", "c", "d", **kwargs)


def test_cli_has_no_test_or_training_cap_and_never_overwrites(tmp_path):
    required = ["--prepared96", "a", "--prepared24", "b", "--checkpoint96", "c", "--checkpoint24", "d"]
    for forbidden in (["--split", "test"], ["--max-train-profiles", "8"]):
        with pytest.raises(SystemExit):
            evaluation.main(required + ["--output", str(tmp_path / "unused.json")] + forbidden)
    output = tmp_path / "existing.json"
    output.write_text("retained")
    with pytest.raises(SystemExit):
        evaluation.main(required + ["--output", str(output)])
    assert output.read_text() == "retained"
