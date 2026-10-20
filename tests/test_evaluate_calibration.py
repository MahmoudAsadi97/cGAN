"""Synthetic integration checks for independent, fixed marginal calibration.

The reference checkpoint contains provenance only. A deterministic generation
stub exercises the workflow without GAN training or real prepared-data access.
"""

import builtins
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import wasserstein_distance
import torch

from fluvius_cgan.battery import simulate_battery
from fluvius_cgan.calibration import ConditionalMarginalCalibrator, fit
from fluvius_cgan.copula import fit_conditional_copula
from fluvius_cgan.evaluation import evaluate_conditional, nearest_neighbor_diagnostics, resample_baseline
from fluvius_cgan.joint import conditional_projected_wasserstein
from fluvius_cgan.training import CHECKPOINT_FORMAT, _config


SPEC = importlib.util.spec_from_file_location(
    "evaluate_calibration", Path(__file__).resolve().parents[1] / "scripts" / "evaluate_calibration.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def digest(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("calibration-evaluation")
    prepared = root / "prepared"
    prepared.mkdir()
    rng = np.random.default_rng(61)
    arrays, manifest = {}, {}
    for split in ("train", "val"):
        arrays[split] = {
            "x": rng.normal(size=(16, 24)).astype(np.float32),
            "y": np.tile([0, 0, 4, 4], 4).astype(np.int64),
            "meter_id": np.repeat([f"{split}-{i}" for i in range(4)], 4),
            "date": np.tile(["2024-01-01", "2024-01-02", "2024-01-06", "2024-01-07"], 4),
        }
        path = prepared / f"{split}.npz"
        np.savez(path, **arrays[split])
        manifest[split] = {
            "profiles": 16, "meters": 4,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "meter_ids": sorted(set(arrays[split]["meter_id"].tolist())),
        }
    (prepared / "test.npz").write_bytes(b"sealed synthetic test sentinel")
    scaler = {"minimum": float(arrays["train"]["x"].min()),
              "maximum": float(arrays["train"]["x"].max())}
    metadata = {"resolution": 24, "target": "net", "units": "kWh/interval",
                "source_sha256": "a" * 64, "split_seed": 42, "splits": manifest}
    (prepared / "scaler.json").write_text(json.dumps(scaler))
    (prepared / "metadata.json").write_text(json.dumps(metadata))
    selected = np.sort(np.random.default_rng(7).choice(16, 12, replace=False))
    state = {
        "format_version": CHECKPOINT_FORMAT, "epoch": 1,
        "config": _config({"seed": 7, "max_train_samples": 12}),
        "metadata": metadata, "scaler": scaler, "output_dim": 24,
        "data_hashes": {
            name: hashlib.sha256((prepared / name).read_bytes()).hexdigest()
            for name in ("train.npz", "val.npz", "metadata.json", "scaler.json")
        },
        "selection": {"train_profiles_available": 16, "train_profiles_used": 12,
                      "train_indices_sha256": digest(selected, "<i8")},
        "supported_labels": np.unique(arrays["train"]["y"][selected]).tolist(),
    }
    checkpoint = root / "reference.pt"
    torch.save(state, checkpoint)
    return prepared, checkpoint, arrays, selected


@pytest.fixture
def forbid_test_access(monkeypatch):
    path_open, builtin_open, numpy_load = Path.open, builtins.open, np.load

    def check(path):
        if isinstance(path, (str, bytes, Path)):
            name = Path(path.decode() if isinstance(path, bytes) else path).name
            assert name != "test.npz", "test split must not be opened, loaded, or hashed"

    def guarded_path_open(path, *args, **kwargs):
        check(path)
        return path_open(path, *args, **kwargs)

    def guarded_builtin_open(path, *args, **kwargs):
        check(path)
        return builtin_open(path, *args, **kwargs)

    def guarded_numpy_load(path, *args, **kwargs):
        check(path)
        return numpy_load(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(np, "load", guarded_numpy_load)


def synthetic_generated(labels, seed):
    labels = np.asarray(labels)
    return (np.random.default_rng(seed).normal(size=(len(labels), 24))
            + labels[:, None] / 8 + np.linspace(-.2, .2, 24))


@pytest.fixture
def generation_calls(monkeypatch):
    calls = []

    def generate(checkpoint, labels, seed=42):
        calls.append({"checkpoint": Path(checkpoint), "labels": np.array(labels, copy=True), "seed": seed})
        return synthetic_generated(labels, seed)

    monkeypatch.setattr(evaluation, "generate", generate)
    return calls


def selected_validation(arrays, seed, cap):
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.choice(np.flatnonzero(arrays["val"]["y"] == label),
                   min(cap, int(np.sum(arrays["val"]["y"] == label))), replace=False)
        for label in np.unique(arrays["val"]["y"])
    ])


@pytest.mark.parametrize("knots", [None, 5])
def test_train_only_selection_independent_draws_and_all_methods_match_manual_metrics(
        experiment, forbid_test_access, generation_calls, tmp_path, knots):
    prepared, checkpoint, arrays, selected_train = experiment
    seed, calibration_seed, cap, calibration_cap = 23, 71, 3, 12
    calibration_output = tmp_path / "fixed-calibration.npz"
    report = evaluation.evaluate(
        prepared, checkpoint, seed=seed, calibration_seed=calibration_seed,
        samples_per_condition=cap, calibration_samples_per_condition=calibration_cap,
        quantile_knots=knots, calibration_output=calibration_output)
    assert len(generation_calls) == 2
    assert all(call["checkpoint"] == checkpoint for call in generation_calls)
    calibration_call = next(call for call in generation_calls if call["seed"] == calibration_seed)
    evaluation_call = next(call for call in generation_calls if call["seed"] == seed)
    calibration_labels = np.repeat([0, 4], calibration_cap)
    np.testing.assert_array_equal(calibration_call["labels"], calibration_labels)
    selected_val = selected_validation(arrays, seed, cap)
    tx, ty = (arrays["train"][key][selected_train] for key in ("x", "y"))
    vx, vy = (arrays["val"][key][selected_val] for key in ("x", "y"))
    np.testing.assert_array_equal(evaluation_call["labels"], vy)
    calibration_reference = synthetic_generated(calibration_labels, calibration_seed)
    raw = synthetic_generated(vy, seed)
    calibrator = fit(tx, ty, calibration_reference, calibration_labels, quantile_knots=knots)
    generated = {
        "raw_cgan": raw,
        "calibrated_cgan": calibrator.transform(raw, vy),
        "gaussian_copula": fit_conditional_copula(tx, ty, quantile_knots=knots, shrinkage=.01).generate(vy, seed=seed),
        "independent_marginals": fit_conditional_copula(tx, ty, quantile_knots=knots, shrinkage=1.).generate(vy, seed=seed),
        "exact_marginal_copula": fit_conditional_copula(tx, ty, quantile_knots=None, shrinkage=.01).generate(vy, seed=seed),
        "conditional_resampling": resample_baseline(tx, ty, vy, seed=seed),
    }
    assert set(report["methods"]) == set(generated)
    assert report["baselines"]["gaussian_copula"]["quantile_knots"] == knots
    assert report["baselines"]["independent_marginals"]["quantile_knots"] == knots
    assert report["baselines"]["exact_marginal_copula"]["quantile_knots"] is None
    assert report["baselines"]["gaussian_copula"]["shrinkage"] == .01
    assert report["baselines"]["independent_marginals"]["shrinkage"] == 1.
    assert report["baselines"]["exact_marginal_copula"]["shrinkage"] == .01
    for method, values in generated.items():
        assert report["methods"][method] == evaluate_conditional(vx, vy, values, vy, interval_hours=1.)
        assert report["frozen_samples"]["profiles_sha256"][method] == digest(values, "<f8")
        assert report["joint_diagnostics"][method] == conditional_projected_wasserstein(vx, vy, values, vy)
        battery = report["battery_diagnostics"][method]
        real_dispatch = simulate_battery(vx, 1.)
        generated_dispatch = simulate_battery(values, 1.)
        assert battery["scenario"] == real_dispatch["scenario"] == generated_dispatch["scenario"]
        assert battery["scenario_sha256"] == hashlib.sha256(json.dumps(
            battery["scenario"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        for label in np.unique(vy):
            selected = vy == label
            condition = battery["per_condition"][str(label)]
            assert condition["n_real"] == condition["n_generated"] == int(selected.sum())
            for quantity in evaluation.BATTERY_QUANTITIES:
                r = real_dispatch["per_profile"][quantity][selected]
                g = generated_dispatch["per_profile"][quantity][selected]
                stem, unit = quantity.rsplit("_", 1)
                assert condition[f"{stem}_wasserstein_{unit}"] == pytest.approx(wasserstein_distance(r, g))
                assert condition[f"real_mean_{quantity}"] == pytest.approx(r.mean())
                assert condition[f"generated_mean_{quantity}"] == pytest.approx(g.mean())
        for key, value in battery["macro"].items():
            assert value == pytest.approx(np.mean([row[key] for row in battery["per_condition"].values()]))
    assert len({row["scenario_sha256"] for row in report["battery_diagnostics"].values()}) == 1
    assert report["battery_diagnostics_status"] == "computed_on_same_frozen_signed_net_profiles"
    directions = [entry["directions"] for entry in report["joint_diagnostics"].values()]
    assert len({entry["sha256"] for entry in directions}) == 1
    assert all(entry["seed"] == 9182 and entry["count"] == 128 for entry in directions)
    assert report["nearest_neighbors"] == nearest_neighbor_diagnostics(tx, generated["calibrated_cgan"], vx, seed=seed)
    assert report["training_profiles"] == {"prepared": 16, "model": 12, "baseline": 12}
    assert report["checkpoint"]["training_selection"]["train_indices_sha256"] == digest(selected_train, "<i8")
    assert report["validation_selection"]["selected_indices_sha256"] == digest(selected_val, "<i8")
    assert report["frozen_samples"]["real_profiles_sha256"] == digest(vx, "<f8")
    assert report["frozen_samples"]["labels_sha256"] == digest(vy, "<i8")
    provenance = report["calibration"]["provenance"]
    assert provenance["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert provenance["train_indices_sha256"] == digest(selected_train, "<i8")
    assert provenance["calibration_seed"] == calibration_seed
    assert provenance["calibration_generated_profiles_sha256"] == digest(calibration_reference, "<f8")
    assert provenance["calibration_labels_sha256"] == digest(calibration_labels, "<i8")
    restored = ConditionalMarginalCalibrator.load(calibration_output)
    assert restored.provenance == provenance and restored.quantile_knots == knots
    np.testing.assert_array_equal(restored.transform(raw, vy), generated["calibrated_cgan"])
    compact_fitting = report["calibration"]["fitting_diagnostics"]["per_condition"]
    for label in (0, 4):
        condition = compact_fitting[str(label)]
        assert condition["train_count"] == int(np.sum(ty == label))
        assert condition["calibration_count"] == calibration_cap
        assert condition["source_tie_mass_by_slot"] == [0.] * 24
        assert condition["source_largest_atom_mass_by_slot"] == [1 / calibration_cap] * 24
        assert condition["source_knot_plateau_mass_by_slot"] == [0.] * 24
        assert condition["source_constant_slots"] == condition["target_constant_slots"] == []
    expected_diagnostics = calibrator.diagnostics(raw, vy)
    compact_application = report["calibration"]["application_diagnostics"]
    for event in ("source_below", "source_above", "cdf_zero", "cdf_one",
                  "target_lower_endpoint", "target_upper_endpoint"):
        total = 0
        for label, condition in expected_diagnostics["per_condition"].items():
            expected_counts = [slot[f"{event}_count"] for slot in condition["slots"]]
            compact = compact_application["per_condition"][label]
            assert compact["count"] == condition["count"]
            assert compact["slot_event_counts"][event] == expected_counts
            assert compact["fractions_over_profile_slots"][event] == sum(expected_counts) / (condition["count"] * 24)
            total += sum(expected_counts)
        assert compact_application["total_event_counts"][event] == total
        assert compact_application["fractions_over_profile_slots"][event] == total / (len(vy) * 24)
    assert report["calibration"]["calibration_profiles"] == len(calibration_reference)
    assert report["calibration"]["samples_per_condition"] == calibration_cap
    assert "EXPLORATORY" in report["scope"]
    assert report["data_curation_status"] == "unverified_no_curation_metadata"
    assert "diversity_ratio" in report["metric_interpretation"]["mechanically_imposed_marginal_or_diversity_diagnostics"]
    assert report["split"] == "val" and report["test_used"] is False
    assert "test.npz" not in report["prepared_data_hashes"]
    json.dumps(report, allow_nan=False)


def test_battery_diagnostic_is_condition_macro_and_detects_order_with_same_marginals():
    # Unequal condition counts cannot change the equal-condition macro weight.
    real = np.zeros((4, 24))
    real[:, :2] = [-1., 1.]
    generated = real.copy()
    generated[0, :2] = [1., -1.]
    labels = np.array([0, 4, 4, 4])
    result = evaluation._battery_diagnostics(simulate_battery(real, 1.), generated, labels, 1.)
    key = "grid_import_reduction_wasserstein_kwh"
    assert result["per_condition"]["0"][key] == pytest.approx(.9)
    assert result["per_condition"]["4"][key] == 0
    assert result["macro"][key] == pytest.approx(.45)
    assert result["macro"][key] != pytest.approx(.225)  # pooled sample weighting
    assert result["n_conditions"] == 2
    assert result["condition_weighting"] == "equal_over_observed_support"
    assert result["per_condition"]["0"]["end_soc_wasserstein_kwh"] == pytest.approx(np.sqrt(.9))
    json.dumps(result, allow_nan=False)


def test_equal_calibration_and_evaluation_seeds_fail_before_data_access(generation_calls):
    with pytest.raises(ValueError, match="(?i)(seed|independent|distinct)"):
        evaluation.evaluate("missing", "missing", seed=71, calibration_seed=71)
    assert generation_calls == []


@pytest.mark.parametrize("tamper", ["data_hash", "selection_hash"])
def test_reference_binding_failures_precede_generation(
        experiment, forbid_test_access, generation_calls, tmp_path, tamper):
    prepared, checkpoint, _, _ = experiment
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if tamper == "data_hash":
        state["data_hashes"]["val.npz"] = "0" * 64
        match = "fingerprint.*val.npz"
    else:
        state["selection"]["train_indices_sha256"] = "0" * 64
        match = "exact training subset"
    invalid = tmp_path / "invalid.pt"
    torch.save(state, invalid)
    with pytest.raises(ValueError, match=match):
        evaluation.evaluate(prepared, invalid, calibration_samples_per_condition=12)
    assert generation_calls == []


def test_changing_evaluation_batch_does_not_refit_calibration_maps(
        experiment, forbid_test_access, generation_calls, tmp_path):
    prepared, checkpoint, _, _ = experiment
    models = []
    for cap in (2, 5):
        path = tmp_path / f"calibration-cap-{cap}.npz"
        evaluation.evaluate(prepared, checkpoint, samples_per_condition=cap,
                            calibration_samples_per_condition=12, calibration_output=path)
        models.append(ConditionalMarginalCalibrator.load(path))
    query = np.linspace(-3., 3., 4 * 24).reshape(4, 24)
    labels = np.array([4, 0, 4, 0])
    np.testing.assert_array_equal(models[0].transform(query, labels), models[1].transform(query, labels))
    assert models[0].provenance == models[1].provenance
    chunks = np.concatenate([models[0].transform(query[:1], labels[:1]),
                             models[0].transform(query[1:], labels[1:])])
    np.testing.assert_array_equal(chunks, models[0].transform(query, labels))
    assert len(generation_calls) == 4


def test_cli_defaults_optional_map_save_and_report_overwrite_protection(
        experiment, forbid_test_access, generation_calls, tmp_path):
    prepared, checkpoint, _, _ = experiment
    output, calibration_output = tmp_path / "report.json", tmp_path / "calibration.npz"
    args = ["--prepared", str(prepared), "--checkpoint", str(checkpoint),
            "--output", str(output), "--calibration-output", str(calibration_output),
            "--samples-per-condition", "3", "--calibration-samples-per-condition", "12"]
    evaluation.main(args)
    original = output.read_bytes()
    report = json.loads(original)
    assert report["seed"] == 2026
    assert report["calibration"]["provenance"]["calibration_seed"] == 7301
    assert ConditionalMarginalCalibrator.load(calibration_output).quantile_knots == 1025
    assert report["baselines"]["gaussian_copula"]["quantile_knots"] == 1025
    assert report["baselines"]["independent_marginals"]["quantile_knots"] == 1025
    assert report["baselines"]["exact_marginal_copula"]["quantile_knots"] is None
    assert report["validation_selection"]["condition_counts"] == {"0": 3, "4": 3}
    assert {call["seed"] for call in generation_calls} == {7301, 2026}
    with pytest.raises(FileExistsError):
        evaluation.main(args)
    assert output.read_bytes() == original and len(generation_calls) == 2
    with pytest.raises(SystemExit):
        evaluation.main(args + ["--split", "test"])
