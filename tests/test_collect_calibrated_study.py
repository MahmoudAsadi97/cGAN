"""Collect synthetic frozen run artifacts without profiles, inference, or training."""

import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluvius_cgan.calibration import ConditionalMarginalCalibrator, fit
from fluvius_cgan.training import CHECKPOINT_FORMAT, _config


SPEC = importlib.util.spec_from_file_location(
    "collect_calibrated_study", Path(__file__).resolve().parents[1] / "scripts" / "collect_calibrated_study.py")
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def metric_bundle(value, diversity=1.):
    row = {
        "pointwise_wasserstein_kwh": value,
        "daily_energy_wasserstein_kwh": 2 * value,
        "peak_power_wasserstein_kw": value + .5,
        "ramp_wasserstein_kw": value + .25,
        "diversity_ratio": diversity,
        "autocorrelation": {"1": {"wasserstein": value / 10}},
    }
    return {"macro": copy.deepcopy(row), "per_condition": {"0": copy.deepcopy(row), "4": copy.deepcopy(row)}}


@pytest.fixture
def make_run(tmp_path):
    def make(name, *, seed=7, value=2., optional=False, hidden_dim=4,
             evaluation_seed=2026, calibration_seed=7301, knots=1025,
             data_tag="shared", joint=False, direction_seed=44, battery=False, battery_capacity=10.,
             curated=True, training_source_updates=None, evaluation_source_updates=None):
        directory = tmp_path / name
        directory.mkdir()
        config = _config({"seed": seed, "epochs": 3, "batch_size": 4,
                          "hidden_dim": hidden_dim, "latent_dim": 4})
        data_hashes = {key: digest(f"{data_tag}:{key}") for key in
                       ("train.npz", "val.npz", "metadata.json", "scaler.json")}
        checkpoint_data_hashes = {**data_hashes}
        if curated:
            checkpoint_data_hashes["curation_manifest.json"] = digest(f"{data_tag}:curation_manifest.json")
        train_labels = np.repeat([0, 4], 4).astype(np.int64)
        train = np.arange(8 * 24, dtype=float).reshape(8, 24) / 10 - 5
        selection = {"train_profiles_available": 8, "train_profiles_used": 8,
                     "train_indices_sha256": digest(np.arange(8, dtype="<i8").tobytes())}
        scaler = {"minimum": float(train.min()), "maximum": float(train.max())}
        metadata = {"source_sha256": "a" * 64, "resolution": 24,
                    "target": "net", "units": "kWh/interval"}
        if curated:
            metadata["curation"] = {
                "schema_version": 1, "fingerprint_version": 2,
                "byte_comparison_verified": True, "original_split_assignments_preserved": True,
                "identity_claim": "No exact full-trace overlap; household independence is not established",
                "requires_new_training_artifacts": True,
            }
        source_hashes = {f"src/fluvius_cgan/{name}.py": digest(f"source:{name}")
                         for name in ("training", "models", "residual", "evaluation", "copula", "calibration")}
        training_provenance = {"git_commit": "1" * 40, "git_dirty": False,
                               "source_files_sha256": {**source_hashes, **(training_source_updates or {})}}
        evaluation_provenance = {"evaluation_script_sha256": digest("evaluation-script"),
                                 "source_files_sha256": {**source_hashes, **(evaluation_source_updates or {})}}
        raw_provenance = {"evaluation_script_sha256": digest("raw-script"),
                          "source_files_sha256": source_hashes}
        history = [{"epoch": epoch, "validation_score": score, "generator_updates": 2}
                   for epoch, score in enumerate((3., 1., 2.), 1)]
        state = {"format_version": CHECKPOINT_FORMAT, "config": config,
                 "metadata": metadata, "data_hashes": checkpoint_data_hashes, "selection": selection,
                 "provenance": training_provenance,
                 "scaler": scaler, "epoch": 2, "best_epoch": 2,
                 "best_validation_score": 1., "validation_score": 1.,
                 "validation_metric": "mean_interval_wasserstein_normalized",
                 "supported_labels": [0, 4], "output_dim": 24, "condition_dim": 64}
        checkpoint = directory / "best.pt"
        torch.save(state, checkpoint)
        checkpoint_hash = digest(checkpoint.read_bytes())
        summary = {"test_used": False, "data_hashes": checkpoint_data_hashes, "selection": selection,
                   "provenance": training_provenance,
                   "scaler": scaler, "source_sha256": metadata["source_sha256"],
                   "epochs_completed": 3, "best_epoch": 2, "best_validation_score": 1.,
                   "validation_metric": state["validation_metric"], "stopped_early": False,
                   "elapsed_seconds": 1., "generator_updates": 6}
        details = {"sha256": checkpoint_hash, "data_hashes": checkpoint_data_hashes, "config": config,
                   "training_selection": selection, "epoch": 2}
        source_labels = np.repeat([0, 4], 6).astype(np.int64)
        source = np.random.default_rng(calibration_seed).normal(size=(12, 24))
        map_provenance = {
            "checkpoint_sha256": checkpoint_hash,
            "train_indices_sha256": selection["train_indices_sha256"],
            "prepared_data_hashes": data_hashes, "calibration_seed": calibration_seed,
            "calibration_samples_per_condition": 6, "calibration_profiles_total": 12,
            "calibration_generated_profiles_sha256": digest(source.astype("<f8").tobytes()),
            "calibration_labels_sha256": digest(source_labels.astype("<i8").tobytes()),
            "generator_inference_batch_size": 1024, "quantile_knots": knots,
        }
        model = fit(train, train_labels, source, source_labels, quantile_knots=knots, provenance=map_provenance)
        map_path = directory / "calibration.npz"
        model.save(map_path)
        map_hash = digest(map_path.read_bytes())
        methods = {"raw_cgan": metric_bundle(value), "calibrated_cgan": metric_bundle(value - 1),
                   **{method: metric_bundle(3.) for method in
                      ("gaussian_copula", "independent_marginals", "exact_marginal_copula", "conditional_resampling")}}
        raw = {"split": "val", "checkpoint_sha256": checkpoint_hash,
               "provenance": raw_provenance,
               "train_sha256": data_hashes["train.npz"], "evaluation_sha256": data_hashes["val.npz"],
               "seed": evaluation_seed, "samples_per_condition_cap": 3, "profile_count": 6,
               "training_profiles": {"prepared": 8, "model": 8, "baselines": 8},
               "methods": {"cgan": copy.deepcopy(methods["raw_cgan"]),
                           "conditional_resampling": copy.deepcopy(methods["conditional_resampling"])}}
        validation = {"selected_indices_sha256": digest(np.arange(6, dtype="<i8").tobytes()),
                      "samples_per_condition_cap": 3, "condition_counts": {"0": 3, "4": 3}}
        profiles_hashes = {method: digest(f"{data_tag}:{evaluation_seed}:{method}:" +
                                          (str(seed) if method in ("raw_cgan", "calibrated_cgan") else "fixed"))
                           for method in methods}
        frozen = {"labels_sha256": digest(np.repeat([0, 4], 3).astype("<i8").tobytes()),
                  "real_profiles_sha256": digest(f"{data_tag}:real"), "profiles_sha256": profiles_hashes}
        bound = {"split": "val", "test_used": False, "checkpoint": details,
                 "prepared_data_hashes": data_hashes, "source_sha256": metadata["source_sha256"]}
        calibrated = {**copy.deepcopy(bound), "seed": evaluation_seed, "calibration_seed": calibration_seed,
                      "provenance": evaluation_provenance,
                      "methods": methods, "validation_selection": validation, "frozen_samples": frozen,
                      "data_curation_status": ("curation_metadata_present_not_reaudited" if curated
                                               else "unverified_no_curation_metadata"),
                      "curation_metadata": metadata.get("curation"),
                      "calibration": {"saved_model": {"path": str(map_path), "sha256": map_hash},
                                      "provenance": map_provenance, "quantile_knots": knots,
                                      "samples_per_condition": 6, "calibration_profiles": 12},
                      "baselines": {"gaussian_copula": {"quantile_knots": knots, "shrinkage": .01},
                                    "independent_marginals": {"quantile_knots": knots, "shrinkage": 1.},
                                    "exact_marginal_copula": {"quantile_knots": None, "shrinkage": .01},
                                    "conditional_resampling": {"seed": evaluation_seed}}}
        if joint:
            directions = {"seed": direction_seed, "count": 32, "sha256": digest(str(direction_seed))}
            calibrated["joint_diagnostics"] = {
                method: {"directions": directions.copy(),
                         "macro": value / 10 if method in ("raw_cgan", "calibrated_cgan") else .3,
                         "per_condition": {label: {"mean_projected_wasserstein_kwh":
                                                   value / 10 if method in ("raw_cgan", "calibrated_cgan") else .3}
                                           for label in ("0", "4")}}
                for method in methods}
        if battery:
            scenario = {"capacity_kwh": battery_capacity, "initial_soc_fraction": .5}
            scenario_hash = digest(json.dumps(scenario, sort_keys=True, separators=(",", ":")))
            calibrated["battery_diagnostics_status"] = "available"
            calibrated["battery_diagnostics"] = {
                method: {"scenario": scenario.copy(), "scenario_sha256": scenario_hash,
                         "macro": {"grid_import_wasserstein_kwh":
                                   value / 10 if method in ("raw_cgan", "calibrated_cgan") else .3},
                         "per_condition": {label: {"grid_import_wasserstein_kwh":
                                                   value / 10 if method in ("raw_cgan", "calibrated_cgan") else .3,
                                                   "n_real": 3, "n_generated": 3}
                                           for label in ("0", "4")}}
                for method in methods}
        for filename, content in (("config.json", config), ("summary.json", summary), ("history.json", history),
                                  ("validation_report.json", raw), ("calibration_validation.json", calibrated)):
            json_write(directory / filename, content)
        if optional:
            utility = {**copy.deepcopy(bound), "calibration": {"sha256": map_hash, "provenance": map_provenance},
                       "config": {"seed": evaluation_seed},
                       "validation_selection": {**validation, "profiles_sha256": frozen["real_profiles_sha256"],
                                                "labels_sha256": frozen["labels_sha256"]},
                       "methods": {"training_calibrated_cgan": {
                           "macro": {"mae_kwh": value, "rmse_kwh": value + .5},
                           "per_condition": {label: {"mae_kwh": value, "rmse_kwh": value + .5, "n_profiles": 3}
                                             for label in ("0", "4")}}},
                       "training": {"sample_count": 8}, "task": "synthetic next-interval prediction",
                       "source_training_pool": {"actual_indices_sha256": selection["train_indices_sha256"],
                                                "actual_gan_profiles": 8},
                       "utility_training_selection": {"seed": 10}}
            json_write(directory / "utility_validation.json", utility)
            uncertainty = {**copy.deepcopy(bound), "candidate": details, "seed": evaluation_seed,
                           "calibrator": {"sha256": map_hash, "provenance": map_provenance},
                           "validation_selection": validation,
                           "frozen_samples": {"labels_sha256": frozen["labels_sha256"],
                                              "real_profiles_sha256": frozen["real_profiles_sha256"],
                                              "candidate_profiles_sha256": profiles_hashes["calibrated_cgan"],
                                              "comparator_profiles_sha256": profiles_hashes["gaussian_copula"]},
                           "comparator": {"kind": "gaussian_copula"}, "replicates": 20,
                           "methods": {"calibrated_cgan": methods["calibrated_cgan"],
                                       "gaussian_copula": methods["gaussian_copula"]},
                           "paired_comparison": {"intervals": {"pointwise_wasserstein_kwh":
                                                                {"estimate": value - 4, "lower": -5., "upper": 2.}}},
                           "interpretation": "Synthetic per-run interval; retain without pooling."}
            json_write(directory / "uncertainty_gaussian.json", uncertainty)
        # Any attempt to read prepared or held-out profiles is outside this collector's remit.
        for filename in ("train.npz", "val.npz", "test.npz"):
            (directory / filename).write_bytes(b"synthetic profile-access sentinel")
        return directory
    return make


@pytest.fixture
def forbid_profile_access(monkeypatch):
    original_path_open, original_open, original_load = Path.open, builtins.open, np.load

    def check(path):
        if isinstance(path, (str, Path)):
            assert Path(path).name not in {"train.npz", "val.npz", "test.npz"}, "collector must not read profile arrays"

    def path_open(path, *args, **kwargs):
        check(path)
        return original_path_open(path, *args, **kwargs)

    def builtin_open(path, *args, **kwargs):
        check(path)
        return original_open(path, *args, **kwargs)

    def numpy_load(path, *args, **kwargs):
        check(path)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(builtins, "open", builtin_open)
    monkeypatch.setattr(np, "load", numpy_load)


def test_collect_run_preserves_bound_metrics_and_missing_optional_artifacts(make_run):
    directory = make_run("seed7")
    result = collector.collect_run("seed7", directory)
    assert result["training_seed"] == 7 and result["epochs_completed"] == 3
    assert result["best_epoch"] == 2 and result["best_selection_score"] == 1.
    assert result["checkpoint_sha256"] == digest((directory / "best.pt").read_bytes())
    assert result["calibration_sha256"] == digest((directory / "calibration.npz").read_bytes())
    assert result["methods"]["raw_cgan"]["macro"] == metric_bundle(2.)["macro"]
    assert result["methods"]["raw_cgan"]["per_condition"]["0"]["lag_one_acf_wasserstein"] == .2
    assert result["utility"] is result["uncertainty"] is result["joint_diagnostics"] is None
    assert result["battery_diagnostics"] is None
    assert result["identity_audit"] is None
    assert set(result["missing_optional_artifacts"]) == {"utility_validation.json", "uncertainty*.json/comparison*.json"}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("curated", [False, True])
def test_checkpoint_and_prepared_fingerprints_keep_distinct_provenance_layers(make_run, curated):
    directory = make_run("curated" if curated else "uncurated", curated=curated, optional=True)
    source = json.loads((directory / "calibration_validation.json").read_text())
    checkpoint_hashes = source["checkpoint"]["data_hashes"]
    prepared_hashes = source["prepared_data_hashes"]
    assert len(checkpoint_hashes) == (5 if curated else 4)
    assert len(prepared_hashes) == 4
    assert source["calibration"]["provenance"]["prepared_data_hashes"] == prepared_hashes
    collected = collector.collect_study([("run", directory)])
    result = collected["runs"][0]
    assert result["prepared_data_hashes"] == prepared_hashes
    assert result["checkpoint_data_hashes"] == checkpoint_hashes
    assert result["group_settings"]["prepared_data_hashes"] == prepared_hashes
    assert result["group_settings"]["checkpoint_data_hashes"] == checkpoint_hashes
    assert result["calibration_provenance"]["prepared_data_hashes"] == prepared_hashes
    summary = json.loads((directory / "summary.json").read_text())
    raw = json.loads((directory / "validation_report.json").read_text())
    assert result["provenance"] == {"training_checkpoint": summary["provenance"],
                                    "training_summary": summary["provenance"],
                                    "calibration_evaluation": source["provenance"],
                                    "raw_evaluation": raw["provenance"]}
    assert result["curation_metadata"] == source["curation_metadata"]
    assert result["data_curation_status"] == ("curation_metadata_present_not_reaudited" if curated
                                             else "unverified_no_curation_metadata")


@pytest.mark.parametrize("key", ["train.npz", "val.npz", "metadata.json", "scaler.json"])
@pytest.mark.parametrize("location", ["report", "map"])
def test_each_prepared_fingerprint_mismatch_is_rejected_with_curated_checkpoint(make_run, key, location):
    directory = make_run("mismatch")
    path = directory / "calibration_validation.json"
    report = json.loads(path.read_text())
    if location == "report":
        report["prepared_data_hashes"][key] = "0" * 64
    else:
        model_path = directory / "calibration.npz"
        model = ConditionalMarginalCalibrator.load(model_path)
        model.provenance["prepared_data_hashes"][key] = "0" * 64
        model.save(model_path)
        report["calibration"]["provenance"] = model.provenance
        report["calibration"]["saved_model"]["sha256"] = digest(model_path.read_bytes())
    json_write(path, report)
    with pytest.raises(ValueError, match="prepared data fingerprints"):
        collector.collect_run("mismatch", directory)


@pytest.mark.parametrize("filename", ["summary.json", "calibration_validation.json"])
def test_extra_checkpoint_curation_fingerprint_remains_bound(make_run, filename):
    directory = make_run("curation-mismatch")
    path = directory / filename
    document = json.loads(path.read_text())
    hashes = document["data_hashes"] if filename == "summary.json" else document["checkpoint"]["data_hashes"]
    hashes["curation_manifest.json"] = "0" * 64
    json_write(path, document)
    with pytest.raises(ValueError, match="data fingerprints"):
        collector.collect_run("curation-mismatch", directory)


@pytest.mark.parametrize("change", ["summary_provenance", "curation_metadata", "curation_status"])
def test_summary_provenance_and_report_curation_claims_are_bound_to_checkpoint(make_run, change):
    directory = make_run("claims")
    filename = "summary.json" if change == "summary_provenance" else "calibration_validation.json"
    path = directory / filename
    report = json.loads(path.read_text())
    if change == "summary_provenance":
        report["provenance"]["git_commit"] = "2" * 40
    elif change == "curation_metadata":
        report["curation_metadata"]["identity_claim"] = "Independent households established"
    else:
        report["data_curation_status"] = "unverified_no_curation_metadata"
    json_write(path, report)
    with pytest.raises(ValueError, match="provenance|curation"):
        collector.collect_run("claims", directory)


@pytest.mark.parametrize("stage,update_key", [
    ("training_checkpoint", "training_source_updates"),
    ("calibration_evaluation", "evaluation_source_updates"),
])
def test_unrelated_source_additions_are_recorded_without_splitting_core_seed_group(make_run, stage, update_key):
    added_path = "src/fluvius_cgan/battery.py"
    added_hash = digest("new-diagnostic-implementation")
    first = make_run("first", seed=7)
    second = make_run("second", seed=8, **{update_key: {added_path: added_hash}})
    report = collector.collect_study([("first", first), ("second", second)])
    assert len(report["groups"]) == 1
    group = report["groups"][0]
    assert group["n_seeds"] == 2
    differences = group["implementation_comparison"]["stages"][stage]["source_file_differences"]
    assert differences == {added_path: {"first": None, "second": added_hash}}
    assert report["runs"][1]["provenance"][stage]["source_files_sha256"][added_path] == added_hash


@pytest.mark.parametrize("update_key,path", [
    ("training_source_updates", "src/fluvius_cgan/training.py"),
    ("evaluation_source_updates", "src/fluvius_cgan/evaluation.py"),
])
def test_core_implementation_changes_split_seed_groups(make_run, update_key, path):
    first = make_run("first", seed=7)
    second = make_run("second", seed=8, **{update_key: {path: digest("changed-core-implementation")}})
    report = collector.collect_study([("first", first), ("second", second)])
    assert len(report["groups"]) == 2
    assert all(group["n_seeds"] == 1 for group in report["groups"])


def test_valid_optional_reports_remain_per_run_and_no_profile_files_are_read(make_run, request):
    directory = make_run("optional", optional=True)
    # Install the access guards only after the synthetic fixture is written.
    request.getfixturevalue("forbid_profile_access")
    result = collector.collect_run("optional", directory)
    assert result["utility"]["methods"]["training_calibrated_cgan"]["macro"] == {"mae_kwh": 2., "rmse_kwh": 2.5}
    assert result["uncertainty"][0]["file"] == "uncertainty_gaussian.json"
    expected = json.loads((directory / "uncertainty_gaussian.json").read_text())["paired_comparison"]
    assert result["uncertainty"][0]["paired_comparison"] == expected
    assert result["missing_optional_artifacts"] == []


def test_training_seed_aggregation_reports_sample_sd_and_retains_optional_coverage(make_run):
    first = make_run("seed7", seed=7, value=2., optional=True)
    second = make_run("seed8", seed=8, value=4.)
    report = collector.collect_study([("seed7", first), ("seed8", second)])
    assert len(report["runs"]) == 2 and len(report["groups"]) == 1
    group = report["groups"][0]
    assert group["training_seeds"] == [7, 8] and group["n_seeds"] == 2
    stats = group["methods"]["raw_cgan"]["macro"]["pointwise_wasserstein_kwh"]
    assert stats == {"n": 2, "n_defined": 2, "mean": 3., "sample_sd": pytest.approx(np.sqrt(2)),
                     "minimum": 2., "maximum": 4.}
    assert group["methods"]["raw_cgan"]["per_condition"]["0"]["pointwise_wasserstein_kwh"] == stats
    assert group["baseline_replay"]["identical_source_subset"] is True
    assert all(group["baseline_replay"]["equal"].values())
    assert group["optional_coverage"] == {"utility": {"seed7": True, "seed8": False},
                                           "uncertainty": {"seed7": True, "seed8": False}}
    assert group["utility_aggregation"]["methods"] is None
    assert "missing" in group["utility_aggregation"]["reason"]
    assert "paired_comparison" not in group and "confidence" not in stats
    assert report["split"] == "val" and report["test_used"] is False


def test_single_seed_sd_is_undefined_and_missing_metrics_are_not_averaged_away(make_run):
    first = make_run("seed7", seed=7)
    only = collector.collect_study([("seed7", first)])["groups"][0]
    assert only["methods"]["raw_cgan"]["macro"]["pointwise_wasserstein_kwh"]["sample_sd"] is None
    second = make_run("seed8", seed=8)
    path = second / "calibration_validation.json"
    report = json.loads(path.read_text())
    report["methods"]["calibrated_cgan"]["macro"]["diversity_ratio"] = None
    for values in report["methods"]["calibrated_cgan"]["per_condition"].values():
        values["diversity_ratio"] = None
    json_write(path, report)
    group = collector.collect_study([("seed7", first), ("seed8", second)])["groups"][0]
    assert group["methods"]["calibrated_cgan"]["macro"]["diversity_ratio"] == {
        "n": 2, "n_defined": 1, "mean": None, "sample_sd": None, "minimum": None, "maximum": None}


def test_utility_aggregates_only_complete_seed_reports_with_equal_settings(make_run):
    first = make_run("first", seed=7, value=2., optional=True)
    second = make_run("second", seed=8, value=4., optional=True)
    group = collector.collect_study([("first", first), ("second", second)])["groups"][0]
    stats = group["utility_aggregation"]["methods"]["training_calibrated_cgan"]["macro"]["mae_kwh"]
    assert stats["mean"] == 3. and stats["sample_sd"] == pytest.approx(np.sqrt(2))
    assert set(group["utility_aggregation"]["methods"]["training_calibrated_cgan"]["per_condition"]["0"]) == {
        "mae_kwh", "rmse_kwh"}
    path = second / "utility_validation.json"
    report = json.loads(path.read_text())
    report["config"]["ridge_alpha"] = 2.
    json_write(path, report)
    group = collector.collect_study([("first", first), ("second", second)])["groups"][0]
    assert group["utility_aggregation"]["methods"] is None
    assert "heterogeneous" in group["utility_aggregation"]["reason"]


@pytest.mark.parametrize("metric", ["rmse_kwh", "mae_kwh"])
@pytest.mark.parametrize("location", ["macro", "per_condition"])
def test_utility_requires_both_metrics_in_every_method_and_condition(make_run, metric, location):
    directory = make_run("missing-metric", optional=True)
    path = directory / "utility_validation.json"
    report = json.loads(path.read_text())
    other = copy.deepcopy(report["methods"]["training_calibrated_cgan"])
    report["methods"]["training_real"] = other
    row = other["macro"] if location == "macro" else other["per_condition"]["4"]
    del row[metric]
    json_write(path, report)
    with pytest.raises(ValueError, match="utility training_real.*RMSE/MAE"):
        collector.collect_run("missing-metric", directory)


@pytest.mark.parametrize("location", ["macro", "per_condition"])
def test_utility_rejects_different_metric_or_count_keys_across_seeds(make_run, location):
    first = make_run("first", seed=7, optional=True)
    second = make_run("second", seed=8, optional=True)
    path = second / "utility_validation.json"
    report = json.loads(path.read_text())
    method = report["methods"]["training_calibrated_cgan"]
    if location == "macro":
        method["macro"]["bias_kwh"] = .1
        for row in method["per_condition"].values():
            row["bias_kwh"] = .1
    else:
        for row in method["per_condition"].values():
            row["n_intervals"] = 24
    json_write(path, report)
    # Each artifact is valid alone; incompatible reporting schemas cannot be pooled.
    collector.collect_run("second", second)
    with pytest.raises(ValueError, match="utility .*metric keys differ"):
        collector.collect_study([("first", first), ("second", second)])


@pytest.mark.parametrize("difference", [
    {"hidden_dim": 8}, {"evaluation_seed": 2027}, {"calibration_seed": 7302},
    {"knots": 3}, {"data_tag": "other-data"},
])
def test_aggregation_separates_different_training_evaluation_calibration_or_data_settings(make_run, difference):
    first, second = make_run("first", seed=7), make_run("second", seed=8, **difference)
    report = collector.collect_study([("first", first), ("second", second)])
    assert len(report["groups"]) == 2
    assert all(group["n_seeds"] == 1 for group in report["groups"])


@pytest.mark.parametrize("filename", collector.REQUIRED_FILES)
def test_missing_required_artifact_is_rejected_instead_of_skipped(make_run, filename):
    directory = make_run("incomplete")
    (directory / filename).unlink()
    with pytest.raises(ValueError, match="missing"):
        collector.collect_study([("incomplete", directory)])


@pytest.mark.parametrize("change", [
    "config", "summary_data", "summary_selection", "raw_hash", "raw_metric",
    "report_subset", "map_hash", "map_checkpoint", "history_gap", "unfinished_budget",
    "primary_knots", "exact_shrinkage", "optional_map", "optional_frozen", "optional_validation", "nonfinite_metric",
])
def test_inconsistent_frozen_artifacts_are_rejected(make_run, change):
    directory = make_run("invalid", optional=change in ("optional_map", "optional_frozen", "optional_validation"))
    filename = "calibration_validation.json"
    if change == "config":
        filename = "config.json"
    elif change in ("summary_data", "summary_selection"):
        filename = "summary.json"
    elif change in ("raw_hash", "raw_metric"):
        filename = "validation_report.json"
    elif change == "history_gap":
        filename = "history.json"
    elif change in ("optional_map", "optional_validation"):
        filename = "utility_validation.json"
    elif change == "optional_frozen":
        filename = "uncertainty_gaussian.json"
    path = directory / filename
    document = json.loads(path.read_text())
    if change == "config":
        document["hidden_dim"] += 1
    elif change == "summary_data":
        document["data_hashes"]["val.npz"] = "0" * 64
    elif change == "summary_selection":
        document["selection"]["train_profiles_used"] = 7
    elif change == "raw_hash":
        document["checkpoint_sha256"] = "0" * 64
    elif change == "raw_metric":
        document["methods"]["cgan"]["macro"]["pointwise_wasserstein_kwh"] += 1
    elif change == "report_subset":
        document["checkpoint"]["training_selection"]["train_indices_sha256"] = "0" * 64
    elif change == "map_hash":
        document["calibration"]["saved_model"]["sha256"] = "0" * 64
    elif change == "map_checkpoint":
        model = ConditionalMarginalCalibrator.load(directory / "calibration.npz")
        model.provenance["checkpoint_sha256"] = "0" * 64
        model.save(directory / "calibration.npz")
        document["calibration"]["provenance"] = model.provenance
        document["calibration"]["saved_model"]["sha256"] = digest((directory / "calibration.npz").read_bytes())
    elif change == "history_gap":
        document[2]["epoch"] = 4
    elif change == "unfinished_budget":
        history_path, summary_path = directory / "history.json", directory / "summary.json"
        json_write(history_path, json.loads(history_path.read_text())[:2])
        summary = json.loads(summary_path.read_text())
        summary["epochs_completed"] = 2
        json_write(summary_path, summary)
    elif change == "primary_knots":
        document["baselines"]["gaussian_copula"]["quantile_knots"] = None
    elif change == "exact_shrinkage":
        document["baselines"]["exact_marginal_copula"]["shrinkage"] = 1.
    elif change == "optional_map":
        document["calibration"]["sha256"] = "0" * 64
    elif change == "optional_frozen":
        document["frozen_samples"]["comparator_profiles_sha256"] = "0" * 64
    elif change == "optional_validation":
        document["validation_selection"]["profiles_sha256"] = "0" * 64
    else:
        document["methods"]["calibrated_cgan"]["macro"]["pointwise_wasserstein_kwh"] = float("nan")
    json_write(path, document)
    with pytest.raises(ValueError):
        collector.collect_run("invalid", directory)


def test_collection_rejects_artifact_changes_during_reading(make_run, monkeypatch):
    directory = make_run("changing")
    original_load = ConditionalMarginalCalibrator.load

    def changed_load(cls, path):
        model = original_load(path)
        with (directory / "history.json").open("a") as stream:
            stream.write("\n")
        return model

    monkeypatch.setattr(ConditionalMarginalCalibrator, "load", classmethod(changed_load))
    with pytest.raises(ValueError, match="changed during collection"):
        collector.collect_run("changing", directory)


def test_duplicate_training_seeds_and_inconsistent_deterministic_controls_are_rejected(make_run):
    first, second = make_run("first", seed=7), make_run("second", seed=7)
    with pytest.raises(ValueError, match="duplicate training seed"):
        collector.collect_study([("first", first), ("second", second)])
    third = make_run("third", seed=8)
    path = third / "calibration_validation.json"
    report = json.loads(path.read_text())
    report["frozen_samples"]["profiles_sha256"]["gaussian_copula"] = "0" * 64
    json_write(path, report)
    with pytest.raises(ValueError, match="baseline hashes"):
        collector.collect_study([("first", first), ("third", third)])


def test_optional_projected_joint_metrics_aggregate_only_with_matching_directions(make_run):
    first = make_run("first", seed=7, value=2., joint=True)
    second = make_run("second", seed=8, value=4., joint=True)
    report = collector.collect_study([("first", first), ("second", second)])
    assert len(report["groups"]) == 1
    stats = report["groups"][0]["joint_diagnostics"]["calibrated_cgan"]["macro"]
    assert stats["mean"] == pytest.approx(.3) and stats["sample_sd"] == pytest.approx(np.sqrt(.02))
    assert stats["n"] == stats["n_defined"] == 2
    third = make_run("third", seed=9, joint=True, direction_seed=45)
    with pytest.raises(ValueError, match="joint_diagnostics.*settings"):
        collector.collect_study([("first", first), ("third", third)])


def test_projected_joint_diagnostics_reject_inconsistent_method_directions(make_run):
    directory = make_run("joint", joint=True)
    path = directory / "calibration_validation.json"
    report = json.loads(path.read_text())
    report["joint_diagnostics"]["calibrated_cgan"]["directions"]["seed"] = 99
    json_write(path, report)
    with pytest.raises(ValueError, match="directions"):
        collector.collect_run("joint", directory)


def test_battery_metrics_retained_and_aggregated_only_with_matching_scenarios(make_run):
    first = make_run("first", seed=7, value=2., battery=True)
    second = make_run("second", seed=8, value=4., battery=True)
    report = collector.collect_study([("first", first), ("second", second)])
    run_metric = report["runs"][0]["battery_diagnostics"]["calibrated_cgan"]
    assert report["runs"][0]["battery_diagnostics_status"] == "available"
    assert run_metric["scenario"]["capacity_kwh"] == 10.
    assert run_metric["per_condition"]["0"]["n_real"] == 3
    group = report["groups"][0]
    stats = group["battery_diagnostics"]["calibrated_cgan"]["macro"]["grid_import_wasserstein_kwh"]
    assert stats == {"n": 2, "n_defined": 2, "mean": pytest.approx(.3),
                     "sample_sd": pytest.approx(np.sqrt(.02)), "minimum": .2, "maximum": .4}
    assert group["battery_diagnostics"]["calibrated_cgan"]["per_condition"]["4"]["grid_import_wasserstein_kwh"] == stats
    third = make_run("third", seed=9, battery=True, battery_capacity=12.)
    with pytest.raises(ValueError, match="battery_diagnostics.*settings"):
        collector.collect_study([("first", first), ("third", third)])


@pytest.mark.parametrize("missing_first", [False, True])
def test_missing_optional_diagnostics_preserve_core_seed_group_and_block_partial_aggregation(make_run, missing_first):
    available = make_run("available", seed=7, joint=True, battery=True)
    missing = make_run("missing", seed=8)
    pairs = [("available", available), ("missing", missing)]
    if missing_first:
        pairs.reverse()
    report = collector.collect_study(pairs)
    assert len(report["groups"]) == 1
    group = report["groups"][0]
    assert group["n_seeds"] == 2
    for diagnostic in ("joint_diagnostics", "battery_diagnostics"):
        assert group[diagnostic] is None
        coverage = group["diagnostic_coverage"][diagnostic]
        assert coverage["complete"] is False
        assert coverage["available_run_names"] == ["available"]
        assert coverage["missing_run_names"] == ["missing"]
    assert group["methods"]["raw_cgan"]["macro"]["pointwise_wasserstein_kwh"]["n"] == 2


@pytest.mark.parametrize("change", ["hash", "scenario"])
def test_battery_scenario_binding_rejects_mismatches(make_run, change):
    directory = make_run("battery", battery=True)
    path = directory / "calibration_validation.json"
    report = json.loads(path.read_text())
    values = report["battery_diagnostics"]["calibrated_cgan"]
    if change == "hash":
        values["scenario_sha256"] = "0" * 64
    else:
        values["scenario"]["capacity_kwh"] += 1.
        values["scenario_sha256"] = digest(json.dumps(values["scenario"], sort_keys=True, separators=(",", ":")))
    json_write(path, report)
    with pytest.raises(ValueError, match="battery scenario"):
        collector.collect_run("battery", directory)


def test_statistical_controls_are_shared_and_verified_across_different_training_arms(make_run):
    first = make_run("first", hidden_dim=4)
    second = make_run("second", hidden_dim=8)
    report = collector.collect_study([("first", first), ("second", second)])
    assert len(report["groups"]) == 2
    assert len(report["shared_statistical_controls"]) == 1
    controls = report["shared_statistical_controls"][0]
    assert controls["run_names"] == ["first", "second"]
    assert controls["identical_frozen_baselines_verified"] is True
    path = second / "calibration_validation.json"
    artifact = json.loads(path.read_text())
    artifact["methods"]["gaussian_copula"]["macro"]["pointwise_wasserstein_kwh"] += .1
    json_write(path, artifact)
    with pytest.raises(ValueError, match="statistical controls.*inconsistent reported metrics"):
        collector.collect_study([("first", first), ("second", second)])


def test_cli_writes_json_and_markdown_and_refuses_overwrite(make_run, tmp_path):
    directory = make_run("seed7", optional=True, battery=True)
    output, markdown = tmp_path / "study.json", tmp_path / "study.md"
    args = ["--run", f"seed7={directory}", "--output", str(output), "--markdown", str(markdown)]
    collector.main(args)
    original = output.read_bytes(), markdown.read_bytes()
    report = json.loads(original[0])
    assert len(report["runs"]) == len(report["groups"]) == 1
    rendered = markdown.read_text()
    assert "present" in rendered and "sample SD" in rendered
    assert "| RMSE | MAE |" in rendered and "battery scenario distribution fidelity" in rendered
    with pytest.raises(FileExistsError):
        collector.main(args)
    assert (output.read_bytes(), markdown.read_bytes()) == original
    with pytest.raises(SystemExit):
        collector.main(["--run", "missing-separator", "--output", str(tmp_path / "unused.json")])
