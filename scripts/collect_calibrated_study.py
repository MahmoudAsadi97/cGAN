"""Collect explicitly named, completed calibrated validation runs without evaluation.

Every --run is NAME=SEED_DIRECTORY. Required artifacts are config.json,
summary.json, history.json, best.pt, validation_report.json,
calibration_validation.json and calibration.npz. Optional utility_validation.json
and uncertainty*.json/comparison*.json are bound to the same frozen artifacts.
No prepared profiles, held-out test artifacts, inference or model fitting are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluvius_cgan.calibration import ConditionalMarginalCalibrator
from fluvius_cgan.release import sha256_file
from fluvius_cgan.training import CHECKPOINT_FORMAT

METRICS = ("pointwise_wasserstein_kwh", "daily_energy_wasserstein_kwh",
           "peak_power_wasserstein_kw", "ramp_wasserstein_kw",
           "lag_one_acf_wasserstein", "diversity_ratio")
BASELINES = ("gaussian_copula", "independent_marginals", "exact_marginal_copula", "conditional_resampling")
REQUIRED_METHODS = {"raw_cgan", "calibrated_cgan", *BASELINES}
REQUIRED_FILES = ("config.json", "summary.json", "history.json", "best.pt",
                  "validation_report.json", "calibration_validation.json", "calibration.npz")
PREPARED_FILES = ("train.npz", "val.npz", "metadata.json", "scaler.json")
TRAINING_CORE_FILES = ("src/fluvius_cgan/training.py", "src/fluvius_cgan/models.py",
                       "src/fluvius_cgan/residual.py", "src/fluvius_cgan/distribution_loss.py",
                       "src/fluvius_cgan/temporal_loss.py")
EVALUATION_CORE_FILES = ("src/fluvius_cgan/evaluation.py", "src/fluvius_cgan/copula.py", "src/fluvius_cgan/calibration.py")


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value in {path}: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def _digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[key], b[key]) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if _finite(a) and _finite(b):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def _validation(report, name, *, legacy=False):
    _check(report.get("split") == "val", f"{name} is not validation-only")
    _check(report.get("test_used", False if legacy else None) is False, f"{name} used test data or lacks test-use status")


def _compact(values):
    result = {metric: values.get(metric) for metric in METRICS}
    result["lag_one_acf_wasserstein"] = values.get("autocorrelation", {}).get("1", {}).get("wasserstein")
    for key, value in result.items():
        _check(value is None or _finite(value), f"invalid numeric metric: {key}")
    return result


def _methods(methods):
    _check(isinstance(methods, dict) and bool(methods), "method reports are missing")
    return {name: {"macro": values["macro"], "joint_and_marginal_macro": _compact(values["macro"]),
                   "per_condition": {label: _compact(row) for label, row in values["per_condition"].items()}}
            for name, values in methods.items()}


def _details(details, state, checkpoint_hash, name):
    _check(details.get("sha256") == checkpoint_hash, f"{name} checkpoint fingerprint mismatch")
    _check(details.get("data_hashes") == state["data_hashes"], f"{name} data fingerprints mismatch")
    _check(details.get("config") == state["config"], f"{name} checkpoint configuration mismatch")
    selected = details.get("training_selection", {})
    for key in ("train_profiles_available", "train_profiles_used", "train_indices_sha256"):
        _check(selected.get(key) == state["selection"].get(key), f"{name} training subset mismatch: {key}")
    _check(details.get("epoch") == state["epoch"], f"{name} checkpoint epoch mismatch")


def _bound_report(report, state, checkpoint_hash, name, details_key="checkpoint"):
    _validation(report, name)
    _details(report[details_key], state, checkpoint_hash, name)
    _check(report.get("prepared_data_hashes") == {key: state["data_hashes"][key] for key in PREPARED_FILES},
           f"{name} prepared data fingerprints mismatch")
    _check(report.get("source_sha256") == state["metadata"]["source_sha256"], f"{name} source fingerprint mismatch")


def _frozen_match(report, calibration, name):
    _check(report["validation_selection"]["selected_indices_sha256"] ==
           calibration["validation_selection"]["selected_indices_sha256"], f"{name} validation rows differ")
    _check(report["frozen_samples"]["labels_sha256"] == calibration["frozen_samples"]["labels_sha256"],
           f"{name} validation labels differ")
    _check(report["frozen_samples"]["real_profiles_sha256"] == calibration["frozen_samples"]["real_profiles_sha256"],
           f"{name} validation profiles differ")


def _collect_run(name, directory):
    """Verify one completed run using its saved artifacts, without prepared data."""
    directory = Path(directory)
    _check(isinstance(name, str) and bool(name.strip()), "run name must be nonempty")
    paths = {filename: directory / filename for filename in REQUIRED_FILES}
    for filename, path in paths.items():
        _check(path.is_file(), f"incomplete run {name}: missing {filename}")
    utility_path = directory / "utility_validation.json"
    if utility_path.exists():
        paths[utility_path.name] = utility_path
    uncertainty_paths = sorted(set(directory.glob("uncertainty*.json")) | set(directory.glob("comparison*.json")))
    paths.update({path.name: path for path in uncertainty_paths})
    hashes = {filename: sha256_file(path) for filename, path in paths.items()}
    config, summary, history, raw, calibrated = (_json(paths[filename]) for filename in
        ("config.json", "summary.json", "history.json", "validation_report.json", "calibration_validation.json"))
    state = torch.load(paths["best.pt"], map_location="cpu", weights_only=True)
    _check(isinstance(state, dict) and state.get("format_version") == CHECKPOINT_FORMAT, "unsupported checkpoint format")
    _check({"train.npz", "val.npz", "metadata.json", "scaler.json"} <= state["data_hashes"].keys() and
           all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in state["data_hashes"].values()),
           "checkpoint lacks valid prepared data fingerprints")
    _check(config == state["config"], "saved config differs from checkpoint config")
    prepared_hashes = {key: state["data_hashes"][key] for key in PREPARED_FILES}
    _check(summary.get("test_used") is False, "training summary is not validation-only")
    _check(summary.get("data_hashes") == state["data_hashes"], "summary data fingerprints mismatch")
    _check(summary.get("selection") == state["selection"], "summary training selection mismatch")
    _check(summary.get("scaler") == state["scaler"], "summary scaler mismatch")
    _check(summary.get("source_sha256") == state["metadata"]["source_sha256"], "summary source fingerprint mismatch")
    _check(summary.get("provenance") == state.get("provenance"), "summary/checkpoint source provenance mismatch")
    completed = summary.get("epochs_completed")
    _check(isinstance(completed, int) and not isinstance(completed, bool) and completed > 0,
           "invalid completed epoch count")
    _check(isinstance(history, list) and len(history) == completed and
           [row.get("epoch") for row in history] == list(range(1, completed + 1)), "history is incomplete or epoch counts disagree")
    _check(all(_finite(row.get("validation_score")) for row in history), "history lacks finite selection scores")
    best = min(history, key=lambda row: row["validation_score"])
    _check(summary.get("best_epoch") == state.get("epoch") == state.get("best_epoch") == best["epoch"],
           "best checkpoint epoch disagrees with history")
    _check(_same(summary.get("best_validation_score"), best["validation_score"]) and
           _same(state.get("best_validation_score"), best["validation_score"]) and
           _same(state.get("validation_score"), best["validation_score"]), "best selection score disagrees with history")
    _check(summary.get("validation_metric") == state.get("validation_metric"), "selection metric mismatch")
    budget, patience = config["epochs"], config.get("patience")
    full_budget = completed == budget and summary.get("stopped_early") is False
    valid_early_stop = (completed < budget and summary.get("stopped_early") is True
                       and isinstance(patience, int) and patience > 0 and completed - best["epoch"] >= patience)
    _check(full_budget or valid_early_stop, "partial run: declared training budget or patience termination was not completed")
    selection = state["selection"]
    available, used = selection["train_profiles_available"], selection["train_profiles_used"]
    _check(0 < used <= available and used == min(available, config.get("max_train_samples") or available),
           "training selection size disagrees with configuration")
    indices = np.arange(available, dtype=np.int64)
    if used < available:
        indices = np.sort(np.random.default_rng(config["seed"]).choice(available, used, replace=False))
    _check(_digest(indices) == selection["train_indices_sha256"], "training subset fingerprint cannot be reconstructed")
    checkpoint_hash, map_hash = hashes["best.pt"], hashes["calibration.npz"]
    _validation(raw, "validation_report.json", legacy=True)
    _check(raw["checkpoint_sha256"] == checkpoint_hash, "raw validation checkpoint fingerprint mismatch")
    _check(raw["train_sha256"] == state["data_hashes"]["train.npz"] and
           raw["evaluation_sha256"] == state["data_hashes"]["val.npz"], "raw validation data fingerprints mismatch")
    _bound_report(calibrated, state, checkpoint_hash, "calibration_validation.json")
    _check(calibrated.get("curation_metadata") == state["metadata"].get("curation"),
           "calibration report curation metadata differs from checkpoint metadata")
    expected_curation = ("curation_metadata_present_not_reaudited" if state["metadata"].get("curation")
                         else "unverified_no_curation_metadata")
    _check(calibrated.get("data_curation_status") == expected_curation, "calibration curation status differs from checkpoint")
    _check(REQUIRED_METHODS <= calibrated["methods"].keys(), "calibration report lacks required comparison methods")
    _check(raw["seed"] == calibrated["seed"] and raw["samples_per_condition_cap"] ==
           calibrated["validation_selection"]["samples_per_condition_cap"], "raw and calibrated evaluation settings differ")
    _check(_same(raw["methods"]["cgan"], calibrated["methods"]["raw_cgan"]), "raw cGAN metrics differ across reports")
    _check(_same(raw["methods"]["conditional_resampling"], calibrated["methods"]["conditional_resampling"]),
           "conditional resampling metrics differ across reports")
    calibration = calibrated["calibration"]
    saved = calibration.get("saved_model")
    _check(isinstance(saved, dict) and saved.get("sha256") == map_hash, "saved calibration fingerprint mismatch or map missing")
    model = ConditionalMarginalCalibrator.load(paths["calibration.npz"])
    map_provenance = model.provenance
    _check(map_provenance == calibration["provenance"], "map/report calibration provenance mismatch")
    _check(map_provenance.get("checkpoint_sha256") == checkpoint_hash, "map checkpoint fingerprint mismatch")
    _check(map_provenance.get("prepared_data_hashes") == prepared_hashes, "map prepared data fingerprints mismatch")
    _check(map_provenance.get("train_indices_sha256") == selection["train_indices_sha256"], "map training subset fingerprint mismatch")
    _check(model.quantile_knots == calibration["quantile_knots"] == map_provenance.get("quantile_knots"), "map quantile setting mismatch")
    _check(model.n_features == state["metadata"]["resolution"], "map feature width mismatch")
    _check(model.supported_labels.tolist() == sorted(state["supported_labels"]), "map condition support mismatch")
    _check(sum(part.train_count for part in model.models.values()) == used, "map training profile count mismatch")
    ncal = calibration["samples_per_condition"]
    _check(all(part.calibration_count == ncal for part in model.models.values()) and
           map_provenance.get("calibration_samples_per_condition") == ncal and
           map_provenance.get("calibration_profiles_total") == calibration["calibration_profiles"] == ncal * len(model.models),
           "map calibration sample count mismatch")
    _check(_digest(np.repeat(model.supported_labels, ncal)) == map_provenance.get("calibration_labels_sha256"),
           "map calibration reference labels mismatch")
    _check(map_provenance.get("calibration_seed") == calibrated["calibration_seed"] != calibrated["seed"],
           "calibration and evaluation seeds must be distinct and provenance-bound")
    for method, shrinkage in (("gaussian_copula", .01), ("independent_marginals", 1.)):
        _check(calibrated["baselines"][method]["quantile_knots"] == model.quantile_knots and
               calibrated["baselines"][method]["shrinkage"] == shrinkage, "primary baseline marginal settings mismatch")
    _check(calibrated["baselines"]["exact_marginal_copula"]["quantile_knots"] is None and
           calibrated["baselines"]["exact_marginal_copula"]["shrinkage"] == .01,
           "exact baseline must retain exact margins and shrinkage .01")
    utility = None
    if utility_path.exists():
        report = _json(utility_path)
        _bound_report(report, state, checkpoint_hash, "utility_validation.json")
        _check(report.get("calibration", {}).get("sha256") == map_hash, "utility calibration fingerprint mismatch")
        _check(report["calibration"].get("provenance") == map_provenance, "utility map provenance mismatch")
        _check(report["config"]["seed"] == calibrated["seed"] and
               report["validation_selection"]["selected_indices_sha256"] == calibrated["validation_selection"]["selected_indices_sha256"],
               "utility evaluation seed or validation rows differ")
        _check(report["validation_selection"]["profiles_sha256"] == calibrated["frozen_samples"]["real_profiles_sha256"] and
               report["validation_selection"]["labels_sha256"] == calibrated["frozen_samples"]["labels_sha256"],
               "utility frozen validation profiles or labels differ")
        _check("training_calibrated_cgan" in report["methods"], "utility report omits calibrated method")
        for method, values in report["methods"].items():
            _check({"rmse_kwh", "mae_kwh"} <= values["macro"].keys(), f"utility {method} macro lacks RMSE/MAE")
            _check(all(_finite(value) for value in values["macro"].values()), f"utility {method} macro has invalid metrics")
            _check(all(all(metric in row and _finite(row[metric]) for metric in values["macro"])
                       for row in values["per_condition"].values()), f"utility {method} per-condition metrics differ or lack RMSE/MAE")
        _check(report["source_training_pool"]["actual_indices_sha256"] == selection["train_indices_sha256"] and
               report["source_training_pool"]["actual_gan_profiles"] == used,
               "utility source training pool differs from verified checkpoint subset")
        utility = {"file": utility_path.name, "sha256": hashes[utility_path.name],
                   "config": report["config"], "methods": report["methods"],
                   "training": report["training"], "task": report["task"],
                   "provenance": report.get("provenance"),
                   "source_training_pool": report["source_training_pool"],
                   "utility_training_selection": report["utility_training_selection"]}
    uncertainty = []
    for path in uncertainty_paths:
        report = _json(path)
        _bound_report(report, state, checkpoint_hash, path.name, details_key="candidate")
        _check(report["calibrator"].get("sha256") == map_hash and report["calibrator"].get("provenance") == map_provenance,
               f"{path.name} map fingerprint or provenance mismatch")
        _frozen_match(report, calibrated, path.name)
        comparator = report["comparator"]["kind"]
        _check(report["seed"] == calibrated["seed"], f"{path.name} evaluation seed differs")
        _check(report["frozen_samples"]["candidate_profiles_sha256"] == calibrated["frozen_samples"]["profiles_sha256"]["calibrated_cgan"] and
               report["frozen_samples"]["comparator_profiles_sha256"] == calibrated["frozen_samples"]["profiles_sha256"][comparator],
               f"{path.name} frozen generated profiles differ")
        _check(all(_same(values, calibrated["methods"][method]) for method, values in report["methods"].items()),
               f"{path.name} metrics differ from frozen calibration report")
        uncertainty.append({"file": path.name, "sha256": hashes[path.name], "comparator": report["comparator"],
                            "replicates": report["replicates"], "paired_comparison": report["paired_comparison"],
                            "interpretation": report["interpretation"], "provenance": report.get("provenance")})
    for filename, path in paths.items():
        _check(sha256_file(path) == hashes[filename], f"artifact changed during collection: {filename}")
    joint = calibrated.get("joint_diagnostics")
    directions = None
    if joint is not None:
        _check(joint.keys() == calibrated["methods"].keys(), "projected joint diagnostic method coverage differs")
        for method, values in joint.items():
            if directions is None:
                directions = values["directions"]
            _check(values["directions"] == directions, "projected joint diagnostic directions differ across methods")
            _check(_finite(values["macro"]), "invalid projected joint macro metric")
            _check(values["per_condition"].keys() == calibrated["methods"][method]["per_condition"].keys(),
                   "projected joint diagnostic condition coverage differs")
            _check(all(_finite(row["mean_projected_wasserstein_kwh"]) for row in values["per_condition"].values()),
                   "invalid projected joint condition metric")
    battery = calibrated.get("battery_diagnostics")
    scenario = None
    if battery:
        _check(battery.keys() == calibrated["methods"].keys(), "battery diagnostic method coverage differs")
        for method, values in battery.items():
            if scenario is None:
                scenario = values["scenario"]
            _check(values["scenario"] == scenario, "battery scenarios differ across methods")
            canonical = json.dumps(values["scenario"], sort_keys=True, separators=(",", ":"), allow_nan=False)
            _check(hashlib.sha256(canonical.encode()).hexdigest() == values["scenario_sha256"],
                   "battery scenario fingerprint mismatch")
            _check(all(_finite(value) for value in values["macro"].values()), "invalid battery macro metric")
            _check(values["per_condition"].keys() == calibrated["methods"][method]["per_condition"].keys(),
                   "battery diagnostic condition coverage differs")
            _check(all(all(_finite(row.get(metric)) for metric in values["macro"])
                       for row in values["per_condition"].values()), "invalid battery condition metric")
    training_sources = (state.get("provenance") or {}).get("source_files_sha256", {})
    evaluation_sources = (calibrated.get("provenance") or {}).get("source_files_sha256", {})
    grouping = {
        "training_config_except_seed": {key: value for key, value in config.items() if key != "seed"},
        "prepared_data_hashes": prepared_hashes,
        "checkpoint_data_hashes": state["data_hashes"],
        "training_core_files_sha256": {key: training_sources.get(key) for key in TRAINING_CORE_FILES},
        "evaluation_core_files_sha256": {key: evaluation_sources.get(key) for key in EVALUATION_CORE_FILES},
        "evaluation_seed": calibrated["seed"],
        "samples_per_condition": calibrated["validation_selection"]["samples_per_condition_cap"],
        "calibration_seed": calibrated["calibration_seed"], "calibration_samples_per_condition": ncal,
        "quantile_knots": model.quantile_knots,
        "baselines": calibrated["baselines"],
        "validation_indices_sha256": calibrated["validation_selection"]["selected_indices_sha256"],
        "validation_labels_sha256": calibrated["frozen_samples"]["labels_sha256"],
        "validation_profiles_sha256": calibrated["frozen_samples"]["real_profiles_sha256"],
    }
    return {
        "name": name, "directory": str(directory), "training_seed": config["seed"], "config": config,
        "group_settings": grouping, "artifact_sha256": hashes, "checkpoint_sha256": checkpoint_hash,
        "calibration_sha256": map_hash, "prepared_data_hashes": prepared_hashes,
        "checkpoint_data_hashes": state["data_hashes"],
        "provenance": {"training_checkpoint": state.get("provenance"), "training_summary": summary.get("provenance"),
                       "calibration_evaluation": calibrated.get("provenance"), "raw_evaluation": raw.get("provenance")},
        "diagnostic_settings": {"joint_projection_directions": directions, "battery_scenario": scenario},
        "training_selection": selection, "best_epoch": summary["best_epoch"],
        "epochs_completed": completed, "stopped_early": summary["stopped_early"],
        "selection_metric": summary["validation_metric"], "best_selection_score": summary["best_validation_score"],
        "elapsed_seconds": summary.get("elapsed_seconds"), "generator_updates": summary.get("generator_updates"),
        "data_curation_status": calibrated.get("data_curation_status"), "curation_metadata": calibrated.get("curation_metadata"),
        "identity_audit": None, "validation_selection": calibrated["validation_selection"],
        "frozen_samples": calibrated["frozen_samples"], "calibration_provenance": map_provenance,
        "methods": _methods(calibrated["methods"]), "raw_validation_methods": _methods(raw["methods"]),
        "joint_diagnostics": joint,
        "battery_diagnostics": battery,
        "battery_diagnostics_status": calibrated.get("battery_diagnostics_status"),
        "battery_diagnostics_note": calibrated.get("battery_diagnostics_note"),
        "nearest_neighbors": calibrated.get("nearest_neighbors"),
        "nearest_neighbor_method": calibrated.get("nearest_neighbor_method"),
        "utility": utility, "uncertainty": uncertainty or None,
        "missing_optional_artifacts": (["utility_validation.json"] if utility is None else []) +
                                      (["uncertainty*.json/comparison*.json"] if not uncertainty else []),
    }


def collect_run(name, directory):
    """Reject malformed or incomplete artifact schemas with a run-specific error."""
    try:
        return _collect_run(name, directory)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"incomplete or malformed artifacts for run {name}: {error}") from error


def _stats(values):
    _check(all(value is None or _finite(value) for value in values), "invalid numeric aggregate value")
    defined = [value for value in values if value is not None]
    complete = len(defined) == len(values)
    return {"n": len(values), "n_defined": len(defined),
            "mean": statistics.mean(defined) if complete else None,
            "sample_sd": statistics.stdev(defined) if complete and len(defined) > 1 else None,
            "minimum": min(defined) if complete else None, "maximum": max(defined) if complete else None}


def _utility_aggregate(members):
    available = [run for run in members if run["utility"] is not None]
    result = {"methods": None, "settings": None, "reason": None,
              "available_run_names": [run["name"] for run in available]}
    if len(available) != len(members):
        result["reason"] = "missing_utility_reports; no partial-seed aggregate"
        return result
    settings = [{"config": {key: value for key, value in run["utility"]["config"].items() if key != "calibrator"},
                 "task": run["utility"]["task"]} for run in members]
    if not all(setting == settings[0] for setting in settings):
        result["reason"] = "heterogeneous_utility_settings; per-run results retained separately"
        return result
    first = members[0]["utility"]["methods"]
    _check(all(run["utility"]["methods"].keys() == first.keys() for run in members),
           "utility method coverage differs despite identical utility settings")
    aggregate = {}
    for method, values in first.items():
        metrics, conditions = values["macro"], values["per_condition"]
        _check({"rmse_kwh", "mae_kwh"} <= metrics.keys(), "utility macro lacks required RMSE/MAE metrics")
        _check(all(run["utility"]["methods"][method]["macro"].keys() == metrics.keys() for run in members),
               "utility macro metric keys differ")
        _check(all(run["utility"]["methods"][method]["per_condition"].keys() == conditions.keys() for run in members),
               "utility condition coverage differs")
        _check(all(all(metric in row for metric in metrics) for run in members
                   for row in run["utility"]["methods"][method]["per_condition"].values()),
               "utility per-condition metric keys differ or omit RMSE/MAE")
        _check(all(run["utility"]["methods"][method]["per_condition"][label].keys() == conditions[label].keys()
                   for run in members for label in conditions), "utility per-condition metric keys differ")
        aggregate[method] = {
            "macro": {metric: _stats([run["utility"]["methods"][method]["macro"].get(metric) for run in members])
                      for metric in metrics},
            "per_condition": {label: {metric: _stats([run["utility"]["methods"][method]["per_condition"][label].get(metric)
                                                      for run in members]) for metric in metrics} for label in conditions},
        }
    result.update(methods=aggregate, settings=settings[0], reason="all_seed_reports_have_identical_utility_settings")
    return result


def _diagnostic_coverage(members, field, settings_key):
    available = [run for run in members if run[field]]
    settings = available[0]["diagnostic_settings"][settings_key] if available else None
    _check(all(run["diagnostic_settings"][settings_key] == settings for run in available),
           f"incompatible nonmissing {field} settings within one core seed group")
    return available, settings, {
        "complete": len(available) == len(members),
        "available_run_names": [run["name"] for run in available],
        "missing_run_names": [run["name"] for run in members if not run[field]],
        "aggregation_policy": "Only complete core-group seed coverage is aggregated; missing diagnostics never split the core arm.",
    }


def _implementation_comparison(members):
    compared = {}
    for stage in ("training_checkpoint", "calibration_evaluation"):
        sources = {run["name"]: (run["provenance"][stage] or {}).get("source_files_sha256", {}) for run in members}
        paths = sorted({key for values in sources.values() for key in values})
        differences = {path: {name: values.get(path) for name, values in sources.items()}
                       for path in paths if len({values.get(path) for values in sources.values()}) > 1}
        compared[stage] = {"source_file_differences": differences,
                           "runner_fingerprint_differences": {
                               key: {run["name"]: (run["provenance"][stage] or {}).get(key) for run in members}
                               for key in ("evaluation_script_sha256", "comparison_helper_sha256")
                               if len({(run["provenance"][stage] or {}).get(key) for run in members}) > 1},
                           "source_provenance_available": {name: bool(values) for name, values in sources.items()}}
    return {"stages": compared,
            "grouping_policy": "Core training/model and primary evaluation module hashes enter group identity. Other source changes are retained and reported; adding unrelated diagnostic modules does not split a training arm."}


def collect_study(pairs):
    """Keep every requested completed seed; group only identical declared settings."""
    pairs = list(pairs)
    _check(bool(pairs), "at least one named run is required")
    _check(len({name for name, _ in pairs}) == len(pairs), "run names must be unique")
    _check(len({str(Path(path).resolve()) for _, path in pairs}) == len(pairs), "a run directory may be listed only once")
    runs = [collect_run(name, path) for name, path in pairs]
    controls = {}
    for run in runs:
        settings = run["group_settings"]
        identity = {key: settings[key] for key in (
            "prepared_data_hashes", "evaluation_core_files_sha256", "evaluation_seed", "samples_per_condition", "quantile_knots", "baselines",
            "validation_indices_sha256", "validation_labels_sha256", "validation_profiles_sha256")}
        identity["train_indices_sha256"] = run["training_selection"]["train_indices_sha256"]
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        controls.setdefault(key, {"settings": identity, "members": []})["members"].append(run)
    shared_controls = []
    for key, control in controls.items():
        members = control["members"]
        baseline_hashes = {method: {run["name"]: run["frozen_samples"]["profiles_sha256"][method] for run in members}
                           for method in BASELINES}
        _check(all(len(set(values.values())) == 1 for values in baseline_hashes.values()),
               "matched statistical controls have differing baseline hashes across named arms")
        _check(all(_same(run["methods"][method], members[0]["methods"][method])
                   for method in BASELINES for run in members), "matched statistical controls have inconsistent reported metrics")
        shared_controls.append({"control_sha256": key, "settings": control["settings"],
                                "run_names": [run["name"] for run in members], "hashes": baseline_hashes,
                                "identical_frozen_baselines_verified": True})
    groups = {}
    for run in runs:
        canonical = json.dumps(run["group_settings"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        key = hashlib.sha256(canonical.encode()).hexdigest()
        groups.setdefault(key, []).append(run)
    collected_groups = []
    for key, members in groups.items():
        seeds = [run["training_seed"] for run in members]
        _check(len(seeds) == len(set(seeds)), "duplicate training seed within an otherwise identical settings group")
        methods = members[0]["methods"]
        _check(all(run["methods"].keys() == methods.keys() for run in members), "method coverage differs within a group")
        joint_runs, joint_settings, joint_coverage = _diagnostic_coverage(members, "joint_diagnostics", "joint_projection_directions")
        battery_runs, battery_settings, battery_coverage = _diagnostic_coverage(members, "battery_diagnostics", "battery_scenario")
        same_subset = len({run["training_selection"]["train_indices_sha256"] for run in members}) == 1
        baseline_hashes = {name: {run["name"]: run["frozen_samples"]["profiles_sha256"][name] for run in members}
                           for name in BASELINES}
        baseline_equal = {name: len(set(values.values())) == 1 for name, values in baseline_hashes.items()}
        if same_subset:
            _check(all(baseline_equal.values()), "identical training subsets and evaluation settings produced differing baseline hashes")
            _check(all(_same(run["methods"][method], members[0]["methods"][method])
                       for method in BASELINES for run in members),
                   "identical frozen baseline arrays have inconsistent reported metrics")
            if joint_runs:
                _check(all(_same(run["joint_diagnostics"][method], joint_runs[0]["joint_diagnostics"][method])
                           for method in BASELINES for run in joint_runs),
                       "identical frozen baseline arrays have inconsistent projected diagnostics")
            if battery_runs:
                _check(all(_same(run["battery_diagnostics"][method], battery_runs[0]["battery_diagnostics"][method])
                           for method in BASELINES for run in battery_runs),
                       "identical frozen baseline arrays have inconsistent battery diagnostics")
        aggregate = {}
        joint_aggregate = {} if joint_coverage["complete"] else None
        battery_aggregate = {} if battery_coverage["complete"] else None
        for method in methods:
            conditions = methods[method]["per_condition"]
            _check(all(run["methods"][method]["per_condition"].keys() == conditions.keys() for run in members),
                   "condition coverage differs within a group")
            aggregate[method] = {
                "macro": {metric: _stats([run["methods"][method]["joint_and_marginal_macro"][metric] for run in members])
                          for metric in METRICS},
                "per_condition": {label: {metric: _stats([run["methods"][method]["per_condition"][label][metric]
                                                          for run in members]) for metric in METRICS}
                                  for label in conditions},
            }
            if joint_aggregate is not None:
                joint_aggregate[method] = {
                    "macro": _stats([run["joint_diagnostics"][method]["macro"] for run in members]),
                    "per_condition": {label: _stats([run["joint_diagnostics"][method]["per_condition"][label]
                                                      ["mean_projected_wasserstein_kwh"] for run in members])
                                      for label in conditions},
                }
            if battery_aggregate is not None:
                battery_metrics = members[0]["battery_diagnostics"][method]["macro"]
                _check(all(run["battery_diagnostics"][method]["macro"].keys() == battery_metrics.keys()
                           for run in members), "battery metric coverage differs")
                battery_aggregate[method] = {
                    "macro": {metric: _stats([run["battery_diagnostics"][method]["macro"][metric] for run in members])
                              for metric in battery_metrics},
                    "per_condition": {label: {metric: _stats([run["battery_diagnostics"][method]["per_condition"][label][metric]
                                                              for run in members]) for metric in battery_metrics}
                                      for label in conditions},
                }
        collected_groups.append({
            "group_sha256": key, "settings": members[0]["group_settings"],
            "run_names": [run["name"] for run in members], "training_seeds": seeds, "n_seeds": len(seeds),
            "methods": aggregate,
            "joint_diagnostics": joint_aggregate,
            "battery_diagnostics": battery_aggregate,
            "diagnostic_settings": {"joint_projection_directions": joint_settings, "battery_scenario": battery_settings},
            "diagnostic_coverage": {"joint_diagnostics": joint_coverage, "battery_diagnostics": battery_coverage},
            "implementation_comparison": _implementation_comparison(members),
            "utility_aggregation": _utility_aggregate(members),
            "baseline_replay": {"identical_source_subset": same_subset,
                                "identical_frozen_baselines_expected": same_subset,
                                "hashes": baseline_hashes, "equal": baseline_equal,
                                "interpretation": "Repeated deterministic controls are not independent seed replications."},
            "optional_coverage": {"utility": {run["name"]: run["utility"] is not None for run in members},
                                  "uncertainty": {run["name"]: run["uncertainty"] is not None for run in members}},
        })
    return {
        "schema_version": 1, "split": "val", "test_used": False,
        "scope": "Exploratory validation-only calibrated study; all explicitly named completed runs retained.",
        "collection_policy": "No automatic run discovery, filtering, best-seed selection, inference, fitting, or prepared-profile reads.",
        "aggregation": "Mean, sample standard deviation and range across declared training seeds with identical settings. Undefined inputs propagate None; one seed has no sample SD. No generated profiles or confidence intervals are pooled or averaged.",
        "identity_qualification": "Artifact bindings and curation metadata are checked; this collector does not perform identity auditing. Exact-trace curation does not prove household independence or exclude near-duplicates.",
        "interpretation": "Calibration directly targets training marginals and marginal diversity, with finite-reference approximation; improvements in these diagnostics are not independent joint evidence. Validation was used for development and checkpoint selection; unseen-test performance remains unknown.",
        "runs": runs, "groups": collected_groups, "shared_statistical_controls": shared_controls,
        "provenance": {"collector_sha256": sha256_file(Path(__file__))},
    }


def markdown(report):
    lines = ["# Calibrated validation study", "", report["scope"], "", report["identity_qualification"], "",
             report["interpretation"], "", report["aggregation"], "", "## Included runs", "",
             "| Run | Seed | Best epoch | Completed | Utility | Uncertainty |", "|---|---:|---:|---:|---|---|"]
    for run in report["runs"]:
        name = run["name"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {name} | {run['training_seed']} | {run['best_epoch']} | {run['epochs_completed']} | "
                     f"{'present' if run['utility'] is not None else 'MISSING'} | {'present' if run['uncertainty'] is not None else 'MISSING'} |")
    labels = ("Point W1", "Daily W1", "Peak W1", "Ramp W1", "Lag-1 ACF W1", "Diversity")
    def cell(stats):
        if stats["mean"] is None:
            return "undefined"
        sd = "undefined" if stats["sample_sd"] is None else f"{stats['sample_sd']:.5g}"
        return f"{stats['mean']:.5g} ± {sd} [{stats['minimum']:.5g}, {stats['maximum']:.5g}]"
    for index, group in enumerate(report["groups"], 1):
        lines.extend(["", f"## Settings group {index}", "", f"Runs: {', '.join(group['run_names'])}. Training seeds: {group['training_seeds']}.",
                      "", "Cells are mean ± sample SD [minimum, maximum]. Diversity is diagnostic, not a quantity to minimize.", "",
                      "| Method | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)])
        for method, values in group["methods"].items():
            lines.append("| " + method + " | " + " | ".join(cell(values["macro"][metric]) for metric in METRICS) + " |")
        for name, coverage in group["diagnostic_coverage"].items():
            if not coverage["complete"]:
                lines.extend(["", f"{name} aggregation unavailable; missing for: {', '.join(coverage['missing_run_names'])}. "
                              "The core seed group is retained in full."])
        if group["joint_diagnostics"] is not None:
            lines.extend(["", "Secondary fixed whole-profile projections (kWh/projection):", "",
                          "| Method | Projected W1 |", "|---|---:|"])
            lines.extend("| " + method + " | " + cell(values["macro"]) + " |"
                         for method, values in group["joint_diagnostics"].items())
        if group["battery_diagnostics"] is not None:
            first = next(iter(group["battery_diagnostics"].values()))
            metrics = [key for key in first["macro"] if "_wasserstein_" in key]
            lines.extend(["", "Secondary fixed battery scenario distribution fidelity:", "",
                          "| Method | " + " | ".join(key.replace("_", " ") for key in metrics) + " |",
                          "|---|" + "---:|" * len(metrics)])
            lines.extend("| " + method + " | " + " | ".join(cell(values["macro"][key]) for key in metrics) + " |"
                         for method, values in group["battery_diagnostics"].items())
        utility = group["utility_aggregation"]
        if utility["methods"] is None:
            lines.extend(["", "Utility aggregation unavailable: " + utility["reason"] + "."])
        else:
            lines.extend(["", "Within-day observed-history utility (kWh):", "",
                          "| Method | RMSE | MAE |", "|---|---:|---:|"])
            lines.extend("| " + method + " | " + cell(values["macro"]["rmse_kwh"]) + " | " +
                         cell(values["macro"]["mae_kwh"]) + " |" for method, values in utility["methods"].items())
        lines.extend(["", "Frozen baseline hashes " + ("match as expected for identical source subsets."
                      if group["baseline_replay"]["identical_source_subset"] else "are recorded; source training subsets differ across seeds."),
                      "Repeated deterministic controls are not independent seed replications. Per-condition results, bindings and uncombined uncertainty reports are retained in JSON."])
    lines.extend(["", "Missing optional artifacts remain missing; no utility or uncertainty claim is inferred from their absence.", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=SEED_DIRECTORY, repeated for every completed run")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report")
    parser.add_argument("--markdown", type=Path, help="Optional new Markdown summary")
    args = parser.parse_args(argv)
    pairs = [item.split("=", 1) for item in args.run]
    if any(len(pair) != 2 or not all(pair) for pair in pairs):
        parser.error("every --run must be NAME=SEED_DIRECTORY")
    outputs = [args.output] + ([args.markdown] if args.markdown is not None else [])
    _check(len({path.resolve() for path in outputs}) == len(outputs), "JSON and Markdown outputs must differ")
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"output already exists; choose a new path: {path}")
    _check(args.output.suffix.lower() == ".json", "output must have a .json extension")
    if args.markdown is not None:
        _check(args.markdown.suffix.lower() == ".md", "markdown output must have a .md extension")
    report = collect_study(pairs)
    payloads = [json.dumps(report, indent=2, allow_nan=False) + "\n"]
    if args.markdown is not None:
        payloads.append(markdown(report))
    for path, payload in zip(outputs, payloads):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(f"Collected {len(report['runs'])} completed validation runs in {len(report['groups'])} settings groups: {args.output}")


if __name__ == "__main__":
    main()
