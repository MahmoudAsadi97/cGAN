"""Assess the fixed three-seed temporal pilot from saved validation artifacts.

Example (outputs must not exist):
  python scripts/assess_temporal_pilot.py --candidate-root runs/curated_derivative1 \
      --baseline-root runs/curated_swd10 --output runs/temporal_pilot.json \
      --markdown runs/temporal_pilot.md

Reads reports, checkpoint metadata and calibration archives through the existing
collector; never loads prepared train/val/test arrays, generates, fits or selects
a model. Missing/incompatible evidence is retained as an incomplete assessment,
not a failed scientific gate or permission to promote. Exit codes: 0 pass,
1 complete but failed gates, 2 incomplete evidence (or invalid invocation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import statistics
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import collect_calibrated_study as collector
from fluvius_cgan.training import _config

SEEDS = (11, 22, 33)
TRAIN_PROFILES = 606060
EPOCHS, UPDATES, UPDATES_PER_EPOCH = 26, 30784, 1184
CONDITIONS = {str(i) for i in range(64)}
METHOD = "calibrated_cgan"
UTILITY_METHODS = ("training_calibrated_cgan", "training_real", "training_resampling")
CALIBRATION_EVENTS = ("source_below", "source_above", "cdf_zero", "cdf_one",
                      "target_lower_endpoint", "target_upper_endpoint")
GATED_METRICS = ("ramp_wasserstein_kw", "daily_energy_wasserstein_kwh",
                 "peak_power_wasserstein_kw", "lag_one_acf_wasserstein",
                 "projected_profile_wasserstein_kwh")
EVALUATION_CORE = tuple(f"src/fluvius_cgan/{name}.py" for name in
                        ("evaluation", "calibration", "copula", "joint", "battery"))
MODEL_CORE = ("src/fluvius_cgan/models.py", "src/fluvius_cgan/residual.py")
UTILITY_CORE = (*MODEL_CORE, "src/fluvius_cgan/utility.py", "src/fluvius_cgan/calibration.py",
                "src/fluvius_cgan/evaluation.py", "src/fluvius_cgan/copula.py",
                "scripts/evaluate_utility.py", "scripts/compare_checkpoints.py")
check, finite, same = collector._check, collector._finite, collector._same


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _source_files(provenance, required, key="source_files_sha256"):
    values = provenance.get(key, {})
    check(all(_hash(values.get(name)) for name in required), "missing/invalid source fingerprints")
    return {name: values[name] for name in required}


def _macro(values, per_condition, name, *, nonnegative=True):
    check(set(per_condition) == CONDITIONS, f"{name}: all 64 conditions required")
    check(finite(values) and all(finite(value) for value in per_condition.values()),
          f"{name}: finite macro and per-condition values required")
    check(not nonnegative or (values >= 0 and min(per_condition.values()) >= 0), f"{name}: negative error")
    check(same(values, statistics.mean(per_condition.values())), f"{name}: macro is not equal-condition mean")


def _calibration_diagnostics(run, calibrated):
    """Validate the compact report schema emitted by evaluate_calibration.

    The calibrator API has verbose slot dictionaries, but the saved report
    deliberately retains 96-value mass/count vectors instead. Validate those
    vectors and their aggregate arithmetic, without loading any profile data.
    """
    fitting = calibrated["calibration"]["fitting_diagnostics"]
    application = calibrated["calibration"]["application_diagnostics"]
    for name, diagnostic in (("fitting_diagnostics", fitting), ("application_diagnostics", application)):
        check(isinstance(diagnostic, dict) and set(diagnostic.get("per_condition", {})) == CONDITIONS,
              f"calibration {name} must retain all 64 support/tie diagnostics")
    check(application.get("n_profiles") == 6400, "calibration application count differs from primary evaluation")
    totals = dict.fromkeys(CALIBRATION_EVENTS, 0)
    train_counts = run["utility"]["source_training_pool"]["condition_counts"]
    for label in CONDITIONS:
        fitted, applied = fitting["per_condition"][label], application["per_condition"][label]
        check(fitted.get("train_count") == train_counts[label] and fitted.get("calibration_count") == 4096,
              "calibration fitting diagnostic counts differ from bound TRAIN/reference pools")
        for key in ("source_tie_mass_by_slot", "source_largest_atom_mass_by_slot", "source_knot_plateau_mass_by_slot"):
            values = fitted.get(key)
            check(isinstance(values, list) and len(values) == 96 and
                  all(finite(value) and 0 <= value <= 1 for value in values),
                  f"calibration fitting {key} requires 96 finite slot masses in [0,1]")
        for key in ("source_constant_slots", "target_constant_slots"):
            slots = fitted.get(key)
            check(isinstance(slots, list) and all(type(slot) is int and 0 <= slot < 96 for slot in slots)
                  and slots == sorted(set(slots)), f"calibration fitting {key} must list unique valid slot indices")
        check(applied.get("count") == 100 and set(applied.get("slot_event_counts", {})) == set(CALIBRATION_EVENTS)
              and set(applied.get("fractions_over_profile_slots", {})) == set(CALIBRATION_EVENTS),
              "calibration application event coverage/count differs")
        for event in CALIBRATION_EVENTS:
            values = applied["slot_event_counts"][event]
            check(isinstance(values, list) and len(values) == 96 and
                  all(type(value) is int and 0 <= value <= 100 for value in values),
                  f"calibration application {event} requires 96 valid slot counts")
            count = sum(values)
            check(same(applied["fractions_over_profile_slots"][event], count / 9600),
                  f"calibration application {event} per-condition fraction mismatch")
            totals[event] += count
    check(application.get("total_event_counts") == totals, "calibration application global event totals mismatch")
    check(set(application.get("fractions_over_profile_slots", {})) == set(CALIBRATION_EVENTS) and
          all(same(application["fractions_over_profile_slots"][event], totals[event] / (6400 * 96))
              for event in CALIBRATION_EVENTS), "calibration application global fractions mismatch")


def expected_config(arm, seed):
    """Freeze the declared settings; normalize only the historical missing weight."""
    return _config({"epochs": 26, "batch_size": 512, "seed": seed, "latent_dim": 24,
                    "hidden_dim": 64, "lr_g": .0001, "lr_d": .0002, "device": "cpu",
                    "num_threads": 1, "unconditional": False, "validation_samples": 4096,
                    "max_train_samples": None, "normalization": "standard",
                    "discriminator_hidden_dim": 128, "objective": "hinge_sn",
                    "adam_beta1": .5, "adam_beta2": .999, "batch_sampling": "balanced_conditions",
                    "conditions_per_batch": 4, "swd_weight": 10., "swd_projections": 128,
                    "validation_selection": "conditional_w1",
                    "derivative_swd_weight": 1. if arm == "candidate" else 0.})


def _utility_checks(run, report):
    utility, selected = run["utility"], run["training_selection"]
    check(utility is not None, "required full-TRAIN utility_validation.json is missing")
    config = utility["config"]
    check(config["seed"] == 2026 and config["samples_per_condition"] == 100
          and config["max_train_profiles"] is None, "utility requires primary seed/cap and uncapped TRAIN")
    pool, selection = utility["source_training_pool"], utility["utility_training_selection"]
    check(pool["prepared_profiles"] == pool["actual_gan_profiles"] == TRAIN_PROFILES,
          "utility source is not full TRAIN")
    check(selection["cap_applied"] is False and selection["selected_profiles"] == TRAIN_PROFILES
          and selection["source_pool_equals_real_ridge_fit_rows"] is True
          and selection["selected_indices_sha256"] == selected["train_indices_sha256"],
          "utility ridge fitting rows do not equal full TRAIN")
    counts = pool["condition_counts"]
    check(set(counts) == CONDITIONS and sum(counts.values()) == TRAIN_PROFILES
          and all(type(n) is int and n > 0 for n in counts.values()), "utility TRAIN condition counts invalid")
    check(selection["condition_counts"] == counts and selection["labels_sha256"] == pool["labels_sha256"],
          "utility TRAIN labels/counts mismatch")
    task = utility["task"]
    required_task = {"name": "within_day_observed_history_next_interval_linear_ridge", "resolution": 96,
                     "lags": 12, "forecast_horizon_intervals": 1, "forecast_horizon_hours": .25,
                     "target_slots_inclusive": [12, 95], "ridge_alpha": 1.}
    check(all(task.get(key) == value for key, value in required_task.items()), "unexpected utility task")
    for key, expected in {"training_condition_counts_match": True, "training_label_order_matches": True,
                          "identical_validation_windows": True, "fit_uses_validation_values": False,
                          "prediction_uses_observed_real_past_lags": True,
                          "prediction_uses_its_own_target_or_future_values": False,
                          "cross_day_windows": False, "calibration_fitted_on_current_generation_batch": False}.items():
        check(report["invariants"].get(key) is expected, f"utility invariant failed: {key}")
    check(report["target"] == "net" and report["units"] == "kWh/interval", "utility target/units mismatch")
    for method in UTILITY_METHODS:
        fit, scores = utility["training"][method], utility["methods"][method]
        check(fit["n_profiles"] == TRAIN_PROFILES and fit["n_windows"] == TRAIN_PROFILES * 84
              and fit["condition_counts"] == counts and fit["labels_sha256"] == pool["labels_sha256"],
              f"utility {method} fitting exposure mismatch")
        check(all(_hash(fit[key]) for key in ("profiles_sha256", "labels_sha256", "coefficients_sha256")),
              f"utility {method} missing fitted artifact fingerprints")
        for metric in ("rmse_kwh", "mae_kwh"):
            _macro(scores["macro"][metric], {key: row[metric] for key, row in scores["per_condition"].items()},
                   f"utility {method} {metric}")
        check(all(row["n_profiles"] == 100 for row in scores["per_condition"].values()),
              "utility must evaluate the same 100 rows in each condition")
    check(utility["training"]["training_real"]["profiles_sha256"] == pool["profiles_sha256"],
          "utility real ridge fit profiles differ from TRAIN pool")
    return _source_files(utility["provenance"], UTILITY_CORE, "source_file_sha256")


def _verify_protocol(run, summary, history, calibrated, utility_report, arm, seed):
    config = {"derivative_swd_weight": 0., **run["config"]}
    check(config == expected_config(arm, seed), f"{arm} seed {seed}: configuration differs from fixed protocol")
    check(run["epochs_completed"] == EPOCHS and run["stopped_early"] is False,
          "full 26-epoch training budget required")
    check(len(history) == EPOCHS and [row["epoch"] for row in history] == list(range(1, 27)),
          "training history must contain all epochs exactly once")
    for key in ("generator_updates", "discriminator_updates"):
        check(summary[key] == UPDATES and all(row[key] == UPDATES_PER_EPOCH for row in history)
              and sum(row[key] for row in history) == UPDATES, f"full 30,784 {key} required")
    check(run["selection_metric"] == "mean_condition_interval_wasserstein_normalized",
          "checkpoint selection must use raw condition-macro interval W1")
    selection = run["training_selection"]
    check(selection["train_profiles_used"] == selection["train_profiles_available"] == TRAIN_PROFILES,
          "training must use all 606,060 eligible TRAIN profiles")
    check(run["data_curation_status"] == "curation_metadata_present_not_reaudited"
          and run["curation_metadata"] and _hash(run["checkpoint_data_hashes"].get("curation_manifest.json")),
          "curated dataset metadata and manifest fingerprint required")
    check(calibrated["target"] == "net" and calibrated["units"] == "kWh/interval"
          and calibrated["resolution"] == 96 and calibrated["interval_hours"] == .25,
          "pilot requires 96-slot signed net import in kWh/interval")
    settings = run["group_settings"]
    for key, expected in {"evaluation_seed": 2026, "samples_per_condition": 100,
                          "calibration_seed": 7301, "calibration_samples_per_condition": 4096,
                          "quantile_knots": 1025}.items():
        check(settings[key] == expected, f"primary setting mismatch: {key}")
    check(run["validation_selection"]["condition_counts"] == {label: 100 for label in CONDITIONS}
          and calibrated["frozen_samples"]["profile_shape"] == [6400, 96], "primary evaluation needs 100 rows in all 64 conditions")
    check(set(calibrated["checkpoint"]["supported_labels"]) == set(range(64)), "checkpoint lacks full condition support")
    check(set(calibrated["methods"]) == collector.REQUIRED_METHODS, "comparison method coverage differs")
    check(run["utility"] is not None, "required full-TRAIN utility_validation.json is missing")
    _calibration_diagnostics(run, calibrated)
    directions = run["diagnostic_settings"]["joint_projection_directions"]
    check(directions and directions["seed"] == 9182 and directions["count"] == 128
          and directions["n_features"] == 96 and _hash(directions["sha256"]), "fixed projected-profile directions missing/incompatible")
    expected_directions = np.random.default_rng(9182).normal(size=(128, 96))
    expected_directions /= np.linalg.norm(expected_directions, axis=1, keepdims=True)
    check(directions["sha256"] == hashlib.sha256(expected_directions.astype("<f8").tobytes()).hexdigest(),
          "projected direction fingerprint differs from fixed 128-by-96 matrix")
    scenario = run["diagnostic_settings"]["battery_scenario"]
    expected_battery = {"capacity_kwh": 10., "charge_power_kw": 4., "discharge_power_kw": 4.,
                        "initial_soc_kwh": 0., "interval_hours": .25, "episode_hours": 24.,
                        "grid_charging_allowed": False, "battery_export_allowed": False,
                        "eta_charge": math.sqrt(.9), "eta_discharge": math.sqrt(.9)}
    check(scenario and all(same(scenario.get(key), value) for key, value in expected_battery.items()),
          "fixed additional-battery scenario missing/incompatible")
    for method in collector.REQUIRED_METHODS:
        physical, joint, battery = (run[key][method] for key in ("methods", "joint_diagnostics", "battery_diagnostics"))
        for metric in collector.METRICS:
            _macro(physical["joint_and_marginal_macro"][metric],
                   {label: row[metric] for label, row in physical["per_condition"].items()}, f"{method} {metric}")
        _macro(joint["macro"], {label: row["mean_projected_wasserstein_kwh"] for label, row in joint["per_condition"].items()},
               f"{method} projected profile")
        check(joint["directions"] == directions and battery["scenario"] == scenario,
              "method diagnostic settings mismatch")
        for label in CONDITIONS:
            check(calibrated["methods"][method]["counts"]["real"][label] ==
                  calibrated["methods"][method]["counts"]["generated"][label] == 100,
                  "physical diagnostic profile counts mismatch")
            check(joint["per_condition"][label]["n_real"] == joint["per_condition"][label]["n_generated"] ==
                  battery["per_condition"][label]["n_real"] == battery["per_condition"][label]["n_generated"] == 100,
                  "joint/battery diagnostic profile counts mismatch")
        check(bool(battery["macro"]), "battery outcomes missing")
        for metric, value in battery["macro"].items():
            _macro(value, {label: row[metric] for label, row in battery["per_condition"].items()},
                   f"{method} battery {metric}", nonnegative=False)
    training_required = list(collector.TRAINING_CORE_FILES)
    if arm == "baseline":
        training_required.remove("src/fluvius_cgan/temporal_loss.py")
    training_sources = _source_files(run["provenance"]["training_checkpoint"], training_required)
    eval_sources = _source_files(run["provenance"]["calibration_evaluation"], EVALUATION_CORE)
    for key in ("evaluation_script_sha256", "comparison_helper_sha256"):
        check(_hash(run["provenance"]["calibration_evaluation"].get(key)), f"calibration runner fingerprint missing: {key}")
        eval_sources[key] = run["provenance"]["calibration_evaluation"][key]
    # The historical raw CLI did not save source provenance. Its values and
    # checkpoint/data hashes are verified against the fully bound calibration
    # report by collect_run; never invent missing historical source hashes.
    raw_provenance = run["provenance"]["raw_evaluation"]
    raw_sources = (_source_files(raw_provenance, (*MODEL_CORE, "src/fluvius_cgan/evaluation.py"))
                   if raw_provenance else None)
    utility_sources = _utility_checks(run, utility_report)
    check(utility_sources["scripts/compare_checkpoints.py"] == eval_sources["comparison_helper_sha256"],
          "utility/calibration checkpoint verifier source mismatch")
    for path in MODEL_CORE:
        check(training_sources[path] == utility_sources[path] ==
              run["provenance"]["calibration_evaluation"]["source_files_sha256"].get(path),
              f"training/inference model source mismatch: {path}")
        if raw_sources:
            check(raw_sources[path] == training_sources[path], f"raw inference source mismatch: {path}")
    if raw_sources:
        check(raw_sources["src/fluvius_cgan/evaluation.py"] == eval_sources["src/fluvius_cgan/evaluation.py"],
              "raw/calibrated metric implementation differs")
    for path in EVALUATION_CORE:
        if path in utility_sources:
            check(utility_sources[path] == eval_sources[path], f"utility/evaluation implementation differs: {path}")
    if arm == "candidate":
        derivative = run["provenance"]["training_checkpoint"]["derivative_swd"]
        scale = derivative["scale"]
        check(derivative["enabled"] is True and derivative["weight"] == 1.
              and derivative["projection_seed"] == seed + 5 and derivative["num_projections"] == 128,
              "derivative objective/projection provenance differs")
        check(summary["derivative_swd_scale"] == scale and scale["n_profiles"] == TRAIN_PROFILES
              and scale["n_features"] == 96 and scale["n_differences"] == 95
              and scale["train_indices_sha256"] == selection["train_indices_sha256"]
              and scale["train_npz_sha256"] == run["prepared_data_hashes"]["train.npz"]
              and scale["fitting_scope"] == "actual_gradient_training_subset_only"
              and scale["centering"] is False and scale["wraparound"] is False,
              "derivative scale is not bound to the actual TRAIN pool")
        check(finite(scale["estimate_float64"]) and scale["estimate_float64"] > 0
              and scale["used_float32"] == float(np.float32(scale["estimate_float64"])), "invalid frozen derivative scale")
        check(scale["condition_counts"] == run["utility"]["source_training_pool"]["condition_counts"]
              and set(scale["condition_mean_squared_differences"]) == CONDITIONS
              and all(finite(v) and v >= 0 for v in scale["condition_mean_squared_differences"].values())
              and same(scale["estimate_float64"] ** 2,
                       statistics.mean(scale["condition_mean_squared_differences"].values())),
              "derivative scale condition weighting/counts differ")
    run["pilot_sources"] = {"training": training_sources, "evaluation": eval_sources, "utility": utility_sources}
    run["calibration_diagnostics"] = {key: calibrated["calibration"][key]
                                      for key in ("fitting_diagnostics", "application_diagnostics")}
    run["arm"] = arm
    run["discriminator_updates"] = summary["discriminator_updates"]
    return run


def verify_run(arm, seed, directory):
    run = collector.collect_run(f"{arm}_seed_{seed}", directory)
    paths = {name: Path(directory) / name for name in
             ("summary.json", "history.json", "calibration_validation.json", "utility_validation.json")}
    check(paths["utility_validation.json"].is_file(), "required full-TRAIN utility_validation.json is missing")
    values = {name: collector._json(path) for name, path in paths.items()}
    result = _verify_protocol(run, values["summary.json"], values["history.json"],
                              values["calibration_validation.json"], values["utility_validation.json"], arm, seed)
    for name, digest in run["artifact_sha256"].items():
        check(collector.sha256_file(Path(directory) / name) == digest, f"artifact changed during pilot assessment: {name}")
    return result


def _compatibility(runs):
    first = runs[0]
    for run in runs[1:]:
        for key in ("prepared_data_hashes", "checkpoint_data_hashes", "curation_metadata",
                    "validation_selection", "diagnostic_settings"):
            check(same(run[key], first[key]), f"paired study incompatible: {key}")
        check(all(run["training_selection"][key] == first["training_selection"][key] for key in
                  ("train_profiles_available", "train_profiles_used", "train_indices_sha256")),
              "paired actual TRAIN subset differs")
        for key in ("labels_sha256", "real_profiles_sha256"):
            check(run["frozen_samples"][key] == first["frozen_samples"][key], f"paired frozen reference differs: {key}")
        for key in ("evaluation", "utility"):
            check(run["pilot_sources"][key] == first["pilot_sources"][key], f"paired {key} source fingerprints differ")
        check(run["utility"]["task"] == first["utility"]["task"], "utility tasks differ")
        check({k: v for k, v in run["utility"]["config"].items() if k != "calibrator"} ==
              {k: v for k, v in first["utility"]["config"].items() if k != "calibrator"}, "utility settings differ")
        for method in collector.BASELINES:
            check(run["frozen_samples"]["profiles_sha256"][method] == first["frozen_samples"]["profiles_sha256"][method],
                  f"matched frozen statistical control differs: {method}")
            for key in ("methods", "joint_diagnostics", "battery_diagnostics"):
                check(same(run[key][method], first[key][method]), f"matched control diagnostics differ: {key}/{method}")
        for method in ("training_real", "training_resampling"):
            check(same(run["utility"]["methods"][method], first["utility"]["methods"][method])
                  and run["utility"]["training"][method] == first["utility"]["training"][method],
                  f"matched utility reference differs: {method}")
    # A training implementation change is declared. Do not compare old/new
    # training.py hashes, nor require historical hashes to match today's files.
    for arm in ("candidate", "baseline"):
        members = [run for run in runs if run["arm"] == arm]
        check(all(run["pilot_sources"]["training"] == members[0]["pilot_sources"]["training"] for run in members),
              f"training core source fingerprints differ within {arm} arm")
    for path in (*MODEL_CORE, "src/fluvius_cgan/distribution_loss.py"):
        check(len({run["pilot_sources"]["training"][path] for run in runs}) == 1,
              f"undeclared architecture/profile-loss source change: {path}")
    for seed in SEEDS:
        pair = [run for run in runs if run["training_seed"] == seed]
        check(pair[0]["training_selection"] == pair[1]["training_selection"],
              f"matched seed {seed} checkpoint-selection rows/noise differ")


def _value(run, metric):
    return (run["joint_diagnostics"][METHOD]["macro"] if metric == "projected_profile_wasserstein_kwh"
            else run["methods"][METHOD]["joint_and_marginal_macro"][metric])


def _comparison(candidate, baseline, maximum_ratio):
    # Zero references remain testable by the absolute inequality; undefined
    # relative changes are JSON null, never infinity or an automatic pass.
    return {"candidate": candidate, "reference": baseline, "maximum_ratio": maximum_ratio,
            "limit": maximum_ratio * baseline, "difference": candidate - baseline,
            "ratio": candidate / baseline if baseline else None,
            "relative_change": (candidate - baseline) / baseline if baseline else None,
            "pass": candidate <= maximum_ratio * baseline}


def measured_gates(runs):
    indexed = {(run["arm"], run["training_seed"]): run for run in runs}
    candidates, baselines = ([indexed[arm, seed] for seed in SEEDS] for arm in ("candidate", "baseline"))
    gates = {}
    paired = [{"seed": seed, **_comparison(_value(c, GATED_METRICS[0]), _value(b, GATED_METRICS[0]), 1.)}
              for seed, c, b in zip(SEEDS, candidates, baselines)]
    for row in paired:
        row["pass"] = row["candidate"] < row["reference"]
        row["comparison"] = "strictly_lower"
    gates["ramp_lower_in_every_paired_seed"] = {"pass": all(row["pass"] for row in paired), "per_seed": paired}
    for metric in GATED_METRICS:
        c, b = ([ _value(run, metric) for run in members] for members in (candidates, baselines))
        gates[f"mean_{metric}"] = {**_comparison(statistics.mean(c), statistics.mean(b),
                                                .90 if metric == GATED_METRICS[0] else 1.10),
                                    "candidate_by_seed": dict(zip(map(str, SEEDS), c)),
                                    "baseline_by_seed": dict(zip(map(str, SEEDS), b))}
    utility_rows = []
    for seed, candidate in zip(SEEDS, candidates):
        values = candidate["utility"]["methods"]
        utility_rows.append({"seed": seed, "comparisons": {
            method: _comparison(values["training_calibrated_cgan"]["macro"]["rmse_kwh"],
                                values[method]["macro"]["rmse_kwh"], 1.10)
            for method in ("training_real", "training_resampling")}})
    aggregate = {method: statistics.mean(run["utility"]["methods"][method]["macro"]["rmse_kwh"] for run in candidates)
                 for method in UTILITY_METHODS}
    gates["utility_each_seed_within_both_references"] = {
        "pass": all(value["pass"] for row in utility_rows for value in row["comparisons"].values()),
        "per_seed": utility_rows, "three_seed_means_rmse_kwh": aggregate,
        "aggregate_comparisons": {method: _comparison(aggregate["training_calibrated_cgan"], aggregate[method], 1.10)
                                  for method in ("training_real", "training_resampling")}}
    return gates


def assess(candidate_root, baseline_root):
    runs, errors = [], []
    roots = {"candidate": Path(candidate_root), "baseline": Path(baseline_root)}
    requested = [{"arm": arm, "seed": seed, "directory": str(root / f"seed_{seed}")}
                 for arm, root in roots.items() for seed in SEEDS]
    if len({str(Path(item["directory"]).resolve()) for item in requested}) != 6:
        errors.append({"run": "study", "error": "candidate and baseline directories must be distinct"})
    for item in requested:
        try:
            runs.append(verify_run(item["arm"], item["seed"], item["directory"]))
        except (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError, RuntimeError,
                EOFError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
            errors.append({**item, "error": f"{type(error).__name__}: {error}"})
    if not errors:
        try:
            _compatibility(runs)
        except (ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
            errors.append({"run": "cross_run_compatibility", "error": str(error)})
    complete = not errors and len(runs) == 6
    gates = measured_gates(runs) if complete else {}
    passed = all(gate["pass"] for gate in gates.values()) if complete else None
    return {"schema_version": 1, "status": "incomplete" if not complete else "pass" if passed else "fail",
            "complete": complete, "pass": passed, "promotion_eligible": complete and passed is True,
            "scope": "Exploratory validation-only promotion screen; not significance, final-test evaluation, or a general superiority claim.",
            "test_used": False, "protocol": {"document": "docs/TEMPORAL_REFINEMENT.md", "seeds": list(SEEDS),
                "training_profiles": TRAIN_PROFILES, "epochs": EPOCHS, "updates_per_adversary": UPDATES,
                "evaluation_seed": 2026, "samples_per_condition": 100, "calibration_seed": 7301,
                "calibration_samples_per_condition": 4096, "quantile_knots": 1025,
                "aggregation": "Arithmetic mean of the three seed-level equal-condition macro errors; no seed dropping.",
                "battery_role": "Required retained secondary diagnostic; not a promotion gate.",
                "source_policy": "Verify within-run bindings and within-arm core source consistency; training implementation hashes may differ across arms. No claim that recorded hashes prove correct execution."},
            "requested_runs": requested, "errors": errors, "gates": gates,
            "failed_gates": [name for name, gate in gates.items() if not gate["pass"]],
            "runs": runs,
            "provenance": {"assessor_sha256": collector.sha256_file(Path(__file__)),
                "collector_sha256": collector.sha256_file(Path(collector.__file__)),
                "protocol_sha256": collector.sha256_file(ROOT / "docs/TEMPORAL_REFINEMENT.md")}}


def markdown(report):
    lines = ["# Temporal pilot assessment", "", f"Status: **{report['status']}**. "
             f"Complete: {report['complete']}. Promotion eligible: {report['promotion_eligible']}.", "", report["scope"], ""]
    for item in report["errors"]:
        lines.append(f"- {item.get('arm', item.get('run'))} {item.get('seed', '')}: {item['error']}")
    if report["complete"]:
        lines += ["| Gate | Pass | Candidate mean | Reference mean | Limit |", "|---|---|---:|---:|---:|"]
        for name, gate in report["gates"].items():
            numbers = [f"{gate[key]:.9g}" if key in gate else "—" for key in ("candidate", "reference", "limit")]
            lines.append(f"| {name} | {gate['pass']} | {' | '.join(numbers)} |")
    lines += ["", "JSON retains exact gate values, paired seed comparisons, all 64 conditions, battery outcomes, "
              "support/tie diagnostics, utility evidence and artifact/source fingerprints. "
              "Incomplete evidence never authorizes promotion. Existing outputs are never overwritten.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    outputs = [path for path in (args.output, args.markdown) if path is not None]
    if len({str(path.resolve()) for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        parser.error("outputs must be distinct new paths; existing assessments are retained")
    report = assess(args.candidate_root, args.baseline_root)
    contents = [json.dumps(report, indent=2, allow_nan=False) + "\n"]
    if args.markdown:
        contents.append(markdown(report))
    for path, content in zip(outputs, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    print(f"{report['status']}: {args.output}")
    return 0 if report["promotion_eligible"] else 1 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
