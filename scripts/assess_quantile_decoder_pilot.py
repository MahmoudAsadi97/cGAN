"""Assess the fixed 11/22/33 quantile-decoder pilot from paired JSON reports only.

This command never opens profile arrays, checkpoints or calibration archives and
never trains, generates, fits or evaluates. The evaluator's recorded bindings are
checked against the prospective baseline/source contracts. This is a report audit,
not independent verification that the recorded numerical execution occurred.
Incomplete evidence never grants promotion. Exit codes: 0 pass, 1 fail, 2 incomplete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33)
CONDITIONS = {str(label) for label in range(64)}
TRAIN_PROFILES = 606060
EPOCHS, UPDATES, UPDATES_PER_EPOCH = 26, 30784, 1184
PHYSICAL_FILE = "quantile_decoder_validation.json"
UTILITY_FILE = "quantile_decoder_utility_validation.json"
BASELINE_BINDINGS = ROOT / "configs/quantile_decoder_baseline_bindings.json"
BASELINE_BINDINGS_SHA256 = "046ddf6d3ba66ac16507539c494422c9bc322505da00bb9a35e848dc4223f9e9"
PROTOCOL_DOCUMENT = ROOT / "docs/QUANTILE_DECODER_REFINEMENT.md"
CANDIDATE_METHOD = "quantile_decoder_cgan"
REFERENCE_METHOD = "reference_calibrated_cgan"
CONTROLS = ("gaussian_copula", "independent_marginals", "exact_marginal_copula",
            "conditional_resampling", "conditional_mean")
PHYSICAL_METHODS = {CANDIDATE_METHOD, REFERENCE_METHOD, "reference_raw_cgan", *CONTROLS}
UTILITY_METHODS = {"training_quantile_decoder", "training_reference_raw", "training_reference_calibrated",
                   "training_real", "training_resampling", "training_copula", "persistence"}
GATED_METRICS = ("ramp_wasserstein_kw", "daily_energy_wasserstein_kwh",
                 "peak_power_wasserstein_kw", "lag_one_acf_wasserstein",
                 "projected_profile_wasserstein_kwh")
PREPARED_FILES = {"train.npz", "val.npz", "metadata.json", "scaler.json"}
PROTOCOL_ID = "conditional_physical_quantile_decoder_v1"
RELEASE_SHA256 = "934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8"
CONFIG_PATH = "configs/curated_quantile_decoder.json"
SOURCE_FILES = {
    *(f"src/fluvius_cgan/{name}.py" for name in
      ("training", "models", "residual", "distribution_loss", "temporal_loss", "quantile_decoder",
       "evaluation", "calibration", "copula", "joint", "battery", "utility")),
    "scripts/evaluate_quantile_decoder.py", "scripts/assess_quantile_decoder_pilot.py",
    "scripts/compare_checkpoints.py", "scripts/evaluate_calibration.py", "scripts/evaluate_utility.py",
}
EXPECTED_BUDGET = {"train_profiles": TRAIN_PROFILES, "epochs": EPOCHS,
                   "updates_per_adversary": UPDATES, "updates_per_epoch": UPDATES_PER_EPOCH}
EXPECTED_EVALUATION = {"seed": 2026, "samples_per_condition": 100, "calibration_seed": 7301,
                       "calibration_samples_per_condition": 4096, "quantile_knots": 1025,
                       "candidate_posthoc_calibration": False, "baseline_calibration": "frozen_original_saved_map"}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _equal(left, right):
    """Exact JSON evidence equality, including boolean versus numeric types."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value, name):
    def invalid(token):
        raise ValueError(f"{name}: nonfinite JSON token {token}")

    def unique(items):
        result = {}
        for key, item in items:
            check(key not in result, f"{name}: duplicate JSON key {key}")
            result[key] = item
        return result

    return json.loads(value, parse_constant=invalid, object_pairs_hook=unique)


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def load_declaration(path):
    """Verify prospective JSON against published commit bytes, without data I/O."""
    check(Path(path).suffix.lower() == ".json", "execution declaration must be a JSON document")
    payload = Path(path).read_bytes()
    declaration = _json_bytes(payload, str(path))
    check(declaration.get("schema_version") == 1 and declaration.get("protocol_id") == PROTOCOL_ID,
          "unsupported execution declaration")
    check(_equal(declaration.get("seeds"), list(SEEDS)), "declaration must fix seeds 11, 22, 33")
    check(_equal(declaration.get("budget"), EXPECTED_BUDGET), "declaration training budget differs")
    check(_equal(declaration.get("evaluation"), EXPECTED_EVALUATION), "declaration evaluation settings differ")
    commit, tree = declaration.get("source_commit"), declaration.get("source_tree")
    check(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) and
          isinstance(tree, str) and re.fullmatch(r"[0-9a-f]{40}", tree), "declaration needs full commit/tree IDs")
    check(_git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip() == commit,
          "declared source commit is unavailable")
    check(_git("rev-parse", f"{commit}^{{tree}}").decode().strip() == tree, "declaration source tree mismatch")
    sources = declaration.get("source_files_sha256", {})
    check(isinstance(sources, dict) and SOURCE_FILES <= set(sources), "declaration source inventory is incomplete")
    for name, digest in sources.items():
        check(isinstance(name, str) and name.startswith(("src/", "scripts/")) and name.endswith(".py")
              and not name.startswith("/") and ".." not in Path(name).parts
              and _hash(digest), "invalid declared source path/hash")
        check(_digest_bytes(_git("show", f"{commit}:{name}")) == digest, f"declared source blob mismatch: {name}")
    check(_digest_bytes(Path(__file__).read_bytes()) == sources["scripts/assess_quantile_decoder_pilot.py"],
          "running assessor differs from the declared published source")
    bound_files = {"config_sha256": CONFIG_PATH, "protocol_sha256": str(PROTOCOL_DOCUMENT.relative_to(ROOT)),
                   "baseline_bindings_sha256": str(BASELINE_BINDINGS.relative_to(ROOT))}
    contents = {}
    for key, name in bound_files.items():
        contents[key] = _git("show", f"{commit}:{name}")
        check(_hash(declaration.get(key)) and _digest_bytes(contents[key]) == declaration[key],
              f"declaration {key} differs from published bytes")
    check(declaration["baseline_bindings_sha256"] == BASELINE_BINDINGS_SHA256,
          "original baseline binding manifest differs from fixed prospective evidence")
    bindings = _json_bytes(contents["baseline_bindings_sha256"], "baseline bindings")
    check(bindings.get("schema_version") == 1 and set(bindings.get("seeds", {})) == set(map(str, SEEDS)),
          "baseline bindings require all three original seeds")
    config = _json_bytes(contents["config_sha256"], "candidate configuration")
    for seed, baseline in bindings["seeds"].items():
        check(_equal(declaration.get("prepared_data_hashes"), baseline["prepared_data_hashes"])
              and _equal(declaration.get("checkpoint_data_hashes"), baseline["checkpoint_data_hashes"]),
              f"declared data differs from original seed {seed}")
    check(set(declaration["prepared_data_hashes"]) == PREPARED_FILES and
          set(declaration["checkpoint_data_hashes"]) == PREPARED_FILES | {"curation_manifest.json"} and
          all(_hash(value) for value in declaration["checkpoint_data_hashes"].values()),
          "declared prepared data hashes incomplete")
    return {**declaration, "_sha256": _digest_bytes(payload), "_path": str(path),
            "_config": config, "_baseline_bindings": bindings}


def expected_config(seed, declaration):
    baseline = declaration["_baseline_bindings"]["seeds"][str(seed)]["config"]
    return {**baseline, "derivative_swd_weight": 0., "conditional_quantile_decoder": True}


def _macro(value, per_condition, name, *, nonnegative=True):
    check(isinstance(per_condition, dict) and set(per_condition) == CONDITIONS,
          f"{name}: all 64 conditions are required")
    check(finite(value) and all(finite(item) for item in per_condition.values()),
          f"{name}: finite macro and condition metrics required")
    check(not nonnegative or (value >= 0 and min(per_condition.values()) >= 0),
          f"{name}: finite nonnegative errors required")
    check(math.isclose(value, statistics.mean(per_condition.values()), rel_tol=1e-12, abs_tol=1e-12),
          f"{name}: macro differs from equal-condition mean")


def _source_pin(provenance, declaration, stage):
    check(provenance.get("git_commit") == declaration["source_commit"] and
          provenance.get("git_commit_tree") == declaration["source_tree"] and provenance.get("git_dirty") is False,
          f"{stage}: source must be the clean declared commit/tree")
    recorded = provenance.get("source_files_sha256", {})
    check(all(recorded.get(name) == digest for name, digest in declaration["source_files_sha256"].items()
              if name.startswith("src/")), f"{stage}: declared package source fingerprints differ")
    if stage != "candidate training":
        check(all(recorded.get(name) == digest for name, digest in declaration["source_files_sha256"].items()),
              f"{stage}: declared runner/helper source fingerprints differ")


def _completion(details, completion, arm, seed, declaration):
    summary, history, config = completion["summary"], completion["history"], completion["config"]
    baseline = declaration["_baseline_bindings"]["seeds"][str(seed)]
    expected = expected_config(seed, declaration) if arm == "candidate" else baseline["config"]
    check(_equal(config, expected) and _equal(details["config"], expected), f"{arm}: configuration differs from fixed protocol")
    if arm == "candidate":
        # The declared input file may omit ordinary training defaults; no field
        # it explicitly declares may change, other than the fixed three seeds.
        check(all(key == "seed" or (key in expected and _equal(value, expected[key]))
                  for key, value in declaration["_config"].items()), "declared candidate configuration differs")
        _source_pin(details["training_provenance"], declaration, "candidate training")
    else:
        check(details["sha256"] == baseline["checkpoint_sha256"] and details["epoch"] == baseline["selected_epoch"],
              "reference is not the original pinned SWD10 checkpoint")
        check(_equal(details["training_provenance"], baseline["training_source"]),
              "original reference training provenance differs")
    check(_hash(details.get("sha256")) and details.get("supported_labels") == list(range(64)),
          f"{arm}: checkpoint fingerprint/support missing")
    check(_equal(details.get("data_hashes"), declaration["checkpoint_data_hashes"]), f"{arm}: checkpoint data hashes differ")
    check(summary.get("epochs_completed") == EPOCHS and summary.get("stopped_early") is False and summary.get("test_used") is False,
          f"{arm}: full 26-epoch budget required")
    check(isinstance(history, list) and [row.get("epoch") for row in history] == list(range(1, EPOCHS + 1)),
          f"{arm}: complete ordered 26-epoch history required")
    for key in ("generator_updates", "discriminator_updates"):
        check(type(summary.get(key)) is int and summary[key] == UPDATES and
              all(type(row.get(key)) is int and row[key] == UPDATES_PER_EPOCH for row in history),
              f"{arm}: full 30,784 {key} required")
    check(summary.get("samples_seen") == UPDATES * 512 and
          all(row.get("samples_seen") == UPDATES_PER_EPOCH * 512 and finite(row.get("validation_score")) for row in history),
          f"{arm}: sample exposure or validation scores differ")
    best = min(history, key=lambda row: row["validation_score"])
    check(summary.get("best_epoch") == details["epoch"] == best["epoch"] and
          summary.get("best_validation_score") == details["checkpoint_selection"]["best_validation_score"] == best["validation_score"],
          f"{arm}: selected checkpoint is not the earliest validation minimum")
    check(_equal(summary.get("provenance"), details["training_provenance"]) and
          _equal(summary.get("data_hashes"), details["data_hashes"]), f"{arm}: summary/checkpoint provenance or data differs")
    check(summary.get("derivative_swd_scale") is None and
          all(finite(row.get("loss_derivative_swd", 0.)) and row.get("loss_derivative_swd", 0.) == 0. for row in history),
          f"{arm}: derivative objective must remain disabled")
    selection = summary["selection"]
    for source in (selection, details["training_selection"]):
        check(source.get("train_profiles_available") == source.get("train_profiles_used") == TRAIN_PROFILES
              and _hash(source.get("train_indices_sha256")), f"{arm}: full TRAIN row binding missing")
    check(details["training_selection"]["train_indices_sha256"] == selection["train_indices_sha256"],
          f"{arm}: checkpoint/summary TRAIN rows differ")
    check(details["checkpoint_selection"]["validation_metric"] == "mean_condition_interval_wasserstein_normalized",
          f"{arm}: checkpoint selection metric changed")
    artifact_hashes = completion["artifact_sha256"]
    check(set(artifact_hashes) == {"best.pt", "last.pt", "config.json", "summary.json", "history.json"}
          and all(_hash(value) for value in artifact_hashes.values())
          and artifact_hashes["best.pt"] == details["sha256"], f"{arm}: completion artifact hashes incomplete")
    return selection


def _physical_metrics(report):
    check(set(report["methods"]) == PHYSICAL_METHODS and set(report["joint_diagnostics"]) == PHYSICAL_METHODS
          and set(report["battery_diagnostics"]) == PHYSICAL_METHODS, "physical/joint/battery method coverage differs")
    values = {}
    directions = None
    scenario = None
    for method in sorted(PHYSICAL_METHODS):
        physical, joint, battery = (report[key][method] for key in ("methods", "joint_diagnostics", "battery_diagnostics"))
        counts = physical["counts"]
        check(counts == {"real": {label: 100 for label in CONDITIONS}, "generated": {label: 100 for label in CONDITIONS}},
              f"{method}: physical evaluation needs identical 100-row condition exposure")
        extracted = {}
        for metric in GATED_METRICS[:-1]:
            getter = ((lambda row: row["autocorrelation"]["1"]["wasserstein"])
                      if metric == "lag_one_acf_wasserstein" else (lambda row, key=metric: row[key]))
            extracted[metric] = getter(physical["macro"])
            _macro(extracted[metric], {label: getter(row) for label, row in physical["per_condition"].items()},
                   f"{method}/{metric}")
        extracted[GATED_METRICS[-1]] = joint["macro"]
        _macro(extracted[GATED_METRICS[-1]], {label: row["mean_projected_wasserstein_kwh"]
                                           for label, row in joint["per_condition"].items()}, f"{method}/projected")
        check(set(battery["per_condition"]) == CONDITIONS and battery["macro"], f"{method}: battery outcomes missing")
        for metric, value in battery["macro"].items():
            _macro(value, {label: row[metric] for label, row in battery["per_condition"].items()},
                   f"{method}/battery/{metric}", nonnegative=False)
        for label in CONDITIONS:
            check(joint["per_condition"][label]["n_real"] == joint["per_condition"][label]["n_generated"] ==
                  battery["per_condition"][label]["n_real"] == battery["per_condition"][label]["n_generated"] == 100,
                  f"{method}: joint/battery profile exposure differs")
        if directions is None:
            directions, scenario = joint["directions"], battery["scenario"]
        check(_equal(joint["directions"], directions) and _equal(battery["scenario"], scenario),
              "physical methods use different diagnostic settings")
        values[method] = extracted
    check(directions.get("seed") == 9182 and directions.get("count") == 128 and
          directions.get("n_features") == 96 and _hash(directions.get("sha256")), "fixed projected directions missing")
    import numpy as np
    matrix = np.random.default_rng(9182).normal(size=(128, 96))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    check(directions["sha256"] == _digest_bytes(matrix.astype("<f8").tobytes()), "fixed projected direction hash differs")
    battery_expected = {"capacity_kwh": 10., "charge_power_kw": 4., "discharge_power_kw": 4., "initial_soc_kwh": 0.,
                        "interval_hours": .25, "episode_hours": 24., "grid_charging_allowed": False,
                        "battery_export_allowed": False, "eta_charge": math.sqrt(.9), "eta_discharge": math.sqrt(.9)}
    check(all(key in scenario and _equal(scenario[key], value) for key, value in battery_expected.items()),
          "fixed battery scenario differs")
    return values


def _utility_metrics(report, train_indices):
    check(set(report["methods"]) == UTILITY_METHODS and set(report["training"]) == UTILITY_METHODS - {"persistence"},
          "utility method/fitting coverage differs")
    selection = report["utility_training_selection"]
    check(selection["cap_applied"] is False and selection["selected_profiles"] == TRAIN_PROFILES and
          selection["selected_indices_sha256"] == train_indices, "utility must use all exact TRAIN rows")
    counts = selection["condition_counts"]
    check(set(counts) == CONDITIONS and all(type(value) is int and value > 0 for value in counts.values())
          and sum(counts.values()) == TRAIN_PROFILES, "utility full-TRAIN condition counts invalid")
    check(_hash(selection["labels_sha256"]) and _hash(selection["real_profiles_sha256"]), "utility TRAIN hashes missing")
    task_expected = {"name": "within_day_observed_history_next_interval_linear_ridge", "resolution": 96,
                     "lags": 12, "forecast_horizon_intervals": 1, "forecast_horizon_hours": .25,
                     "target_slots_inclusive": [12, 95], "ridge_alpha": 1., "windows_per_profile": 84,
                     "batch_profiles": 256, "ridge_objective": "sum_squared_error_plus_alpha_nonintercept_squared_norm"}
    check(all(key in report["task"] and _equal(report["task"][key], value) for key, value in task_expected.items()),
          "utility fixed observed-history task changed")
    invariants = {"full_actual_train_required": True, "identical_training_label_order": True,
                  "fit_uses_validation_values": False, "candidate_posthoc_map": False, "reference_map_refitted": False}
    check(all(report["invariants"].get(key) is value for key, value in invariants.items()), "utility invariant violated")
    for method, scores in report["methods"].items():
        for metric in ("rmse_kwh", "mae_kwh"):
            _macro(scores["macro"][metric], {label: row[metric] for label, row in scores["per_condition"].items()},
                   f"utility/{method}/{metric}")
        check(all(row["n_profiles"] == 100 for row in scores["per_condition"].values()), "utility validation rows differ")
        if method == "persistence":
            continue
        fit = report["training"][method]
        check(fit["n_profiles"] == TRAIN_PROFILES and fit["n_windows"] == TRAIN_PROFILES * 84 and
              fit["condition_counts"] == counts and fit["labels_sha256"] == selection["labels_sha256"] and
              _hash(fit["profiles_sha256"]) and _hash(fit["coefficients_sha256"]), f"utility/{method}: fitting rows/hashes differ")
    check(report["training"]["training_real"]["profiles_sha256"] == selection["real_profiles_sha256"],
          "utility real fitting pool hash differs")
    check(report["training"]["training_quantile_decoder"]["profile_dtype"] == "float32", "utility candidate must be raw decoded float32")
    check(report["copula"] == {"enabled": True, "quantile_knots": 1025, "shrinkage": .01, "source_profiles": TRAIN_PROFILES},
          "utility copula settings differ")
    return {"candidate": report["methods"]["training_quantile_decoder"]["macro"]["rmse_kwh"],
            **{method: report["methods"][method]["macro"]["rmse_kwh"] for method in ("training_real", "training_resampling")}}


def _decoder_checks(physical, utility, train_indices):
    diagnostics = physical["quantile_decoder"]
    metadata = diagnostics["fit_metadata"]
    check(_equal(metadata, utility["quantile_decoder"]["fit_metadata"]) and
          _equal(metadata, physical["candidate"]["training_provenance"]["quantile_decoder"]),
          "decoder fit metadata differs across training/physical/utility")
    check(metadata.get("schema_version") == 1 and metadata.get("algorithm") == PROTOCOL_ID and
          metadata.get("n_quantiles") == 1025 and metadata.get("condition_dim") == 64 and
          metadata.get("output_dim") == 96 and metadata.get("selected_profiles") == TRAIN_PROFILES and
          metadata.get("supported_labels") == list(range(64)), "decoder representation/count/support differs")
    check(metadata["selected_indices_sha256"] == train_indices and
          metadata["selected_labels_sha256"] == utility["utility_training_selection"]["labels_sha256"] and
          metadata["condition_counts"] == utility["utility_training_selection"]["condition_counts"],
          "decoder fitted TRAIN rows/labels/counts differ")
    summary = physical["training_completion"]["candidate"]["summary"]
    selection = summary["selection"]
    check(metadata["selected_profiles_sha256"] == selection["train_profiles_sha256"] and
          metadata["selected_labels_sha256"] == selection["train_labels_sha256"] and
          metadata["condition_counts"] == selection["train_profiles_per_condition"] and
          _equal(metadata, summary["quantile_decoder"]), "decoder fit metadata differs from completion evidence")
    binding = {"fitting_scope": "actual_gradient_training_subset_only",
               "train_npz_sha256": physical["prepared_data_hashes"]["train.npz"],
               "train_indices_sha256": train_indices, "train_labels_sha256": selection["train_labels_sha256"],
               "train_profiles_available": TRAIN_PROFILES, "train_profiles_used": TRAIN_PROFILES,
               "data_hashes": summary["data_hashes"], "scaler": summary["scaler"],
               "source_files_sha256": physical["candidate"]["training_provenance"]["source_files_sha256"]}
    check(_equal(metadata["source_binding"], binding), "decoder TRAIN/scaler/source binding differs")
    for key in ("selected_profiles_sha256", "probability_grid_sha256", "quantiles_sha256", "support_sha256", "counts_sha256"):
        check(_hash(metadata.get(key)), f"decoder fitted-array hash missing: {key}")
    check(diagnostics["physical_dtype"] == utility["quantile_decoder"]["physical_dtype"] == "float32" and
          diagnostics["probability_dtype"] == diagnostics["quantile_table_dtype"] == "float64" and
          diagnostics["near_saturation_threshold"] == 1e-6, "decoder arithmetic/diagnostic settings differ")
    check(set(diagnostics["per_condition"]) == CONDITIONS, "decoder diagnostics require all 64 conditions")
    events = {"probability_zero", "probability_one", "probability_near_zero", "probability_near_one",
              "below_decoded_endpoint", "above_decoded_endpoint", "at_lower_decoded_endpoint", "at_upper_decoded_endpoint",
              "below_actual_train_minimum", "above_actual_train_maximum"}
    for label, row in diagnostics["per_condition"].items():
        check(row["count"] == 100 and set(row["slot_event_counts"]) == set(row["fractions_over_profile_slots"]) == events,
              f"decoder condition {label}: diagnostic event/count coverage differs")
        for event, slots in row["slot_event_counts"].items():
            check(isinstance(slots, list) and len(slots) == 96 and all(type(value) is int and 0 <= value <= 100 for value in slots),
                  f"decoder {event}: 96 valid integer slot counts required")
            fraction = row["fractions_over_profile_slots"][event]
            check(finite(fraction) and math.isclose(fraction, sum(slots) / 9600, rel_tol=1e-12, abs_tol=1e-12),
                  f"decoder {event}: inconsistent event fraction")
        check(sum(row["slot_event_counts"]["below_decoded_endpoint"]) ==
              sum(row["slot_event_counts"]["above_decoded_endpoint"]) == 0, "decoder exceeded fixed decoded support")
        flats = row["fixed_zero_width_segment_fraction_by_slot"]
        check(len(flats) == 96 and all(finite(value) and 0 <= value <= 1 for value in flats), "decoder flat segment masses invalid")
        for name in ("sample_tied_observation_counts_by_slot", "sample_largest_atom_counts_by_slot"):
            check(len(row[name]) == 96 and all(type(value) is int and 0 <= value <= 100 for value in row[name]),
                  "decoder sample tie/atom counts invalid")
        slots = row["fixed_constant_slots"]
        check(isinstance(slots, list) and all(type(value) is int and 0 <= value < 96 for value in slots)
              and slots == sorted(set(slots)), "decoder constant-slot indices invalid")


def _secondary_evidence(physical, utility):
    expected_baselines = {"gaussian_copula": {"quantile_knots": 1025, "shrinkage": .01},
                          "independent_marginals": {"quantile_knots": 1025, "shrinkage": 1.},
                          "exact_marginal_copula": {"quantile_knots": None, "shrinkage": .01}}
    check(_equal(physical["baselines"], expected_baselines), "fixed statistical-control settings differ")
    diagnostics = physical["reference_map_diagnostics"]
    fitting, application = diagnostics["fitting"], diagnostics["application"]
    check(set(fitting["per_condition"]) == set(application["per_condition"]) == CONDITIONS and
          application["n_profiles"] == 6400, "original reference map diagnostic coverage differs")
    events = {"source_below", "source_above", "cdf_zero", "cdf_one", "target_lower_endpoint", "target_upper_endpoint"}
    totals = {event: 0 for event in events}
    for label in CONDITIONS:
        fit, applied = fitting["per_condition"][label], application["per_condition"][label]
        check(fit["train_count"] == utility["utility_training_selection"]["condition_counts"][label] and
              fit["calibration_count"] == 4096 and applied["count"] == 100, "reference map diagnostic fitting/application counts differ")
        for key in ("source_tie_mass_by_slot", "source_largest_atom_mass_by_slot", "source_knot_plateau_mass_by_slot"):
            check(len(fit[key]) == 96 and all(finite(value) and 0 <= value <= 1 for value in fit[key]),
                  "reference map diagnostic slot masses invalid")
        check(set(applied["slot_event_counts"]) == set(applied["fractions_over_profile_slots"]) == events,
              "reference map diagnostic event coverage differs")
        for event, slots in applied["slot_event_counts"].items():
            check(len(slots) == 96 and all(type(value) is int and 0 <= value <= 100 for value in slots),
                  "reference map diagnostic event counts invalid")
            fraction = applied["fractions_over_profile_slots"][event]
            check(finite(fraction) and math.isclose(fraction, sum(slots) / 9600, rel_tol=1e-12, abs_tol=1e-12),
                  "reference map diagnostic event fraction differs")
            totals[event] += sum(slots)
    check(application["total_event_counts"] == totals and set(application["fractions_over_profile_slots"]) == events and
          all(finite(application["fractions_over_profile_slots"][event]) and
              math.isclose(application["fractions_over_profile_slots"][event], totals[event] / (6400 * 96), rel_tol=1e-12, abs_tol=1e-12)
              for event in events), "reference map global diagnostic totals/fractions differ")
    neighbors = physical["nearest_neighbors"]
    check(set(neighbors) == {CANDIDATE_METHOD, REFERENCE_METHOD}, "both selected pipelines require neighbor diagnostics")
    for method, row in neighbors.items():
        check(row["reference_scope"] == "sampled_training_profiles" and
              row["total_counts"] == {"train": TRAIN_PROFILES, "generated": 6400, "heldout": 6400} and
              row["sample_counts"] == {"train": 1000, "generated": 1000, "heldout": 1000} and row["seed"] == 2026,
              f"{method}: nearest-neighbor bounded sample settings differ")
        for sample in ("generated", "heldout"):
            fraction = row[f"{sample}_exact_copy_fraction"]
            check(finite(fraction) and 0 <= fraction <= 1, "nearest-neighbor exact-copy fraction invalid")
            check(all(finite(row[f"{sample}_to_train_{stat}_rms_kwh"]) and row[f"{sample}_to_train_{stat}_rms_kwh"] >= 0
                      for stat in ("mean", "median", "p05", "p95")), "nearest-neighbor distances invalid")


def verify_pair(physical, utility, seed, declaration):
    """Validate two JSON objects; no input paths from reports are ever opened."""
    check(type(seed) is int and seed in SEEDS, "only fixed seeds 11, 22, 33 are eligible")
    original = declaration["_baseline_bindings"]["seeds"][str(seed)]
    shared = ("execution_declaration", "candidate", "reference", "reference_calibration", "training_completion",
              "baseline_binding", "prepared_data_hashes", "curation_manifest_sha256", "data_curation_status",
              "curation_metadata", "source_sha256", "target", "units", "resolution", "interval_hours",
              "training_profiles", "seed", "samples_per_condition_cap", "seeds", "validation_selection",
              "input_sha256", "post_input_sha256", "provenance")
    for report, mode in ((physical, "physical"), (utility, "utility")):
        check(report.get("schema_version") == 1 and report.get("report_kind") == f"quantile_decoder_pilot_{mode}" and
              report.get("split") == "val" and report.get("test_used") is False and report.get("training_seed") == seed,
              f"{mode}: explicit validation-only matching-seed report required")
        check(report.get("candidate_posthoc_calibration") is False and
              report.get("reference_calibration_refitted") is False and
              _equal(report.get("endpoints"), EXPECTED_EVALUATION), "candidate/reference endpoint declaration differs")
        expected_declaration = {"sha256": declaration["_sha256"], "source_commit": declaration["source_commit"],
                                "source_tree": declaration["source_tree"]}
        check(_equal(report["execution_declaration"], expected_declaration), "report execution declaration binding differs")
        _source_pin(report["provenance"], declaration, mode)
        check(_equal(report["prepared_data_hashes"], declaration["prepared_data_hashes"]) and
              report["curation_manifest_sha256"] == declaration["checkpoint_data_hashes"]["curation_manifest.json"],
              f"{mode}: prepared data/curation fingerprints differ")
        check(report["data_curation_status"] == "verified_curation_v2" and isinstance(report["curation_metadata"], dict)
              and report["curation_metadata"] and report["source_sha256"] == RELEASE_SHA256, "verified curated release metadata required")
        check(report["target"] == "net" and report["units"] == "kWh/interval" and report["resolution"] == 96 and
              report["interval_hours"] == .25, "pilot target/resolution/physical units differ")
        check(report["training_profiles"] == {"prepared": TRAIN_PROFILES, "candidate": TRAIN_PROFILES,
                                               "reference": TRAIN_PROFILES, "controls": TRAIN_PROFILES}, "full TRAIN exposure differs")
        check(report["seed"] == 2026 and report["samples_per_condition_cap"] == 100 and
              report["seeds"] == {"validation_selection": 2026, "generation": 2026, "controls": 2026, "reference_calibration": 7301},
              "fixed evaluation seeds/cap differ")
        selected = report["validation_selection"]
        check(selected["samples_per_condition_cap"] == 100 and selected["seed"] == 2026 and
              selected["selected_profiles"] == 6400 and selected["condition_counts"] == {label: 100 for label in CONDITIONS} and
              all(_hash(selected[key]) for key in ("selected_indices_sha256", "labels_sha256", "profiles_sha256", "meter_ids_sha256", "dates_sha256")),
              "fixed validation rows/labels/meters/dates hashes or counts missing")
        check(report["baseline_binding"] == {"sha256": BASELINE_BINDINGS_SHA256, "record": original},
              "report original baseline binding differs")
        inputs = report["input_sha256"]
        check(_equal(inputs, report["post_input_sha256"]) and all(_hash(value) for value in inputs.values()),
              "input artifact fingerprints changed during evaluation")
        required_inputs = {**{f"prepared/{name}": value for name, value in declaration["checkpoint_data_hashes"].items()},
                           "baseline_bindings": BASELINE_BINDINGS_SHA256, "reference_calibrator": original["calibration_sha256"],
                           "execution_declaration": declaration["_sha256"], "candidate_config": declaration["config_sha256"],
                           "protocol": declaration["protocol_sha256"]}
        for arm in ("candidate", "reference"):
            required_inputs.update({f"{arm}/{name}": value for name, value in report["training_completion"][arm]["artifact_sha256"].items()})
        check(inputs == required_inputs, "report input hashes do not bind the declared data/model/map/completion files")
    check(all(_equal(physical[key], utility[key]) for key in shared), "paired physical/utility source, inputs, or row evidence differs")
    selections = {arm: _completion(physical[arm], physical["training_completion"][arm], arm, seed, declaration)
                  for arm in ("candidate", "reference")}
    check(_equal(physical["training_completion"]["candidate"]["summary"]["scaler"],
                 physical["training_completion"]["reference"]["summary"]["scaler"]),
          "candidate/reference full-TRAIN standardizer differs")
    added_selection = {"train_labels_sha256", "train_profiles_per_condition", "train_profiles_sha256"}
    check(_equal({key: value for key, value in selections["candidate"].items() if key not in added_selection},
                 selections["reference"]), "paired checkpoint-selection rows/noise/settings differ")
    selected = selections["candidate"]
    check(isinstance(selected.get("validation_indices"), list) and len(selected["validation_indices"]) == 4096
          and all(type(value) is int and value >= 0 for value in selected["validation_indices"])
          and len(set(selected["validation_indices"])) == 4096 and
          all(_hash(selected.get(key)) for key in ("validation_labels_sha256", "validation_noise_sha256")),
          "checkpoint-selection row/label/noise evidence incomplete")
    train_indices = selections["candidate"]["train_indices_sha256"]
    calibration = physical["reference_calibration"]
    map_provenance = calibration["provenance"]
    check(calibration["sha256"] == original["calibration_sha256"] and map_provenance["checkpoint_sha256"] == original["checkpoint_sha256"]
          and map_provenance["train_indices_sha256"] == train_indices and map_provenance["prepared_data_hashes"] == declaration["prepared_data_hashes"]
          and calibration["calibration_seed"] == map_provenance["calibration_seed"] == 7301 and
          calibration["generation_seed"] == 2026 and map_provenance["calibration_samples_per_condition"] == 4096
          and calibration["quantile_knots"] == map_provenance["quantile_knots"] == 1025, "original frozen calibration binding/settings differ")
    physical_values = _physical_metrics(physical)
    utility_values = _utility_metrics(utility, train_indices)
    _decoder_checks(physical, utility, train_indices)
    _secondary_evidence(physical, utility)
    frozen = physical["frozen_samples"]
    check(frozen["real_profiles_sha256"] == physical["validation_selection"]["profiles_sha256"] and
          frozen["labels_sha256"] == physical["validation_selection"]["labels_sha256"] and
          set(frozen["profiles_sha256"]) == PHYSICAL_METHODS and all(_hash(value) for value in frozen["profiles_sha256"].values())
          and _hash(frozen["candidate_probabilities_sha256"]), "frozen profile/label/control hashes missing or inconsistent")
    check(set(frozen["profile_dtypes"]) == PHYSICAL_METHODS and frozen["profile_dtypes"][CANDIDATE_METHOD] == "float32",
          "candidate physical endpoint must be raw decoded float32")
    return {"seed": seed, "metrics": {"candidate": physical_values[CANDIDATE_METHOD], "reference": physical_values[REFERENCE_METHOD]},
            "utility_rmse": utility_values, "physical": physical, "utility": utility}


def _compatibility(pairs):
    check(len(pairs) == 3 and sorted(pair["seed"] for pair in pairs) == list(SEEDS), "exact paired seeds 11, 22, 33 required")
    first = pairs[0]
    for pair in pairs[1:]:
        left, right = pair["physical"], first["physical"]
        for key in ("execution_declaration", "prepared_data_hashes", "curation_manifest_sha256", "source_sha256",
                    "curation_metadata", "validation_selection", "training_profiles", "provenance"):
            check(_equal(left[key], right[key]), f"cross-seed shared source/data/rows differ: {key}")
        check(_equal(left["quantile_decoder"]["fit_metadata"], right["quantile_decoder"]["fit_metadata"]),
              "fixed TRAIN decoder differs across seeds")
        for method in CONTROLS:
            check(left["frozen_samples"]["profiles_sha256"][method] == right["frozen_samples"]["profiles_sha256"][method],
                  f"shared statistical control hash differs: {method}")
            for key in ("methods", "joint_diagnostics", "battery_diagnostics"):
                check(_equal(left[key][method], right[key][method]), f"shared control diagnostics differ: {method}/{key}")
        for key in ("task", "utility_training_selection", "copula"):
            check(_equal(pair["utility"][key], first["utility"][key]), f"cross-seed utility settings differ: {key}")
        for method in ("training_real", "training_resampling", "training_copula", "persistence"):
            check(_equal(pair["utility"]["methods"][method], first["utility"]["methods"][method]),
                  f"shared utility control scores differ: {method}")
            if method != "persistence":
                check(_equal(pair["utility"]["training"][method], first["utility"]["training"][method]),
                      f"shared utility control hashes/exposure differ: {method}")


def verify_run(seed, directory, declaration):
    paths = {"physical": Path(directory) / PHYSICAL_FILE, "utility": Path(directory) / UTILITY_FILE}
    payloads = {key: path.read_bytes() for key, path in paths.items()}
    reports = {key: _json_bytes(value, str(paths[key])) for key, value in payloads.items()}
    result = verify_pair(reports["physical"], reports["utility"], seed, declaration)
    check(all(path.read_bytes() == payloads[key] for key, path in paths.items()), "report changed during assessment")
    return {**result, "report_sha256": {key: _digest_bytes(value) for key, value in payloads.items()}}


def assess(candidate_root, declaration):
    pairs, errors = [], []
    requested = [{"seed": seed, "directory": str(Path(candidate_root) / f"seed_{seed}")} for seed in SEEDS]
    try:
        contract = load_declaration(declaration)
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        contract = None
        errors.append({"stage": "declaration", "error": f"{type(error).__name__}: {error}"})
    if contract is not None:
        for item in requested:
            try:
                pairs.append(verify_run(item["seed"], item["directory"], contract))
            except (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError) as error:
                errors.append({**item, "error": f"{type(error).__name__}: {error}"})
        if not errors:
            try:
                _compatibility(pairs)
            except (ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
                errors.append({"stage": "cross_seed_compatibility", "error": str(error)})
    complete = not errors and len(pairs) == 3
    gates = measured_gates(pairs) if complete else {}
    passed = all(gate["pass"] for gate in gates.values()) if complete else None
    return {"schema_version": 1, "status": "incomplete" if not complete else "pass" if passed else "fail",
            "complete": complete, "pass": passed, "promotion_eligible": complete and passed is True,
            "test_used": False, "scope": "Exploratory validation-only promotion screen; no significance, final-test or general superiority claim.",
            "protocol": {"id": PROTOCOL_ID, "seeds": list(SEEDS), "budget": EXPECTED_BUDGET,
                         "evaluation": EXPECTED_EVALUATION, "candidate_endpoint": CANDIDATE_METHOD,
                         "reference_endpoint": REFERENCE_METHOD, "reference": "Exact original SWD10 checkpoints and saved maps, freshly evaluated under the same declared corrected source.",
                         "aggregation": "Arithmetic mean of three seed-level equal-condition macro errors; no seed dropping.",
                         "battery_role": "Required retained secondary diagnostic; not a promotion gate."},
            "declaration": {key: value for key, value in contract.items() if not key.startswith("_")} if contract else None,
            "provenance": {"assessor_sha256": _digest_bytes(Path(__file__).read_bytes()),
                           "declaration_sha256": contract["_sha256"] if contract else None,
                           "binding_scope": "Report-only evidence audit; no checkpoints, maps, prepared profiles or TEST artifacts opened."},
            "requested_runs": requested, "errors": errors, "gates": gates,
            "failed_gates": [name for name, gate in gates.items() if not gate["pass"]], "pairs": pairs}


def markdown(report):
    lines = ["# Quantile-decoder pilot assessment", "", f"Status: **{report['status']}**. Promotion eligible: {report['promotion_eligible']}.",
             "", report["scope"], "", "Candidate: raw decoded cGAN. Reference: original SWD10 with its frozen saved map; both freshly evaluated under one declared corrected source.", ""]
    for item in report["errors"]:
        lines.append(f"- {item.get('stage', 'seed ' + str(item.get('seed')))}: {item['error']}")
    if report["complete"]:
        lines += ["| Gate | Pass | Candidate mean | Reference mean | Limit |", "|---|---|---:|---:|---:|"]
        for name, gate in report["gates"].items():
            numbers = [f"{gate[key]:.9g}" if key in gate else "—" for key in ("candidate", "reference", "limit")]
            lines.append(f"| {name} | {gate['pass']} | {' | '.join(numbers)} |")
    return "\n".join([*lines, "", "JSON retains all paired reports, condition metrics, support/tie diagnostics, utility controls and provenance. Incomplete evidence never authorizes promotion.", ""])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    outputs = [path for path in (args.output, args.markdown) if path is not None]
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        parser.error("outputs must be distinct new paths; existing assessments are retained")
    report = assess(args.candidate_root, args.declaration)
    contents = [json.dumps(report, indent=2, allow_nan=False) + "\n"]
    if args.markdown:
        contents.append(markdown(report))
    for path, content in zip(outputs, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
    print(f"{report['status']}: {args.output}")
    return 0 if report["promotion_eligible"] else 1 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


def _comparison(candidate, reference, maximum_ratio):
    return {"candidate": candidate, "reference": reference, "maximum_ratio": maximum_ratio,
            "limit": maximum_ratio * reference, "difference": candidate - reference,
            "ratio": candidate / reference if reference else None,
            "relative_change": (candidate - reference) / reference if reference else None,
            "pass": candidate <= maximum_ratio * reference}


def measured_gates(pairs):
    """Unchanged temporal-pilot arithmetic, applied to explicitly named endpoints."""
    check(len(pairs) == 3 and sorted(pair["seed"] for pair in pairs) == list(SEEDS),
          "exactly one paired result for each seed 11, 22, 33 is required")
    indexed = {pair["seed"]: pair for pair in pairs}
    pairs = [indexed[seed] for seed in SEEDS]
    for pair in pairs:
        values = [pair["metrics"][arm][metric] for arm in ("candidate", "reference") for metric in GATED_METRICS]
        values.extend(pair["utility_rmse"][method] for method in ("candidate", "training_real", "training_resampling"))
        check(all(finite(value) and value >= 0 for value in values), "all gate errors must be finite nonnegative numbers")
    paired = [{"seed": pair["seed"], **_comparison(pair["metrics"]["candidate"][GATED_METRICS[0]],
                                                    pair["metrics"]["reference"][GATED_METRICS[0]], 1.)}
              for pair in pairs]
    for row in paired:
        row["pass"] = row["candidate"] < row["reference"]
        row["comparison"] = "strictly_lower"
    gates = {"ramp_lower_in_every_paired_seed": {"pass": all(row["pass"] for row in paired), "per_seed": paired}}
    for metric in GATED_METRICS:
        candidate = [pair["metrics"]["candidate"][metric] for pair in pairs]
        reference = [pair["metrics"]["reference"][metric] for pair in pairs]
        gates[f"mean_{metric}"] = {
            **_comparison(statistics.mean(candidate), statistics.mean(reference), .9 if metric == GATED_METRICS[0] else 1.1),
            "candidate_by_seed": dict(zip(map(str, SEEDS), candidate)),
            "baseline_by_seed": dict(zip(map(str, SEEDS), reference)),
        }
    utility_rows = [{"seed": pair["seed"], "comparisons": {
        method: _comparison(pair["utility_rmse"]["candidate"], pair["utility_rmse"][method], 1.1)
        for method in ("training_real", "training_resampling")}} for pair in pairs]
    means = {method: statistics.mean(pair["utility_rmse"][method] for pair in pairs)
             for method in ("candidate", "training_real", "training_resampling")}
    gates["utility_each_seed_within_both_references"] = {
        "pass": all(value["pass"] for row in utility_rows for value in row["comparisons"].values()),
        "per_seed": utility_rows, "three_seed_means_rmse_kwh": means,
        "aggregate_comparisons": {method: _comparison(means["candidate"], means[method], 1.1)
                                  for method in ("training_real", "training_resampling")},
    }
    return gates
