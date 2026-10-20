"""Synthetic report contracts only: no profiles, checkpoint loads or inference."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("assess_quantile_decoder_pilot", Path(__file__).resolve().parents[1] /
                                            "scripts" / "assess_quantile_decoder_pilot.py")
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def gate_pairs():
    return [{"seed": seed,
             "metrics": {"candidate": {metric: .8 for metric in pilot.GATED_METRICS},
                         "reference": {metric: 1. for metric in pilot.GATED_METRICS}},
             "utility_rmse": {"candidate": 1., "training_real": 1., "training_resampling": 1.}}
            for seed in pilot.SEEDS]


def test_gate_arithmetic_matches_declared_original_thresholds():
    pairs = gate_pairs()
    gates = pilot.measured_gates(pairs)
    assert all(row["pass"] for row in gates.values())
    assert gates["mean_ramp_wasserstein_kw"]["relative_change"] == pytest.approx(-.2)
    assert gates["mean_ramp_wasserstein_kw"]["maximum_ratio"] == .9
    for metric in pilot.GATED_METRICS[1:]:
        assert gates[f"mean_{metric}"]["maximum_ratio"] == 1.1


def test_mean_ramp_improvement_cannot_hide_one_nonimproving_seed():
    pairs = gate_pairs()
    pairs[0]["metrics"]["candidate"]["ramp_wasserstein_kw"] = 1.
    gates = pilot.measured_gates(pairs)
    assert gates["mean_ramp_wasserstein_kw"]["pass"]
    assert not gates["ramp_lower_in_every_paired_seed"]["pass"]


def test_each_seed_improvement_does_not_relax_ten_percent_mean_gate():
    pairs = gate_pairs()
    for pair in pairs:
        pair["metrics"]["candidate"]["ramp_wasserstein_kw"] = .91
    gates = pilot.measured_gates(pairs)
    assert gates["ramp_lower_in_every_paired_seed"]["pass"]
    assert not gates["mean_ramp_wasserstein_kw"]["pass"]


@pytest.mark.parametrize("metric", ["daily_energy_wasserstein_kwh", "peak_power_wasserstein_kw",
                                    "lag_one_acf_wasserstein", "projected_profile_wasserstein_kwh"])
def test_each_mean_guardrail_is_independent(metric):
    pairs = gate_pairs()
    for pair in pairs:
        pair["metrics"]["candidate"][metric] = 1.1001
    gates = pilot.measured_gates(pairs)
    assert not gates[f"mean_{metric}"]["pass"]
    assert gates["mean_ramp_wasserstein_kw"]["pass"]


@pytest.mark.parametrize("reference", ["training_real", "training_resampling"])
def test_each_seed_must_pass_both_full_train_utility_references(reference):
    pairs = gate_pairs()
    pairs[1]["utility_rmse"][reference] = .9
    gates = pilot.measured_gates(pairs)
    assert not gates["utility_each_seed_within_both_references"]["pass"]
    assert not gates["utility_each_seed_within_both_references"]["per_seed"][1]["comparisons"][reference]["pass"]


def test_zero_reference_and_inclusive_boundaries():
    assert pilot._comparison(.9, 1., .9)["pass"]
    assert pilot._comparison(1.1, 1., 1.1)["pass"]
    assert pilot._comparison(0., 0., 1.1)["pass"]
    assert pilot._comparison(0., 0., 1.1)["ratio"] is None
    assert not pilot._comparison(.0001, 0., 1.1)["pass"]


@pytest.mark.parametrize("seeds", [[11, 22], [11, 22, 22], [11, 22, 44], [11, 22, 33, 55]])
def test_gate_api_never_drops_or_replaces_a_seed(seeds):
    pairs = gate_pairs()
    by_seed = {pair["seed"]: pair for pair in pairs}
    changed = [{**copy.deepcopy(by_seed.get(seed, pairs[0])), "seed": seed} for seed in seeds]
    with pytest.raises(ValueError, match="11, 22, 33"):
        pilot.measured_gates(changed)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -.1, None])
def test_gate_api_rejects_invalid_errors(value):
    pairs = gate_pairs()
    pairs[0]["metrics"]["candidate"]["ramp_wasserstein_kw"] = value
    with pytest.raises(ValueError, match="finite nonnegative"):
        pilot.measured_gates(pairs)


def contract():
    bindings = json.loads(pilot.BASELINE_BINDINGS.read_text())
    first = bindings["seeds"]["11"]
    result = {"schema_version": 1, "protocol_id": pilot.PROTOCOL_ID,
              "source_commit": "a" * 40, "source_tree": "b" * 40,
              "source_files_sha256": {name: digest(name) for name in pilot.SOURCE_FILES},
              "config_sha256": digest("config"), "protocol_sha256": digest("protocol"),
              "baseline_bindings_sha256": pilot.BASELINE_BINDINGS_SHA256,
              "seeds": list(pilot.SEEDS), "budget": copy.deepcopy(pilot.EXPECTED_BUDGET),
              "evaluation": copy.deepcopy(pilot.EXPECTED_EVALUATION),
              "prepared_data_hashes": copy.deepcopy(first["prepared_data_hashes"]),
              "checkpoint_data_hashes": copy.deepcopy(first["checkpoint_data_hashes"]),
              "_sha256": digest("declaration"), "_path": "synthetic.json", "_baseline_bindings": bindings}
    result["_config"] = pilot.expected_config(11, result)
    return result


def synthetic(seed=11, declaration=None):
    import numpy as np
    declaration = declaration or contract()
    original = declaration["_baseline_bindings"]["seeds"][str(seed)]
    labels = sorted(pilot.CONDITIONS, key=int)
    counts = {label: 9470 if int(label) < 44 else 9469 for label in labels}
    provenance = {"git_commit": declaration["source_commit"], "git_commit_tree": declaration["source_tree"],
                  "git_dirty": False, "source_files_sha256": copy.deepcopy(declaration["source_files_sha256"])}
    selection = {"train_profiles_available": 606060, "train_profiles_used": 606060,
                 "train_indices_sha256": digest("indices"), "validation_indices": list(range(4096)),
                 "validation_labels_sha256": digest(f"selection-labels-{seed}"),
                 "validation_noise_sha256": digest(f"selection-noise-{seed}")}
    fit = {"schema_version": 1, "algorithm": pilot.PROTOCOL_ID, "n_quantiles": 1025, "condition_dim": 64,
           "output_dim": 96, "selected_profiles": 606060, "supported_labels": list(range(64)),
           "selected_indices_sha256": selection["train_indices_sha256"], "selected_labels_sha256": digest("train-labels"),
           "condition_counts": counts, **{key: digest(key) for key in ("selected_profiles_sha256", "probability_grid_sha256",
                                                                      "quantiles_sha256", "support_sha256", "counts_sha256")}}
    scaler = {"kind": "standard", "mean": 0., "std": 1.}
    fit["source_binding"] = {"fitting_scope": "actual_gradient_training_subset_only",
        "train_npz_sha256": declaration["prepared_data_hashes"]["train.npz"], "train_indices_sha256": digest("indices"),
        "train_labels_sha256": digest("train-labels"), "train_profiles_available": 606060, "train_profiles_used": 606060,
        "data_hashes": copy.deepcopy(declaration["checkpoint_data_hashes"]), "scaler": scaler,
        "source_files_sha256": copy.deepcopy(provenance["source_files_sha256"])}
    candidates, completions = {}, {}
    for arm in ("candidate", "reference"):
        checkpoint_hash = digest(f"candidate-{seed}") if arm == "candidate" else original["checkpoint_sha256"]
        config = pilot.expected_config(seed, declaration) if arm == "candidate" else original["config"]
        training_provenance = {**copy.deepcopy(provenance), "quantile_decoder": fit} if arm == "candidate" else original["training_source"]
        selected_epoch = 20 if arm == "candidate" else original["selected_epoch"]
        candidates[arm] = {"sha256": checkpoint_hash, "epoch": selected_epoch, "config": copy.deepcopy(config),
                           "training_provenance": copy.deepcopy(training_provenance),
                           "data_hashes": copy.deepcopy(declaration["checkpoint_data_hashes"]), "supported_labels": list(range(64)),
                           "training_selection": copy.deepcopy(selection),
                           "checkpoint_selection": {"validation_metric": "mean_condition_interval_wasserstein_normalized",
                                                     "best_validation_score": .1}}
        history = [{"epoch": epoch, "generator_updates": 1184, "discriminator_updates": 1184,
                    "samples_seen": 1184 * 512, "validation_score": .1 if epoch == selected_epoch else .2}
                   for epoch in range(1, 27)]
        summary = {"epochs_completed": 26, "stopped_early": False, "test_used": False,
                   "generator_updates": 30784, "discriminator_updates": 30784, "samples_seen": 30784 * 512,
                   "best_epoch": selected_epoch, "best_validation_score": .1, "selection": copy.deepcopy(selection),
                   "provenance": copy.deepcopy(training_provenance), "data_hashes": copy.deepcopy(declaration["checkpoint_data_hashes"]),
                   "scaler": copy.deepcopy(scaler)}
        if arm == "candidate":
            summary["selection"].update(train_labels_sha256=fit["selected_labels_sha256"],
                                         train_profiles_sha256=fit["selected_profiles_sha256"], train_profiles_per_condition=counts)
            summary.update(quantile_decoder=copy.deepcopy(fit), scaler=copy.deepcopy(scaler))
        completions[arm] = {"summary": summary, "history": history, "config": copy.deepcopy(config),
                            "artifact_sha256": {"best.pt": checkpoint_hash, **{key: digest(f"{arm}-{seed}-{key}")
                                for key in ("last.pt", "config.json", "summary.json", "history.json")}}}
    inputs = {**{f"prepared/{key}": value for key, value in declaration["checkpoint_data_hashes"].items()},
              "reference_calibrator": original["calibration_sha256"], "baseline_bindings": pilot.BASELINE_BINDINGS_SHA256,
              "execution_declaration": declaration["_sha256"], "candidate_config": declaration["config_sha256"],
              "protocol": declaration["protocol_sha256"]}
    for arm in candidates:
        inputs.update({f"{arm}/{key}": value for key, value in completions[arm]["artifact_sha256"].items()})
    shared = {"schema_version": 1, "split": "val", "test_used": False, "training_seed": seed,
              **candidates, "training_completion": completions,
              "reference_calibration": {"sha256": original["calibration_sha256"], "calibration_seed": 7301,
                  "generation_seed": 2026, "quantile_knots": 1025,
                  "provenance": {"checkpoint_sha256": original["checkpoint_sha256"], "train_indices_sha256": digest("indices"),
                                 "prepared_data_hashes": declaration["prepared_data_hashes"], "calibration_seed": 7301,
                                 "calibration_samples_per_condition": 4096, "quantile_knots": 1025}},
              "baseline_binding": {"sha256": pilot.BASELINE_BINDINGS_SHA256, "record": original},
              "execution_declaration": {"sha256": declaration["_sha256"], "source_commit": declaration["source_commit"],
                                        "source_tree": declaration["source_tree"]},
              "endpoints": copy.deepcopy(pilot.EXPECTED_EVALUATION), "candidate_posthoc_calibration": False,
              "reference_calibration_refitted": False, "prepared_data_hashes": declaration["prepared_data_hashes"],
              "curation_manifest_sha256": declaration["checkpoint_data_hashes"]["curation_manifest.json"],
              "data_curation_status": "verified_curation_v2", "curation_metadata": {"version": 2},
              "source_sha256": pilot.RELEASE_SHA256, "target": "net", "units": "kWh/interval", "resolution": 96,
              "interval_hours": .25, "training_profiles": {key: 606060 for key in ("prepared", "candidate", "reference", "controls")},
              "seed": 2026, "samples_per_condition_cap": 100,
              "seeds": {"validation_selection": 2026, "generation": 2026, "controls": 2026, "reference_calibration": 7301},
              "validation_selection": {"samples_per_condition_cap": 100, "seed": 2026, "selected_profiles": 6400,
                  "condition_counts": {label: 100 for label in labels}, **{key: digest(key) for key in
                      ("selected_indices_sha256", "labels_sha256", "profiles_sha256", "meter_ids_sha256", "dates_sha256")}},
              "input_sha256": inputs, "post_input_sha256": copy.deepcopy(inputs), "provenance": provenance}
    physical, utility = copy.deepcopy(shared), copy.deepcopy(shared)
    physical["report_kind"], utility["report_kind"] = "quantile_decoder_pilot_physical", "quantile_decoder_pilot_utility"
    directions = np.random.default_rng(9182).normal(size=(128, 96))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    direction_meta = {"seed": 9182, "count": 128, "n_features": 96, "sha256": hashlib.sha256(directions.astype("<f8").tobytes()).hexdigest()}
    scenario = {"capacity_kwh": 10., "charge_power_kw": 4., "discharge_power_kw": 4., "initial_soc_kwh": 0.,
                "interval_hours": .25, "episode_hours": 24., "grid_charging_allowed": False, "battery_export_allowed": False,
                "eta_charge": float(np.sqrt(.9)), "eta_discharge": float(np.sqrt(.9))}
    physical.update(methods={}, joint_diagnostics={}, battery_diagnostics={})
    for method in pilot.PHYSICAL_METHODS:
        value = .8 if method == pilot.CANDIDATE_METHOD else 1.
        row = {key: value for key in pilot.GATED_METRICS[:3]}
        row["autocorrelation"] = {"1": {"wasserstein": value}}
        physical["methods"][method] = {"macro": copy.deepcopy(row), "per_condition": {label: copy.deepcopy(row) for label in labels},
                                       "counts": {side: {label: 100 for label in labels} for side in ("real", "generated")}}
        physical["joint_diagnostics"][method] = {"macro": value, "directions": direction_meta, "per_condition": {
            label: {"n_real": 100, "n_generated": 100, "mean_projected_wasserstein_kwh": value} for label in labels}}
        physical["battery_diagnostics"][method] = {"macro": {"import_w1": value}, "scenario": scenario, "per_condition": {
            label: {"n_real": 100, "n_generated": 100, "import_w1": value} for label in labels}}
    events = ("probability_zero", "probability_one", "probability_near_zero", "probability_near_one", "below_decoded_endpoint",
              "above_decoded_endpoint", "at_lower_decoded_endpoint", "at_upper_decoded_endpoint", "below_actual_train_minimum", "above_actual_train_maximum")
    physical["quantile_decoder"] = {"fit_metadata": copy.deepcopy(fit), "physical_dtype": "float32", "probability_dtype": "float64",
        "quantile_table_dtype": "float64", "near_saturation_threshold": 1e-6, "per_condition": {label: {
            "count": 100, "slot_event_counts": {event: [0] * 96 for event in events},
            "fractions_over_profile_slots": {event: 0. for event in events}, "fixed_zero_width_segment_fraction_by_slot": [0.] * 96,
            "sample_tied_observation_counts_by_slot": [0] * 96, "sample_largest_atom_counts_by_slot": [1] * 96,
            "fixed_constant_slots": []} for label in labels}}
    physical["frozen_samples"] = {"real_profiles_sha256": digest("profiles_sha256"), "labels_sha256": digest("labels_sha256"),
        "profiles_sha256": {method: digest(method) for method in pilot.PHYSICAL_METHODS},
        "profile_dtypes": {method: "float32" for method in pilot.PHYSICAL_METHODS}, "candidate_probabilities_sha256": digest("probabilities")}
    physical["baselines"] = {"gaussian_copula": {"quantile_knots": 1025, "shrinkage": .01},
                             "independent_marginals": {"quantile_knots": 1025, "shrinkage": 1.},
                             "exact_marginal_copula": {"quantile_knots": None, "shrinkage": .01}}
    map_events = ("source_below", "source_above", "cdf_zero", "cdf_one", "target_lower_endpoint", "target_upper_endpoint")
    physical["reference_map_diagnostics"] = {"fitting": {"per_condition": {label: {
        "train_count": counts[label], "calibration_count": 4096,
        "source_tie_mass_by_slot": [0.] * 96, "source_largest_atom_mass_by_slot": [1/4096] * 96,
        "source_knot_plateau_mass_by_slot": [0.] * 96} for label in labels}},
        "application": {"n_profiles": 6400, "per_condition": {label: {"count": 100,
            "slot_event_counts": {event: [0] * 96 for event in map_events},
            "fractions_over_profile_slots": {event: 0. for event in map_events}} for label in labels},
            "total_event_counts": {event: 0 for event in map_events}, "fractions_over_profile_slots": {event: 0. for event in map_events}}}
    neighbor = {"reference_scope": "sampled_training_profiles", "total_counts": {"train": 606060, "generated": 6400, "heldout": 6400},
                "sample_counts": {key: 1000 for key in ("train", "generated", "heldout")}, "seed": 2026}
    for sample in ("generated", "heldout"):
        neighbor[f"{sample}_exact_copy_fraction"] = 0.
        neighbor.update({f"{sample}_to_train_{stat}_rms_kwh": .1 for stat in ("mean", "median", "p05", "p95")})
    physical["nearest_neighbors"] = {method: copy.deepcopy(neighbor) for method in (pilot.CANDIDATE_METHOD, pilot.REFERENCE_METHOD)}
    utility["quantile_decoder"] = {"fit_metadata": copy.deepcopy(fit), "physical_dtype": "float32"}
    utility["utility_training_selection"] = {"cap_applied": False, "selected_profiles": 606060, "selected_indices_sha256": digest("indices"),
        "labels_sha256": digest("train-labels"), "condition_counts": counts, "real_profiles_sha256": digest("real-training")}
    utility["task"] = {"name": "within_day_observed_history_next_interval_linear_ridge", "resolution": 96, "lags": 12,
        "forecast_horizon_intervals": 1, "forecast_horizon_hours": .25, "windows_per_profile": 84, "target_slots_inclusive": [12, 95],
        "ridge_alpha": 1., "batch_profiles": 256, "ridge_objective": "sum_squared_error_plus_alpha_nonintercept_squared_norm"}
    utility["invariants"] = {"full_actual_train_required": True, "identical_training_label_order": True,
                            "fit_uses_validation_values": False, "candidate_posthoc_map": False, "reference_map_refitted": False}
    utility["methods"], utility["training"] = {}, {}
    for method in pilot.UTILITY_METHODS:
        utility["methods"][method] = {"macro": {"rmse_kwh": 1., "mae_kwh": .8}, "per_condition": {
            label: {"rmse_kwh": 1., "mae_kwh": .8, "n_profiles": 100} for label in labels}}
        if method != "persistence":
            utility["training"][method] = {"n_profiles": 606060, "n_windows": 606060 * 84, "condition_counts": counts,
                "labels_sha256": digest("train-labels"), "profiles_sha256": digest("real-training") if method == "training_real" else digest(method),
                "coefficients_sha256": digest(method + "coef"), "profile_dtype": "float32"}
    utility["copula"] = {"enabled": True, "quantile_knots": 1025, "shrinkage": .01, "source_profiles": 606060}
    return physical, utility, declaration


def test_complete_synthetic_paired_reports_validate_and_keep_endpoint_identity():
    physical, utility, declaration = synthetic()
    result = pilot.verify_pair(physical, utility, 11, declaration)
    assert result["metrics"]["candidate"]["ramp_wasserstein_kw"] == .8
    assert result["metrics"]["reference"]["ramp_wasserstein_kw"] == 1.
    assert result["physical"]["candidate_posthoc_calibration"] is False


@pytest.mark.parametrize("mutation,match", [
    (lambda p,u: p.update(test_used=True), "validation-only"),
    (lambda p,u: p.update(training_seed=44), "matching-seed"),
    (lambda p,u: p.update(candidate_posthoc_calibration=True), "endpoint"),
    (lambda p,u: p["endpoints"].update(baseline_calibration="new_fit"), "endpoint"),
    (lambda p,u: p["provenance"].update(git_dirty=True), "clean declared"),
    (lambda p,u: p["provenance"]["source_files_sha256"].update({"src/fluvius_cgan/copula.py": digest("old-copula")}), "source fingerprints"),
    (lambda p,u: p["prepared_data_hashes"].update({"val.npz": digest("other-data")}), "prepared data"),
    (lambda p,u: p["validation_selection"].update(selected_profiles=6399), "validation rows"),
    (lambda p,u: p["post_input_sha256"].update(reference_calibrator=digest("changed-map")), "changed during"),
    (lambda p,u: u["validation_selection"].update(meter_ids_sha256=digest("different-meters")), "paired physical/utility"),
    (lambda p,u: u["utility_training_selection"].update(selected_profiles=10000), "exact TRAIN"),
    (lambda p,u: u["invariants"].update(fit_uses_validation_values=True), "invariant"),
    (lambda p,u: u["methods"].pop("training_resampling"), "utility method"),
    (lambda p,u: p["methods"][pilot.CANDIDATE_METHOD]["per_condition"].pop("63"), "64 conditions"),
    (lambda p,u: p["methods"][pilot.CANDIDATE_METHOD]["macro"].update(ramp_wasserstein_kw=.5), "equal-condition mean"),
    (lambda p,u: p["joint_diagnostics"][pilot.CANDIDATE_METHOD]["directions"].update(sha256=digest("other-directions")), "direction hash"),
    (lambda p,u: p["frozen_samples"]["profiles_sha256"].pop("gaussian_copula"), "control hashes"),
    (lambda p,u: p["quantile_decoder"]["per_condition"]["0"]["fractions_over_profile_slots"].update(probability_zero=.1), "event fraction"),
])
def test_missing_or_mismatched_evidence_fails_closed(mutation, match):
    physical, utility, declaration = synthetic()
    mutation(physical, utility)
    with pytest.raises(ValueError, match=match):
        pilot.verify_pair(physical, utility, 11, declaration)


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["candidate"]["config"].update(latent_dim=96), "configuration"),
    (lambda r: r["training_completion"]["candidate"]["summary"].update(generator_updates=30783), "30,784"),
    (lambda r: r["training_completion"]["reference"]["history"].pop(), "26-epoch history"),
    (lambda r: r["training_completion"]["candidate"]["summary"]["selection"].update(validation_noise_sha256=digest("wrong-noise")), "rows/noise"),
    (lambda r: r["reference"].update(sha256=digest("not-original")), "original pinned"),
    (lambda r: r["reference_calibration"]["provenance"].update(calibration_samples_per_condition=8192), "calibration binding"),
])
def test_consistently_wrong_evidence_in_both_reports_is_still_rejected(mutation, match):
    physical, utility, declaration = synthetic()
    mutation(physical)
    mutation(utility)
    with pytest.raises(ValueError, match=match):
        pilot.verify_pair(physical, utility, 11, declaration)


def test_cross_seed_control_hash_change_cannot_be_hidden_by_equal_metrics():
    declaration = contract()
    pairs = [pilot.verify_pair(*synthetic(seed, declaration)[:2], seed, declaration) for seed in pilot.SEEDS]
    pilot._compatibility(pairs)
    pairs[1]["physical"]["frozen_samples"]["profiles_sha256"]["gaussian_copula"] = digest("other-draw")
    with pytest.raises(ValueError, match="control hash"):
        pilot._compatibility(pairs)


def test_duplicate_json_fields_and_nonfinite_tokens_rejected():
    for value in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
        with pytest.raises(ValueError):
            pilot._json_bytes(value, "synthetic")


def declaration_file(tmp_path, monkeypatch):
    value = contract()
    config = json.dumps(value["_config"], sort_keys=True).encode()
    protocol = b"Synthetic prospective protocol.\n"
    bindings = pilot.BASELINE_BINDINGS.read_bytes()
    blobs = {name: name.encode() for name in pilot.SOURCE_FILES}
    assessor = "scripts/assess_quantile_decoder_pilot.py"
    blobs[assessor] = Path(pilot.__file__).read_bytes()
    value["source_files_sha256"][assessor] = hashlib.sha256(blobs[assessor]).hexdigest()
    blobs.update({pilot.CONFIG_PATH: config, str(pilot.PROTOCOL_DOCUMENT.relative_to(pilot.ROOT)): protocol,
                  str(pilot.BASELINE_BINDINGS.relative_to(pilot.ROOT)): bindings})
    value["config_sha256"] = hashlib.sha256(config).hexdigest()
    value["protocol_sha256"] = hashlib.sha256(protocol).hexdigest()
    def fake_git(*args):
        if args[0] == "rev-parse":
            return ((value["source_commit"] if args[-1].endswith("^{commit}") else value["source_tree"]) + "\n").encode()
        assert args[0] == "show"
        return blobs[args[1].split(":", 1)[1]]
    monkeypatch.setattr(pilot, "_git", fake_git)
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps({key: item for key, item in value.items() if not key.startswith("_")}))
    return path, blobs


def test_declaration_is_verified_against_published_git_blobs_without_profile_io(tmp_path, monkeypatch):
    path, _ = declaration_file(tmp_path, monkeypatch)
    result = pilot.load_declaration(path)
    assert result["_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["baseline_bindings_sha256"] == pilot.BASELINE_BINDINGS_SHA256


@pytest.mark.parametrize("mutation,match", [
    (lambda value: value.update(seeds=[11, 22, 44]), "11, 22, 33"),
    (lambda value: value["budget"].update(epochs=25), "budget"),
    (lambda value: value["evaluation"].update(candidate_posthoc_calibration=True), "evaluation settings"),
    (lambda value: value["source_files_sha256"].pop("src/fluvius_cgan/copula.py"), "inventory"),
    (lambda value: value["source_files_sha256"].update({"src/fluvius_cgan/copula.py": digest("historical")}), "source blob"),
    (lambda value: value.update(baseline_bindings_sha256=digest("new-comparator")), "baseline_bindings_sha256"),
    (lambda value: value["prepared_data_hashes"].update({"train.npz": digest("different-training")}), "declared data"),
])
def test_incompatible_execution_declaration_is_rejected(tmp_path, monkeypatch, mutation, match):
    path, _ = declaration_file(tmp_path, monkeypatch)
    value = json.loads(path.read_text())
    mutation(value)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=match):
        pilot.load_declaration(path)


def test_assessor_reads_only_two_report_files_and_never_report_input_paths(tmp_path, monkeypatch):
    physical, utility, declaration = synthetic()
    physical["input_paths"] = utility["input_paths"] = {"malicious": "/must/not/open/test.npz"}
    (tmp_path / pilot.PHYSICAL_FILE).write_text(json.dumps(physical))
    (tmp_path / pilot.UTILITY_FILE).write_text(json.dumps(utility))
    original = Path.open
    allowed = {pilot.PHYSICAL_FILE, pilot.UTILITY_FILE}
    def guarded(path, *args, **kwargs):
        assert path.name in allowed, f"unexpected artifact access: {path}"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    assert pilot.verify_run(11, tmp_path, declaration)["seed"] == 11


def test_missing_report_retains_seed_and_marks_incomplete(tmp_path, monkeypatch):
    declaration = contract()
    monkeypatch.setattr(pilot, "load_declaration", lambda path: declaration)
    def verify(seed, directory, declaration):
        if seed == 22:
            raise FileNotFoundError("missing full-TRAIN utility")
        return pilot.verify_pair(*synthetic(seed, declaration)[:2], seed, declaration)
    monkeypatch.setattr(pilot, "verify_run", verify)
    result = pilot.assess(tmp_path, "synthetic-declaration.json")
    assert result["status"] == "incomplete" and not result["complete"] and result["pass"] is None
    assert not result["promotion_eligible"] and result["gates"] == {} and len(result["pairs"]) == 2
    assert result["errors"][0]["seed"] == 22 and len(result["requested_runs"]) == 3
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("status,code", [("pass", 0), ("fail", 1), ("incomplete", 2)])
def test_cli_preserves_every_outcome_and_never_overwrites(tmp_path, monkeypatch, status, code):
    declaration = contract()
    monkeypatch.setattr(pilot, "load_declaration", lambda path: declaration)
    def verify(seed, directory, declaration):
        if status == "incomplete" and seed == 33:
            raise ValueError("missing report")
        result = pilot.verify_pair(*synthetic(seed, declaration)[:2], seed, declaration)
        if status == "fail":
            result["metrics"]["candidate"]["ramp_wasserstein_kw"] = .99
        return result
    monkeypatch.setattr(pilot, "verify_run", verify)
    output, markdown = tmp_path / "assessment.json", tmp_path / "assessment.md"
    args = ["--candidate-root", str(tmp_path / "candidate"), "--declaration", "synthetic-declaration.json",
            "--output", str(output), "--markdown", str(markdown)]
    assert pilot.main(args) == code
    assert json.loads(output.read_text())["status"] == status
    assert f"**{status}**" in markdown.read_text()
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        pilot.main(args)
    assert output.read_bytes() == before
