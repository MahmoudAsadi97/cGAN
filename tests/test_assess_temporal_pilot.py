"""Synthetic report-only promotion gates; no real data, training or generation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("assess_temporal_pilot", Path(__file__).resolve().parents[1] /
                                            "scripts" / "assess_temporal_pilot.py")
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def synthetic(arm="candidate", seed=11):
    """Post-collector evidence: byte/map binding is covered by collector tests.

    No 606k-row arrays or model weights are fabricated; these are scalar schemas
    for testing the additional fixed-protocol and decision layer independently.
    """
    labels = sorted(pilot.CONDITIONS, key=int)
    counts = {label: 9470 if int(label) < 44 else 9469 for label in labels}
    selection = {"train_profiles_available": 606060, "train_profiles_used": 606060,
                 "train_indices_sha256": digest("indices"), "validation_noise_sha256": digest(f"noise{seed}")}
    data = {key: digest(key) for key in pilot.collector.PREPARED_FILES}
    sources = {key: digest(key) for key in set((*pilot.collector.TRAINING_CORE_FILES,
                                             *pilot.EVALUATION_CORE, *pilot.UTILITY_CORE))}
    training_sources = {**sources, "src/fluvius_cgan/training.py": digest(f"training-{arm}")}
    training_provenance = {"source_files_sha256": training_sources}
    scale = {"n_profiles": 606060, "n_features": 96, "n_differences": 95,
             "train_indices_sha256": selection["train_indices_sha256"], "train_npz_sha256": data["train.npz"],
             "fitting_scope": "actual_gradient_training_subset_only", "centering": False, "wraparound": False,
             "estimate_float64": 1., "used_float32": 1., "condition_counts": counts,
             "condition_mean_squared_differences": {label: 1. for label in labels}}
    if arm == "candidate":
        training_provenance["derivative_swd"] = {"enabled": True, "weight": 1., "scale": scale,
                                                "projection_seed": seed + 5, "num_projections": 128}
    directions = np.random.default_rng(9182).normal(size=(128, 96))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    direction_meta = {"count": 128, "seed": 9182, "n_features": 96,
                      "sha256": hashlib.sha256(directions.astype("<f8").tobytes()).hexdigest()}
    scenario = {"capacity_kwh": 10., "charge_power_kw": 4., "discharge_power_kw": 4., "initial_soc_kwh": 0.,
                "interval_hours": .25, "episode_hours": 24., "grid_charging_allowed": False,
                "battery_export_allowed": False, "eta_charge": float(np.sqrt(.9)), "eta_discharge": float(np.sqrt(.9))}
    methods, joint, battery = {}, {}, {}
    for method in pilot.collector.REQUIRED_METHODS:
        value = .8 if method == "calibrated_cgan" and arm == "candidate" else 1.
        row = {key: value for key in pilot.collector.METRICS}
        methods[method] = {"joint_and_marginal_macro": row, "macro": copy.deepcopy(row),
                           "per_condition": {label: copy.deepcopy(row) for label in labels}}
        joint[method] = {"macro": value, "directions": direction_meta,
                         "per_condition": {label: {"mean_projected_wasserstein_kwh": value,
                                                   "n_real": 100, "n_generated": 100} for label in labels}}
        battery[method] = {"scenario": scenario, "macro": {"grid_import_wasserstein_kwh": value},
                           "per_condition": {label: {"grid_import_wasserstein_kwh": value,
                                                     "n_real": 100, "n_generated": 100} for label in labels}}
    utility_methods, training = {}, {}
    for method in pilot.UTILITY_METHODS:
        utility_methods[method] = {"macro": {"rmse_kwh": 1., "mae_kwh": .8},
                                   "per_condition": {label: {"rmse_kwh": 1., "mae_kwh": .8, "n_profiles": 100}
                                                     for label in labels}}
        training[method] = {"n_profiles": 606060, "n_windows": 606060 * 84, "condition_counts": counts,
                            "labels_sha256": digest("train-labels"), "profiles_sha256": digest(method),
                            "coefficients_sha256": digest(method + "coef")}
    task = {"name": "within_day_observed_history_next_interval_linear_ridge", "resolution": 96, "lags": 12,
            "forecast_horizon_intervals": 1, "forecast_horizon_hours": .25, "target_slots_inclusive": [12, 95],
            "ridge_alpha": 1.}
    utility = {"config": {"seed": 2026, "samples_per_condition": 100, "max_train_profiles": None,
                           "copula": True, "calibrator": f"{arm}/seed_{seed}/calibration.npz"},
               "task": task, "methods": utility_methods, "training": training,
               "source_training_pool": {"prepared_profiles": 606060, "actual_gan_profiles": 606060,
                                         "condition_counts": counts, "labels_sha256": digest("train-labels"),
                                         "profiles_sha256": digest("training_real")},
               "utility_training_selection": {"cap_applied": False, "selected_profiles": 606060,
                                               "source_pool_equals_real_ridge_fit_rows": True,
                                               "selected_indices_sha256": selection["train_indices_sha256"],
                                               "condition_counts": counts, "labels_sha256": digest("train-labels")},
               "provenance": {"source_file_sha256": sources}}
    run = {"name": f"{arm}_seed_{seed}", "training_seed": seed, "arm": arm,
           "config": pilot.expected_config(arm, seed), "epochs_completed": 26, "stopped_early": False,
           "selection_metric": "mean_condition_interval_wasserstein_normalized", "training_selection": selection,
           "prepared_data_hashes": data, "checkpoint_data_hashes": {**data, "curation_manifest.json": digest("curation")},
           "data_curation_status": "curation_metadata_present_not_reaudited", "curation_metadata": {"curated": True},
           "group_settings": {"evaluation_seed": 2026, "samples_per_condition": 100, "calibration_seed": 7301,
                               "calibration_samples_per_condition": 4096, "quantile_knots": 1025},
           "validation_selection": {"condition_counts": {label: 100 for label in labels}},
           "diagnostic_settings": {"joint_projection_directions": direction_meta, "battery_scenario": scenario},
           "methods": methods, "joint_diagnostics": joint, "battery_diagnostics": battery,
           "provenance": {"training_checkpoint": training_provenance,
                          "calibration_evaluation": {"source_files_sha256": sources, "evaluation_script_sha256": digest("eval"),
                                                     "comparison_helper_sha256": sources["scripts/compare_checkpoints.py"]},
                          "raw_evaluation": None}, "utility": utility,
           "frozen_samples": {"labels_sha256": digest("val-labels"), "real_profiles_sha256": digest("real"),
                              "profiles_sha256": {method: digest(method) for method in methods}}}
    # Saved evaluation reports compact the calibrator's verbose API diagnostics
    # into these exact vectors; no `slots` list is present in report JSON.
    fitting = {"per_condition": {label: {
        "train_count": counts[label], "calibration_count": 4096,
        "source_tie_mass_by_slot": [0.] * 96, "source_largest_atom_mass_by_slot": [1 / 4096] * 96,
        "source_knot_plateau_mass_by_slot": [0.] * 96,
        "source_constant_slots": [], "target_constant_slots": []} for label in labels}}
    application = {"n_profiles": 6400, "per_condition": {label: {
        "count": 100, "slot_event_counts": {event: [1] * 96 for event in pilot.CALIBRATION_EVENTS},
        "fractions_over_profile_slots": {event: .01 for event in pilot.CALIBRATION_EVENTS}} for label in labels},
        "total_event_counts": {event: 64 * 96 for event in pilot.CALIBRATION_EVENTS},
        "fractions_over_profile_slots": {event: .01 for event in pilot.CALIBRATION_EVENTS}}
    calibrated = {"target": "net", "units": "kWh/interval", "resolution": 96, "interval_hours": .25,
                  "checkpoint": {"supported_labels": list(range(64))}, "frozen_samples": {"profile_shape": [6400, 96]},
                  "calibration": {"fitting_diagnostics": fitting, "application_diagnostics": application},
                  "methods": {method: {"counts": {"real": {label: 100 for label in labels},
                                                  "generated": {label: 100 for label in labels}}} for method in methods}}
    report = {"target": "net", "units": "kWh/interval", "invariants": {
        "training_condition_counts_match": True, "training_label_order_matches": True,
        "identical_validation_windows": True, "fit_uses_validation_values": False,
        "prediction_uses_observed_real_past_lags": True, "prediction_uses_its_own_target_or_future_values": False,
        "cross_day_windows": False, "calibration_fitted_on_current_generation_batch": False}}
    summary = {"generator_updates": 30784, "discriminator_updates": 30784, "derivative_swd_scale": scale}
    history = [{"epoch": epoch, "generator_updates": 1184, "discriminator_updates": 1184} for epoch in range(1, 27)]
    return run, summary, history, calibrated, report


def verified(arm, seed):
    return pilot._verify_protocol(*synthetic(arm, seed), arm, seed)


@pytest.fixture
def six_runs():
    return [verified(arm, seed) for arm in ("candidate", "baseline") for seed in pilot.SEEDS]


def change_metric(run, metric, value):
    method = run["methods"]["calibrated_cgan"]
    method["joint_and_marginal_macro"][metric] = value
    for row in method["per_condition"].values():
        row[metric] = value


def test_fixed_valid_protocol_and_different_training_implementation_pass(six_runs):
    pilot._compatibility(six_runs)
    gates = pilot.measured_gates(six_runs)
    assert all(row["pass"] for row in gates.values())
    assert gates["mean_ramp_wasserstein_kw"]["relative_change"] == pytest.approx(-.2)
    assert gates["utility_each_seed_within_both_references"]["three_seed_means_rmse_kwh"]["training_real"] == 1.


@pytest.mark.parametrize("mutation,match", [
    (lambda f,a: f["per_condition"]["0"]["source_tie_mass_by_slot"].pop(), "96 finite slot masses"),
    (lambda f,a: f["per_condition"]["0"].update(train_count=1), "fitting diagnostic counts"),
    (lambda f,a: f["per_condition"]["0"].update(source_constant_slots=[96]), "valid slot indices"),
    (lambda f,a: a["per_condition"]["0"]["slot_event_counts"]["cdf_zero"].pop(), "96 valid slot counts"),
    (lambda f,a: a["per_condition"]["0"]["slot_event_counts"]["cdf_zero"].__setitem__(0, 101), "96 valid slot counts"),
    (lambda f,a: a["per_condition"]["0"]["fractions_over_profile_slots"].update(cdf_zero=.5), "per-condition fraction"),
    (lambda f,a: a["total_event_counts"].update(cdf_zero=0), "global event totals"),
    (lambda f,a: a["fractions_over_profile_slots"].update(cdf_zero=.5), "global fractions"),
])
def test_saved_compact_diagnostic_vectors_and_totals_are_validated(mutation, match):
    values = synthetic()
    calibration = values[3]["calibration"]
    assert "slots" not in calibration["fitting_diagnostics"]["per_condition"]["0"]
    mutation(calibration["fitting_diagnostics"], calibration["application_diagnostics"])
    with pytest.raises(ValueError, match=match):
        pilot._verify_protocol(*values, "candidate", 11)


def test_mean_improvement_does_not_excuse_one_nonimproving_seed(six_runs):
    for run, value in zip(six_runs[:3], (.5, .5, 1.)):
        change_metric(run, "ramp_wasserstein_kw", value)
    gates = pilot.measured_gates(six_runs)
    assert gates["mean_ramp_wasserstein_kw"]["pass"]
    assert not gates["ramp_lower_in_every_paired_seed"]["pass"]


def test_every_seed_improves_but_mean_under_ten_percent_fails(six_runs):
    for run in six_runs[:3]:
        change_metric(run, "ramp_wasserstein_kw", .95)
    gates = pilot.measured_gates(six_runs)
    assert gates["ramp_lower_in_every_paired_seed"]["pass"]
    assert not gates["mean_ramp_wasserstein_kw"]["pass"]


@pytest.mark.parametrize("metric", pilot.GATED_METRICS[1:])
def test_each_guardrail_can_fail_independently(six_runs, metric):
    for run in six_runs[:3]:
        if metric.startswith("projected"):
            run["joint_diagnostics"]["calibrated_cgan"]["macro"] = 1.1001
        else:
            change_metric(run, metric, 1.1001)
    gates = pilot.measured_gates(six_runs)
    assert not gates[f"mean_{metric}"]["pass"]
    assert gates["mean_ramp_wasserstein_kw"]["pass"]


@pytest.mark.parametrize("reference", ["training_real", "training_resampling"])
def test_each_seed_must_pass_both_utility_references(six_runs, reference):
    six_runs[1]["utility"]["methods"][reference]["macro"]["rmse_kwh"] = .5
    gate = pilot.measured_gates(six_runs)["utility_each_seed_within_both_references"]
    assert not gate["pass"] and not gate["per_seed"][1]["comparisons"][reference]["pass"]
    assert gate["per_seed"][0]["comparisons"][reference]["pass"]


def test_inclusive_boundaries_and_zero_reference_are_explicit():
    assert pilot._comparison(.9, 1., .9)["pass"]
    assert pilot._comparison(1.1, 1., 1.1)["pass"]
    assert pilot._comparison(0., 0., 1.1) == {"candidate": 0., "reference": 0., "maximum_ratio": 1.1,
        "limit": 0., "difference": 0., "ratio": None, "relative_change": None, "pass": True}
    assert not pilot._comparison(.01, 0., 1.1)["pass"]


@pytest.mark.parametrize("mutation,match", [
    (lambda r,s,h,c,u: r["config"].update(swd_weight=9.), "configuration"),
    (lambda r,s,h,c,u: s.update(discriminator_updates=30783), "discriminator_updates"),
    (lambda r,s,h,c,u: h[0].update(generator_updates=1183), "generator_updates"),
    (lambda r,s,h,c,u: r.update(epochs_completed=25), "26-epoch"),
    (lambda r,s,h,c,u: r["group_settings"].update(calibration_samples_per_condition=8192), "primary setting"),
    (lambda r,s,h,c,u: r["group_settings"].update(calibration_seed=7302), "primary setting"),
    (lambda r,s,h,c,u: r["group_settings"].update(samples_per_condition=1000), "primary setting"),
    (lambda r,s,h,c,u: r["diagnostic_settings"]["joint_projection_directions"].update(sha256=digest("wrong")), "direction fingerprint"),
    (lambda r,s,h,c,u: r["diagnostic_settings"]["battery_scenario"].update(capacity_kwh=5.), "battery scenario"),
    (lambda r,s,h,c,u: r["utility"]["config"].update(max_train_profiles=10000), "uncapped TRAIN"),
    (lambda r,s,h,c,u: r["utility"]["utility_training_selection"].update(selected_indices_sha256=digest("wrong")), "fitting rows"),
    (lambda r,s,h,c,u: u["invariants"].update(fit_uses_validation_values=True), "utility invariant"),
    (lambda r,s,h,c,u: r["methods"]["calibrated_cgan"]["per_condition"].pop("63"), "64 conditions"),
    (lambda r,s,h,c,u: r["methods"]["calibrated_cgan"]["joint_and_marginal_macro"].update(ramp_wasserstein_kw=.5), "equal-condition"),
    (lambda r,s,h,c,u: s["derivative_swd_scale"].update(train_indices_sha256=digest("bad")), "TRAIN pool"),
])
def test_incomplete_or_incompatible_protocol_is_rejected(mutation, match):
    values = synthetic()
    mutation(*values)
    with pytest.raises(ValueError, match=match):
        pilot._verify_protocol(*values, "candidate", 11)


def test_cross_run_reference_and_source_changes_fail_closed(six_runs):
    six_runs[0]["frozen_samples"]["real_profiles_sha256"] = digest("changed")
    with pytest.raises(ValueError, match="frozen reference"):
        pilot._compatibility(six_runs)
    six_runs[0]["frozen_samples"]["real_profiles_sha256"] = digest("real")
    six_runs[0]["pilot_sources"]["training"]["src/fluvius_cgan/training.py"] = digest("third-implementation")
    with pytest.raises(ValueError, match="within candidate"):
        pilot._compatibility(six_runs)


def fake_verifier(arm, seed, directory):
    return verified(arm, seed)


def test_incomplete_report_retains_errors_and_does_not_drop_missing_seed(monkeypatch, tmp_path):
    def missing(arm, seed, directory):
        if arm == "candidate" and seed == 22:
            raise ValueError("saved calibration fingerprint mismatch")
        return verified(arm, seed)
    monkeypatch.setattr(pilot, "verify_run", missing)
    report = pilot.assess(tmp_path / "candidate", tmp_path / "baseline")
    assert report["status"] == "incomplete" and report["complete"] is False and report["pass"] is None
    assert not report["promotion_eligible"] and not report["gates"] and len(report["runs"]) == 5
    assert "fingerprint mismatch" in report["errors"][0]["error"]
    json.dumps(report, allow_nan=False)


def test_collector_binding_failure_is_retained_and_no_profile_access(monkeypatch, tmp_path):
    calls = []
    def rejecting(name, directory):
        calls.append(name)
        raise ValueError("checkpoint/report/map binding mismatch")
    monkeypatch.setattr(pilot.collector, "collect_run", rejecting)
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name not in {"train.npz", "val.npz", "test.npz"}
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    result = pilot.assess(tmp_path / "c", tmp_path / "b")
    assert len(calls) == 6 and len(result["errors"]) == 6 and result["status"] == "incomplete"


@pytest.mark.parametrize("status,code", [("pass", 0), ("fail", 1), ("incomplete", 2)])
def test_cli_writes_every_outcome_and_never_overwrites(monkeypatch, tmp_path, status, code):
    def verify(arm, seed, directory):
        if status == "incomplete" and seed == 33:
            raise ValueError("missing required utility")
        run = verified(arm, seed)
        if status == "fail" and arm == "candidate":
            change_metric(run, "ramp_wasserstein_kw", .99)
        return run
    monkeypatch.setattr(pilot, "verify_run", verify)
    output, md = tmp_path / "assessment.json", tmp_path / "assessment.md"
    args = ["--candidate-root", str(tmp_path / "c"), "--baseline-root", str(tmp_path / "b"),
            "--output", str(output), "--markdown", str(md)]
    assert pilot.main(args) == code
    assert json.loads(output.read_text())["status"] == status
    assert f"**{status}**" in md.read_text()
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        pilot.main(args)
    assert output.read_bytes() == original


def test_identical_arm_paths_never_authorize_promotion(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "verify_run", fake_verifier)
    report = pilot.assess(tmp_path, tmp_path)
    assert report["status"] == "incomplete" and "distinct" in report["errors"][0]["error"]
