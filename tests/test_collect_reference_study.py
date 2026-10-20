"""Saved-report contracts only: no prepared arrays, model fitting or inference."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "collect_reference_study", Path(__file__).resolve().parents[1] / "scripts/collect_reference_study.py")
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else str(value).encode()).hexdigest()


def declared_inventory():
    seeds = tuple(map(str, collector.SEEDS))
    return {
        "schema_version": 1, "candidate_kind": "swd10",
        "prepared96": "prepared96", "prepared24": "prepared24",
        "runs": {arm: {seed: f"runs/{arm}/{seed}" for seed in seeds} for arm in collector.ARMS},
        "development_evidence": {
            "native": {arm: {seed: {role: f"{arm}/{seed}/{role}.json" for role in collector.NATIVE_ROLES}
                             for seed in seeds} for arm in collector.ARMS},
            "common_hourly": {seed: {role: f"hourly/{seed}/{role}.json" for role in collector.HOURLY_ROLES}
                              for seed in seeds},
            "paired": {contrast: {seed: f"paired/{contrast}/{seed}.json" for seed in seeds}
                       for contrast in collector.CONTRASTS},
            "additional": [],
        },
    }


def counts(total):
    base, extra = divmod(total, 64)
    return {str(i): base + int(i < extra) for i in range(64)}


def physical_result(value=1., *, cap=100, width=96):
    metrics = {name: value * (i + 1) for i, name in enumerate(collector.METRICS[:-1])}
    metrics["autocorrelation"] = {"1": {"wasserstein": value / 10}}
    row = {**metrics, "n_real": cap, "n_generated": cap, "n_intervals": width,
           "interval_hours": 24 / width}
    return {"macro": deepcopy(metrics), "per_condition": {str(i): deepcopy(row) for i in range(64)}}


def utility_report(*, unconditional=False, width=96, cap=100):
    lags = width // 8
    names = {"training_real", "training_cgan", "training_resampling", "training_copula", "persistence"}
    if not unconditional:
        names.add("training_calibrated_cgan")
    labels_hash = sha("complete ordered TRAIN labels")
    source_indices = sha(np.arange(collector.TRAIN_ROWS, dtype="<i8").tobytes())
    fits = {}
    methods = {}
    condition_counts = counts(collector.TRAIN_ROWS)
    for name in sorted(names):
        row = {"rmse_kwh": 2., "mae_kwh": 1., "n_profiles": cap, "n_windows": cap * (width - lags)}
        methods[name] = {"macro": {"rmse_kwh": 2., "mae_kwh": 1.},
                         "per_condition": {str(i): dict(row) for i in range(64)}}
        if name != "persistence":
            fits[name] = {
                "n_profiles": collector.TRAIN_ROWS, "n_windows": collector.TRAIN_ROWS * (width - lags),
                "n_features": lags + 69, "resolution": width, "lags": lags, "ridge_alpha": 1.,
                "ridge_objective": "sum_squared_error_plus_alpha_nonintercept_squared_norm",
                "feature_scaling": "none; lagged energy remains in native kWh/interval",
                "condition_counts": dict(condition_counts), "profiles_sha256": sha(name),
                "labels_sha256": labels_hash, "coefficients_sha256": sha("coefficients:" + name),
                "validation_errors_sha256": sha(f"errors:{name}:{cap}"),
            }
    map_provenance = {"checkpoint_sha256": sha("checkpoint")}
    report = {
        "split": "val", "test_used": False, "units": "kWh/interval", "target": "net",
        "config": {"seed": 2026, "samples_per_condition": cap, "max_train_profiles": None,
                   "copula": True, "calibrator": None if unconditional else "calibration.npz"},
        "seeds": {"validation_selection": 2026, "utility_training_cap": 2126, "cgan_generation": 2026,
                  "conditional_resampling": 2026, "copula_generation": 2026,
                  "calibration_reference": None if unconditional else 7301},
        "task": {
            "name": "within_day_observed_history_next_interval_linear_ridge", "resolution": width,
            "lags": lags, "forecast_horizon_intervals": 1, "forecast_horizon_hours": 24 / width,
            "ridge_alpha": 1., "target_slots_inclusive": [lags, width - 1],
            "feature_names": ([f"lag_{lag}_intervals" for lag in range(lags, 0, -1)] +
                              ["clock_sin_1", "clock_cos_1", "clock_sin_2", "clock_cos_2"] +
                              [f"condition_{i}" for i in range(64)] + ["intercept"]),
            "ridge_objective": "sum_squared_error_plus_alpha_nonintercept_squared_norm",
            "feature_scaling": "none; native kWh/interval", "batch_profiles": 256,
        },
        "source_training_pool": {
            "prepared_profiles": collector.TRAIN_ROWS, "actual_gan_profiles": collector.TRAIN_ROWS,
            "actual_indices_sha256": source_indices, "profiles_sha256": fits["training_real"]["profiles_sha256"],
            "labels_sha256": labels_hash, "condition_counts": dict(condition_counts),
        },
        "utility_training_selection": {
            "cap_applied": False, "selected_profiles": collector.TRAIN_ROWS,
            "condition_counts": dict(condition_counts), "labels_sha256": labels_hash,
            "selected_indices_sha256": source_indices, "source_pool_equals_real_ridge_fit_rows": True,
        },
        "validation_selection": {
            "samples_per_condition_cap": cap, "condition_counts": {str(i): cap for i in range(64)},
            "selected_indices_sha256": sha(f"indices:{cap}"), "profiles_sha256": sha(f"real:{cap}"),
            "labels_sha256": sha(np.repeat(np.arange(64, dtype="<i8"), cap).tobytes()),
            "selected_profiles": cap * 64, "selected_meters": 353, "meter_ids_sha256": sha(f"meters:{cap}"),
        },
        "calibration": {"enabled": not unconditional, "sha256": sha("map"), "provenance": map_provenance},
        "copula": {"enabled": True, "quantile_knots": None if unconditional else 1025, "shrinkage": .01},
        "training": fits, "methods": methods,
        "invariants": {
            "training_condition_counts_match": True, "training_label_order_matches": True,
            "identical_validation_windows": True, "baseline_profile_pools_match_gan_gradient_training_subset": True,
            "fit_uses_validation_values": False, "prediction_uses_observed_real_past_lags": True,
            "prediction_uses_its_own_target_or_future_values": False, "cross_day_windows": False,
            "post_generation_clipping_applied": False, "calibration_fitted_on_current_generation_batch": False,
        },
        "provenance": {"source_files_sha256": dict(collector.TRAINING_SOURCE_HASHES)},
    }
    run = {"arm": "unconditional96" if unconditional else ("bce24" if width == 24 else "candidate96"),
           "width": width, "training_selection": {"train_indices_sha256": source_indices}}
    map_record = {"sha256": sha("map"), "metadata": {"provenance": map_provenance}}
    return report, run, map_record


def bootstrap(*, utility=False, invalid=0):
    metrics = ("rmse_kwh", "mae_kwh") if utility else collector.METRICS
    points = [{name: 1. + i for i, name in enumerate(metrics)}, {name: 2. + i for i, name in enumerate(metrics)}]
    status = "complete" if not invalid else ("withheld_missing_conditions" if utility else "unavailable_missing_condition_draws")
    result = {"n_bootstrap": 500, "confidence": .95, "seed": 42027 if utility else 2026,
              "n_valid_bootstrap": 500 - invalid, "n_invalid_bootstrap": invalid,
              "interval_status": status,
              "point_estimates": dict(zip(("candidate" if utility else "generated", "comparator"), points)),
              "intervals": {name: {"estimate": -1., "lower": None if invalid else -2.,
                                    "upper": None if invalid else 0.} for name in metrics}}
    return result, metrics, points


def test_inventory_requires_all_four_arms_five_seeds_and_all_roles():
    spec = declared_inventory()
    assert len(collector.inventory(spec)) == 4 * 5 * 5 + 5 * 4 + 4 * 5
    for mutation in (
        lambda x: x["runs"]["bce96"].pop("55"),
        lambda x: x["development_evidence"]["native"]["unconditional96"]["11"].pop("precision"),
        lambda x: x["development_evidence"]["common_hourly"]["33"].pop("utility_precision"),
        lambda x: x["development_evidence"]["paired"].pop("resolution_calibrated"),
        lambda x: x.update(candidate_kind="best_of_seeds"),
    ):
        broken = deepcopy(spec)
        mutation(broken)
        with pytest.raises(ValueError):
            collector.inventory(broken)


@pytest.mark.parametrize("filename", ["generated_profiles.npz", "raw_release.zip", "extra_checkpoint.pt"])
def test_extra_evidence_cannot_read_profile_or_binary_archives(tmp_path, monkeypatch, filename):
    spec = declared_inventory()
    spec["development_evidence"]["additional"] = [filename]
    (tmp_path / filename).write_bytes(b"unread binary sentinel")
    monkeypatch.setattr(collector, "sha256_file", lambda path: pytest.fail("content read before evidence type preflight"))
    with pytest.raises(ValueError, match="JSON reports"):
        collector.collect(spec, base=tmp_path)


@pytest.mark.parametrize("name", ["train.npz", "val.npz", "test.npz"])
def test_preflight_refuses_profile_content_and_aliases(tmp_path, monkeypatch, name):
    directory = tmp_path / "prepared96"
    directory.mkdir()
    protected = directory / name
    protected.write_bytes(b"opaque sentinel; must never be opened")
    hardlink = tmp_path / "artifact.json"
    os.link(protected, hardlink)
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(protected)
    inputs = collector.Inputs(tmp_path, "prepared96", "prepared24")
    monkeypatch.setattr(collector, "sha256_file", lambda path: pytest.fail("protected profile content was read"))
    for path in (protected, hardlink, symlink):
        with pytest.raises(ValueError, match="profile"):
            inputs.record(path)


def test_inputs_detect_changed_artifact(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"value":1}')
    inputs = collector.Inputs(tmp_path, "prepared96", "prepared24")
    assert inputs.json(path) == {"value": 1}
    path.write_text('{"value":2}')
    with pytest.raises(ValueError, match="changed"):
        inputs.finish()


def test_collect_preflights_last_evidence_path_before_any_metadata_read(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared96"
    prepared.mkdir()
    protected = prepared / "test.npz"
    protected.write_bytes(b"reserved TEST sentinel")
    alias = tmp_path / "additional.json"
    os.link(protected, alias)
    spec = declared_inventory()
    spec["development_evidence"]["additional"] = [str(alias)]
    monkeypatch.setattr(collector, "preparation", lambda *args: pytest.fail("metadata read before complete path preflight"))
    monkeypatch.setattr(collector, "sha256_file", lambda *args: pytest.fail("content read before complete path preflight"))
    with pytest.raises(ValueError, match="profile"):
        collector.collect(spec, base=tmp_path)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_json_duplicate_or_nonfinite_values_rejected(tmp_path, text):
    path = tmp_path / "report.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        collector.read_json(path)


def test_physical_requires_every_condition_full_cap_and_correct_units():
    methods = {"candidate": physical_result()}
    collector.physical_methods(methods, 96, 100)
    for mutation in (
        lambda x: x["candidate"]["per_condition"].pop("63"),
        lambda x: x["candidate"]["per_condition"]["0"].update(n_generated=99),
        lambda x: x["candidate"]["per_condition"]["0"].update(interval_hours=1.),
        lambda x: x["candidate"]["macro"].update(pointwise_wasserstein_kwh=2.),
    ):
        broken = deepcopy(methods)
        mutation(broken)
        with pytest.raises(ValueError):
            collector.physical_methods(broken, 96, 100)


@pytest.mark.parametrize("width", [24, 96])
def test_full_train_utility_contract(width):
    report, _, _ = utility_report(width=width)
    collector.utility_methods(report, set(report["methods"]), width, 100)
    report["training"]["training_cgan"]["n_profiles"] -= 1
    with pytest.raises(ValueError, match="TRAIN"):
        collector.utility_methods(report, set(report["methods"]), width, 100)


@pytest.mark.parametrize("field", ["profiles_sha256", "labels_sha256", "coefficients_sha256", "condition_counts"])
def test_precision_cap_cannot_change_any_training_fit(field):
    first, _, _ = utility_report(cap=100)
    second, _, _ = utility_report(cap=1000)
    collector.fixed_fits(first, second)
    second["training"]["training_cgan"][field] = {"0": 606060} if field == "condition_counts" else sha("changed")
    with pytest.raises(ValueError, match="cap changed"):
        collector.fixed_fits(first, second)


@pytest.mark.parametrize("unconditional", [False, True])
def test_native_gaussian_utility_retains_exact_empirical_exception(monkeypatch, unconditional):
    report, run, map_record = utility_report(unconditional=unconditional)
    monkeypatch.setattr(collector, "bound", lambda *args: None)
    frozen = collector.selection(report, 100, utility=True)
    disclosure = collector.native_utility(report, run, 100, frozen, map_record)
    assert disclosure["gaussian_quantile_knots"] == (None if unconditional else 1025)
    assert disclosure["shared_gaussian_across_arms"] is False
    report["copula"]["quantile_knots"] = 1025 if unconditional else None
    with pytest.raises(ValueError, match="Gaussian"):
        collector.native_utility(report, run, 100, frozen, map_record)


def test_unconditional_utility_refuses_a_conditional_map(monkeypatch):
    report, run, map_record = utility_report(unconditional=True)
    monkeypatch.setattr(collector, "bound", lambda *args: None)
    frozen = collector.selection(report, 100, utility=True)
    report["calibration"]["enabled"] = True
    report["config"]["calibrator"] = "conditional_map.npz"
    with pytest.raises(ValueError, match="raw-only"):
        collector.native_utility(report, run, 100, frozen, map_record)


@pytest.mark.parametrize("utility", [False, True])
@pytest.mark.parametrize("invalid", [0, 1, 500])
def test_bootstrap_preserves_withheld_status_and_null_bounds(utility, invalid):
    report, metrics, points = bootstrap(utility=utility, invalid=invalid)
    assert collector.interval_contract(report, metrics, points, utility=utility) is (invalid == 0)
    if invalid:
        report["intervals"][metrics[0]]["lower"] = -2.
    else:
        report["intervals"][metrics[0]]["estimate"] = 1.
    with pytest.raises(ValueError):
        collector.interval_contract(report, metrics, points, utility=utility)


@pytest.mark.parametrize("change", ["missing_metric", "replicates", "status", "point_estimate"])
def test_bootstrap_rejects_incomplete_or_misaligned_evidence(change):
    report, metrics, points = bootstrap()
    if change == "missing_metric":
        report["intervals"].pop(metrics[-1])
    elif change == "replicates":
        report["n_valid_bootstrap"] = 499
    elif change == "status":
        report["interval_status"] = "withheld_missing_conditions"
    else:
        report["point_estimates"]["generated"][metrics[0]] += .5
    with pytest.raises(ValueError):
        collector.interval_contract(report, metrics, points)


def test_seed_summary_keeps_five_independent_seeds_without_pooling_ci_or_controls():
    reports = [{"methods": {"neural": physical_result(float(i)), "control": physical_result(7.)},
                "paired_comparison": {"intervals": {"metric": {"lower": -1000 * i, "upper": 1000 * i}}}}
               for i in range(1, 6)]
    result = collector.summarize(reports, {"neural"})
    row = result["neural"]["macro"]["pointwise_wasserstein_kwh"]
    assert row["n_seeds"] == 5 and row["mean"] == 3.
    assert row["sample_sd"] == pytest.approx(np.sqrt(2.5))
    assert row["values_by_seed"] == {str(s): float(i) for s, i in zip(collector.SEEDS, range(1, 6))}
    assert result["control"]["independent_repetitions"] == 1
    assert result["control"]["seed_summaries_computed"] is False
    assert "lower" not in json.dumps(result) and "upper" not in json.dumps(result)
    with pytest.raises(ValueError, match="five"):
        collector.summarize(reports[:-1], {"neural"})


def test_summary_never_drops_nonfinite_or_incompatible_seed():
    reports = [{"methods": {"neural": physical_result(float(i))}} for i in range(1, 6)]
    reports[-1]["methods"]["neural"]["macro"]["pointwise_wasserstein_kwh"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        collector.summarize(reports, {"neural"})


def test_evidence_index_binds_collector_report_and_every_artifact(tmp_path):
    path = tmp_path / "collected.json"
    path.write_text('{"synthetic":true}\n')
    report = {"candidate_kind": "capacity", "development_evidence": declared_inventory()["development_evidence"],
              "artifacts": {"original": {"sha256": sha("original"), "bytes": 8}}}
    index = collector.evidence_index(report, path)
    assert index["candidate_kind"] == "capacity"
    assert index["development_evidence"]["additional"] == [str(path)]
    assert index["collector_report"]["sha256"] == sha(path.read_bytes())
    assert index["artifacts"]["original"] == report["artifacts"]["original"]
    assert report["development_evidence"]["additional"] == []


def source_report(family, *, version="historical"):
    hashes = {f"src/fluvius_cgan/{name}.py": sha(name) for name in (
        "evaluation", "models", "residual", "calibration", "copula", "joint", "battery", "utility",
        "utility_uncertainty", "uncertainty")}
    hashes["src/fluvius_cgan/copula.py"] = sha("copula:" + version)
    source = {"source_files_sha256": hashes}
    helper = {f"scripts/{name}.py": sha("runner:" + name) for name in (
        "evaluate_unconditional_reference", "compare_resolutions", "evaluate_resolution_utility",
        "compare_checkpoints", "evaluate_calibration", "evaluate_utility")}
    if family.startswith("native_physical"):
        source.update(evaluation_script_sha256=helper["scripts/evaluate_calibration.py"],
                      comparison_helper_sha256=helper["scripts/compare_checkpoints.py"])
    elif family.startswith("native_utility"):
        # The actual historical native96 reports have no evaluate_calibration.py helper entry.
        source["source_file_sha256"] = {key: helper[key] for key in
                                         ("scripts/evaluate_utility.py", "scripts/compare_checkpoints.py")}
        source["source_file_sha256"].update(hashes)
    elif family == "supplemental":
        source["supplemental_evaluation_source_files_sha256"] = {**hashes, **helper}
    elif family == "common_physical":
        source.update(runner_sha256=helper["scripts/compare_resolutions.py"],
                      comparison_helper_sha256=helper["scripts/compare_checkpoints.py"],
                      diagnostic_helper_sha256=helper["scripts/evaluate_calibration.py"])
    else:
        source.update(runner_sha256=sha("runner:" + family), helper_files_sha256=helper)
    return {"provenance": source}


@pytest.mark.parametrize("family", ["native_physical96", "native_physical24", "native_utility96",
                                   "native_utility24", "supplemental", "common_physical", "common_utility", "paired"])
def test_source_groups_bind_numerical_versions_and_runner_helpers(family):
    groups = {}
    report = source_report(family)
    collector.bind_evaluator(groups, report, family)
    collector.bind_evaluator(groups, deepcopy(report), family)
    changed = deepcopy(report)
    changed["provenance"]["source_files_sha256"]["src/fluvius_cgan/evaluation.py"] = sha("changed evaluator")
    with pytest.raises(ValueError, match="fingerprints conflict|mixed evaluator"):
        collector.bind_evaluator(groups, changed, family)


def test_deliberate_historical_copula_is_separate_but_current_common_sources_match():
    groups = {}
    collector.bind_evaluator(groups, source_report("native_physical96"), "native_physical96")
    collector.bind_evaluator(groups, source_report("common_physical", version="current"), "common_physical", current=True)
    collector.bind_evaluator(groups, source_report("paired", version="current"), "paired", current=True)
    changed = source_report("common_utility", version="different-current")
    with pytest.raises(ValueError, match="numerical versions differ"):
        collector.bind_evaluator(groups, changed, "common_utility", current=True)


@pytest.mark.parametrize("cap", [100, 1000])
def test_unconditional_supplement_binds_sixth_legacy_raw_report(cap):
    run = {"artifacts": {name: {"sha256": sha(name)} for name in collector.RUN_FILES}}
    raw = {"sha256": sha("validation_report.json")}
    report = {
        "run_artifacts_sha256": {**{name: row["sha256"] for name, row in run["artifacts"].items()},
                                 "validation_report.json": raw["sha256"]},
        "completed_training_run": {"primary_raw_report": {
            "sha256": raw["sha256"], "filename": "validation_report.json", "evaluation_seed": 2026,
            "samples_per_condition_cap": 100, "declared_native_evaluator_source_commit": collector.SOURCE_COMMIT,
            "superseded": False, "attribution_scope": "Legacy report has protocol-declared source attribution only."}},
        "evaluation_role": "primary_cap100_supplement" if cap == 100 else "precision_cap1000_supplement",
    }
    collector.supplemental_raw_binding(report, run, raw, cap)
    for mutation in (
        lambda x: x["run_artifacts_sha256"].pop("validation_report.json"),
        lambda x: x["completed_training_run"]["primary_raw_report"].update(sha256=sha("different raw")),
        lambda x: x["completed_training_run"]["primary_raw_report"].update(superseded=True),
        lambda x: x["completed_training_run"]["primary_raw_report"].update(samples_per_condition_cap=1000),
    ):
        broken = deepcopy(report)
        mutation(broken)
        with pytest.raises(ValueError):
            collector.supplemental_raw_binding(broken, run, raw, cap)


def hourly_pipeline_report(*, utility=False):
    pipelines = {}
    for arm in ("candidate96", "bce96", "bce24"):
        width = 24 if arm == "bce24" else 96
        for mode in ("raw", "calibrated"):
            mapped = mode == "calibrated"
            row = {"native_resolution": width, "evaluated_resolution": 24, "saved_map": mapped,
                   "aggregation": deepcopy(collector.AGGREGATION) if width == 96 else None,
                   "training_dtype" if utility else "evaluated_dtype": "float64" if width == 24 and mapped else "float32"}
            if mapped:
                row["order"] = ("native generator -> saved native map -> hourly sum" if width == 96 else
                                "hourly generator -> saved hourly map (float64 retained)" if utility else
                                "hourly generator -> saved hourly map (float64 map output retained)")
            if utility:
                if mapped:
                    row.update(native_raw_profiles_sha256=sha(arm + ":raw"), native_mapped_profiles_sha256=sha(arm + ":mapped"))
                else:
                    row["native_profiles_sha256"] = sha(arm + ":raw")
            pipelines[arm + "_" + mode] = row
    for name, shrinkage in (("gaussian_copula24", .01), ("independent_marginals24", 1.)):
        if not utility or name == "gaussian_copula24":
            pipelines[name] = {"fitted_on": "full TRAIN24", "seed": 2026, "quantile_knots": 1025, "shrinkage": shrinkage}
    return {"aggregation": deepcopy(collector.AGGREGATION), "pipelines": pipelines}


@pytest.mark.parametrize("utility", [False, True])
def test_hourly_pipeline_keeps_saved_native_map_before_sum(utility):
    report = hourly_pipeline_report(utility=utility)
    collector.hourly_pipelines(report, utility=utility)
    for mutation in (
        lambda x: x["pipelines"]["bce96_calibrated"].update(order="native generator -> hourly sum -> fitted map"),
        lambda x: x["pipelines"]["bce24_calibrated"].update(aggregation=deepcopy(collector.AGGREGATION)),
        lambda x: x["pipelines"]["gaussian_copula24"].update(quantile_knots=None),
        lambda x: x["aggregation"].update(accumulation_dtype="float32"),
    ):
        broken = deepcopy(report)
        mutation(broken)
        with pytest.raises(ValueError):
            collector.hourly_pipelines(broken, utility=utility)


def test_hourly_utility_mapped_profiles_use_same_native_raw_draw():
    report = hourly_pipeline_report(utility=True)
    report["pipelines"]["candidate96_calibrated"]["native_raw_profiles_sha256"] = sha("another draw")
    with pytest.raises(ValueError, match="same raw generation"):
        collector.hourly_pipelines(report, utility=True)


@pytest.mark.parametrize("field,value", [
    ("labels_sha256", sha("different label order")), ("condition_counts", {"0": 606060}),
    ("n_features", 80), ("ridge_objective", "mean_squared_error_plus_alpha"),
])
def test_utility_all_fits_share_full_train_label_order_and_fixed_predictor(field, value):
    report, _, _ = utility_report()
    report["training"]["training_cgan"][field] = value
    with pytest.raises(ValueError):
        collector.utility_methods(report, set(report["methods"]), 96, 100)


@pytest.mark.parametrize("utility", [False, True])
def test_bootstrap_uses_whole_meters_and_retains_missing_condition_draws(utility):
    selected = {"selected_meters": 353, "condition_counts": {str(i): 100 for i in range(64)}}
    value = {"condition_weighting": "equal_over_original_observed_support",
             "resampling_scheme": "within_technology_real_meter" if utility else "within_stratum_real_meter",
             "stratum_meter_counts": {str(i): 44 + int(i == 0) for i in range(8)}, "n_meters": 353,
             "n_invalid_bootstrap": 2, "missing_condition_draws" if utility else "missing_condition_draw_counts":
             {str(i): 2 if i == 0 else 0 for i in range(64)},
             "condition_counts": selected["condition_counts"], "n_profiles": 6400, "windows_per_profile": 21,
             "resampling_unit": "real_meter", "generated_sample_policy": "fixed_condition_reference",
             "comparator_sample_policy": "fixed_condition_reference"}
    collector.bootstrap_design(value, selected, utility=utility)
    value["resampling_scheme"] = "independent_profile_rows"
    with pytest.raises(ValueError, match="whole meters"):
        collector.bootstrap_design(value, selected, utility=utility)


def test_inventory_changed_during_collection_cannot_be_published(tmp_path, monkeypatch):
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(declared_inventory()))
    output, index = tmp_path / "report.json", tmp_path / "index.json"

    def changed_inventory(*args, **kwargs):
        inventory_path.write_text(json.dumps({**declared_inventory(), "candidate_kind": "capacity"}))
        return {}

    monkeypatch.setattr(collector, "collect", changed_inventory)
    with pytest.raises(ValueError, match="inventory changed"):
        collector.main(["--inventory", str(inventory_path), "--output", str(output), "--evidence-index", str(index)])
    assert not output.exists() and not index.exists()


def test_cli_rejects_hardlinked_inventory_before_any_content_read(tmp_path, monkeypatch):
    protected = tmp_path / "test.npz"
    protected.write_bytes(b"opaque reserved test sentinel")
    alias = tmp_path / "inventory.json"
    os.link(protected, alias)
    monkeypatch.setattr(collector, "sha256_file", lambda path: pytest.fail("hardlinked content was hashed"))
    monkeypatch.setattr(collector, "read_json", lambda path: pytest.fail("hardlinked content was parsed"))
    with pytest.raises(ValueError, match="hardlink|link"):
        collector.main(["--inventory", str(alias), "--output", str(tmp_path / "report.json"),
                        "--evidence-index", str(tmp_path / "index.json")])


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return sha(path.read_bytes())


def synthetic_preparation(tmp_path):
    parent, child = tmp_path / "prepared96", tmp_path / "prepared24"
    metadata_by_width = {}
    hashes_by_width = {}
    for width, directory in ((96, parent), (24, child)):
        directory.mkdir()
        # Invalid archives are deliberate: every collector read must stop at recorded JSON metadata.
        for name in ("train.npz", "val.npz", "test.npz"):
            (directory / name).write_bytes(b"opaque synthetic sentinel; never open this file")
        metadata = {
            "resolution": width, "target": "net", "units": "kWh/interval",
            "source_sha256": collector.RELEASE_SHA256, "split_seed": 42,
            "splits": {split: {"profiles": profiles, "meters": meters,
                               "meter_ids": [f"{split}-meter-{i}" for i in range(meters)],
                               "sha256": sha(f"recorded:{width}:{split}")}
                       for split, profiles, meters in (("train", 606060, 1665), ("val", 128492, 353))},
        }
        scaler_hash = write_json(directory / "scaler.json", {"minimum": -1., "maximum": 2.})
        if width == 96:
            metadata["curation"] = {"fingerprint_version": 2}
        else:
            metadata["derived"] = {
                "schema_version": 1, "operation": collector.OPERATION, "qualification": collector.QUALIFICATION,
                "test24_materialized": False,
                "parent": {"development_files_sha256": hashes_by_width[96],
                           "curation_manifest_sha256": sha((parent / "curation_manifest.json").read_bytes()),
                           "path_policy": "relative_to_child_prepared_directory", "prepared_dir": "../prepared96"},
            }
        metadata_hash = write_json(directory / "metadata.json", metadata)
        hashes = {"metadata.json": metadata_hash, "scaler.json": scaler_hash,
                  **{split + ".npz": metadata["splits"][split]["sha256"] for split in ("train", "val")}}
        manifest = {"output_files_sha256": hashes}
        if width == 24:
            manifest["derived"] = deepcopy(metadata["derived"])
        write_json(directory / ("curation_manifest.json" if width == 96 else "derivation_manifest.json"), manifest)
        metadata_by_width[width], hashes_by_width[width] = metadata, hashes
    return parent, child, metadata_by_width, hashes_by_width


def protect_sentinel_content(monkeypatch):
    original_hash, original_read_text, original_read_bytes = collector.sha256_file, Path.read_text, Path.read_bytes

    def permitted(path):
        assert Path(path).name not in ("train.npz", "val.npz", "test.npz"), "prepared sentinel content accessed"

    def checked_hash(path):
        permitted(path)
        return original_hash(path)

    def checked_text(path, *args, **kwargs):
        permitted(path)
        return original_read_text(path, *args, **kwargs)

    def checked_bytes(path, *args, **kwargs):
        permitted(path)
        return original_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(collector, "sha256_file", checked_hash)
    monkeypatch.setattr(Path, "read_text", checked_text)
    monkeypatch.setattr(Path, "read_bytes", checked_bytes)


def test_preparation_binds_full_metadata_lineage_without_opening_arrays(tmp_path, monkeypatch):
    parent, child, _, hashes = synthetic_preparation(tmp_path)
    protect_sentinel_content(monkeypatch)
    inputs = collector.Inputs(tmp_path, parent, child)
    result = collector.preparation(inputs)
    assert result[96]["hashes"] == hashes[96]
    assert result[24]["hashes"] == hashes[24]
    assert result[24]["metadata"]["derived"]["parent"]["development_files_sha256"] == hashes[96]
    assert all(Path(path).suffix == ".json" for path in inputs.records)
    inputs.finish()


@pytest.mark.parametrize("change", ["train_budget", "meter_overlap", "lineage_hash", "wrong_parent",
                                    "test_materialized", "curation_on_child", "manifest_output"])
def test_preparation_refuses_inconsistent_or_unqualified_lineage(tmp_path, monkeypatch, change):
    parent, child, metadata, _ = synthetic_preparation(tmp_path)
    width = 96 if change in ("train_budget", "meter_overlap") else 24
    directory = parent if width == 96 else child
    row = deepcopy(metadata[width])
    if change == "train_budget":
        row["splits"]["train"]["profiles"] -= 1
    elif change == "meter_overlap":
        row["splits"]["val"]["meter_ids"][0] = row["splits"]["train"]["meter_ids"][0]
    elif change == "lineage_hash":
        row["derived"]["parent"]["development_files_sha256"]["train.npz"] = sha("different parent")
    elif change == "wrong_parent":
        row["derived"]["parent"]["prepared_dir"] = "../another_parent"
    elif change == "test_materialized":
        row["derived"]["test24_materialized"] = True
    elif change == "curation_on_child":
        row["curation"] = {"fingerprint_version": 2}
    manifest_path = directory / ("curation_manifest.json" if width == 96 else "derivation_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["output_files_sha256"]["metadata.json"] = write_json(directory / "metadata.json", row)
    if width == 24:
        manifest["derived"] = deepcopy(row["derived"])
    if change == "manifest_output":
        manifest["output_files_sha256"]["val.npz"] = sha("unrelated validation")
    write_json(manifest_path, manifest)
    protect_sentinel_content(monkeypatch)
    with pytest.raises(ValueError):
        collector.preparation(collector.Inputs(tmp_path, parent, child))


def synthetic_map(tmp_path, *, transform=None):
    run = {"directory": str(tmp_path), "width": 96, "checkpoint_sha256": sha("best checkpoint"),
           "prepared_data_hashes": {name: sha(name) for name in collector.PREPARED},
           "training_selection": {"train_indices_sha256": sha("full canonical indices")}}
    metadata = {
        "schema_version": 1, "n_features": 96, "quantile_knots": 1025,
        "provenance": {"checkpoint_sha256": run["checkpoint_sha256"],
                       "prepared_data_hashes": dict(run["prepared_data_hashes"]),
                       "train_indices_sha256": run["training_selection"]["train_indices_sha256"],
                       "calibration_seed": 7301, "calibration_samples_per_condition": 4096,
                       "calibration_profiles_total": 64 * 4096, "quantile_knots": 1025},
    }
    if transform is not None:
        transform(metadata)
    content = io.BytesIO()
    np.save(content, np.array(json.dumps(metadata)), allow_pickle=False)
    path = tmp_path / "calibration.npz"
    with ZipFile(path, "w") as archive:
        archive.writestr("metadata.npy", content.getvalue())
        archive.writestr("source_quantiles.npy", b"invalid quantile member; metadata-only collector must not decode")
        archive.writestr("target_quantiles.npy", b"another invalid quantile member")
    return run, metadata, path


def test_map_metadata_decodes_only_scalar_metadata_and_hashes_entire_archive(tmp_path, monkeypatch):
    run, metadata, path = synthetic_map(tmp_path)
    original_read = ZipFile.read
    accessed = []

    def read_metadata_only(archive, name, *args, **kwargs):
        member = name.filename if hasattr(name, "filename") else name
        accessed.append(member)
        assert member == "metadata.npy", "quantile/profile member decoded"
        return original_read(archive, name, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "read", read_metadata_only)
    inputs = collector.Inputs(tmp_path, "prepared96", "prepared24")
    result = collector.map_metadata(inputs, run)
    assert result["metadata"] == metadata
    assert result["sha256"] == sha(path.read_bytes())
    assert accessed == ["metadata.npy"]
    inputs.finish()


@pytest.mark.parametrize("field,value", [
    ("checkpoint_sha256", sha("other checkpoint")), ("train_indices_sha256", sha("other subset")),
    ("calibration_seed", 2026), ("calibration_samples_per_condition", 1000),
    ("calibration_profiles_total", 4096), ("quantile_knots", 513),
    ("prepared_data_hashes", {"train.npz": sha("other TRAIN")}),
])
def test_map_metadata_requires_fixed_checkpoint_train_and_sampling_protocol(tmp_path, field, value):
    run, _, _ = synthetic_map(tmp_path, transform=lambda metadata: metadata["provenance"].update({field: value}))
    with pytest.raises(ValueError, match="binding differs"):
        collector.map_metadata(collector.Inputs(tmp_path, "prepared96", "prepared24"), run)


@pytest.mark.parametrize("field,value", [("n_features", 24), ("schema_version", 2), ("quantile_knots", 513)])
def test_map_metadata_rejects_incompatible_native_representation(tmp_path, field, value):
    run, _, _ = synthetic_map(tmp_path, transform=lambda metadata: metadata.update({field: value}))
    with pytest.raises(ValueError, match="width/knots"):
        collector.map_metadata(collector.Inputs(tmp_path, "prepared96", "prepared24"), run)
