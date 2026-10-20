"""One sealed candidate/reference study; TEST96 is opened only by the scorer.

Schema 2 is deliberately separate from frozen_test's strict schema-1 API.
All generated arrays, maps and downstream predictors are frozen before access.
"""
from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
import shutil
import uuid

import numpy as np
import torch

from . import frozen_test as ft
from . import training
from .derived import _preflight, _metadata as _derived_metadata, resolve_parent_directory, OPERATION, QUALIFICATION
from .evaluation import mean_baseline

ARMS = ("candidate96", "bce96", "unconditional96", "bce24")
SEEDS = ft.SEEDS
SUPPORT = ft.SUPPORT
TRAIN_ROWS = 606060
BOOTSTRAP_REPLICATES = 500
CONTRASTS = ("conditioning", "candidate_reference", "resolution_raw", "resolution_calibrated")
VIEWS = {"candidate96": ("raw96", "mapped96", "sum_raw24", "sum_mapped24"),
         "bce96": ("raw96", "mapped96", "sum_raw24", "sum_mapped24"),
         "unconditional96": ("raw96",), "bce24": ("raw24", "mapped24")}
CONTROLS = {96: (*ft.BASELINES, "conditional_mean"),
            24: ("gaussian_copula", "independent_marginals", "conditional_resampling", "conditional_mean")}
RIDGE_CONTROLS = ("training_real", "training_resampling", "training_copula")
NATIVE_EVIDENCE = ("primary", "precision", "utility", "utility_precision", "raw_primary")
HOURLY_EVIDENCE = ("primary", "precision", "utility", "utility_precision")
AGGREGATION = {"operation": "astype(float64).reshape(N,24,4).sum(axis=2).astype(float32)",
               "accumulator_dtype": "float64", "output_dtype": "float32",
               "map_order": "native generator -> saved native map, if any -> sum4; no hourly remap"}


def _protocol():
    return {**ft.PROTOCOL, "utility": True, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_confidence": .95, "full_train_profiles": TRAIN_ROWS,
            "arms": list(ARMS), "training_seeds": list(SEEDS), "reference_contrasts": list(CONTRASTS),
            "aggregation": AGGREGATION, "development_primary_cap": 100,
            "development_precision_cap": 1000, "development_seed": 2026}


def _sources():
    sources = ft._sources()
    root = Path(__file__).resolve().parents[2]
    for name in ("freeze_reference_study.py", "evaluate_reference_study.py"):
        key = f"scripts/{name}"
        sources["files_sha256"][key] = ft.sha256_file(root / key)
    return sources


def _expected_config(arm, seed, candidate_kind):
    ft._check(arm in ARMS and seed in SEEDS and candidate_kind in ("swd10", "capacity"), "invalid arm/seed/candidate kind")
    values = dict(epochs=26, batch_size=512, seed=seed, latent_dim=24, hidden_dim=64,
        discriminator_hidden_dim=64, objective="bce", lr_g=.0001, lr_d=.0002,
        adam_beta1=.9, adam_beta2=.999, normalization="minmax", conditional_mean_residual=False,
        swd_weight=0., derivative_swd_weight=0., swd_projections=128,
        batch_sampling="balanced_conditions", conditions_per_batch=4,
        validation_selection="conditional_w1", validation_samples=4096, max_train_samples=None,
        patience=None, device="cpu", num_threads=1, unconditional=arm == "unconditional96",
        allow_uncurated_data=False)
    if arm == "candidate96":
        values.update(discriminator_hidden_dim=128, objective="hinge_sn", adam_beta1=.5,
                      normalization="standard", swd_weight=10.)
        if candidate_kind == "capacity":
            values.update(hidden_dim=128, latent_dim=96)
    return training._config(values)


def _roster(arms):
    ft._check(isinstance(arms, dict) and set(arms) == set(ARMS), "exactly four declared arms required")
    paths, implicit = [], {}
    for arm in ARMS:
        records = arms[arm]
        ft._check(isinstance(records, list) and len(records) == 5, f"{arm} requires five runs")
        for index, record in enumerate(records):
            ft._check(isinstance(record, dict) and set(record) == {"checkpoint", "calibrator"}, "run requires checkpoint and calibrator fields")
            ft._check(isinstance(record["checkpoint"], (str, Path)), "checkpoint path required")
            ft._check((record["calibrator"] is None) == (arm == "unconditional96"), "unconditional arm is raw-only; all other arms require maps")
            if record["calibrator"] is not None:
                ft._check(isinstance(record["calibrator"], (str, Path)), "calibrator path required")
                paths.append(Path(record["calibrator"]))
            checkpoint = Path(record["checkpoint"])
            paths.append(checkpoint)
            siblings = sorted(set(checkpoint.parent.glob("*.json")) |
                              {checkpoint.parent / name for name in ("config.json", "summary.json", "history.json")})
            implicit[(arm, index)] = siblings
            paths.extend([*siblings, checkpoint.parent / "last.pt"])
    return paths, implicit


def _evidence_slots(evidence):
    """Closed coverage inventory. Its contents are retained, not used to select TEST models."""
    ft._check(isinstance(evidence, dict) and set(evidence) == {"native", "common_hourly", "paired", "additional"}, "invalid development evidence index")
    slots = []
    ft._check(set(evidence["native"]) == set(ARMS), "native evidence must cover all four arms")
    for arm in ARMS:
        ft._check(set(evidence["native"][arm]) == {str(s) for s in SEEDS}, "native evidence must cover all five seeds")
        for seed in SEEDS:
            rows = evidence["native"][arm][str(seed)]
            ft._check(set(rows) == set(NATIVE_EVIDENCE), "native evidence coverage differs")
            for kind in NATIVE_EVIDENCE:
                slots.append((f"native.{arm}.{seed}.{kind}", rows[kind], (arm,), seed, kind))
    ft._check(set(evidence["common_hourly"]) == {str(s) for s in SEEDS}, "hourly evidence must cover all five seeds")
    for seed in SEEDS:
        rows = evidence["common_hourly"][str(seed)]
        ft._check(set(rows) == set(HOURLY_EVIDENCE), "hourly evidence coverage differs")
        for kind in HOURLY_EVIDENCE:
            slots.append((f"common_hourly.{seed}.{kind}", rows[kind], ("candidate96", "bce96", "bce24"), seed, kind))
    ft._check(set(evidence["paired"]) == set(CONTRASTS), "all four paired reference contrasts required")
    for contrast in CONTRASTS:
        rows = evidence["paired"][contrast]
        ft._check(set(rows) == {str(s) for s in SEEDS}, "paired evidence must cover all five seeds")
        roles = ("bce96", "unconditional96") if contrast == "conditioning" else (
            ("candidate96", "bce96") if contrast == "candidate_reference" else ("bce24", "bce96"))
        for seed in SEEDS:
            slots.append((f"paired.{contrast}.{seed}", rows[str(seed)], roles, seed, "paired"))
    ft._check(isinstance(evidence["additional"], list), "additional evidence must be a list")
    ft._check(all(isinstance(row[1], (str, Path)) for row in slots), "every evidence slot requires a file path")
    return slots


def _sum4(values):
    values = np.asarray(values)
    ft._check(values.ndim == 2 and values.shape[1] == 96 and values.dtype.kind in "fiu" and np.isfinite(values).all(), "sum4 requires finite 96-slot profiles")
    with np.errstate(over="raise", invalid="raise"):
        result = values.astype(np.float64).reshape(-1, 24, 4).sum(axis=2).astype(np.float32)
    ft._check(np.isfinite(result).all(), "hourly sum overflow")
    return result


def _load_development(parent, child):
    _preflight(parent, child)
    hourly = training._load_prepared(child)  # verifies full streamed parent/child lineage
    ft._check(resolve_parent_directory(child, hourly[2].get("derived")) == parent, "hourly preparation has a different parent")
    native = training._load_prepared(parent)
    guard = training._training_curation_guard(parent, {"allow_uncurated_data": False})
    ft._check(guard["status"] == "verified_curation_v2" and guard["pinned_release"], "pinned verified curated96 required")
    derivation_hash = ft.sha256_file(child / "derivation_manifest.json")
    result = {}
    for width, directory, data in ((96, parent, native), (24, child, hourly)):
        train, val, metadata, scaler, hashes = data
        ft._check(metadata["resolution"] == width and metadata["target"] == "net" and metadata["units"] == "kWh/interval", "native physical preparation differs")
        ft._check(len(train["x"]) == TRAIN_ROWS, "full declared TRAIN count required")
        for name, part in (("train", train), ("val", val)):
            ft._check(np.unique(part["y"]).tolist() == SUPPORT, "all 64 TRAIN/VAL conditions required")
            ft._check(metadata["splits"][name]["sha256"] == hashes[f"{name}.npz"] and
                      np.unique(part["meter_id"]).tolist() == metadata["splits"][name]["meter_ids"], "actual development membership/hash differs from metadata")
        manifest_name = "curation_manifest.json" if width == 96 else "derivation_manifest.json"
        manifest_hash = guard["manifest_sha256"] if width == 96 else derivation_hash
        result[width] = {"train": train, "metadata": metadata, "scaler": scaler, "hashes": hashes,
                         "directory": directory, "manifest_name": manifest_name, "manifest_sha256": manifest_hash}
    ft._check(native[4] == guard["verified_file_hashes"], "parent data changed during guard/load")
    ft._check(hourly[2]["derived"]["parent"]["development_files_sha256"] == native[4] and
              hourly[2]["derived"]["parent"]["curation_manifest_sha256"] == guard["manifest_sha256"], "hourly parent fingerprints differ")
    for key in ("y", "meter_id", "date"):
        ft._check(np.array_equal(native[0][key], hourly[0][key]), "native/hourly TRAIN row identity differs")
    return result


def _verify_run(path, arm, data, candidate_kind):
    before = ft.sha256_file(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    ft._check(ft.sha256_file(path) == before, "checkpoint changed while loading")
    metadata, train = data["metadata"], data["train"]
    ft._check(state.get("format_version") == training.CHECKPOINT_FORMAT and state.get("metadata") == metadata and
              state.get("output_dim") == metadata["resolution"], "checkpoint native metadata differs")
    config = training._config(state["config"])
    seed = config["seed"]
    ft._check(config == _expected_config(arm, seed, candidate_kind), "checkpoint differs from declared arm configuration")
    ft._check(state.get("condition_dim") == (0 if arm == "unconditional96" else 64) and state.get("supported_labels") == SUPPORT, "checkpoint condition dimension/support differs")
    expected_hashes = {**data["hashes"], data["manifest_name"]: data["manifest_sha256"]}
    ft._check(state.get("data_hashes") == expected_hashes, "checkpoint preparation/manifest binding differs")
    ft._check(all(state["scaler"].get(k) == v for k, v in data["scaler"].items()), "checkpoint TRAIN scaler differs")
    if config["normalization"] == "standard":
        if "standard_scaler" not in data:
            values = train["x"].astype(np.float64)
            data["standard_scaler"] = {**data["scaler"], "kind": "standard", "mean": float(values.mean()), "std": float(values.std())}
            del values
        ft._check(state["scaler"] == data["standard_scaler"], "standardization must use exact full TRAIN statistics")
    else:
        ft._check(state["scaler"] == data["scaler"], "minmax checkpoint scaler differs")
    selection = state["selection"]
    ft._check(selection["train_profiles_available"] == selection["train_profiles_used"] == TRAIN_ROWS and
              selection["train_indices_sha256"] == ft._digest(np.arange(TRAIN_ROWS), "<i8"), "checkpoint must use exact complete TRAIN rows")
    directory = Path(path).parent
    summary = ft._json(directory / "summary.json")
    history = json.loads((directory / "history.json").read_bytes())
    ft._check(ft._json(directory / "config.json") == state["config"], "saved config differs from checkpoint")
    ft._check(summary["test_used"] is False and summary["epochs_completed"] == 26 and summary["stopped_early"] is False,
              "every selected run must complete the 26-epoch budget")
    updates = (TRAIN_ROWS + config["batch_size"] - 1) // config["batch_size"]
    ft._check(isinstance(history, list) and len(history) == 26 and [r["epoch"] for r in history] == list(range(1, 27)), "complete ordered history required")
    ft._check(all(type(r["validation_score"]) in (int, float) and math.isfinite(r["validation_score"]) and
              r["generator_updates"] == r["discriminator_updates"] == updates and
              r["samples_seen"] == updates * config["batch_size"] for r in history), "training history budget/score differs")
    ft._check(summary["generator_updates"] == summary["discriminator_updates"] == updates * 26 and
              summary["samples_seen"] == updates * config["batch_size"] * 26, "completed training exposure differs")
    best = min(history, key=lambda row: row["validation_score"])
    ft._check(state["epoch"] == summary["best_epoch"] == best["epoch"] and
              state["best_validation_score"] == summary["best_validation_score"] == best["validation_score"], "checkpoint is not earliest raw-selector minimum")
    for key in ("selection", "data_hashes", "provenance", "data_curation"):
        ft._check(summary.get(key) == state.get(key), f"summary/checkpoint {key} differs")
    ft._check(summary["source_sha256"] == metadata["source_sha256"], "training source release differs")
    ft._check(state.get("validation_metric") == summary.get("validation_metric") == training.CONDITIONAL_VALIDATION_METRIC, "raw condition-macro selector required")
    status = state.get("data_curation", {})
    ft._check(status.get("status") == ("verified_curation_v2" if arm != "bce24" else "verified_hourly_derivation_v1") and
              status.get("verified_file_hashes") == data["hashes"] and
              status.get("manifest_sha256" if arm != "bce24" else "derived_manifest_sha256") == data["manifest_sha256"], "checkpoint curation/derivation qualification differs")
    sources = state.get("provenance", {}).get("source_files_sha256", {})
    ft._check(all(ft._is_hash(sources.get(f"src/fluvius_cgan/{name}")) for name in ft.TRAINING_CORE[:-1]), "training core provenance missing")
    last_path = directory / "last.pt"
    last_hash = ft.sha256_file(last_path)
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    ft._check(ft.sha256_file(last_path) == last_hash, "last checkpoint changed while loading")
    for key in ("format_version", "config", "condition_dim", "output_dim", "metadata", "scaler", "data_hashes", "selection", "provenance", "data_curation", "supported_labels", "validation_metric"):
        ft._check(last.get(key) == state.get(key), f"last/best checkpoint {key} differs")
    ft._check(last.get("epoch") == 26 and last.get("generator_updates") == last.get("discriminator_updates") == updates * 26 and
              last.get("samples_seen") == updates * config["batch_size"] * 26 and last.get("best_epoch") == best["epoch"] and
              last.get("best_validation_score") == best["validation_score"] and last.get("validation_score") == history[-1]["validation_score"], "last checkpoint does not prove full frozen training budget")
    for checkpoint, completed in ((state, updates * state["epoch"]), (last, updates * 26)):
        for name in ("optimizer_g_state_dict", "optimizer_d_state_dict"):
            if name in checkpoint:
                steps = [float(value["step"]) for value in checkpoint[name]["state"].values() if "step" in value]
                ft._check(steps and all(value == completed for value in steps), "optimizer step counts differ from update budget")
    return state, before, last_hash


def _copy_asset(source, output, name, assets):
    before = ft.sha256_file(source)
    with Path(source).open("rb") as src, ft._safe_asset(output, name).open("xb") as dst:
        shutil.copyfileobj(src, dst)
    ft._record_asset(output, name, assets)
    ft._check(ft.sha256_file(source) == before == assets[name]["sha256"], "input changed while copying")
    return name


def _report_models(report):
    if "models" in report:
        return {row["sha256"] for row in report["models"].values()}
    if "candidate" in report and "comparator" in report:
        return {report[key]["sha256"] for key in ("candidate", "comparator")}
    if "checkpoint" in report:
        return {report["checkpoint"]["sha256"]}
    return {report.get("checkpoint_sha256")}


def _physical_evidence(value):
    ft._check(isinstance(value, dict) and set(value.get("per_condition", {})) == {str(c) for c in SUPPORT}, "physical evidence requires all64 condition metrics")
    for row in [value.get("macro", {}), *value["per_condition"].values()]:
        for key in ft.METRICS[:-1]:
            number = row.get(key)
            ft._check(type(number) in (int, float) and math.isfinite(number) and number >= 0, "physical evidence metric missing/nonfinite")
        number = row.get("autocorrelation", {}).get("1", {}).get("wasserstein")
        ft._check(type(number) in (int, float) and math.isfinite(number) and number >= 0, "physical evidence ACF metric missing/nonfinite")


def _utility_evidence(value):
    ft._check(isinstance(value, dict) and set(value.get("per_condition", {})) == {str(c) for c in SUPPORT}, "utility evidence requires all64 condition metrics")
    for row in [value.get("macro", {}), *value["per_condition"].values()]:
        for key in ("rmse_kwh", "mae_kwh"):
            number = row.get(key)
            ft._check(type(number) in (int, float) and math.isfinite(number) and number >= 0, "utility evidence metric missing/nonfinite")


def _paired_evidence(value, width):
    ft._check(isinstance(value, dict), "paired evidence results missing")
    ft._check(value.get("n_bootstrap") == 500 and value.get("confidence") == .95 and value.get("seed") == 2026 and
              value.get("interval_hours") == 24 / width and value.get("condition_support") == SUPPORT and
              value.get("n_conditions") == 64 and value.get("resampling_unit") == "real_meter" and
              value.get("resampling_scheme") == "within_stratum_real_meter" and
              value.get("condition_weighting") == "equal_over_original_observed_support" and
              value.get("comparison") == "generated_minus_comparator", "paired evidence results protocol differs")
    valid, invalid = value.get("n_valid_bootstrap"), value.get("n_invalid_bootstrap")
    ft._check(type(valid) is int and type(invalid) is int and min(valid, invalid) >= 0 and valid + invalid == 500,
              "paired evidence draw counts differ")
    status = "unavailable_missing_condition_draws" if invalid else "complete"
    ft._check(value.get("interval_status") == status and
              value.get("missing_condition_policy") == "withhold_all_intervals_if_any_draw_lacks_original_support" and
              set(value.get("per_condition", {})) == {str(c) for c in SUPPORT}, "paired evidence condition results missing")
    for row in [value, *value["per_condition"].values()]:
        ft._check(isinstance(row, dict) and row.get("interval_status") == status and
                  set(row.get("intervals", {})) == set(ft.METRICS) and
                  set(row.get("point_estimates", {})) == {"generated", "comparator"}, "paired evidence metric results missing")
        for method in ("generated", "comparator"):
            points = row["point_estimates"][method]
            ft._check(set(points) == set(ft.METRICS) and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in points.values()),
                      "paired evidence point estimates invalid")
        for metric in ft.METRICS:
            interval = row["intervals"][metric]
            estimate = interval.get("estimate")
            expected = row["point_estimates"]["generated"][metric] - row["point_estimates"]["comparator"][metric]
            ft._check(type(estimate) in (int, float) and math.isfinite(estimate) and
                      math.isclose(estimate, expected, rel_tol=1e-12, abs_tol=1e-12), "paired evidence contrast estimate differs")
            lower, upper = interval.get("lower"), interval.get("upper")
            ft._check((lower is None and upper is None) if invalid else
                      (type(lower) in (int, float) and type(upper) in (int, float) and
                       math.isfinite(lower) and math.isfinite(upper) and lower <= upper), "paired evidence bounds/status differ")


def _validate_evidence_report(report, roles, seed, kind, runs, contrast=None):
    ft._check(report.get("split") == "val" and report.get("test_used", False) is False, "development evidence must be validation-only")
    expected = {runs[(arm, seed)]["checkpoint_sha256"] for arm in roles}
    ft._check(_report_models(report) == expected, "development report checkpoint coverage differs")
    selection = report.get("validation_selection", {})
    cap = selection.get("samples_per_condition_cap", report.get("samples_per_condition_cap", report.get("samples_per_condition")))
    ft._check(cap == (1000 if kind in ("precision", "utility_precision") else 100), "development primary/precision cap differs")
    draw = report.get("seed", report.get("seeds", {}).get("validation_selection", selection.get("seed")))
    ft._check(draw == 2026, "development selection seed differs")
    expected_parts = {"parent96": runs[("bce96", seed)]["prepared_data_hashes"],
                      "derived24": runs[("bce24", seed)]["prepared_data_hashes"]}
    if len(roles) == 1:
        row = runs[(roles[0], seed)]
        if "prepared_data_hashes" in report:
            ft._check(report["prepared_data_hashes"] == row["prepared_data_hashes"], "native evidence preparation differs")
        else:  # original raw CLI schema
            ft._check(report.get("train_sha256") == row["prepared_data_hashes"]["train.npz"] and
                      report.get("evaluation_sha256") == row["prepared_data_hashes"]["val.npz"], "raw evidence TRAIN/VAL binding differs")
    else:
        expected = expected_parts if "bce24" in roles else {"parent96": expected_parts["parent96"]}
        ft._check(report.get("prepared_data_hashes") == expected, "comparison evidence preparation differs")
    def map_record(record, arm):
        row = runs[(arm, seed)]
        ft._check(isinstance(record, dict) and record.get("sha256") == row["calibrator_sha256"] and
                  record.get("provenance") == row["calibration_provenance"], "evidence saved-map binding differs")
    if kind == "paired":
        ft._check(report.get("training_seed") == seed and report.get("replicates") == 500 and
                  set(report.get("metrics", [])) == set(ft.METRICS), "paired development protocol differs")
        ft._check(report.get("contrast") == contrast and report.get("candidate", {}).get("sha256") == runs[(roles[0], seed)]["checkpoint_sha256"] and
                  report.get("comparator", {}).get("sha256") == runs[(roles[1], seed)]["checkpoint_sha256"], "paired contrast/order differs")
        maps = report.get("saved_maps", {})
        if contrast in ("candidate_reference", "resolution_calibrated"):
            ft._check(set(maps) == {"candidate", "comparator"}, "calibrated paired maps missing")
            for name, arm in zip(("candidate", "comparator"), roles):
                map_record(maps[name], arm)
        else:
            ft._check(maps == {}, "raw paired comparison contains maps")
        width = 24 if contrast in ("resolution_raw", "resolution_calibrated") else 96
        ft._check(report.get("resolution") == width and report.get("interval_hours") == 24 / width, "paired evidence resolution differs")
        _paired_evidence(report.get("paired_comparison"), width)
    elif len(roles) == 3:
        ft._check(set(report.get("models", {})) == set(roles) and set(report.get("saved_maps", {})) == set(roles), "common-hourly models/maps missing")
        for arm in roles:
            ft._check(report["models"][arm]["sha256"] == runs[(arm, seed)]["checkpoint_sha256"], "common-hourly role binding differs")
            map_record(report["saved_maps"][arm], arm)
        needed = {f"{arm}_{variant}" for arm in roles for variant in ("raw", "calibrated")}
        ft._check(needed <= set(report.get("methods", {})), "common-hourly raw/calibrated outcomes missing")
    elif kind != "raw_primary":
        arm = roles[0]
        methods = set(report.get("methods", {}))
        if arm != "unconditional96":
            calibration = report.get("calibration", {})
            if kind in ("utility", "utility_precision"):
                map_record(calibration, arm)
                ft._check({"training_cgan", "training_calibrated_cgan", "training_real", "training_resampling"} <= methods, "native utility outcomes missing")
            else:
                ft._check(calibration.get("provenance") == runs[(arm, seed)]["calibration_provenance"] and
                          {"raw_cgan", "calibrated_cgan"} <= methods, "native calibration evidence differs")
                artifact = calibration.get("saved_model")
                if kind == "primary" or artifact is not None:
                    ft._check(isinstance(artifact, dict) and artifact.get("sha256") == runs[(arm, seed)]["calibrator_sha256"], "primary evidence saved map differs")
        else:
            ft._check("calibrated_cgan" not in methods and "training_calibrated_cgan" not in methods and
                      not report.get("calibration", {}).get("enabled", False), "unconditional evidence contains conditional calibration")
            if kind in ("primary", "precision"):
                ft._check(report.get("report_kind") == "supplemental_native96_raw_unconditional_reference" and
                          report.get("conditional_calibrator_used") is False and
                          {"raw_unconditional_cgan", "conditional_mean", "conditional_resampling", "gaussian_copula", "independent_marginals"} <= methods,
                          "supplemental unconditional physical evidence missing")
    if kind in ("utility", "utility_precision"):
        width = 24 if len(roles) == 3 or roles == ("bce24",) else 96
        task, selection = report.get("task", {}), report.get("utility_training_selection", {})
        ft._check(task.get("resolution") == width and task.get("lags") == width // 8 and
                  selection.get("selected_profiles") == TRAIN_ROWS and selection.get("cap_applied") is False,
                  "utility evidence task/full TRAIN selection differs")
        required = ({f"{arm}_{variant}" for arm in roles for variant in ("raw", "calibrated")} if len(roles) == 3 else
                    {"training_cgan", "training_real", "training_resampling"} | ({"training_calibrated_cgan"} if roles != ("unconditional96",) else set()))
        for name in required:
            _utility_evidence(report.get("methods", {}).get(name))
    elif kind != "paired":
        if len(roles) == 3:
            required = {f"{arm}_{variant}" for arm in roles for variant in ("raw", "calibrated")}
        elif kind == "raw_primary":
            required = {"cgan", "conditional_mean", "conditional_resampling"}
        elif roles == ("unconditional96",):
            required = {"raw_unconditional_cgan", "conditional_mean", "conditional_resampling", "gaussian_copula", "independent_marginals"}
        else:
            required = {"raw_cgan", "calibrated_cgan"}
        for name in required:
            _physical_evidence(report.get("methods", {}).get(name))


def freeze_reference_study(prepared96, prepared24, arms, output_dir, *, candidate_kind,
                           decision_rationale, protocol_files, development_evidence, attempt_history):
    """Seal complete TRAIN-only assets and development evidence; never read TEST.

    The mandatory evidence index is documented in FROZEN_REFERENCE_STUDY.md.
    Utility and 500 replicates are fixed; there is no public reduced-study mode.
    """
    parent, child, output = Path(prepared96).resolve(), Path(prepared24).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    ft._write(output / "freeze_attempt.json", {"started_at": ft._now(), "test_accessed": False})
    try:
        with ft._one_thread():
            result = _freeze(parent, child, arms, output, candidate_kind, decision_rationale,
                             list(protocol_files), development_evidence, list(attempt_history))
        ft._write(output / "freeze_complete.json", {**result, "completed_at": ft._now(), "test_accessed": False})
        return result
    except BaseException as error:
        ft._write(output / "freeze_failure.json", {"failed_at": ft._now(), "test_accessed": False,
            "error_type": type(error).__name__, "error": str(error)})
        raise


def _freeze(parent, child, arms, output, candidate_kind, rationale, protocols, evidence, attempts):
    ft._check(candidate_kind in ("swd10", "capacity") and isinstance(rationale, str) and rationale.strip(), "declared candidate kind and decision rationale required")
    ft._check(protocols and attempts, "protocol files and complete attempted-arm history required")
    run_paths, implicit = _roster(arms)
    slots = _evidence_slots(evidence)
    paths = run_paths + protocols + attempts + [s[1] for s in slots] + evidence["additional"]
    paths += [parent / name for name in (*ft.PREPARED, "curation_manifest.json")]
    paths += [child / name for name in (*ft.PREPARED, "derivation_manifest.json")]
    ft._check(all(Path(path).name != "test.npz" and Path(path).resolve().name != "test.npz" for path in paths), "freeze input names a TEST archive")
    for reserved in (parent, child):
        ft._forbid_test_inputs(reserved, paths)
    _preflight(parent, child)
    sources, environment = _sources(), ft._environment()
    data = _load_development(parent, child)
    binding = ft._test_binding(parent, data[96]["metadata"], ft._json(parent / "curation_manifest.json"))
    ft._check(binding["resolution"] == 96, "reserved cohort must be TEST96")
    (output / "artifacts").mkdir()
    (output / "attempts").mkdir()
    assets, preparation = {}, {}
    for width, part in data.items():
        records = {name: _copy_asset(part["directory"] / name, output, f"prepared{width}_{name}", assets)
                   for name in ("metadata.json", "scaler.json", part["manifest_name"])}
        preparation[str(width)] = {"files": records, "hashes": part["hashes"], "manifest_name": part["manifest_name"],
                                  "manifest_sha256": part["manifest_sha256"]}
    labels = np.repeat(np.asarray(SUPPORT, dtype=np.int64),
        [min(ft.PROTOCOL["samples_per_condition_cap"], binding["condition_counts"][str(c)]) for c in SUPPORT])
    labels_asset = ft._save_array(output, "labels.npy", labels, assets)
    runs, roster = {}, {arm: [] for arm in ARMS}
    for arm in ARMS:
        part = data[24 if arm == "bce24" else 96]
        for index, spec in enumerate(arms[arm]):
            state, checkpoint_hash, last_hash = _verify_run(Path(spec["checkpoint"]), arm, part, candidate_kind)
            seed = state["config"]["seed"]
            ft._check((arm, seed) not in runs, "duplicate training seed in arm")
            prefix = f"{arm}_{seed}"
            checkpoint = _copy_asset(spec["checkpoint"], output, prefix + ".pt", assets)
            ft._check(assets[checkpoint]["sha256"] == checkpoint_hash, "checkpoint changed since validation")
            row = {"arm": arm, "seed": seed, "checkpoint": checkpoint, "checkpoint_sha256": checkpoint_hash,
                   "last_checkpoint": _copy_asset(Path(spec["checkpoint"]).parent / "last.pt", output, prefix + "_last.pt", assets),
                   "last_checkpoint_sha256": last_hash,
                   "config": state["config"], "epoch": state["epoch"], "training_selection": state["selection"],
                   "prepared_data_hashes": part["hashes"], "source_sha256": part["metadata"]["source_sha256"],
                   "training_provenance": state["provenance"], "calibrator": None, "calibrator_sha256": None,
                   "calibration_provenance": None, "calibration_diagnostics": None,
                   "evidence": ft._evidence(output, implicit[(arm, index)], assets, prefix + "_evidence")}
            ft._check(assets[row["last_checkpoint"]]["sha256"] == last_hash, "last checkpoint changed since verification")
            if arm != "unconditional96":
                model, details = ft.load_verified_calibrator(spec["calibrator"], checkpoint_sha256=checkpoint_hash,
                    train_indices_sha256=state["selection"]["train_indices_sha256"], prepared_data_hashes=part["hashes"],
                    generation_seed=ft.PROTOCOL["generation_seed"], n_features=part["metadata"]["resolution"], train_labels=part["train"]["y"])
                ft._check(model.quantile_knots == 1025 and details["calibration_seed"] == 7301 and
                          model.provenance["calibration_samples_per_condition"] == 4096, "saved map violates 4096/7301/1025 protocol")
                row["calibrator"] = _copy_asset(spec["calibrator"], output, prefix + "_map.npz", assets)
                row["calibrator_sha256"] = details["sha256"]
                ft._check(assets[row["calibrator"]]["sha256"] == details["sha256"], "map changed since validation")
                row["calibration_provenance"] = model.provenance
                del model
            runs[(arm, seed)] = row
        ft._check({s for a, s in runs if a == arm} == set(SEEDS), "all five fixed seeds required in every arm")
        roster[arm] = [runs[(arm, seed)] for seed in SEEDS]
        core = [{name: r["training_provenance"]["source_files_sha256"].get(f"src/fluvius_cgan/{name}") for name in ft.TRAINING_CORE}
                for r in roster[arm]]
        ft._check(all(value == core[0] for value in core), "training core differs within an arm")
    for seed in SEEDS:
        first, second = (runs[(arm, seed)]["training_provenance"]["source_files_sha256"] for arm in ("bce96", "unconditional96"))
        ft._check(all(first.get(f"src/fluvius_cgan/{name}") == second.get(f"src/fluvius_cgan/{name}") for name in ft.TRAINING_CORE), "conditioning comparison requires identical training core")
    coverage = {}
    for index, (key, path, roles, seed, kind) in enumerate(slots):
        report = ft._json(path)
        _validate_evidence_report(report, roles, seed, kind, runs, key.split(".")[1] if kind == "paired" else None)
        coverage[key] = ft._evidence(output, [path], assets, f"coverage_{index}")[0]
    decision = {"candidate_kind": candidate_kind, "rationale": rationale,
        "protocol_references": ft._evidence(output, protocols, assets, "protocol"),
        "attempt_history": ft._evidence(output, attempts, assets, "attempt_history"),
        "additional_evidence": ft._evidence(output, evidence["additional"], assets, "additional"),
        "coverage": coverage, "scope": "Development evidence records the fixed choice and all supplied attempted-arm history; it is not TEST evidence or a new selection rule."}
    controls, utility_controls = {}, {}
    for width, part in data.items():
        tx, ty = part["train"]["x"], part["train"]["y"]
        controls[str(width)], utility_controls[str(width)] = {}, {}
        utility_controls[str(width)]["training_real"] = ft._save_ridge(output, f"ridge_real{width}.npz", tx, ty, assets)
        for method in CONTROLS[width]:
            model = None
            if method == "conditional_mean":
                values = mean_baseline(tx, ty, labels)
            elif method == "conditional_resampling":
                values = ft.resample_baseline(tx, ty, labels, seed=ft.PROTOCOL["generation_seed"])
            else:
                model = ft.fit_conditional_copula(tx, ty, quantile_knots=None if method == "exact_marginal_copula" else 1025,
                                                 shrinkage=1. if method == "independent_marginals" else .01)
                values = model.generate(labels, seed=ft.PROTOCOL["generation_seed"])
            controls[str(width)][method] = ft._save_array(output, f"control{width}_{method}.npy", np.asarray(values, dtype=np.float64), assets)
            del values
            if method in ("gaussian_copula", "conditional_resampling"):
                values = (ft.resample_baseline(tx, ty, ty, seed=ft.PROTOCOL["utility_seed"]) if model is None
                          else model.generate(ty, seed=ft.PROTOCOL["utility_seed"]))
                name = "training_copula" if model is not None else "training_resampling"
                utility_controls[str(width)][name] = ft._save_ridge(output, f"ridge_control{width}_{name}.npz", values, ty, assets)
                del values
            del model
    for arm in ARMS:
        part = data[24 if arm == "bce24" else 96]
        for row in roster[arm]:
            prefix, model = f"{arm}_{row['seed']}", None
            if row["calibrator"]:
                model, _ = ft.load_verified_calibrator(ft._safe_asset(output, row["calibrator"]),
                    checkpoint_sha256=row["checkpoint_sha256"], train_indices_sha256=row["training_selection"]["train_indices_sha256"],
                    prepared_data_hashes=part["hashes"], generation_seed=ft.PROTOCOL["generation_seed"],
                    n_features=part["metadata"]["resolution"], train_labels=part["train"]["y"])
            raw = training.generate(ft._safe_asset(output, row["checkpoint"]), labels, seed=ft.PROTOCOL["generation_seed"], batch_size=1024)
            views = _representations(arm, raw, model, labels)
            row["arrays"] = {name: ft._save_array(output, f"{prefix}_{name}.npy", np.asarray(values, dtype=np.float64), assets)
                             for name, values in views.items()}
            row["representations"] = {name: _representation_record(arm, name, views) for name in views}
            if model is not None:
                name = prefix + "_map_diagnostics.json"
                ft._write(ft._safe_asset(output, name), {"fitting": model.fitting_diagnostics(), "application": model.diagnostics(raw, labels),
                    "scope": "Fixed native-map diagnostics on TRAIN references and fresh generated arrays; no TEST values."})
                row["calibration_diagnostics"] = ft._record_asset(output, name, assets)
            del raw, views
            ty = part["train"]["y"]
            raw = training.generate(ft._safe_asset(output, row["checkpoint"]), ty, seed=ft.PROTOCOL["utility_seed"], batch_size=1024)
            views = _representations(arm, raw, model, ty)
            row["utility"] = {name: ft._save_ridge(output, f"{prefix}_ridge_{name}.npz", values, ty, assets) for name, values in views.items()}
            del raw, views, model
    ft._check(_sources() == sources and ft._environment() == environment, "source/environment changed during freeze")
    for part in data.values():
        ft._check(all(ft.sha256_file(part["directory"] / name) == digest for name, digest in part["hashes"].items()) and
                  ft.sha256_file(part["directory"] / part["manifest_name"]) == part["manifest_sha256"], "development data changed during freeze")
    manifest = {"schema_version": 2, "kind": "frozen_candidate_reference_study", "created_at": ft._now(),
        "test_accessed_during_freeze": False, "protocol": _protocol(), "candidate_kind": candidate_kind,
        "sources": sources, "environment": environment, "preparation": preparation, "test_binding": binding,
        "decision": decision, "labels_artifact": labels_asset, "controls": controls, "utility_controls": utility_controls,
        "arms": roster, "assets": assets,
        "limitations": [s for s in ft.LIMITATIONS if not s.startswith("Reference-architecture")] + [
            "Reference conditioning is a BCE-network ablation with different input parameter counts; candidate versus BCE is a package comparison.",
            "Hourly profiles inherit quarter-hour canonical membership; aggregation may create coincidences and is not a new identity audit.",
            "Native96 and native24 utility horizons, ramps, ACF and projection spaces differ; common-hourly comparisons use the same transformed rows."]}
    _validate_manifest(output, manifest)
    ft._write(output / "manifest.json", manifest)
    seal = ft.sha256_file(output / "manifest.json")
    with (output / "seal.sha256").open("x", encoding="ascii") as handle:
        handle.write(seal + "\n")
    return {"manifest": str(output / "manifest.json"), "seal_sha256": seal}


def _representations(arm, raw, model, labels):
    width = 24 if arm == "bce24" else 96
    raw = np.asarray(raw)
    ft._check(raw.shape == (len(labels), width) and np.isfinite(raw).all(), "generated shape/finiteness differs")
    values = {f"raw{width}": raw}
    if model is not None:
        values[f"mapped{width}"] = model.transform(raw, labels)
    if arm in ("candidate96", "bce96"):
        values.update(sum_raw24=_sum4(values["raw96"]), sum_mapped24=_sum4(values["mapped96"]))
    ft._check(set(values) == set(VIEWS[arm]), "native/map representation inventory differs")
    return values


def _representation_record(arm, name, views):
    predecessor = "mapped96" if name == "sum_mapped24" else "raw96" if name == "sum_raw24" else None
    return {"native_resolution": 24 if arm == "bce24" else 96, "resolution": views[name].shape[1],
            "stored_dtype": "float64", "computed_dtype": str(views[name].dtype),
            "profiles_sha256": ft._digest(views[name], "<f8"), "aggregation": AGGREGATION if predecessor else None,
            "predecessor": predecessor, "predecessor_sha256": ft._digest(views[predecessor], "<f8") if predecessor else None}


def _required_coverage():
    return ({f"native.{arm}.{seed}.{kind}" for arm in ARMS for seed in SEEDS for kind in NATIVE_EVIDENCE} |
            {f"common_hourly.{seed}.{kind}" for seed in SEEDS for kind in HOURLY_EVIDENCE} |
            {f"paired.{contrast}.{seed}" for contrast in CONTRASTS for seed in SEEDS})


def _validate_manifest(bundle, manifest):
    ft._check(manifest.get("schema_version") == 2 and manifest.get("kind") == "frozen_candidate_reference_study" and
              manifest.get("test_accessed_during_freeze") is False, "unsupported unified study schema")
    ft._check(ft._canonical(manifest["protocol"]) == ft._canonical(_protocol()), "sealed unified protocol differs")
    ft._check(manifest["sources"] == _sources() and manifest["environment"] == ft._environment(), "sealed source/environment differs")
    ft._verify_assets(bundle, manifest["assets"])
    ft._check(set(manifest["preparation"]) == {"96", "24"}, "both native preparations required")
    expected_assets, metadata = {manifest["labels_artifact"]}, {}
    for width in (96, 24):
        part = manifest["preparation"][str(width)]
        name = "curation_manifest.json" if width == 96 else "derivation_manifest.json"
        ft._check(part["manifest_name"] == name and set(part["hashes"]) == set(ft.PREPARED) and
                  set(part["files"]) == {"metadata.json", "scaler.json", name}, "sealed preparation inventory differs")
        for logical, filename in part["files"].items():
            expected_assets.add(filename)
            digest = part["manifest_sha256"] if logical == name else part["hashes"][logical]
            ft._check(manifest["assets"][filename]["sha256"] == digest, "preparation asset binding differs")
        metadata[width] = ft._json(ft._safe_asset(bundle, part["files"]["metadata.json"]))
        ft._check(metadata[width]["resolution"] == width and metadata[width]["target"] == "net" and
                  metadata[width]["units"] == "kWh/interval" and metadata[width]["splits"]["train"]["profiles"] == TRAIN_ROWS,
                  "sealed native preparation dimensions/counts differ")
    curation = ft._json(ft._safe_asset(bundle, manifest["preparation"]["96"]["files"]["curation_manifest.json"]))
    binding = ft._test_binding(Path(manifest["test_binding"]["path"]).parent, metadata[96], curation)
    ft._check(binding == manifest["test_binding"] and binding["resolution"] == 96, "reserved TEST96 binding differs")
    derived = metadata[24].get("derived", {})
    parent = derived.get("parent", {})
    ft._check(derived.get("schema_version") == 1 and derived.get("operation") == OPERATION and
              derived.get("qualification") == QUALIFICATION and derived.get("test24_materialized") is False and
              parent.get("development_files_sha256") == manifest["preparation"]["96"]["hashes"] and
              parent.get("curation_manifest_sha256") == manifest["preparation"]["96"]["manifest_sha256"] and
              parent.get("path_policy") == "relative_to_child_prepared_directory" and "curation" not in metadata[24],
              "sealed hourly lineage differs")
    derivation = ft._json(ft._safe_asset(bundle, manifest["preparation"]["24"]["files"]["derivation_manifest.json"]))
    ft._check(derivation["derived"] == derived and derivation["output_files_sha256"] == manifest["preparation"]["24"]["hashes"], "sealed hourly manifest differs")
    reference = parent.get("prepared_dir")
    ft._check(isinstance(reference, str) and reference not in ("", ".") and "\\" not in reference and "\0" not in reference and
              not PurePosixPath(reference).is_absolute() and PurePosixPath(reference).as_posix() == reference and
              ":" not in reference.split("/")[0], "sealed portable parent path is invalid")
    # Reuse the derivation's deterministic metadata constructor without opening
    # either preparation. These symbolic paths only reproduce the stored relpath.
    symbolic_child = Path("/__frozen_lineage__/child")
    expected_splits = {name: {**metadata[96]["splits"][name], "sha256": manifest["preparation"]["24"]["hashes"][f"{name}.npz"]}
                       for name in ("train", "val")}
    expected_hourly = _derived_metadata(metadata[96], symbolic_child / reference, symbolic_child,
        {"verified_file_hashes": manifest["preparation"]["96"]["hashes"],
         "manifest_sha256": manifest["preparation"]["96"]["manifest_sha256"]}, expected_splits)
    ft._check(ft._canonical(metadata[24]) == ft._canonical(expected_hourly), "sealed hourly metadata/calendar/reserved binding differs")
    counts = [min(ft.PROTOCOL["samples_per_condition_cap"], binding["condition_counts"][str(c)]) for c in SUPPORT]
    labels = np.load(ft._safe_asset(bundle, manifest["labels_artifact"]), allow_pickle=False)
    ft._check(labels.dtype == np.int64 and np.array_equal(labels, np.repeat(np.asarray(SUPPORT, dtype=np.int64), counts)), "sealed labels differ from reserved metadata")
    ft._check(set(manifest["arms"]) == set(ARMS) and manifest["candidate_kind"] in ("swd10", "capacity"), "sealed arm roster differs")
    run_lookup = {}
    for arm in ARMS:
        rows = manifest["arms"][arm]
        ft._check([r["seed"] for r in rows] == list(SEEDS), "all five ordered seeds required in every arm")
        for row in rows:
            seed = row["seed"]
            ft._check(row["arm"] == arm and training._config(row["config"]) == _expected_config(arm, seed, manifest["candidate_kind"]), "sealed role configuration differs")
            ft._check(manifest["assets"][row["checkpoint"]]["sha256"] == row["checkpoint_sha256"], "sealed checkpoint binding differs")
            ft._check(manifest["assets"][row["last_checkpoint"]]["sha256"] == row["last_checkpoint_sha256"], "sealed final checkpoint binding differs")
            ft._check(set(row["arrays"]) == set(row["utility"]) == set(row["representations"]) == set(VIEWS[arm]), "sealed representation inventory differs")
            expected_assets.update((row["checkpoint"], row["last_checkpoint"]))
            if arm == "unconditional96":
                ft._check(all(row[k] is None for k in ("calibrator", "calibrator_sha256", "calibration_provenance", "calibration_diagnostics")), "unconditional pipeline must remain raw-only")
            else:
                ft._check(manifest["assets"][row["calibrator"]]["sha256"] == row["calibrator_sha256"], "sealed map binding differs")
                provenance = row["calibration_provenance"]
                width = 24 if arm == "bce24" else 96
                ft._check(provenance["checkpoint_sha256"] == row["checkpoint_sha256"] and
                          provenance["train_indices_sha256"] == row["training_selection"]["train_indices_sha256"] and
                          provenance["prepared_data_hashes"] == manifest["preparation"][str(width)]["hashes"] and
                          provenance["calibration_seed"] == 7301 and provenance["calibration_samples_per_condition"] == 4096 and
                          provenance["quantile_knots"] == 1025, "sealed map provenance differs")
                expected_assets.update((row["calibrator"], row["calibration_diagnostics"]))
                ft._json(ft._safe_asset(bundle, row["calibration_diagnostics"]))
            values = {}
            for name in VIEWS[arm]:
                width = 96 if name.endswith("96") else 24
                values[name] = ft._array(bundle, row["arrays"][name], (len(labels), width))
                record = row["representations"][name]
                ft._check(record["resolution"] == width and record["profiles_sha256"] == ft._digest(values[name], "<f8"), "representation digest/width differs")
                ridge = ft._load_ridge(bundle, row["utility"][name])
                ft._check(ridge.resolution_ == width and row["utility"][name]["training"]["fit_summary"]["n_profiles"] == TRAIN_ROWS,
                          "ridge native width/full TRAIN counts differ")
                expected_assets.update((row["arrays"][name], row["utility"][name]["artifact"]))
            if arm in ("candidate96", "bce96"):
                for name, predecessor in (("sum_raw24", "raw96"), ("sum_mapped24", "mapped96")):
                    record = row["representations"][name]
                    ft._check(np.array_equal(values[name], _sum4(values[predecessor]).astype(np.float64)) and
                              record["aggregation"] == AGGREGATION and record["predecessor"] == predecessor and
                              record["predecessor_sha256"] == ft._digest(values[predecessor], "<f8"), "hourly generated lineage differs")
            for item in row["evidence"]:
                expected_assets.add(item["artifact"])
                ft._check(manifest["assets"][item["artifact"]]["sha256"] == item["sha256"], "run evidence binding differs")
            run_lookup[(arm, seed)] = row
    ft._check(set(manifest["controls"]) == set(manifest["utility_controls"]) == {"96", "24"}, "both control representations required")
    for width in (96, 24):
        controls, predictors = manifest["controls"][str(width)], manifest["utility_controls"][str(width)]
        ft._check(set(controls) == set(CONTROLS[width]) and set(predictors) == set(RIDGE_CONTROLS), "control inventory differs")
        for filename in controls.values():
            expected_assets.add(filename)
            ft._array(bundle, filename, (len(labels), width))
        for item in predictors.values():
            expected_assets.add(item["artifact"])
            ft._check(ft._load_ridge(bundle, item).resolution_ == width and item["training"]["fit_summary"]["n_profiles"] == TRAIN_ROWS, "control utility width/count differs")
    decision = manifest["decision"]
    ft._check(decision["candidate_kind"] == manifest["candidate_kind"] and isinstance(decision["rationale"], str) and decision["rationale"].strip() and
              decision["protocol_references"] and decision["attempt_history"], "decision/protocol/attempt history missing")
    ft._check(set(decision["coverage"]) == _required_coverage(), "development evidence coverage incomplete")
    for key, item in decision["coverage"].items():
        parts = key.split(".")
        if parts[0] == "native":
            roles, seed, kind = (parts[1],), int(parts[2]), parts[3]
        elif parts[0] == "common_hourly":
            roles, seed, kind = ("candidate96", "bce96", "bce24"), int(parts[1]), parts[2]
        else:
            contrast, seed, kind = parts[1], int(parts[2]), "paired"
            roles = ("bce96", "unconditional96") if contrast == "conditioning" else (("candidate96", "bce96") if contrast == "candidate_reference" else ("bce24", "bce96"))
        _validate_evidence_report(ft._json(ft._safe_asset(bundle, item["artifact"])), roles, seed, kind, run_lookup,
                                  parts[1] if kind == "paired" else None)
    for item in [*decision["coverage"].values(), *decision["protocol_references"], *decision["attempt_history"], *decision["additional_evidence"]]:
        expected_assets.add(item["artifact"])
        ft._check(manifest["assets"][item["artifact"]]["sha256"] == item["sha256"], "decision evidence fingerprint differs")
    ft._check(expected_assets == set(manifest["assets"]), "sealed artifact inventory differs")


def _score(real, labels, values, battery):
    width = real.shape[1]
    joint = ft.conditional_projected_wasserstein(real, labels, values, labels)
    ft._check(joint["directions"]["count"] == 128 and joint["directions"]["seed"] == 9182, "joint projection protocol differs")
    return {"physical": ft.evaluate_conditional(real, labels, values, labels, interval_hours=24 / width),
            "joint": joint, "battery": ft._battery_compare(battery, values, labels, 24 / width)}


def _utility(bundle, description, real, labels):
    model = ft._load_ridge(bundle, description)
    report = model.evaluate(real, labels)
    errors = model.predict(real, labels) - real[:, model.lags_:]
    return report, errors


def _physical_pair(real, labels, meters, first, second):
    return ft.paired_condition_macro_meter_bootstrap(real, labels, first, labels, second, labels, meters,
        interval_hours=24 / real.shape[1], n_bootstrap=BOOTSTRAP_REPLICATES, confidence=.95,
        seed=ft.PROTOCOL["bootstrap_seed"], real_meter_strata=labels // 8, metrics=ft.METRICS, return_per_condition=True)


def _utility_pair(labels, meters, first, second):
    return ft.paired_utility_meter_bootstrap(first, second, labels, meters, n_bootstrap=BOOTSTRAP_REPLICATES,
                                           confidence=.95, seed=ft.PROTOCOL["bootstrap_seed"])


def _contrast_views():
    return {"conditioning": (("bce96", "raw96"), ("unconditional96", "raw96")),
            "candidate_reference": (("candidate96", "mapped96"), ("bce96", "mapped96")),
            "resolution_raw": (("bce24", "raw24"), ("bce96", "sum_raw24")),
            "resolution_calibrated": (("bce24", "mapped24"), ("bce96", "sum_mapped24"))}


def _five_seed_summary(rows):
    result = {}
    for name in rows[0]["methods"]:
        method = [row["methods"][name] for row in rows]
        result[name] = {"physical_macro": {key: ft._stats([ft._compact(m["physical"]["macro"])[key] for m in method]) for key in (*ft.METRICS, "diversity_ratio")},
            "joint_macro": ft._stats([m["joint"]["macro"] for m in method]),
            "battery_macro": {key: ft._stats([m["battery"]["macro"][key] for m in method]) for key in method[0]["battery"]["macro"]},
            "utility_macro": {key: ft._stats([r["utility"][name]["macro"][key] for r in rows]) for key in ("rmse_kwh", "mae_kwh")}}
    return result


def _evaluate(bundle, manifest):
    data = ft._read_verified_test(manifest["test_binding"])
    rng = np.random.default_rng(ft.PROTOCOL["selection_seed"])
    selected = np.concatenate([rng.choice(np.flatnonzero(data["y"] == c),
        min(ft.PROTOCOL["samples_per_condition_cap"], int(np.sum(data["y"] == c))), replace=False) for c in SUPPORT])
    real96, labels, meters, dates = (data[key][selected] for key in ("x", "y", "meter_id", "date"))
    stored = np.load(ft._safe_asset(bundle, manifest["labels_artifact"]), allow_pickle=False)
    ft._check(np.array_equal(labels, stored), "selected TEST labels differ from sealed generated labels")
    del data
    real = {96: real96, 24: _sum4(real96)}
    battery = {width: ft._battery(values, 24 / width) for width, values in real.items()}
    control_reports, control_utility, control_errors, control_arrays = {}, {}, {}, {}
    for width in (96, 24):
        key = str(width)
        control_arrays[width] = {name: ft._array(bundle, path, real[width].shape) for name, path in manifest["controls"][key].items()}
        control_reports[key] = {name: _score(real[width], labels, values, battery[width]) for name, values in control_arrays[width].items()}
        control_utility[key], control_errors[width] = {}, {}
        for name, description in manifest["utility_controls"][key].items():
            report, errors = _utility(bundle, description, real[width], labels)
            control_utility[key][name], control_errors[width][name] = report, errors
    results = {arm: [] for arm in ARMS}
    comparisons = {name: [] for name in CONTRASTS}
    for seed in SEEDS:
        arrays, errors, scored = {}, {}, {}
        for arm in ARMS:
            frozen = next(r for r in manifest["arms"][arm] if r["seed"] == seed)
            methods, utility, per_view_controls = {}, {}, {}
            for view in VIEWS[arm]:
                width = 96 if view.endswith("96") else 24
                values = ft._array(bundle, frozen["arrays"][view], real[width].shape)
                arrays[(arm, view)] = values
                methods[view] = _score(real[width], labels, values, battery[width])
                utility[view], errors[(arm, view)] = _utility(bundle, frozen["utility"][view], real[width], labels)
                ft._check(utility[view]["persistence"] == control_utility[str(width)]["training_real"]["persistence"], "utility tasks/windows differ")
                per_view_controls[view] = {name: {"rmse_ratio": utility[view]["macro"]["rmse_kwh"] / control_utility[str(width)][name]["macro"]["rmse_kwh"]
                    if control_utility[str(width)][name]["macro"]["rmse_kwh"] != 0 else None}
                    for name in ("training_real", "training_resampling")}
            row = {"training_seed": seed, "checkpoint_sha256": frozen["checkpoint_sha256"], "calibrator_sha256": frozen["calibrator_sha256"],
                "methods": methods, "utility": utility, "utility_reference_ratios": per_view_controls,
                "representations": frozen["representations"], "training_provenance": frozen["training_provenance"],
                "calibration_diagnostics": ft._json(ft._safe_asset(bundle, frozen["calibration_diagnostics"])) if frozen["calibration_diagnostics"] else None}
            results[arm].append(row)
            scored[arm] = row
        candidate = scored["candidate96"]
        candidate["paired_physical_controls"] = {name: _physical_pair(real96, labels, meters, arrays[("candidate96", "mapped96")], control_arrays[96][name])
                                                   for name in ("gaussian_copula", "conditional_resampling")}
        candidate["paired_utility_controls"] = {name: _utility_pair(labels, meters, errors[("candidate96", "mapped96")], control_errors[96][name])
                                                  for name in ("training_real", "training_resampling")}
        for contrast, (first, second) in _contrast_views().items():
            width = arrays[first].shape[1]
            ft._check(arrays[second].shape == arrays[first].shape, "paired representation dimensions differ")
            comparisons[contrast].append({"training_seed": seed, "candidate": list(first), "comparator": list(second),
                "difference": "candidate minus comparator; lower physical W1/RMSE/MAE is better",
                "physical": _physical_pair(real[width], labels, meters, arrays[first], arrays[second]),
                "utility": _utility_pair(labels, meters, errors[first], errors[second])})
        del arrays, errors
    summary = {arm: _five_seed_summary(rows) for arm, rows in results.items()}
    paired_summary = {contrast: {kind: {metric: ft._stats([row[kind]["intervals"][metric]["estimate"] for row in rows])
                      for metric in (ft.METRICS if kind == "physical" else ("rmse_kwh", "mae_kwh"))}
                      for kind in ("physical", "utility")} for contrast, rows in comparisons.items()}
    native = summary["candidate96"]["mapped96"]["physical_macro"]
    baseline = control_reports["96"]["gaussian_copula"]["physical"]["macro"]["ramp_wasserstein_kw"]
    reduction = 1 - native["ramp_wasserstein_kw"]["mean"] / baseline if baseline else None
    n_better = sum(row["methods"]["mapped96"]["physical"]["macro"]["ramp_wasserstein_kw"] < baseline for row in results["candidate96"])
    gates = {"scope": "Selected calibrated candidate native96 only; no hourly/reference promotion thresholds.",
        "relative_ramp_reduction": reduction, "seeds_better_than_gaussian": n_better,
        "relative_ramp_criterion_pass": reduction is not None and reduction >= .20 and n_better >= 4,
        "five_seed_mean_physical_pass": {key: native[key]["mean"] <= limit for key, limit in ft.PROTOCOL["physical_limits"].items()},
        "per_seed_physical_pass": {str(row["training_seed"]): {key: ft._compact(row["methods"]["mapped96"]["physical"]["macro"])[key] <= limit
            for key, limit in ft.PROTOCOL["physical_limits"].items()} for row in results["candidate96"]},
        "per_seed_utility_tolerance_pass": {str(row["training_seed"]): all(row["utility"]["mapped96"]["macro"]["rmse_kwh"] <=
            1.1 * control_utility["96"][name]["macro"]["rmse_kwh"] for name in ("training_real", "training_resampling")) for row in results["candidate96"]}}
    ft._check(ft.sha256_file(manifest["test_binding"]["path"]) == manifest["test_binding"]["sha256"], "TEST changed during scoring")
    return {"schema_version": 2, "split": "test", "test_used": True, "fitting_performed": False,
        "protocol": manifest["protocol"], "sources": manifest["sources"], "environment": manifest["environment"],
        "decision": manifest["decision"], "preparation": manifest["preparation"], "test_binding": manifest["test_binding"],
        "test_selection": {"policy": "One uniform-without-replacement selection per condition, shared by all arms/views.",
            "selected_profiles": len(labels), "selected_meters": len(np.unique(meters)), "condition_counts": {str(c): int(np.sum(labels == c)) for c in SUPPORT},
            "indices_sha256": ft._digest(selected, "<i8"), "labels_sha256": ft._digest(labels, "<i8"),
            "meter_ids_sha256": ft.hashlib.sha256(ft._canonical(meters.tolist())).hexdigest(),
            "dates_sha256": ft.hashlib.sha256(ft._canonical(dates.tolist())).hexdigest(),
            "profiles_sha256": {str(width): ft._digest(values, "<f8") for width, values in real.items()}, "hourly_aggregation": AGGREGATION},
        "shared_statistical_controls": control_reports, "shared_utility": control_utility,
        "arms": results, "paired_reference_comparisons": comparisons, "five_seed_summary": summary,
        "paired_point_estimate_summary": paired_summary, "practical_criteria": gates,
        "aggregation": "All five seeds per arm; arithmetic means/sample SD/ranges of point estimates. Shared controls once. Native24 reused for common24. No ensemble, CI averaging or cross-resolution native RMSE contrast.",
        "uncertainty_scope": "Per-seed descriptive marginal technology-stratified meter intervals conditional on fixed arrays/models/maps and selected rows. No five-seed mean CI, selection adjustment or formal noninferiority claim.",
        "limitations": manifest["limitations"]}


def evaluate_reference_study(bundle, expected_seal_sha256, output_dir):
    """Score one immutable unified bundle; no fit, generation or runtime overrides.

    An exclusive access claim prevents concurrent or subsequent TEST attempts on
    this bundle. Pre-access failures remain in the ledger and may be corrected
    only by an explicit new invocation; there is no automatic retry.
    """
    ft._check(ft._is_hash(expected_seal_sha256), "expected seal SHA256 required")
    bundle, output = Path(bundle).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    attempt_id, opened, started_ledger = uuid.uuid4().hex, False, False
    started = {"attempt_id": attempt_id, "started_at": ft._now(), "bundle": str(bundle), "output": str(output),
               "expected_seal_sha256": expected_seal_sha256, "test_access_started": False}
    ft._write(output / "attempt.json", started)
    ledger = bundle / "attempts"
    try:
        ft._check(not (bundle / "manifest.json").is_symlink() and ft.sha256_file(bundle / "manifest.json") == expected_seal_sha256, "manifest seal mismatch")
        manifest = ft._json(bundle / "manifest.json")
        ft._check(ledger.is_dir() and not ledger.is_symlink(), "attempt ledger missing")
        ft._write(ledger / f"{attempt_id}.start.json", started)
        started_ledger = True
        with ft._one_thread():
            _validate_manifest(bundle, manifest)
            ft._write(ledger / "test_access.claim.json", {"attempt_id": attempt_id, "at": ft._now(), "seal_sha256": expected_seal_sha256})
            ft._write(output / "test_access_started.json", {"at": ft._now(), "test_sha256": manifest["test_binding"]["sha256"]})
            opened = True
            report = _evaluate(bundle, manifest)
            ft._check(_sources() == manifest["sources"] and ft._environment() == manifest["environment"], "source/environment changed during scoring")
            ft._verify_assets(bundle, manifest["assets"])
            ft._check(ft.sha256_file(bundle / "manifest.json") == expected_seal_sha256, "seal changed during scoring")
        report.update(seal_sha256=expected_seal_sha256, attempt_id=attempt_id)
        ft._write(output / "report.json", report)
        complete = {"attempt_id": attempt_id, "completed_at": ft._now(), "test_access_started": True,
                    "report_sha256": ft.sha256_file(output / "report.json")}
        ft._write(output / "complete.json", complete)
        ft._write(ledger / f"{attempt_id}.complete.json", complete)
        return report
    except BaseException as error:
        failed = {"attempt_id": attempt_id, "failed_at": ft._now(), "test_access_started": opened,
                  "error_type": type(error).__name__, "error": str(error)}
        ft._write(output / "failure.json", failed)
        if started_ledger:
            ft._write(ledger / f"{attempt_id}.failure.json", failed)
        raise
