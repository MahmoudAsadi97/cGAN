"""Paired validation of a direct TRAIN-quantile decoder and frozen original SWD10.

The candidate has no post-hoc map. Both arms and their shared statistical
controls are reevaluated in this process under the same numerical source.
Physical and full-TRAIN utility reports are separate serialized operations.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_checkpoints import _checkpoint, _digest, _validate_manifest
from evaluate_calibration import _battery_diagnostics, _compact_application, _compact_fitting, _profile_digest
from fluvius_cgan import training
from fluvius_cgan.battery import simulate_battery
from fluvius_cgan.calibration import load_verified_calibrator
from fluvius_cgan.copula import fit_conditional_copula
from fluvius_cgan.evaluation import evaluate_conditional, mean_baseline, nearest_neighbor_diagnostics, resample_baseline
from fluvius_cgan.frozen_test import _forbid_test_inputs
from fluvius_cgan.joint import conditional_projected_wasserstein
from fluvius_cgan.release import sha256_file
from fluvius_cgan.utility import RidgeProfileForecaster

SEEDS = (11, 22, 33)
SUPPORT = list(range(64))
TRAIN_ROWS = 606060
EPOCHS = 26
CAP, SEED = 100, 2026
SATURATION_THRESHOLD = 1e-6
BASELINE_BINDINGS = ROOT / "configs/quantile_decoder_baseline_bindings.json"
BASELINE_BINDINGS_SHA256 = "046ddf6d3ba66ac16507539c494422c9bc322505da00bb9a35e848dc4223f9e9"
CONFIG_PATH = ROOT / "configs/curated_quantile_decoder.json"
PROTOCOL_PATH = ROOT / "docs/QUANTILE_DECODER_REFINEMENT.md"
REQUIRED_SOURCES = tuple(f"src/fluvius_cgan/{name}.py" for name in
    ("training", "models", "residual", "distribution_loss", "temporal_loss", "quantile_decoder",
     "evaluation", "calibration", "copula", "joint", "battery", "utility")) + tuple(f"scripts/{name}.py" for name in
    ("evaluate_quantile_decoder", "assess_quantile_decoder_pilot", "compare_checkpoints", "evaluate_calibration", "evaluate_utility"))
PREPARED_FILES = ("train.npz", "val.npz", "metadata.json", "scaler.json", "curation_manifest.json")
RUN_FILES = ("best.pt", "last.pt", "config.json", "summary.json", "history.json")
PHYSICAL_METHODS = ("quantile_decoder_cgan", "reference_raw_cgan", "reference_calibrated_cgan",
                    "gaussian_copula", "independent_marginals", "exact_marginal_copula",
                    "conditional_resampling", "conditional_mean")


def _check(value, message):
    if not value:
        raise ValueError(message)


def _json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _raw_digest(values):
    values = np.asarray(values)
    dtype = values.dtype.newbyteorder("<")
    digest = hashlib.sha256()
    for start in range(0, len(values), 4096):
        digest.update(np.asarray(values[start:start + 4096], dtype=dtype).tobytes(order="C"))
    return digest.hexdigest()


def _string_digest(values):
    return hashlib.sha256(json.dumps(np.asarray(values).tolist(), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _preflight(prepared, paths):
    _check(all(Path(path).name != "test.npz" and Path(path).resolve().name != "test.npz" for path in paths), "TEST input forbidden")
    _forbid_test_inputs(Path(prepared).resolve(), paths)


def _snapshot(prepared, paths):
    _preflight(prepared, paths.values())
    return {name: sha256_file(path) for name, path in paths.items()}


def _source(declared_paths):
    result = training._provenance(torch.device("cpu"))
    result["source_files_sha256"].update({name: sha256_file(ROOT / name) for name in declared_paths})
    result.update(scipy=str(scipy.__version__), runner_sha256=sha256_file(__file__),
                  helper_files_sha256={name: sha256_file(ROOT / name) for name in
                      ("scripts/compare_checkpoints.py", "scripts/evaluate_calibration.py")})
    return result


def _declaration(path, prepared):
    _preflight(prepared, [path])
    value = _json(path)
    names = value.get("source_files_sha256", {})
    _check(isinstance(names, dict) and set(REQUIRED_SOURCES) <= set(names), "execution declaration source inventory incomplete")
    for name in names:
        parsed = Path(name)
        _check(isinstance(name, str) and not parsed.is_absolute() and ".." not in parsed.parts and
               parsed.as_posix() == name and (ROOT / parsed).resolve().is_relative_to(ROOT.resolve()), "invalid declared source path")
    _preflight(prepared, [ROOT / name for name in names])
    source = _source(names)
    _check(value.get("schema_version") == 1 and value.get("protocol_id") == "conditional_physical_quantile_decoder_v1" and
           value.get("source_commit") == source["git_commit"] and value.get("source_tree") == source["git_commit_tree"] and
           source["git_dirty"] is False and all(source["source_files_sha256"][name] == digest for name, digest in names.items()),
           "execution declaration does not bind this clean numerical source")
    expected = {"seed": SEED, "samples_per_condition": CAP, "calibration_seed": 7301,
        "calibration_samples_per_condition": 4096, "quantile_knots": 1025, "candidate_posthoc_calibration": False,
        "baseline_calibration": "frozen_original_saved_map"}
    _check(value.get("evaluation") == expected and value.get("seeds") == list(SEEDS), "execution declaration endpoint settings differ")
    return value, source


def _completion(path, state, hashes):
    directory = Path(path).parent
    summary, history, config = (_json(directory / name) for name in ("summary.json", "history.json", "config.json"))
    _check(config == state["config"], "saved training config differs")
    updates = (TRAIN_ROWS + config["batch_size"] - 1) // config["batch_size"]
    _check(summary.get("test_used") is False and summary.get("epochs_completed") == EPOCHS and
           summary.get("stopped_early") is False and len(history) == EPOCHS and
           [row["epoch"] for row in history] == list(range(1, EPOCHS + 1)), "complete fixed training history required")
    _check(all(row["generator_updates"] == row["discriminator_updates"] == updates and
               row["samples_seen"] == updates * config["batch_size"] and math.isfinite(row["validation_score"]) for row in history),
           "per-epoch exposure or score differs")
    _check(summary.get("generator_updates") == summary.get("discriminator_updates") == updates * EPOCHS and
           summary.get("samples_seen") == updates * EPOCHS * config["batch_size"], "training exposure differs")
    best = min(history, key=lambda row: row["validation_score"])
    _check(state["epoch"] == summary["best_epoch"] == best["epoch"] and
           state["best_validation_score"] == summary["best_validation_score"] == best["validation_score"], "selected checkpoint differs from earliest selector minimum")
    for key in ("data_hashes", "selection", "provenance", "data_curation", "scaler"):
        _check(summary.get(key) == state.get(key), f"summary/checkpoint {key} differs")
    last = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
    for key in ("config", "metadata", "scaler", "data_hashes", "selection", "provenance", "data_curation", "quantile_decoder"):
        _check(last.get(key) == state.get(key), f"last/best {key} differs")
    _check(last.get("epoch") == EPOCHS and last.get("generator_updates") == last.get("discriminator_updates") == updates * EPOCHS and
           last.get("samples_seen") == updates * EPOCHS * config["batch_size"] and last.get("best_epoch") == best["epoch"] and
           last.get("validation_score") == history[-1]["validation_score"], "last checkpoint does not prove full budget")
    for saved, count in ((state, updates * state["epoch"]), (last, updates * EPOCHS)):
        for key in ("optimizer_g_state_dict", "optimizer_d_state_dict"):
            if key in saved:
                steps = [float(row["step"]) for row in saved[key]["state"].values() if "step" in row]
                _check(steps and all(step == count for step in steps), "optimizer step count differs")
    return {"summary": summary, "history": history, "config": config, "artifact_sha256": hashes}


def _prepare(prepared, checkpoint, reference_checkpoint, reference_calibrator, declaration, execution):
    prepared, checkpoint, reference_checkpoint, reference_calibrator = map(Path, (prepared, checkpoint, reference_checkpoint, reference_calibrator))
    _check(checkpoint.name == reference_checkpoint.name == "best.pt", "fixed selected best.pt artifacts required")
    paths = {f"prepared/{name}": prepared / name for name in PREPARED_FILES}
    paths.update({f"{arm}/{name}": path.parent / name for arm, path in (("candidate", checkpoint), ("reference", reference_checkpoint)) for name in RUN_FILES})
    paths.update(reference_calibrator=reference_calibrator, baseline_bindings=BASELINE_BINDINGS,
                 execution_declaration=Path(declaration), candidate_config=CONFIG_PATH, protocol=PROTOCOL_PATH)
    before = _snapshot(prepared, paths)
    _check(before["baseline_bindings"] == BASELINE_BINDINGS_SHA256, "declared original comparator binding changed")
    _check(execution["baseline_bindings_sha256"] == before["baseline_bindings"] and execution["config_sha256"] == before["candidate_config"] and
           execution["protocol_sha256"] == before["protocol"], "execution declaration artifact fingerprints differ")
    bindings = _json(BASELINE_BINDINGS)
    guard = training._training_curation_guard(prepared, {"allow_uncurated_data": False})
    _check(guard["status"] == "verified_curation_v2" and guard["pinned_release"], "pinned verified curated96 required")
    train, val, metadata, scaler, hashes = training._load_prepared(prepared)
    _validate_manifest(metadata, hashes, train, val)
    _check(metadata["resolution"] == 96 and metadata["target"] == "net" and metadata["units"] == "kWh/interval" and
           len(train["x"]) == TRAIN_ROWS and all(np.unique(part["y"]).tolist() == SUPPORT for part in (train, val)), "full native96 TRAIN/all64 support required")
    _check(hashes == guard["verified_file_hashes"] and all(before[f"prepared/{name}"] == value for name, value in hashes.items()), "prepared fingerprints differ")
    bound_hashes = {**hashes, "curation_manifest.json": guard["manifest_sha256"]}
    updates = (TRAIN_ROWS + 511) // 512
    _check(execution["prepared_data_hashes"] == hashes and execution["checkpoint_data_hashes"] == bound_hashes and
           execution["budget"] == {"train_profiles": TRAIN_ROWS, "epochs": EPOCHS, "updates_per_adversary": updates * EPOCHS,
                                   "updates_per_epoch": updates}, "execution declaration data/budget differs")
    details, states, completions = {}, {}, {}
    for arm, path in (("candidate", checkpoint), ("reference", reference_checkpoint)):
        details[arm], indices = _checkpoint(path, train, metadata, scaler, bound_hashes)
        _check(np.array_equal(indices, np.arange(TRAIN_ROWS)), "both arms must use every actual TRAIN row in order")
        states[arm] = torch.load(path, map_location="cpu", weights_only=True)
        status = states[arm].get("data_curation", {})
        _check(status.get("status") == "verified_curation_v2" and status.get("verified_file_hashes") == hashes and
               status.get("manifest_sha256") == guard["manifest_sha256"] and
               states[arm].get("validation_metric") == training.CONDITIONAL_VALIDATION_METRIC,
               "curation or raw condition-macro checkpoint selector differs")
        completions[arm] = _completion(path, states[arm], {name: before[f"{arm}/{name}"] for name in RUN_FILES})
    seed = details["candidate"]["config"]["seed"]
    _check(seed in SEEDS and seed == details["reference"]["config"]["seed"], "matching fixed pilot training seed required")
    original = bindings["seeds"][str(seed)]
    reference = details["reference"]
    _check(reference["sha256"] == original["checkpoint_sha256"] and before["reference_calibrator"] == original["calibration_sha256"] and
           reference["epoch"] == original["selected_epoch"] and reference["config"] == original["config"] and
           reference["training_provenance"] == original["training_source"] and hashes == original["prepared_data_hashes"] and
           reference["data_hashes"] == original["checkpoint_data_hashes"], "reference is not the prospectively fixed original SWD10 artifact")
    # The original artifact is hash-pinned and uses the identical full TRAIN
    # pool. Bind its standardizer without allocating another full float64 copy.
    _check(states["candidate"]["scaler"] == states["reference"]["scaler"], "candidate/reference TRAIN standardizers differ")
    added_selection = {"train_labels_sha256", "train_profiles_sha256", "train_profiles_per_condition"}
    paired_selection = {arm: {key: value for key, value in state["selection"].items() if key not in added_selection}
                        for arm, state in states.items()}
    _check(paired_selection["candidate"] == paired_selection["reference"], "candidate/reference fixed selector or sampling records differ")
    configs = {name: training._config(row["config"]) for name, row in details.items()}
    _check(configs["candidate"]["conditional_quantile_decoder"] is True and configs["reference"]["conditional_quantile_decoder"] is False,
           "candidate must decode directly; original reference must remain unchanged")
    _check({k: v for k, v in configs["candidate"].items() if k != "conditional_quantile_decoder"} ==
           {k: v for k, v in configs["reference"].items() if k != "conditional_quantile_decoder"}, "candidate changes more than the declared decoder package")
    _check(details["candidate"]["training_provenance"]["git_dirty"] is False, "candidate training source must be clean")
    provenance = details["candidate"]["training_provenance"]
    _check(provenance["git_commit"] == execution["source_commit"] and provenance["git_commit_tree"] == execution["source_tree"] and
           all(provenance["source_files_sha256"].get(name) == digest for name, digest in execution["source_files_sha256"].items()
               if name.startswith("src/fluvius_cgan/")), "candidate training differs from execution source declaration")
    declared_config = training._config({**_json(CONFIG_PATH), "seed": seed})
    _check(configs["candidate"] == declared_config, "candidate differs from declared config")
    decoder = training.load_quantile_decoded_generator(states["candidate"], device="cpu").decoder
    fit_metadata = decoder.fit_metadata
    _check(fit_metadata["selected_profiles"] == TRAIN_ROWS and fit_metadata["selected_indices_sha256"] == _digest(np.arange(TRAIN_ROWS), "<i8") and
           fit_metadata["selected_profiles_sha256"] == _raw_digest(train["x"]) and fit_metadata["selected_labels_sha256"] == _digest(train["y"], "<i8"),
           "decoder fitted-row hashes differ from actual selected TRAIN")
    fixed_map, map_details = load_verified_calibrator(reference_calibrator, checkpoint_sha256=reference["sha256"],
        train_indices_sha256=reference["training_selection"]["train_indices_sha256"], prepared_data_hashes=hashes,
        generation_seed=SEED, n_features=96, train_labels=train["y"])
    _check(fixed_map.quantile_knots == 1025 and fixed_map.provenance["calibration_seed"] == 7301 and
           fixed_map.provenance["calibration_samples_per_condition"] == 4096, "reference map must use4096/7301/1025")
    rng = np.random.default_rng(SEED)
    selected = np.concatenate([rng.choice(np.flatnonzero(val["y"] == label), min(CAP, int(np.sum(val["y"] == label))), replace=False) for label in SUPPORT])
    return {"prepared": prepared, "paths": paths, "input_sha256": before, "train": train, "val": val, "metadata": metadata,
            "hashes": hashes, "guard": guard, "details": details, "states": states, "training_completion": completions,
            "decoder": decoder, "fixed_map": fixed_map, "reference_calibration": map_details, "selected": selected,
            "original_binding": original, "training_seed": seed}


def _generate(context, arm, labels):
    path = context["paths"][f"{arm}/best.pt"]
    values = training.generate(path, labels, seed=SEED, batch_size=1024)
    _check(sha256_file(path) == context["details"][arm]["sha256"], "checkpoint changed during generation")
    _check(values.shape == (len(labels), 96) and values.dtype == np.float32 and np.isfinite(values).all(), "physical generation dtype/shape/finiteness differs")
    return values


def _diagnostics(context, values, probabilities, labels):
    table = context["decoder"].quantiles.detach().cpu().numpy()
    _check(values.shape == (len(labels), 96) and values.dtype == np.float32 and np.isfinite(values).all() and
           table.shape == (64, 1025, 96) and table.dtype == np.float64 and probabilities.shape == values.shape and
           probabilities.dtype == np.float64 and np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1)), "decoder diagnostics dimensions/dtypes differ")
    per_condition = {}
    for label in SUPPORT:
        mask = labels == label
        x, p = values[mask], probabilities[mask]
        train = context["train"]["x"][context["train"]["y"] == label]
        low, high = table[label, 0].astype(np.float32), table[label, -1].astype(np.float32)
        atom_counts, tie_counts = [], []
        for slot in range(96):
            counts = np.unique(x[:, slot], return_counts=True)[1]
            atom_counts.append(int(counts.max()))
            tie_counts.append(int(counts[counts > 1].sum()))
        events = {"probability_zero": p == 0, "probability_one": p == 1,
                  "probability_near_zero": p <= SATURATION_THRESHOLD, "probability_near_one": p >= 1-SATURATION_THRESHOLD,
                  "below_decoded_endpoint": x < low, "above_decoded_endpoint": x > high,
                  "at_lower_decoded_endpoint": x == low, "at_upper_decoded_endpoint": x == high,
                  "below_actual_train_minimum": x < train.min(axis=0), "above_actual_train_maximum": x > train.max(axis=0)}
        per_condition[str(label)] = {"count": len(x), "slot_event_counts": {key: value.sum(axis=0).tolist() for key, value in events.items()},
            "fractions_over_profile_slots": {key: float(value.mean()) for key, value in events.items()},
            "fixed_zero_width_segment_fraction_by_slot": np.mean(table[label, 1:] == table[label, :-1], axis=0).tolist(),
            "fixed_constant_slots": np.flatnonzero(table[label, 0] == table[label, -1]).tolist(),
            "sample_tied_observation_counts_by_slot": tie_counts, "sample_largest_atom_counts_by_slot": atom_counts}
    _check(not any(sum(row["slot_event_counts"][key]) for row in per_condition.values() for key in ("below_decoded_endpoint", "above_decoded_endpoint")), "decoded physical output left fixed endpoint support")
    return {"fit_metadata": context["decoder"].fit_metadata, "physical_dtype": str(values.dtype), "probability_dtype": str(probabilities.dtype),
            "quantile_table_dtype": str(table.dtype), "near_saturation_threshold": SATURATION_THRESHOLD,
            "sample_policy": "Same cap100/seed2026 physical draw; no extra reference draw, fitting, ranking, or clipping.",
            "per_condition": per_condition,
            "interpretation": "Fixed zero-width segment fractions refer to a uniform probability grid, not learned atom probabilities. Sample ties/saturation are descriptive. TRAIN extrema bound this representation; no exact empirical marginal or tail extrapolation claim."}


def _physical(context):
    selected, val, train = context["selected"], context["val"], context["train"]
    real, labels = val["x"][selected], val["y"][selected]
    candidate, probabilities, metadata = training.generate_quantile_decoder_diagnostics(context["paths"]["candidate/best.pt"], labels, seed=SEED, batch_size=1024)
    _check(metadata == context["decoder"].fit_metadata, "diagnostic generation decoder metadata differs")
    diagnostics = _diagnostics(context, candidate, probabilities, labels)
    raw = _generate(context, "reference", labels)
    arrays = {"quantile_decoder_cgan": candidate, "reference_raw_cgan": raw,
              "reference_calibrated_cgan": context["fixed_map"].transform(raw, labels)}
    for name, knots, shrinkage in (("gaussian_copula", 1025, .01), ("independent_marginals", 1025, 1.), ("exact_marginal_copula", None, .01)):
        model = fit_conditional_copula(train["x"], train["y"], quantile_knots=knots, shrinkage=shrinkage)
        arrays[name] = model.generate(labels, seed=SEED)
        del model
    arrays["conditional_resampling"] = resample_baseline(train["x"], train["y"], labels, seed=SEED)
    arrays["conditional_mean"] = mean_baseline(train["x"], train["y"], labels)
    real_battery = simulate_battery(real, .25)
    return {"methods": {name: evaluate_conditional(real, labels, values, labels, interval_hours=.25) for name, values in arrays.items()},
            "joint_diagnostics": {name: conditional_projected_wasserstein(real, labels, values, labels) for name, values in arrays.items()},
            "battery_diagnostics": {name: _battery_diagnostics(real_battery, values, labels, .25) for name, values in arrays.items()},
            "quantile_decoder": diagnostics, "reference_map_diagnostics": {
                "fitting": _compact_fitting(context["fixed_map"].fitting_diagnostics()),
                "application": _compact_application(context["fixed_map"].diagnostics(raw, labels), 96)},
            "nearest_neighbors": {name: nearest_neighbor_diagnostics(train["x"], arrays[name], real, seed=SEED)
                                  for name in ("quantile_decoder_cgan", "reference_calibrated_cgan")},
            "frozen_samples": {"real_profiles_sha256": _profile_digest(real), "labels_sha256": _digest(labels, "<i8"),
                "profiles_sha256": {name: _profile_digest(values) for name, values in arrays.items()},
                "profile_dtypes": {name: str(values.dtype) for name, values in arrays.items()},
                "candidate_probabilities_sha256": _profile_digest(probabilities)},
            "baselines": {"gaussian_copula": {"quantile_knots": 1025, "shrinkage": .01},
                          "independent_marginals": {"quantile_knots": 1025, "shrinkage": 1.},
                          "exact_marginal_copula": {"quantile_knots": None, "shrinkage": .01}}}


def _utility(context):
    train, val, selected = context["train"], context["val"], context["selected"]
    tx, ty, vx, vy = train["x"], train["y"], val["x"][selected], val["y"][selected]
    methods, fits, persistence, features = {}, {}, None, None
    def score(name, values):
        nonlocal persistence, features
        _check(values.shape == tx.shape and np.isfinite(values).all(), "utility TRAIN profile shape/finiteness differs")
        model = RidgeProfileForecaster(batch_profiles=256).fit(values, ty)
        metrics = model.evaluate(vx, vy)
        observed = metrics.pop("persistence")
        if persistence is None:
            persistence, features = observed, model.feature_names_
        _check(persistence == observed and features == model.feature_names_, "utility tasks/windows differ")
        methods[name] = metrics
        fits[name] = {**model.fit_summary_, "profiles_sha256": _profile_digest(values), "profile_dtype": str(values.dtype),
                      "labels_sha256": _digest(ty, "<i8"), "coefficients_sha256": _digest(model.coef_, "<f8")}
    score("training_real", tx)
    values = _generate(context, "candidate", ty)
    score("training_quantile_decoder", values)
    del values
    values = _generate(context, "reference", ty)
    score("training_reference_raw", values)
    # Transform in small blocks into a single physical array. Avoid an additional
    # full reference/probability workspace while preserving the fixed map output.
    mapped = np.empty(values.shape, dtype=np.float64)
    for start in range(0, len(values), 256):
        mapped[start:start+256] = context["fixed_map"].transform(values[start:start+256], ty[start:start+256])
    del values
    score("training_reference_calibrated", mapped)
    del mapped
    values = resample_baseline(tx, ty, ty, seed=SEED)
    score("training_resampling", values)
    del values
    copula = fit_conditional_copula(tx, ty, quantile_knots=1025, shrinkage=.01)
    values = copula.generate(ty, seed=SEED, batch_size=4096)
    score("training_copula", values)
    del values, copula
    methods["persistence"] = persistence
    return {"methods": methods, "training": fits,
            "quantile_decoder": {"fit_metadata": context["decoder"].fit_metadata, "physical_dtype": "float32",
                "diagnostics_scope": "Probability/support diagnostics are retained in the paired physical report, not allocated for full TRAIN utility."},
            "task": {"name": "within_day_observed_history_next_interval_linear_ridge", "resolution": 96, "lags": 12,
                "forecast_horizon_intervals": 1, "forecast_horizon_hours": .25, "windows_per_profile": 84,
                "target_slots_inclusive": [12, 95], "feature_names": features, "ridge_alpha": 1.,
                "ridge_objective": "sum_squared_error_plus_alpha_nonintercept_squared_norm", "batch_profiles": 256},
            "utility_training_selection": {"cap_applied": False, "selected_profiles": len(tx),
                "selected_indices_sha256": _digest(np.arange(len(tx)), "<i8"), "labels_sha256": _digest(ty, "<i8"),
                "condition_counts": {str(c): int(np.sum(ty == c)) for c in SUPPORT}, "real_profiles_sha256": _profile_digest(tx)},
            "copula": {"enabled": True, "quantile_knots": 1025, "shrinkage": .01, "source_profiles": len(tx)},
            "invariants": {"full_actual_train_required": True, "identical_training_label_order": True,
                "fit_uses_validation_values": False, "candidate_posthoc_map": False, "reference_map_refitted": False}}


def evaluate(prepared, checkpoint, reference_checkpoint, reference_calibrator, *, mode, declaration):
    _check(mode in ("physical", "utility"), "mode must be physical or utility")
    started = time.perf_counter()
    execution, source = _declaration(declaration, prepared)
    context = _prepare(prepared, checkpoint, reference_checkpoint, reference_calibrator, declaration, execution)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        limits = nullcontext()
    else:
        limits = threadpool_limits(limits=1, user_api="blas")
    with limits:
        result = _physical(context) if mode == "physical" else _utility(context)
    post = _snapshot(context["prepared"], context["paths"])
    _check(post == context["input_sha256"] and _source(execution["source_files_sha256"]) == source, "inputs/source changed during evaluation")
    selected, val = context["selected"], context["val"]
    labels = val["y"][selected]
    return {"schema_version": 1, "report_kind": f"quantile_decoder_pilot_{mode}", "split": "val", "test_used": False,
            "scope": "Exploratory fixed three-seed package pilot; no checkpoint selection, final TEST, or new promotion criteria.",
            "training_seed": context["training_seed"], "candidate": context["details"]["candidate"], "reference": context["details"]["reference"],
            "reference_calibration": context["reference_calibration"], "training_completion": context["training_completion"],
            "baseline_binding": {"sha256": BASELINE_BINDINGS_SHA256, "record": context["original_binding"]},
            "execution_declaration": {"sha256": context["input_sha256"]["execution_declaration"],
                "source_commit": execution["source_commit"], "source_tree": execution["source_tree"]},
            "endpoints": execution["evaluation"],
            "prepared_data_hashes": context["hashes"], "curation_manifest_sha256": context["guard"]["manifest_sha256"],
            "data_curation_status": "verified_curation_v2", "curation_metadata": context["metadata"]["curation"],
            "source_sha256": context["metadata"]["source_sha256"], "target": "net", "units": "kWh/interval", "resolution": 96, "interval_hours": .25,
            "training_profiles": {"prepared": TRAIN_ROWS, "candidate": TRAIN_ROWS, "reference": TRAIN_ROWS, "controls": TRAIN_ROWS},
            "seed": SEED, "samples_per_condition_cap": CAP,
            "seeds": {"validation_selection": SEED, "generation": SEED, "controls": SEED, "reference_calibration": 7301},
            "validation_selection": {"samples_per_condition_cap": CAP, "seed": SEED, "selected_profiles": len(selected),
                "selected_indices_sha256": _digest(selected, "<i8"), "labels_sha256": _digest(labels, "<i8"),
                "profiles_sha256": _profile_digest(val["x"][selected]), "meter_ids_sha256": _string_digest(val["meter_id"][selected]),
                "dates_sha256": _string_digest(val["date"][selected]), "condition_counts": {str(c): int(np.sum(labels == c)) for c in SUPPORT}},
            "input_paths": {name: str(path.resolve()) for name, path in context["paths"].items()},
            "input_sha256": context["input_sha256"], "post_input_sha256": post, "provenance": source,
            "candidate_posthoc_calibration": False, "reference_calibration_refitted": False,
            "seconds": time.perf_counter() - started, **result,
            "limitations": ["The fixed decoder supplies TRAIN-bounded support; sigmoid outputs do not guarantee uniform probabilities or exact empirical margins.",
                "Only the original reference uses its saved post-hoc map. Both selected pipelines and controls are reevaluated under the same corrected numerical source.",
                "Nearest-neighbor summaries are descriptive and provide no privacy guarantee.",
                "Validation has been reused for model development. One observed-history ridge task and one hypothetical daily battery do not establish general forecasting or annual economic validity."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "checkpoint", "reference-checkpoint", "reference-calibrator", "declaration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("physical", "utility"), required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.suffix.lower() != ".json":
        parser.error("output must be a new .json path")
    result = evaluate(args.prepared, args.checkpoint, args.reference_checkpoint, args.reference_calibrator, mode=args.mode, declaration=args.declaration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved validation-only quantile-decoder {args.mode}: {args.output}")


if __name__ == "__main__":
    main()
