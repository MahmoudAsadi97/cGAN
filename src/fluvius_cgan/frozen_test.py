"""Seal TRAIN-only outputs, then evaluate an immutable five-seed TEST study.

The freeze stage never opens or hashes TEST. Evaluation never generates or fits
anything: it scores sealed arrays and ridge states. This is an internal holdout
evaluation, not a guarantee of independent households or external validity.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import uuid

import numpy as np
import scipy
from scipy.stats import wasserstein_distance
import torch

from . import training
from .battery import simulate_battery
from .calibration import load_verified_calibrator
from .copula import fit_conditional_copula
from .evaluation import evaluate_conditional, resample_baseline
from .joint import conditional_projected_wasserstein
from .release import sha256_file
from .uncertainty import paired_condition_macro_meter_bootstrap
from .utility import RidgeProfileForecaster, _feature_names
from .utility_uncertainty import paired_utility_meter_bootstrap

SEEDS = (11, 22, 33, 44, 55)
SUPPORT = list(range(64))
PREPARED = ("train.npz", "val.npz", "metadata.json", "scaler.json")
BASELINES = ("gaussian_copula", "independent_marginals", "exact_marginal_copula", "conditional_resampling")
TRAINING_CORE = ("training.py", "models.py", "residual.py", "distribution_loss.py", "temporal_loss.py")
METRICS = ("pointwise_wasserstein_kwh", "daily_energy_wasserstein_kwh",
           "peak_power_wasserstein_kw", "ramp_wasserstein_kw", "lag_one_acf_wasserstein")
PROTOCOL = {"samples_per_condition_cap": 1000, "selection_seed": 9201,
            "generation_seed": 9202, "bootstrap_seed": 9203, "utility_seed": 9204,
            "condition_support": SUPPORT, "condition_weighting": "equal_all_64",
            "quantile_knots": 1025, "calibration_seed": 7301,
            "calibration_samples_per_condition": 4096, "copula_shrinkage": .01,
            "primary_metric": "ramp_wasserstein_kw", "primary_comparator": "gaussian_copula",
            "relative_ramp_improvement": .20, "seeds_required_better_than_copula": 4,
            "physical_limits": {"daily_energy_wasserstein_kwh": 5., "peak_power_wasserstein_kw": 1.,
                                "ramp_wasserstein_kw": .10, "lag_one_acf_wasserstein": .10},
            "utility_rmse_ratio_limit": 1.10,
            "projection_count": 128, "projection_seed": 9182,
            "battery": {"capacity_kwh": 10., "charge_power_kw": 4., "discharge_power_kw": 4.,
                        "eta_charge": math.sqrt(.9), "eta_discharge": math.sqrt(.9),
                        "initial_soc_kwh": 0.}}
LIMITATIONS = [
    "Identity-audited internal meter holdout from the same source/year; exact trace uniqueness does not establish household independence or exclude near-duplicates.",
    "The method and checkpoints were selected using development validation. This frozen test does not validate all earlier exploratory comparisons.",
    "Calibration supplies approximate bounded training marginals; marginal fidelity is not independent evidence of learned dependence or tail extrapolation.",
    "Five-seed SD is distinct from conditional real-meter bootstrap uncertainty; neither captures all model, calibration, selection or synthetic Monte Carlo uncertainty.",
    "Finite projections and a single hypothetical daily battery scenario do not certify complete joint fidelity, annual battery performance, costs or PV attribution.",
    "Observed-history ridge utility is not day-ahead, recursive or future-period forecasting; its 10% tolerance is not a formal noninferiority test.",
    "Reference-architecture, unconditional and hourly manuscript ablations remain separate requirements; this evaluates one candidate and statistical controls, without an ensemble.",
    "Subsequent tuning using these outcomes exhausts this set as performance-unseen confirmation.",
]


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    value = json.loads(Path(path).read_bytes(), parse_constant=invalid)
    _check(isinstance(value, dict), "expected a JSON object")
    return value


def _write(path, value):
    with Path(path).open("xb") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False).encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _digest(values, dtype):
    digest = hashlib.sha256()
    for start in range(0, len(values), 256):
        digest.update(np.asarray(values[start:start + 256], dtype=dtype).tobytes(order="C"))
    return digest.hexdigest()


def _is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            yield
        else:
            with threadpool_limits(limits=1, user_api="blas"):
                yield
    finally:
        torch.set_num_threads(previous)


def _environment():
    try:
        import threadpoolctl
    except ImportError:
        pools, version = [], None
    else:
        pools = [{key: row.get(key) for key in ("internal_api", "prefix", "version", "architecture", "num_threads")}
                 for row in threadpoolctl.threadpool_info()]
        pools.sort(key=lambda row: json.dumps(row, sort_keys=True))
        version = threadpoolctl.__version__
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": str(torch.__version__), "threadpoolctl": version, "blas": pools,
            "platform": platform.system(), "machine": platform.machine(), "torch_threads": torch.get_num_threads()}


def _sources():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src" / "fluvius_cgan").glob("*.py"))
    paths += [root / "scripts" / name for name in ("freeze_test_manifest.py", "evaluate_frozen_test.py")]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"git_commit": commit, "files_sha256": {str(p.relative_to(root)): sha256_file(p) for p in paths}}


def _safe_asset(bundle, filename):
    _check(isinstance(filename, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", filename) is not None,
           "invalid artifact path")
    directory = Path(bundle) / "artifacts"
    path = directory / filename
    _check(not directory.is_symlink() and not path.is_symlink() and path.resolve().parent == directory.resolve(),
           "artifact path must remain inside the bundle without symlinks")
    return path


def _record_asset(bundle, filename, assets):
    path = _safe_asset(bundle, filename)
    assets[filename] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return filename


def _save_array(bundle, filename, values, assets):
    with _safe_asset(bundle, filename).open("xb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
    return _record_asset(bundle, filename, assets)


def _save_ridge(bundle, filename, profiles, labels, assets):
    model = RidgeProfileForecaster(batch_profiles=256).fit(profiles, labels)
    state = {"resolution": model.resolution_, "lags": model.lags_, "support": model.condition_support_.tolist(),
             "feature_names": model.feature_names_, "fit_summary": model.fit_summary_,
             "profiles_sha256": _digest(profiles, "<f8"), "labels_sha256": _digest(labels, "<i8")}
    with _safe_asset(bundle, filename).open("xb") as handle:
        np.savez(handle, coefficients=model.coef_, metadata=np.asarray(json.dumps(state, allow_nan=False)))
    return {"artifact": _record_asset(bundle, filename, assets), "training": state}


def _load_ridge(bundle, description):
    with np.load(_safe_asset(bundle, description["artifact"]), allow_pickle=False) as saved:
        _check(set(saved.files) == {"coefficients", "metadata"}, "invalid ridge artifact arrays")
        state = json.loads(str(saved["metadata"].item()))
        coefficients = np.asarray(saved["coefficients"])
    _check(state == description["training"], "ridge metadata disagrees with seal")
    width, lags = state["resolution"], state["lags"]
    _check(width in (24, 96) and lags == width // 8 and state["support"] == SUPPORT and
           state["feature_names"] == _feature_names(lags), "invalid frozen ridge features/support")
    _check(coefficients.dtype == np.float64 and coefficients.shape == (lags + 69,) and
           np.isfinite(coefficients).all(), "invalid frozen ridge coefficients")
    model = RidgeProfileForecaster(lags=lags, batch_profiles=256)
    model.coef_, model.resolution_, model.lags_ = coefficients, width, lags
    model.condition_support_ = np.asarray(SUPPORT, dtype=np.int64)
    model.feature_names_, model.fit_summary_ = state["feature_names"], state["fit_summary"]
    return model


def _test_binding(prepared, metadata, curation_manifest):
    curation, split = metadata["curation"], metadata["splits"]["test"]
    expected = curation_manifest["output_files_sha256"]["test.npz"]
    _check(_is_hash(expected) and split["sha256"] == expected, "curated TEST hashes disagree")
    meters = split["meter_ids"]
    _check(isinstance(meters, list) and meters == sorted(set(meters)) and bool(meters), "invalid TEST membership")
    holdout = curation["holdout"]
    _check(curation_manifest["holdout"] == holdout and holdout["retained_original_test_meter_ids"] == meters,
           "curation holdout membership disagrees")
    original = set(curation["before"]["test"]["meter_ids"])
    excluded = set(holdout["excluded_original_test_meter_ids"])
    _check(set(meters) <= original and set(meters).isdisjoint(excluded) and set(meters) | excluded == original,
           "TEST membership is not the retained original holdout")
    priority = {name: i for i, name in enumerate(("train", "val", "test"))}
    canonicals = {name: {} for name in priority}
    for group in curation["identity_groups"]:
        winner = min(group["members"], key=lambda member: (priority[member["split"]], member["meter_id"]))
        _check(group["canonical"] == winner, "identity representative violates frozen split priority")
        _check(winner["meter_id"] not in canonicals[winner["split"]], "duplicate identity representative")
        canonicals[winner["split"]][winner["meter_id"]] = winner
    for name in priority:
        part = metadata["splits"][name]
        _check(sorted(canonicals[name]) == part["meter_ids"], "split membership differs from identity representatives")
        _check(curation["after"][name] == {key: value for key, value in part.items() if key != "sha256"},
               "curation retained split metadata disagrees")
    development = sorted(set(metadata["splits"]["train"]["meter_ids"]) | set(metadata["splits"]["val"]["meter_ids"]))
    _check(set(meters).isdisjoint(development), "TEST IDs overlap development")
    dates = curation["expected_dates"]
    _check(isinstance(dates, list) and dates == sorted(set(dates)) and len(dates) >= 2, "invalid declared calendar")
    days = np.asarray(dates, dtype="datetime64[D]")
    _check(not np.isnat(days).any() and np.datetime_as_string(days, unit="D").tolist() == dates, "invalid calendar dates")
    counts = split["condition_counts"]
    _check(set(counts) == {str(x) for x in SUPPORT} and
           all(type(n) is int and n > 0 for n in counts.values()), "TEST metadata must support all 64 conditions")
    _check(split["meters"] == len(meters) and split["profiles"] == len(meters) * len(dates) == sum(counts.values()),
           "TEST counts/calendar disagree")
    technologies = {meter: member["technology"] for meter, member in canonicals["test"].items()}
    _check(all(type(value) is int and 0 <= value < 8 for value in technologies.values()) and
           all(member["profiles"] == len(dates) for member in canonicals["test"].values()), "invalid canonical TEST meter details")
    technology_counts = {str(t): sum(value == t for value in technologies.values()) for t in range(8)}
    _check(technology_counts == split["technology_meter_counts"], "TEST technology counts disagree with identity representatives")
    months = days.astype("datetime64[M]").astype(np.int64) % 12 + 1
    calendar_labels = (((days.astype(np.int64) + 3) % 7 >= 5).astype(np.int64) * 4 + (months % 12) // 3)
    expected_counts = {str(c): technology_counts[str(c // 8)] * int(np.sum(calendar_labels == c % 8)) for c in SUPPORT}
    _check(counts == expected_counts, "TEST condition counts disagree with canonical meters/calendar")
    return {"path": str(Path(prepared) / "test.npz"), "sha256": expected, "meter_ids": meters,
            "excluded_original_test_meter_ids": sorted(excluded), "development_meter_ids": development,
            "expected_dates": dates, "profiles": split["profiles"], "condition_counts": counts,
            "technology_meter_counts": technology_counts, "technology_by_meter": technologies, "resolution": metadata["resolution"],
            "status_before_execution": "identity_audited_performance_unseen_by_this_harness"}


def _verify_checkpoint(path, train, metadata, scaler, hashes, manifest_hash):
    before = sha256_file(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    _check(sha256_file(path) == before, "checkpoint changed while loading")
    _check(state.get("format_version") == training.CHECKPOINT_FORMAT, "unsupported checkpoint")
    _check(state["metadata"] == metadata and state["output_dim"] == metadata["resolution"], "checkpoint metadata differs")
    _check(all(state["data_hashes"].get(key) == value for key, value in hashes.items()) and
           state["data_hashes"].get("curation_manifest.json") == manifest_hash, "checkpoint data/curation binding differs")
    _check(all(state["scaler"].get(key) == value for key, value in scaler.items()), "checkpoint scaler differs")
    config = training._config({**state["config"], "device": "cpu"})
    _check(config["seed"] in SEEDS and not config["unconditional"] and state["supported_labels"] == SUPPORT,
           "five supported conditional training seeds are required")
    selection = state["selection"]
    indices = np.arange(len(train["x"]), dtype=np.int64)
    used = min(len(indices), config["max_train_samples"] or len(indices))
    if used < len(indices):
        indices = np.sort(np.random.default_rng(config["seed"]).choice(len(indices), used, replace=False))
    _check(selection["train_profiles_available"] == len(train["x"]) and selection["train_profiles_used"] == used and
           selection["train_indices_sha256"] == _digest(indices, "<i8"), "checkpoint training subset cannot be verified")
    _check(np.unique(train["y"][indices]).tolist() == SUPPORT, "actual TRAIN subset lacks support")
    directory = Path(path).parent
    summary = _json(directory / "summary.json")
    history = json.loads((directory / "history.json").read_bytes())
    _check(summary["test_used"] is False and summary["epochs_completed"] == config["epochs"] and
           summary["stopped_early"] is False, "every run must complete its frozen training budget")
    _check(isinstance(history, list) and len(history) == config["epochs"] and
           [r["epoch"] for r in history] == list(range(1, config["epochs"] + 1)) and
           all(isinstance(r["validation_score"], (int, float)) and math.isfinite(r["validation_score"]) for r in history),
           "invalid completed training history")
    best = min(history, key=lambda row: row["validation_score"])
    _check(state["epoch"] == summary["best_epoch"] == best["epoch"] and
           state["best_validation_score"] == summary["best_validation_score"] == best["validation_score"],
           "checkpoint is not the recorded validation-selected best epoch")
    _check(summary["data_hashes"] == state["data_hashes"] and summary["selection"] == selection,
           "training summary binding differs")
    _check(summary.get("provenance") == state.get("provenance") and
           summary.get("source_sha256") == metadata["source_sha256"], "training summary provenance differs")
    sources = state.get("provenance", {}).get("source_files_sha256", {})
    _check(all(_is_hash(sources.get(f"src/fluvius_cgan/{name}")) for name in TRAINING_CORE[:-1]),
           "training core source provenance missing")
    _check(sources.get("src/fluvius_cgan/temporal_loss.py") is None or
           _is_hash(sources["src/fluvius_cgan/temporal_loss.py"]), "invalid temporal training source fingerprint")
    _check(_json(directory / "config.json") == state["config"], "saved training configuration differs")
    return state, indices, before


def _evidence(output, paths, assets, prefix):
    records = []
    for index, item in enumerate(paths):
        path = Path(item).resolve()
        _check(path.suffix.lower() in (".json", ".md", ".txt"), "evidence must be a JSON, Markdown or text document")
        before = sha256_file(path)
        name = f"{prefix}_{index}{path.suffix.lower()}"
        shutil.copyfile(path, _safe_asset(output, name))
        _record_asset(output, name, assets)
        _check(sha256_file(path) == before == assets[name]["sha256"], "decision evidence changed while copying")
        records.append({"path_at_freeze": str(path), "artifact": name, "sha256": before})
    return records


def _forbid_test_inputs(prepared, paths):
    """Reject mistaken paths/aliases using filesystem identity without reading TEST."""
    reserved = prepared / "test.npz"
    for value in paths:
        path = Path(value).resolve()
        _check(path != reserved.resolve(), "freeze input points to reserved TEST")
        if reserved.exists() and path.exists():
            _check(not os.path.samefile(path, reserved), "freeze input aliases reserved TEST")


def freeze_test(prepared, runs, output_dir, *, decision_rationale, protocol_files,
                evidence_files=(), utility=True, bootstrap_replicates=500):
    """Build and seal a TRAIN-only bundle; never open/hash TEST, even on failure.

    ``runs`` is five (checkpoint, saved_calibrator) pairs. Returns the manifest
    path and its exact-byte SHA256. Output directories are exclusive; failed
    attempts remain on disk and must not be resumed or overwritten.
    """
    _check(type(utility) is bool, "utility must be boolean")
    _check(isinstance(decision_rationale, str) and bool(decision_rationale.strip()), "explicit freeze decision rationale required")
    protocol_files = list(protocol_files)
    _check(bool(protocol_files), "at least one protocol document is required")
    _check(type(bootstrap_replicates) is int and (bootstrap_replicates == 0 or bootstrap_replicates >= 2),
           "bootstrap_replicates must be zero or an integer >=2")
    runs = list(runs)
    _check(len(runs) == 5 and all(len(pair) == 2 for pair in runs), "exactly five checkpoint/calibrator pairs required")
    prepared, output = Path(prepared).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "freeze_attempt.json", {"started_at": _now(), "test_accessed": False})
    try:
        with _one_thread():
            result = _freeze(prepared, runs, output, utility, bootstrap_replicates,
                             decision_rationale, protocol_files, list(evidence_files))
        _write(output / "freeze_complete.json", {**result, "completed_at": _now(), "test_accessed": False})
        return result
    except BaseException as error:
        _write(output / "freeze_failure.json", {"failed_at": _now(), "test_accessed": False,
                                                "error_type": type(error).__name__, "error": str(error)})
        raise


def _freeze(prepared, pairs, output, utility, bootstrap_replicates, rationale, protocol_files, evidence_files):
    _forbid_test_inputs(prepared, [prepared / name for name in (*PREPARED, "curation_manifest.json")] +
                        [path for pair in pairs for path in pair] + protocol_files + evidence_files)
    sources, environment = _sources(), _environment()
    guard = training._training_curation_guard(prepared, {"allow_uncurated_data": False})
    _check(guard["status"] == "verified_curation_v2", "final study requires verified identity-curated v2 data")
    train, val, metadata, scaler, hashes = training._load_prepared(prepared)
    _check(metadata["target"] == "net" and metadata["units"] == "kWh/interval", "signed net kWh profiles required")
    _check(all(metadata["splits"][name]["sha256"] == hashes[f"{name}.npz"] for name in ("train", "val")),
           "TRAIN/VAL metadata hashes differ")
    _check(all(np.unique(part["meter_id"]).tolist() == metadata["splits"][name]["meter_ids"]
               for name, part in (("train", train), ("val", val))), "actual TRAIN/VAL meter membership differs from metadata")
    curation_manifest = _json(prepared / "curation_manifest.json")
    binding = _test_binding(prepared, metadata, curation_manifest)
    del val
    (output / "artifacts").mkdir()
    (output / "attempts").mkdir()
    assets = {}
    decision = {"rationale": rationale, "protocol_references": _evidence(output, protocol_files, assets, "protocol"),
                "additional_evidence": _evidence(output, evidence_files, assets, "evidence"),
                "scope": "Development evidence is retained to document selection, not reinterpreted as TEST evidence."}
    for name in ("metadata.json", "scaler.json", "curation_manifest.json"):
        shutil.copyfile(prepared / name, _safe_asset(output, name))
        _record_asset(output, name, assets)
    labels = np.repeat(np.asarray(SUPPORT, dtype=np.int64),
                       [min(PROTOCOL["samples_per_condition_cap"], binding["condition_counts"][str(c)]) for c in SUPPORT])
    labels_asset = _save_array(output, "labels.npy", labels, assets)
    verified, configurations, subsets, calibration_settings, core_sources = [], [], [], [], []
    seen = set()
    for checkpoint, calibrator in pairs:
        checkpoint, calibrator = Path(checkpoint).resolve(), Path(calibrator).resolve()
        state, indices, checkpoint_hash = _verify_checkpoint(checkpoint, train, metadata, scaler, hashes, guard["manifest_sha256"])
        seed = state["config"]["seed"]
        _check(seed not in seen, "training seeds must be exactly 11,22,33,44,55 without duplicates")
        seen.add(seed)
        model, map_details = load_verified_calibrator(calibrator, checkpoint_sha256=checkpoint_hash,
            train_indices_sha256=state["selection"]["train_indices_sha256"], prepared_data_hashes=hashes,
            generation_seed=PROTOCOL["generation_seed"], n_features=metadata["resolution"], train_labels=train["y"][indices])
        _check(model.quantile_knots == PROTOCOL["quantile_knots"] and
               map_details["calibration_seed"] == PROTOCOL["calibration_seed"] and
               model.provenance["calibration_samples_per_condition"] == PROTOCOL["calibration_samples_per_condition"],
               "calibrator count/knots/seed violate the frozen protocol")
        core_sources.append({name: state["provenance"]["source_files_sha256"].get(f"src/fluvius_cgan/{name}") for name in TRAINING_CORE})
        configurations.append({k: v for k, v in state["config"].items() if k != "seed"})
        subsets.append(state["selection"]["train_indices_sha256"])
        calibration_settings.append({k: model.provenance[k] for k in ("calibration_seed", "calibration_samples_per_condition", "quantile_knots")})
        for source, name in ((checkpoint, f"seed_{seed}.pt"), (calibrator, f"seed_{seed}_calibration.npz")):
            shutil.copyfile(source, _safe_asset(output, name))
            _record_asset(output, name, assets)
        _check(assets[f"seed_{seed}.pt"]["sha256"] == checkpoint_hash and
               assets[f"seed_{seed}_calibration.npz"]["sha256"] == map_details["sha256"], "model artifacts changed while copying")
        evidence_paths = sorted({checkpoint.parent / name for name in ("config.json", "summary.json", "history.json")} |
            {p for p in checkpoint.parent.glob("*.json")
             if any(tag in p.stem for tag in ("validation", "uncertainty", "robustness", "comparison", "sensitivity"))})
        development_evidence = _evidence(output, evidence_paths, assets, f"seed_{seed}_evidence")
        verified.append({"seed": seed, "checkpoint": f"seed_{seed}.pt", "calibrator": f"seed_{seed}_calibration.npz",
                         "checkpoint_sha256": checkpoint_hash, "calibrator_sha256": map_details["sha256"],
                         "epoch": state["epoch"], "config": state["config"], "training_selection": state["selection"],
                         "training_provenance": state.get("provenance"), "calibration_provenance": model.provenance,
                         "development_evidence": development_evidence})
    _check(seen == set(SEEDS), "exact five training seeds required")
    _check(all(c == configurations[0] for c in configurations) and len(set(subsets)) == 1 and
           all(c == calibration_settings[0] for c in calibration_settings), "models must share configuration, actual TRAIN subset and calibration settings")
    _check(all(sources == core_sources[0] for sources in core_sources), "training core source differs across seeds")
    verified.sort(key=lambda row: row["seed"])
    tx, ty = train["x"][indices], train["y"][indices]
    del train
    shared_arrays, shared_utility = {}, {}
    if utility:
        shared_utility["training_real"] = _save_ridge(output, "ridge_real.npz", tx, ty, assets)
    for method in BASELINES:
        if method == "conditional_resampling":
            values = resample_baseline(tx, ty, labels, seed=PROTOCOL["generation_seed"])
            baseline = None
        else:
            baseline = fit_conditional_copula(tx, ty,
                quantile_knots=None if method == "exact_marginal_copula" else PROTOCOL["quantile_knots"],
                shrinkage=1. if method == "independent_marginals" else PROTOCOL["copula_shrinkage"])
            values = baseline.generate(labels, seed=PROTOCOL["generation_seed"])
        shared_arrays[method] = _save_array(output, f"{method}.npy", np.asarray(values, dtype=np.float64), assets)
        del values
        if utility and method in ("gaussian_copula", "conditional_resampling"):
            values = (resample_baseline(tx, ty, ty, seed=PROTOCOL["utility_seed"]) if baseline is None
                      else baseline.generate(ty, seed=PROTOCOL["utility_seed"]))
            utility_name = "training_copula" if baseline is not None else "training_resampling"
            shared_utility[utility_name] = _save_ridge(output, f"ridge_{method}.npz", values, ty, assets)
            del values
        del baseline
    for run in verified:
        checkpoint = _safe_asset(output, run["checkpoint"])
        model, _ = load_verified_calibrator(_safe_asset(output, run["calibrator"]),
            checkpoint_sha256=run["checkpoint_sha256"], train_indices_sha256=subsets[0], prepared_data_hashes=hashes,
            generation_seed=PROTOCOL["generation_seed"], n_features=metadata["resolution"], train_labels=ty)
        raw = training.generate(checkpoint, labels, seed=PROTOCOL["generation_seed"], batch_size=1024)
        mapped = model.transform(raw, labels)
        diagnostics = {"fitting": model.fitting_diagnostics(), "application": model.diagnostics(raw, labels),
                       "scope": "Saved fixed-map diagnostics on TRAIN calibration references and fresh sealed generated arrays; no TEST values used."}
        diagnostic_name = f"calibration_diagnostics_{run['seed']}.json"
        _write(_safe_asset(output, diagnostic_name), diagnostics)
        run["calibration_diagnostics"] = _record_asset(output, diagnostic_name, assets)
        run["arrays"] = {"raw_cgan": _save_array(output, f"raw_{run['seed']}.npy", np.asarray(raw, dtype=np.float64), assets),
                         "calibrated_cgan": _save_array(output, f"calibrated_{run['seed']}.npy", mapped, assets)}
        del raw, mapped
        run["utility"] = {}
        if utility:
            raw = training.generate(checkpoint, ty, seed=PROTOCOL["utility_seed"], batch_size=1024)
            run["utility"]["training_cgan"] = _save_ridge(output, f"ridge_raw_{run['seed']}.npz", raw, ty, assets)
            mapped = model.transform(raw, ty)
            run["utility"]["training_calibrated_cgan"] = _save_ridge(output, f"ridge_calibrated_{run['seed']}.npz", mapped, ty, assets)
            del raw, mapped
    _check(_sources() == sources and _environment() == environment, "source/environment changed during freeze")
    _check(all(sha256_file(prepared / name) == value for name, value in hashes.items()) and
           sha256_file(prepared / "curation_manifest.json") == guard["manifest_sha256"], "development inputs changed during freeze")
    _verify_assets(output, assets)
    manifest = {"schema_version": 1, "kind": "frozen_five_seed_test_bundle", "created_at": _now(),
                "test_accessed_during_freeze": False, "protocol": {**PROTOCOL, "utility": utility,
                    "bootstrap_replicates": bootstrap_replicates, "bootstrap_confidence": .95},
                "sources": sources, "environment": environment, "decision": decision, "training_data_hashes": hashes,
                "curation_manifest_sha256": guard["manifest_sha256"], "test_binding": binding,
                "labels_artifact": labels_asset, "shared_arrays": shared_arrays, "shared_utility": shared_utility,
                "runs": verified, "assets": assets, "limitations": LIMITATIONS,
                "scope": "One selected candidate, all five training seeds, train-only statistical controls; no ensemble."}
    _write(output / "manifest.json", manifest)
    seal = sha256_file(output / "manifest.json")
    with (output / "seal.sha256").open("x", encoding="ascii") as handle:
        handle.write(seal + "\n")
    return {"manifest": str(output / "manifest.json"), "seal_sha256": seal}


def _verify_assets(bundle, assets):
    _check(isinstance(assets, dict) and bool(assets), "sealed artifacts are missing")
    for filename, record in assets.items():
        path = _safe_asset(bundle, filename)
        _check(_is_hash(record.get("sha256")) and type(record.get("bytes")) is int and
               path.is_file() and path.stat().st_size == record["bytes"] and sha256_file(path) == record["sha256"],
               f"sealed artifact changed: {filename}")


def _validate_manifest(bundle, manifest):
    _check(manifest.get("schema_version") == 1 and manifest.get("kind") == "frozen_five_seed_test_bundle" and
           manifest.get("test_accessed_during_freeze") is False, "unsupported or unsealed test bundle")
    protocol = manifest["protocol"]
    _check(all(protocol.get(k) == v for k, v in PROTOCOL.items()), "frozen test protocol differs")
    _check(type(protocol.get("utility")) is bool and type(protocol.get("bootstrap_replicates")) is int and
           (protocol["bootstrap_replicates"] == 0 or protocol["bootstrap_replicates"] >= 2) and
           protocol["bootstrap_confidence"] == .95, "invalid frozen optional settings")
    _check([r["seed"] for r in manifest["runs"]] == list(SEEDS), "sealed run list must contain all five seeds")
    _check(set(manifest["shared_arrays"]) == set(BASELINES), "statistical controls missing")
    _check(manifest["sources"] == _sources(), "evaluator source differs from seal")
    _check(manifest["environment"] == _environment(), "evaluator environment differs from seal")
    _verify_assets(bundle, manifest["assets"])
    metadata = _json(_safe_asset(bundle, "metadata.json"))
    curation = _json(_safe_asset(bundle, "curation_manifest.json"))
    binding = _test_binding(Path(manifest["test_binding"]["path"]).parent, metadata, curation)
    _check(binding == manifest["test_binding"], "sealed TEST binding disagrees with curated metadata")
    _check(manifest["assets"]["metadata.json"]["sha256"] == manifest["training_data_hashes"]["metadata.json"] and
           manifest["assets"]["curation_manifest.json"]["sha256"] == manifest["curation_manifest_sha256"], "metadata artifact binding differs")
    expected_assets = {manifest["labels_artifact"], "metadata.json", "curation_manifest.json", "scaler.json", *manifest["shared_arrays"].values()}
    decision = manifest["decision"]
    _check(isinstance(decision["rationale"], str) and bool(decision["rationale"].strip()) and
           bool(decision["protocol_references"]), "sealed decision rationale/protocol missing")
    evidence = [*decision["protocol_references"], *decision["additional_evidence"]]
    for run in manifest["runs"]:
        _check(set(run["arrays"]) == {"raw_cgan", "calibrated_cgan"}, "candidate arrays missing")
        expected_assets.update((run["checkpoint"], run["calibrator"], run["calibration_diagnostics"], *run["arrays"].values()))
        _check(manifest["assets"][run["checkpoint"]]["sha256"] == run["checkpoint_sha256"] and
               manifest["assets"][run["calibrator"]]["sha256"] == run["calibrator_sha256"], "candidate artifact bindings differ")
        expected = {"training_cgan", "training_calibrated_cgan"} if protocol["utility"] else set()
        _check(set(run["utility"]) == expected, "candidate utility coverage differs")
        expected_assets.update(x["artifact"] for x in run["utility"].values())
        evidence.extend(run["development_evidence"])
    expected = {"training_real", "training_resampling", "training_copula"} if protocol["utility"] else set()
    _check(set(manifest["shared_utility"]) == expected, "shared utility coverage differs")
    expected_assets.update(x["artifact"] for x in manifest["shared_utility"].values())
    for item in evidence:
        _check(manifest["assets"][item["artifact"]]["sha256"] == item["sha256"], "decision evidence binding differs")
        expected_assets.add(item["artifact"])
    _check(expected_assets == set(manifest["assets"]), "artifact inventory differs from declared methods")
    expected_labels = np.repeat(np.asarray(SUPPORT, dtype=np.int64),
        [min(protocol["samples_per_condition_cap"], binding["condition_counts"][str(c)]) for c in SUPPORT])
    labels = np.load(_safe_asset(bundle, manifest["labels_artifact"]), allow_pickle=False)
    _check(labels.dtype == np.int64 and np.array_equal(labels, expected_labels), "sealed generation labels disagree with metadata")
    shape = (len(labels), binding["resolution"])
    for name in list(manifest["shared_arrays"].values()) + [p for run in manifest["runs"] for p in run["arrays"].values()]:
        _array(bundle, name, shape)
    for item in list(manifest["shared_utility"].values()) + [item for run in manifest["runs"] for item in run["utility"].values()]:
        _check(_load_ridge(bundle, item).resolution_ == binding["resolution"], "frozen utility resolution differs")


def _read_verified_test(binding):
    path = Path(binding["path"])
    _check(not path.is_symlink(), "TEST path cannot be a symlink")
    _check(sha256_file(path) == binding["sha256"], "full TEST hash differs from seal")
    with np.load(path, allow_pickle=False) as archive:
        _check(set(archive.files) == {"x", "y", "meter_id", "date"}, "invalid TEST arrays")
        data = {key: archive[key] for key in archive.files}
    _check(sha256_file(path) == binding["sha256"], "TEST changed while loading")
    x, y, meters, dates = (data[key] for key in ("x", "y", "meter_id", "date"))
    n, width = binding["profiles"], binding["resolution"]
    _check(x.dtype == np.float32 and x.shape == (n, width) and np.isfinite(x).all(), "invalid TEST profiles/dtype/width")
    _check(y.dtype == np.int64 and y.shape == (n,) and np.unique(y).tolist() == SUPPORT, "invalid TEST condition support/dtype")
    _check(all(v.shape == (n,) and v.dtype.kind == "U" and not np.any(v == "") for v in (meters, dates)),
           "invalid TEST meter/date vectors")
    _check(np.unique(meters).tolist() == binding["meter_ids"] and
           not set(meters).intersection(binding["development_meter_ids"] + binding["excluded_original_test_meter_ids"]),
           "TEST meter membership differs")
    try:
        days = dates.astype("datetime64[D]")
    except ValueError as error:
        raise ValueError("invalid TEST dates") from error
    _check(not np.isnat(days).any() and np.array_equal(np.datetime_as_string(days, unit="D"), dates), "invalid TEST calendar format")
    months = days.astype("datetime64[M]").astype(np.int64) % 12 + 1
    weekends = ((days.astype(np.int64) + 3) % 7 >= 5).astype(np.int64)
    _check(np.array_equal(y % 8, weekends * 4 + (months % 12) // 3), "TEST labels disagree with calendar")
    tech_counts = {}
    for meter in binding["meter_ids"]:
        mask = meters == meter
        _check(sorted(dates[mask].tolist()) == binding["expected_dates"], "TEST meter has missing/duplicate/unexpected dates")
        technology = np.unique(y[mask] // 8)
        _check(len(technology) == 1, "TEST technology changes within a meter")
        _check(int(technology[0]) == binding["technology_by_meter"][meter], "TEST technology differs from canonical meter")
        key = str(int(technology[0])); tech_counts[key] = tech_counts.get(key, 0) + 1
    _check(tech_counts == binding["technology_meter_counts"] and
           {str(c): int(np.sum(y == c)) for c in SUPPORT} == binding["condition_counts"], "TEST condition/technology counts differ")
    return data


def _array(bundle, filename, shape=None):
    values = np.load(_safe_asset(bundle, filename), allow_pickle=False)
    _check(values.dtype == np.float64 and (shape is None or values.shape == shape) and np.isfinite(values).all(),
           "invalid sealed generated profiles")
    return values


def _battery_compare(real_result, generated, labels, hours):
    other = _battery(generated, hours)
    _check(other["scenario"] == real_result["scenario"], "battery scenarios differ")
    quantities = ("post_grid_import_kwh", "post_grid_export_kwh", "grid_import_reduction_kwh",
                  "post_peak_import_power_kw", "end_soc_kwh")
    rows = {}
    for c in SUPPORT:
        mask = labels == c
        row = {"n_real": int(mask.sum()), "n_generated": int(mask.sum())}
        for quantity in quantities:
            a, b = real_result["per_profile"][quantity][mask], other["per_profile"][quantity][mask]
            stem, unit = quantity.rsplit("_", 1)
            row[f"{stem}_wasserstein_{unit}"] = float(wasserstein_distance(a, b))
            row[f"real_mean_{quantity}"] = float(a.mean())
            row[f"generated_mean_{quantity}"] = float(b.mean())
        rows[str(c)] = row
    keys = [k for k in rows["0"] if k not in ("n_real", "n_generated")]
    return {"scenario": other["scenario"], "condition_weighting": "equal_all_64", "per_condition": rows,
            "macro": {key: statistics.mean(row[key] for row in rows.values()) for key in keys}}


def _battery(values, hours):
    return simulate_battery(values, hours, **{key: value for key, value in PROTOCOL["battery"].items()
                                             if key != "initial_soc_kwh"})


def _stats(values):
    if any(value is None for value in values):
        return {"n": len(values), "n_defined": sum(v is not None for v in values), "mean": None, "sample_sd": None}
    return {"n": len(values), "n_defined": len(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "minimum": min(values), "maximum": max(values)}


def _compact(row):
    return {**{key: row[key] for key in METRICS if key != "lag_one_acf_wasserstein"},
            "lag_one_acf_wasserstein": row["autocorrelation"]["1"]["wasserstein"], "diversity_ratio": row["diversity_ratio"]}


def evaluate_frozen_test(bundle, expected_seal_sha256, output_dir):
    """Evaluate a sealed bundle with no fitting/generation or runtime overrides.

    Exclusive output and append-only bundle attempt events preserve failures.
    Per-seed bootstrap intervals are never pooled into a five-seed interval.
    """
    _check(_is_hash(expected_seal_sha256), "expected seal SHA256 is mandatory")
    bundle, output = Path(bundle).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    attempt_id = uuid.uuid4().hex
    started = {"attempt_id": attempt_id, "started_at": _now(), "bundle": str(bundle),
               "output": str(output), "expected_seal_sha256": expected_seal_sha256, "test_access_started": False}
    _write(output / "attempt.json", started)
    ledger = bundle / "attempts"
    ledger_started = False
    opened = False
    try:
        _check(not (bundle / "manifest.json").is_symlink() and sha256_file(bundle / "manifest.json") == expected_seal_sha256,
               "manifest seal SHA256 mismatch")
        manifest = _json(bundle / "manifest.json")
        _check(ledger.is_dir() and not ledger.is_symlink(), "bundle attempt ledger missing")
        _write(ledger / f"{attempt_id}.start.json", started); ledger_started = True
        with _one_thread():
            _validate_manifest(bundle, manifest)
            _write(output / "test_access_started.json", {"at": _now(), "test_sha256": manifest["test_binding"]["sha256"]})
            _write(ledger / f"{attempt_id}.test_access.json", {"at": _now()})
            opened = True
            report = _evaluate(bundle, manifest)
            _check(_sources() == manifest["sources"] and _environment() == manifest["environment"], "source/environment changed during evaluation")
            _verify_assets(bundle, manifest["assets"])
            _check(sha256_file(bundle / "manifest.json") == expected_seal_sha256, "manifest changed during evaluation")
        report.update({"seal_sha256": expected_seal_sha256, "attempt_id": attempt_id})
        _write(output / "report.json", report)
        finished = {"attempt_id": attempt_id, "completed_at": _now(), "test_access_started": True,
                    "report_sha256": sha256_file(output / "report.json")}
        _write(output / "complete.json", finished)
        _write(ledger / f"{attempt_id}.complete.json", finished)
        return report
    except BaseException as error:
        failed = {"attempt_id": attempt_id, "failed_at": _now(), "test_access_started": opened,
                  "error_type": type(error).__name__, "error": str(error)}
        _write(output / "failure.json", failed)
        if ledger_started:
            _write(ledger / f"{attempt_id}.failure.json", failed)
        raise


def _evaluate(bundle, manifest):
    data = _read_verified_test(manifest["test_binding"])
    protocol = manifest["protocol"]
    rng = np.random.default_rng(protocol["selection_seed"])
    selected = np.concatenate([rng.choice(np.flatnonzero(data["y"] == c),
        min(protocol["samples_per_condition_cap"], int(np.sum(data["y"] == c))), replace=False) for c in SUPPORT])
    real, labels, meters = (data[key][selected] for key in ("x", "y", "meter_id"))
    width = real.shape[1]; hours = 24 / width
    stored_labels = np.load(_safe_asset(bundle, manifest["labels_artifact"]), allow_pickle=False)
    _check(stored_labels.dtype == np.int64 and np.array_equal(stored_labels, labels), "sealed generation labels differ from TEST selection")
    del data
    real_battery = _battery(real, hours)
    shared = {name: _array(bundle, path, real.shape) for name, path in manifest["shared_arrays"].items()}
    shared_utility = {name: _load_ridge(bundle, item) for name, item in manifest["shared_utility"].items()}
    shared_utility_reports = {name: model.evaluate(real, labels) for name, model in shared_utility.items()}
    persistence = None
    for report in shared_utility_reports.values():
        value = report.pop("persistence")
        _check(persistence is None or persistence == value, "utility windows differ")
        persistence = value
    def score(values):
        joint = conditional_projected_wasserstein(real, labels, values, labels)
        _check(joint["directions"]["count"] == protocol["projection_count"] and
               joint["directions"]["seed"] == protocol["projection_seed"], "joint directions differ from protocol")
        return {"physical": evaluate_conditional(real, labels, values, labels, interval_hours=hours),
                "joint": joint,
                "battery": _battery_compare(real_battery, values, labels, hours)}
    shared_reports = {name: score(values) for name, values in shared.items()}
    runs = []
    for frozen in manifest["runs"]:
        arrays = {name: _array(bundle, path, real.shape) for name, path in frozen["arrays"].items()}
        methods = {name: score(values) for name, values in arrays.items()}
        utility_models = {name: _load_ridge(bundle, item) for name, item in frozen["utility"].items()}
        utility = {name: model.evaluate(real, labels) for name, model in utility_models.items()}
        for item in utility.values():
            _check(item.pop("persistence") == persistence, "utility evaluation windows differ")
        comparisons, utility_comparisons = {}, {}
        if protocol["bootstrap_replicates"]:
            for name in ("gaussian_copula", "conditional_resampling"):
                comparisons[name] = paired_condition_macro_meter_bootstrap(
                    real, labels, arrays["calibrated_cgan"], labels, shared[name], labels, meters,
                    interval_hours=hours, n_bootstrap=protocol["bootstrap_replicates"],
                    confidence=protocol["bootstrap_confidence"], seed=protocol["bootstrap_seed"],
                    real_meter_strata=labels // 8, metrics=METRICS, return_per_condition=True)
            if protocol["utility"]:
                candidate = utility_models["training_calibrated_cgan"]
                errors = candidate.predict(real, labels) - real[:, candidate.lags_:]
                for name in ("training_real", "training_resampling"):
                    comparator = shared_utility[name]
                    utility_comparisons[name] = paired_utility_meter_bootstrap(errors,
                        comparator.predict(real, labels) - real[:, comparator.lags_:], labels, meters,
                        n_bootstrap=protocol["bootstrap_replicates"], confidence=protocol["bootstrap_confidence"],
                        seed=protocol["bootstrap_seed"])
        runs.append({"training_seed": frozen["seed"], "checkpoint_sha256": frozen["checkpoint_sha256"],
                     "calibrator_sha256": frozen["calibrator_sha256"], "methods": methods,
                     "utility": utility or None, "paired_uncertainty": comparisons or None,
                     "paired_utility_uncertainty": utility_comparisons or None,
                     "calibration_diagnostics": {"artifact": frozen["calibration_diagnostics"],
                         "sha256": manifest["assets"][frozen["calibration_diagnostics"]]["sha256"],
                         "data": _json(_safe_asset(bundle, frozen["calibration_diagnostics"]))},
                     "generated_profiles_sha256": {name: _digest(values, "<f8") for name, values in arrays.items()}})
    aggregate = {}
    for method in ("raw_cgan", "calibrated_cgan"):
        aggregate[method] = {
            "physical_macro": {key: _stats([_compact(run["methods"][method]["physical"]["macro"])[key] for run in runs])
                               for key in (*METRICS, "diversity_ratio")},
            "joint_macro": _stats([run["methods"][method]["joint"]["macro"] for run in runs]),
            "battery_macro": {key: _stats([run["methods"][method]["battery"]["macro"][key] for run in runs])
                              for key in runs[0]["methods"][method]["battery"]["macro"]},
            "utility_macro": {key: _stats([run["utility"]["training_cgan" if method == "raw_cgan" else "training_calibrated_cgan"]["macro"][key]
                                           for run in runs]) for key in ("rmse_kwh", "mae_kwh")} if protocol["utility"] else None}
    calibrated = aggregate["calibrated_cgan"]["physical_macro"]
    baseline_ramp = shared_reports["gaussian_copula"]["physical"]["macro"]["ramp_wasserstein_kw"]
    improvements = [run["methods"]["calibrated_cgan"]["physical"]["macro"]["ramp_wasserstein_kw"] < baseline_ramp for run in runs]
    relative = None if baseline_ramp == 0 else 1 - calibrated["ramp_wasserstein_kw"]["mean"] / baseline_ramp
    gates = {"relative_ramp_reduction": relative, "seeds_better_than_gaussian": sum(improvements),
             "relative_ramp_criterion_pass": relative is not None and relative >= .20 and sum(improvements) >= 4,
             "five_seed_mean_physical_pass": {key: calibrated[key]["mean"] <= limit for key, limit in protocol["physical_limits"].items()},
             "per_seed_physical_pass": {str(run["training_seed"]): {key: _compact(run["methods"]["calibrated_cgan"]["physical"]["macro"])[key] <= limit
                  for key, limit in protocol["physical_limits"].items()} for run in runs}}
    if protocol["utility"]:
        gates["per_seed_utility_tolerance_pass"] = {str(run["training_seed"]): all(
            run["utility"]["training_calibrated_cgan"]["macro"]["rmse_kwh"] <= 1.10 * shared_utility_reports[name]["macro"]["rmse_kwh"]
            for name in ("training_real", "training_resampling")) for run in runs}
    _check(sha256_file(manifest["test_binding"]["path"]) == manifest["test_binding"]["sha256"], "TEST changed during evaluation")
    return {"schema_version": 1, "split": "test", "test_used": True, "fitting_performed": False,
            "protocol": protocol, "sources": manifest["sources"], "environment": manifest["environment"],
            "decision": manifest["decision"],
            "training_data_hashes": manifest["training_data_hashes"], "test_binding": manifest["test_binding"],
            "test_selection": {"policy": "uniform_without_replacement_within_each_of_64_conditions",
                "selected_profiles": len(real), "selected_meters": len(np.unique(meters)),
                "condition_counts": {str(c): int(np.sum(labels == c)) for c in SUPPORT},
                "indices_sha256": _digest(selected, "<i8"), "labels_sha256": _digest(labels, "<i8"),
                "profiles_sha256": _digest(real, "<f8"),
                "meter_ids_sha256": hashlib.sha256(_canonical(meters.tolist())).hexdigest()},
            "shared_statistical_controls": shared_reports,
            "shared_utility": {**shared_utility_reports, **({"persistence": persistence} if persistence is not None else {})} or None,
            "runs": runs, "five_seed_summary": aggregate, "practical_criteria": gates,
            "aggregation": "All five seeds; arithmetic mean and sample SD. Shared controls appear once. No profile pooling, seed selection or averaged confidence intervals.",
            "uncertainty_scope": "Per-seed descriptive marginal technology-stratified meter intervals conditional on fixed models/maps, arrays and selected rows; no five-seed mean interval or formal significance claim.",
            "limitations": manifest["limitations"]}
