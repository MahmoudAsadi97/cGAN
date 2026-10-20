"""Evaluate fixed linear downstream utility on validation profiles only.

Example:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/evaluate_utility.py \
        --prepared data/prepared96 --checkpoint runs/candidate/best.pt \
        --output reports/new_utility.json --copula

Real, cGAN, conditional-resampling and optional Gaussian-copula training sets
have identical profile counts and condition labels. Baseline profile pools match
the checkpoint's verified gradient-training subset; GAN normalization was fitted
on the full prepared training split before that subset was selected. A profile cap
limits downstream fitting rows, not the source pool previously available to the
GAN and baseline generators. This command has no test-split option.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_checkpoints import _checkpoint, _digest, _generate_verified, _validate_manifest
from fluvius_cgan.calibration import load_verified_calibrator
from fluvius_cgan.copula import fit_conditional_copula
from fluvius_cgan.evaluation import resample_baseline
from fluvius_cgan.release import sha256_file
from fluvius_cgan.training import _load_prepared, _provenance
from fluvius_cgan.utility import RidgeProfileForecaster


def _profile_digest(values):
    """Match little-endian float64 C-order hashes without a full-size copy."""
    digest = hashlib.sha256()
    for start in range(0, len(values), 256):
        digest.update(np.asarray(values[start:start + 256], dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _counts(labels):
    conditions, counts = np.unique(labels, return_counts=True)
    return {str(label): int(count) for label, count in zip(conditions, counts)}


def evaluate(prepared, checkpoint, *, samples_per_condition=100, seed=2026,
             max_train_profiles=None, copula=False, calibrator=None):
    """Return a provenance-bound, validation-only observed-history TSTR report.

    If capped, real downstream training rows are a uniform sample without
    replacement from the actual GAN training subset, using seed+100. All other
    downstream training sets have the exact same labels and count. The copula
    and resampling source pool remains the full actual GAN training subset, so
    baseline fitting rows match GAN gradient-training rows. GAN normalization
    separately uses the full prepared train split. Both distinctions are reported.
    A saved calibrator adds a fixed-map method after strict checkpoint, prepared
    data, exact training-subset and independent calibration-seed verification.
    """
    for name, value, minimum in (("samples_per_condition", samples_per_condition, 1),
                                 ("seed", seed, 0), ("max_train_profiles", max_train_profiles, 1)):
        if name == "max_train_profiles" and value is None:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if seed >= 2 ** 32:
        raise ValueError("seed must be smaller than 2**32")
    if not isinstance(copula, (bool, np.bool_)):
        raise ValueError("copula must be a boolean")
    seed = int(seed)
    started = time.perf_counter()
    train, val, metadata, scaler, hashes = _load_prepared(prepared)
    _validate_manifest(metadata, hashes, train, val)
    checkpoint_details, actual_indices = _checkpoint(checkpoint, train, metadata, scaler, hashes)
    conditions = np.unique(val["y"])
    missing = sorted(set(conditions.tolist()) - set(checkpoint_details["supported_labels"]))
    if missing:
        raise ValueError(f"checkpoint training subset lacks observed validation conditions: {missing}")

    utility_indices = actual_indices
    capped = max_train_profiles is not None and max_train_profiles < len(actual_indices)
    if capped:
        utility_indices = np.sort(np.random.default_rng(seed + 100).choice(
            actual_indices, int(max_train_profiles), replace=False))
    # Avoid copying full prepared arrays when the exact subset is all rows.
    source_x = train["x"] if len(actual_indices) == len(train["x"]) else train["x"][actual_indices]
    source_y = train["y"] if len(actual_indices) == len(train["y"]) else train["y"][actual_indices]
    calibration_model = None
    calibration_details = {"enabled": calibrator is not None}
    if calibrator is not None:
        calibration_model, verified = load_verified_calibrator(
            calibrator, checkpoint_sha256=checkpoint_details["sha256"],
            train_indices_sha256=_digest(actual_indices, "<i8"),
            prepared_data_hashes=hashes, generation_seed=seed,
            n_features=metadata["resolution"], train_labels=source_y)
        calibration_details.update(verified)
        calibration_details["policy"] = (
            "Frozen per-condition maps fitted on the exact GAN training subset and a separate fixed generated reference; "
            "the current downstream training batch never fits or ranks the maps.")
    utility_x = train["x"][utility_indices] if capped else source_x
    utility_y = train["y"][utility_indices] if capped else source_y
    missing = np.setdiff1d(conditions, np.unique(utility_y))
    if len(missing):
        raise ValueError("downstream training cap omitted validation conditions; increase the cap: "
                         f"{missing.tolist()}")
    rng = np.random.default_rng(seed)
    selected_val = np.concatenate([
        rng.choice(np.flatnonzero(val["y"] == label),
                   min(int(samples_per_condition), int(np.sum(val["y"] == label))), replace=False)
        for label in conditions
    ])
    vx, vy, meter_ids = (val[key][selected_val] for key in ("x", "y", "meter_id"))
    loaded_at = time.perf_counter()
    methods, fits, timings = {}, {}, {}
    persistence = None
    fixed_feature_names = None
    training_label_digest = _digest(utility_y, "<i8")

    def fit_evaluate(name, profiles):
        nonlocal persistence, fixed_feature_names
        if np.asarray(profiles).shape != utility_x.shape:
            raise ValueError(f"{name} training profile shape does not match the real reference")
        fitted_started = time.perf_counter()
        model = RidgeProfileForecaster(batch_profiles=256).fit(profiles, utility_y)
        fitted_at = time.perf_counter()
        report = model.evaluate(vx, vy)
        evaluated_at = time.perf_counter()
        this_persistence = report.pop("persistence")
        if persistence is None:
            persistence = this_persistence
            fixed_feature_names = model.feature_names_
        elif persistence != this_persistence or fixed_feature_names != model.feature_names_:
            raise ValueError("utility methods do not share identical evaluation windows and features")
        methods[name] = report
        fits[name] = {
            **model.fit_summary_,
            "profiles_sha256": _profile_digest(profiles),
            "labels_sha256": training_label_digest,
            "coefficients_sha256": _digest(model.coef_, "<f8"),
        }
        timings[name] = {"ridge_fit": fitted_at - fitted_started,
                         "heldout_evaluation": evaluated_at - fitted_at}

    fit_evaluate("training_real", utility_x)
    generation_started = time.perf_counter()
    generated = _generate_verified(checkpoint, checkpoint_details, utility_y, seed)
    generated_at = time.perf_counter()
    fit_evaluate("training_cgan", generated)
    timings["training_cgan"]["generation"] = generated_at - generation_started
    if calibration_model is not None:
        transform_started = time.perf_counter()
        calibrated = calibration_model.transform(generated, utility_y)
        transformed_at = time.perf_counter()
        fit_evaluate("training_calibrated_cgan", calibrated)
        timings["training_calibrated_cgan"].update(
            generation_shared_with_raw_cgan=generated_at - generation_started,
            fixed_map_transform=transformed_at - transform_started)
        del calibrated
    del generated
    generation_started = time.perf_counter()
    resampled = resample_baseline(source_x, source_y, utility_y, seed=seed)
    generated_at = time.perf_counter()
    fit_evaluate("training_resampling", resampled)
    timings["training_resampling"]["generation"] = generated_at - generation_started
    del resampled
    copula_details = {"enabled": bool(copula)}
    if copula:
        fitting_started = time.perf_counter()
        copula_knots = calibration_model.quantile_knots if calibration_model is not None else None
        copula_model = fit_conditional_copula(source_x, source_y, quantile_knots=copula_knots, shrinkage=.01)
        fitted_at = time.perf_counter()
        generated = copula_model.generate(utility_y, seed=seed, batch_size=4096)
        generated_at = time.perf_counter()
        fit_evaluate("training_copula", generated)
        timings["training_copula"].update(copula_fit=fitted_at - fitting_started,
                                          generation=generated_at - fitted_at)
        copula_details.update(
            quantile_knots=copula_knots, shrinkage=.01,
            margins=("All sorted source-pool observations, with linear inverse quantiles."
                     if copula_knots is None else "Uniform linear empirical quantile knots matching the saved calibrator."),
            source_profiles=len(source_x), source_condition_counts=_counts(source_y),
            source_indices_sha256=_digest(actual_indices, "<i8"),
            limitations="Gaussian dependence; interpolated empirical margins do not extrapolate training tails.",
        )
        del generated, copula_model
    methods["persistence"] = persistence
    if sha256_file(checkpoint) != checkpoint_details["sha256"]:
        raise ValueError("checkpoint changed during utility evaluation")
    if calibrator is not None and sha256_file(calibrator) != calibration_details["sha256"]:
        raise ValueError("calibration archive changed during utility evaluation")

    counts = _counts(utility_y)
    matched_counts = all(fit["condition_counts"] == counts for fit in fits.values())
    matched_labels = all(fit["labels_sha256"] == training_label_digest for fit in fits.values())
    lags = metadata["resolution"] // 8
    provenance = _provenance(torch.device("cpu"))
    provenance["scipy"] = str(scipy.__version__)
    provenance["source_file_sha256"] = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in (
            Path(__file__), ROOT / "scripts" / "compare_checkpoints.py",
            ROOT / "src" / "fluvius_cgan" / "utility.py",
            ROOT / "src" / "fluvius_cgan" / "training.py",
            ROOT / "src" / "fluvius_cgan" / "models.py",
            ROOT / "src" / "fluvius_cgan" / "residual.py",
            ROOT / "src" / "fluvius_cgan" / "copula.py",
            ROOT / "src" / "fluvius_cgan" / "calibration.py",
            ROOT / "src" / "fluvius_cgan" / "evaluation.py",
        )
    }
    provenance["thread_environment"] = {
        name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    }
    return {
        "schema_version": 1,
        "split": "val",
        "test_used": False,
        "scope": "Exploratory validation-only downstream utility; no final test evaluation.",
        "source_sha256": metadata["source_sha256"],
        "prepared_data_hashes": hashes,
        "target": metadata["target"],
        "units": metadata["units"],
        "checkpoint": checkpoint_details,
        "gan_preprocessing": {
            "normalization": checkpoint_details["config"].get("normalization", "minmax"),
            "fitted_on": "full_prepared_train_before_training_subset_selection",
            "fitting_profiles": len(train["x"]),
            "uses_validation_values": False,
            "note": "Preprocessing statistics may use additional prepared training rows beyond GAN gradient-training rows.",
        },
        "config": {"samples_per_condition": int(samples_per_condition), "seed": seed,
                   "max_train_profiles": None if max_train_profiles is None else int(max_train_profiles),
                   "copula": bool(copula), "calibrator": None if calibrator is None else str(calibrator)},
        "seeds": {"validation_selection": seed, "utility_training_cap": seed + 100,
                  "cgan_generation": seed, "conditional_resampling": seed,
                  "copula_generation": seed if copula else None,
                  "calibration_reference": calibration_details.get("calibration_seed")},
        "task": {
            "name": "within_day_observed_history_next_interval_linear_ridge",
            "resolution": metadata["resolution"], "lags": lags,
            "forecast_horizon_intervals": 1, "forecast_horizon_hours": 24 / metadata["resolution"],
            "target_slots_inclusive": [lags, metadata["resolution"] - 1],
            "feature_names": fixed_feature_names,
            "ridge_alpha": 1., "ridge_objective": "sum_squared_error_plus_alpha_nonintercept_squared_norm",
            "feature_scaling": "none; native kWh/interval", "batch_profiles": 256,
            "macro_definition": "Arithmetic mean of per-condition RMSE/MAE; each condition has equal weight.",
        },
        "source_training_pool": {
            "prepared_profiles": len(train["x"]), "actual_gan_profiles": len(source_x),
            "actual_indices_sha256": _digest(actual_indices, "<i8"),
            "profiles_sha256": _profile_digest(source_x), "labels_sha256": _digest(source_y, "<i8"),
            "condition_counts": _counts(source_y),
            "policy": "Exact verified GAN gradient-training subset; also the resampling/copula profile-fitting source.",
        },
        "utility_training_selection": {
            "cap_applied": bool(capped), "selected_profiles": len(utility_x),
            "selected_indices_sha256": _digest(utility_indices, "<i8"),
            "selected_meters": len(np.unique(train["meter_id"][utility_indices])),
            "condition_counts": counts, "labels_sha256": training_label_digest,
            "policy": "All actual training rows, or uniform without replacement using seed+100 if capped.",
            "source_pool_equals_real_ridge_fit_rows": not capped,
            "cap_limitation": ("Real ridge sees capped rows; GAN and baseline generators saw the full actual source pool."
                               if capped else None),
        },
        "validation_selection": {
            "policy": "Uniform without replacement within every observed condition; matches cli.evaluate_checkpoint.",
            "samples_per_condition_cap": int(samples_per_condition),
            "available_profiles": len(val["x"]), "selected_profiles": len(vx),
            "selected_meters": len(np.unique(meter_ids)), "condition_counts": _counts(vy),
            "selected_indices_sha256": _digest(selected_val, "<i8"),
            "profiles_sha256": _profile_digest(vx), "labels_sha256": _digest(vy, "<i8"),
            "meter_ids_sha256": hashlib.sha256(json.dumps(meter_ids.tolist(), ensure_ascii=False,
                                                         separators=(",", ":")).encode("utf-8")).hexdigest(),
        },
        "hash_encoding": "Numeric profiles/coefficients: little-endian float64; labels/indices: little-endian int64; C-order bytes.",
        "invariants": {
            "training_condition_counts_match": matched_counts,
            "training_label_order_matches": matched_labels,
            "identical_validation_windows": True,
            "baseline_profile_pools_match_gan_gradient_training_subset": True,
            "fit_uses_validation_values": False,
            "prediction_uses_observed_real_past_lags": True,
            "prediction_uses_its_own_target_or_future_values": False,
            "cross_day_windows": False, "post_generation_clipping_applied": False,
            "calibration_fitted_on_current_generation_batch": False,
        },
        "training": fits, "methods": methods, "copula": copula_details,
        "calibration": calibration_details,
        "timings_seconds": {"load_and_verify": loaded_at - started, "methods": timings,
                            "total_before_report_serialization": time.perf_counter() - started},
        "provenance": provenance,
        "limitations": [
            "This fixed linear next-interval diagnostic is not recursive, day-ahead, future-year or general application validation.",
            "Observed real held-out past lags may dominate performance; persistence and real-trained references are mandatory.",
            "Default context excludes midnight-to-03:00 targets; hourly and quarter-hourly runs have different forecast horizons.",
            "Overlapping windows and repeated days per meter are dependent; no IID-window uncertainty or significance is claimed.",
            "Validation reuse, checkpoint selection and repeated candidate screening make this exploratory.",
            "Equal synthetic/real counts pair condition exposure only, not individual synthetic profiles with real days.",
            "GAN normalization uses the full prepared train split; matching gradient/profile pools does not imply identical preprocessing exposure.",
            "No privacy, causal or broad application-utility guarantee follows from these scores.",
            "Fixed marginal calibration imposes marginal distributions and diversity; it does not establish correct joint dependence or application utility.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New validation JSON report")
    parser.add_argument("--samples-per-condition", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-profiles", type=int, default=None,
                        help="Optional uniform downstream-training cap; default uses every actual GAN training row")
    parser.add_argument("--copula", action="store_true",
                        help="Include Gaussian copula; match saved calibrator quantile knots if supplied, otherwise exact margins")
    parser.add_argument("--calibrator", default=None,
                        help="Saved calibration NPZ bound to this checkpoint and exact training subset")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new report path: {output}")
    if output.suffix.lower() != ".json":
        raise ValueError("output must have a .json extension")
    report = evaluate(args.prepared, args.checkpoint, samples_per_condition=args.samples_per_condition,
                      seed=args.seed, max_train_profiles=args.max_train_profiles, copula=args.copula,
                      calibrator=args.calibrator)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(f"Wrote validation-only utility report: {output}")
    print(json.dumps({name: values["macro"] for name, values in report["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
