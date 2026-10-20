"""Frozen common-hourly observed-history ridge utility on validation only.

Every fitted predictor uses all actual TRAIN rows in the same label order.
Quarter-hour generator output is optionally mapped at its native resolution,
then summed to hourly energy. No calibration map is fitted here. This command
has no TEST option, training cap, feature tuning, or quarter-hour score endpoint.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_resolutions import (
    AGGREGATION, _model, _prepared_pair, _reference_configs, _saved_map,
    _string_digest, hourly_sum,
)
from compare_checkpoints import _digest
from evaluate_calibration import _generate_verified, _integer
from evaluate_utility import _counts, _profile_digest
from fluvius_cgan.copula import fit_conditional_copula
from fluvius_cgan.evaluation import resample_baseline
from fluvius_cgan.release import sha256_file
from fluvius_cgan.training import _provenance
from fluvius_cgan.utility import RidgeProfileForecaster
from fluvius_cgan.utility_uncertainty import paired_utility_meter_bootstrap


TRANSFORM_BATCH_PROFILES = 4096
RIDGE_BATCH_PROFILES = 256
BOOTSTRAP = {"n_bootstrap": 500, "confidence": .95, "seed": 42027}


def _hourly_training(values, labels, width, saved_map=None, *, batch_profiles=TRANSFORM_BATCH_PROFILES):
    """Return hourly profiles and a native-output hash using bounded transforms.

    Generation precedes this helper and happens once for the complete label
    vector. Chunking must never restart the generator's latent RNG stream.
    """
    batch_profiles = _integer("batch_profiles", batch_profiles)
    values, labels = np.asarray(values), np.asarray(labels)
    if (width not in (24, 96) or labels.ndim != 1 or not len(labels)
            or values.shape != (len(labels), width) or values.dtype.kind not in "fiu"):
        raise ValueError("native training profiles and labels have incompatible shape or resolution")
    dtype = np.float32 if width == 96 else (np.float64 if saved_map is not None else values.dtype)
    hourly = np.empty((len(values), 24), dtype=dtype)
    native_hash = hashlib.sha256()
    for start in range(0, len(values), batch_profiles):
        stop = min(start + batch_profiles, len(values))
        block = values[start:stop]
        if saved_map is not None:
            block = np.asarray(saved_map.transform(block, labels[start:stop]))
        if block.shape != (stop - start, width) or not np.isfinite(block).all():
            raise ValueError("native generation/map output must be finite and preserve profile shape")
        native_hash.update(np.asarray(block, dtype="<f8").tobytes(order="C"))
        hourly[start:stop] = hourly_sum(block) if width == 96 else block
    return hourly, native_hash.hexdigest()


def _uncertainty_pairs(methods):
    """Only the prospectively declared resolution and calibrated-control pairs."""
    pairs = [("bce24_raw", "bce96_raw")]
    if "bce24_calibrated" in methods:
        pairs.append(("bce24_calibrated", "bce96_calibrated"))
    for name in methods:
        if name.endswith("_calibrated"):
            pairs.extend((name, control) for control in ("training_real24", "conditional_resampling24"))
    return pairs


def evaluate(prepared96, prepared24, checkpoint96, checkpoint24, *, map96=None, map24=None,
             candidate96=None, candidate_map96=None, samples_per_condition=100, seed=2026):
    """Fit fixed hourly predictors and bootstrap their frozen real-row errors."""
    cap = _integer("samples_per_condition", samples_per_condition)
    seed = _integer("seed", seed, 0)
    if seed >= 2**32:
        raise ValueError("seed must be smaller than 2**32")
    if (map96 is None) != (map24 is None):
        raise ValueError("provide both reference maps or neither")
    if candidate_map96 is not None and candidate96 is None:
        raise ValueError("candidate_map96 requires candidate96")
    parent, child, row_bindings, manifests = _prepared_pair(prepared96, prepared24)
    models = {
        "bce96": _model(checkpoint96, parent, "curation_manifest.json", manifests["parent_curation_manifest"]["sha256"]),
        "bce24": _model(checkpoint24, child, "derivation_manifest.json", manifests["hourly_derivation_manifest"]["sha256"]),
    }
    config = _reference_configs(models["bce96"], models["bce24"])
    specs = [("bce96", checkpoint96, map96, 96), ("bce24", checkpoint24, map24, 24)]
    bundles = {"bce96": parent, "bce24": child}
    if candidate96 is not None:
        models["candidate96"] = _model(candidate96, parent, "curation_manifest.json", manifests["parent_curation_manifest"]["sha256"])
        if models["candidate96"]["config"]["seed"] != config["seed"]:
            raise ValueError("candidate96 must match the reference training seed")
        specs.append(("candidate96", candidate96, candidate_map96, 96))
        bundles["candidate96"] = parent
    train, val = child[:2]
    conditions = np.unique(parent[1]["y"])
    for name, details in models.items():
        if np.setdiff1d(conditions, details["supported_labels"]).size:
            raise ValueError(f"{name} lacks observed validation conditions")
    # Validate every saved map before any downstream fit or generation.
    maps, map_models = {}, {}
    for name, _, map_path, _ in specs:
        if map_path is not None:
            map_models[name], maps[name] = _saved_map(map_path, models[name], bundles[name], seed)
    rng = np.random.default_rng(seed)
    selected = np.concatenate([
        rng.choice(np.flatnonzero(parent[1]["y"] == label),
                   min(cap, int(np.sum(parent[1]["y"] == label))), replace=False)
        for label in conditions
    ])
    vx = hourly_sum(parent[1]["x"][selected])
    vy, meter_ids, dates = (val[key][selected] for key in ("y", "meter_id", "date"))
    prepared_hashes = {"parent96": parent[4], "derived24": child[4]}
    # Only hourly TRAIN and selected hourly VAL arrays are needed after binding.
    del bundles, parent
    labels = train["y"]
    label_hash = _digest(labels, "<i8")
    expected_counts = _counts(labels)
    methods, fits, errors, pipelines = {}, {}, {}, {}
    persistence = None
    feature_names = None

    def fit_evaluate(name, profiles, pipeline):
        nonlocal persistence, feature_names
        if np.asarray(profiles).shape != train["x"].shape:
            raise ValueError(f"{name}: every ridge fit must use the complete matched hourly TRAIN count")
        model = RidgeProfileForecaster(lags=3, batch_profiles=RIDGE_BATCH_PROFILES).fit(profiles, labels)
        report = model.evaluate(vx, vy)
        baseline = report.pop("persistence")
        if persistence is None:
            persistence, feature_names = baseline, model.feature_names_
        elif persistence != baseline or feature_names != model.feature_names_:
            raise ValueError("all methods must share identical hourly features and validation windows")
        if model.fit_summary_["condition_counts"] != expected_counts:
            raise ValueError("downstream training condition counts differ")
        residuals = model.predict(vx, vy) - np.asarray(vx[:, 3:], dtype=np.float64)
        if residuals.shape != (len(vx), 21) or not np.isfinite(residuals).all():
            raise ValueError("hourly prediction errors must be finite and preserve real row/window order")
        methods[name], errors[name] = report, residuals
        fits[name] = {**model.fit_summary_, "profiles_sha256": _profile_digest(profiles),
                      "labels_sha256": label_hash, "coefficients_sha256": _digest(model.coef_, "<f8"),
                      "validation_errors_sha256": _profile_digest(residuals)}
        pipelines[name] = {**pipeline, "evaluated_resolution": 24, "training_dtype": str(np.asarray(profiles).dtype)}

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        context, blas_limit = nullcontext(), None
    else:
        context, blas_limit = threadpool_limits(limits=1, user_api="blas"), 1
    with context:
        fit_evaluate("training_real24", train["x"], {"fitted_on": "full observed TRAIN24"})
        for name, path, map_path, width in specs:
            generated = _generate_verified(path, models[name], labels, seed)
            hourly, native_hash = _hourly_training(generated, labels, width)
            fit_evaluate(name + "_raw", hourly,
                         {"native_resolution": width, "native_profiles_sha256": native_hash, "saved_map": False,
                          "aggregation": AGGREGATION if width == 96 else None})
            del hourly
            if map_path is not None:
                hourly, mapped_hash = _hourly_training(generated, labels, width, map_models[name])
                fit_evaluate(name + "_calibrated", hourly,
                             {"native_resolution": width, "native_raw_profiles_sha256": native_hash,
                              "native_mapped_profiles_sha256": mapped_hash, "saved_map": True,
                              "aggregation": AGGREGATION if width == 96 else None,
                              "order": "native generator -> saved native map -> hourly sum" if width == 96
                                       else "hourly generator -> saved hourly map (float64 retained)"})
                del hourly
                del map_models[name]
            del generated
        generated = resample_baseline(train["x"], labels, labels, seed=seed)
        fit_evaluate("conditional_resampling24", generated, {"fitted_on": "full TRAIN24", "seed": seed})
        del generated
        copula = fit_conditional_copula(train["x"], labels, quantile_knots=1025, shrinkage=.01)
        generated = copula.generate(labels, seed=seed, batch_size=4096)
        fit_evaluate("gaussian_copula24", generated,
                     {"fitted_on": "full TRAIN24", "seed": seed, "quantile_knots": 1025, "shrinkage": .01})
        del generated, copula
        uncertainty = {}
        for candidate, comparator in _uncertainty_pairs(methods):
            name = candidate + "_minus_" + comparator
            result = paired_utility_meter_bootstrap(errors[candidate], errors[comparator], vy, meter_ids, **BOOTSTRAP)
            uncertainty[name] = {
                "candidate": candidate, "comparator": comparator,
                "direction": "candidate minus comparator; negative favors candidate",
                "candidate_errors_sha256": fits[candidate]["validation_errors_sha256"],
                "comparator_errors_sha256": fits[comparator]["validation_errors_sha256"],
                "bootstrap": result,
            }
    methods["persistence"] = persistence
    for name, path, map_path, _ in specs:
        if sha256_file(path) != models[name]["sha256"]:
            raise ValueError("checkpoint changed during hourly utility evaluation")
        if map_path is not None and sha256_file(map_path) != maps[name]["sha256"]:
            raise ValueError("saved map changed during hourly utility evaluation")
    for manifest in manifests.values():
        if sha256_file(manifest["path"]) != manifest["sha256"]:
            raise ValueError("preparation manifest changed during hourly utility evaluation")
    provenance = _provenance(torch.device("cpu"))
    provenance.update({"scipy": str(scipy.__version__), "blas_thread_limit": blas_limit,
                       "runner_sha256": sha256_file(Path(__file__)),
                       "helper_files_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
                           ROOT / "scripts/compare_resolutions.py", ROOT / "scripts/compare_checkpoints.py",
                           ROOT / "scripts/evaluate_calibration.py", ROOT / "scripts/evaluate_utility.py")}})
    return {
        "schema_version": 1, "split": "val", "test_used": False,
        "scope": "Exploratory common-hourly observed-history utility and conditional real-meter uncertainty; no TEST.",
        "source_sha256": child[2]["source_sha256"], "target": "net", "units": "kWh/interval",
        "prepared_data_hashes": prepared_hashes, "preparation_manifests": manifests,
        "derived_provenance": child[2]["derived"], "row_bindings": row_bindings,
        "models": models, "saved_maps": maps, "aggregation": AGGREGATION,
        "seeds": {"training": config["seed"], "validation_selection": seed,
                  "full_train_generation": seed, "conditional_resampling": seed, "gaussian_copula": seed,
                  "calibration_reference": 7301 if maps else None, "meter_bootstrap": BOOTSTRAP["seed"]},
        "task": {"name": "within_day_observed_history_next_interval_linear_ridge",
                 "resolution": 24, "lags": 3, "forecast_horizon_intervals": 1, "forecast_horizon_hours": 1.,
                 "target_slots_inclusive": [3, 23], "windows_per_profile": 21, "feature_names": feature_names,
                 "clock_features": "sin/cos at harmonic orders one and two (four columns)",
                 "ridge_alpha": 1., "ridge_objective": "sum_squared_error_plus_alpha_nonintercept_squared_norm",
                 "feature_scaling": "none; hourly kWh/interval", "batch_profiles": RIDGE_BATCH_PROFILES,
                 "macro_definition": "Arithmetic mean of per-condition RMSE/MAE, equal condition weight."},
        "utility_training_selection": {
            "cap_applied": False, "selected_profiles": len(labels), "condition_counts": expected_counts,
            "labels_sha256": label_hash, "selected_indices_sha256": _digest(np.arange(len(labels)), "<i8"),
            "policy": "Every actual TRAIN row in original prepared order; identical labels for all training pools.",
            "real_profiles_sha256": _profile_digest(train["x"]),
        },
        "validation_selection": {
            "policy": "One uniform without-replacement selection from 96-slot VAL reused for every hourly predictor.",
            "samples_per_condition_cap": cap, "indices": selected.tolist(),
            "selected_indices_sha256": _digest(selected, "<i8"), "profiles_sha256": _profile_digest(vx),
            "labels": vy.tolist(), "labels_sha256": _digest(vy, "<i8"), "condition_counts": _counts(vy),
            "meter_ids": meter_ids.tolist(), "meter_ids_sha256": _string_digest(meter_ids),
            "dates": dates.tolist(), "dates_sha256": _string_digest(dates),
            "selected_profiles": len(vx), "selected_meters": len(np.unique(meter_ids)),
        },
        "training": fits, "methods": methods, "pipelines": pipelines, "paired_uncertainty": uncertainty,
        "uncertainty_policy": {**BOOTSTRAP, "resampling": "paired whole real meter within technology",
                               "scope": "Frozen predictors and selected hourly validation rows/windows; separate per-seed intervals, never averaged.",
                               "missing_conditions": "Retain helper's withheld status and null endpoints; no replacement draws."},
        "memory_policy": {"generator_calls": "Once per checkpoint over the complete TRAIN label vector",
                          "map_and_aggregation_batch_profiles": TRANSFORM_BATCH_PROFILES,
                          "native96_calibrated_array_materialized": False,
                          "large_profile_hashes": "Incremental little-endian float64 C-order bytes"},
        "invariants": {"full_actual_train_required": True, "training_condition_counts_match": True,
                       "training_label_order_matches": True, "identical_validation_windows": True,
                       "fit_uses_validation_values": False, "calibration_fitted_here": False,
                       "post_generation_clipping_applied": False, "cross_day_windows": False},
        "provenance": provenance,
        "limitations": [
            "All methods predict one hour ahead from three actual observed hourly lags; native 15-minute errors are not compared.",
            "Targets start at 03:00. This is not recursive, day-ahead, future-year or general application validation.",
            "The fixed map is applied before summing 96-slot profiles; summed output is never recalibrated.",
            "Equal training labels/counts do not pair generated profiles with individual observed days.",
            "Bootstrap intervals condition on frozen predictors, generation, maps and selected validation rows; fitting and model-selection uncertainty is excluded.",
            "Intervals are descriptive, not simultaneous or formal noninferiority tests. Trace deduplication does not establish independent households.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared96", "prepared24", "checkpoint96", "checkpoint24"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("map96", "map24", "candidate96", "candidate-map96"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--samples-per-condition", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".json" or args.output.exists():
        parser.error("output must be a new .json path")
    result = evaluate(args.prepared96, args.prepared24, args.checkpoint96, args.checkpoint24,
                      map96=args.map96, map24=args.map24, candidate96=args.candidate96,
                      candidate_map96=args.candidate_map96, samples_per_condition=args.samples_per_condition, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved common-hourly validation utility: {args.output}")


if __name__ == "__main__":
    main()
