"""Compare frozen 96-slot and 24-slot models on common hourly validation rows.

The 96-slot pipeline is aggregated only after any saved marginal map. No map is
fitted here, and this command has no TEST option. Statistical controls fit only
the complete, verified hourly TRAIN pool. Native-resolution metrics, utility
and uncertainty are deliberately separate analyses.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_checkpoints import _checkpoint, _digest, _validate_manifest
from evaluate_calibration import _battery_diagnostics, _generate_verified, _integer
from fluvius_cgan.battery import simulate_battery
from fluvius_cgan.calibration import load_verified_calibrator
from fluvius_cgan.copula import fit_conditional_copula
from fluvius_cgan.derived import _preflight, resolve_parent_directory
from fluvius_cgan.evaluation import evaluate_conditional, mean_baseline, resample_baseline
from fluvius_cgan.joint import conditional_projected_wasserstein
from fluvius_cgan.release import sha256_file
from fluvius_cgan.training import _config, _load_prepared, _provenance


AGGREGATION = {
    "input_width": 96, "output_width": 24, "consecutive_intervals": 4,
    "operation": "astype(float64).reshape(N,24,4).sum(axis=2).astype(float32)",
    "accumulation_dtype": "float64", "output_dtype": "float32",
    "interval_hours": 1.0, "units": "kWh/interval",
    "order": "Apply a saved 96-slot map before aggregation; never recalibrate hourly sums.",
}


def hourly_sum(values):
    """Sum four consecutive interval energies, with declared finite rounding."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 96 or values.dtype.kind not in "fiu":
        raise ValueError("hourly aggregation requires numeric (N, 96) profiles")
    if not np.isfinite(values).all():
        raise ValueError("hourly aggregation requires finite profiles")
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = values.astype(np.float64).reshape(-1, 24, 4).sum(axis=2).astype(np.float32)
    except FloatingPointError as error:
        raise ValueError("hourly aggregation exceeds finite float32 range") from error
    if not np.isfinite(result).all():
        raise ValueError("hourly aggregation exceeds finite float32 range")
    return result


def _string_digest(values):
    """Unambiguous Unicode identity encoding, independent of array string width."""
    return hashlib.sha256(json.dumps(np.asarray(values).tolist(), ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _alignment(parent, child, *, chunk_rows=4096):
    """Bind the loaded rows as well as their on-disk derived-data provenance."""
    rows = {}
    for split, a, b in zip(("train", "val"), parent[:2], child[:2]):
        if a["x"].shape != (len(b["x"]), 96) or b["x"].shape[1] != 24:
            raise ValueError(f"{split}: 96/24 row counts or widths differ")
        if b["x"].dtype != np.float32:
            raise ValueError(f"{split}: derived hourly profiles must be float32")
        for key in ("y", "meter_id", "date"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"{split}: {key} row identities/order differ across resolutions")
        for start in range(0, len(a["x"]), chunk_rows):
            expected = hourly_sum(a["x"][start:start + chunk_rows])
            observed = np.asarray(b["x"][start:start + chunk_rows], dtype="<f4")
            if expected.astype("<f4", copy=False).tobytes() != observed.tobytes():
                raise ValueError(f"{split}: hourly profiles differ from the declared four-interval sums")
        rows[split] = {"profiles": len(a["x"]), "meters": len(np.unique(a["meter_id"])),
                       "labels_sha256": _digest(a["y"], "<i8"),
                       "meter_ids_sha256": _string_digest(a["meter_id"]),
                       "dates_sha256": _string_digest(a["date"]),
                       "same_order_and_values": True, "hourly_values_verified": True}
    return rows


def _prepared_pair(prepared96, prepared24):
    # Explicit parent knowledge allows alias checks even before the child's
    # metadata is read to discover its recorded parent.
    _preflight(Path(prepared96).resolve(), Path(prepared24).resolve())
    # For derived metadata the loader verifies the parent curation and streamed
    # transformation, including reserved-TEST alias preflight, before returning
    # arrays. Bind its recorded parent before opening the explicit parent path.
    child = _load_prepared(prepared24)
    derived = child[2].get("derived")
    if not isinstance(derived, dict) or derived.get("schema_version") != 1:
        raise ValueError("hourly data must have verified derived schema_version 1")
    recorded_parent = derived.get("parent", {})
    if resolve_parent_directory(prepared24, derived) != Path(prepared96).resolve():
        raise ValueError("derived hourly data are not bound to the supplied 96-slot parent")
    parent = _load_prepared(prepared96)
    for bundle in (parent, child):
        _validate_manifest(bundle[2], bundle[4], bundle[0], bundle[1])
        if bundle[2]["target"] != "net":
            raise ValueError("common-hourly battery comparison requires signed net energy")
    if parent[2]["resolution"] != 96 or child[2]["resolution"] != 24:
        raise ValueError("prepared96 and prepared24 must have resolutions 96 and 24 respectively")
    if recorded_parent.get("development_files_sha256") != parent[4]:
        raise ValueError("derived hourly data are not bound to the supplied 96-slot parent")
    parent_manifest = sha256_file(Path(prepared96) / "curation_manifest.json")
    if recorded_parent.get("curation_manifest_sha256") != parent_manifest:
        raise ValueError("derived parent curation manifest fingerprint differs")
    if (parent[2]["source_sha256"] != child[2]["source_sha256"]
            or parent[2]["split_seed"] != child[2]["split_seed"]):
        raise ValueError("source release or original split seed differs across resolutions")
    manifests = {"parent_curation_manifest": {"path": str(Path(prepared96) / "curation_manifest.json"),
                                               "sha256": parent_manifest},
                 "hourly_derivation_manifest": {"path": str(Path(prepared24) / "derivation_manifest.json"),
                                                "sha256": sha256_file(Path(prepared24) / "derivation_manifest.json")}}
    return parent, child, _alignment(parent, child), manifests


def _model(path, bundle, manifest_name, manifest_hash):
    # Checkpoint identities include their preparation manifest; saved-map
    # identities deliberately retain the established four-file mapping.
    train, _, metadata, scaler, hashes = bundle
    details, indices = _checkpoint(path, train, metadata, scaler,
                                   {**hashes, manifest_name: manifest_hash})
    if (len(indices) != len(train["x"])
            or details["training_selection"]["max_train_samples"] is not None
            or not np.array_equal(indices, np.arange(len(train["x"])))):
        raise ValueError("every resolution-comparison checkpoint must use the full actual TRAIN pool")
    return details


def _reference_configs(first, second):
    configs = [_config({**details["config"], "device": "cpu"}) for details in (first, second)]
    if configs[0] != configs[1]:
        raise ValueError("BCE96/BCE24 configurations and training seeds must match")
    c = configs[0]
    if (c["objective"] != "bce" or c["normalization"] != "minmax" or c["unconditional"]
            or c["conditional_mean_residual"] or c["swd_weight"] or c["derivative_swd_weight"]):
        raise ValueError("the reference pair must be conditional minmax BCE without auxiliary/residual terms")
    return configs[0]


def _saved_map(path, details, bundle, seed):
    model, provenance = load_verified_calibrator(
        path, checkpoint_sha256=details["sha256"],
        train_indices_sha256=details["training_selection"]["train_indices_sha256"],
        prepared_data_hashes=bundle[4], generation_seed=seed,
        n_features=bundle[2]["resolution"], train_labels=bundle[0]["y"])
    if (provenance["quantile_knots"] != 1025 or provenance["calibration_seed"] != 7301
            or provenance["provenance"]["calibration_samples_per_condition"] != 4096):
        raise ValueError("saved maps must use 1025 knots, calibration seed 7301 and 4096 draws/condition")
    return model, provenance


def compare(prepared96, prepared24, checkpoint96, checkpoint24, *, map96=None, map24=None,
            candidate96=None, candidate_map96=None, samples_per_condition=100, seed=2026):
    """Evaluate regenerated frozen pipelines in one common hourly coordinate space."""
    cap = _integer("samples_per_condition", samples_per_condition)
    seed = _integer("seed", seed, 0)
    if seed >= 2**32:
        raise ValueError("seed must be smaller than 2**32")
    if (map96 is None) != (map24 is None):
        raise ValueError("provide both reference maps or neither")
    if candidate_map96 is not None and candidate96 is None:
        raise ValueError("candidate_map96 requires candidate96")
    parent, child, row_bindings, manifests = _prepared_pair(prepared96, prepared24)
    m96 = _model(checkpoint96, parent, "curation_manifest.json", manifests["parent_curation_manifest"]["sha256"])
    m24 = _model(checkpoint24, child, "derivation_manifest.json", manifests["hourly_derivation_manifest"]["sha256"])
    shared_config = _reference_configs(m96, m24)
    specs = [("bce96", checkpoint96, map96, m96, parent), ("bce24", checkpoint24, map24, m24, child)]
    if candidate96 is not None:
        candidate = _model(candidate96, parent, "curation_manifest.json", manifests["parent_curation_manifest"]["sha256"])
        if candidate["config"]["seed"] != shared_config["seed"]:
            raise ValueError("candidate96 must match the reference training seed")
        specs.append(("candidate96", candidate96, candidate_map96, candidate, parent))
    train, val = child[:2]
    conditions = np.unique(parent[1]["y"])
    for name, _, _, details, _ in specs:
        if np.setdiff1d(conditions, details["supported_labels"]).size:
            raise ValueError(f"{name} lacks observed validation conditions")
    rng = np.random.default_rng(seed)
    selected = np.concatenate([rng.choice(np.flatnonzero(parent[1]["y"] == label),
                                         min(cap, int(np.sum(parent[1]["y"] == label))), replace=False)
                               for label in conditions])
    real = hourly_sum(parent[1]["x"][selected])
    labels, meter_ids, dates = (val[k][selected] for k in ("y", "meter_id", "date"))
    real_battery = simulate_battery(real, 1.0)
    methods, joint, battery, frozen, pipelines, model_details, maps = {}, {}, {}, {}, {}, {}, {}

    def record(name, values, pipeline):
        values = np.asarray(values)
        if values.shape != real.shape or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError(f"{name}: generated hourly profiles must be finite and match real shape")
        methods[name] = evaluate_conditional(real, labels, values, labels, interval_hours=1.0)
        joint[name] = conditional_projected_wasserstein(real, labels, values, labels)
        battery[name] = _battery_diagnostics(real_battery, values, labels, 1.0)
        frozen[name] = _digest(values, "<f8")
        pipelines[name] = {**pipeline, "evaluated_dtype": str(values.dtype), "evaluated_resolution": 24}

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        context, limit = nullcontext(), None
    else:
        context, limit = threadpool_limits(limits=1, user_api="blas"), 1
    with context:
        for name, path, map_path, details, bundle in specs:
            saved_map = None
            if map_path is not None:
                saved_map, maps[name] = _saved_map(map_path, details, bundle, seed)
            model_details[name] = details
            raw = _generate_verified(path, details, labels, seed)
            width = bundle[2]["resolution"]
            if raw.shape != (len(labels), width):
                raise ValueError(f"{name}: generator resolution differs from verified checkpoint")
            to_hourly = hourly_sum if width == 96 else np.asarray
            record(name + "_raw", to_hourly(raw), {"native_resolution": width, "saved_map": False,
                                                  "aggregation": AGGREGATION if width == 96 else None})
            if saved_map is not None:
                mapped = saved_map.transform(raw, labels)
                record(name + "_calibrated", to_hourly(mapped),
                       {"native_resolution": width, "saved_map": True,
                        "aggregation": AGGREGATION if width == 96 else None,
                        "order": "native generator -> saved native map -> hourly sum" if width == 96
                                 else "hourly generator -> saved hourly map (float64 map output retained)"})
                if sha256_file(map_path) != maps[name]["sha256"]:
                    raise ValueError("saved map changed during evaluation")
            del raw, saved_map
        record("conditional_mean24", mean_baseline(train["x"], train["y"], labels), {"fitted_on": "full TRAIN24"})
        record("conditional_resampling24", resample_baseline(train["x"], train["y"], labels, seed=seed),
               {"fitted_on": "full TRAIN24", "seed": seed})
        for name, shrinkage in (("gaussian_copula24", .01), ("independent_marginals24", 1.0)):
            control = fit_conditional_copula(train["x"], train["y"], quantile_knots=1025, shrinkage=shrinkage)
            record(name, control.generate(labels, seed=seed), {"fitted_on": "full TRAIN24", "seed": seed,
                                                              "quantile_knots": 1025, "shrinkage": shrinkage})
            del control
    for manifest in manifests.values():
        if sha256_file(manifest["path"]) != manifest["sha256"]:
            raise ValueError("preparation manifest changed during evaluation")
    provenance = _provenance(torch.device("cpu"))
    provenance.update({"scipy": scipy.__version__, "blas_thread_limit": limit,
                       "runner_sha256": sha256_file(Path(__file__)),
                       "comparison_helper_sha256": sha256_file(ROOT / "scripts/compare_checkpoints.py"),
                       "diagnostic_helper_sha256": sha256_file(ROOT / "scripts/evaluate_calibration.py")})
    return {"schema_version": 1, "split": "val", "test_used": False,
            "scope": "Exploratory matched-row common-hourly validation; no final TEST, native96 score comparison, utility or uncertainty.",
            "source_sha256": parent[2]["source_sha256"], "resolution": 24, "interval_hours": 1.0,
            "units": "kWh/interval", "aggregation": AGGREGATION,
            "prepared_data_hashes": {"parent96": parent[4], "derived24": child[4]},
            "preparation_manifests": manifests, "derived_provenance": child[2]["derived"],
            "row_bindings": row_bindings, "models": model_details, "saved_maps": maps,
            "training_profiles": {"parent96": len(parent[0]["x"]), "derived24": len(train["x"]),
                                  "controls": len(train["x"]), "full_actual_train_required": True},
            "seed": seed, "samples_per_condition": cap,
            "validation_selection": {"policy": "One uniform without-replacement selection from96, reused for every hourly representation.",
                                     "indices": selected.tolist(), "selected_indices_sha256": _digest(selected, "<i8"),
                                     "meter_ids": meter_ids.tolist(), "dates": dates.tolist(), "labels": labels.tolist(),
                                     "condition_counts": {str(c): int(np.sum(labels == c)) for c in conditions}},
            "frozen_samples": {"profile_hash_encoding": "little-endian float64 C-order bytes",
                               "identity_hash_encoding": "UTF-8 compact JSON arrays for Unicode; little-endian int64 C-order labels/indices",
                               "real_hourly_sha256": _digest(real, "<f8"), "labels_sha256": _digest(labels, "<i8"),
                               "profiles_sha256": frozen},
            "pipelines": pipelines, "methods": methods, "joint_diagnostics": joint, "battery_diagnostics": battery,
            "physical_lag_interpretation": {"common_hourly": "lag1 = 1 hour; lag6 = 6 hours in every condition/method",
                                            "native96_not_compared": "native96 lag1 is 15 minutes; native96 lag4/lag24 correspond to common-hourly lag1/lag6",
                                            "ramps_and_battery": "Recomputed at interval_hours=1 on hourly energies; native15-minute ramps and dispatch are different endpoints."},
            "limitations": ["Calibration and aggregation do not commute;96 calibrated sums were not recalibrated.",
                            "Finite24D projections use seed9182 and are not comparable to native96D distances.",
                            "Saved-map provenance is validated; no map fitting or validation-based transform occurs here.",
                            "Exact row/trace bindings do not prove independent households or exclude near-duplicates.",
                            "Controls share the full TRAIN24 pool; repeated deterministic controls are not independent replications."],
            "provenance": provenance}


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
    report = compare(args.prepared96, args.prepared24, args.checkpoint96, args.checkpoint24,
                     map96=args.map96, map24=args.map24, candidate96=args.candidate96,
                     candidate_map96=args.candidate_map96, samples_per_condition=args.samples_per_condition,
                     seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved common-hourly validation comparison: {args.output}")


if __name__ == "__main__":
    main()
