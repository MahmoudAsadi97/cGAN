"""Validation-only diagnostic of fixed marginal calibration for a frozen cGAN.

Calibration and evaluation seeds must differ. Each fixed map uses the verified
checkpoint training subset and an independent generated calibration reference.
This command has no test-split option. Uncurated original-study data remain
EXPLORATORY DIAGNOSTICS ONLY, including when marginal metrics improve.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.stats import wasserstein_distance
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_checkpoints import _checkpoint, _digest, _validate_manifest
from fluvius_cgan.battery import simulate_battery
from fluvius_cgan.calibration import fit
from fluvius_cgan.copula import fit_conditional_copula
from fluvius_cgan.evaluation import evaluate_conditional, nearest_neighbor_diagnostics, resample_baseline
from fluvius_cgan.joint import conditional_projected_wasserstein
from fluvius_cgan.release import sha256_file
from fluvius_cgan.training import _load_prepared, _provenance, generate


EVENTS = ("source_below", "source_above", "cdf_zero", "cdf_one",
          "target_lower_endpoint", "target_upper_endpoint")
METHODS = ("raw_cgan", "calibrated_cgan", "gaussian_copula", "independent_marginals",
           "exact_marginal_copula", "conditional_resampling")
BATTERY_QUANTITIES = ("post_grid_import_kwh", "post_grid_export_kwh",
                      "grid_import_reduction_kwh", "post_peak_import_power_kw", "end_soc_kwh")


def _profile_digest(values):
    """Canonical float64 hash without copying the full calibration reference."""
    digest = hashlib.sha256()
    for start in range(0, len(values), 4096):
        digest.update(np.asarray(values[start:start + 4096], dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _integer(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _new_calibration_path(path):
    if path is None:
        return None
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("calibration_output must have a .npz extension")
    if path.exists():
        raise FileExistsError(f"calibration output already exists: {path}")
    return path


def _compact_fitting(diagnostics):
    conditions = {}
    for label, condition in diagnostics["per_condition"].items():
        slots = condition["slots"]
        conditions[label] = {
            "train_count": condition["train_count"],
            "calibration_count": condition["calibration_count"],
            "source_tie_mass_by_slot": [s["source_tie_mass"] for s in slots],
            "source_largest_atom_mass_by_slot": [s["source_largest_atom_mass"] for s in slots],
            "source_knot_plateau_mass_by_slot": [sum(p["mass"] for p in s["source_knot_plateaus"]) for s in slots],
            "source_constant_slots": [s["slot"] for s in slots if s["source_constant"]],
            "target_constant_slots": [s["slot"] for s in slots if s["target_constant"]],
        }
    return {"per_condition": conditions, "tie_convention": diagnostics["tie_convention"],
            "source_tie_mass_definition": diagnostics["source_tie_mass_definition"],
            "knot_mass_definition": diagnostics["knot_mass_definition"],
            "detail_policy": "Per-slot summaries only; full tie values and fixed maps are retained in the optional NPZ."}


def _compact_application(diagnostics, width):
    conditions, totals = {}, dict.fromkeys(EVENTS, 0)
    for label, condition in diagnostics["per_condition"].items():
        count = condition["count"]
        counts = {event: [s[f"{event}_count"] for s in condition["slots"]] for event in EVENTS}
        fractions = {event: sum(values) / (count * width) for event, values in counts.items()}
        conditions[label] = {"count": count, "slot_event_counts": counts,
                             "fractions_over_profile_slots": fractions}
        for event, values in counts.items():
            totals[event] += sum(values)
    denominator = diagnostics["n_profiles"] * width
    return {"n_profiles": diagnostics["n_profiles"], "per_condition": conditions,
            "total_event_counts": totals,
            "fractions_over_profile_slots": {event: count / denominator if denominator else None
                                               for event, count in totals.items()},
            "interpretation": diagnostics["interpretation"]}


def _generate_verified(checkpoint, details, labels, seed):
    output = generate(checkpoint, labels, seed=seed)
    if sha256_file(checkpoint) != details["sha256"]:
        raise ValueError("checkpoint changed between verification and generation")
    return output


def _preparation_binding(prepared, metadata, hashes):
    """Qualify data already verified by _load_prepared; keep map hashes at four files."""
    if "derived" in metadata:
        manifest_hash = sha256_file(Path(prepared) / "derivation_manifest.json")
        return {**hashes, "derivation_manifest.json": manifest_hash}, {
            "data_curation_status": "verified_hourly_derivation_v1",
            "curation_metadata": None,
            "derivation_metadata": metadata["derived"],
            "derivation_manifest_sha256": manifest_hash,
            "reserved_parent_test96": metadata["reserved_parent_test96"],
            "data_integrity_note": (
                "The prepared-data loader verified the parent curation binding, canonical development "
                "rows and exact hourly transformation. Parent quarter-hour identity curation is inherited; "
                "no new hourly identity audit was performed. Hourly trace uniqueness and independent "
                "households are not established. Reserved TEST96 is recorded from metadata only; "
                "TEST24 remains unmaterialized. Validation remains exploratory."
            ),
        }
    curated = isinstance(metadata.get("curation"), dict) and bool(metadata["curation"])
    return hashes, {
        "data_curation_status": "curation_metadata_present_not_reaudited" if curated else "unverified_no_curation_metadata",
        "curation_metadata": metadata.get("curation"),
        "data_integrity_note": (
            "Curation metadata are retained, but this runner does not independently audit cross-meter duplication."
            if curated else
            "No curation metadata are present. Original-study train/validation data had cross-meter annual duplicates; these results do not establish contamination-free evaluation."
        ),
    }


def _check_preparation_unchanged(prepared, qualification):
    if ("derivation_manifest_sha256" in qualification and
            sha256_file(Path(prepared) / "derivation_manifest.json") != qualification["derivation_manifest_sha256"]):
        raise ValueError("derivation manifest changed during evaluation")


def _battery_diagnostics(real_battery, generated, labels, interval_hours):
    """Compare daily application outputs; real dispatch is cached across methods."""
    generated_battery = simulate_battery(generated, interval_hours)
    if real_battery["scenario"] != generated_battery["scenario"]:
        raise ValueError("battery diagnostics require the same real/generated scenario")
    scenario = generated_battery["scenario"]
    per_condition = {}
    for label in np.unique(labels):
        selected = labels == label
        result = {"n_real": int(selected.sum()), "n_generated": int(selected.sum())}
        for quantity in BATTERY_QUANTITIES:
            real_values = real_battery["per_profile"][quantity][selected]
            generated_values = generated_battery["per_profile"][quantity][selected]
            stem, unit = quantity.rsplit("_", 1)
            result[f"{stem}_wasserstein_{unit}"] = float(wasserstein_distance(real_values, generated_values))
            result[f"real_mean_{quantity}"] = float(real_values.mean())
            result[f"generated_mean_{quantity}"] = float(generated_values.mean())
        per_condition[str(label)] = result
    # Only scalar summaries leave this function; dispatch and SOC arrays are
    # released for each generated method rather than retained in the report.
    metric_keys = [key for key in next(iter(per_condition.values())) if key not in ("n_real", "n_generated")]
    return {
        "scenario": scenario,
        "scenario_sha256": hashlib.sha256(json.dumps(scenario, sort_keys=True, separators=(",", ":"),
                                                     allow_nan=False).encode("utf-8")).hexdigest(),
        "scenario_hash_encoding": "UTF-8 JSON with sorted keys, compact separators and no NaN",
        "role": "secondary_application_fidelity_diagnostic",
        "n_conditions": len(per_condition),
        "condition_weighting": "equal_over_observed_support",
        "per_condition": per_condition,
        "macro": {key: float(np.mean([row[key] for row in per_condition.values()])) for key in metric_keys},
        "interpretation": "Descriptive fidelity of a fixed hypothetical additional export-charged battery; no annual, cost, PV-attribution or hardware claim. Peak is nonnegative grid-import power, not signed net peak. Compare only identical scenarios at the same interval resolution.",
    }


def evaluate(prepared, checkpoint, *, calibration_samples_per_condition=4096,
             calibration_seed=7301, seed=2026, samples_per_condition=100,
             quantile_knots=1025, calibration_output=None):
    """Evaluate the prespecified methods on one frozen validation sample."""
    calibration_samples_per_condition = _integer("calibration_samples_per_condition", calibration_samples_per_condition)
    samples_per_condition = _integer("samples_per_condition", samples_per_condition)
    calibration_seed, seed = _integer("calibration_seed", calibration_seed, 0), _integer("seed", seed, 0)
    if seed >= 2 ** 32 or calibration_seed >= 2 ** 32:
        raise ValueError("seeds must be smaller than 2**32")
    if calibration_seed == seed:
        raise ValueError("calibration and evaluation seeds must be distinct")
    if quantile_knots is not None:
        quantile_knots = _integer("quantile_knots", quantile_knots, 2)
    calibration_output = _new_calibration_path(calibration_output)

    started = time.perf_counter()
    train, val, metadata, scaler, hashes = _load_prepared(prepared)
    _validate_manifest(metadata, hashes, train, val)
    checkpoint_hashes, qualification = _preparation_binding(prepared, metadata, hashes)
    details, indices = _checkpoint(checkpoint, train, metadata, scaler, checkpoint_hashes)
    conditions = np.unique(val["y"])
    missing = sorted(set(conditions.tolist()) - set(details["supported_labels"]))
    if missing:
        raise ValueError(f"checkpoint lacks observed validation conditions: {missing}")
    tx, ty = train["x"][indices], train["y"][indices]
    rng = np.random.default_rng(seed)
    selected = np.concatenate([
        rng.choice(np.flatnonzero(val["y"] == label),
                   min(samples_per_condition, int(np.sum(val["y"] == label))), replace=False)
        for label in conditions
    ])
    real, labels, meters = (val[key][selected] for key in ("x", "y", "meter_id"))
    source_labels = np.repeat(np.unique(ty), calibration_samples_per_condition)
    loaded_at = time.perf_counter()
    source = _generate_verified(checkpoint, details, source_labels, calibration_seed)
    source_at = time.perf_counter()
    map_provenance = {
        "checkpoint_sha256": details["sha256"],
        "train_indices_sha256": details["training_selection"]["train_indices_sha256"],
        "prepared_data_hashes": hashes,
        "calibration_seed": calibration_seed,
        "calibration_samples_per_condition": calibration_samples_per_condition,
        "calibration_profiles_total": len(source),
        "calibration_generated_profiles_sha256": _profile_digest(source),
        "calibration_labels_sha256": _digest(source_labels, "<i8"),
        "generator_inference_batch_size": 1024,
        "quantile_knots": quantile_knots,
    }
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        thread_context, thread_limit = nullcontext(), None
    else:
        thread_context, thread_limit = threadpool_limits(limits=1, user_api="blas"), 1
    with thread_context:
        calibration = fit(tx, ty, source, source_labels, quantile_knots=quantile_knots, provenance=map_provenance)
        fitted_at = time.perf_counter()
        # New draws are made after fitting, from a distinct random stream. The
        # map never sees or ranks the current evaluation batch during fitting.
        raw = _generate_verified(checkpoint, details, labels, seed)
        generated_at = time.perf_counter()
        calibrated = calibration.transform(raw, labels)
        fitting_diagnostics = _compact_fitting(calibration.fitting_diagnostics())
        application_diagnostics = _compact_application(calibration.diagnostics(raw, labels), real.shape[1])
        calibrated_at = time.perf_counter()
        gaussian = fit_conditional_copula(tx, ty, quantile_knots=quantile_knots, shrinkage=.01)
        gaussian_profiles = gaussian.generate(labels, seed=seed)
        del gaussian
        gaussian_at = time.perf_counter()
        independent = fit_conditional_copula(tx, ty, quantile_knots=quantile_knots, shrinkage=1.)
        independent_profiles = independent.generate(labels, seed=seed)
        del independent
        independent_at = time.perf_counter()
        if quantile_knots is None:
            exact_profiles = gaussian_profiles
        else:
            exact_copula = fit_conditional_copula(tx, ty, quantile_knots=None, shrinkage=.01)
            exact_profiles = exact_copula.generate(labels, seed=seed)
            del exact_copula
        exact_at = time.perf_counter()
    arrays = {"raw_cgan": raw, "calibrated_cgan": calibrated, "gaussian_copula": gaussian_profiles,
              "independent_marginals": independent_profiles,
              "exact_marginal_copula": exact_profiles,
              "conditional_resampling": resample_baseline(tx, ty, labels, seed=seed)}
    metrics = {name: evaluate_conditional(real, labels, values, labels, interval_hours=24 / metadata["resolution"])
               for name, values in arrays.items()}
    evaluated_at = time.perf_counter()
    # Frozen secondary diagnostic: independent of the stochastic training SWD
    # directions. Every method sees the same complete-profile unit projections.
    with threadpool_limits(limits=1, user_api="blas") if thread_limit else nullcontext():
        joint_diagnostics = {name: conditional_projected_wasserstein(real, labels, values, labels)
                             for name, values in arrays.items()}
    joint_at = time.perf_counter()
    battery_diagnostics = {}
    if metadata["target"] == "net":
        real_battery = simulate_battery(real, 24 / metadata["resolution"])
        battery_diagnostics = {name: _battery_diagnostics(real_battery, values, labels, 24 / metadata["resolution"])
                               for name, values in arrays.items()}
        del real_battery
    battery_at = time.perf_counter()
    neighbors = nearest_neighbor_diagnostics(tx, calibrated, real, seed=seed)
    neighbors_at = time.perf_counter()
    if sha256_file(checkpoint) != details["sha256"]:
        raise ValueError("checkpoint changed during calibration evaluation")
    _check_preparation_unchanged(prepared, qualification)
    provenance = _provenance(torch.device("cpu"))
    provenance.update({"scipy": str(scipy.__version__), "blas_thread_limit": thread_limit,
                       "evaluation_script_sha256": sha256_file(Path(__file__)),
                       "comparison_helper_sha256": sha256_file(ROOT / "scripts" / "compare_checkpoints.py"),
                       "checkpoint_sha256": details["sha256"],
                       "train_indices_sha256": details["training_selection"]["train_indices_sha256"]})

    artifact = None
    if calibration_output is not None:
        _new_calibration_path(calibration_output)
        calibration_output.parent.mkdir(parents=True, exist_ok=True)
        calibration.save(calibration_output)
        artifact = {"path": str(calibration_output), "sha256": sha256_file(calibration_output),
                    "format": "ConditionalMarginalCalibrator NPZ schema version 1; no pickle"}
    saved_at = time.perf_counter()
    return {
        "schema_version": 1, "split": "val", "test_used": False,
        "scope": "EXPLORATORY DIAGNOSTICS ONLY; no confirmatory claim and no checkpoint selection by this runner.",
        **qualification,
        "source_sha256": metadata["source_sha256"], "prepared_data_hashes": hashes,
        "units": metadata["units"], "target": metadata["target"], "resolution": metadata["resolution"],
        "interval_hours": 24 / metadata["resolution"], "checkpoint": details,
        "training_profiles": {"prepared": len(train["x"]), "model": len(tx), "baseline": len(tx)},
        "training_subset_source": "All fitted maps and statistical baselines use the checkpoint's exact verified training rows.",
        "seed": seed, "calibration_seed": calibration_seed,
        "seeds": {"calibration_reference": calibration_seed, "validation_selection": seed,
                  "evaluation_generation": seed, "baseline_generation": seed, "nearest_neighbor_subsampling": seed},
        "calibration": {
            "method": "Fixed Q_train(H_generator(x)) per condition and slot; no ranking of application batches.",
            "samples_per_condition": calibration_samples_per_condition,
            "calibration_profiles": len(source), "quantile_knots": quantile_knots,
            "distinct_calibration_evaluation_seeds": True,
            "provenance": map_provenance, "fitting_diagnostics": fitting_diagnostics,
            "application_diagnostics": application_diagnostics, "saved_model": artifact,
            "tail_policy": "Source values below/above calibration support map to 0/1; targets do not exceed train extrema.",
            "finite_reference_limitations": "Finite calibration samples, quantile compression, and source atoms make marginal matching approximate; exact target matching is not guaranteed.",
        },
        "baselines": {
            "gaussian_copula": {"quantile_knots": quantile_knots, "shrinkage": .01,
                                 "purpose": "Primary comparator using the same target marginal knots as calibration."},
            "independent_marginals": {"quantile_knots": quantile_knots, "shrinkage": 1.,
                                      "purpose": "Same target marginal knots as calibration, with independent latent coordinates."},
            "exact_marginal_copula": {"quantile_knots": None, "shrinkage": .01,
                                       "purpose": "Secondary reference retaining every sorted training observation."},
            "conditional_resampling": {"reference": "Same exact selected training subset", "seed": seed},
        },
        "validation_selection": {
            "policy": "Uniform without replacement per observed condition; same seed/cap/order as cli.evaluate_checkpoint.",
            "samples_per_condition_cap": samples_per_condition, "available_profiles": len(val["x"]),
            "available_meters": len(np.unique(val["meter_id"])), "selected_profiles": len(real),
            "selected_meters": len(np.unique(meters)), "selected_indices_sha256": _digest(selected, "<i8"),
            "conditions": conditions.tolist(),
            "condition_counts": {str(label): int(np.sum(labels == label)) for label in conditions},
        },
        "frozen_samples": {
            "hash_encoding": "Profiles: little-endian float64 C-order bytes; labels/indices: little-endian int64 C-order bytes.",
            "profile_shape": list(real.shape), "labels_sha256": _digest(labels, "<i8"),
            "real_profiles_sha256": _profile_digest(real),
            "profiles_sha256": {name: _profile_digest(values) for name, values in arrays.items()},
            "postprocessing": "Only calibrated_cgan applies the disclosed fixed maps; no additional clipping or scaling.",
        },
        "methods": metrics, "joint_diagnostics": joint_diagnostics,
        "battery_diagnostics": battery_diagnostics,
        "battery_diagnostics_status": ("computed_on_same_frozen_signed_net_profiles" if battery_diagnostics
                                       else "unavailable_requires_signed_net_target"),
        "nearest_neighbors": neighbors, "nearest_neighbor_method": "calibrated_cgan",
        "metric_interpretation": {
            "mechanically_imposed_marginal_or_diversity_diagnostics": [
                "pointwise_wasserstein_kwh", "mean_profile_rmse_kwh", "pointwise_variance_rmse_kwh2",
                "diversity_ratio", "generated_mean_pointwise_variance_kwh2", "generated_mean_daily_energy_kwh"],
            "joint_diagnostics_to_inspect": ["daily_energy_wasserstein_kwh", "peak_power_wasserstein_kw",
                                             "ramp_wasserstein_kw", "autocorrelation"],
            "warning": "Calibration approximately moves outputs toward training marginals; marginal and diversity improvements are partly supplied by calibration and are not independent joint-model evidence. Joint diagnostics also change with marginals and require comparison to independent_marginals and other baselines.",
        },
        "timings_seconds": {
            "load_and_verify": loaded_at - started, "generate_calibration_reference": source_at - loaded_at,
            "fit_calibration_and_reference_hashes": fitted_at - source_at,
            "generate_evaluation": generated_at - fitted_at,
            "apply_calibration_and_diagnostics": calibrated_at - generated_at,
            "fit_and_generate_gaussian_copula": gaussian_at - calibrated_at,
            "fit_and_generate_independent_marginals": independent_at - gaussian_at,
            "fit_and_generate_exact_marginal_copula": exact_at - independent_at,
            "resample_and_evaluate_all_methods": evaluated_at - exact_at,
            "fixed_joint_projection_diagnostics": joint_at - evaluated_at,
            "fixed_battery_application_diagnostics": battery_at - joint_at,
            "nearest_neighbors": neighbors_at - battery_at,
            "provenance_and_optional_save": saved_at - neighbors_at,
            "total_before_report_serialization": time.perf_counter() - started,
        },
        "provenance": provenance,
        "interpretation": "Descriptive validation diagnostics only. Fixed calibration cannot restore missing dependence or split source atoms. Copula baselines are statistical comparators, not evidence of GAN novelty. Nearest-neighbor checks provide no privacy guarantee. Original uncurated results must not be treated as clean confirmation.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-output")
    parser.add_argument("--calibration-samples-per-condition", type=int, default=4096)
    parser.add_argument("--calibration-seed", type=int, default=7301)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples-per-condition", type=int, default=100)
    parser.add_argument("--quantile-knots", type=int, default=1025)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new JSON path: {output}")
    if output.suffix.lower() != ".json":
        raise ValueError("output must have a .json extension")
    if args.calibration_output is not None and output.resolve() == Path(args.calibration_output).resolve():
        raise ValueError("report and calibration output paths must differ")
    report = evaluate(args.prepared, args.checkpoint,
                      calibration_samples_per_condition=args.calibration_samples_per_condition,
                      calibration_seed=args.calibration_seed, seed=args.seed,
                      samples_per_condition=args.samples_per_condition,
                      quantile_knots=args.quantile_knots, calibration_output=args.calibration_output)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    summary_keys = ("pointwise_wasserstein_kwh", "daily_energy_wasserstein_kwh", "peak_power_wasserstein_kw",
                    "ramp_wasserstein_kw", "diversity_ratio")
    print(f"Wrote EXPLORATORY validation calibration diagnostics: {output}")
    print(json.dumps({"macro": {name: {key: result["macro"][key] for key in summary_keys}
                                for name, result in report["methods"].items()},
                      "timings_seconds": report["timings_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
