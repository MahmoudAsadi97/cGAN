"""Paired validation uncertainty for the four declared reference contrasts.

One report describes one matched training seed. No fitting, TEST option,
cross-seed pooling or averaging of confidence limits is implemented here.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_checkpoints import _digest, _validate_manifest
from compare_resolutions import AGGREGATION, _model, _prepared_pair, _saved_map, _string_digest, hourly_sum
from evaluate_calibration import _generate_verified
from fluvius_cgan.derived import _preflight
from fluvius_cgan.evaluation import evaluate_conditional
from fluvius_cgan.release import sha256_file
from fluvius_cgan.training import _config, _load_prepared, _provenance, _training_curation_guard
from fluvius_cgan.uncertainty import paired_condition_macro_meter_bootstrap


CONTRASTS = ("conditioning", "candidate_reference", "resolution_raw", "resolution_calibrated")
METRICS = ("pointwise_wasserstein_kwh", "daily_energy_wasserstein_kwh",
           "peak_power_wasserstein_kw", "ramp_wasserstein_kw", "lag_one_acf_wasserstein")
SEEDS = (11, 22, 33, 44, 55)
CAP, DRAW_SEED, REPLICATES = 100, 2026, 500
TRAIN_PROFILES = 606060
CONDITION_SUPPORT = tuple(range(64))
SOURCE_CORE = ("src/fluvius_cgan/training.py", "src/fluvius_cgan/models.py",
               "src/fluvius_cgan/residual.py", "src/fluvius_cgan/distribution_loss.py")
HELPERS = ("scripts/compare_resolutions.py", "scripts/compare_checkpoints.py",
           "scripts/evaluate_calibration.py")


def _declared_config(seed, *, unconditional=False, candidate_kind=None):
    """The declared packages; defaults are resolved before exact comparison."""
    config = dict(epochs=26, batch_size=512, seed=seed, latent_dim=24,
                  hidden_dim=64, discriminator_hidden_dim=64, objective="bce",
                  lr_g=.0001, lr_d=.0002, adam_beta1=.9, adam_beta2=.999,
                  normalization="minmax", conditional_mean_residual=False,
                  swd_weight=0., derivative_swd_weight=0., swd_projections=128,
                  batch_sampling="balanced_conditions", conditions_per_batch=4,
                  validation_selection="conditional_w1", validation_samples=4096,
                  max_train_samples=None, patience=None, device="cpu", num_threads=1,
                  unconditional=unconditional, allow_uncurated_data=False)
    if candidate_kind is not None:
        if candidate_kind not in ("swd10", "capacity"):
            raise ValueError("candidate_kind must be swd10 or capacity")
        config.update(discriminator_hidden_dim=128, objective="hinge_sn",
                      adam_beta1=.5, normalization="standard", swd_weight=10.)
        if candidate_kind == "capacity":
            config.update(hidden_dim=128, latent_dim=96)
    return _config(config)


def _config_contract(contrast, first, second, candidate_kind):
    configs = [_config(details["config"]) for details in (first, second)]
    seed = configs[0]["seed"]
    if seed not in SEEDS or configs[1]["seed"] != seed:
        raise ValueError("both checkpoints must share one declared training seed: 11,22,33,44,55")
    expected = [_declared_config(seed), _declared_config(seed)]
    if contrast == "conditioning":
        expected[1] = _declared_config(seed, unconditional=True)
    elif contrast == "candidate_reference":
        expected[0] = _declared_config(seed, candidate_kind=candidate_kind)
    for name, observed, wanted in zip(("candidate", "comparator"), configs, expected):
        if observed != wanted:
            changed = sorted(key for key in wanted if observed.get(key) != wanted[key])
            raise ValueError(f"{name} differs from the declared {contrast} package: {changed}")
    first_selection, second_selection = first["training_selection"], second["training_selection"]
    for key in ("train_indices_sha256", "train_profiles_available", "train_profiles_used"):
        if first_selection[key] != second_selection[key]:
            raise ValueError(f"canonical full TRAIN selections differ: {key}")
    return seed


def _source_contract(contrast, first, second):
    sources = []
    for name, details in (("candidate", first), ("comparator", second)):
        provenance = details.get("training_provenance")
        hashes = provenance.get("source_files_sha256") if isinstance(provenance, dict) else None
        if (not isinstance(hashes, dict) or any(not isinstance(hashes.get(path), str)
                or re.fullmatch(r"[0-9a-f]{64}", hashes[path]) is None for path in SOURCE_CORE)):
            raise ValueError(f"{name} is missing training source fingerprints")
        sources.append(hashes)
    differences = {path: {"candidate": sources[0].get(path), "comparator": sources[1].get(path)}
                   for path in sorted(set(sources[0]) | set(sources[1]))
                   if sources[0].get(path) != sources[1].get(path)}
    if contrast == "conditioning" and any(path in differences for path in SOURCE_CORE):
        raise ValueError("conditioning contrast requires matching core training implementation")
    return {"training_source_differences": differences,
            "training_core_byte_identical": not any(path in differences for path in SOURCE_CORE),
            "qualification": (
                "Both frozen arrays are regenerated with this one recorded evaluator. Historical endpoint "
                "reports are not imported or replaced. Declared package/resolution contrasts may have "
                "different recorded training sources; these are retained, not asserted identical.")}


def _arguments(contrast, prepared24, candidate_map, comparator_map, candidate_kind):
    if contrast not in CONTRASTS:
        raise ValueError(f"contrast must be one of {CONTRASTS}")
    resolution = contrast.startswith("resolution_")
    calibrated = contrast in ("candidate_reference", "resolution_calibrated")
    if resolution != (prepared24 is not None):
        raise ValueError("prepared24 is required only for a resolution contrast")
    if calibrated:
        if candidate_map is None or comparator_map is None:
            raise ValueError("this contrast requires both frozen native-resolution maps")
    elif candidate_map is not None or comparator_map is not None:
        raise ValueError("raw contrasts forbid maps, including conditional maps on unconditional generators")
    if contrast == "candidate_reference":
        if candidate_kind not in ("swd10", "capacity"):
            raise ValueError("candidate_reference requires explicit candidate_kind swd10 or capacity")
    elif candidate_kind is not None:
        raise ValueError("candidate_kind is only valid for candidate_reference")
    return resolution, calibrated


def _native_prepared(prepared96):
    _preflight(Path(prepared96).resolve())
    qualification = _training_curation_guard(prepared96, {"allow_uncurated_data": False})
    if qualification.get("status") != "verified_curation_v2" or qualification.get("pinned_release") is not True:
        raise ValueError("reference96 requires verified curation v2 of the pinned release")
    bundle = _load_prepared(prepared96)
    train, val, metadata, _, hashes = bundle
    _validate_manifest(metadata, hashes, train, val)
    if (metadata["resolution"] != 96 or metadata["target"] != "net"
            or not isinstance(metadata.get("curation"), dict) or not metadata["curation"]):
        raise ValueError("reference96 requires identity-curated 96-slot signed net preparation")
    if qualification.get("verified_file_hashes") != hashes:
        raise ValueError("curated development files changed between verification and loading")
    path = Path(prepared96) / "curation_manifest.json"
    digest = sha256_file(path)
    if digest != qualification.get("manifest_sha256"):
        raise ValueError("curation manifest changed between verification and loading")
    manifests = {"parent_curation_manifest": {"path": str(path), "sha256": digest}}
    return bundle, manifests


def _population_contract(bundle):
    train, val = bundle[:2]
    if len(train["x"]) != TRAIN_PROFILES:
        raise ValueError(f"declared reference TRAIN pool requires {TRAIN_PROFILES} profiles")
    for name, split in (("TRAIN", train), ("VAL", val)):
        if not np.array_equal(np.unique(split["y"]), CONDITION_SUPPORT):
            raise ValueError(f"declared reference {name} must retain all 64 conditions")


def _checked_output(values, rows, width, name):
    values = np.asarray(values)
    if (values.shape != (rows, width) or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()):
        raise ValueError(f"{name}: generated profiles must be finite numeric ({rows}, {width})")
    return values


def _artifact_preflight(prepared96, prepared24, *artifacts):
    """Reject TEST aliases in model/map arguments using path metadata only."""
    reserved = [Path(directory).resolve() / "test.npz"
                for directory in (prepared96, prepared24) if directory is not None]
    for value in artifacts:
        if value is None:
            continue
        path = Path(value)
        if path.name == "test.npz" or path.resolve().name == "test.npz":
            raise ValueError("checkpoint/map input aliases reserved TEST")
        if path.exists() and any(target.exists() and path.samefile(target) for target in reserved):
            raise ValueError("checkpoint/map input aliases reserved TEST")


def compare(prepared96, candidate, comparator, *, contrast, prepared24=None,
            candidate_map=None, comparator_map=None, candidate_kind=None):
    """One matched-seed contrast; all sampling/CI settings are protocol constants."""
    resolution, calibrated = _arguments(contrast, prepared24, candidate_map, comparator_map, candidate_kind)
    _artifact_preflight(prepared96, prepared24, candidate, comparator, candidate_map, comparator_map)
    if resolution:
        parent, child, row_bindings, manifests = _prepared_pair(prepared96, prepared24)
        first_bundle, second_bundle = child, parent
        first_manifest = ("derivation_manifest.json", manifests["hourly_derivation_manifest"]["sha256"])
    else:
        parent, manifests = _native_prepared(prepared96)
        child, row_bindings = None, None
        first_bundle = second_bundle = parent
        first_manifest = ("curation_manifest.json", manifests["parent_curation_manifest"]["sha256"])
    _population_contract(parent)
    if resolution:
        _population_contract(child)
    second_manifest = ("curation_manifest.json", manifests["parent_curation_manifest"]["sha256"])
    first = _model(candidate, first_bundle, *first_manifest)
    second = _model(comparator, second_bundle, *second_manifest)
    training_seed = _config_contract(contrast, first, second, candidate_kind)
    source_binding = _source_contract(contrast, first, second)
    conditions = np.unique(parent[1]["y"])
    for name, details in (("candidate", first), ("comparator", second)):
        if np.setdiff1d(conditions, details["supported_labels"]).size:
            raise ValueError(f"{name} lacks observed validation conditions")

    rng = np.random.default_rng(DRAW_SEED)
    selected = np.concatenate([rng.choice(np.flatnonzero(parent[1]["y"] == label),
                                        min(CAP, int(np.sum(parent[1]["y"] == label))), replace=False)
                               for label in conditions])
    val = parent[1]
    labels, meters, dates = (val[key][selected] for key in ("y", "meter_id", "date"))
    real = hourly_sum(val["x"][selected]) if resolution else val["x"][selected]
    width, hours = (24, 1.) if resolution else (96, .25)
    outputs, frozen, maps, pipelines = {}, {}, {}, {}
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        context, thread_limit = nullcontext(), None
    else:
        context, thread_limit = threadpool_limits(limits=1, user_api="blas"), 1
    with context:
        for name, path, map_path, details, bundle in (
                ("candidate", candidate, candidate_map, first, first_bundle),
                ("comparator", comparator, comparator_map, second, second_bundle)):
            native_width = bundle[2]["resolution"]
            saved_map = None
            if calibrated:
                saved_map, maps[name] = _saved_map(map_path, details, bundle, DRAW_SEED)
            raw = _checked_output(_generate_verified(path, details, labels, DRAW_SEED),
                                  len(labels), native_width, name)
            values = raw if saved_map is None else _checked_output(saved_map.transform(raw, labels),
                                                                  len(labels), native_width, name)
            frozen[name] = {"native_raw_profiles_sha256": _digest(raw, "<f8"),
                            "native_deployed_profiles_sha256": _digest(values, "<f8")}
            aggregate = resolution and native_width == 96
            outputs[name] = hourly_sum(values) if aggregate else values
            _checked_output(outputs[name], len(labels), width, name)
            frozen[name].update(evaluated_profiles_sha256=_digest(outputs[name], "<f8"),
                                evaluated_dtype=str(outputs[name].dtype), shape=list(outputs[name].shape))
            pipelines[name] = {"native_resolution": native_width, "evaluated_resolution": width,
                               "saved_map": calibrated, "aggregation": AGGREGATION if aggregate else None,
                               "order": "native generator -> " + ("fixed native map -> " if calibrated else "")
                                        + ("hourly sum" if aggregate else "evaluation")}
            if calibrated and sha256_file(map_path) != maps[name]["sha256"]:
                raise ValueError("saved map changed during evaluation")
        methods = {name: evaluate_conditional(real, labels, values, labels, interval_hours=hours)
                   for name, values in outputs.items()}
        paired = paired_condition_macro_meter_bootstrap(
            real, labels, outputs["candidate"], labels, outputs["comparator"], labels, meters,
            interval_hours=hours, n_bootstrap=REPLICATES, confidence=.95, seed=DRAW_SEED,
            real_meter_strata=labels // 8, metrics=METRICS, return_per_condition=True)

    for path, details in ((candidate, first), (comparator, second)):
        if sha256_file(path) != details["sha256"]:
            raise ValueError("checkpoint changed during comparison")
    for name, path in (("candidate", candidate_map), ("comparator", comparator_map)):
        if calibrated and sha256_file(path) != maps[name]["sha256"]:
            raise ValueError("saved map changed during comparison")
    for manifest in manifests.values():
        if sha256_file(manifest["path"]) != manifest["sha256"]:
            raise ValueError("preparation manifest changed during comparison")
    preparations = [(prepared96, parent)] + ([(prepared24, child)] if resolution else [])
    for directory, bundle in preparations:
        for filename, digest in bundle[4].items():
            if sha256_file(Path(directory) / filename) != digest:
                raise ValueError(f"prepared development file changed during comparison: {filename}")
    provenance = _provenance(torch.device("cpu"))
    provenance.update(runner_sha256=sha256_file(Path(__file__)), blas_thread_limit=thread_limit,
                      helper_files_sha256={path: sha256_file(ROOT / path) for path in HELPERS})
    names = {
        "conditioning": ("raw_conditional_bce96", "raw_unconditional_bce96"),
        "candidate_reference": (f"calibrated_{candidate_kind}96", "calibrated_bce96"),
        "resolution_raw": ("raw_bce24", "hourly_sum_raw_bce96"),
        "resolution_calibrated": ("calibrated_bce24", "hourly_sum_calibrated_bce96"),
    }[contrast]
    return {"schema_version": 1, "split": "val", "test_used": False,
            "scope": "Exploratory per-seed declared-reference contrast; no final TEST or new method selection.",
            "contrast": contrast, "training_seed": training_seed,
            "candidate_kind": candidate_kind, "candidate": first, "comparator": second,
            "method_names": dict(zip(("candidate", "comparator"), names)),
            "difference_definition": f"W1(real, {names[0]}) - W1(real, {names[1]})",
            "negative_difference_favors": names[0],
            "source_sha256": parent[2]["source_sha256"], "target": "net", "units": "kWh/interval",
            "resolution": width, "interval_hours": hours,
            "prepared_data_hashes": {"parent96": parent[4], **({"derived24": child[4]} if resolution else {})},
            "preparation_manifests": manifests, "row_bindings": row_bindings,
            "curation_metadata": parent[2].get("curation"),
            "data_curation_status": "verified_curation_v2" if not resolution else "verified_hourly_derivation_v1",
            "derived_provenance": child[2].get("derived") if resolution else None,
            "training_subsets_match": True, "canonical_full_train_selection_verified": True,
            "training_completion_qualification": "Checkpoint configuration and exact rows verified; full-run completion is checked by the study collector, not inferred from best.pt.",
            "source_binding": source_binding, "saved_maps": maps, "pipelines": pipelines,
            "validation_selection": {
                "seed": DRAW_SEED, "samples_per_condition_cap": CAP,
                "available_profiles": len(val["x"]), "selected_profiles": len(selected),
                "selected_meters": len(np.unique(meters)),
                "selected_indices_sha256": _digest(selected, "<i8"),
                "selected_meter_ids_sha256": _string_digest(meters), "selected_dates_sha256": _string_digest(dates),
                "labels_sha256": _digest(labels, "<i8"), "conditions": conditions.tolist(),
                "condition_counts": {str(label): int(np.sum(labels == label)) for label in conditions},
                "policy": "One sorted-condition parent96 selection, uniform without replacement; verified hourly identities retain the same rows."},
            "seeds": {"validation_selection": DRAW_SEED, "generation_both_methods": DRAW_SEED,
                      "meter_bootstrap": DRAW_SEED, "calibration_reference": 7301 if calibrated else None},
            "frozen_samples": {"real_profiles_sha256": _digest(real, "<f8"),
                               "labels_sha256": _digest(labels, "<i8"), **frozen,
                               "hash_encoding": "Profiles: little-endian float64 C-order; labels/indices: little-endian int64; identities: compact UTF-8 JSON.",
                               "pairing": "Identical real-meter multiplicities across both methods and every condition; no synthetic-to-real day match or common latent realization implied."},
            "methods": methods, "metrics": list(METRICS), "replicates": REPLICATES,
            "paired_comparison": paired,
            "interpretation": (
                "All intervals are descriptive marginal percentile intervals, not simultaneous or significance claims. "
                "The five metrics and per-condition intervals use the same 500 technology-stratified whole-meter "
                "draws, conditional on the selected validation days, frozen checkpoints, maps and generated arrays. "
                "Missing-condition draws withhold all bounds. Uncertainty excludes training, model selection, "
                "calibration estimation, synthetic Monte Carlo, validation subsampling and future periods. "
                "Do not average these per-seed intervals or pool seeds as independent meters. "
                "Marginal improvements supplied by fixed maps are not independent learned-dependence evidence. "
                "Identity curation does not establish independent households; reused validation is exploratory."),
            "provenance": provenance}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrast", choices=CONTRASTS, required=True)
    parser.add_argument("--prepared96", required=True)
    parser.add_argument("--prepared24")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--comparator", required=True)
    parser.add_argument("--candidate-map")
    parser.add_argument("--comparator-map")
    parser.add_argument("--candidate-kind", choices=("swd10", "capacity"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new path: {output}")
    report = compare(args.prepared96, args.candidate, args.comparator, contrast=args.contrast,
                     prepared24=args.prepared24, candidate_map=args.candidate_map,
                     comparator_map=args.comparator_map, candidate_kind=args.candidate_kind)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    print(f"Wrote validation-only reference uncertainty: {output}")


if __name__ == "__main__":
    main()
