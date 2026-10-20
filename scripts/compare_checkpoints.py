"""Compare frozen generators or a training-only baseline on validation data only.

Example:
    python scripts/compare_checkpoints.py --prepared data/prepared96 \
        --candidate runs/candidate/best.pt --comparator runs/pilot/best.pt \
        --output reports/new_paired_comparison.json

Omit --comparator to use conditional_resampling, or select --baseline
conditional_mean. This command has no test-split option and never opens test.npz.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluvius_cgan.evaluation import mean_baseline, resample_baseline
from fluvius_cgan.release import sha256_file
from fluvius_cgan.training import CHECKPOINT_FORMAT, _config, _load_prepared, generate
from fluvius_cgan.uncertainty import paired_condition_macro_meter_bootstrap


def _digest(values, dtype):
    """Hash C-order bytes in an explicit, little-endian numeric representation."""
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def _validate_manifest(metadata, hashes, train, val):
    for split, data in (("train", train), ("val", val)):
        recorded = metadata["splits"][split]
        if recorded.get("sha256") != hashes[f"{split}.npz"]:
            raise ValueError(f"{split} fingerprint differs from the prepared manifest")
        if "meter_ids" in recorded and set(recorded["meter_ids"]) != set(data["meter_id"]):
            raise ValueError(f"{split} meter IDs disagree with the prepared manifest")


def _checkpoint(path, train, metadata, scaler, hashes):
    """Verify data binding and reconstruct the exact training row selection."""
    path = Path(path)
    digest = sha256_file(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if sha256_file(path) != digest:
        raise ValueError("checkpoint changed while loading")
    if not isinstance(state, dict) or state.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint format")
    for name, actual in hashes.items():
        if state.get("data_hashes", {}).get(name) != actual:
            raise ValueError(f"checkpoint data fingerprint does not match prepared {name}")
    if state.get("metadata") != metadata or state.get("output_dim") != metadata["resolution"]:
        raise ValueError("checkpoint metadata or resolution disagrees with prepared data")
    if any(state.get("scaler", {}).get(key) != value for key, value in scaler.items()):
        raise ValueError("checkpoint scaler disagrees with prepared training scaler")
    # Validate CPU-compatible inference configuration even for a CUDA-trained model.
    config = _config({**state["config"], "device": "cpu"})
    selection = state.get("selection", {})
    available, used = len(train["x"]), selection.get("train_profiles_used")
    if (not isinstance(used, int) or isinstance(used, bool) or not 0 < used <= available
            or selection.get("train_profiles_available") != available):
        raise ValueError("invalid checkpoint training subset size")
    expected_count = min(available, config["max_train_samples"] or available)
    if used != expected_count:
        raise ValueError("checkpoint training subset size disagrees with configuration")
    indices = np.arange(available, dtype=np.int64)
    if used < available:
        indices = np.sort(np.random.default_rng(config["seed"]).choice(available, used, replace=False))
    if _digest(indices, "<i8") != selection.get("train_indices_sha256"):
        raise ValueError("cannot reconstruct the checkpoint's exact training subset")
    support = np.unique(train["y"][indices]).tolist()
    if support != sorted(state.get("supported_labels", [])):
        raise ValueError("checkpoint label support disagrees with its actual training subset")
    details = {
        "kind": "checkpoint",
        "path": str(path),
        "sha256": digest,
        "epoch": state["epoch"],
        "config": state["config"],
        "training_provenance": state.get("provenance"),
        "data_hashes": state["data_hashes"],
        "supported_labels": support,
        "training_selection": {
            "train_profiles_available": available,
            "train_profiles_used": used,
            "train_indices_sha256": selection["train_indices_sha256"],
            "training_seed": config["seed"],
            "max_train_samples": config["max_train_samples"],
            "reconstruction": "Training seed and count; exact selected-index SHA-256 verified.",
        },
        "checkpoint_selection": {
            "validation_metric": state.get("validation_metric"),
            "validation_score": state.get("validation_score"),
            "best_epoch": state.get("best_epoch"),
            "best_validation_score": state.get("best_validation_score"),
        },
    }
    return details, indices


def _generate_verified(path, details, labels, seed):
    values = generate(path, labels, seed=seed)
    if sha256_file(path) != details["sha256"]:
        raise ValueError("checkpoint changed between verification and generation")
    return values


def compare(prepared, candidate, *, comparator=None, baseline=None,
            samples_per_condition=100, seed=2026, bootstrap=200):
    """Return a validation-only paired comparison of fully frozen profile arrays.

    Real row selection matches ``cli.evaluate_checkpoint``: uniform without
    replacement within every observed validation condition, in sorted condition
    order, using the requested cap and seed. Unsupported conditions cause failure;
    neither checkpoint may silently narrow the common validation target.

    Cluster resampling retains all *selected* days of each selected real meter,
    preserving the selected meter count within each technology stratum (label//8).
    Consequently intervals are conditional on this validation subsample as well as
    the frozen generated/baseline samples. Condition labels determine empirical
    distributions only; equal row counts do not pair fake profiles with real days.
    Validation is used for model development, so these intervals are exploratory
    and do not correct for checkpoint selection or repeated candidate screening.
    """
    for name, value, minimum in (("samples_per_condition", samples_per_condition, 1),
                                 ("bootstrap", bootstrap, 2), ("seed", seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if seed >= 2 ** 32:
        raise ValueError("seed must be smaller than 2**32")
    if comparator is not None and baseline is not None:
        raise ValueError("choose either a comparator checkpoint or a baseline")
    baseline = baseline or "conditional_resampling"
    if baseline not in ("conditional_resampling", "conditional_mean"):
        raise ValueError("unknown baseline")

    # The existing loader reads only train/val and verifies physical units,
    # disjoint meters, counts, labels, dates, and the training-fitted scaler.
    train, val, metadata, scaler, hashes = _load_prepared(prepared)
    _validate_manifest(metadata, hashes, train, val)
    candidate_details, train_indices = _checkpoint(candidate, train, metadata, scaler, hashes)
    comparator_details = None
    if comparator is not None:
        comparator_details, _ = _checkpoint(comparator, train, metadata, scaler, hashes)

    conditions = np.unique(val["y"])
    for name, details in (("candidate", candidate_details), ("comparator", comparator_details)):
        if details is not None:
            missing = sorted(set(conditions.tolist()) - set(details["supported_labels"]))
            if missing:
                raise ValueError(f"{name} lacks observed validation conditions: {missing}")
    rng = np.random.default_rng(seed)
    selected = np.concatenate([
        rng.choice(np.flatnonzero(val["y"] == label),
                   min(samples_per_condition, int(np.sum(val["y"] == label))), replace=False)
        for label in conditions
    ])
    real, labels, meter_ids = (val[key][selected] for key in ("x", "y", "meter_id"))
    generated = _generate_verified(candidate, candidate_details, labels, seed)
    if comparator is not None:
        reference = _generate_verified(comparator, comparator_details, labels, seed)
    else:
        tx, ty = train["x"][train_indices], train["y"][train_indices]
        reference = (resample_baseline(tx, ty, labels, seed=seed)
                     if baseline == "conditional_resampling" else mean_baseline(tx, ty, labels))
        comparator_details = {
            "kind": baseline,
            "training_subset_source": "candidate checkpoint's exact verified training subset",
            "training_selection": candidate_details["training_selection"].copy(),
            "seed": int(seed) if baseline == "conditional_resampling" else None,
        }
    same_subset = (candidate_details["training_selection"]["train_indices_sha256"]
                   == comparator_details["training_selection"]["train_indices_sha256"])
    inference = paired_condition_macro_meter_bootstrap(
        real, labels, generated, labels, reference, labels, meter_ids,
        interval_hours=24 / metadata["resolution"], n_bootstrap=bootstrap, seed=seed,
        real_meter_strata=labels // 8,
    )
    return {
        "schema_version": 1,
        "split": "val",
        "test_used": False,
        "scope": "Exploratory validation-only comparison; not a final test evaluation.",
        "source_sha256": metadata["source_sha256"],
        "prepared_data_hashes": hashes,
        "target": metadata["target"],
        "units": metadata["units"],
        "resolution": metadata["resolution"],
        "interval_hours": 24 / metadata["resolution"],
        "candidate": candidate_details,
        "comparator": comparator_details,
        "training_subsets_match": same_subset,
        "training_subset_interpretation": (
            "Both methods use the same verified training rows." if same_subset else
            "Checkpoint training subsets differ; the comparison jointly reflects model and training-data differences."
        ),
        "seed": int(seed),
        "seeds": {"validation_selection": int(seed), "candidate_generation": int(seed),
                  "comparator_generation_or_resampling": int(seed) if comparator is not None or baseline == "conditional_resampling" else None,
                  "meter_bootstrap": int(seed)},
        "validation_selection": {
            "policy": "Uniform without replacement within every observed condition; matches cli.evaluate_checkpoint.",
            "samples_per_condition_cap": int(samples_per_condition),
            "available_profiles": len(val["x"]),
            "available_meters": len(np.unique(val["meter_id"])),
            "selected_profiles": len(real),
            "selected_meters": len(np.unique(meter_ids)),
            "selected_indices_sha256": _digest(selected, "<i8"),
            "index_hash_encoding": "Zero-based row indices as little-endian int64 C-order bytes, in evaluation order.",
            "conditions": conditions.tolist(),
            "condition_counts": {str(label): int(np.sum(labels == label)) for label in conditions},
            "cluster_scope": "All selected days per sampled meter; conditional on this frozen validation subsample.",
            "bootstrap_strata": "Technology group = condition label // 8; preserve selected meter count per group.",
        },
        "frozen_samples": {
            "hash_encoding": "Profiles: little-endian float64 C-order bytes; labels: little-endian int64 C-order bytes.",
            "profile_shape": list(real.shape),
            "labels_sha256": _digest(labels, "<i8"),
            "real_profiles_sha256": _digest(real, "<f8"),
            "candidate_profiles_sha256": _digest(generated, "<f8"),
            "comparator_profiles_sha256": _digest(reference, "<f8"),
            "pairing": "Shared real-meter draws and condition distributions only; no synthetic-to-real day correspondence.",
            "postprocessing": "None; original-unit raw model/baseline outputs, no clipping.",
        },
        "paired_comparison": inference,
        "interpretation": (
            "Differences are candidate minus comparator; negative values favor the candidate. "
            "Descriptive marginal percentile intervals are conditional on frozen reference arrays and the selected real rows. "
            "Nominal coverage is not guaranteed, especially near coincident CDFs where W1 is nonsmooth. "
            "They exclude model-selection, training-seed, generated-sample, comparator-sample, "
            "validation-subsampling, and future-period uncertainty. Validation reuse and repeated "
            "candidate screening preclude a confirmatory significance claim."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--candidate", required=True)
    alternatives = parser.add_mutually_exclusive_group()
    alternatives.add_argument("--comparator", help="Comparator checkpoint from the same prepared train/val data")
    alternatives.add_argument("--baseline", choices=("conditional_resampling", "conditional_mean"),
                              help="Training-only comparator; default: conditional_resampling")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-condition", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new report path: {output}")
    result = compare(args.prepared, args.candidate, comparator=args.comparator, baseline=args.baseline,
                     samples_per_condition=args.samples_per_condition, seed=args.seed, bootstrap=args.bootstrap)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    print(f"Wrote validation-only paired comparison: {output}")


if __name__ == "__main__":
    main()
