"""Evaluate a training-subset-matched Gaussian copula on validation data only.

Example:
    python scripts/evaluate_copula.py --prepared data/prepared96 \
        --reference-checkpoint runs/screen_standard/seed_42/best.pt \
        --output runs/copula_screen/validation_report.json

The reference GAN checkpoint binds the prepared data and actual training rows;
its weights are not used. Exact empirical linear marginal quantiles are the
reported-study default. Set --quantile-knots 257 to request a compressed margin
approximation explicitly. This script has no test-split option.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
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
from fluvius_cgan.copula import _settings, fit_conditional_copula
from fluvius_cgan.evaluation import evaluate_conditional, nearest_neighbor_diagnostics
from fluvius_cgan.release import sha256_file
from fluvius_cgan.training import _load_prepared, _provenance


def evaluate(prepared, reference_checkpoint, *, samples_per_condition=100, seed=2026,
             quantile_knots=None, model_output=None):
    """Return comparable validation metrics without opening or hashing test data.

    Training selection is reconstructed and verified against the checkpoint's
    selected-index hash. Validation selection matches cli.evaluate_checkpoint
    and compare_checkpoints: sorted conditions, uniform without replacement,
    the same cap and seed. The copula is fit on physical-unit training rows only.
    """
    for name, value, minimum in (("samples_per_condition", samples_per_condition, 1), ("seed", seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if seed >= 2 ** 32:
        raise ValueError("seed must be smaller than 2**32")
    _, quantile_knots, shrinkage = _settings(64, quantile_knots, .01)
    if model_output is not None:
        model_output = Path(model_output)
        if model_output.suffix.lower() != ".npz":
            raise ValueError("model_output must have a .npz extension")
        if model_output.exists():
            raise FileExistsError(f"model output already exists: {model_output}")

    started = time.perf_counter()
    train, val, metadata, scaler, hashes = _load_prepared(prepared)
    _validate_manifest(metadata, hashes, train, val)
    reference_details, train_indices = _checkpoint(reference_checkpoint, train, metadata, scaler, hashes)
    conditions = np.unique(val["y"])
    missing = sorted(set(conditions.tolist()) - set(reference_details["supported_labels"]))
    if missing:
        raise ValueError(f"reference training subset lacks observed validation conditions: {missing}")
    rng = np.random.default_rng(seed)
    selected = np.concatenate([
        rng.choice(np.flatnonzero(val["y"] == label),
                   min(samples_per_condition, int(np.sum(val["y"] == label))), replace=False)
        for label in conditions
    ])
    real, labels, meter_ids = (val[key][selected] for key in ("x", "y", "meter_id"))
    tx, ty = train["x"][train_indices], train["y"][train_indices]
    loaded_at = time.perf_counter()

    # Avoid small dense matrix operations spawning an excessive BLAS thread pool.
    # This optional dependency is not needed for the algorithm or installation.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        thread_context, thread_limit = nullcontext(), None
    else:
        thread_context, thread_limit = threadpool_limits(limits=1, user_api="blas"), 1
    with thread_context:
        fit_started = time.perf_counter()
        model = fit_conditional_copula(tx, ty, quantile_knots=quantile_knots, shrinkage=shrinkage)
        fitted_at = time.perf_counter()
        generated = model.generate(labels, seed=int(seed), batch_size=4096)
        generated_at = time.perf_counter()
    metrics = evaluate_conditional(real, labels, generated, labels,
                                   interval_hours=24 / metadata["resolution"])
    evaluated_at = time.perf_counter()
    neighbors = nearest_neighbor_diagnostics(tx, generated, real, seed=int(seed))
    neighbors_at = time.perf_counter()
    if sha256_file(reference_checkpoint) != reference_details["sha256"]:
        raise ValueError("reference checkpoint changed during copula evaluation")

    artifact = None
    if model_output is not None:
        model_output.parent.mkdir(parents=True, exist_ok=True)
        if model_output.exists():
            raise FileExistsError(f"model output already exists: {model_output}")
        model.save(model_output)
        artifact = {"path": str(model_output), "sha256": sha256_file(model_output),
                    "format": "ConditionalGaussianCopula NPZ schema version 1; no pickle"}
    saved_at = time.perf_counter()
    provenance = _provenance(torch.device("cpu"))
    provenance["scipy"] = str(scipy.__version__)
    provenance["evaluation_script_sha256"] = sha256_file(Path(__file__))
    provenance["comparison_helper_sha256"] = sha256_file(ROOT / "scripts" / "compare_checkpoints.py")
    provenance["blas_thread_limit"] = thread_limit
    model_bytes = sum(array.nbytes for fitted in model.models.values()
                      for array in (fitted.quantiles, fitted.correlation, fitted.cholesky))
    return {
        "schema_version": 1,
        "split": "val",
        "test_used": False,
        "scope": "Exploratory validation-only statistical baseline; not a final test evaluation.",
        "source_sha256": metadata["source_sha256"],
        "prepared_data_hashes": hashes,
        "target": metadata["target"],
        "units": metadata["units"],
        "resolution": metadata["resolution"],
        "interval_hours": 24 / metadata["resolution"],
        "reference_checkpoint": reference_details,
        "training_profiles": {"prepared": len(train["x"]), "model": len(tx), "baseline": len(tx)},
        "training_subset_source": "Reference checkpoint's exact verified training subset, in physical units.",
        "seed": int(seed),
        "seeds": {"validation_selection": int(seed), "copula_generation": int(seed),
                  "nearest_neighbor_subsampling": int(seed)},
        "copula": {
            "method": "conditional_gaussian_copula",
            "quantile_knots": quantile_knots,
            "marginals": ("All sorted training observations with linear inverse quantiles."
                          if quantile_knots is None else
                          "Uniform quantile knots with linear interpolation; compressed approximation including endpoints."),
            "normal_scores": "ndtri((average_rank - 0.5) / condition_training_count)",
            "dependence": "Full normal-score sample correlation per condition, shrunk toward identity.",
            "shrinkage": shrinkage,
            "regularization_selection": "Fixed a priori; not tuned on validation targets.",
            "psd_repair": "Symmetrize, eigenvalue floor 1e-10, unit-diagonal renormalization.",
            "tail_policy": "No extrapolation beyond each condition/feature's observed training range.",
            "constant_columns": "Constant empirical inverse; zero latent cross-correlations and unit latent variance.",
            "supported_labels": model.supported_labels.tolist(),
            "train_counts": {str(label): count for label, count in model.train_counts.items()},
            "generation_batch_size": 4096,
            "parameter_array_bytes": model_bytes,
            "saved_model": artifact,
        },
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
        },
        "frozen_samples": {
            "hash_encoding": "Profiles: little-endian float64 C-order bytes; labels: little-endian int64 C-order bytes.",
            "profile_shape": list(real.shape),
            "labels_sha256": _digest(labels, "<i8"),
            "real_profiles_sha256": _digest(real, "<f8"),
            "generated_profiles_sha256": _digest(generated, "<f8"),
            "postprocessing": "Original-unit inverse empirical margins; no additional clipping or scaling.",
        },
        "methods": {"conditional_gaussian_copula": metrics},
        "nearest_neighbors": neighbors,
        "raw_output": {
            "minimum_kwh": float(generated.min()), "maximum_kwh": float(generated.max()),
            "negative_fraction": float(np.mean(generated < 0)),
            "negative_values_valid_for_target": metadata["target"] == "net",
            "post_generation_clipping_applied": False,
        },
        "timings_seconds": {
            "load_and_verify": loaded_at - started,
            "fit": fitted_at - fit_started,
            "generate": generated_at - fitted_at,
            "evaluate": evaluated_at - generated_at,
            "nearest_neighbors": neighbors_at - evaluated_at,
            "save_model_and_final_checkpoint_check": saved_at - neighbors_at,
            "total_before_report_serialization": time.perf_counter() - started,
        },
        "provenance": provenance,
        "interpretation": (
            "Fresh joint draws from a statistical comparator, not row resampling or evidence of GAN novelty. "
            "Linear empirical inverse quantiles smooth atoms; midrank Gaussian scores approximate tied/discrete margins. "
            "Compressed quantile knots can additionally distort sparse tails. Gaussian dependence cannot capture "
            "arbitrary joint multimodality or tail dependence, and empirical tails do not extrapolate. "
            "Metrics are descriptive validation diagnostics; nearest-neighbor checks provide no privacy guarantee. "
            "Repeated screening on validation precludes a confirmatory significance claim."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New validation JSON report")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples-per-condition", type=int, default=100)
    parser.add_argument("--quantile-knots", type=int, default=None,
                        help="Optional marginal compression (e.g. 257); omitted means exact sorted margins")
    parser.add_argument("--model-output", help="Optional new NPZ file containing fitted copula parameters")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new report path: {output}")
    if output.suffix.lower() != ".json":
        raise ValueError("output must have a .json extension")
    if args.model_output is not None and output.resolve() == Path(args.model_output).resolve():
        raise ValueError("report and model output paths must differ")
    report = evaluate(args.prepared, args.reference_checkpoint, seed=args.seed,
                      samples_per_condition=args.samples_per_condition,
                      quantile_knots=args.quantile_knots, model_output=args.model_output)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(f"Wrote validation-only Gaussian-copula report: {output}")
    print(json.dumps({"macro": report["methods"]["conditional_gaussian_copula"]["macro"],
                      "timings_seconds": report["timings_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
