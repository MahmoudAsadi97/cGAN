"""Compare a frozen, provenance-bound calibrated cGAN on validation only.

The Gaussian-copula primary comparator uses the calibrator's quantile-knot
setting. ``exact_marginal_copula`` is a separately named sensitivity comparator.
No map is fit here, no current-batch ranks are used, and test.npz is never read.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_checkpoints import _checkpoint, _digest, _validate_manifest
from evaluate_calibration import _preparation_binding, _check_preparation_unchanged
from fluvius_cgan.calibration import load_verified_calibrator
from fluvius_cgan.copula import fit_conditional_copula
from fluvius_cgan.evaluation import evaluate_conditional, resample_baseline
from fluvius_cgan.release import sha256_file
from fluvius_cgan.training import _load_prepared, _provenance, generate
from fluvius_cgan.uncertainty import paired_condition_macro_meter_bootstrap


BASELINES = ("gaussian_copula", "exact_marginal_copula", "conditional_resampling", "raw_cgan")
METRICS = (
    "pointwise_wasserstein_kwh", "daily_energy_wasserstein_kwh",
    "peak_power_wasserstein_kw", "ramp_wasserstein_kw", "lag_one_acf_wasserstein",
)


def compare(prepared, checkpoint, calibrator, *, baseline="gaussian_copula",
            samples_per_condition=100, seed=2026, replicates=500):
    """Return full metrics and paired real-meter uncertainty for fixed arrays.

    Both methods use the same selected real conditions and the checkpoint's exact
    training subset. Meter draws are paired across methods and all conditions,
    with fixed counts within technology strata. Condition CIs are marginal, not
    simultaneous. The calibrator is loaded and verified, never fitted or updated.
    """
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {BASELINES}")
    for name, value, minimum in (("samples_per_condition", samples_per_condition, 1),
                                 ("seed", seed, 0), ("replicates", replicates, 2)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if seed >= 2 ** 32:
        raise ValueError("seed must be smaller than 2**32")

    train, val, metadata, scaler, hashes = _load_prepared(prepared)
    _validate_manifest(metadata, hashes, train, val)
    checkpoint_hashes, qualification = _preparation_binding(prepared, metadata, hashes)
    details, indices = _checkpoint(checkpoint, train, metadata, scaler, checkpoint_hashes)
    tx, ty = train["x"][indices], train["y"][indices]
    calibration, calibration_details = load_verified_calibrator(
        calibrator, checkpoint_sha256=details["sha256"],
        train_indices_sha256=details["training_selection"]["train_indices_sha256"],
        prepared_data_hashes=hashes, generation_seed=int(seed),
        n_features=metadata["resolution"], train_labels=ty,
    )
    conditions = np.unique(val["y"])
    missing = sorted(set(conditions.tolist()) - set(details["supported_labels"]))
    if missing:
        raise ValueError(f"checkpoint lacks observed validation conditions: {missing}")
    rng = np.random.default_rng(seed)
    selected = np.concatenate([
        rng.choice(np.flatnonzero(val["y"] == label),
                   min(samples_per_condition, int(np.sum(val["y"] == label))), replace=False)
        for label in conditions
    ])
    real, labels, meter_ids = (val[key][selected] for key in ("x", "y", "meter_id"))
    raw = generate(checkpoint, labels, seed=int(seed))
    if sha256_file(checkpoint) != details["sha256"]:
        raise ValueError("checkpoint changed between verification and generation")
    generated = calibration.transform(raw, labels)

    comparator_details = {
        "kind": baseline,
        "training_selection": details["training_selection"].copy(),
        "training_subset_source": "Checkpoint's exact verified training rows",
        "seed": int(seed),
    }
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        thread_context, thread_limit = nullcontext(), None
    else:
        thread_context, thread_limit = threadpool_limits(limits=1, user_api="blas"), 1
    with thread_context:
        if baseline in ("gaussian_copula", "exact_marginal_copula"):
            knots = calibration.quantile_knots if baseline == "gaussian_copula" else None
            model = fit_conditional_copula(tx, ty, quantile_knots=knots, shrinkage=.01)
            reference = model.generate(labels, seed=int(seed))
            comparator_details.update({
                "quantile_knots": knots, "shrinkage": .01,
                "quantile_setting_matches_calibrator": knots == calibration.quantile_knots,
                "marginals": "All training order statistics with linear inverse-quantile interpolation; not a discrete empirical CDF"
                             if knots is None else "Uniform quantile knots with linear inverse-quantile interpolation",
                "role": "Primary marginal-setting-matched control" if baseline == "gaussian_copula"
                        else "Secondary exact-marginal sensitivity control",
            })
        elif baseline == "conditional_resampling":
            reference = resample_baseline(tx, ty, labels, seed=int(seed))
        else:
            # Share the exact pre-calibration draws to isolate the fixed mapping.
            reference = raw
    hours = 24 / metadata["resolution"]
    inference = paired_condition_macro_meter_bootstrap(
        real, labels, generated, labels, reference, labels, meter_ids,
        interval_hours=hours, n_bootstrap=int(replicates), seed=int(seed),
        real_meter_strata=labels // 8, metrics=METRICS, return_per_condition=True,
    )
    methods = {
        "calibrated_cgan": evaluate_conditional(real, labels, generated, labels, interval_hours=hours),
        baseline: evaluate_conditional(real, labels, reference, labels, interval_hours=hours),
    }
    if sha256_file(checkpoint) != details["sha256"]:
        raise ValueError("checkpoint changed during comparison")
    if sha256_file(calibrator) != calibration_details["sha256"]:
        raise ValueError("calibrator changed during comparison")
    _check_preparation_unchanged(prepared, qualification)
    provenance = _provenance(torch.device("cpu"))
    provenance.update({"comparison_script_sha256": sha256_file(Path(__file__)),
                       "comparison_helper_sha256": sha256_file(ROOT / "scripts" / "compare_checkpoints.py"),
                       "preparation_helper_sha256": sha256_file(ROOT / "scripts" / "evaluate_calibration.py"),
                       "blas_thread_limit": thread_limit})
    return {
        "schema_version": 1, "split": "val", "test_used": False,
        "scope": "Exploratory validation-only paired comparison; not a final test or significance claim.",
        "source_sha256": metadata["source_sha256"], "prepared_data_hashes": hashes,
        "target": metadata["target"], "units": metadata["units"],
        "resolution": metadata["resolution"], "interval_hours": hours,
        **qualification,
        "identity_qualification": (
            "This command does not repeat identity auditing. Exact-trace curation does not prove "
            "independent households or exclude near-duplicates. Earlier original-split results "
            "contained cross-split exact trace aliases; uncurated results remain exploratory diagnostics."
        ),
        "candidate": details, "calibrator": calibration_details, "comparator": comparator_details,
        "training_subsets_match": True,
        "methods": methods, "paired_comparison": inference,
        "seed": int(seed), "replicates": int(replicates),
        "seeds": {"validation_selection": int(seed), "candidate_generation": int(seed),
                  "comparator_generation_or_resampling": int(seed), "meter_bootstrap": int(seed),
                  "calibration_reference": calibration.provenance["calibration_seed"]},
        "validation_selection": {
            "policy": "Uniform without replacement within every observed condition; matches cli.evaluate_checkpoint.",
            "samples_per_condition_cap": int(samples_per_condition),
            "available_profiles": len(val["x"]), "available_meters": len(np.unique(val["meter_id"])),
            "selected_profiles": len(real), "selected_meters": len(np.unique(meter_ids)),
            "selected_indices_sha256": _digest(selected, "<i8"),
            "conditions": conditions.tolist(),
            "condition_counts": {str(label): int(np.sum(labels == label)) for label in conditions},
            "bootstrap_strata": "Technology = condition label // 8; preserve selected meter counts per group.",
            "cluster_scope": "All selected days of each sampled real meter; conditional on frozen validation subsample.",
        },
        "frozen_samples": {
            "hash_encoding": "Profiles: little-endian float64 C-order; labels/row indices: little-endian int64 C-order.",
            "profile_shape": list(real.shape), "labels_sha256": _digest(labels, "<i8"),
            "real_profiles_sha256": _digest(real, "<f8"),
            "raw_profiles_sha256": _digest(raw, "<f8"),
            "candidate_profiles_sha256": _digest(generated, "<f8"),
            "comparator_profiles_sha256": _digest(reference, "<f8"),
            "pairing": "Shared real-meter draws across methods and conditions; no synthetic-to-real day correspondence.",
            "postprocessing": "Candidate uses the previously fitted fixed conditional marginal map; no application-batch ranks.",
        },
        "interpretation": (
            "Differences are calibrated cGAN minus comparator; negative values favor calibrated cGAN. "
            "Marginal improvements are mechanically imposed and cannot independently validate learned dependence. "
            "Intervals are descriptive, marginal (not simultaneous), and conditional on frozen models, maps, "
            "generated arrays and validation rows. They exclude calibration-fit/seed, training-seed, synthetic "
            "Monte Carlo, validation-subsampling, model-selection and future-period uncertainty. Nominal coverage "
            "is not guaranteed near coincident CDFs. Validation reuse precludes confirmatory significance claims."
        ),
        "provenance": provenance,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--baseline", choices=BASELINES, default="gaussian_copula")
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--samples-per-condition", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new report path: {output}")
    report = compare(args.prepared, args.checkpoint, args.calibrator, baseline=args.baseline,
                     samples_per_condition=args.samples_per_condition, seed=args.seed, replicates=args.replicates)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    print(f"Wrote validation-only calibrated comparison: {output}")


if __name__ == "__main__":
    main()
