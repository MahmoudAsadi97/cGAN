"""Diagnose a checkpoint on validation data; never open the test split.

Example:
    python scripts/diagnose_checkpoint.py --prepared data/prepared96 \
        --checkpoint runs/pilot_full_seed42/best.pt \
        --output reports/new_diagnosis.json

Mean replacement and smoothing are diagnostic counterfactuals, not fitted,
validated, or deployable models. Resampling repetitions describe Monte Carlo
variation conditional on these datasets, not independent-meter confidence bands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import wasserstein_distance
import torch

# Also permit execution directly from an uninstalled source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluvius_cgan.evaluation import _acf, evaluate_conditional, resample_baseline
from fluvius_cgan.release import sha256_file
from fluvius_cgan.training import _load_prepared, _scaler_parameters, generate


METRICS = (
    "pointwise_wasserstein_kwh", "daily_energy_wasserstein_kwh",
    "peak_power_wasserstein_kw", "ramp_wasserstein_kw",
    "mean_profile_rmse_kwh", "pointwise_variance_rmse_kwh2", "diversity_ratio",
    "generated_mean_daily_energy_kwh", "generated_mean_peak_power_kw",
    "generated_mean_pointwise_variance_kwh2",
)
SAMPLING_METRICS = METRICS[:6]


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def _correlation(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _sample_indices(labels, conditions, cap, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.choice(indices, min(cap, len(indices)), replace=False)
        for label in conditions
        for indices in [np.flatnonzero(labels == label)]
    ])


def _conditional_means(values, labels, conditions):
    return np.stack([values[labels == label].mean(axis=0, dtype=np.float64)
                     for label in conditions])


def _metric(real, labels, generated, interval_hours):
    result = evaluate_conditional(real, labels, generated, labels, interval_hours)["macro"]
    return {**{key: result[key] for key in METRICS},
            "acf1_mean": result["autocorrelation"]["1"]["generated_mean"],
            "acf1_wasserstein": result["autocorrelation"]["1"]["wasserstein"]}


def _temporal(values, interval_hours):
    centered = values - values.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    ramps = np.diff(values, axis=1) / interval_hours
    return {
        "mean_daily_energy": float(values.sum(axis=1).mean()),
        "std_daily_energy": float(values.sum(axis=1).std()),
        "mean_row_std": float(values.std(axis=1).mean()),
        "ramp_rms_kw": float(np.sqrt(np.mean(ramps ** 2))),
        "ramp_abs_mean_kw": float(np.abs(ramps).mean()),
        "high_frequency_fraction_above_12_cycles_per_day":
            _ratio(power[:, 13:].sum(), power[:, 1:].sum()),
        "negative_fraction": float((values < 0).mean()),
        "acf1": float(_acf(values, 1).mean()),
    }


def _condition_spread(means):
    centered = means - means.mean(axis=0, keepdims=True)
    return {
        "between_condition_mean_profile_rms_kwh": float(np.sqrt(np.mean(centered ** 2))),
        "daily_energy_condition_std_kwh": float(means.sum(axis=1).std()),
    }


def _summarize(rows):
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
            "min": float(min(row[key] for row in rows)),
            "max": float(max(row[key] for row in rows)),
        }
        for key in SAMPLING_METRICS
    }


def diagnose(prepared, checkpoint, *, samples_per_condition=100, seed=2026, replicates=5):
    """Use the checkpoint's actual training subset and supported validation labels."""
    if samples_per_condition < 1 or replicates < 1 or not 0 <= seed < 2 ** 32:
        raise ValueError("samples/replicates must be positive; seed must be in [0, 2**32)")
    prepared, checkpoint = Path(prepared), Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # This existing loader reads and verifies only train.npz and val.npz.
    train, val, metadata, _, data_hashes = _load_prepared(prepared)
    if any(state.get("data_hashes", {}).get(name) != digest
           for name, digest in data_hashes.items()):
        raise ValueError("checkpoint fingerprints do not match prepared train/validation data")
    available = len(train["x"])
    used = state["selection"]["train_profiles_used"]
    if not isinstance(used, int) or isinstance(used, bool) or not 0 < used <= available:
        raise ValueError("invalid checkpoint training subset size")
    selected = np.arange(available, dtype=np.int64)
    if used < available:
        selected = np.sort(np.random.default_rng(state["config"]["seed"]).choice(
            available, used, replace=False))
    subset_hash = hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest()
    if subset_hash != state["selection"]["train_indices_sha256"]:
        raise ValueError("cannot reconstruct the checkpoint's exact training subset")
    tx, ty = train["x"][selected], train["y"][selected]
    support = np.unique(ty)
    if support.tolist() != sorted(state["supported_labels"]):
        raise ValueError("checkpoint label support disagrees with actual training subset")
    vx, vy = val["x"], val["y"]
    conditions = np.intersect1d(support, np.unique(vy))
    if not len(conditions):
        raise ValueError("no validation conditions are supported by the checkpoint")
    indices = _sample_indices(vy, conditions, samples_per_condition, seed)
    real, labels = vx[indices].astype(np.float64), vy[indices]
    fake = generate(checkpoint, labels, seed=seed).astype(np.float64)
    means = _conditional_means(tx, ty, conditions)
    fmeans = _conditional_means(fake, labels, conditions)
    rmeans = _conditional_means(real, labels, conditions)
    vmeans = _conditional_means(vx, vy, conditions)
    positions = np.searchsorted(conditions, labels)
    repeated_fmeans, repeated_means = fmeans[positions], means[positions]
    centered = fake - repeated_fmeans
    corrected = centered + repeated_means
    interval_hours = 24 / real.shape[1]
    offset, scale = _scaler_parameters(state["scaler"])
    result = {
        "scope": "Train/validation only. Counterfactual mean correction uses the generated "
                 "sample's own estimated condition means and actual training-subset means. "
                 "It is a diagnostic, not a fitted or deployable model.",
        "provenance": {
            "source_sha256": metadata["source_sha256"],
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_epoch": state["epoch"],
            "checkpoint_training_provenance": state.get("provenance"),
            "data_hashes": data_hashes,
            "training_profiles": {"prepared": available, "model": used, "diagnostic": len(tx)},
            "train_indices_sha256": subset_hash,
            "training_subset_policy": "Reconstructed from training seed/count; verified against checkpoint index hash.",
            "generator_used_train_subset_record": {
                "train_profiles_available": available, "train_profiles_used": used,
                "train_indices_sha256": subset_hash,
                "training_seed": state["config"]["seed"],
                "max_train_samples": state["config"].get("max_train_samples"),
            },
            "split": "val", "test_used": False,
            "seed": seed, "samples_per_condition_cap": samples_per_condition,
            "replicates": replicates,
            "conditions": conditions.tolist(),
            "counts": {str(label): int(np.sum(labels == label)) for label in conditions},
            "validation_profiles": len(vx),
            "validation_unsupported_profiles": int(np.sum(~np.isin(vy, support))),
            "sampled_validation_profiles": len(real),
            "sampled_validation_meters": len(np.unique(val["meter_id"][indices])),
            "resolution": real.shape[1], "interval_hours": interval_hours,
            "target": metadata["target"], "units": metadata["units"],
        },
        "metrics": {}, "temporal": {}, "normalization": {}, "conditioning": {},
    }
    for name, values in (
        ("real", real), ("fake", fake), ("conditional_fake_mean", repeated_fmeans),
        ("fake_condition_residual", centered), ("fake_train_mean_corrected", corrected),
        ("train_mean", repeated_means),
    ):
        result["temporal"][name] = _temporal(values, interval_hours)
        if name not in ("real", "fake_condition_residual"):
            result["metrics"][name] = _metric(real, labels, values, interval_hours)
    for width in (3, 5):
        padded = np.pad(fake, ((0, 0), (width // 2, width // 2)), mode="reflect")
        smooth = np.stack([np.convolve(row, np.ones(width) / width, mode="valid")
                           for row in padded])
        result["metrics"][f"fake_smooth{width}"] = _metric(real, labels, smooth, interval_hours)
    result["bias"] = {
        "macro_mean_absolute_slot_bias_kwh": float(np.abs(fmeans - rmeans).mean()),
        "macro_mean_absolute_daily_bias_kwh": float(np.abs((fmeans - rmeans).sum(axis=1)).mean()),
        # Repeat condition means to retain empirical weights if caps differ by label.
        "mean_ramp_squared_fraction_in_conditional_mean": _ratio(
            np.mean(np.diff(repeated_fmeans, axis=1) ** 2), np.mean(np.diff(fake, axis=1) ** 2)),
        "mean_within_row_squared_fraction_in_conditional_mean": _ratio(
            np.mean((repeated_fmeans - repeated_fmeans.mean(axis=1, keepdims=True)) ** 2),
            np.mean((fake - fake.mean(axis=1, keepdims=True)) ** 2)),
    }
    quantiles = [0, .001, .01, .05, .25, .5, .75, .95, .99, .999, 1]
    q = np.quantile(tx, quantiles)
    standard_deviation = float(tx.std(dtype=np.float64))
    result["normalization"] = {
        "statistics_scope": "Actual checkpoint training subset; affine transform from checkpoint.",
        "kind": state["scaler"].get("kind", "minmax"),
        "offset": offset, "scale": scale,
        "train_quantiles_kwh": dict(zip(map(str, quantiles), q.tolist())),
        "train_quantiles_normalized": dict(zip(map(str, quantiles), ((q - offset) / scale).tolist())),
        "physical_range": float(q[-1] - q[0]), "zero_normalized": -offset / scale,
        "train_std_kwh": standard_deviation, "train_std_normalized": standard_deviation / scale,
        "middle_90pct_width_normalized": float((q[7] - q[3]) / scale),
    }
    non_pv = (labels // 8) % 2 == 0
    result["conditioning"] = {
        "train": _condition_spread(means), "val_full": _condition_spread(vmeans),
        "fake": _condition_spread(fmeans),
        "fake_real_conditional_energy_correlation": _correlation(fmeans.sum(axis=1), vmeans.sum(axis=1)),
        "fake_real_conditional_shape_correlation": _correlation(
            fmeans - fmeans.mean(axis=0), vmeans - vmeans.mean(axis=0)),
        "negative_fraction_non_pv_real": float((real[non_pv] < 0).mean()) if non_pv.any() else None,
        "negative_fraction_non_pv_fake": float((fake[non_pv] < 0).mean()) if non_pv.any() else None,
        "technology": {
            str(group): {
                "train_daily_mean": float(means[conditions // 8 == group].sum(axis=1).mean()),
                "val_daily_mean": float(vmeans[conditions // 8 == group].sum(axis=1).mean()),
                "fake_daily_mean": float(fmeans[conditions // 8 == group].sum(axis=1).mean()),
            }
            for group in np.unique(conditions // 8)
        },
    }
    baselines, floors, sampling = [], [], []
    for replicate in range(replicates):
        replicate_seed = (seed + replicate) % 2 ** 32
        baseline = resample_baseline(tx, ty, labels, seed=replicate_seed)
        baselines.append(_metric(real, labels, baseline, interval_hours))
        reference = resample_baseline(vx, vy, labels, seed=(replicate_seed + 1000) % 2 ** 32)
        floors.append(_metric(real, labels, reference, interval_hours))
        fresh = _sample_indices(vy, conditions, samples_per_condition, (replicate_seed + 2000) % 2 ** 32)
        macro = evaluate_conditional(vx[fresh], vy[fresh], fake, labels, interval_hours)["macro"]
        sampling.append({key: macro[key] for key in SAMPLING_METRICS})
    result["sampling"] = {
        "replicates": replicates,
        "interpretation": "Descriptive Monte Carlo variability, not confidence intervals. "
                          "Same-validation-population draws may overlap the fixed sample and "
                          "reuse meters; they estimate finite-sample error, not generalization.",
        "train_resample_vs_fixed_validation": _summarize(baselines),
        "same_validation_population_resample_floor": _summarize(floors),
        "fixed_fake_vs_fresh_validation_sample": _summarize(sampling),
    }
    eligible = np.flatnonzero(np.isin(vy, support))
    count = min(len(eligible), state["config"]["validation_samples"])
    scores = []
    for replicate in range(replicates):
        replicate_seed = (seed + 10 + replicate) % 2 ** 32
        indices = np.random.default_rng(replicate_seed).choice(eligible, count, replace=False)
        generated = generate(checkpoint, vy[indices], seed=replicate_seed)
        scores.append(float(np.mean([
            wasserstein_distance(vx[indices, slot], generated[:, slot])
            for slot in range(real.shape[1])
        ]) / scale))
    result["sampling"]["checkpoint_metric_fresh_samples"] = {
        "metric": "mean_interval_wasserstein_normalized", "sample_count": count,
        "scores": scores, "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-condition", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--replicates", type=int, default=5)
    args = parser.parse_args()
    if args.samples_per_condition < 1 or args.replicates < 1 or not 0 <= args.seed < 2 ** 32:
        parser.error("samples/replicates must be positive; seed must be in [0, 2**32)")
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; choose a new report path")
    result = diagnose(args.prepared, args.checkpoint, samples_per_condition=args.samples_per_condition,
                      seed=args.seed, replicates=args.replicates)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote validation-only diagnosis: {output}")


if __name__ == "__main__":
    main()
