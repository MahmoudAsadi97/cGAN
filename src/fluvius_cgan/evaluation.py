"""Distributional evaluation of profiles in their original kWh/interval units.

Rows need not be paired and the real/generated row counts may differ. Daily energy
is a row sum; peak power is max(interval energy)/interval_hours; ramps are signed
within-row power differences. No metric clips negative net import or raw outputs.
The routines describe empirical samples, not a privacy guarantee or a claim of
independent daily observations.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance


_DEFAULT_BOOTSTRAP_METRICS = (
    "pointwise_wasserstein_kwh",
    "daily_energy_wasserstein_kwh",
    "peak_power_wasserstein_kw",
    "ramp_wasserstein_kw",
    "mean_profile_rmse_kwh",
    "pointwise_variance_rmse_kwh2",
)


def _profiles(values, name: str) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonempty numeric rank-2 array") from exc
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a nonempty rank-2 array")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _matching_width(*arrays: np.ndarray) -> None:
    if len({array.shape[1] for array in arrays}) != 1:
        raise ValueError("profile arrays must have the same width")


def _labels(values, name: str, count: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not len(array):
        raise ValueError(f"{name} must be a nonempty rank-1 integer array")
    if count is not None and len(array) != count:
        raise ValueError(f"{name} length must match its profile count")
    if array.dtype.kind not in "iu" or (array < 0).any() or (array >= 64).any():
        raise ValueError(f"{name} must contain integer condition IDs in [0, 63]")
    return array.astype(np.int64, copy=False)


def _interval_hours(value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("interval_hours must be a finite positive number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("interval_hours must be a finite positive number") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("interval_hours must be a finite positive number")
    return value


def _acf(profiles: np.ndarray, lag: int) -> np.ndarray:
    """Biased ACF using each row's full mean and sum of squared deviations.

    A constant profile has ACF=0 by convention; its count is reported separately.
    This convention avoids silently dropping constant or collapsed generations.
    """
    centered = profiles - profiles.mean(axis=1, keepdims=True)
    denominator = np.sum(centered * centered, axis=1)
    numerator = np.sum(centered[:, lag:] * centered[:, :-lag], axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator),
                     where=denominator != 0)


def _evaluate(real: np.ndarray, generated: np.ndarray, hours: float) -> dict:
    real_daily, generated_daily = real.sum(axis=1), generated.sum(axis=1)
    real_peak, generated_peak = real.max(axis=1) / hours, generated.max(axis=1) / hours
    real_variance, generated_variance = real.var(axis=0), generated.var(axis=0)
    real_diversity = float(real_variance.mean())
    generated_diversity = float(generated_variance.mean())
    result = {
        "n_real": len(real),
        "n_generated": len(generated),
        "n_intervals": real.shape[1],
        "interval_hours": hours,
        "pointwise_wasserstein_kwh": float(np.mean([
            wasserstein_distance(real[:, slot], generated[:, slot])
            for slot in range(real.shape[1])
        ])),
        "daily_energy_wasserstein_kwh": float(wasserstein_distance(real_daily, generated_daily)),
        "peak_power_wasserstein_kw": float(wasserstein_distance(real_peak, generated_peak)),
        "ramp_wasserstein_kw": None,
        "mean_profile_rmse_kwh": float(np.sqrt(np.mean((real.mean(axis=0) - generated.mean(axis=0)) ** 2))),
        "pointwise_variance_rmse_kwh2": float(np.sqrt(np.mean((real_variance - generated_variance) ** 2))),
        "real_mean_daily_energy_kwh": float(real_daily.mean()),
        "generated_mean_daily_energy_kwh": float(generated_daily.mean()),
        "real_mean_peak_power_kw": float(real_peak.mean()),
        "generated_mean_peak_power_kw": float(generated_peak.mean()),
        "real_mean_ramp_kw": None,
        "generated_mean_ramp_kw": None,
        "real_mean_pointwise_variance_kwh2": real_diversity,
        "generated_mean_pointwise_variance_kwh2": generated_diversity,
        "diversity_ratio": generated_diversity / real_diversity if real_diversity > 0 else None,
        "real_constant_profile_count": int(np.count_nonzero(np.ptp(real, axis=1) == 0)),
        "generated_constant_profile_count": int(np.count_nonzero(np.ptp(generated, axis=1) == 0)),
        "autocorrelation": {},
    }
    if real.shape[1] > 1:
        real_ramps = (np.diff(real, axis=1) / hours).ravel()
        generated_ramps = (np.diff(generated, axis=1) / hours).ravel()
        result["ramp_wasserstein_kw"] = float(wasserstein_distance(real_ramps, generated_ramps))
        result["real_mean_ramp_kw"] = float(real_ramps.mean())
        result["generated_mean_ramp_kw"] = float(generated_ramps.mean())
    # One interval, one hour, and six hours, where exactly representable and shorter
    # than a profile. Hourly data therefore uses lags 1 and 6; quarter-hourly 1,4,24.
    lags = {1}
    for duration in (1., 6.):
        intervals = duration / hours
        if np.isfinite(intervals) and np.isclose(intervals, round(intervals)):
            lags.add(int(round(intervals)))
    for lag in sorted(lag for lag in lags if 0 < lag < real.shape[1]):
        real_acf, generated_acf = _acf(real, lag), _acf(generated, lag)
        result["autocorrelation"][str(lag)] = {
            "lag_hours": lag * hours,
            "real_mean": float(real_acf.mean()),
            "generated_mean": float(generated_acf.mean()),
            "mean_difference_abs": float(abs(real_acf.mean() - generated_acf.mean())),
            "wasserstein": float(wasserstein_distance(real_acf, generated_acf)),
        }
    return result


def evaluate_profiles(real, generated, interval_hours: float = .25) -> dict:
    """Compare empirical profile distributions in physical units.

    Pointwise Wasserstein averages independent distribution distances over slots.
    Ramps pool signed, within-day changes; daily energy and peaks are per row.
    Diversity is mean population variance over slots, with generated/real ratio
    undefined (None) when real diversity is zero. Single-slot input is supported,
    with temporal metrics unavailable. ACF uses each profile independently, with
    full-row centering/normalization and constant-profile ACF=0.
    """
    real = _profiles(real, "real")
    generated = _profiles(generated, "generated")
    _matching_width(real, generated)
    return _evaluate(real, generated, _interval_hours(interval_hours))


def evaluate_conditional(real, real_labels, generated, generated_labels,
                         interval_hours: float = .25) -> dict:
    """Overall, per-condition, and equally weighted condition-macro metrics.

    Supports must match exactly: absent generated or real conditions raise rather
    than bias macro summaries through omission. Unobserved IDs in both inputs are
    not evaluated. Counts expose unequal empirical condition proportions. Undefined
    metrics propagate as None in the macro and their condition counts are reported.
    """
    real = _profiles(real, "real")
    generated = _profiles(generated, "generated")
    _matching_width(real, generated)
    real_labels = _labels(real_labels, "real_labels", len(real))
    generated_labels = _labels(generated_labels, "generated_labels", len(generated))
    hours = _interval_hours(interval_hours)
    real_support, generated_support = set(real_labels.tolist()), set(generated_labels.tolist())
    if real_support != generated_support:
        raise ValueError(
            "condition support differs; missing generated conditions: "
            f"{sorted(real_support - generated_support)}; missing real conditions: "
            f"{sorted(generated_support - real_support)}"
        )
    per_condition = {
        str(label): _evaluate(real[real_labels == label], generated[generated_labels == label], hours)
        for label in sorted(real_support)
    }
    rows = list(per_condition.values())
    excluded = {"n_real", "n_generated", "n_intervals", "interval_hours",
                "real_constant_profile_count", "generated_constant_profile_count", "autocorrelation"}
    macro, undefined = {}, {}
    for key in sorted(rows[0].keys() - excluded):
        missing = sum(row[key] is None for row in rows)
        macro[key] = None if missing else float(np.mean([row[key] for row in rows]))
        if missing:
            undefined[key] = missing
    macro["autocorrelation"] = {
        lag: {key: float(np.mean([row["autocorrelation"][lag][key] for row in rows]))
              for key in rows[0]["autocorrelation"][lag]}
        for lag in rows[0]["autocorrelation"]
    }
    return {
        "overall": _evaluate(real, generated, hours),
        "per_condition": per_condition,
        "macro": macro,
        "macro_undefined_condition_counts": undefined,
        "n_conditions": len(per_condition),
        "counts": {
            "real": {label: row["n_real"] for label, row in per_condition.items()},
            "generated": {label: row["n_generated"] for label, row in per_condition.items()},
        },
    }


def _baseline_inputs(train, train_labels, labels):
    train = _profiles(train, "train")
    train_labels = _labels(train_labels, "train_labels", len(train))
    labels = _labels(labels, "labels")
    unsupported = set(labels.tolist()) - set(train_labels.tolist())
    if unsupported:
        raise ValueError(f"unsupported requested conditions: {sorted(unsupported)}")
    return train, train_labels, labels


def resample_baseline(train, train_labels, labels, seed: int = 42) -> np.ndarray:
    """Draw with replacement from training rows of each requested condition."""
    train, train_labels, labels = _baseline_inputs(train, train_labels, labels)
    rng = np.random.default_rng(seed)
    output = np.empty((len(labels), train.shape[1]), dtype=train.dtype)
    for label in np.unique(labels):
        mask = labels == label
        choices = rng.choice(np.flatnonzero(train_labels == label), size=int(mask.sum()), replace=True)
        output[mask] = train[choices]
    return output


def mean_baseline(train, train_labels, labels) -> np.ndarray:
    """Repeat the train-only conditional mean in the requested label order."""
    train, train_labels, labels = _baseline_inputs(train, train_labels, labels)
    output = np.empty((len(labels), train.shape[1]), dtype=train.dtype)
    for label in np.unique(labels):
        output[labels == label] = train[train_labels == label].mean(axis=0)
    return output


def nearest_neighbor_diagnostics(train, generated, heldout, max_samples: int = 1000,
                                 seed: int = 42) -> dict:
    """RMS nearest-neighbor distances to a bounded training reference sample.

    Uniform sampling without replacement independently bounds all three inputs.
    Exact copies mean zero distance to the *sampled* reference, so this diagnostic
    can miss copies of training rows excluded by subsampling. Distances use all
    slots and Euclidean distance/sqrt(number of slots), in kWh/interval. A KD-tree
    avoids allocating a query-by-reference-by-slot or dense pairwise array.
    These descriptive checks provide no formal privacy guarantee.
    """
    arrays = {"train": _profiles(train, "train"),
              "generated": _profiles(generated, "generated"),
              "heldout": _profiles(heldout, "heldout")}
    _matching_width(*arrays.values())
    if isinstance(max_samples, (bool, np.bool_)) or not isinstance(max_samples, (int, np.integer)) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer")
    rng = np.random.default_rng(seed)
    sampled = {
        name: array if len(array) <= max_samples else array[rng.choice(len(array), max_samples, replace=False)]
        for name, array in arrays.items()
    }
    tree = cKDTree(sampled["train"])
    result = {
        "reference_scope": "sampled_training_profiles",
        "total_counts": {name: len(array) for name, array in arrays.items()},
        "sample_counts": {name: len(array) for name, array in sampled.items()},
        "seed": int(seed),
        "distance_units": "kWh/interval (Euclidean RMS across slots)",
    }
    for name in ("generated", "heldout"):
        euclidean, neighbors = tree.query(sampled[name], k=1)
        distances = euclidean / np.sqrt(sampled[name].shape[1])
        result[f"{name}_to_train_mean_rms_kwh"] = float(distances.mean())
        result[f"{name}_to_train_median_rms_kwh"] = float(np.median(distances))
        result[f"{name}_to_train_p05_rms_kwh"] = float(np.quantile(distances, .05))
        result[f"{name}_to_train_p95_rms_kwh"] = float(np.quantile(distances, .95))
        copies = np.all(sampled[name] == sampled["train"][neighbors], axis=1)
        result[f"{name}_exact_copy_fraction"] = float(copies.mean())
    return result


def meter_bootstrap_confidence_intervals(real, generated, real_meter_ids,
                                       interval_hours: float = .25, *,
                                       n_bootstrap: int = 200, confidence: float = .95,
                                       seed: int = 42,
                                       metrics: Sequence[str] = _DEFAULT_BOOTSTRAP_METRICS) -> dict:
    """Percentile intervals resampling entire real meters with replacement.

    Every selected meter contributes all its days, including repeated copies when
    selected more than once. The generated sample is fixed. Thus these intervals
    describe held-out-meter sampling uncertainty conditional on one generated
    sample/model, not training-seed, generative Monte Carlo, future-year, or full
    model uncertainty. At least two real meters are required; more are advisable.
    """
    real, generated = _profiles(real, "real"), _profiles(generated, "generated")
    _matching_width(real, generated)
    hours = _interval_hours(interval_hours)
    ids = np.asarray(real_meter_ids)
    if ids.ndim != 1 or len(ids) != len(real) or ids.dtype.kind not in "iuSU":
        raise ValueError("real_meter_ids must be a rank-1 string/integer array matching real rows")
    if ids.dtype.kind in "SU" and (np.char.str_len(np.char.strip(ids.astype(str))) == 0).any():
        raise ValueError("real_meter_ids cannot contain empty identifiers")
    meters = np.unique(ids)
    if len(meters) < 2:
        raise ValueError("meter bootstrap requires at least two distinct real meters")
    if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer of at least two")
    if isinstance(confidence, (bool, np.bool_)) or not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    estimate = _evaluate(real, generated, hours)
    if isinstance(metrics, str):
        raise ValueError("metrics must be a nonempty sequence of scalar metric names")
    metrics = list(metrics)
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("metrics must be a nonempty sequence of unique metric names")
    for metric in metrics:
        if metric not in estimate or not isinstance(estimate[metric], float) or metric == "interval_hours":
            raise ValueError(f"unavailable or non-metric scalar requested: {metric}")
    groups = [np.flatnonzero(ids == meter) for meter in meters]
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in metrics}
    for _ in range(n_bootstrap):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        result = _evaluate(real[indices], generated, hours)
        for metric in metrics:
            if result[metric] is None:
                raise ValueError(f"metric {metric} becomes undefined in a meter-bootstrap replicate")
            draws[metric].append(result[metric])
    alpha = (1 - confidence) / 2
    return {
        "method": "percentile_cluster_bootstrap",
        "resampling_unit": "real_meter",
        "generated_sample_policy": "fixed",
        "uncertainty_scope": "real-meter sampling only; excludes training-seed and generated-sample uncertainty",
        "n_meters": len(meters),
        "n_bootstrap": int(n_bootstrap),
        "confidence": float(confidence),
        "seed": int(seed),
        "intervals": {
            metric: {"estimate": estimate[metric], "lower": float(np.quantile(draws[metric], alpha)),
                     "upper": float(np.quantile(draws[metric], 1 - alpha))}
            for metric in metrics
        },
    }
