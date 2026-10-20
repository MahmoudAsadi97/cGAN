"""Descriptive electricity-profile evaluation in original physical units.

Inputs must be inverse transformed before evaluation. A row represents one full
profile (normally a day); ``daily_energy_wasserstein_kwh`` sums that row, so a
non-24-hour input is explicitly identified in metadata and warnings. These
metrics are descriptive comparisons, not hypothesis tests or a privacy audit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


_NN_MAX_SAMPLES = 2000
_NN_BATCH_SIZE = 128
_NN_SEED = 42
_SMALL_SUPPORT = 20


def _profiles(values: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real numeric values")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError(f"{name} must be a nonempty N-by-T array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _labels(values: np.ndarray, name: str, n: int | None = None) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if n is not None and len(labels) != n:
        raise ValueError(f"{name} must have one label for each profile")
    seen: dict[str, Any] = {}
    for value in labels.tolist():
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"{name} labels must be strings or finite numbers")
        if isinstance(value, (int, float)) and not np.isfinite(value):
            raise ValueError(f"{name} labels must be strings or finite numbers")
        key = str(value)
        if key in seen and value != seen[key]:
            raise ValueError(f"{name} contains ambiguous string representations of labels")
        seen[key] = value
    return labels


def _check_units(interval_hours: float, unit: str) -> float:
    try:
        interval = float(interval_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError("interval_hours must be finite and positive") from exc
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("interval_hours must be finite and positive")
    if unit not in ("kWh", "kW"):
        raise ValueError("unit must be 'kWh' (interval energy) or 'kW' (average power)")
    return interval


def _mean_autocorrelation(values: np.ndarray, lag: int) -> tuple[float | None, int]:
    """Mean of per-row biased, normalized autocovariance; constants excluded."""
    centered = values - values.mean(axis=1, keepdims=True)
    denominator = np.einsum("ij,ij->i", centered, centered)
    valid = denominator > 0
    count = int(valid.sum())
    if not count or lag >= values.shape[1]:
        return None, count
    numerator = np.einsum("ij,ij->i", centered[:, :-lag], centered[:, lag:])
    return float(np.mean(numerator[valid] / denominator[valid])), count


def _mean_nearest_rmse(queries: np.ndarray, reference: np.ndarray) -> float:
    """At most 128-by-2000 pairwise distances are allocated at a time."""
    total = 0.0
    for start in range(0, len(queries), _NN_BATCH_SIZE):
        block = queries[start : start + _NN_BATCH_SIZE]
        squared = cdist(block, reference, metric="sqeuclidean")
        total += float(np.sqrt(squared.min(axis=1) / queries.shape[1]).sum())
    return total / len(queries)


def _nearest_neighbor_diagnostic(
    real: np.ndarray, fake: np.ndarray, train: np.ndarray, unit: str
) -> dict[str, Any]:
    rng = np.random.default_rng(_NN_SEED)
    n_reference = min(len(real), len(train), _NN_MAX_SAMPLES)
    train_subset = train[rng.choice(len(train), n_reference, replace=False)]
    holdout_subset = real[rng.choice(len(real), n_reference, replace=False)]
    n_fake = min(len(fake), _NN_MAX_SAMPLES)
    queries = fake[rng.choice(len(fake), n_fake, replace=False)]
    train_distance = _mean_nearest_rmse(queries, train_subset)
    holdout_distance = _mean_nearest_rmse(queries, holdout_subset)
    return {
        "mean_rmse_to_train": train_distance,
        "mean_rmse_to_holdout": holdout_distance,
        "train_to_holdout_rmse_ratio": (
            train_distance / holdout_distance if holdout_distance > 0 else None
        ),
        "rmse_unit": unit,
        "n_fake": n_fake,
        "n_reference_train": n_reference,
        "n_reference_holdout": n_reference,
        "max_samples": _NN_MAX_SAMPLES,
        "seed": _NN_SEED,
        "caveat": (
            "Descriptive memorization diagnostic, not a privacy proof. References "
            "are subsampled to equal sizes; holdout data must be disjoint from "
            "training. A lower ratio means generated profiles are closer to "
            "training than holdout. Sampling and distribution shift affect the "
            "ratio; a zero holdout distance makes it undefined."
        ),
    }


def evaluate_profiles(
    real: np.ndarray,
    fake: np.ndarray,
    interval_hours: float = 0.25,
    unit: str = "kWh",
    reference_train: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare N-by-T distributions using original, unnormalized values.

    Sample counts can differ. ``kWh`` means energy per interval, while ``kW``
    means mean power during that interval. Ramps are signed adjacent changes in
    power, pooled within profiles without crossing profile boundaries. Standard
    deviation uses ``ddof=0``. Autocorrelation errors compare the mean per-profile
    autocorrelations; constant profiles are excluded and unavailable metrics are
    ``None``. The second lag is one quarter of the profile length, normally six
    hours for daily profiles, clamped to valid positive lags.
    """
    real = _profiles(real, "real")
    fake = _profiles(fake, "fake")
    if real.shape[1] != fake.shape[1]:
        raise ValueError("real and fake must have the same number of time slots")
    interval = _check_units(interval_hours, unit)
    train = None
    if reference_train is not None:
        train = _profiles(reference_train, "reference_train")
        if train.shape[1] != real.shape[1]:
            raise ValueError("reference_train must have the same number of time slots")
    t = real.shape[1]
    power_real = real / interval if unit == "kWh" else real
    power_fake = fake / interval if unit == "kWh" else fake
    energy_real = real.sum(axis=1) * (interval if unit == "kW" else 1.0)
    energy_fake = fake.sum(axis=1) * (interval if unit == "kW" else 1.0)
    quarter_lag = max(1, min(t // 4, t - 1))
    acf_real, valid_real = _mean_autocorrelation(real, 1)
    acf_fake, valid_fake = _mean_autocorrelation(fake, 1)
    quarter_real, _ = _mean_autocorrelation(real, quarter_lag)
    quarter_fake, _ = _mean_autocorrelation(fake, quarter_lag)
    warnings: list[str] = []
    if not np.isclose(t * interval, 24.0):
        warnings.append(
            "Profiles do not span 24 hours; daily_energy_wasserstein_kwh compares "
            "total energy over the provided profile duration."
        )
    if valid_real < len(real) or valid_fake < len(fake):
        warnings.append("Constant profiles are excluded from autocorrelation calculations.")
    if t < 2:
        warnings.append("Ramps and autocorrelation are unavailable for single-slot profiles.")
    if min(len(real), len(fake)) < 2:
        warnings.append("Fewer than two profiles: distributional support is insufficient.")

    result: dict[str, Any] = {
        "per_slot_wasserstein": float(np.mean([
            wasserstein_distance(real[:, slot], fake[:, slot]) for slot in range(t)
        ])),
        "daily_energy_wasserstein_kwh": float(wasserstein_distance(energy_real, energy_fake)),
        "peak_power_wasserstein_kw": float(wasserstein_distance(
            power_real.max(axis=1), power_fake.max(axis=1)
        )),
        "ramp_power_wasserstein_kw": (
            float(wasserstein_distance(
                np.diff(power_real, axis=1).ravel(), np.diff(power_fake, axis=1).ravel()
            )) if t > 1 else None
        ),
        "mean_profile_mae": float(np.abs(real.mean(axis=0) - fake.mean(axis=0)).mean()),
        "std_profile_mae": float(np.abs(real.std(axis=0) - fake.std(axis=0)).mean()),
        "autocorrelation_mae_lag_1": (
            abs(acf_real - acf_fake) if acf_real is not None and acf_fake is not None else None
        ),
        "autocorrelation_mae_lag_quarter_day": (
            abs(quarter_real - quarter_fake)
            if quarter_real is not None and quarter_fake is not None else None
        ),
        "negative_fraction": float(np.mean(fake < 0)),
        "metadata": {
            "n_real": len(real),
            "n_fake": len(fake),
            "n_time_slots": t,
            "interval_hours": interval,
            "profile_duration_hours": t * interval,
            "profile_value_unit": unit,
            "per_slot_wasserstein_unit": unit,
            "mean_profile_mae_unit": unit,
            "std_profile_mae_unit": unit,
            "autocorrelation_lag_quarter_day": quarter_lag if t > 1 else None,
            "autocorrelation_valid_real": valid_real,
            "autocorrelation_valid_fake": valid_fake,
            "autocorrelation_definition": "mean biased normalized within-profile autocovariance",
        },
        "warnings": warnings,
    }
    if train is not None:
        result["nearest_neighbor"] = _nearest_neighbor_diagnostic(real, fake, train, unit)
    return result


def conditional_bootstrap(
    train_x: np.ndarray,
    train_y: np.ndarray,
    requested_y: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Sample training rows with replacement within each requested condition.

    This is an empirical resampling baseline and deliberately reproduces real
    training rows. It must not be described as privacy preserving synthesis.
    The result preserves the order of ``requested_y`` and does not alias input.
    """
    train = _profiles(train_x, "train_x")
    labels = _labels(train_y, "train_y", len(train))
    requested = _labels(requested_y, "requested_y")
    rng = np.random.default_rng(seed)
    output = np.empty((len(requested), train.shape[1]), dtype=np.float64)
    for label in dict.fromkeys(requested.tolist()):
        source_indices = np.flatnonzero(labels == label)
        if not len(source_indices):
            raise ValueError(f"unsupported requested condition: {label!r}")
        targets = np.flatnonzero(requested == label)
        chosen = rng.choice(source_indices, size=len(targets), replace=True)
        output[targets] = train[chosen]
    return output


def evaluate_conditional(
    real: np.ndarray,
    fake: np.ndarray,
    labels: np.ndarray,
    interval_hours: float = 0.25,
    unit: str = "kWh",
    reference_train: np.ndarray | None = None,
    train_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate aligned requested conditions globally and per condition.

    ``labels[i]`` describes both ``real[i]`` and ``fake[i]``; correspondence means
    the same condition, not a paired prediction target. Training references are
    condition-matched when ``train_labels`` is supplied. Support labels are
    descriptive flags: <2 rows is insufficient; <20 is limited; otherwise
    available. These cutoffs do not establish statistical power. Absent training
    conditions are explicitly unsupported when training labels are known.
    Conditions observed only in training are reported as absent from evaluation,
    with zero evaluation counts and no metrics.
    """
    real = _profiles(real, "real")
    fake = _profiles(fake, "fake")
    if real.shape != fake.shape:
        raise ValueError("conditional real and fake must have the same N-by-T shape")
    labels = _labels(labels, "labels", len(real))
    train = None
    training_labels = None
    warnings: list[str] = []
    if train_labels is not None and reference_train is None:
        raise ValueError("train_labels requires reference_train")
    if reference_train is not None:
        train = _profiles(reference_train, "reference_train")
        if train_labels is not None:
            training_labels = _labels(train_labels, "train_labels", len(train))
        else:
            warnings.append(
                "train_labels was not supplied: training support and nearest-neighbor "
                "diagnostics are unavailable per condition."
            )
    global_metrics = evaluate_profiles(real, fake, interval_hours, unit, train)
    per_condition: dict[str, Any] = {}
    observed_labels = labels.tolist()
    if training_labels is not None:
        observed_labels += training_labels.tolist()
    for label in dict.fromkeys(observed_labels):
        mask = labels == label
        n = int(mask.sum())
        if n == 0:
            per_condition[str(label)] = {
                "label": label,
                "n_real": 0,
                "n_fake": 0,
                "n_train": int(np.count_nonzero(training_labels == label)),
                "support_status": "absent",
                "warnings": [
                    "Condition observed in training but absent from evaluation profiles; "
                    "no held-out metrics are available."
                ],
                "metrics": None,
            }
            continue
        condition_train = None
        n_train = None
        condition_warnings: list[str] = []
        status = "insufficient" if n < 2 else "limited" if n < _SMALL_SUPPORT else "available"
        if status != "available":
            condition_warnings.append(
                f"Only {n} real and {n} generated profiles; support is {status}. "
                "Treat metrics as descriptive, not reliable population estimates."
            )
        if training_labels is not None:
            train_mask = training_labels == label
            n_train = int(train_mask.sum())
            if n_train:
                condition_train = train[train_mask]
                if n_train < _SMALL_SUPPORT:
                    condition_warnings.append(f"Training support is small ({n_train} profiles).")
                    if n_train < 2:
                        status = "insufficient"
                    elif status == "available":
                        status = "limited"
            else:
                status = "unsupported"
                condition_warnings.append("Condition absent from training references; unsupported.")
        metrics = evaluate_profiles(real[mask], fake[mask], interval_hours, unit, condition_train)
        per_condition[str(label)] = {
            "label": label,
            "n_real": n,
            "n_fake": n,
            "n_train": n_train,
            "support_status": status,
            "warnings": condition_warnings + metrics["warnings"],
            "metrics": metrics,
        }
    return {
        "global": global_metrics,
        "per_condition": per_condition,
        "warnings": warnings,
        "support_policy": {
            "insufficient_below": 2,
            "limited_below": _SMALL_SUPPORT,
            "caveat": "Descriptive sample-count flags, not a statistical power calculation.",
        },
    }
